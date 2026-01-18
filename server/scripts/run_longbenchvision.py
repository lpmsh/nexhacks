import argparse
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types

from cosmos.longbench_eval import (
    get_token_counter,
    load_longbench,
    LongBenchRunner,
    write_jsonl,
)
from cosmos.token_client import TokenCoClient

# Lazy import PIL to avoid startup cost if not using canvas mode
_PIL_AVAILABLE = None
_FONT_CACHE = {}  # Cache for fonts: {(font_path, size): font_object}


def _ensure_pil():
    """Lazy import PIL/Pillow."""
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is None:
        try:
            from PIL import Image, ImageDraw, ImageFont
            _PIL_AVAILABLE = True
        except ImportError:
            _PIL_AVAILABLE = False
    return _PIL_AVAILABLE


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def build_gemini_prompt(prompt: str, cot: bool) -> str:
    """Build full prompt with system instruction."""
    if cot:
        system_instruction = (
            "Think step by step, then provide your final answer.\n Keep your response concise and to the point."
            "CRITICAL: You MUST end your response with exactly: [[X]] where X is A, B, C, or D.\n"
            "Examples: [[A]] or [[B]] or [[C]] or [[D]]"
        )
    else:
        system_instruction = (
            "Answer the question by selecting the correct option.\n Keep your response concise and to the point."
            "CRITICAL: You MUST end your response with exactly: [[X]] where X is A, B, C, or D.\n"
            "Examples: [[A]] or [[B]] or [[C]] or [[D]]"
        )
    
    return f"{system_instruction}\n\n{prompt}"


def extract_non_thinking_content(response) -> str:
    """Extract text content from response, filtering out thinking/thought parts."""
    if not response.candidates or len(response.candidates) == 0:
        return ""
    
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return ""
    
    text_parts = []
    for part in candidate.content.parts:
        # Skip parts that are marked as thinking/thought content
        if hasattr(part, "thought") and part.thought:
            continue
        # Skip parts with thinking indicators
        if hasattr(part, "is_thinking") and part.is_thinking:
            continue
        text = getattr(part, "text", "")
        if text:
            text_parts.append(text)
    
    return "".join(text_parts)


def extract_usage_metadata(response) -> Dict:
    """Extract usage metadata from response."""
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return {}
    return {
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
    }


def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract the answer letter from model response.
    Looks for our required format [[X]] first, then falls back to other patterns.
    """
    import re
    
    if not text:
        return None
    
    text = text.strip()
    
    # Priority 1: Look for our required format [[A]], [[B]], [[C]], [[D]]
    bracket_match = re.search(r"\[\[([A-D])\]\]", text, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).upper()
    
    # Priority 2: Other explicit answer patterns
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*([A-D])\b",  # "Answer: A", "The answer is B"
        r"\*\*([A-D])\*\*",  # Bold letter: **A**
        r"\b([A-D])\s*[.)]?\s*$",  # Letter at end: "A." or "A)" or just "A"
        r"^\s*([A-D])\s*$",  # Just the letter alone
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Priority 3: Find the last occurrence of standalone A, B, C, or D
    matches = re.findall(r"\b([A-D])\b", text.upper())
    if matches:
        return matches[-1]
    
    return None


def call_gemini(
    prompt: str,
    client: genai.Client,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    cot: bool,
) -> Dict:
    """Call Gemini API with retry logic."""
    full_prompt = build_gemini_prompt(prompt, cot)
    
    # Disable thinking mode for Gemini 3 models (use minimal thinking level)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=config,
            )
            
            content = extract_non_thinking_content(response)
            usage = extract_usage_metadata(response)
            
            return {"content": content, "usage": usage}
        except Exception as e:
            if attempt + 1 >= max_retries:
                raise
            print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(backoff)
            backoff *= 1.6
    
    return {"content": "", "usage": {}}


# =============================================================================
# Canvas Compression: Text-to-Image Rendering
# =============================================================================

def estimate_tokens_local(text: str) -> int:
    """
    Fast local token estimation for chunking.
    Uses ~4 chars per token as a reasonable approximation.
    """
    return len(text) // 4


def count_tokens_gemini(client: genai.Client, model: str, text: str) -> int:
    """Count tokens using Gemini's token counting API."""
    try:
        result = client.models.count_tokens(model=model, contents=text)
        return result.total_tokens
    except Exception:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4


def count_tokens_for_images(client: genai.Client, model: str, images: List[bytes], text_prompt: str) -> int:
    """Count tokens for images + text prompt using Gemini API."""
    try:
        contents = []
        for img_bytes in images:
            # Use inline_data for image bytes
            contents.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}})
        contents.append(text_prompt)
        
        result = client.models.count_tokens(model=model, contents=contents)
        return result.total_tokens
    except Exception:
        # Fallback estimate: ~258 tokens per image at low res + text tokens
        return len(images) * 258 + len(text_prompt) // 4


def get_default_font(size: int = 14):
    """Get a default monospace font for rendering text (with caching)."""
    from PIL import ImageFont
    
    # Check cache first
    cache_key = (None, size)  # We'll use path as None for default font
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]
    
    # Try common monospace fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/System/Library/Fonts/Monaco.ttf",  # macOS fallback
        "C:\\Windows\\Fonts\\consola.ttf",  # Windows
        "C:\\Windows\\Fonts\\cour.ttf",  # Windows fallback
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[(path, size)] = font  # Cache by path and size
                break
            except Exception:
                continue
    
    # Fallback to default
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        _FONT_CACHE[cache_key] = font
    
    return font
def wrap_text_fixed_grid_words(text: str, cols: int) -> List[str]:
    """
    Word-aware wrapping for a monospace grid (no pixel measurement).
    - Preserves explicit newlines.
    - Wraps by words when possible.
    - If a single word is longer than `cols`, it is hard-sliced.
    """
    lines: List[str] = []

    for raw_line in text.split("\n"):
        # Preserve blank lines
        if raw_line.strip() == "":
            lines.append("")
            continue

        words = raw_line.split(" ")
        cur = ""

        for w in words:
            if w == "":
                # multiple spaces -> treat as a single separator
                continue

            if cur == "":
                # start a new line
                if len(w) <= cols:
                    cur = w
                else:
                    # word too long: hard-slice
                    for i in range(0, len(w), cols):
                        chunk = w[i : i + cols]
                        if len(chunk) == cols:
                            lines.append(chunk)
                        else:
                            cur = chunk
            else:
                # try to append " " + w
                needed = 1 + len(w)
                if len(cur) + needed <= cols:
                    cur = f"{cur} {w}"
                else:
                    # commit current line
                    lines.append(cur)
                    cur = ""

                    # place w on new line (or slice if too long)
                    if len(w) <= cols:
                        cur = w
                    else:
                        for i in range(0, len(w), cols):
                            chunk = w[i : i + cols]
                            if len(chunk) == cols:
                                lines.append(chunk)
                            else:
                                cur = chunk

        if cur != "":
            lines.append(cur)

    return lines


def render_text_to_image_fast(
    text: str,
    canvas_size: int = 1024,
    font_size: int = 14,
    padding: int = 20,
    bg: int = 255,              # grayscale white
    fg: int = 0,                # grayscale black
    line_spacing: float = 1.2,
    jpeg_quality: int = 80,
) -> Tuple[bytes, int]:
    """
    Fast monospace grid renderer: no bbox per word.
    Outputs JPEG bytes + number of lines rendered.
    """
    from PIL import Image, ImageDraw

    font = get_default_font(font_size)

    # Grayscale canvas (faster, smaller)
    img = Image.new("L", (canvas_size, canvas_size), color=bg)
    draw = ImageDraw.Draw(img)

    usable_w = canvas_size - 2 * padding
    usable_h = canvas_size - 2 * padding

    # Measure a single character once
    # Using "M" as typical monospace width
    bbox = draw.textbbox((0, 0), "M", font=font)
    char_w = max(1, bbox[2] - bbox[0])
    char_h = max(1, bbox[3] - bbox[1])

    line_h = max(1, int(char_h * line_spacing))
    cols = max(1, usable_w // char_w)
    rows = max(1, usable_h // line_h)

    wrapped = wrap_text_fixed_grid_words(text, cols=cols)
    lines_to_render = wrapped[:rows]

    y = padding
    for line in lines_to_render:
        # No measuring; just draw
        draw.text((padding, y), line, font=font, fill=fg)
        y += line_h

    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=jpeg_quality,
        optimize=False,
        progressive=False,
    )
    buf.seek(0)
    return buf.getvalue(), len(lines_to_render)

def estimate_chars_per_image(
    canvas_size: int = 1024,
    font_size: int = 14,
    padding: int = 20,
    line_spacing: float = 1.2,
    avg_char_width_ratio: float = 0.6,  # Monospace font width relative to height
) -> int:
    """Estimate how many characters fit in one canvas image."""
    usable_width = canvas_size - (2 * padding)
    usable_height = canvas_size - (2 * padding)
    
    # Estimate char dimensions
    char_width = int(font_size * avg_char_width_ratio)
    line_height = int(font_size * line_spacing)
    
    chars_per_line = usable_width // char_width
    lines_per_image = usable_height // line_height
    
    return chars_per_line * lines_per_image


def split_text_for_canvas(
    text: str,
    tokens_per_image: int,
    token_counter=None,  # Optional, use local estimation if None
) -> List[str]:
    """
    Split text into chunks that each contain approximately tokens_per_image tokens.
    Uses fast local estimation for chunking (no API calls).
    """
    # Use local estimation for chunking (fast, no API calls)
    estimate_fn = estimate_tokens_local
    
    total_tokens = estimate_fn(text)
    
    if total_tokens <= tokens_per_image:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = estimate_fn(para)
        
        if current_tokens + para_tokens <= tokens_per_image:
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # If single paragraph is too large, split by sentences
            if para_tokens > tokens_per_image:
                sentences = para.replace('. ', '.\n').split('\n')
                for sent in sentences:
                    sent_tokens = estimate_fn(sent)
                    if current_tokens + sent_tokens <= tokens_per_image:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_tokens = sent_tokens
            else:
                current_chunk = [para]
                current_tokens = para_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def compress_text_to_canvas(
    text: str,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    tokens_per_image: int = 2000,
    canvas_size: int = 1024,
    font_size: int = 14,
    padding: int = 20,
    token_counter=None,
) -> Dict:
    """
    Compress text by rendering it to canvas images.
    
    Returns dict with:
        - images: List of image bytes
        - num_images: Number of images created
        - original_tokens: Original text token count (estimated)
        - estimated_image_tokens: Estimated token count for images
    """
    # Use local estimation for fast chunking (no API calls)
    original_tokens = estimate_tokens_local(text)
    
    # Split text into chunks using local estimation
    chunks = split_text_for_canvas(text, tokens_per_image, token_counter=None)
    
    # Render each chunk to an image
    images = []
    for chunk in chunks:
        img_bytes, _ = render_text_to_image_fast(
            chunk,
            canvas_size=canvas_size,
            font_size=font_size,
            padding=padding,
            jpeg_quality=80,
        )
        images.append(img_bytes)
    
    return {
        "images": images,
        "num_images": len(images),
        "original_tokens": original_tokens,
        "chunks": chunks,
    }


def call_gemini_with_images(
    images: List[bytes],
    text_prompt: str,
    client: genai.Client,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Dict:
    """
    Call Gemini API with images using lowest resolution setting.
    """
    # Build contents: images first, then text prompt
    # Use inline_data dict format for images
    contents = []
    for img_bytes in images:
        contents.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}})
    contents.append(text_prompt)
    
    # Configure with lowest resolution and minimal thinking
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            
            content = extract_non_thinking_content(response)
            usage = extract_usage_metadata(response)
            
            return {"content": content, "usage": usage}
        except Exception as e:
            if attempt + 1 >= max_retries:
                raise
            print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(backoff)
            backoff *= 1.6
    
    return {"content": "", "usage": {}}


def build_canvas_prompt(question: str, choices: List[str], cot: bool = False) -> str:
    """Build prompt for canvas-based evaluation."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
    
    if cot:
        instruction = (
            "Think step by step, then provide your final answer.\n"
            "You MUST end your response with exactly this format: [[X]] where X is A, B, C, or D."
        )
    else:
        instruction = (
            "Answer the question by selecting the correct option.\n"
            "You MUST respond with exactly this format: [[X]] where X is A, B, C, or D.\n"
            "Example: [[A]] or [[B]] or [[C]] or [[D]]"
        )
    
    return (
        f"{instruction}\n\n"
        "The context is provided in the image(s) above. Read the text carefully.\n\n"
        f"Question: {question}\n"
        "Options:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Your answer:"
    )


def _compress_single_sample(
    sample_with_idx: Tuple[int, Dict],
    tokens_per_image: int,
    canvas_size: int,
    font_size: int,
    padding: int,
) -> Tuple[int, Dict]:
    """Worker function for parallel canvas compression."""
    idx, sample = sample_with_idx
    sample_id = sample.get("id", idx)
    
    # Use compressed_context if available (for compressed_canvas mode), otherwise use context
    text_to_compress = sample.get("compressed_context") or sample["context"]
    text_label = "compressed" if "compressed_context" in sample else "original"
    
    print(f"  Sample {idx+1} (ID: {sample_id}): Converting {len(text_to_compress)} chars ({text_label}) to images...", end=" ")
    
    start = time.perf_counter()
    
    canvas_result = compress_text_to_canvas(
        text=text_to_compress,
        client=None,  # Not needed for local token estimation
        model=None,
        tokens_per_image=tokens_per_image,
        canvas_size=canvas_size,
        font_size=font_size,
        padding=padding,
        token_counter=None,
    )
    
    duration = time.perf_counter() - start
    
    # Estimate token savings
    # At low resolution, images are ~258 tokens each
    estimated_image_tokens = canvas_result["num_images"] * 258
    original_tokens = canvas_result["original_tokens"]
    savings_percent = ((original_tokens - estimated_image_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0
    
    print(f"✓ {canvas_result['num_images']} images ({duration:.2f}s)")
    
    # Preserve existing metrics if present (for compressed_canvas mode)
    existing_metrics = sample.get("metrics", {})
    
    # Merge canvas metrics with existing metrics
    canvas_metrics = {
        "original_tokens": original_tokens,
        "estimated_image_tokens": estimated_image_tokens,
        "num_images": canvas_result["num_images"],
        "savings_percent": round(savings_percent, 2),
        "compression_ratio": round(estimated_image_tokens / original_tokens, 4) if original_tokens > 0 else 1.0,
    }
    
    # If we have existing metrics (from compression), preserve them
    if existing_metrics:
        # Keep original compression metrics
        canvas_metrics["compressed_tokens"] = existing_metrics.get("compressed_tokens", original_tokens)
        canvas_metrics["compression_savings_percent"] = existing_metrics.get("savings_percent", 0.0)
        # Use original_tokens from compression if available
        if "original_tokens" in existing_metrics:
            canvas_metrics["original_tokens"] = existing_metrics["original_tokens"]
    
    result = {
        **sample,
        "canvas_images": canvas_result["images"],
        "num_images": canvas_result["num_images"],
        "metrics": canvas_metrics,
        "compression_latency_s": round(duration, 4),
    }
    
    return (idx, result)


def compress_samples_to_canvas(
    samples: List[Dict],
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    tokens_per_image: int = 2000,
    canvas_size: int = 1024,
    font_size: int = 14,
    padding: int = 20,
    token_counter=None,
    max_workers: int = 4,
) -> List[Dict]:
    """
    Compress all samples using canvas rendering (with parallel processing).
    """
    # Use parallel processing for faster compression
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _compress_single_sample,
                (i, sample),
                tokens_per_image,
                canvas_size,
                font_size,
                padding,
            )
            for i, sample in enumerate(samples)
        ]
        
        # Collect results with indices
        indexed_results = {}
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                indexed_results[idx] = result
            except Exception as e:
                print(f"Error compressing sample: {e}")
                # Continue with other samples
                continue
    
    # Reconstruct in original order
    results = [indexed_results[i] for i in range(len(samples)) if i in indexed_results]
    
    return results


def evaluate_canvas_with_gemini(
    canvas_samples: List[Dict],
    client: genai.Client,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    cot: bool,
    price_in: float,
    price_out: float,
    price_unit: int,
) -> Dict:
    """Evaluate canvas-compressed samples using Gemini vision."""
    total = 0
    correct = 0
    rows = []
    latencies: List[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    cost_total = 0.0
    
    for item in canvas_samples:
        # Build the text prompt (question + choices)
        text_prompt = build_canvas_prompt(
            item["question"],
            item["choices"],
            cot=cot,
        )
        
        start = time.perf_counter()
        response = call_gemini_with_images(
            images=item["canvas_images"],
            text_prompt=text_prompt,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        duration = time.perf_counter() - start
        latencies.append(duration)
        
        raw = response.get("content", "")
        usage = response.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        cost = estimate_cost(prompt_tokens, completion_tokens, price_in, price_out, price_unit)
        prompt_tokens_total += prompt_tokens
        completion_tokens_total += completion_tokens
        cost_total += cost

        choice = extract_answer_letter(raw or "")
        
        # Debug: print prompt and response
        num_images = item.get("num_images", 0)
        prompt_preview = text_prompt[:300] + "..." if len(text_prompt) > 300 else text_prompt
        print(f"    [{total}] Images: {num_images} | Raw: '{raw[:100]}...' -> {choice} (answer: {item.get('answer')})")
        print(f"    Prompt: '{prompt_preview}'")
        
        total += 1
        is_correct = choice == item.get("answer")
        if is_correct:
            correct += 1
        
        # Don't include image bytes in output rows (too large)
        row_data = {k: v for k, v in item.items() if k != "canvas_images"}
        rows.append({
            **row_data,
            "prediction": choice,
            "correct": is_correct,
            "latency_s": round(duration, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
        })
    
    accuracy = correct / total if total else 0.0
    latency_summary = summarize_latencies(latencies)
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "cost_usd_total": round(cost_total, 6),
        **latency_summary,
        "rows": rows,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def summarize_latencies(latencies: List[float]) -> Dict:
    if not latencies:
        return {"latency_avg_s": 0.0, "latency_p50_s": 0.0, "latency_p95_s": 0.0}
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    p50 = lat_sorted[int(0.5 * (n - 1))]
    p95 = lat_sorted[int(0.95 * (n - 1))]
    avg = sum(lat_sorted) / n
    return {
        "latency_avg_s": round(avg, 4),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
    }


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_in: float,
    price_out: float,
    price_unit: int,
) -> float:
    if price_in <= 0 and price_out <= 0:
        return 0.0
    return (prompt_tokens / price_unit) * price_in + (completion_tokens / price_unit) * price_out


def evaluate_with_gemini(
    prompts: List[Dict],
    client: genai.Client,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    cot: bool,
    token_counter,
    price_in: float,
    price_out: float,
    price_unit: int,
) -> Dict:
    total = 0
    correct = 0
    rows = []
    latencies: List[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    cost_total = 0.0
    
    for item in prompts:
        start = time.perf_counter()
        response = call_gemini(
            prompt=item["prompt"],
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cot=cot,
        )
        duration = time.perf_counter() - start
        latencies.append(duration)
        
        raw = response.get("content", "")
        usage = response.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens") or token_counter(item["prompt"])
        completion_tokens = usage.get("completion_tokens") or token_counter(raw or "")
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        cost = estimate_cost(prompt_tokens, completion_tokens, price_in, price_out, price_unit)
        prompt_tokens_total += prompt_tokens
        completion_tokens_total += completion_tokens
        cost_total += cost

        choice = extract_answer_letter(raw or "")
        
        # Debug: print prompt and response
        prompt_text = item.get("prompt", "")
        prompt_preview = prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text
        print(f"    [{total}] Raw: '{raw[:100]}...' -> {choice} (answer: {item.get('answer')})")
        print(f"    Prompt: '{prompt_preview}'")
        
        total += 1
        is_correct = choice == item.get("answer")
        if is_correct:
            correct += 1
        context_text = item.get("compressed_context") or item.get("context") or ""
        context_tokens = token_counter(context_text)
        rows.append(
            {
                **item,
                "prediction": choice,
                "correct": is_correct,
                "latency_s": round(duration, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "context_tokens": context_tokens,
                "cost_usd": round(cost, 6),
            }
        )
    
    accuracy = correct / total if total else 0.0
    latency_summary = summarize_latencies(latencies)
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "cost_usd_total": round(cost_total, 6),
        **latency_summary,
        "rows": rows,
    }


def summarize_savings(samples: List[Dict]) -> Dict:
    if not samples:
        return {"avg_savings_percent": 0.0}
    savings = [item["metrics"]["savings_percent"] for item in samples if item.get("metrics")]
    avg = sum(savings) / max(len(savings), 1)
    return {"avg_savings_percent": round(avg, 2)}


def compress_with_bear(
    samples: List[Dict],
    token_client: TokenCoClient,
    aggressiveness: float,
    model: str,
    max_output_tokens: Optional[int],
    min_output_tokens: Optional[int],
) -> List[Dict]:
    results: List[Dict] = []
    for sample in samples:
        start = time.perf_counter()
        payload = token_client.compress(
            sample["context"],
            aggressiveness=aggressiveness,
            max_output_tokens=max_output_tokens,
            min_output_tokens=min_output_tokens,
            model=model,
        )
        duration = time.perf_counter() - start
        if not payload.get("available"):
            raise SystemExit(f"TokenCo compress failed: {payload.get('error')}")
        compressed_text = payload.get("text", "")
        results.append(
            {
                **sample,
                "compressed_context": compressed_text,
                "metrics": payload.get("metrics", {}),
                "bear": payload,
                "compression_latency_s": round(duration, 4),
            }
        )
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_env_file(env_path)
    
    parser = argparse.ArgumentParser(description="Run LongBench v2 evaluation with Gemini Flash and canvas compression.")
    parser.add_argument("--data", required=True, help="Path to LongBench v2 JSON/JSONL file.")
    parser.add_argument("--limit", type=int, default=230)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=100000)
    parser.add_argument("--target-ratio", type=float, default=0.4)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--passes", type=int, default=1, help="Number of compression passes.")
    parser.add_argument("--mode", choices=["baseline", "compressed", "canvas", "compressed_canvas", "bear", "both", "all"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--cot", action="store_true")
    
    # Canvas compression options
    parser.add_argument("--canvas", action="store_true", help="Enable canvas (text-to-image) compression")
    parser.add_argument("--canvas-tokens-per-image", type=int, default=2000, 
                        help="Target tokens per canvas image (default: 2000)")
    parser.add_argument("--canvas-size", type=int, default=768, 
                        help="Canvas image size in pixels (default: 768)")
    parser.add_argument("--canvas-font-size", type=int, default=14, 
                        help="Font size for canvas text (default: 14)")
    parser.add_argument("--canvas-padding", type=int, default=20, 
                        help="Canvas padding in pixels (default: 20)")
    parser.add_argument("--canvas-workers", type=int, default=4,
                        help="Number of parallel workers for canvas compression (default: 4)")
    parser.add_argument("--save-canvas-images", action="store_true",
                        help="Save canvas images to disk for inspection")
    
    # Bear compression options
    parser.add_argument("--bear", action="store_true")
    parser.add_argument("--bear-aggressiveness", type=float, default=0.4)
    parser.add_argument("--bear-model", default=os.getenv("TOKENC_MODEL", "bear-1"))
    parser.add_argument("--bear-max-output-tokens", type=int, default=None)
    parser.add_argument("--bear-min-output-tokens", type=int, default=None)
    
    # Gemini API options
    parser.add_argument("--api-key", default=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=3)
    
    # Pricing (Gemini 2.0 Flash pricing per 1M tokens)
    parser.add_argument("--price-in", type=float, default=0.10)
    parser.add_argument("--price-out", type=float, default=0.40)
    parser.add_argument("--price-unit", type=int, default=1000000)
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Check if canvas mode requires PIL
    use_canvas = args.canvas or args.mode in ("canvas", "compressed_canvas", "all")
    if use_canvas and not _ensure_pil():
        raise SystemExit("Canvas mode requires Pillow. Install with: pip install Pillow")

    # Initialize Gemini client
    if (args.eval or use_canvas) and not args.api_key:
        raise SystemExit("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY or pass --api-key.")
    
    gemini_client = None
    if args.api_key:
        gemini_client = genai.Client(api_key=args.api_key)

    token_counter = get_token_counter(model_name=args.model)
    samples = load_longbench(
        path=args.data,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        max_context_tokens=args.max_context_tokens,
        token_counter=token_counter,
    )
    
    print(f"Loaded {len(samples)} samples from {args.data}")
    print(f"Model: {args.model}")
    
    runner = LongBenchRunner()

    baseline_prompts = []
    compressed_prompts = []
    compressed_samples = []
    bear_prompts = []
    bear_samples = []
    bear_savings = {}
    canvas_samples = []
    canvas_savings = {}
    compressed_canvas_samples = []
    compressed_canvas_savings = {}
    
    # Build baseline prompts
    if args.mode in ("baseline", "both", "all"):
        baseline_prompts = runner.build_prompts(samples)
        write_jsonl(os.path.join(args.out_dir, "baseline_prompts.jsonl"), baseline_prompts)
        print(f"Wrote {len(baseline_prompts)} baseline prompts")
    
    # Build compressed prompts (original compression)
    if args.mode in ("compressed", "both", "all", "compressed_canvas"):
        print("[prepare] running longbench compressor")
        compressed_samples = runner.compress_samples(
            samples,
            target_ratio=args.target_ratio,
            token_budget=args.token_budget,
            seed=args.seed,
            passes=args.passes,
        )
        compressed_prompts = runner.build_compressed_prompts(compressed_samples)
        write_jsonl(os.path.join(args.out_dir, "compressed_prompts.jsonl"), compressed_prompts)
        print(f"Wrote {len(compressed_prompts)} compressed prompts")
        
        savings = summarize_savings(compressed_samples)
        with open(os.path.join(args.out_dir, "compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(savings, handle, ensure_ascii=True, indent=2)
    
    # Build compressed_canvas mode: compress first, then convert to canvas
    if args.mode == "compressed_canvas":
        if not compressed_samples:
            print("[prepare] running longbench compressor for compressed_canvas mode")
            compressed_samples = runner.compress_samples(
                samples,
                target_ratio=args.target_ratio,
                token_budget=args.token_budget,
                seed=args.seed,
                passes=args.passes,
            )
        
        print(f"Compressing {len(compressed_samples)} pre-compressed samples to canvas images...")
        print(f"  Target tokens per image: {args.canvas_tokens_per_image}")
        print(f"  Canvas size: {args.canvas_size}x{args.canvas_size}")
        print(f"  Font size: {args.canvas_font_size}")
        
        compressed_canvas_samples = compress_samples_to_canvas(
            compressed_samples,  # Use compressed samples instead of original samples
            client=gemini_client,
            model=args.model,
            tokens_per_image=args.canvas_tokens_per_image,
            canvas_size=args.canvas_size,
            font_size=args.canvas_font_size,
            padding=args.canvas_padding,
            token_counter=token_counter,
            max_workers=args.canvas_workers,
        )
        
        # Optionally save images for inspection
        if args.save_canvas_images:
            img_dir = os.path.join(args.out_dir, "compressed_canvas_images")
            os.makedirs(img_dir, exist_ok=True)
            for i, sample in enumerate(compressed_canvas_samples):
                sample_id = sample.get("id", i)
                for j, img_bytes in enumerate(sample["canvas_images"]):
                    img_path = os.path.join(img_dir, f"{sample_id}_page{j}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
            print(f"  Saved compressed canvas images to {img_dir}")
        
        # Write summary (without image bytes)
        compressed_canvas_summary_data = []
        for sample in compressed_canvas_samples:
            summary = {k: v for k, v in sample.items() if k != "canvas_images"}
            compressed_canvas_summary_data.append(summary)
        write_jsonl(os.path.join(args.out_dir, "compressed_canvas_samples.jsonl"), compressed_canvas_summary_data)
        
        # Calculate and display detailed compression stats
        compressed_canvas_savings = summarize_savings(compressed_canvas_samples)
        total_images = sum(s["num_images"] for s in compressed_canvas_samples)
        # For compressed_canvas, use metrics from compression step if available
        # The compressed_samples have metrics with original_tokens and compressed_tokens
        total_original = sum(s.get("metrics", {}).get("original_tokens", 0) for s in compressed_canvas_samples)
        # If metrics don't have original_tokens, fall back to token_counter
        if total_original == 0:
            total_original = sum(token_counter(s.get("context", "")) for s in compressed_canvas_samples)
        total_compressed_text = sum(s.get("metrics", {}).get("compressed_tokens", 0) for s in compressed_canvas_samples)
        # If metrics don't have compressed_tokens, fall back to token_counter
        if total_compressed_text == 0:
            total_compressed_text = sum(token_counter(s.get("compressed_context", "")) for s in compressed_canvas_samples)
        total_estimated_image = sum(s.get("metrics", {}).get("estimated_image_tokens", 0) for s in compressed_canvas_samples)
        total_saved_vs_original = total_original - total_estimated_image
        total_saved_vs_compressed = total_compressed_text - total_estimated_image
        savings_percent_vs_original = (total_saved_vs_original / total_original * 100) if total_original > 0 else 0.0
        savings_percent_vs_compressed = (total_saved_vs_compressed / total_compressed_text * 100) if total_compressed_text > 0 else 0.0
        
        print(f"\nCompressed Canvas Compression Statistics:")
        print(f"  Original tokens: {total_original:,}")
        print(f"  Compressed text tokens: {total_compressed_text:,}")
        print(f"  Estimated image tokens: {total_estimated_image:,}")
        print(f"  Tokens saved vs original: {total_saved_vs_original:,} ({savings_percent_vs_original:.1f}%)")
        print(f"  Tokens saved vs compressed: {total_saved_vs_compressed:,} ({savings_percent_vs_compressed:.1f}%)")
        print(f"  Compression ratio (vs original): {total_estimated_image/total_original:.2%}" if total_original > 0 else "  Compression ratio: N/A")
        print(f"  Total images: {total_images} (avg {round(total_images / len(compressed_canvas_samples), 2)} per sample)")
        
        compressed_canvas_savings["total_original_tokens"] = total_original
        compressed_canvas_savings["total_compressed_text_tokens"] = total_compressed_text
        compressed_canvas_savings["total_estimated_image_tokens"] = total_estimated_image
        compressed_canvas_savings["total_tokens_saved_vs_original"] = total_saved_vs_original
        compressed_canvas_savings["total_tokens_saved_vs_compressed"] = total_saved_vs_compressed
        compressed_canvas_savings["savings_percent_vs_original"] = round(savings_percent_vs_original, 2)
        compressed_canvas_savings["savings_percent_vs_compressed"] = round(savings_percent_vs_compressed, 2)
        compressed_canvas_savings["compression_ratio_vs_original"] = round(total_estimated_image / total_original, 4) if total_original > 0 else 1.0
        compressed_canvas_savings["total_images"] = total_images
        compressed_canvas_savings["avg_images_per_sample"] = round(total_images / len(compressed_canvas_samples), 2)
        
        with open(os.path.join(args.out_dir, "compressed_canvas_compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(compressed_canvas_savings, handle, ensure_ascii=True, indent=2)
    
    # Build canvas compression (text-to-image) - regular mode
    if use_canvas and args.mode != "compressed_canvas":
        print(f"Compressing {len(samples)} samples to canvas images...")
        print(f"  Target tokens per image: {args.canvas_tokens_per_image}")
        print(f"  Canvas size: {args.canvas_size}x{args.canvas_size}")
        print(f"  Font size: {args.canvas_font_size}")
        
        canvas_samples = compress_samples_to_canvas(
            samples,
            client=gemini_client,
            model=args.model,
            tokens_per_image=args.canvas_tokens_per_image,
            canvas_size=args.canvas_size,
            font_size=args.canvas_font_size,
            padding=args.canvas_padding,
            token_counter=token_counter,
            max_workers=args.canvas_workers,
        )
        
        # Optionally save images for inspection
        if args.save_canvas_images:
            img_dir = os.path.join(args.out_dir, "canvas_images")
            os.makedirs(img_dir, exist_ok=True)
            for i, sample in enumerate(canvas_samples):
                sample_id = sample.get("id", i)
                for j, img_bytes in enumerate(sample["canvas_images"]):
                    img_path = os.path.join(img_dir, f"{sample_id}_page{j}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
            print(f"  Saved canvas images to {img_dir}")
        
        # Write summary (without image bytes)
        canvas_summary_data = []
        for sample in canvas_samples:
            summary = {k: v for k, v in sample.items() if k != "canvas_images"}
            canvas_summary_data.append(summary)
        write_jsonl(os.path.join(args.out_dir, "canvas_samples.jsonl"), canvas_summary_data)
        
        # Calculate and display detailed compression stats
        canvas_savings = summarize_savings(canvas_samples)
        total_images = sum(s["num_images"] for s in canvas_samples)
        total_original = sum(s.get("metrics", {}).get("original_tokens", 0) for s in canvas_samples)
        total_estimated_image = sum(s.get("metrics", {}).get("estimated_image_tokens", 0) for s in canvas_samples)
        total_saved = total_original - total_estimated_image
        savings_percent = (total_saved / total_original * 100) if total_original > 0 else 0.0
        
        print(f"\nCanvas Compression Statistics:")
        print(f"  Original tokens: {total_original:,}")
        print(f"  Estimated image tokens: {total_estimated_image:,}")
        print(f"  Tokens saved: {total_saved:,}")
        print(f"  Compression ratio: {total_estimated_image/total_original:.2%}" if total_original > 0 else "  Compression ratio: N/A")
        print(f"  Average savings: {canvas_savings['avg_savings_percent']:.1f}%")
        print(f"  Total images: {total_images} (avg {round(total_images / len(canvas_samples), 2)} per sample)")
        
        canvas_savings["total_original_tokens"] = total_original
        canvas_savings["total_estimated_image_tokens"] = total_estimated_image
        canvas_savings["total_tokens_saved"] = total_saved
        canvas_savings["compression_ratio"] = round(total_estimated_image / total_original, 4) if total_original > 0 else 1.0
        canvas_savings["total_images"] = total_images
        canvas_savings["avg_images_per_sample"] = round(total_images / len(canvas_samples), 2)
        
        with open(os.path.join(args.out_dir, "canvas_compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(canvas_savings, handle, ensure_ascii=True, indent=2)

    # Build Bear compression prompts
    use_bear = args.bear or args.mode in ("bear", "all")
    if use_bear:
        token_client = TokenCoClient()
        if not token_client.available:
            error_msg = getattr(token_client, 'error', 'Unknown error')
            raise SystemExit(f"TokenCo client unavailable: {error_msg}")
        
        print(f"Compressing {len(samples)} samples with Bear compression...")
        print(f"  Aggressiveness: {args.bear_aggressiveness}")
        print(f"  Model: {args.bear_model}")
        
        bear_samples = compress_with_bear(
            samples,
            token_client=token_client,
            aggressiveness=args.bear_aggressiveness,
            model=args.bear_model,
            max_output_tokens=args.bear_max_output_tokens,
            min_output_tokens=args.bear_min_output_tokens,
        )
        bear_prompts = runner.build_compressed_prompts(bear_samples)
        write_jsonl(os.path.join(args.out_dir, "bear_prompts.jsonl"), bear_prompts)
        print(f"Wrote {len(bear_prompts)} bear compression prompts")
        
        # Calculate and display detailed compression stats
        bear_savings = summarize_savings(bear_samples)
        total_original = sum(s.get("metrics", {}).get("original_tokens", 0) for s in bear_samples)
        total_compressed = sum(s.get("metrics", {}).get("compressed_tokens", 0) for s in bear_samples)
        total_saved = total_original - total_compressed
        savings_percent = (total_saved / total_original * 100) if total_original > 0 else 0.0
        
        print(f"\nBear Compression Statistics:")
        print(f"  Original tokens: {total_original:,}")
        print(f"  Compressed tokens: {total_compressed:,}")
        print(f"  Tokens saved: {total_saved:,}")
        print(f"  Compression ratio: {total_compressed/total_original:.2%}" if total_original > 0 else "  Compression ratio: N/A")
        print(f"  Average savings: {bear_savings['avg_savings_percent']:.2f}%")
        
        bear_savings["total_original_tokens"] = total_original
        bear_savings["total_compressed_tokens"] = total_compressed
        bear_savings["total_tokens_saved"] = total_saved
        bear_savings["compression_ratio"] = round(total_compressed / total_original, 4) if total_original > 0 else 1.0
        
        with open(os.path.join(args.out_dir, "bear_compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(bear_savings, handle, ensure_ascii=True, indent=2)

    if not args.eval:
        print("Skipping evaluation (--eval not set)")
        return

    if not gemini_client:
        raise SystemExit("Gemini client not initialized. Check API key.")

    for run_idx in range(args.runs):
        print(f"\n=== Evaluation Run {run_idx + 1}/{args.runs} ===")
        
        if baseline_prompts:
            print(f"Evaluating baseline ({len(baseline_prompts)} samples)...")
            baseline_eval = evaluate_with_gemini(
                baseline_prompts,
                client=gemini_client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                cot=args.cot,
                token_counter=token_counter,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"baseline_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(baseline_eval, handle, ensure_ascii=True, indent=2)
            print(f"  Baseline accuracy: {baseline_eval['accuracy']:.2%}")
            print(f"  Baseline tokens: {baseline_eval['prompt_tokens_total']:,}")
        
        if compressed_prompts:
            print(f"Evaluating compressed ({len(compressed_prompts)} samples)...")
            compressed_eval = evaluate_with_gemini(
                compressed_prompts,
                client=gemini_client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                cot=args.cot,
                token_counter=token_counter,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"compressed_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(compressed_eval, handle, ensure_ascii=True, indent=2)
            print(f"  Compressed accuracy: {compressed_eval['accuracy']:.2%}")
            print(f"  Compressed tokens: {compressed_eval['prompt_tokens_total']:,}")
        
        if canvas_samples:
            print(f"Evaluating canvas compression ({len(canvas_samples)} samples)...")
            canvas_eval = evaluate_canvas_with_gemini(
                canvas_samples,
                client=gemini_client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                cot=args.cot,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"canvas_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(canvas_eval, handle, ensure_ascii=True, indent=2)
            print(f"  Canvas accuracy: {canvas_eval['accuracy']:.2%}")
            print(f"  Canvas tokens: {canvas_eval['prompt_tokens_total']:,}")
            if canvas_samples:
                avg_compression = canvas_savings.get('avg_savings_percent', 0)
                total_images = canvas_savings.get('total_images', 0)
                print(f"  Avg compression: {avg_compression:.1f}%")
                print(f"  Total images used: {total_images}")
        
        if compressed_canvas_samples:
            print(f"Evaluating compressed canvas compression ({len(compressed_canvas_samples)} samples)...")
            compressed_canvas_eval = evaluate_canvas_with_gemini(
                compressed_canvas_samples,
                client=gemini_client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                cot=args.cot,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"compressed_canvas_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(compressed_canvas_eval, handle, ensure_ascii=True, indent=2)
            print(f"  Compressed canvas accuracy: {compressed_canvas_eval['accuracy']:.2%}")
            print(f"  Compressed canvas tokens: {compressed_canvas_eval['prompt_tokens_total']:,}")
            if compressed_canvas_samples:
                savings_vs_original = compressed_canvas_savings.get('savings_percent_vs_original', 0)
                savings_vs_compressed = compressed_canvas_savings.get('savings_percent_vs_compressed', 0)
                total_images = compressed_canvas_savings.get('total_images', 0)
                print(f"  Savings vs original: {savings_vs_original:.1f}%")
                print(f"  Savings vs compressed: {savings_vs_compressed:.1f}%")
                print(f"  Total images used: {total_images}")
        
        if bear_prompts:
            print(f"Evaluating bear compression ({len(bear_prompts)} samples)...")
            bear_eval = evaluate_with_gemini(
                bear_prompts,
                client=gemini_client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                cot=args.cot,
                token_counter=token_counter,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"bear_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(bear_eval, handle, ensure_ascii=True, indent=2)
            print(f"  Bear accuracy: {bear_eval['accuracy']:.2%}")
            print(f"  Bear tokens: {bear_eval['prompt_tokens_total']:,}")
            if bear_samples:
                avg_compression = bear_savings.get('avg_savings_percent', 0)
                print(f"  Avg compression: {avg_compression:.1f}%")

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
