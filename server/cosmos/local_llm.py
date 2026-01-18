import math
import os
from typing import Callable, Optional, Sequence, Tuple

import requests

from .types import Span


def build_signal_and_paraphrase() -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Optionally build a model-backed signal provider and paraphrase function.

    Config (env):
    - LOCAL_LLM_MODEL: path to a local HF folder (safetensors).
    - LOCAL_LLM_RUNTIME: "hf" (default).
    - GROQ_API_KEY: if set, use Groq for paraphrase (fast) instead of local LLM.
    - GROQ_MODEL: optional Groq model name (default: llama-3.1-8b-instant).

    Returns (signal_provider, paraphrase_fn) or (None, None) if unavailable.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    signal_provider = None
    paraphrase_fn = None

    model_path = os.getenv("LOCAL_LLM_MODEL")
    runtime = os.getenv("LOCAL_LLM_RUNTIME", "hf").lower()

    # Build local HF signal provider if model path is set.
    if model_path and runtime == "hf":
        signal_provider, local_paraphrase = _build_hf_runtime(model_path)
        paraphrase_fn = local_paraphrase

    # Override paraphrase with Groq if available (faster).
    if groq_key:
        paraphrase_fn = _build_groq_paraphrase(groq_key, groq_model)

    return signal_provider, paraphrase_fn


def _build_hf_runtime(model_path: str) -> Tuple[Optional[Callable], Optional[Callable]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        # Dependencies not installed; stay silent to avoid breaking startup.
        return None, None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )
        model.eval()
    except Exception:
        return None, None

    def signal_provider(
        spans: Sequence[Span], documents: Sequence[str], query: Optional[str]
    ) -> Sequence[float]:
        scores = []
        for span in spans:
            if span.must_keep or span.metadata.get("contains_code") or span.metadata.get("contains_role_marker"):
                scores.append(1.0)
                continue
            text = span.text.strip()
            if not text:
                scores.append(0.0)
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = float(outputs.loss)
            # Higher perplexity -> more surprising -> boost importance.
            ppl = math.exp(min(loss, 50.0))
            score = min(1.0, max(0.0, (ppl - 1.0) / 15.0))
            scores.append(round(score, 4))
        max_score = max(scores) if scores else 0.0
        if max_score <= 0:
            return [0.0 for _ in scores]
        return [round(s / max_score, 4) for s in scores]

    def paraphrase_fn(text: str) -> str:
        prompt = (
            "You are compressing text WITHOUT losing key facts.\n"
            "- Preserve all numbers, dates, IDs, names, quoted strings, and code fences.\n"
            "- Keep instructions, constraints, and role markers.\n"
            "- Shorten wording only. Do NOT add new facts.\n\n"
            "Text:\n"
            f"{text}\n\n"
            "Rewrite (concise, same facts):\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_tokens = gen[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        return completion or text

    return signal_provider, paraphrase_fn


def _build_groq_paraphrase(api_key: str, model: str) -> Callable[[str], str]:
    def paraphrase_fn(text: str) -> str:
        prompt = (
            "You are compressing text WITHOUT losing key facts.\n"
            "- Preserve all numbers, dates, IDs, names, quoted strings, and code fences.\n"
            "- Keep instructions, constraints, and role markers.\n"
            "- Shorten wording only. Do NOT add new facts.\n\n"
            "Text:\n"
            f"{text}\n\n"
            "Rewrite (concise, same facts):\n"
        )
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a careful compression assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 120,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            completion = data["choices"][0]["message"]["content"].strip()
            return completion or text
        except Exception:
            return text

    return paraphrase_fn
