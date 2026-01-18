#!/usr/bin/env python3
"""
Test script that simulates browser-like requests including CORS preflight.
This helps identify if CORS handling is causing latency issues.

Usage:
    python test_browser_request.py [--tokens N] [--url URL]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add server to path for token counting
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cosmos.chunker import count_tokens
except ImportError:

    def count_tokens(text: str) -> int:
        return len(text.split())


def generate_text(target_tokens: int) -> str:
    """Generate synthetic text."""
    words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "it",
        "for",
        "not",
        "on",
        "with",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "data",
        "system",
        "model",
        "process",
        "function",
        "server",
        "client",
    ]

    paragraphs = []
    current_tokens = 0

    while current_tokens < target_tokens:
        sentences = []
        for _ in range(random.randint(5, 10)):
            sentence = " ".join(random.choices(words, k=random.randint(10, 20)))
            sentence = sentence.capitalize() + "."
            sentences.append(sentence)

        paragraph = " ".join(sentences)
        paragraphs.append(paragraph)
        current_tokens += count_tokens(paragraph)

    return "\n\n".join(paragraphs)


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def test_cors_preflight(url: str) -> dict:
    """Simulate CORS preflight OPTIONS request."""
    timings = {}

    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }

    req = Request(url, headers=headers, method="OPTIONS")

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=30) as response:
            timings["status"] = response.status
            timings["duration"] = time.perf_counter() - t0
            timings["cors_headers"] = {
                "Access-Control-Allow-Origin": response.headers.get(
                    "Access-Control-Allow-Origin"
                ),
                "Access-Control-Allow-Methods": response.headers.get(
                    "Access-Control-Allow-Methods"
                ),
                "Access-Control-Allow-Headers": response.headers.get(
                    "Access-Control-Allow-Headers"
                ),
            }
            timings["success"] = True
    except HTTPError as e:
        timings["status"] = e.code
        timings["duration"] = time.perf_counter() - t0
        timings["success"] = False
        timings["error"] = str(e)
    except Exception as e:
        timings["duration"] = time.perf_counter() - t0
        timings["success"] = False
        timings["error"] = str(e)

    return timings


def test_post_request(
    url: str, payload: dict, origin: str = "http://localhost:5173"
) -> dict:
    """Simulate browser POST request with CORS headers."""
    timings = {}

    json_body = json.dumps(payload).encode("utf-8")
    timings["body_size"] = len(json_body)

    headers = {
        "Content-Type": "application/json",
        "Origin": origin,
        "Accept": "application/json",
        # Simulate browser headers
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    req = Request(url, data=json_body, headers=headers, method="POST")

    # Measure time to send request and get first byte
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=300) as response:
            timings["ttfb"] = time.perf_counter() - t0

            # Read response
            t1 = time.perf_counter()
            body = response.read()
            timings["read_time"] = time.perf_counter() - t1

            timings["response_size"] = len(body)
            timings["status"] = response.status
            timings["success"] = True

            # Check for timing headers
            timings["server_time_header"] = response.headers.get("X-Process-Time")

            # Parse response
            t2 = time.perf_counter()
            result = json.loads(body)
            timings["parse_time"] = time.perf_counter() - t2
            timings["result"] = result

    except HTTPError as e:
        timings["ttfb"] = time.perf_counter() - t0
        timings["status"] = e.code
        timings["success"] = False
        timings["error"] = str(e)
        try:
            timings["error_body"] = e.read().decode()[:500]
        except:
            pass
    except Exception as e:
        timings["ttfb"] = time.perf_counter() - t0
        timings["success"] = False
        timings["error"] = str(e)

    timings["total"] = time.perf_counter() - t0
    return timings


def main():
    parser = argparse.ArgumentParser(description="Test browser-like requests")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument(
        "--tokens", "-t", type=int, default=10000, help="Number of tokens"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Skip OPTIONS preflight test"
    )

    args = parser.parse_args()

    compress_url = f"{args.url}/compress"

    print("=" * 70)
    print("BROWSER-LIKE REQUEST TEST")
    print("=" * 70)

    # Test 1: CORS Preflight
    if not args.skip_preflight:
        print("\n1. Testing CORS preflight (OPTIONS request)...")
        preflight = test_cors_preflight(compress_url)

        if preflight.get("success"):
            print(f"   ✓ Preflight OK ({format_time(preflight['duration'])})")
            print(f"   CORS Headers: {preflight.get('cors_headers')}")
        else:
            print(f"   ✗ Preflight FAILED: {preflight.get('error')}")
            if preflight.get("status"):
                print(f"     Status: {preflight['status']}")

    # Test 2: Generate payload
    print(f"\n2. Generating ~{args.tokens:,} tokens of text...")
    t0 = time.perf_counter()
    text = generate_text(args.tokens)
    actual_tokens = count_tokens(text)
    print(
        f"   Generated {actual_tokens:,} tokens in {format_time(time.perf_counter() - t0)}"
    )

    payload = {
        "text": text,
        "query": "Summarize the key points.",
        "target_ratio": 0.5,
        "run_baselines": True,
        "toggles": {
            "keep_numbers_entities": True,
            "keep_headings": True,
            "keep_code_blocks": True,
            "keep_role_markers": True,
            "use_signal_scores": True,
            "signal_boost": 0.65,
            "novelty_boost": 0.35,
            "paraphrase_mode": "none",
        },
    }

    # Test 3: POST request
    print(f"\n3. Sending POST request to {compress_url}...")
    result = test_post_request(compress_url, payload)

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    if result.get("success"):
        print(f"\n✓ Request successful!")
        print(f"\nTiming breakdown:")
        print(f"   Request body size:     {result['body_size']:,} bytes")
        print(
            f"   Time to first byte:    {format_time(result['ttfb'])} ← Server processing"
        )
        print(f"   Response read time:    {format_time(result.get('read_time', 0))}")
        print(f"   Response size:         {result.get('response_size', 0):,} bytes")
        print(f"   JSON parse time:       {format_time(result.get('parse_time', 0))}")
        print(f"   TOTAL:                 {format_time(result['total'])}")

        if result.get("server_time_header"):
            print(f"\n   Server X-Process-Time: {result['server_time_header']}s")

        # Show compression result
        metrics = result.get("result", {}).get("metrics", {})
        if metrics:
            print(f"\nCompression metrics:")
            print(f"   Original tokens:       {metrics.get('original_tokens', 'N/A')}")
            print(
                f"   Compressed tokens:     {metrics.get('compressed_tokens', 'N/A')}"
            )
            print(f"   Savings:               {metrics.get('savings_percent', 'N/A')}%")
    else:
        print(f"\n✗ Request FAILED")
        print(f"   Error: {result.get('error')}")
        if result.get("status"):
            print(f"   Status: {result['status']}")
        if result.get("error_body"):
            print(f"   Body: {result['error_body']}")
        print(f"   Time until failure: {format_time(result['ttfb'])}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if result.get("success"):
        ttfb = result["ttfb"]
        server_time = (
            float(result.get("server_time_header", ttfb))
            if result.get("server_time_header")
            else ttfb
        )

        if ttfb > 10:
            print(f"""
⚠️  SLOW REQUEST DETECTED ({format_time(ttfb)})

Possible causes:
1. Server is blocking - compression algorithm is CPU-intensive
2. Large response serialization taking time
3. Network issues between client and server

If server_time ({format_time(server_time)}) << ttfb ({format_time(ttfb)}):
   → Network/transfer is the bottleneck

If server_time ≈ ttfb:
   → Server processing is the bottleneck
   → Consider running compression in a thread pool
""")
        elif ttfb > 1:
            print(f"""
Request took {format_time(ttfb)}, which is expected for {actual_tokens:,} tokens.
The compression algorithm is O(n²) for representation drop scoring.
""")
        else:
            print(f"""
✓ Request completed quickly ({format_time(ttfb)})
No obvious performance issues detected.
""")


if __name__ == "__main__":
    main()
