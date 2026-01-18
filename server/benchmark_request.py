#!/usr/bin/env python3
"""
Benchmark script to test HTTP request latency for the /compress endpoint.
This helps identify if the bottleneck is in:
1. Network transfer (sending large payload)
2. JSON parsing / Pydantic validation
3. The actual compression algorithm
4. Response serialization

Usage:
    # First, start the server in another terminal:
    # cd server && uvicorn main:app --reload

    # Then run this script:
    python benchmark_request.py --tokens 10000
    python benchmark_request.py --tokens 50000
    python benchmark_request.py --tokens 100000 --no-baselines
"""

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add server to path for token counting
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cosmos.chunker import count_tokens
except ImportError:
    # Fallback if cosmos not available
    def count_tokens(text: str) -> int:
        return len(text.split())


def generate_text(target_tokens: int) -> str:
    """Generate synthetic text with approximately target tokens."""
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
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "data",
        "system",
        "model",
        "process",
        "function",
        "method",
        "class",
        "server",
        "client",
        "request",
        "response",
        "API",
        "database",
        "query",
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


def make_request(
    url: str,
    payload: dict,
    compress: bool = False,
    timeout: int = 300,
) -> dict:
    """Make HTTP request and return timing breakdown."""
    timings = {}

    # Step 1: Serialize to JSON
    t0 = time.perf_counter()
    json_body = json.dumps(payload)
    timings["json_serialize"] = time.perf_counter() - t0

    body_size = len(json_body.encode("utf-8"))
    timings["body_size_bytes"] = body_size

    # Step 2: Optionally compress
    if compress:
        t0 = time.perf_counter()
        compressed_body = gzip.compress(json_body.encode("utf-8"))
        timings["gzip_compress"] = time.perf_counter() - t0
        timings["compressed_size_bytes"] = len(compressed_body)
        timings["compression_ratio"] = len(compressed_body) / body_size
        body = compressed_body
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        }
    else:
        body = json_body.encode("utf-8")
        headers = {"Content-Type": "application/json"}

    # Step 3: Send request
    req = Request(url, data=body, headers=headers, method="POST")

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            timings["ttfb"] = time.perf_counter() - t0  # Time to first byte

            # Step 4: Read response
            t1 = time.perf_counter()
            response_body = response.read()
            timings["response_read"] = time.perf_counter() - t1
            timings["response_size_bytes"] = len(response_body)

            # Step 5: Parse response JSON
            t2 = time.perf_counter()
            result = json.loads(response_body)
            timings["json_parse"] = time.perf_counter() - t2

            timings["total_request"] = time.perf_counter() - t0
            timings["status"] = response.status
            timings["success"] = True
            timings["result"] = result

    except HTTPError as e:
        timings["total_request"] = time.perf_counter() - t0
        timings["status"] = e.code
        timings["success"] = False
        timings["error"] = str(e)
        try:
            timings["error_body"] = e.read().decode()[:500]
        except:
            pass
    except URLError as e:
        timings["total_request"] = time.perf_counter() - t0
        timings["success"] = False
        timings["error"] = str(e)
    except Exception as e:
        timings["total_request"] = time.perf_counter() - t0
        timings["success"] = False
        timings["error"] = str(e)

    return timings


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def run_benchmark(
    base_url: str,
    tokens: int,
    compress: bool,
    run_baselines: bool,
    query: Optional[str],
):
    """Run the benchmark and print results."""
    print("\n" + "=" * 70)
    print("HTTP REQUEST BENCHMARK")
    print("=" * 70)

    # Generate text
    print(f"\nGenerating ~{tokens:,} tokens of text...")
    t0 = time.perf_counter()
    text = generate_text(tokens)
    actual_tokens = count_tokens(text)
    gen_time = time.perf_counter() - t0
    print(f"Generated {actual_tokens:,} tokens in {format_time(gen_time)}")

    # Build payload
    payload = {
        "text": text,
        "query": query or "Summarize the key points.",
        "target_ratio": 0.5,
        "run_baselines": run_baselines,
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

    url = f"{base_url}/compress"
    print(f"\nSending request to: {url}")
    print(f"Compression: {'enabled' if compress else 'disabled'}")
    print(f"Run baselines: {run_baselines}")

    # Make request
    timings = make_request(url, payload, compress=compress)

    # Print results
    print("\n" + "-" * 70)
    print("TIMING BREAKDOWN")
    print("-" * 70)

    print(f"\n📤 REQUEST:")
    print(f"   JSON serialize:     {format_time(timings.get('json_serialize', 0)):>12}")
    print(
        f"   Body size:          {format_bytes(timings.get('body_size_bytes', 0)):>12}"
    )

    if compress:
        print(
            f"   Gzip compress:      {format_time(timings.get('gzip_compress', 0)):>12}"
        )
        print(
            f"   Compressed size:    {format_bytes(timings.get('compressed_size_bytes', 0)):>12}"
        )
        print(f"   Compression ratio:  {timings.get('compression_ratio', 0):.1%}")

    print(f"\n⏱️  SERVER:")
    ttfb = timings.get("ttfb", 0)
    print(
        f"   Time to first byte: {format_time(ttfb):>12}  ← This is server processing time"
    )

    print(f"\n📥 RESPONSE:")
    print(f"   Response read:      {format_time(timings.get('response_read', 0)):>12}")
    print(
        f"   Response size:      {format_bytes(timings.get('response_size_bytes', 0)):>12}"
    )
    print(f"   JSON parse:         {format_time(timings.get('json_parse', 0)):>12}")

    print(f"\n{'=' * 70}")
    total = timings.get("total_request", 0)
    print(f"   TOTAL REQUEST TIME: {format_time(total):>12}")
    print(f"   Tokens/second:      {actual_tokens / total if total > 0 else 0:,.0f}")
    print(f"{'=' * 70}")

    if not timings.get("success"):
        print(f"\n❌ REQUEST FAILED")
        print(f"   Error: {timings.get('error', 'Unknown')}")
        if "error_body" in timings:
            print(f"   Body: {timings['error_body']}")
        return timings

    # Analyze where time is spent
    print("\n📊 TIME DISTRIBUTION:")

    client_time = (
        timings.get("json_serialize", 0)
        + timings.get("gzip_compress", 0)
        + timings.get("response_read", 0)
        + timings.get("json_parse", 0)
    )
    server_time = ttfb

    print(
        f"   Client-side:        {format_time(client_time):>12} ({client_time / total * 100:.1f}%)"
    )
    print(
        f"   Server-side:        {format_time(server_time):>12} ({server_time / total * 100:.1f}%)"
    )

    if server_time > 5:
        print("\n⚠️  SERVER TIME IS HIGH!")
        print("   The bottleneck is server-side processing.")
        print("   Check if the server is:")
        print("   - Blocking on Pydantic validation of large text fields")
        print("   - Running compression synchronously in async handler")
        print("   - Serializing large response objects")

    # Check compression result
    result = timings.get("result", {})
    if result:
        metrics = result.get("metrics", {})
        print(f"\n📈 COMPRESSION RESULT:")
        print(f"   Input tokens:       {metrics.get('original_tokens', 'N/A'):>12}")
        print(f"   Output tokens:      {metrics.get('compressed_tokens', 'N/A'):>12}")
        print(f"   Savings:            {metrics.get('savings_percent', 'N/A')}%")

    return timings


def main():
    parser = argparse.ArgumentParser(description="Benchmark /compress HTTP requests")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=int,
        default=10000,
        help="Target number of tokens (default: 10000)",
    )
    parser.add_argument(
        "--compress",
        "-c",
        action="store_true",
        help="Enable gzip compression for request body",
    )
    parser.add_argument(
        "--no-baselines", action="store_true", help="Skip running baseline comparisons"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None, help="Query string for compression"
    )
    parser.add_argument(
        "--repeat", "-n", type=int, default=1, help="Number of times to repeat"
    )

    args = parser.parse_args()

    # Check server is running
    print(f"Checking server at {args.url}...")
    try:
        health_url = f"{args.url}/health"
        req = Request(health_url)
        with urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("✓ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure the server is running:")
        print(f"   cd server && uvicorn main:app --reload")
        sys.exit(1)

    all_timings = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#' * 70}")
            print(f"# RUN {i + 1}/{args.repeat}")
            print(f"{'#' * 70}")

        timings = run_benchmark(
            base_url=args.url,
            tokens=args.tokens,
            compress=args.compress,
            run_baselines=not args.no_baselines,
            query=args.query,
        )
        all_timings.append(timings)

    if args.repeat > 1:
        print(f"\n{'=' * 70}")
        print(f"SUMMARY ({args.repeat} runs)")
        print(f"{'=' * 70}")
        times = [t.get("total_request", 0) for t in all_timings]
        ttfbs = [t.get("ttfb", 0) for t in all_timings]
        print(
            f"Total time - Avg: {sum(times) / len(times):.2f}s, Min: {min(times):.2f}s, Max: {max(times):.2f}s"
        )
        print(
            f"TTFB       - Avg: {sum(ttfbs) / len(ttfbs):.2f}s, Min: {min(ttfbs):.2f}s, Max: {max(ttfbs):.2f}s"
        )


if __name__ == "__main__":
    main()
