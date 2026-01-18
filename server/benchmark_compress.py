#!/usr/bin/env python3
"""
Benchmark script to test COSMOS compression performance directly,
bypassing HTTP overhead to identify bottlenecks.

Usage:
    python benchmark_compress.py [--file PATH] [--tokens N] [--ratio R] [--profile]

Examples:
    # Test with generated text of ~100k tokens
    python benchmark_compress.py --tokens 100000

    # Test with a specific file
    python benchmark_compress.py --file /path/to/large_text.txt

    # Run with profiling enabled
    python benchmark_compress.py --tokens 50000 --profile

    # Run step-by-step breakdown
    python benchmark_compress.py --tokens 50000 --step-by-step
"""

import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cosmos import BaselineSuite, CosmosEngine
from cosmos.chunker import chunk_text, count_tokens, tokenize
from cosmos.embedder import SimpleEmbedder, cosine, similarity_matrix
from cosmos.local_llm import build_signal_and_paraphrase


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, results: Dict[str, float]):
        self.name = name
        self.results = results
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.results[self.name] = elapsed
        return False


def benchmark_step_by_step(
    text: str,
    query: Optional[str] = None,
    target_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Run compression step-by-step with detailed timing for each phase.
    This helps identify exactly where bottlenecks occur.
    """
    timings: Dict[str, float] = {}
    results: Dict[str, Any] = {"timings": timings}

    print("\n" + "=" * 70)
    print("STEP-BY-STEP BENCHMARK")
    print("=" * 70)

    # Step 1: Count input tokens
    with Timer("1_count_input_tokens", timings):
        input_tokens = count_tokens(text)
    print(f"\n1. Count input tokens: {timings['1_count_input_tokens']:.4f}s")
    print(f"   → {input_tokens:,} tokens")
    results["input_tokens"] = input_tokens

    # Step 2: Chunk text into spans
    with Timer("2_chunk_text", timings):
        spans = chunk_text(
            text=text,
            query=query,
            keep_last_n=1,
            keep_headings=True,
            keep_code_blocks=True,
            keep_role_markers=True,
        )
    print(f"\n2. Chunk text into spans: {timings['2_chunk_text']:.4f}s")
    print(f"   → {len(spans)} spans")
    results["num_spans"] = len(spans)

    # Step 3: Extract documents from spans
    with Timer("3_extract_documents", timings):
        documents = [s.text for s in spans]
    print(f"\n3. Extract documents: {timings['3_extract_documents']:.4f}s")

    # Step 4: Fit embedder (build IDF)
    with Timer("4_embedder_fit", timings):
        embedder = SimpleEmbedder()
        embedder.fit(documents)
    print(f"\n4. Fit embedder (build IDF): {timings['4_embedder_fit']:.4f}s")
    print(f"   → Vocabulary size: {len(embedder.idf):,}")

    # Step 5: Transform documents to vectors
    with Timer("5_embedder_transform", timings):
        span_embeddings = embedder.transform(documents)
    print(
        f"\n5. Transform documents to vectors: {timings['5_embedder_transform']:.4f}s"
    )

    # Step 6: Build similarity matrix
    with Timer("6_similarity_matrix", timings):
        sim_matrix = similarity_matrix(span_embeddings)
    print(f"\n6. Build similarity matrix: {timings['6_similarity_matrix']:.4f}s")
    print(f"   → Matrix size: {len(sim_matrix)}x{len(sim_matrix)}")
    n_spans = len(spans)
    print(f"   → Comparisons: {n_spans * (n_spans - 1) // 2:,}")

    # Step 7: Representation drop (THE BOTTLENECK)
    print(f"\n7. Representation drop signal scores...")
    with Timer("7_representation_drop_total", timings):
        # This is what _representation_drop does internally
        full_text = "\n\n".join(documents)

        with Timer("7a_encode_full_text", timings):
            base_vector = embedder.encode(full_text)
        print(f"   7a. Encode full text: {timings['7a_encode_full_text']:.4f}s")

        scores: List[float] = []
        encode_times = []

        with Timer("7b_encode_masked_docs", timings):
            for idx in range(len(spans)):
                t0 = time.perf_counter()
                masked_docs = [doc for j, doc in enumerate(documents) if j != idx]
                if not masked_docs:
                    scores.append(0.0)
                    continue
                masked_vector = embedder.encode("\n\n".join(masked_docs))
                drop = max(0.0, 1 - cosine(base_vector, masked_vector))
                scores.append(round(drop, 4))
                encode_times.append(time.perf_counter() - t0)

        print(
            f"   7b. Encode {len(spans)} masked docs: {timings['7b_encode_masked_docs']:.4f}s"
        )
        print(
            f"       → Avg per span: {sum(encode_times) / len(encode_times) * 1000:.2f}ms"
        )
        print(f"       → This is O(n²) - each span re-encodes ALL other docs!")

    print(
        f"   TOTAL representation drop: {timings['7_representation_drop_total']:.4f}s"
    )

    # Step 8: Encode query (if provided)
    if query:
        with Timer("8_encode_query", timings):
            query_embedding = embedder.encode(query)
        print(f"\n8. Encode query: {timings['8_encode_query']:.4f}s")

    # Summary
    total_time = sum(
        v
        for k, v in timings.items()
        if not k.startswith("7a") and not k.startswith("7b")
    )

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    bottlenecks = [
        ("Representation drop", timings.get("7_representation_drop_total", 0)),
        ("Similarity matrix", timings.get("6_similarity_matrix", 0)),
        ("Embedder transform", timings.get("5_embedder_transform", 0)),
        ("Chunking", timings.get("2_chunk_text", 0)),
    ]

    for name, t in sorted(bottlenecks, key=lambda x: -x[1]):
        pct = (t / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{name:25} {t:8.3f}s ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION SUGGESTIONS")
    print("=" * 70)

    rep_drop_time = timings.get("7_representation_drop_total", 0)
    if rep_drop_time > 1.0:
        print(
            """
⚠️  REPRESENTATION DROP is the main bottleneck!

Current implementation is O(n²) where n = number of spans:
  - For each span, it joins ALL other documents and re-encodes
  - With {n} spans, this means {n} full re-encodings

SUGGESTED OPTIMIZATIONS:

1. CACHING: Pre-compute token counts per document, use incremental updates
   Instead of re-encoding everything, subtract the removed doc's contribution

2. SAMPLING: For large docs, sample a subset of spans for signal scores
   Use all spans for final selection, but compute signals on ~100-200 spans

3. BATCH PROCESSING: Process spans in batches with vectorized operations
   Use numpy for vector math instead of Python dicts

4. EARLY TERMINATION: Skip signal computation for obviously low-value spans
   (very short spans, high similarity to others already computed)

5. APPROXIMATE NEAREST NEIGHBORS: Use LSH or FAISS for similarity
   Avoid O(n²) pairwise comparisons
""".format(n=len(spans))
        )

    sim_time = timings.get("6_similarity_matrix", 0)
    if sim_time > 0.5:
        print(f"""
⚠️  SIMILARITY MATRIX takes {sim_time:.2f}s

With {len(spans)} spans, computing {len(spans) * (len(spans) - 1) // 2:,} pairwise similarities.

SUGGESTED OPTIMIZATIONS:

1. Use numpy for vectorized cosine similarity (10-100x faster)
2. Use sparse matrices for TF-IDF vectors
3. Consider approximate methods (LSH) for very large span counts
""")

    results["timings"] = timings
    return results


def generate_large_text(target_tokens: int) -> str:
    """Generate synthetic text with approximately the target number of tokens."""
    # Common words for realistic text generation
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
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "data",
        "system",
        "model",
        "process",
        "function",
        "method",
        "class",
        "object",
        "value",
        "type",
        "string",
        "number",
        "array",
        "list",
        "map",
        "set",
        "key",
        "server",
        "client",
        "request",
        "response",
        "API",
        "endpoint",
        "database",
        "query",
        "table",
        "column",
        "row",
        "index",
        "cache",
        "memory",
        "storage",
        "network",
        "protocol",
        "packet",
        "connection",
        "session",
        "token",
        "auth",
        "user",
        "admin",
        "role",
        "permission",
        "access",
        "security",
        "encryption",
    ]

    # Technical phrases for more realistic content
    phrases = [
        "The system processes",
        "According to the data",
        "This implementation uses",
        "The following section describes",
        "In order to optimize",
        "The main advantage of",
        "This approach allows",
        "Based on the analysis",
        "The configuration requires",
        "To ensure consistency",
        "The algorithm performs",
        "This module handles",
        "The API endpoint returns",
        "For better performance",
        "The database stores",
    ]

    paragraphs = []
    current_tokens = 0
    paragraph_count = 0

    while current_tokens < target_tokens:
        # Generate a paragraph
        sentences = []
        para_length = random.randint(5, 12)  # sentences per paragraph

        for _ in range(para_length):
            if random.random() < 0.2:
                # Use a phrase starter
                sentence = random.choice(phrases)
                sentence += " " + " ".join(
                    random.choices(words, k=random.randint(8, 20))
                )
            else:
                sentence = " ".join(random.choices(words, k=random.randint(10, 25)))
                sentence = sentence.capitalize()
            sentence += random.choice([".", ".", ".", "!", "?"])
            sentences.append(sentence)

        # Add section headers occasionally
        if paragraph_count % 5 == 0 and paragraph_count > 0:
            header = f"\n## Section {paragraph_count // 5}: {random.choice(['Overview', 'Details', 'Implementation', 'Analysis', 'Summary', 'Configuration', 'Architecture'])}\n"
            paragraphs.append(header)
            current_tokens += count_tokens(header)

        paragraph = " ".join(sentences)
        paragraphs.append(paragraph)
        current_tokens += count_tokens(paragraph)
        paragraph_count += 1

    return "\n\n".join(paragraphs)


def load_text_file(filepath: str) -> str:
    """Load text from a file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return path.read_text(encoding="utf-8")


def benchmark_compression(
    text: str,
    query: Optional[str] = None,
    target_ratio: float = 0.5,
    run_baselines: bool = True,
    profile: bool = False,
    step_by_step: bool = False,
) -> dict:
    """
    Benchmark the compression pipeline with detailed timing.

    Returns a dict with timing breakdowns for each stage.
    """
    # Run step-by-step analysis first if requested
    if step_by_step:
        return benchmark_step_by_step(text, query, target_ratio)

    results = {
        "input_tokens": 0,
        "output_tokens": 0,
        "compression_ratio": 0.0,
        "timings": {},
        "profile_stats": None,
    }

    # Count input tokens
    t0 = time.perf_counter()
    input_tokens = count_tokens(text)
    results["timings"]["count_input_tokens"] = time.perf_counter() - t0
    results["input_tokens"] = input_tokens

    print(f"\n{'=' * 60}")
    print(f"Input: {input_tokens:,} tokens ({len(text):,} chars)")
    print(f"Target ratio: {target_ratio} ({int(input_tokens * target_ratio):,} tokens)")
    print(f"{'=' * 60}\n")

    # Initialize engine
    print("Initializing engine...")
    t0 = time.perf_counter()
    signal_provider, paraphrase_fn = build_signal_and_paraphrase()
    results["timings"]["build_signal_paraphrase"] = time.perf_counter() - t0
    print(
        f"  build_signal_and_paraphrase: {results['timings']['build_signal_paraphrase']:.3f}s"
    )

    t0 = time.perf_counter()
    engine = CosmosEngine(signal_provider=signal_provider, paraphrase_fn=paraphrase_fn)
    results["timings"]["engine_init"] = time.perf_counter() - t0
    print(f"  CosmosEngine init: {results['timings']['engine_init']:.3f}s")

    if run_baselines:
        t0 = time.perf_counter()
        baselines = BaselineSuite()
        results["timings"]["baselines_init"] = time.perf_counter() - t0
        print(f"  BaselineSuite init: {results['timings']['baselines_init']:.3f}s")
    else:
        baselines = None

    # Run compression
    print("\nRunning compression...")

    toggles = {
        "keep_numbers_entities": True,
        "keep_headings": True,
        "keep_code_blocks": True,
        "keep_role_markers": True,
        "use_signal_scores": True,
        "signal_boost": 0.65,
        "novelty_boost": 0.35,
        "paraphrase_mode": "none",
    }

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    t_compress_start = time.perf_counter()

    result = engine.compress(
        text=text,
        query=query,
        target_ratio=target_ratio,
        keep_last_n=1,
        toggles=toggles,
        run_baselines=run_baselines,
        baseline_suite=baselines,
        seed=13,
    )

    t_compress_end = time.perf_counter()
    results["timings"]["total_compress"] = t_compress_end - t_compress_start

    if profile:
        profiler.disable()
        # Capture profile stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        results["profile_stats"] = stream.getvalue()

    # Extract results
    compressed_text = result.get("compressed_text", "")
    output_tokens = count_tokens(compressed_text)
    results["output_tokens"] = output_tokens
    results["compression_ratio"] = (
        output_tokens / input_tokens if input_tokens > 0 else 0
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Input tokens:      {input_tokens:,}")
    print(f"Output tokens:     {output_tokens:,}")
    print(f"Compression ratio: {results['compression_ratio']:.2%}")
    print(
        f"Tokens saved:      {input_tokens - output_tokens:,} ({(1 - results['compression_ratio']):.1%})"
    )

    print(f"\n{'=' * 60}")
    print("TIMING BREAKDOWN")
    print(f"{'=' * 60}")

    # Detailed timings from the result if available
    total_time = results["timings"]["total_compress"]
    print(f"Total compression time: {total_time:.3f}s")
    print(f"Tokens per second:      {input_tokens / total_time:,.0f}")

    for stage, duration in sorted(results["timings"].items()):
        if stage != "total_compress":
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {stage}: {duration:.3f}s ({pct:.1f}%)")

    # Metrics from compression result
    if "metrics" in result:
        print(f"\nMetrics from engine:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")

    # Span info
    if "spans" in result:
        print(f"\nSpans: {len(result['spans'])} total")
        selected = sum(1 for s in result["spans"] if s.get("selected", False))
        print(f"  Selected: {selected}")

    # Baselines
    if "baselines" in result and result["baselines"]:
        print(f"\nBaselines ({len(result['baselines'])}):")
        for b in result["baselines"]:
            name = b.get("name", "unknown")
            metrics = b.get("metrics", {})
            print(
                f"  {name}: {metrics.get('compressed_tokens', 'N/A')} tokens, {metrics.get('savings_percent', 'N/A')}% saved"
            )

    if profile and results["profile_stats"]:
        print(f"\n{'=' * 60}")
        print("PROFILE (top 30 by cumulative time)")
        print(f"{'=' * 60}")
        print(results["profile_stats"])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark COSMOS compression performance directly"
    )
    parser.add_argument(
        "--file", "-f", type=str, help="Path to a text file to compress"
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=int,
        default=10000,
        help="Target number of tokens for generated text (default: 10000)",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=0.5,
        help="Target compression ratio (default: 0.5)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Summarize the key points from this document.",
        help="Query for compression context",
    )
    parser.add_argument(
        "--no-baselines", action="store_true", help="Skip running baseline comparisons"
    )
    parser.add_argument(
        "--profile", "-p", action="store_true", help="Enable cProfile profiling"
    )
    parser.add_argument(
        "--repeat",
        "-n",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark",
    )
    parser.add_argument(
        "--step-by-step",
        "-s",
        action="store_true",
        help="Run step-by-step analysis to identify bottlenecks",
    )

    args = parser.parse_args()

    # Load or generate text
    if args.file:
        print(f"Loading text from: {args.file}")
        text = load_text_file(args.file)
    else:
        print(f"Generating synthetic text (~{args.tokens:,} tokens)...")
        t0 = time.perf_counter()
        text = generate_large_text(args.tokens)
        gen_time = time.perf_counter() - t0
        print(f"Generated {count_tokens(text):,} tokens in {gen_time:.2f}s")

    # Run benchmark(s)
    all_results = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#' * 60}")
            print(f"# RUN {i + 1}/{args.repeat}")
            print(f"{'#' * 60}")

        results = benchmark_compression(
            text=text,
            query=args.query,
            target_ratio=args.ratio,
            run_baselines=not args.no_baselines,
            profile=args.profile,
            step_by_step=args.step_by_step,
        )
        all_results.append(results)

        # Only run once for step-by-step mode
        if args.step_by_step:
            break

    # Summary for multiple runs
    if args.repeat > 1:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({args.repeat} runs)")
        print(f"{'=' * 60}")
        times = [r["timings"]["total_compress"] for r in all_results]
        print(f"Average time: {sum(times) / len(times):.3f}s")
        print(f"Min time:     {min(times):.3f}s")
        print(f"Max time:     {max(times):.3f}s")


if __name__ == "__main__":
    main()
