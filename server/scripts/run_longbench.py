import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from cosmos.longbench_eval import (
    get_token_counter,
    load_longbench,
    LongBenchRunner,
    normalize_choice,
    write_jsonl,
)
from cosmos.longbench_variants import get_variant, list_variants
from cosmos.token_client import TokenCoClient
from tokenc import TokenClient


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


def build_chat_payload(prompt: str, model: str, temperature: float, max_tokens: int, cot: bool) -> Dict:
    if cot:
        system = "Think step by step, then answer with a single letter."
    else:
        system = "Answer with a single letter."
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def call_openai_compatible(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    cot: bool,
) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = build_chat_payload(prompt, model, temperature, max_tokens, cot)
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
            }
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 1.6
    return {"content": "", "usage": {}}


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


def evaluate_with_model(
    prompts: List[Dict],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
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
    total_items = len(prompts)
    for idx, item in enumerate(prompts, start=1):
        if idx == 1 or idx % 5 == 0 or idx == total_items:
            print(f"[eval] {idx}/{total_items} prompts...")
        start = time.perf_counter()
        response = call_openai_compatible(
            prompt=item["prompt"],
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
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

        choice = normalize_choice(raw or "")
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
    total_items = len(samples)
    for idx, sample in enumerate(samples, start=1):
        if idx == 1 or idx % 5 == 0 or idx == total_items:
            print(f"[bear] {idx}/{total_items} compressions...")
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


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_env_file(env_path)
    parser = argparse.ArgumentParser(description="Run LongBench v2 compression evaluation.")
    parser.add_argument("--data", required=True, help="Path to LongBench v2 JSON/JSONL file.")
    parser.add_argument("--limit", type=int, default=230)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=100000)
    parser.add_argument("--target-ratio", type=float, default=0.4)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--passes", type=int, default=1, help="Number of compression passes.")
    parser.add_argument("--mode", choices=["baseline", "compressed", "both"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--bear", action="store_true")
    parser.add_argument("--custom", action="store_true", help="Use local CustomCompressor (small_compress)")
    parser.add_argument("--custom-aggressiveness", type=float, default=0.4)
    parser.add_argument("--custom-similarity-cutoff", type=float, default=0.1)
    parser.add_argument("--custom-chunk-size", type=int, default=3)
    parser.add_argument("--bear-aggressiveness", type=float, default=0.4)
    parser.add_argument("--bear-model", default=os.getenv("TOKENC_MODEL", "bear-1"))
    parser.add_argument("--bear-max-output-tokens", type=int, default=None)
    parser.add_argument("--bear-min-output-tokens", type=int, default=None)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    parser.add_argument("--price-unit", type=int, default=1000000)
    parser.add_argument("--variant", default=None, help="Predefined compressor variant name.")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit.")
    parser.add_argument(
        "--use-variant-ratio",
        action="store_true",
        help="Apply the variant's target ratio if provided.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["task_aware", "task_agnostic", "hybrid"],
        default=None,
        help="Override compressor task mode.",
    )
    parser.add_argument("--agnostic-weight", type=float, default=None)
    parser.add_argument("--agnostic-doc-sim-weight", type=float, default=None)
    parser.add_argument("--agnostic-idf-weight", type=float, default=None)
    parser.add_argument("--agnostic-diversity-weight", type=float, default=None)
    parser.add_argument("--agnostic-min-score", type=float, default=None)
    parser.add_argument(
        "--toggle",
        action="append",
        default=None,
        help="Override compressor toggle (key=value). Repeatable.",
    )
    args = parser.parse_args()

    if args.list_variants:
        print(list_variants())
        return

    os.makedirs(args.out_dir, exist_ok=True)

    token_counter = get_token_counter(model_name=args.model)
    samples = load_longbench(
        path=args.data,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        max_context_tokens=args.max_context_tokens,
        token_counter=token_counter,
    )
    runner = LongBenchRunner()

    baseline_prompts = []
    compressed_prompts = []
    compressed_samples = []
    bear_prompts = []
    bear_samples = []
    toggles: Dict = {}
    if args.variant:
        variant = get_variant(args.variant)
        toggles.update(variant.toggles)
        if args.use_variant_ratio and variant.target_ratio is not None:
            args.target_ratio = variant.target_ratio
    if args.task_mode:
        toggles["task_mode"] = args.task_mode
    if args.agnostic_weight is not None:
        toggles["agnostic_weight"] = args.agnostic_weight
    if args.agnostic_doc_sim_weight is not None:
        toggles["agnostic_doc_sim_weight"] = args.agnostic_doc_sim_weight
    if args.agnostic_idf_weight is not None:
        toggles["agnostic_idf_weight"] = args.agnostic_idf_weight
    if args.agnostic_diversity_weight is not None:
        toggles["agnostic_diversity_weight"] = args.agnostic_diversity_weight
    if args.agnostic_min_score is not None:
        toggles["agnostic_min_score"] = args.agnostic_min_score
    if args.toggle:
        for raw in args.toggle:
            if "=" not in raw:
                raise SystemExit(f"Invalid --toggle value: {raw} (expected key=value)")
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise SystemExit(f"Invalid --toggle value: {raw} (empty key)")
            lowered = value.lower()
            if lowered in ("true", "false"):
                parsed: object = lowered == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            toggles[key] = parsed
    if not toggles:
        toggles = None
    if args.mode in ("baseline", "both"):
        print("[prepare] building baseline prompts")
        baseline_prompts = runner.build_prompts(samples)
        write_jsonl(os.path.join(args.out_dir, "baseline_prompts.jsonl"), baseline_prompts)
    if args.mode in ("compressed", "both"):
        print("[prepare] running longbench compressor")
        compressed_samples = runner.compress_samples(
            samples,
            target_ratio=args.target_ratio,
            token_budget=args.token_budget,
            seed=args.seed,
            passes=args.passes,
            toggles=toggles,
        )
        compressed_prompts = runner.build_compressed_prompts(compressed_samples)
        write_jsonl(os.path.join(args.out_dir, "compressed_prompts.jsonl"), compressed_prompts)

    if compressed_samples:
        savings = summarize_savings(compressed_samples)
        with open(os.path.join(args.out_dir, "compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(savings, handle, ensure_ascii=True, indent=2)

    if args.bear:
        print("[prepare] running bear compression")
        token_client = TokenCoClient()
        if not token_client.available:
            raise SystemExit("TokenCo client unavailable. Check TOKENC_API_KEY and tokenc installation.")
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
        bear_savings = summarize_savings(bear_samples)
        with open(os.path.join(args.out_dir, "bear_compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(bear_savings, handle, ensure_ascii=True, indent=2)

    if args.custom:
        try:
            from small_compress.custom_compressor import CustomCompressor
            from small_compress.vector_store import VectorStore
            from small_compress.logger import Logger
        except Exception as exc:
            raise SystemExit(
                "Custom compressor dependencies missing. Install small_compress requirements or skip --custom."
            ) from exc
        tokenc_key = os.getenv("TOKEN_COMPANY_API_KEY")
        if not tokenc_key:
            raise SystemExit("Missing TOKEN_COMPANY_API_KEY for custom compressor")
        # create clients and helpers similar to analyzer.py
        tc = TokenClient(api_key=tokenc_key)
        logger = Logger()
        vector_store = VectorStore(logger=logger)
        custom_comp = CustomCompressor(
            tokenc_client=tc,
            vector_store=vector_store,
            logger=logger,
            similarity_cutoff=args.custom_similarity_cutoff,
            chunk_size=args.custom_chunk_size,
        )

        custom_samples: List[Dict] = []
        for sample in samples:
            start = time.perf_counter()
            # use question+choices as context signal for the vector filter
            question_context = sample.get("question", "") + " " + " ".join(sample.get("choices", []))
            compressed = custom_comp.compress(text=sample["context"], context=question_context, aggressiveness=args.custom_aggressiveness)
            duration = time.perf_counter() - start
            compressed_text = getattr(compressed, "output", "")
            metrics = getattr(compressed, "metrics", {}) or {}
            compression_time = getattr(compressed, "compression_time", None)
            if compression_time is None:
                compression_time = round(duration, 4)
            custom_samples.append(
                {
                    **sample,
                    "compressed_context": compressed_text,
                    "metrics": metrics,
                    "custom": {
                        "compression_time_s": compression_time,
                    },
                    "compression_latency_s": round(duration, 4),
                }
            )

        custom_prompts = runner.build_compressed_prompts(custom_samples)
        write_jsonl(os.path.join(args.out_dir, "custom_prompts.jsonl"), custom_prompts)
        custom_savings = summarize_savings(custom_samples)
        with open(os.path.join(args.out_dir, "custom_compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(custom_savings, handle, ensure_ascii=True, indent=2)

    if not args.eval:
        return

    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    for run_idx in range(args.runs):
        if baseline_prompts:
            print(f"[eval] baseline run {run_idx + 1}/{args.runs}")
            baseline_eval = evaluate_with_model(
                baseline_prompts,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
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
        if compressed_prompts:
            print(f"[eval] compressed run {run_idx + 1}/{args.runs}")
            compressed_eval = evaluate_with_model(
                compressed_prompts,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
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
        if bear_prompts:
            print(f"[eval] bear run {run_idx + 1}/{args.runs}")
            bear_eval = evaluate_with_model(
                bear_prompts,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
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
        if args.custom:
            # evaluate custom prompts if present
            custom_eval = evaluate_with_model(
                custom_prompts,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                cot=args.cot,
                token_counter=token_counter,
                price_in=args.price_in,
                price_out=args.price_out,
                price_unit=args.price_unit,
            )
            with open(
                os.path.join(args.out_dir, f"custom_eval_run{run_idx + 1}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(custom_eval, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
