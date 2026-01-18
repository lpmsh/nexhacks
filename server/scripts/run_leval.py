import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from cosmos.chunker import count_tokens
from cosmos.longbench_compressor import LongBenchEngine
from cosmos.longbench_eval import get_token_counter, normalize_choice, write_jsonl
from cosmos.scoring import build_metrics


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


def build_chat_payload(prompt: str, model: str, temperature: float, max_tokens: int, cot: bool, mcq: bool) -> Dict:
    if cot:
        system = "Think step by step, then answer with a single letter." if mcq else "Think step by step, then answer."
    else:
        system = "Answer with a single letter." if mcq else "Answer concisely."
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
    mcq: bool,
) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = build_chat_payload(prompt, model, temperature, max_tokens, cot, mcq)
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


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return text


def extract_number(text: str) -> Optional[str]:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else None


def normalize_choice_sequence(text: str) -> Optional[str]:
    letters = re.findall(r"[A-D]", (text or "").upper())
    return "".join(letters) if letters else None


def is_letter_answer(answer: str) -> bool:
    if not answer:
        return False
    answer = answer.strip().upper()
    return bool(re.fullmatch(r"[A-D]+", answer))


def parse_instruction(instruction: str) -> Tuple[str, List[str]]:
    choices: List[str] = []
    question_lines: List[str] = []
    for line in (instruction or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([A-D])[\).\s-]+(.+)$", line)
        if match:
            choices.append(match.group(2).strip())
        else:
            question_lines.append(line)
    question = " ".join(question_lines).strip()
    if not question:
        question = instruction.strip()
    return question, choices


def build_prompt(context: str, instruction: str, mcq: bool) -> str:
    header = "Answer with a single letter (A, B, C, D)." if mcq else "Answer concisely."
    return (
        "Read the context and answer the question.\n"
        f"{header}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{instruction}\n\n"
        "Answer:"
    )


def load_leval_samples(
    task: str,
    split: str,
    limit: Optional[int],
    seed: int,
    shuffle: bool,
    max_context_tokens: Optional[int],
    token_counter,
    evaluation: Optional[str],
    data_path: Optional[str] = None,
) -> List[Dict]:
    rows: List[Dict] = []
    if data_path:
        if not os.path.exists(data_path):
            raise SystemExit(f"Data file not found: {data_path}")
        with open(data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "datasets is required. Run: pip install datasets or use --data-path."
            ) from exc
        dataset = load_dataset("L4NLP/LEval", task, split=split)
        rows = list(dataset)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    samples: List[Dict] = []
    for row_idx, row in enumerate(rows):
        context = row.get("input") or ""
        if not context:
            continue
        if max_context_tokens is not None and token_counter(context) > max_context_tokens:
            continue
        instructions = row.get("instructions") or []
        outputs = row.get("outputs") or []
        eval_type = row.get("evaluation")
        if evaluation and eval_type != evaluation:
            continue
        for ins_idx, instruction in enumerate(instructions):
            if not instruction or ins_idx >= len(outputs):
                continue
            answer = outputs[ins_idx]
            if not isinstance(answer, str):
                answer = str(answer)
            question, choices = parse_instruction(instruction)
            mcq = bool(choices) or is_letter_answer(answer)
            samples.append(
                {
                    "id": f"{task}-{row_idx}-{ins_idx}",
                    "task": task,
                    "source": row.get("source"),
                    "evaluation": eval_type,
                    "context": context,
                    "instruction": instruction,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "mcq": mcq,
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


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
            mcq=item.get("mcq", False),
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

        total += 1
        answer = item.get("answer") or ""
        is_correct = False
        if item.get("mcq"):
            pred_choice = normalize_choice(raw or "")
            gold_choice = normalize_choice_sequence(answer)
            is_correct = pred_choice == gold_choice
        else:
            gold_number = extract_number(answer)
            pred_number = extract_number(raw or "")
            if gold_number is not None and pred_number is not None:
                is_correct = gold_number == pred_number
            else:
                is_correct = normalize_text(raw) == normalize_text(answer)

        context_text = item.get("compressed_context") or item.get("context") or ""
        context_tokens = token_counter(context_text)
        rows.append(
            {
                **item,
                "prediction": raw,
                "correct": is_correct,
                "latency_s": round(duration, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "context_tokens": context_tokens,
                "cost_usd": round(cost, 6),
            }
        )
        if is_correct:
            correct += 1
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


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_env_file(env_path)
    parser = argparse.ArgumentParser(description="Run L-Eval compression evaluation (closed-ended tasks).")
    parser.add_argument("--task", required=True, help="L-Eval task name, e.g. quality, coursera, gsm100.")
    parser.add_argument("--data-path", default=None, help="Optional path to local LEval jsonl file.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=100000)
    parser.add_argument("--evaluation", default="exam", help="Filter by evaluation type (e.g., exam).")
    parser.add_argument("--target-ratio", type=float, default=0.4)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--mode", choices=["baseline", "compressed", "both"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--compress-per-doc", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    parser.add_argument("--price-unit", type=int, default=1000000)
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

    os.makedirs(args.out_dir, exist_ok=True)

    token_counter = get_token_counter(model_name=args.model)
    samples = load_leval_samples(
        task=args.task,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        max_context_tokens=args.max_context_tokens,
        token_counter=token_counter,
        evaluation=args.evaluation,
        data_path=args.data_path,
    )

    engine = LongBenchEngine()
    toggles: Dict = {}
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

    baseline_prompts = []
    compressed_prompts = []
    compressed_samples = []
    if args.mode in ("baseline", "both"):
        print("[prepare] building baseline prompts")
        for sample in samples:
            prompt = build_prompt(sample["context"], sample["instruction"], sample["mcq"])
            baseline_prompts.append({**sample, "prompt": prompt})
        write_jsonl(os.path.join(args.out_dir, "baseline_prompts.jsonl"), baseline_prompts)

    if args.mode in ("compressed", "both"):
        print("[prepare] running compressor")
        compressed_samples = []
        cached_context = None
        cached_metrics = None
        cached_key = None
        for sample in samples:
            if not args.compress_per_doc:
                cached_context = None
                cached_metrics = None
            context = sample["context"]
            question = sample["question"]
            choices = sample["choices"]
            if args.compress_per_doc:
                key = sample.get("id", "").rsplit("-", 1)[0]
                if cached_key != key:
                    cached_context = None
                    cached_metrics = None
                    cached_key = key
            if cached_context is None:
                compressed_context = context
                for _ in range(max(1, args.passes)):
                    result = engine.compress(
                        context=compressed_context,
                        question=question,
                        choices=choices,
                        token_budget=args.token_budget,
                        target_ratio=args.target_ratio,
                        seed=args.seed,
                        toggles=toggles,
                    )
                    compressed_context = result["compressed_context"]
                if args.passes == 1:
                    metrics = result["metrics"]
                else:
                    metrics = build_metrics(
                        context,
                        compressed_context,
                        original_tokens_override=count_tokens(context),
                        compressed_tokens_override=count_tokens(compressed_context),
                    )
                cached_context = compressed_context
                cached_metrics = metrics
            compressed_samples.append(
                {**sample, "compressed_context": cached_context, "metrics": cached_metrics}
            )
        for sample in compressed_samples:
            prompt = build_prompt(sample["compressed_context"], sample["instruction"], sample["mcq"])
            compressed_prompts.append({**sample, "prompt": prompt})
        write_jsonl(os.path.join(args.out_dir, "compressed_prompts.jsonl"), compressed_prompts)

    if compressed_samples:
        savings = summarize_savings(compressed_samples)
        with open(os.path.join(args.out_dir, "compression_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(savings, handle, ensure_ascii=True, indent=2)

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


if __name__ == "__main__":
    main()
