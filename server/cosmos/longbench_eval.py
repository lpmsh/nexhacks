import json
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .chunker import count_tokens
from .longbench_compressor import LongBenchEngine

TokenCounter = Callable[[str], int]


def get_token_counter(encoding_name: str = "o200k_base", model_name: Optional[str] = None) -> TokenCounter:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return count_tokens

    enc = None
    if model_name:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = None
    if enc is None:
        try:
            enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return count_tokens

    def counter(text: str) -> int:
        return len(enc.encode(text))

    return counter


def load_longbench(
    path: str,
    limit: Optional[int] = None,
    seed: int = 13,
    shuffle: bool = False,
    max_context_tokens: Optional[int] = None,
    token_counter: Optional[TokenCounter] = None,
) -> List[Dict]:
    token_counter = token_counter or count_tokens
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            rows = payload if isinstance(payload, list) else payload.get("data", [])

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    samples: List[Dict] = []
    for row in rows:
        sample = normalize_longbench_sample(row)
        if not sample:
            continue
        if max_context_tokens is not None and token_counter(sample["context"]) > max_context_tokens:
            continue
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def normalize_longbench_sample(row: Dict) -> Optional[Dict]:
    context = row.get("context")
    question = row.get("question")
    if not context or not question:
        return None
    if "choices" in row and isinstance(row["choices"], list):
        choices = row["choices"]
    else:
        choices = [row.get("choice_A"), row.get("choice_B"), row.get("choice_C"), row.get("choice_D")]
    choices = [c for c in choices if isinstance(c, str)]
    if len(choices) < 2:
        return None
    return {
        "id": row.get("_id") or row.get("id"),
        "domain": row.get("domain"),
        "sub_domain": row.get("sub_domain"),
        "difficulty": row.get("difficulty"),
        "length": row.get("length"),
        "question": question,
        "choices": choices,
        "answer": row.get("answer"),
        "context": context,
    }


def build_prompt(context: str, question: str, choices: Sequence[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
    return (
        "Read the context and answer the multiple-choice question.\n"
        "IMPORTANT: You MUST end your response with exactly: [[X]] where X is A, B, C, or D.\n"
        "Examples: [[A]] or [[B]] or [[C]] or [[D]]\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Options:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Your answer (must end with [[X]] format):"
    )


def normalize_choice(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"[A-Z]", text.upper())
    return match.group(0) if match else None


class LongBenchRunner:
    def __init__(self, engine: Optional[LongBenchEngine] = None, token_counter: Optional[TokenCounter] = None) -> None:
        self.engine = engine or LongBenchEngine()
        self.token_counter = token_counter or count_tokens

    def compress_samples(
        self,
        samples: Sequence[Dict],
        target_ratio: float = 0.5,
        token_budget: Optional[int] = None,
        toggles: Optional[Dict] = None,
        seed: int = 13,
    ) -> List[Dict]:
        results: List[Dict] = []
        for sample in samples:
            result = self.engine.compress(
                context=sample["context"],
                question=sample["question"],
                choices=sample["choices"],
                token_budget=token_budget,
                target_ratio=target_ratio,
                seed=seed,
                toggles=toggles,
            )
            results.append({**sample, "compressed_context": result["compressed_context"], "metrics": result["metrics"]})
        return results

    def build_prompts(self, samples: Sequence[Dict]) -> List[Dict]:
        payloads: List[Dict] = []
        for sample in samples:
            prompt = build_prompt(sample["context"], sample["question"], sample["choices"])
            payloads.append({**sample, "prompt": prompt})
        return payloads

    def build_compressed_prompts(self, compressed_samples: Sequence[Dict]) -> List[Dict]:
        payloads: List[Dict] = []
        for sample in compressed_samples:
            prompt = build_prompt(sample["compressed_context"], sample["question"], sample["choices"])
            payloads.append({**sample, "prompt": prompt})
        return payloads

    def evaluate(
        self,
        prompts: Iterable[Dict],
        predictor: Callable[[str], str],
    ) -> Dict:
        total = 0
        correct = 0
        rows = []
        for item in prompts:
            pred = predictor(item["prompt"])
            choice = normalize_choice(pred or "")
            total += 1
            is_correct = choice == item.get("answer")
            if is_correct:
                correct += 1
            rows.append({**item, "prediction": choice, "correct": is_correct})
        accuracy = correct / total if total else 0.0
        return {"accuracy": round(accuracy, 4), "total": total, "correct": correct, "rows": rows}


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
