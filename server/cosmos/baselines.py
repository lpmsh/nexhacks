import random
from typing import Dict, List, Sequence

from .scoring import build_metrics
from .types import Span


class BaselineSuite:
    def truncate(self, text: str, token_budget: int) -> Dict:
        tokens = text.split()
        kept = tokens[:token_budget]
        compressed = " ".join(kept)
        return {
            "name": "truncate",
            "text": compressed,
            "metrics": build_metrics(text, compressed),
        }

    def head_tail(self, text: str, token_budget: int, head_pct: float = 0.7) -> Dict:
        tokens = text.split()
        head_count = int(token_budget * head_pct)
        tail_count = token_budget - head_count
        head = tokens[:head_count]
        tail = tokens[-tail_count:] if tail_count > 0 else []
        compressed = " ".join(head + tail)
        return {
            "name": "head_tail",
            "text": compressed,
            "metrics": build_metrics(text, compressed),
        }

    def random_spans(self, spans: Sequence[Span], token_budget: int, seed: int = 13) -> Dict:
        rng = random.Random(seed)
        span_ids = list(range(len(spans)))
        rng.shuffle(span_ids)
        kept: List[Span] = []
        used = 0
        for idx in span_ids:
            span = spans[idx]
            if used + span.token_count > token_budget:
                continue
            kept.append(span)
            used += span.token_count
            if used >= token_budget:
                break
        compressed = "\n\n".join(span.text for span in sorted(kept, key=lambda s: s.id))
        full_text = "\n\n".join(span.text for span in spans)
        return {
            "name": "random_spans",
            "text": compressed,
            "metrics": build_metrics(full_text, compressed),
        }

    def deduplicate(
        self,
        spans: Sequence[Span],
        similarity: List[List[float]],
        token_budget: int,
        threshold: float = 0.8,
    ) -> Dict:
        kept: List[Span] = []
        used = 0
        for i, span in enumerate(spans):
            if used + span.token_count > token_budget:
                continue
            if any(similarity[i][k.id] > threshold for k in kept):
                continue
            kept.append(span)
            used += span.token_count
        compressed = "\n\n".join(span.text for span in kept)
        full_text = "\n\n".join(span.text for span in spans)
        return {
            "name": "deduplicate",
            "text": compressed,
            "metrics": build_metrics(full_text, compressed),
        }

    def run_all(
        self,
        text: str,
        spans: Sequence[Span],
        similarity: List[List[float]],
        token_budget: int,
        seed: int = 13,
    ) -> List[Dict]:
        return [
            self.truncate(text, token_budget),
            self.head_tail(text, token_budget),
            self.random_spans(spans, token_budget, seed=seed),
            self.deduplicate(spans, similarity, token_budget),
        ]
