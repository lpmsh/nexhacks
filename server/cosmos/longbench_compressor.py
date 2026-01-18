import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .chunker import chunk_text, count_tokens, tokenize
from .scoring import build_metrics
from .types import Span


@dataclass
class Window:
    id: int
    span_ids: List[int]
    token_count: int
    score: float = 0.0


class BM25Scorer:
    def __init__(
        self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_term_counts: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        df: Dict[str, int] = {}
        for tokens in documents:
            counts: Dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self.doc_term_counts.append(counts)
            self.doc_len.append(len(tokens))
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        self.n_docs = len(documents)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.idf: Dict[str, float] = {}
        for tok, freq in df.items():
            self.idf[tok] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0 for _ in range(self.n_docs)]
        if not query_tokens or self.n_docs == 0:
            return scores
        q_counts: Dict[str, int] = {}
        for tok in query_tokens:
            q_counts[tok] = q_counts.get(tok, 0) + 1
        for tok, qf in q_counts.items():
            idf = self.idf.get(tok, 0.0)
            if idf <= 0:
                continue
            for i in range(self.n_docs):
                freq = self.doc_term_counts[i].get(tok, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class LongBenchEngine:
    """LongBench-focused compressor using query+option aware scoring and windowed selection.

    This engine can be used in two modes:
    1. LongBench mode: Use compress_longbench() with context, question, and choices
    2. Cosmos-compatible mode: Use compress() with text and query for frontend integration
    """

    def __init__(self, signal_provider=None, paraphrase_fn=None) -> None:
        self.signal_provider = (
            signal_provider  # Accepted for API compatibility with CosmosEngine
        )
        self.paraphrase_fn = paraphrase_fn

    def compress(
        self,
        text: str,
        query: Optional[str] = None,
        token_budget: Optional[int] = None,
        target_ratio: float = 0.5,
        keep_last_n: int = 1,
        toggles: Optional[Dict] = None,
        run_baselines: bool = False,
        baseline_suite=None,
        seed: int = 13,
    ) -> Dict:
        """Cosmos-compatible compress interface for frontend integration.

        Maps the Cosmos API to the LongBench engine internally.
        """
        toggles = toggles or {}
        toggles.setdefault("keep_last_n", keep_last_n)

        # Run the LongBench compression with empty choices (query-only mode)
        result = self.compress_longbench(
            context=text,
            question=query or "",
            choices=[],
            token_budget=token_budget,
            target_ratio=target_ratio,
            seed=seed,
            toggles=toggles,
        )

        # Transform response to match CosmosEngine output format
        cosmos_result = {
            "compressed_text": result.get("compressed_context", ""),
            "selected_spans": result.get("selected_spans", []),
            "spans": result.get("spans", []),
            "clusters": [],  # LongBench doesn't compute clusters
            "metrics": result.get("metrics", {}),
            "budget": result.get("budget", 0),
            "input_tokens": result.get("input_tokens", 0),
            "source_tokens": result.get(
                "input_tokens", 0
            ),  # Same as input for LongBench
            "span_counts": result.get("span_counts", {"selected": 0, "total": 0}),
            "baselines": [],  # LongBench doesn't run baselines currently
        }

        # Run baselines if requested and suite is provided
        if run_baselines and baseline_suite and result.get("spans"):
            from .embedder import SimpleEmbedder, similarity_matrix

            spans_objs = [
                Span(
                    id=s["id"],
                    text=s["text"],
                    token_count=s["token_count"],
                    is_heading=s.get("is_heading", False),
                    is_question=s.get("is_question", False),
                    must_keep=s.get("must_keep", False),
                    weight=s.get("weight", 1.0),
                    score=s.get("score", 0.0),
                    selected=s.get("selected", False),
                )
                for s in result.get("spans", [])
            ]
            if spans_objs:
                embedder = SimpleEmbedder()
                documents = [s.text for s in spans_objs]
                embedder.fit(documents)
                span_embeddings = embedder.transform(documents)
                similarity = similarity_matrix(span_embeddings)
                cosmos_result["baselines"] = baseline_suite.run_all(
                    text=text,
                    spans=spans_objs,
                    similarity=similarity,
                    token_budget=result.get("budget", 0),
                    seed=seed,
                )

        return cosmos_result

    def compress_longbench(
        self,
        context: str,
        question: str,
        choices: Sequence[str],
        token_budget: Optional[int] = None,
        target_ratio: float = 0.5,
        seed: int = 13,
        toggles: Optional[Dict] = None,
    ) -> Dict:
        """Original LongBench compression method with question and multiple-choice options."""
        toggles = toggles or {}
        if not context:
            return {"compressed_context": "", "selected_spans": [], "metrics": {}}
        spans = chunk_text(
            text=context,
            query=None,
            target_span_tokens=int(toggles.get("target_span_tokens", 200)),
            min_span_tokens=int(toggles.get("min_span_tokens", 80)),
            max_span_tokens=int(toggles.get("max_span_tokens", 260)),
            keep_headings=toggles.get("keep_headings", True),
            keep_code_blocks=toggles.get("keep_code_blocks", True),
            keep_role_markers=toggles.get("keep_role_markers", True),
            must_keep_keywords=toggles.get("must_keep_keywords"),
            keep_last_n=int(toggles.get("keep_last_n", 1)),
        )
        if not spans:
            return {"compressed_context": "", "selected_spans": [], "metrics": {}}

        span_tokens = [tokenize(span.text) for span in spans]
        span_token_sets = [set(tokens) for tokens in span_tokens]
        bm25 = BM25Scorer(
            span_tokens,
            k1=float(toggles.get("bm25_k1", 1.5)),
            b=float(toggles.get("bm25_b", 0.75)),
        )

        question_tokens = tokenize(question or "")
        choice_tokens = [tokenize(choice) for choice in choices]

        query_scores = bm25.score(question_tokens)
        option_scores = []
        for c_tokens in choice_tokens:
            option_scores.append(bm25.score(question_tokens + c_tokens))

        importance = self._compute_importance(
            spans=spans,
            span_tokens=span_tokens,
            question_tokens=question_tokens,
            choice_tokens=choice_tokens,
            query_scores=query_scores,
            option_scores=option_scores,
            toggles=toggles,
        )

        budget = token_budget or int(count_tokens(context) * target_ratio)
        budget = max(budget, min(span.token_count for span in spans))

        selected_ids = self._select_spans(
            spans=spans,
            span_token_sets=span_token_sets,
            importance=importance,
            budget=budget,
            seed=seed,
            toggles=toggles,
        )
        for idx in selected_ids:
            spans[idx].selected = True
            spans[idx].score = round(importance[idx], 4)

        selected_spans = [spans[i] for i in selected_ids]
        compressed_context = "\n\n".join(span.text for span in selected_spans)

        metrics = build_metrics(
            context,
            compressed_context,
            original_tokens_override=count_tokens(context),
            compressed_tokens_override=count_tokens(compressed_context),
        )

        return {
            "compressed_context": compressed_context,
            "selected_spans": [span.to_dict() for span in selected_spans],
            "spans": [span.to_dict() for span in spans],
            "metrics": metrics,
            "budget": budget,
            "input_tokens": count_tokens(context),
            "span_counts": {"selected": len(selected_spans), "total": len(spans)},
        }

    def _compute_importance(
        self,
        spans: Sequence[Span],
        span_tokens: Sequence[Sequence[str]],
        question_tokens: Sequence[str],
        choice_tokens: Sequence[Sequence[str]],
        query_scores: Sequence[float],
        option_scores: Sequence[Sequence[float]],
        toggles: Dict,
    ) -> List[float]:
        question_weight = float(toggles.get("question_weight", 0.45))
        option_weight = float(toggles.get("option_weight", 1.0))
        contrast_weight = float(toggles.get("contrast_weight", 0.7))
        entity_boost = float(toggles.get("entity_boost", 0.2))
        number_boost = float(toggles.get("number_boost", 0.2))
        must_keep_boost = float(toggles.get("must_keep_boost", 0.35))
        query_hit_boost = float(toggles.get("query_hit_boost", 0.2))
        min_query_hits = int(toggles.get("min_query_hits", 2))

        query_terms = set(question_tokens)
        for tokens in choice_tokens:
            query_terms.update(tokens)

        importance: List[float] = []
        for i, span in enumerate(spans):
            opt_vals = (
                [scores[i] for scores in option_scores] if option_scores else [0.0]
            )
            opt_sorted = sorted(opt_vals, reverse=True)
            best = opt_sorted[0] if opt_sorted else 0.0
            second = opt_sorted[1] if len(opt_sorted) > 1 else 0.0
            contrast = max(0.0, best - second)
            score = (
                question_weight * query_scores[i]
                + option_weight * best
                + contrast_weight * contrast
            )

            if span.must_keep:
                score *= 1 + must_keep_boost

            if query_terms and span_tokens[i]:
                hits = len(set(span_tokens[i]) & query_terms)
                if hits >= min_query_hits:
                    score *= 1 + query_hit_boost

            if re.search(r"\d", span.text):
                score *= 1 + number_boost
            if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", span.text):
                score *= 1 + entity_boost

            importance.append(score)
        return importance

    def _build_windows(
        self, spans: Sequence[Span], window_token_budget: int
    ) -> List[Window]:
        windows: List[Window] = []
        current_ids: List[int] = []
        current_tokens = 0
        window_id = 0
        for span in spans:
            if current_ids and current_tokens + span.token_count > window_token_budget:
                windows.append(
                    Window(
                        id=window_id, span_ids=current_ids, token_count=current_tokens
                    )
                )
                window_id += 1
                current_ids = []
                current_tokens = 0
            current_ids.append(span.id)
            current_tokens += span.token_count
        if current_ids:
            windows.append(
                Window(id=window_id, span_ids=current_ids, token_count=current_tokens)
            )
        return windows

    def _select_spans(
        self,
        spans: Sequence[Span],
        span_token_sets: Sequence[set],
        importance: Sequence[float],
        budget: int,
        seed: int,
        toggles: Dict,
    ) -> List[int]:
        rng = random.Random(seed)
        redundancy_penalty = float(toggles.get("redundancy_penalty", 0.45))
        window_token_budget = int(toggles.get("window_token_budget", 1800))
        window_top_k = int(toggles.get("window_top_k", 3))
        window_max_fraction = float(toggles.get("window_max_fraction", 0.35))
        neighbor_span_count = int(toggles.get("neighbor_span_count", 1))

        selected: List[int] = []
        selected_sets: List[set] = []
        used = 0

        must_keep_ids = [i for i, span in enumerate(spans) if span.must_keep]
        for idx in must_keep_ids:
            if idx in selected:
                continue
            if used + spans[idx].token_count > budget:
                budget += spans[idx].token_count
            selected.append(idx)
            selected_sets.append(span_token_sets[idx])
            used += spans[idx].token_count

        windows = self._build_windows(spans, window_token_budget)
        for window in windows:
            scores = sorted((importance[i] for i in window.span_ids), reverse=True)
            window.score = sum(scores[:window_top_k]) if scores else 0.0

        windows.sort(key=lambda w: (w.score / max(w.token_count, 1)), reverse=True)
        window_cap = max(1, int(budget * window_max_fraction))

        def select_from_window(window: Window, cap: int) -> None:
            nonlocal used
            candidates = [i for i in window.span_ids if i not in selected]
            rng.shuffle(candidates)
            local_used = 0
            while candidates and used < budget and local_used < cap:
                best_idx = None
                best_score = 0.0
                for i in candidates:
                    cost = spans[i].token_count
                    if used + cost > budget or local_used + cost > cap:
                        continue
                    redundancy = 0.0
                    if selected_sets:
                        redundancy = max(
                            _jaccard(span_token_sets[i], s) for s in selected_sets
                        )
                    score = importance[i] * (1 - redundancy_penalty * redundancy)
                    score_per_token = score / max(cost, 1)
                    if score_per_token > best_score:
                        best_idx = i
                        best_score = score_per_token
                if best_idx is None or best_score <= 1e-6:
                    break
                selected.append(best_idx)
                selected_sets.append(span_token_sets[best_idx])
                used += spans[best_idx].token_count
                local_used += spans[best_idx].token_count
                candidates.remove(best_idx)

        for window in windows:
            if used >= budget:
                break
            select_from_window(window, window_cap)

        if used < budget:
            for window in windows:
                if used >= budget:
                    break
                select_from_window(window, budget - used)

        if neighbor_span_count > 0:
            neighbor_ids = set()
            for idx in selected:
                for offset in range(1, neighbor_span_count + 1):
                    neighbor_ids.add(idx - offset)
                    neighbor_ids.add(idx + offset)
            for idx in sorted(neighbor_ids):
                if idx < 0 or idx >= len(spans):
                    continue
                if idx in selected:
                    continue
                cost = spans[idx].token_count
                if used + cost > budget:
                    continue
                selected.append(idx)
                selected_sets.append(span_token_sets[idx])
                used += cost

        selected = sorted(set(selected))
        return selected


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
