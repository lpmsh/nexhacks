import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .chunker import chunk_text, count_tokens
from .embedder import SimpleEmbedder, cosine, similarity_matrix
from .scoring import build_metrics
from .types import Span

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "as",
    "from",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "it",
    "its",
    "we",
    "you",
    "your",
    "yours",
    "our",
    "ours",
    "their",
    "theirs",
    "i",
    "me",
    "my",
    "mine",
}


class CosmosEngine:
    """Facility-location style compressor with guardrails and optional paraphrasing."""

    def __init__(self, signal_provider=None, paraphrase_fn=None) -> None:
        self.embedder = SimpleEmbedder()
        self.signal_provider = signal_provider  # Optional callable for logit/gradient scores.
        self.paraphrase_fn = paraphrase_fn  # Optional callable(text) -> text for constrained paraphrase.

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
        toggles = toggles or {}
        source_text = text if not query else f"Query: {query}\n\n{text}"
        full_tokens = max(count_tokens(source_text), 1)
        spans = chunk_text(
            text=text,
            query=query,
            keep_last_n=keep_last_n,
            keep_headings=toggles.get("keep_headings", True),
            keep_code_blocks=toggles.get("keep_code_blocks", True),
            keep_role_markers=toggles.get("keep_role_markers", True),
            must_keep_keywords=toggles.get("must_keep_keywords"),
        )
        if not spans:
            return {"compressed_text": "", "selected_spans": [], "metrics": {}}

        documents = [s.text for s in spans]
        self.embedder.fit(documents)
        span_embeddings = self.embedder.transform(documents)
        similarity = similarity_matrix(span_embeddings)
        signal_scores = self._representation_drop(spans, documents, toggles)

        query_embedding = self.embedder.encode(query) if query else None
        # Prefer external signal provider if available.
        if self.signal_provider and toggles.get("use_signal_scores", True):
            try:
                signal_scores = self.signal_provider(spans=spans, documents=documents, query=query)
            except Exception:
                signal_scores = self._representation_drop(spans, documents, toggles)

        weights = self._compute_weights(
            spans,
            span_embeddings,
            similarity,
            query_embedding=query_embedding,
            signal_scores=signal_scores,
            toggles=toggles,
        )

        budget = token_budget or int(full_tokens * target_ratio)
        budget = max(budget, min(s.token_count for s in spans))  # ensure at least one span fits

        selected, coverage = self._select_spans(
            spans, similarity, weights, budget=budget, seed=seed
        )
        for idx in selected:
            spans[idx].selected = True
        selected_spans = [spans[i] for i in selected]
        compressed_text = "\n\n".join(span.text for span in selected_spans)

        # Optional paraphrase stage for extra compression while preserving constraints.
        paraphrase_mode = toggles.get("paraphrase_mode", "none")
        if paraphrase_mode != "none":
            selected_spans, compressed_text = self._paraphrase_selected(
                selected_spans, mode=paraphrase_mode, toggles=toggles
            )

        coverage_score = self._coverage_score(weights, coverage)
        metrics = build_metrics(
            source_text,
            compressed_text,
            coverage_score=coverage_score,
            original_tokens_override=count_tokens(source_text),
            compressed_tokens_override=count_tokens(compressed_text),
        )
        cluster_threshold = toggles.get("cluster_threshold", 0.62)
        clusters = self._build_clusters(selected, similarity, threshold=cluster_threshold)

        cluster_lookup = {span_id: cluster["cluster"] for cluster in clusters for span_id in cluster["spans"]}
        for span in selected_spans:
            span.cluster = cluster_lookup.get(span.id)

        baselines = []
        if run_baselines and baseline_suite:
            baselines = baseline_suite.run_all(
                text=text,
                spans=spans,
                similarity=similarity,
                token_budget=budget,
                seed=seed,
            )

        return {
            "compressed_text": compressed_text,
            "selected_spans": [span.to_dict() for span in selected_spans],
            "spans": [span.to_dict() for span in spans],
            "clusters": clusters,
            "metrics": metrics,
            "budget": budget,
            "input_tokens": count_tokens(text),
            "source_tokens": count_tokens(source_text),
            "span_counts": {"selected": len(selected_spans), "total": len(spans)},
            "baselines": baselines,
        }

    def _representation_drop(
        self, spans: Sequence[Span], documents: Sequence[str], toggles: Dict
    ) -> List[float]:
        if not toggles.get("use_signal_scores", True):
            return []
        if not documents:
            return []
        full_text = "\n\n".join(documents)
        base_vector = self.embedder.encode(full_text)
        scores: List[float] = []
        for idx, _ in enumerate(spans):
            masked_docs = [doc for j, doc in enumerate(documents) if j != idx]
            if not masked_docs:
                scores.append(0.0)
                continue
            masked_vector = self.embedder.encode("\n\n".join(masked_docs))
            drop = max(0.0, 1 - cosine(base_vector, masked_vector))
            scores.append(round(drop, 4))
        max_drop = max(scores) or 1.0
        return [s / max_drop for s in scores]

    def _compute_weights(
        self,
        spans: Sequence[Span],
        embeddings: Sequence[Dict[str, float]],
        similarity: List[List[float]],
        query_embedding: Optional[Dict[str, float]],
        toggles: Dict,
        signal_scores: Optional[Sequence[float]] = None,
    ) -> List[float]:
        alpha = toggles.get("alpha", 0.35)
        entity_boost = toggles.get("entity_boost", 0.25)
        novelty_boost = toggles.get("novelty_boost", 0.35)
        signal_boost = toggles.get("signal_boost", 0.65)
        weights: List[float] = []
        for i, span in enumerate(spans):
            weight = span.weight
            if query_embedding:
                weight *= alpha + (1 - alpha) * cosine(embeddings[i], query_embedding)
            if signal_scores:
                weight *= 1 + signal_boost * max(signal_scores[i], 0.0)

            if toggles.get("keep_numbers_entities", True):
                if re.search(r"\d", span.text):
                    weight *= 1 + entity_boost
                if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", span.text):
                    weight *= 1 + entity_boost

            sims = similarity[i]
            max_sim = max(sim for idx, sim in enumerate(sims) if idx != i) if len(sims) > 1 else 0
            novelty = max(0.0, 1 - max_sim)
            if novelty > 0.45:
                weight *= 1 + novelty_boost * novelty

            if span.must_keep:
                weight *= 1.4

            weights.append(weight)
            spans[i].weight = weight
        return weights

    def _select_spans(
        self,
        spans: Sequence[Span],
        similarity: List[List[float]],
        weights: Sequence[float],
        budget: int,
        seed: int = 13,
    ) -> Tuple[List[int], List[float]]:
        n = len(spans)
        coverage = [0.0 for _ in range(n)]
        selected: List[int] = []
        used = 0
        rng = random.Random(seed)

        def add_span(idx: int) -> None:
            nonlocal used
            selected.append(idx)
            used += spans[idx].token_count
            for i in range(n):
                coverage[i] = max(coverage[i], similarity[i][idx])

        must_keep_ids = [i for i, span in enumerate(spans) if span.must_keep]
        for idx in must_keep_ids:
            if used + spans[idx].token_count > budget:
                budget += spans[idx].token_count
            add_span(idx)

        # Greedy facility location with marginal gain per token.
        while used < budget:
            best_idx = None
            best_gain = 0.0
            best_score = 0.0
            candidates = list(range(n))
            rng.shuffle(candidates)
            for j in candidates:
                if j in selected:
                    continue
                cost = spans[j].token_count
                if used + cost > budget:
                    continue
                gain = 0.0
                for i in range(n):
                    gain += weights[i] * max(0.0, similarity[i][j] - coverage[i])
                score = gain / max(cost, 1)
                if score > best_score:
                    best_idx = j
                    best_score = score
                    best_gain = gain
            if best_idx is None or best_gain <= 1e-5:
                break
            spans[best_idx].gain = best_gain
            add_span(best_idx)

        selected.sort()
        return selected, coverage

    def _coverage_score(self, weights: Sequence[float], coverage: Sequence[float]) -> float:
        numerator = sum(w * c for w, c in zip(weights, coverage))
        denominator = sum(weights) or 1.0
        return round(numerator / denominator, 4)

    def _paraphrase_selected(
        self, selected_spans: Sequence[Span], mode: str, toggles: Dict
    ) -> Tuple[Sequence[Span], str]:
        paraphrased: List[Span] = []
        for span in selected_spans:
            original = span.text
            new_text = original
            if span.metadata.get("contains_code") or span.metadata.get("contains_role_marker"):
                new_text = original
            elif self.paraphrase_fn and mode == "llm":
                try:
                    new_text = self.paraphrase_fn(original)
                except Exception:
                    new_text = original
            elif mode == "heuristic":
                new_text = self._heuristic_shrink(original)

            rewritten = Span(
                id=span.id,
                text=new_text,
                token_count=count_tokens(new_text),
                is_heading=span.is_heading,
                is_question=span.is_question,
                must_keep=span.must_keep,
                weight=span.weight,
                cluster=span.cluster,
                score=span.score,
                gain=span.gain,
                selected=True,
                metadata={**span.metadata, "original_text": original},
            )
            paraphrased.append(rewritten)
        compressed_text = "\n\n".join(span.text for span in paraphrased)
        return paraphrased, compressed_text

    def _heuristic_shrink(self, text: str) -> str:
        # Simple stopword drop while preserving numbers, uppercase, and quoted content.
        protected_segments = re.findall(r'\"[^\"]+\"|\\\'[^\\\']+\\\'', text)
        if protected_segments:
            return text  # Avoid touching quoted strings.

        tokens = re.findall(r"\\w+|[^\\w\\s]", text)
        kept: List[str] = []
        for tok in tokens:
            if re.search(r"\\d", tok):
                kept.append(tok)
                continue
            if tok.isupper() and len(tok) > 1:
                kept.append(tok)
                continue
            lower = tok.lower()
            if lower not in STOPWORDS and re.search(r"[A-Za-z]", tok):
                kept.append(tok)
            elif re.match(r"[^A-Za-z0-9]", tok):
                kept.append(tok)
        if not kept:
            return text
        compressed = " ".join(kept)
        compressed = re.sub(r"\\s+([,.;:])", r"\\1", compressed)
        return compressed.strip()

    def _build_clusters(
        self, selected: Sequence[int], similarity: List[List[float]], threshold: float = 0.62
    ) -> List[Dict]:
        clusters: List[List[int]] = []
        visited = set()
        for idx in selected:
            if idx in visited:
                continue
            cluster = [idx]
            visited.add(idx)
            for j in selected:
                if j in visited:
                    continue
                if similarity[idx][j] >= threshold:
                    visited.add(j)
                    cluster.append(j)
            clusters.append(cluster)
        payload = []
        for cid, members in enumerate(clusters):
            payload.append({"cluster": cid, "spans": sorted(members)})
        return payload
