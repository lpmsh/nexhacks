from typing import Dict, List, Optional, Sequence, Union

from .baselines import BaselineSuite
from .chunker import chunk_text
from .compressor import CosmosEngine
from .demo_data import SAMPLE_BATCH
from .embedder import SimpleEmbedder, similarity_matrix
from .scoring import quality_score
from .token_client import TokenCoClient


class EvaluationRunner:
    def __init__(
        self,
        engine: CosmosEngine,
        baselines: BaselineSuite,
        token_client: Optional[TokenCoClient] = None,
    ) -> None:
        self.engine = engine
        self.baselines = baselines
        self.token_client = token_client

    def run(
        self,
        budgets: Sequence[Union[int, float]] = (0.35, 0.5, 0.7),
        quality_threshold: float = 0.72,
        include_tokenc: bool = False,
    ) -> Dict:
        curves: List[Dict] = []
        example_results: List[Dict] = []

        for example in SAMPLE_BATCH:
            text = example["text"].strip()
            query = example.get("query")
            spans = chunk_text(text=text, query=query)
            embedder = SimpleEmbedder()
            embedder.fit([s.text for s in spans])
            similarity = similarity_matrix(embedder.transform([s.text for s in spans]))
            for ratio in budgets:
                budget = ratio if isinstance(ratio, int) else None
                target_ratio = ratio if isinstance(ratio, float) else 0.5
                cosmos_result = self.engine.compress(
                    text=text,
                    query=query,
                    token_budget=budget,
                    target_ratio=target_ratio,
                    run_baselines=False,
                )
                cosmos_quality = quality_score(text, cosmos_result["compressed_text"])
                cosmos_payload = {
                    "metrics": cosmos_result["metrics"],
                    "quality": cosmos_quality,
                    "compressed_text": cosmos_result["compressed_text"],
                    "budget": cosmos_result["budget"],
                }

                baseline_results = []
                for baseline in self.baselines.run_all(
                    text=text,
                    spans=spans,
                    similarity=similarity,
                    token_budget=cosmos_result["budget"],
                ):
                    baseline_quality = quality_score(text, baseline["text"])
                    baseline_results.append(
                        {
                            **baseline,
                            "quality": baseline_quality,
                        }
                    )

                tokenc_result = None
                if include_tokenc and self.token_client and self.token_client.available:
                    tokenc_payload = self.token_client.compress(
                        text,
                        aggressiveness=1 - target_ratio,
                    )
                    if tokenc_payload.get("available"):
                        tokenc_result = {
                            **tokenc_payload,
                            "quality": quality_score(text, tokenc_payload["text"]),
                        }
                    else:
                        tokenc_result = tokenc_payload

                example_results.append(
                    {
                        "category": example["category"],
                        "query": query,
                        "target_ratio": target_ratio,
                        "budget": cosmos_result["budget"],
                        "cosmos": cosmos_payload,
                        "baselines": baseline_results,
                        "tokenc": tokenc_result,
                    }
                )

                curves.append(
                    {
                        "ratio": target_ratio,
                        "quality": cosmos_quality,
                        "savings": cosmos_payload["metrics"]["savings_percent"],
                    }
                )

        avg_quality = sum(item["cosmos"]["quality"] for item in example_results) / max(
            len(example_results), 1
        )
        avg_savings = sum(
            item["cosmos"]["metrics"]["savings_percent"] for item in example_results
        ) / max(len(example_results), 1)
        fail_rate = sum(
            1
            for item in example_results
            if item["cosmos"]["quality"] < quality_threshold
        ) / max(len(example_results), 1)

        return {
            "summary": {
                "examples": len(example_results),
                "avg_quality": round(avg_quality, 3),
                "avg_savings_percent": round(avg_savings, 2),
                "fail_rate": round(fail_rate, 3),
                "quality_threshold": quality_threshold,
            },
            "curve": curves,
            "examples": example_results,
        }
