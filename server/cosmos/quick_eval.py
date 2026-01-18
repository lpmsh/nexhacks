import json
from typing import Dict, List

from .baselines import BaselineSuite
from .compressor import CosmosEngine
from .demo_data import SAMPLE_BATCH
from .evaluation import quality_score
from .local_llm import build_signal_and_paraphrase
from .token_client import TokenCoClient


def run_quick_eval(limit: int = 5, target_ratio: float = 0.5) -> Dict:
    signal_provider, paraphrase_fn = build_signal_and_paraphrase()
    engine = CosmosEngine(signal_provider=signal_provider, paraphrase_fn=paraphrase_fn)
    baselines = BaselineSuite()
    tokenc = TokenCoClient()

    results: List[Dict] = []
    for example in SAMPLE_BATCH[:limit]:
        text = example["text"].strip()
        query = example.get("query")

        def cosmos_with(toggles: Dict) -> Dict:
            res = engine.compress(
                text=text,
                query=query,
                target_ratio=target_ratio,
                run_baselines=False,
                toggles=toggles,
            )
            return {
                "metrics": res["metrics"],
                "quality": quality_score(text, res["compressed_text"]),
                "text": res["compressed_text"],
            }

        classic = cosmos_with(
            {
                "use_signal_scores": False,
                "paraphrase_mode": "none",
                "keep_code_blocks": True,
                "keep_role_markers": True,
            }
        )
        signal = cosmos_with(
            {
                "use_signal_scores": True,
                "signal_boost": 0.65,
                "paraphrase_mode": "none",
                "keep_code_blocks": True,
                "keep_role_markers": True,
            }
        )
        signal_para = cosmos_with(
            {
                "use_signal_scores": True,
                "signal_boost": 0.65,
                "paraphrase_mode": "llm",
                "keep_code_blocks": True,
                "keep_role_markers": True,
            }
        )

        tokenc_result = None
        if tokenc.available:
            tok_payload = tokenc.compress(text, aggressiveness=1 - target_ratio)
            if tok_payload.get("available"):
                tokenc_result = {
                    **tok_payload,
                    "quality": quality_score(text, tok_payload["text"]),
                }
            else:
                tokenc_result = tok_payload

        results.append(
            {
                "category": example["category"],
                "cosmos_classic": classic,
                "cosmos_signal": signal,
                "cosmos_signal_paraphrase": signal_para,
                "tokenc": tokenc_result,
            }
        )

    summary = {
        "count": len(results),
        "avg_quality_classic": round(
            sum(r["cosmos_classic"]["quality"] for r in results) / max(len(results), 1), 4
        ),
        "avg_quality_signal": round(
            sum(r["cosmos_signal"]["quality"] for r in results) / max(len(results), 1), 4
        ),
        "avg_quality_signal_paraphrase": round(
            sum(r["cosmos_signal_paraphrase"]["quality"] for r in results) / max(len(results), 1), 4
        ),
        "avg_savings_classic": round(
            sum(r["cosmos_classic"]["metrics"]["savings_percent"] for r in results) / max(len(results), 1),
            2,
        ),
        "avg_savings_signal": round(
            sum(r["cosmos_signal"]["metrics"]["savings_percent"] for r in results) / max(len(results), 1),
            2,
        ),
        "avg_savings_signal_paraphrase": round(
            sum(r["cosmos_signal_paraphrase"]["metrics"]["savings_percent"] for r in results)
            / max(len(results), 1),
            2,
        ),
    }
    return {"summary": summary, "examples": results}


if __name__ == "__main__":
    payload = run_quick_eval()
    print(json.dumps(payload, indent=2))
