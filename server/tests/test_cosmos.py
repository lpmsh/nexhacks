import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cosmos.compressor import CosmosEngine
from cosmos.evaluation import EvaluationRunner
from cosmos.baselines import BaselineSuite
from cosmos.token_client import TokenCoClient


def test_cosmos_compress_runs():
    engine = CosmosEngine()
    text = (
        "Heading: Launch Readiness\n\n"
        "We are preparing the launch. Risks include latency, duplicate charges, and missing alerts. "
        "Controls include idempotency keys and circuit breakers."
    )
    result = engine.compress(text=text, query="What are the risks?", token_budget=40, run_baselines=False)
    assert result["compressed_text"]
    assert result["metrics"]["compressed_tokens"] <= result["metrics"]["original_tokens"]
    assert result["metrics"]["savings_percent"] >= 0


def test_evaluation_runner_smoke():
    engine = CosmosEngine()
    baselines = BaselineSuite()
    tokenc = TokenCoClient(api_key="dummy")  # stays offline friendly
    runner = EvaluationRunner(engine, baselines, tokenc)
    result = runner.run(budgets=[0.4], include_tokenc=False)
    assert result["summary"]["examples"] >= 1
