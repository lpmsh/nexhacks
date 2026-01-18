import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cosmos.longbench_compressor import LongBenchEngine


def test_longbench_engine_smoke():
    engine = LongBenchEngine()
    context = (
        "Launch Update:\n\n"
        "The team flagged latency spikes and missing alerts as top risks. "
        "Mitigations include circuit breakers and paging rotations. "
        "The launch window is August 12."
    )
    question = "Which risk was mentioned?"
    choices = ["Latency spikes", "Hiring delays", "Pricing errors", "Supplier issues"]
    result = engine.compress_longbench(
        context=context,
        question=question,
        choices=choices,
        token_budget=25,
        target_ratio=0.4,
        seed=7,
    )
    assert result["compressed_context"]
    assert (
        result["metrics"]["compressed_tokens"] <= result["metrics"]["original_tokens"]
    )


def test_longbench_engine_deterministic():
    engine = LongBenchEngine()
    context = (
        "Spec:\n\n"
        "We must keep names, dates, and numbers. The deadline is September 30. "
        "The API should return 200 for valid inputs and 400 for invalid ones."
    )
    question = "What is the deadline?"
    choices = ["September 30", "October 30", "November 30", "December 30"]
    first = engine.compress_longbench(
        context=context,
        question=question,
        choices=choices,
        token_budget=30,
        target_ratio=0.5,
        seed=11,
    )
    second = engine.compress_longbench(
        context=context,
        question=question,
        choices=choices,
        token_budget=30,
        target_ratio=0.5,
        seed=11,
    )
    assert first["compressed_context"] == second["compressed_context"]


def test_longbench_cosmos_compatible_interface():
    """Test that the Cosmos-compatible compress() method works."""
    engine = LongBenchEngine()
    text = (
        "Launch Update:\n\n"
        "The team flagged latency spikes and missing alerts as top risks. "
        "Mitigations include circuit breakers and paging rotations. "
        "The launch window is August 12."
    )
    query = "Which risk was mentioned?"
    result = engine.compress(
        text=text,
        query=query,
        token_budget=25,
        target_ratio=0.4,
        seed=7,
    )
    # Should return Cosmos-compatible response format
    assert "compressed_text" in result
    assert "spans" in result
    assert "clusters" in result
    assert "baselines" in result
    assert result["compressed_text"]
    assert (
        result["metrics"]["compressed_tokens"] <= result["metrics"]["original_tokens"]
    )
