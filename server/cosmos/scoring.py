from typing import Dict, Optional

from .chunker import count_tokens
from .embedder import SimpleEmbedder, cosine


def build_metrics(
    original_text: str,
    compressed_text: str,
    coverage_score: float = 0.0,
    original_tokens_override: Optional[int] = None,
    compressed_tokens_override: Optional[int] = None,
) -> Dict:
    original_tokens = original_tokens_override or max(count_tokens(original_text), 1)
    compressed_tokens = (
        compressed_tokens_override
        if compressed_tokens_override is not None
        else count_tokens(compressed_text)
    )
    savings = max(0.0, 1 - (compressed_tokens / original_tokens)) if original_tokens else 0.0
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "savings_percent": round(savings * 100, 2),
        "compression_ratio": round(compressed_tokens / original_tokens, 3) if original_tokens else 0,
        "coverage_score": round(coverage_score, 4),
    }


def quality_score(full_text: str, compressed_text: str) -> float:
    if not full_text or not compressed_text:
        return 0.0
    embedder = SimpleEmbedder()
    embedder.fit([full_text, compressed_text])
    ref = embedder.encode(full_text)
    hyp = embedder.encode(compressed_text)
    return round(cosine(ref, hyp), 4)
