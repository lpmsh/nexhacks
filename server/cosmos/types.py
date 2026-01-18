from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Span:
    """Lightweight representation of a text span."""

    id: int
    text: str
    token_count: int
    is_heading: bool = False
    is_question: bool = False
    must_keep: bool = False
    weight: float = 1.0
    cluster: Optional[int] = None
    score: float = 0.0
    gain: float = 0.0
    selected: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "token_count": self.token_count,
            "is_heading": self.is_heading,
            "is_question": self.is_question,
            "must_keep": self.must_keep,
            "weight": self.weight,
            "cluster": self.cluster,
            "score": round(self.score, 4),
            "gain": round(self.gain, 4),
            "selected": self.selected,
            "metadata": self.metadata,
        }
