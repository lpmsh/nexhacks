from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    toggles: Dict[str, object]
    target_ratio: Optional[float] = None


VARIANTS: Dict[str, Variant] = {
    "v1_single": Variant(
        name="v1_single",
        description="Task-aware baseline (LongBench v1 settings).",
        toggles={"task_mode": "task_aware"},
        target_ratio=0.4,
    ),
    "hybrid_w0_2": Variant(
        name="hybrid_w0_2",
        description="Hybrid task-aware + task-agnostic (weight=0.2).",
        toggles={"task_mode": "hybrid", "agnostic_weight": 0.2},
        target_ratio=0.4,
    ),
    "hybrid_w0_2_c0_85": Variant(
        name="hybrid_w0_2_c0_85",
        description="Hybrid weight=0.2 with higher contrast weight.",
        toggles={"task_mode": "hybrid", "agnostic_weight": 0.2, "contrast_weight": 0.85},
        target_ratio=0.4,
    ),
    "v1_q0_6_r0_35": Variant(
        name="v1_q0_6_r0_35",
        description="Task-aware with higher question weight and lower redundancy penalty.",
        toggles={"task_mode": "task_aware", "question_weight": 0.6, "redundancy_penalty": 0.35},
        target_ratio=0.4,
    ),
}


def get_variant(name: str) -> Variant:
    key = name.strip()
    if key not in VARIANTS:
        available = ", ".join(sorted(VARIANTS.keys()))
        raise KeyError(f"Unknown variant '{name}'. Available: {available}")
    return VARIANTS[key]


def list_variants() -> str:
    lines = []
    for variant in VARIANTS.values():
        ratio = f"target_ratio={variant.target_ratio}" if variant.target_ratio is not None else "target_ratio=default"
        lines.append(f"- {variant.name}: {variant.description} ({ratio})")
    return "\n".join(lines)
