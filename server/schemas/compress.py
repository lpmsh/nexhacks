from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ToggleSettings(BaseModel):
    keep_numbers_entities: bool = True
    keep_headings: bool = True
    keep_code_blocks: bool = True
    keep_role_markers: bool = True
    alpha: float = Field(0.35, description="Query awareness mixing parameter")
    entity_boost: float = 0.25
    novelty_boost: float = 0.35
    use_signal_scores: bool = True
    signal_boost: float = 0.65
    paraphrase_mode: str = Field(
        "none",
        description="Paraphrase pass after selection: none|heuristic|llm. LLM requires a provided paraphrase_fn in engine.",
    )
    cluster_threshold: float = 0.62
    must_keep_keywords: Optional[List[str]] = None


class LongBenchToggleSettings(BaseModel):
    target_span_tokens: int = 200
    min_span_tokens: int = 80
    max_span_tokens: int = 260
    keep_headings: bool = True
    keep_code_blocks: bool = True
    keep_role_markers: bool = True
    keep_last_n: int = 1
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    question_weight: float = 0.45
    option_weight: float = 1.0
    contrast_weight: float = 0.7
    entity_boost: float = 0.2
    number_boost: float = 0.2
    must_keep_boost: float = 0.35
    query_hit_boost: float = 0.2
    min_query_hits: int = 2
    redundancy_penalty: float = 0.45
    window_token_budget: int = 1800
    window_top_k: int = 3
    window_max_fraction: float = 0.35
    neighbor_span_count: int = 1
    must_keep_keywords: Optional[List[str]] = None


class CompressionRequest(BaseModel):
    text: str
    query: Optional[str] = None
    target_ratio: float = Field(0.5, ge=0.05, le=0.95)
    token_budget: Optional[int] = Field(default=None, gt=0)
    keep_last_n: int = 1
    run_baselines: bool = True
    use_tokenc: bool = False
    seed: Optional[int] = 13
    toggles: ToggleSettings = ToggleSettings()


class LongBenchCompressionRequest(BaseModel):
    context: str
    question: str
    choices: List[str]
    target_ratio: float = Field(0.5, ge=0.05, le=0.95)
    token_budget: Optional[int] = Field(default=None, gt=0)
    seed: Optional[int] = 13
    toggles: LongBenchToggleSettings = LongBenchToggleSettings()


class CompareRequest(BaseModel):
    text: str
    query: Optional[str] = None
    target_ratio: float = Field(0.5, ge=0.05, le=0.95)
    token_budget: Optional[int] = Field(default=None, gt=0)
    aggressiveness: float = Field(0.5, ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    min_output_tokens: Optional[int] = Field(default=None, gt=0)
    model: str = "bear-1"
    api_key: Optional[str] = Field(
        default=None, description="TokenCo API key; falls back to env TOKENC_API_KEY"
    )
    seed: Optional[int] = 13
    toggles: ToggleSettings = ToggleSettings()


class EvaluationRequest(BaseModel):
    budgets: Optional[List[float]] = Field(
        default_factory=lambda: [0.35, 0.5, 0.7],
        description="List of target ratios (0-1) or absolute token budgets (ints).",
    )
    quality_threshold: float = 0.72
    include_tokenc: bool = False


class SpanResponse(BaseModel):
    id: int
    text: str
    token_count: int
    weight: float
    gain: float
    is_heading: bool
    is_question: bool
    must_keep: bool
    cluster: Optional[int] = None
    selected: bool = False
    metadata: Dict = Field(default_factory=dict)


class BaselineResponse(BaseModel):
    name: str
    text: str
    metrics: Dict
    quality: Optional[float] = None


class CompressionResponse(BaseModel):
    compressed_text: str
    selected_spans: List[SpanResponse]
    spans: List[SpanResponse]
    clusters: List[Dict]
    metrics: Dict
    budget: int
    input_tokens: int
    source_tokens: int
    span_counts: Dict
    baselines: List[BaselineResponse] = Field(default_factory=list)


class LongBenchCompressionResponse(BaseModel):
    compressed_context: str
    selected_spans: List[SpanResponse]
    spans: List[SpanResponse]
    metrics: Dict
    budget: int
    input_tokens: int
    span_counts: Dict


class EvaluationResponse(BaseModel):
    summary: Dict
    curve: List[Dict]
    examples: List[Dict]
