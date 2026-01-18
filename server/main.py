from pathlib import Path
from typing import Dict

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from components.api_handler import APIHandler
from cosmos import (
    BaselineSuite,
    CosmosEngine,
    EvaluationRunner,
    LongBenchEngine,
    TokenCoClient,
)
from cosmos.demo_data import SAMPLE_BATCH
from cosmos.local_llm import build_signal_and_paraphrase
from schemas.compress import (
    CompareRequest,
    CompressionRequest,
    CompressionResponse,
    EvaluationRequest,
    EvaluationResponse,
    LongBenchCompressionRequest,
    LongBenchCompressionResponse,
)

app = FastAPI(
    title="COSMOS Compression API",
    description="Facility-location compressor with baselines and a live demo",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

root_dir = Path(__file__).resolve().parent.parent
client_dir = root_dir / "client"
if client_dir.exists():
    app.mount("/app", StaticFiles(directory=client_dir, html=True), name="app")

signal_provider, paraphrase_fn = build_signal_and_paraphrase()
engine = CosmosEngine(signal_provider=signal_provider, paraphrase_fn=paraphrase_fn)
longbench_engine = LongBenchEngine()
baselines = BaselineSuite()
tokenc_client = TokenCoClient()
evaluator = EvaluationRunner(engine, baselines, tokenc_client)


@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/compress", response_model=CompressionResponse)
async def compress(request: CompressionRequest) -> CompressionResponse:
    result = engine.compress(
        text=request.text,
        query=request.query,
        token_budget=request.token_budget,
        target_ratio=request.target_ratio,
        keep_last_n=request.keep_last_n,
        toggles=request.toggles.model_dump(),
        run_baselines=request.run_baselines,
        baseline_suite=baselines,
        seed=request.seed or 13,
    )
    return CompressionResponse(**result)


@app.post("/compress/longbench", response_model=LongBenchCompressionResponse)
async def compress_longbench(
    request: LongBenchCompressionRequest,
) -> LongBenchCompressionResponse:
    result = longbench_engine.compress_longbench(
        context=request.context,
        question=request.question,
        choices=request.choices,
        token_budget=request.token_budget,
        target_ratio=request.target_ratio,
        seed=request.seed or 13,
        toggles=request.toggles.model_dump(),
    )
    return LongBenchCompressionResponse(**result)


@app.post("/compare")
async def compare(request: CompareRequest) -> Dict:
    # Allow per-request TokenCo API key to avoid hardcoded keys.
    tokenc = (
        tokenc_client if not request.api_key else TokenCoClient(api_key=request.api_key)
    )
    cosmos_result = engine.compress(
        text=request.text,
        query=request.query,
        token_budget=request.token_budget,
        target_ratio=request.target_ratio,
        keep_last_n=1,
        toggles=request.toggles.model_dump(),
        run_baselines=True,
        baseline_suite=baselines,
        seed=request.seed or 13,
    )
    tokenc_result = tokenc.compress(
        request.text,
        aggressiveness=request.aggressiveness,
        max_output_tokens=request.max_output_tokens,
        min_output_tokens=request.min_output_tokens,
        model=request.model,
        api_key_override=request.api_key,
    )
    return {
        "cosmos": cosmos_result,
        "tokenc": tokenc_result,
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    result = evaluator.run(
        budgets=request.budgets or [0.35, 0.5, 0.7],
        quality_threshold=request.quality_threshold,
        include_tokenc=request.include_tokenc,
    )
    return EvaluationResponse(**result)


@app.get("/examples")
async def examples() -> Dict:
    return {"examples": SAMPLE_BATCH}
