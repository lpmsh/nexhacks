import asyncio
import gzip
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosmos.timing")

# Thread pool for CPU-intensive compression operations
# This prevents blocking the async event loop
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compress_")

app = FastAPI(
    title="COSMOS Compression API",
    description="Facility-location compressor with baselines and a live demo",
    version="0.2.0",
)


class GzipRequestMiddleware(BaseHTTPMiddleware):
    """Middleware to decompress gzip-encoded request bodies."""

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-encoding") == "gzip":
            body = await request.body()
            original_size = len(body)
            t0 = time.perf_counter()
            try:
                decompressed = gzip.decompress(body)
                decompress_time = time.perf_counter() - t0
                logger.info(
                    f"[GZIP] Decompressed request: {original_size:,} -> {len(decompressed):,} bytes "
                    f"({len(decompressed) / original_size:.1f}x) in {decompress_time * 1000:.1f}ms"
                )
                # Create a new scope with decompressed body
                request._body = decompressed
            except Exception as e:
                logger.error(f"[GZIP] Failed to decompress: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Failed to decompress gzip body: {str(e)}"},
                )
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware for compressing responses (min 1KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add middleware to handle gzip-compressed request bodies
app.add_middleware(GzipRequestMiddleware)

root_dir = Path(__file__).resolve().parent.parent
client_dir = root_dir / "client"
if client_dir.exists():
    app.mount("/app", StaticFiles(directory=client_dir, html=True), name="app")

logger.info("Initializing COSMOS engine...")
t0 = time.perf_counter()
signal_provider, paraphrase_fn = build_signal_and_paraphrase()
engine = CosmosEngine(signal_provider=signal_provider, paraphrase_fn=paraphrase_fn)
longbench_engine = LongBenchEngine()
baselines = BaselineSuite()
tokenc_client = TokenCoClient()
evaluator = EvaluationRunner(engine, baselines, tokenc_client)
logger.info(f"Engine initialized in {time.perf_counter() - t0:.2f}s")


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log timing for all requests."""
    start = time.perf_counter()

    # Log request info
    content_length = request.headers.get("content-length", "unknown")
    content_encoding = request.headers.get("content-encoding", "none")
    logger.info(
        f"[REQ] {request.method} {request.url.path} - Size: {content_length} bytes, Encoding: {content_encoding}"
    )

    response = await call_next(request)

    elapsed = time.perf_counter() - start
    logger.info(
        f"[RES] {request.method} {request.url.path} - Status: {response.status_code}, Time: {elapsed:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"

    return response


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up thread pool on shutdown."""
    logger.info("Shutting down thread pool...")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shut down.")


@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


def _run_compression(
    text: str,
    query: str | None,
    token_budget: int | None,
    target_ratio: float,
    keep_last_n: int,
    toggles: dict,
    run_baselines: bool,
    seed: int,
) -> dict:
    """Run compression synchronously - meant to be called in thread pool."""
    return longbench_engine.compress(
        text=text,
        query=query,
        token_budget=token_budget,
        target_ratio=target_ratio,
        keep_last_n=keep_last_n,
        toggles=toggles,
        run_baselines=run_baselines,
        baseline_suite=baselines,
        seed=seed,
    )


@app.post("/compress", response_model=CompressionResponse)
async def compress(request: CompressionRequest) -> CompressionResponse:
    timings = {}
    total_start = time.perf_counter()

    # Log input size
    text_len = len(request.text)
    text_tokens_approx = len(request.text.split())
    logger.info(f"[COMPRESS] Input: {text_len:,} chars, ~{text_tokens_approx:,} tokens")

    # Run compression in thread pool to avoid blocking the event loop
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        partial(
            _run_compression,
            text=request.text,
            query=request.query,
            token_budget=request.token_budget,
            target_ratio=request.target_ratio,
            keep_last_n=request.keep_last_n,
            toggles=request.toggles.model_dump(),
            run_baselines=request.run_baselines,
            seed=request.seed or 13,
        ),
    )
    timings["compression"] = time.perf_counter() - t0

    # Time response construction
    t0 = time.perf_counter()
    response = CompressionResponse(**result)
    timings["response_construction"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - total_start

    # Log timing breakdown
    logger.info(
        f"[COMPRESS] Timings - "
        f"compression: {timings['compression']:.3f}s, "
        f"response_construction: {timings['response_construction']:.3f}s, "
        f"total: {timings['total']:.3f}s"
    )

    return response


def _run_longbench_compression(
    context: str,
    question: str,
    choices: list,
    token_budget: int | None,
    target_ratio: float,
    seed: int,
    toggles: dict,
) -> dict:
    """Run longbench compression synchronously - meant to be called in thread pool."""
    return longbench_engine.compress_longbench(
        context=context,
        question=question,
        choices=choices,
        token_budget=token_budget,
        target_ratio=target_ratio,
        seed=seed,
        toggles=toggles,
    )


@app.post("/compress/longbench", response_model=LongBenchCompressionResponse)
async def compress_longbench(
    request: LongBenchCompressionRequest,
) -> LongBenchCompressionResponse:
    t0 = time.perf_counter()

    # Run in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        partial(
            _run_longbench_compression,
            context=request.context,
            question=request.question,
            choices=request.choices,
            token_budget=request.token_budget,
            target_ratio=request.target_ratio,
            seed=request.seed or 13,
            toggles=request.toggles.model_dump(),
        ),
    )

    logger.info(f"[LONGBENCH] Compression took {time.perf_counter() - t0:.3f}s")
    return LongBenchCompressionResponse(**result)


def _run_compare(
    text: str,
    query: str | None,
    token_budget: int | None,
    target_ratio: float,
    toggles: dict,
    seed: int,
    tokenc_client: TokenCoClient,
    aggressiveness: float,
    max_output_tokens: int | None,
    min_output_tokens: int | None,
    model: str,
    api_key: str | None,
) -> dict:
    """Run comparison synchronously - meant to be called in thread pool."""
    cosmos_result = engine.compress(
        text=text,
        query=query,
        token_budget=token_budget,
        target_ratio=target_ratio,
        keep_last_n=1,
        toggles=toggles,
        run_baselines=True,
        baseline_suite=baselines,
        seed=seed,
    )
    tokenc_result = tokenc_client.compress(
        text,
        aggressiveness=aggressiveness,
        max_output_tokens=max_output_tokens,
        min_output_tokens=min_output_tokens,
        model=model,
        api_key_override=api_key,
    )
    return {
        "cosmos": cosmos_result,
        "tokenc": tokenc_result,
    }


@app.post("/compare")
async def compare(request: CompareRequest) -> Dict:
    total_start = time.perf_counter()

    text_len = len(request.text)
    logger.info(f"[COMPARE] Input: {text_len:,} chars")

    # Allow per-request TokenCo API key to avoid hardcoded keys.
    tokenc = (
        tokenc_client if not request.api_key else TokenCoClient(api_key=request.api_key)
    )

    # Run in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        partial(
            _run_compare,
            text=request.text,
            query=request.query,
            token_budget=request.token_budget,
            target_ratio=request.target_ratio,
            toggles=request.toggles.model_dump(),
            seed=request.seed or 13,
            tokenc_client=tokenc,
            aggressiveness=request.aggressiveness,
            max_output_tokens=request.max_output_tokens,
            min_output_tokens=request.min_output_tokens,
            model=request.model,
            api_key=request.api_key,
        ),
    )

    logger.info(f"[COMPARE] Total time: {time.perf_counter() - total_start:.3f}s")

    return result


def _run_evaluation(
    budgets: list,
    quality_threshold: float,
    include_tokenc: bool,
) -> dict:
    """Run evaluation synchronously - meant to be called in thread pool."""
    return evaluator.run(
        budgets=budgets,
        quality_threshold=quality_threshold,
        include_tokenc=include_tokenc,
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    t0 = time.perf_counter()

    # Run in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        partial(
            _run_evaluation,
            budgets=request.budgets or [0.35, 0.5, 0.7],
            quality_threshold=request.quality_threshold,
            include_tokenc=request.include_tokenc,
        ),
    )

    logger.info(f"[EVALUATE] Evaluation took {time.perf_counter() - t0:.3f}s")
    return EvaluationResponse(**result)


@app.get("/examples")
async def examples() -> Dict:
    return {"examples": SAMPLE_BATCH}
