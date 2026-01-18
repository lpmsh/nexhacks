# COSMOS – Token-Aware Compressor

Facility-location compression pipeline for the Token Company hackathon. The FastAPI backend exposes `/compress`, `/compare`, and `/evaluate`; the static `/app` UI shows the three-pane demo (original spans, redundancy map, compressed prompt) plus baselines.

## Quick start

```bash
cd server
uv sync  # or pip install -r <(uv pip compile) if you prefer pip
uv run uvicorn main:app --reload
# open http://127.0.0.1:8000/app for the UI
```

### TokenCo API key

The Token Company compressor is optional; set `TOKENC_API_KEY` (or `TTC_API_KEY`) to enable `/compare` and the TokenCo baseline.

## Endpoints

- `GET /health` – liveness
- `POST /compress` – runs COSMOS compression with guardrails, clusters, metrics, and baselines
- `POST /compare` – COSMOS + TokenCo output (when the key is set)
- `POST /evaluate` – small curated eval set with quality vs. savings curve
- `GET /examples` – sample scenarios used in the demo/eval

## What’s implemented

- Chunker that keeps headings, questions, and recency; merges sentences into ~80–200 token spans.
- Query-aware facility-location selection under a token budget with novelty and entity/number boosts.
- Baselines: truncate, head+tail, random spans, simple dedup.
- Optional TokenCo (tokenc SDK) baseline.
- Evaluation runner over curated RAG/meeting/policy examples with quality and savings metrics.
- Frontend demo in `client/` mounted at `/app`.

## Track 1 enhancements

- Model-aware scoring: representation-drop signal rescales span weights (`use_signal_scores`, `signal_boost`).
- Hard guardrails: keep code/role markers and constraints by default (`keep_code_blocks`, `keep_role_markers`, plus existing must-keep keywords/numbers/entities).
- The facility-location core stays intact; new signals boost important spans without breaking budgets. Tune these via `/compress` toggles.
- Optional paraphrase pass: set `paraphrase_mode` to `heuristic` (built-in stopword shrink) or `llm` (requires a paraphrase_fn in the engine) to squeeze extra tokens after selection.

## Optional local LLM for signals/paraphrase

- Set `LOCAL_LLM_MODEL` to a local HF model folder to enable logit-based signals and LLM paraphrase (e.g., `export LOCAL_LLM_MODEL=~/models/llama3.2-3b`).
- The runtime defaults to Hugging Face (`LOCAL_LLM_RUNTIME=hf`). Install extras: `uv add torch transformers accelerate sentencepiece`.
- If the model/env is not available, COSMOS falls back to its built-in heuristic signals and no LLM paraphrase.

## Optional Groq for fast paraphrase

- If you have a Groq key, set `GROQ_API_KEY` (and optionally `GROQ_MODEL`, default `llama-3.1-8b-instant`). COSMOS will use Groq for the paraphrase stage while keeping local signals.

## Frontend + API

- Frontend is served at `/app` once the server is running (`http://127.0.0.1:8000/app`).
- API docs at `/docs`; health at `/health`.

## Local run (arm64, keep existing venv)

```bash
cd server
source .venv/bin/activate                      # arm64 Python 3.11 venv
export GROQ_API_KEY=...                        # paraphrase speed
export TOKENC_API_KEY=...                      # optional bear-1 baseline
unset LOCAL_LLM_MODEL                          # skip local scorer for fastest path
python -m uvicorn main:app --reload
```

Quick eval:
```bash
uv run python -m cosmos.quick_eval
```

Env template: see `server/.env.example` for the expected variables.
