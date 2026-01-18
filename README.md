# Irreduce
Irreduce compresses long prompts by splitting text into token spans, scoring each span for task relevance and global importance, then selecting the highest information-per-token spans under a tight budget while penalizing redundancy. The result keeps about 95% task performance while cutting roughly 90% of tokens, making large context inference far cheaper and more scalable.

## Demo
- UI: http://127.0.0.1:8000/app
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Requirements
- Python 3.13+

## Quick start
```bash
cd server
uv sync
uv run uvicorn main:app --reload
```
Open http://127.0.0.1:8000/app.

### Alternative (existing venv)
```bash
cd server
source .venv/bin/activate
python -m uvicorn main:app --reload
```

## What it does
- Chunks long context into spans and applies guardrails (headings, entities, code/role markers).
- Scores spans with novelty/entity/number boosts plus optional signal scoring.
- Runs greedy facility-location selection under a token budget.
- Optional paraphrase squeeze (heuristic, local LLM, or Groq).

## API endpoints
- `GET /health` - liveness
- `POST /compress` - main Irreduce compression
- `POST /compress/longbench` - LongBench-style compression
- `POST /compare` - Irreduce vs TokenCo baseline
- `POST /evaluate` - quality vs savings curve
- `GET /examples` - demo scenarios

## Configuration
Irreduce runs without keys; extra features unlock with these environment variables:
- `TOKENC_API_KEY` or `TTC_API_KEY` - TokenCo baseline in `/compare`
- `TOKENC_MODEL` - TokenCo model override for scripts
- `TOKEN_COMPANY_API_KEY` - required for the custom compressor in scripts
- `GROQ_API_KEY` (and optional `GROQ_MODEL`) - Groq paraphrase
- `LOCAL_LLM_MODEL` (and optional `LOCAL_LLM_RUNTIME=hf`) - local signals/paraphrase
- `OPENAI_API_KEY` - evaluation scripts
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` - LongBench vision script

Optional extras for local LLM:
```bash
uv add torch transformers accelerate sentencepiece
```

## Evaluation
```bash
cd server
uv run python -m cosmos.quick_eval
```

## Project structure
- `client/` - static demo UI
- `server/` - FastAPI API, compression engine, eval scripts

## External tools and libraries
- FastAPI
- Pydantic
- Uvicorn (via FastAPI standard)
- uv (Python package manager)
- Token Company tokenc SDK
- Requests
- Google GenAI SDK (Gemini)
- Pillow
- ChromaDB
- python-dotenv
- OpenAI Python SDK
- Groq API (optional)
- Hugging Face Transformers, PyTorch, Accelerate, SentencePiece (optional)
- Google Fonts (Fraunces, Manrope, IBM Plex Mono)
