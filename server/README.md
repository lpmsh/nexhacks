# Irreduce server

FastAPI backend for Irreduce (COSMOS compression engine). Serves the API and mounts the demo UI at `/app`.

## Run locally
```bash
cd server
uv sync
uv run uvicorn main:app --reload
```
Open http://127.0.0.1:8000/app.

## Endpoints
- `/health` - liveness
- `/compress` - main compression API
- `/compress/longbench` - LongBench-style compression
- `/compare` - compare against TokenCo
- `/evaluate` - quick eval runner
- `/examples` - demo scenarios
- `/docs` - OpenAPI docs

## Configuration and credits
See `../README.md` for environment variables, evaluation commands, and external tool credits.
