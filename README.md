# Irreduce

Irreduce is a token-aware prompt compressor for long-context inputs. It preserves critical spans with guardrails and query-aware facility-location selection, then optionally paraphrases to hit strict budgets. The project ships with a FastAPI backend and a static demo UI.

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

## NexHacks readiness
- Project name: Irreduce
- Registration deadline: 3:00 PM Saturday, January 17, 2026 (Devpost)
- Submission deadline: 1:00 PM Sunday, January 18, 2026 (Devpost)
- Build window: 1:00 PM Saturday to 1:00 PM Sunday
- Team size: up to 4; list all members on Devpost
- Code + demo must be linked on Devpost
- External tools/libraries are credited below

<details>
<summary>NexHacks rules (reference)</summary>

### Dates
- Teams must register on Devpost by 3 PM Saturday, January 17, 2026.
- Final submissions are due 1:00 PM Sunday, January 18, 2026. Submissions are accepted only through Devpost.

### Eligibility
- NexHacks follows the NexHacks Rules, MLH Code of Conduct, and Carnegie Mellon University Student Handbook.
- Directors, organizers, volunteers, mentors, sponsors, vendors, or anyone involved in running NexHacks cannot submit projects.
- All projects must be built during NexHacks. Prior ideas are allowed, but pre-existing projects cannot be submitted.
- AI tools may be used to assist development, but fully AI-generated submissions are prohibited and may be disqualified.
- Maximum team size is four (4). All team members must be listed on Devpost to receive prizes.

### Project and submission rules
- Teams may include up to four participants; smaller teams and solo hackers are allowed.
- Only submit projects to NexHacks; cross-submissions to other hackathons are prohibited.
- All code must be uploaded and linked on Devpost before the deadline.
- Submissions with a demo but no accessible code will be disqualified.
- Public frameworks, libraries, and AI tools are allowed; copied or uncredited code will result in disqualification.
- Each participant may contribute to only one project.
- Credit all external tools or libraries in your README.
- There is no limit on the number of non-sponsor tracks you may submit to.
- Work shown during judging must have been built between 1:00 PM Saturday and 1:00 PM Sunday.
- No coding before 1:00 PM Saturday, but ideation is allowed.
- Judging will be in-person expo style; at least one team member must be present to qualify for prizes.
- You may submit to up to 2 sponsor tracks; MLH-labeled challenges do not count toward this limit.

### Prizes
- To claim prizes, your team must demo during judging and attend the closing ceremony.
- If a team member cannot attend due to exceptional circumstances, NexHacks may arrange to send their prize.

### Judging criteria
- Innovation and originality
- Technical execution
- Impact and scalability
- Design and user experience
- Presentation and demo

### What we do not judge on
- Code perfection or formatting
- Whether the idea is 100% original
- Pitch deck polish
- Prior experience or credentials
- Team size (unless >4)
- Hours of sleep
</details>

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
