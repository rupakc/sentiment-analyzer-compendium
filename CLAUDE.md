# Sentiment Analysis Comparison App

Compare 5 sentiment models side-by-side. React (`frontend/`) + FastAPI serverless (`api/`), one Vercel deploy.

## Run locally
- Backend: `python3 -m venv .venv && .venv/bin/pip install -r api/requirements.txt && .venv/bin/uvicorn api.index:app --reload`
- Frontend: `cd frontend && npm install && npm run dev` (proxies `/api` → `localhost:8000`)
- Train pickles: `.venv/bin/python scripts/train_models.py`

## Conventions
- One model = one file in `api/models/`, registered in `registry.py`. **Never modify existing models to add a new one** (Open/Closed). Use the `model-adder` agent.
- Uniform result shape `{label, confidence, scores}`; CRF additionally sets `aspects`.
- Secrets: `ANTHROPIC_API_KEY`, `HF_API_TOKEN` env vars. Missing key → model reports `available: false`, never crashes a request.
- Tests mock all remote (HF, Claude) calls — no network in CI.
- No torch/transformers in the backend (DistilBERT is a remote HF Inference API call) — keeps the function under Vercel's 250 MB limit.

## Test
- Backend: `.venv/bin/python -m pytest`
- Frontend: `cd frontend && npm test`

## Docs
- `RESEARCH.md` — survey of all model families. `THEORY.md` — math + worked examples.
- `docs/architecture.md` — system architecture. `docs/superpowers/specs/` + `plans/` — design + implementation plan.
