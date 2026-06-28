# Sentiment Lab — Sentiment Analysis Comparison App

Run the same text through one representative of each sentiment-analysis family
(1990s → today) and compare verdicts, confidence, and *why* each model decided
what it did. Built as a 20-year retrospective made tangible.

| Family | Model | Hosting | Explanation |
|--------|-------|---------|-------------|
| Lexicon/rules | VADER | local | native (word valences) |
| Classical ML | Logistic Regression (TF-IDF) | local | native (token weights) |
| Sequence | CRF (aspect extraction) | local | native (tagged aspects) |
| Deep learning | DistilBERT (SST-2) | HF Inference API | AI-narrated |
| LLM | Claude | Anthropic API | native (model's rationale) |

## Architecture

React + TypeScript + Vite SPA (`frontend/`) calls a FastAPI Python backend
(`api/`) as Vercel serverless functions. Light models run in-function from
committed pickles in `artifacts/`; DistilBERT and Claude are remote API calls,
keeping the function under Vercel's 250 MB limit. History is per-browser
localStorage — no database, no auth. See `docs/architecture.md`.

## Tabs

- **Compare** — enter text, multi-select models, get a card per model; click a card for the "why" modal.
- **Guides** — how each model family works (sourced from `RESEARCH.md` / `THEORY.md`).
- **History** — past analyses, stored in your browser.

## Run locally

```bash
# Backend
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.txt
.venv/bin/python scripts/train_models.py        # generates artifacts/*.pkl
.venv/bin/uvicorn api.index:app --reload        # http://localhost:8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev        # http://localhost:5173
```

VADER / LogReg / CRF work offline. DistilBERT and Claude need API keys (below);
without them those two cards show "unavailable" by design.

## Deploy to Vercel

1. Push this repo to GitHub.
2. Import the repo in Vercel — it auto-detects `vercel.json`.
3. In **Project Settings → Environment Variables**, set:
   - `ANTHROPIC_API_KEY` — enables the Claude LLM model + AI-narrated explanations.
   - `HF_API_TOKEN` — enables the DistilBERT model.
4. Deploy. The build runs the `frontend` build and deploys `api/` as a Python
   serverless function. Missing keys degrade gracefully (those models report
   `available: false`).

## Tests

```bash
.venv/bin/python -m pytest          # backend (22 tests)
cd frontend && npm test             # frontend (Vitest)
```

## Add a model

One file in `api/models/`, registered in `registry.py` — never modify existing
models (Open/Closed). Use the `model-adder` Claude agent in `.claude/agents/`.
