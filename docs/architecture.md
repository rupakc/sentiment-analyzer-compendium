# Architecture

## Overview

Single Vercel project, monorepo. A React SPA talks to a FastAPI backend exposed
as Vercel Python serverless functions under `/api`.

```
┌─────────────── Vercel ───────────────┐
│  React SPA (Vite)  ──fetch──►  /api/* │   Python serverless (FastAPI)
│  3 tabs            ◄──JSON───          │   ├─ model registry
└───────────────────────────────────────┘   ├─ VADER     (local, light)
                                             ├─ LogReg    (local, sklearn pickle)
        external calls ◄────────────────────┼─ CRF        (local, crfsuite pickle)
                                             ├─ DistilBERT → HF Inference API
                                             └─ LLM        → Claude API
```

## Backend (SOLID)

- `api/models/base.py` — `SentimentModel` protocol + `AnalysisResult`,
  `Explanation`, `Aspect` dataclasses. The single interface every model implements.
- `api/models/<id>.py` — one model per file, each with one responsibility.
- `api/models/registry.py` — `REGISTRY` dict + `get_model()` / `list_models()`.
  Single source of truth for the API and the frontend dropdown. Adding a model
  touches only its own file + the registry (Open/Closed).
- `api/runner.py` — `analyze_many()` runs selected models concurrently with
  per-model error isolation (one failing model never breaks the others).
- `api/index.py` — FastAPI app: `GET /api/models`, `POST /api/analyze`,
  `POST /api/explain`.
- `api/schemas.py` — pydantic request validation.

### Uniform contract

Every model returns `AnalysisResult{model_id, label, confidence, scores, aspects?,
available, error?}`. This makes cards comparable. CRF additionally fills `aspects`.

### Explanations

`explain()` returns either model-native evidence (VADER word valences, LogReg
token weights, CRF tagged aspects) or AI-narrated prose (DistilBERT, LLM), tagged
via `explanation_type` so the UI labels it honestly. Lazy-loaded — only fetched
when a user opens a card's modal.

## Data flow

1. `GET /api/models` populates the multi-select (disabling unavailable models).
2. `POST /api/analyze {text, model_ids}` → `analyze_many` runs models in a thread
   pool → array of `AnalysisResult` → one card each.
3. Click a card → `POST /api/explain {text, model_id}` → `Explanation` → modal.
4. Each successful run is appended to `localStorage` (History tab).

## Frontend

- React + TypeScript + Vite + Tailwind. `types.ts` mirrors backend schemas.
- `api.ts` — typed fetch client. `hooks/useHistory.ts` — localStorage persistence.
- Components: `CompareTab`, `MultiSelect`, `ModelCard`, `ExplanationModal`,
  `GuidesTab`, `HistoryTab`. Each is focused and independently testable.

## Hosting rationale

PyTorch + transformers (~1–2 GB) exceed Vercel's 250 MB serverless limit, so
DistilBERT is served via the Hugging Face Inference API and the LLM via the
Anthropic API. Light models (VADER, sklearn LogReg, crfsuite CRF) run in-function
from small committed pickles. One deploy target, all five families.

## Failure modes

- Missing API key → model `available: false`; others still run.
- Remote API error/timeout → that model's card shows an error; rest succeed.
- localStorage quota/parse error → history degrades to empty, never blocks Compare.
