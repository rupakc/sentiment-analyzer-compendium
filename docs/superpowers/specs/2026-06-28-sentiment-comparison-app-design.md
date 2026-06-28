# Design Spec — Sentiment Analysis Comparison App

**Date:** 2026-06-28
**Status:** Approved (design), pending implementation plan
**Companions:** `RESEARCH.md` (survey), `THEORY.md` (mechanics)

## 1. Goal

A web app that runs the same input text through one representative of each
sentiment-analysis family side-by-side, shows each model's verdict + confidence,
and explains *why* each model predicted what it did. Plus a Guides tab (teaching
content) and a History tab (past runs). React frontend, Python backend, deployed
as a single Vercel project.

## 2. Resolved decisions (from brainstorming)

| Decision | Choice | Consequence |
|----------|--------|-------------|
| Model hosting | **Hybrid** | Light models (VADER, LogReg, CRF) run in Vercel Python serverless; DistilBERT via Hugging Face Inference API; LLM via Claude API. Stays under Vercel's 250 MB limit. |
| History / identity | **Per-browser localStorage** | No database, no auth, no accounts. "User" = this browser. |
| Model lineup | **5 representatives, uniform polarity** | VADER, Logistic Regression (TF-IDF), DistilBERT, Claude LLM emit polarity. CRF emits aspect-extraction (distinct card). HMM lives in Guides only. |
| Explanations | **Native where cheap, Claude-narrated for DistilBERT** | VADER/LogReg/CRF show real internal evidence; DistilBERT + LLM get Claude-generated prose. UI labels each as native vs. AI-narrated. |
| Pretrained models | **Train tiny models on a small public set (e.g. SST-2 subset), commit `.pkl`** | Plus a documented training script in `scripts/`. |
| API keys | **`ANTHROPIC_API_KEY`, `HF_API_TOKEN` as Vercel env vars** | App degrades gracefully — models needing a missing key report "unavailable" rather than erroring the whole request. |

## 3. Architecture

Single Vercel project, monorepo:

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

- Vercel serves the built SPA as static assets and `api/` as serverless functions
  (native pattern, no extra infra).
- Light models ship as small pretrained pickles committed to the repo.
- Heavy/remote models are external API calls, keeping the function slim.

## 4. Backend — SOLID model layer

One interface, five implementations, one registry. Adding a model never modifies
existing ones (Open/Closed). Each model is testable in isolation (single
responsibility, depends only on the `SentimentModel` protocol).

```python
# api/models/base.py
class SentimentModel(Protocol):
    id: str; name: str; family: str
    def analyze(self, text: str) -> AnalysisResult: ...
    def explain(self, text: str, result: AnalysisResult) -> Explanation: ...
```

- `AnalysisResult` = `{label, confidence, scores}` — uniform so every card is
  comparable. CRF additionally fills optional `aspects: [{term, polarity}]`.
- Each model in its own file: `vader.py`, `logreg.py`, `crf.py`, `distilbert.py`,
  `llm.py`. Registered in a `REGISTRY` dict keyed by `id` — single source of truth
  for both the API and the frontend dropdown.
- `explain()` returns native evidence (VADER word scores, LogReg top coefficients,
  CRF features) or Claude-narrated prose (DistilBERT, LLM), with an
  `explanation_type` field so the UI labels it honestly.

### Model representatives

| id | Family | Output | Explanation |
|----|--------|--------|-------------|
| `vader` | Lexicon/rules | polarity | native: per-word valence contributions |
| `logreg` | Classical ML | polarity | native: top contributing tokens (coef × tf-idf) |
| `crf` | Sequence | aspects + per-aspect polarity | native: features that fired |
| `distilbert` | Deep learning | polarity | Claude-narrated |
| `llm` | LLM | polarity (+ nuance) | native: Claude's own explanation |

## 5. API contract

| Endpoint | Request | Response |
|----------|---------|----------|
| `GET /api/models` | — | `[{id, name, family, description, available}]` |
| `POST /api/analyze` | `{text, model_ids[]}` | `{results: [AnalysisResult...]}` (selected models run concurrently) |
| `POST /api/explain` | `{text, model_id, result}` | `Explanation` (lazy — only when a card is clicked) |

Splitting `analyze` from `explain` keeps comparison fast and only pays the
Claude-narration cost when a user opens a modal. A model whose key is missing
returns `available: false` and is shown disabled in the dropdown.

## 6. Frontend — 3 tabs

React + TypeScript + Vite + Tailwind. Accessible (keyboard-navigable modal, ARIA,
focus trap). `frontend-design` skill applied at build time for a distinctive look.
React state + localStorage only — no state-management library.

1. **Compare** — textarea + multi-select model dropdown + Analyze button → grid of
   result cards (label, confidence bar, family badge). Click a card → modal showing
   the explanation from `/api/explain`. On success, the run is appended to history.
2. **Guides** — RESEARCH.md + THEORY.md content rendered per family, including the
   HMM Viterbi teaching demo. Markdown shipped as static content.
3. **History** — past runs from localStorage (text snippet, models, verdicts,
   timestamp); click to re-open; clear button.

## 7. Claude tooling & scaffolding

```
CLAUDE.md                      # overview, conventions, run/deploy
.claude/settings.json          # permissions, env var names, hooks config
.claude/agents/                # model-adder.md (scaffolds a new SentimentModel)
.claude/skills/                # add-model skill following the SOLID pattern
.claude/hooks/                 # post-edit ruff/eslint format hook
README.md                      # setup, run, deploy
docs/architecture.md           # architecture, diagrams, data flow
docs/superpowers/specs/...     # this spec
vercel.json                    # routes: SPA + /api
api/requirements.txt           # light deps only (no torch)
scripts/train_models.py        # trains + pickles logreg & crf
frontend/                      # React app
api/                           # FastAPI serverless
```

## 8. Error handling

- Missing API key → model reported `available: false`; never crashes a multi-model
  request. Other selected models still return.
- Remote API failure (HF/Claude timeout, rate limit) → that model's card shows an
  error state; the rest succeed. Per-model try/except in the concurrent runner.
- Input validation: non-empty text, length cap, model_ids must exist in registry.
- localStorage quota/parse errors → history degrades to empty, never blocks Compare.

## 9. Testing

- **Backend:** pytest per model (`analyze` returns valid `AnalysisResult` shape,
  known-sentiment sanity checks for VADER/LogReg), registry integrity, API
  endpoint tests with remote calls mocked. CRF/DistilBERT/LLM tests mock the
  heavy/remote parts.
- **Frontend:** component tests for the card grid, modal, and history persistence
  (Vitest + Testing Library). One happy-path flow test.
- Smallest-useful-check philosophy: one runnable test per non-trivial unit, no
  per-function suites.

## 10. Deliberately out of scope (YAGNI)

- No database, no auth, no accounts (localStorage covers history).
- No local PyTorch, no training-at-runtime (pretrained pickles + remote APIs).
- No SVM/NB/HMM as separate comparison models (one classical representative;
  others in Guides).
- No state-management library, no SSR, no i18n.

## 11. Open items

None blocking. Both confirmation items resolved in §2 (train+commit pickles with a
script; env-var keys with graceful degradation).
