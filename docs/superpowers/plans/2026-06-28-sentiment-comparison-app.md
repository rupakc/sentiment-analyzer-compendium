# Sentiment Analysis Comparison App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Vercel-deployed web app that runs input text through 5 sentiment-analysis models side-by-side, shows each verdict + confidence, and explains each prediction in a modal; plus Guides and History tabs.

**Architecture:** Monorepo on a single Vercel project. React+TS+Vite SPA (static) calls a Python FastAPI serverless backend under `/api`. Light models (VADER, LogReg, CRF) run in-function from committed pickles; DistilBERT calls the Hugging Face Inference API; the LLM calls the Claude API. History is browser localStorage — no DB, no auth.

**Tech Stack:** Python 3.12, FastAPI, scikit-learn, sklearn-crfsuite, vaderSentiment, httpx, anthropic, pytest, respx; React 18, TypeScript, Vite, Tailwind CSS, Vitest, Testing Library.

## Global Constraints

- Python **3.12**; Node **20+**.
- Backend deps must stay light — **no torch, no transformers** installed (DistilBERT is a remote HTTP call). Keep `api/requirements.txt` minimal.
- Single Vercel project: SPA static + `api/` Python serverless. Function bundle must stay under 250 MB.
- Uniform result shape across all models: `{label, confidence, scores}`; CRF additionally sets `aspects`.
- Secrets are env vars only: `ANTHROPIC_API_KEY`, `HF_API_TOKEN`. Never commit keys. Missing key → model reports `available: false`, never crashes a request.
- One model = one file in `api/models/`, registered in `REGISTRY` (single source of truth for API + dropdown). Adding a model must not modify existing model files (Open/Closed).
- Model id strings are canonical and identical backend↔frontend: `vader`, `logreg`, `crf`, `distilbert`, `llm`.
- TDD: write the failing test first, watch it fail, minimal implementation, watch it pass, commit. Mock all remote calls (HF, Claude) in tests.

---

## File Structure

```
api/
  index.py                 # FastAPI app + routes (Vercel entrypoint)
  schemas.py               # AnalysisResult, Explanation, request/response models
  models/
    base.py                # SentimentModel protocol, AnalysisResult/Explanation dataclasses
    registry.py            # REGISTRY dict, get_model(), list_models()
    vader.py
    logreg.py
    crf.py
    distilbert.py
    llm.py
  runner.py                # concurrent multi-model analyze with per-model error isolation
  requirements.txt
artifacts/                 # committed pretrained pickles
  logreg.pkl
  crf.pkl
scripts/
  train_models.py          # trains + pickles logreg & crf from a small SST-2 subset
tests/                     # pytest (backend)
  test_registry.py
  test_vader.py
  test_logreg.py
  test_crf.py
  test_distilbert.py
  test_llm.py
  test_runner.py
  test_api.py
frontend/
  index.html
  package.json
  vite.config.ts
  tailwind.config.js
  postcss.config.js
  tsconfig.json
  src/
    main.tsx
    App.tsx                # tab shell
    api.ts                 # typed fetch client
    types.ts               # shared TS types mirroring backend schemas
    hooks/useHistory.ts    # localStorage history
    components/
      CompareTab.tsx
      ModelCard.tsx
      ExplanationModal.tsx
      GuidesTab.tsx
      HistoryTab.tsx
      MultiSelect.tsx
    content/guides.ts      # guide content per family (from RESEARCH/THEORY)
    __tests__/             # Vitest
CLAUDE.md
README.md
docs/architecture.md
vercel.json
.gitignore                # (exists)
```

---

## Task 1: Repo scaffolding + Vercel config + Claude tooling

**Files:**
- Create: `vercel.json`, `api/requirements.txt`, `CLAUDE.md`, `README.md`, `docs/architecture.md`, `.claude/settings.json`, `.claude/agents/model-adder.md`, `.claude/hooks/format.sh`, `frontend/package.json`
- Test: manual (config task — no unit test; validated by later tasks building on it)

**Interfaces:**
- Produces: project layout, `vercel.json` routing, `api/requirements.txt` dependency set that all backend tasks rely on.

- [ ] **Step 1: Create `api/requirements.txt`** (light deps only)

```
fastapi==0.115.*
vaderSentiment==3.3.2
scikit-learn==1.5.*
sklearn-crfsuite==0.5.0
httpx==0.27.*
anthropic==0.40.*
pydantic==2.*
```

- [ ] **Step 2: Create `vercel.json`**

```json
{
  "buildCommand": "cd frontend && npm install && npm run build",
  "outputDirectory": "frontend/dist",
  "functions": { "api/index.py": { "runtime": "python3.12" } },
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

- [ ] **Step 3: Create `.claude/settings.json`**

```json
{
  "permissions": {
    "allow": ["Bash(pytest*)", "Bash(npm run*)", "Bash(python scripts/*)"],
    "deny": ["Read(./.env)", "Read(./.env.local)"]
  },
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "bash .claude/hooks/format.sh" }] }
    ]
  }
}
```

- [ ] **Step 4: Create `.claude/hooks/format.sh`** (best-effort formatter; never fails the edit)

```bash
#!/usr/bin/env bash
# ponytail: best-effort format, never block an edit
ruff format api scripts 2>/dev/null || true
cd frontend 2>/dev/null && npx --no-install prettier -w 'src/**/*.{ts,tsx}' 2>/dev/null || true
exit 0
```

- [ ] **Step 5: Create `.claude/agents/model-adder.md`** (agent that scaffolds a new model the SOLID way)

```markdown
---
name: model-adder
description: Scaffold a new sentiment model following the SOLID registry pattern. Use when adding a model family to the comparison.
tools: Read, Write, Edit, Bash
---
Add a new `SentimentModel` without modifying existing model files.
1. Read `api/models/base.py` for the protocol and `api/models/vader.py` as the reference implementation.
2. Create `api/models/<id>.py` implementing `analyze()` and `explain()`, returning `AnalysisResult`/`Explanation`.
3. Register it in `api/models/registry.py` REGISTRY only — touch no other model file.
4. Create `tests/test_<id>.py` mirroring `tests/test_vader.py`; mock any remote calls.
5. Run `pytest tests/test_<id>.py -v` and confirm green.
```

- [ ] **Step 6: Create `CLAUDE.md`, `README.md`, `docs/architecture.md`, `frontend/package.json`**

`CLAUDE.md`:
```markdown
# Sentiment Analysis Comparison App

Compare 5 sentiment models side-by-side. React (frontend/) + FastAPI serverless (api/), one Vercel deploy.

## Run locally
- Backend: `pip install -r api/requirements.txt && uvicorn api.index:app --reload`
- Frontend: `cd frontend && npm install && npm run dev`
- Train pickles: `python scripts/train_models.py`

## Conventions
- One model = one file in `api/models/`, registered in `registry.py`. Never modify existing models to add a new one (Open/Closed).
- Uniform result shape `{label, confidence, scores}`; CRF adds `aspects`.
- Secrets: `ANTHROPIC_API_KEY`, `HF_API_TOKEN` env vars. Missing key → `available: false`, never crash.
- Tests mock all remote (HF, Claude) calls.

## Test
- Backend: `pytest`
- Frontend: `cd frontend && npm test`
```

`frontend/package.json`:
```json
{
  "name": "sentiment-frontend",
  "private": true,
  "type": "module",
  "scripts": { "dev": "vite", "build": "tsc && vite build", "test": "vitest run" },
  "dependencies": { "react": "^18.3.1", "react-dom": "^18.3.1" },
  "devDependencies": {
    "@testing-library/react": "^16.0.0", "@testing-library/jest-dom": "^6.4.0",
    "@types/react": "^18.3.0", "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0", "autoprefixer": "^10.4.0",
    "jsdom": "^25.0.0", "postcss": "^8.4.0", "tailwindcss": "^3.4.0",
    "typescript": "^5.5.0", "vite": "^5.4.0", "vitest": "^2.1.0"
  }
}
```

`README.md` and `docs/architecture.md`: copy the architecture diagram and run/deploy instructions from `docs/superpowers/specs/2026-06-28-sentiment-comparison-app-design.md` §3, §7, plus a Vercel deploy section (set `ANTHROPIC_API_KEY` and `HF_API_TOKEN` in Project Settings → Environment Variables; push to git; import to Vercel).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold monorepo, vercel config, claude tooling"
```

---

## Task 2: Backend core — schemas, protocol, registry

**Files:**
- Create: `api/models/base.py`, `api/models/registry.py`, `api/schemas.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `AnalysisResult(model_id: str, label: str, confidence: float, scores: dict[str,float], aspects: list[Aspect] | None = None, available: bool = True, error: str | None = None)`
  - `Explanation(model_id: str, explanation_type: str, summary: str, evidence: list[dict])` — `explanation_type ∈ {"native","ai-narrated"}`
  - `SentimentModel` protocol: attrs `id, name, family, description`; methods `analyze(text)->AnalysisResult`, `explain(text, result)->Explanation`, `available()->bool`
  - `REGISTRY: dict[str, SentimentModel]`, `list_models()->list[dict]`, `get_model(id)->SentimentModel`

- [ ] **Step 1: Write the failing test** — `tests/test_registry.py`

```python
from api.models.registry import REGISTRY, list_models, get_model

def test_registry_has_five_models():
    assert set(REGISTRY) == {"vader", "logreg", "crf", "distilbert", "llm"}

def test_list_models_shape():
    items = list_models()
    assert all({"id", "name", "family", "description", "available"} <= set(m) for m in items)

def test_get_model_returns_matching_id():
    assert get_model("vader").id == "vader"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL (module `api.models.registry` not found)

- [ ] **Step 3: Implement `api/models/base.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass
class Aspect:
    term: str
    polarity: str

@dataclass
class AnalysisResult:
    model_id: str
    label: str                      # "positive" | "negative" | "neutral"
    confidence: float               # 0..1
    scores: dict[str, float] = field(default_factory=dict)
    aspects: list[Aspect] | None = None
    available: bool = True
    error: str | None = None

@dataclass
class Explanation:
    model_id: str
    explanation_type: str           # "native" | "ai-narrated"
    summary: str
    evidence: list[dict] = field(default_factory=list)

@runtime_checkable
class SentimentModel(Protocol):
    id: str
    name: str
    family: str
    description: str
    def available(self) -> bool: ...
    def analyze(self, text: str) -> AnalysisResult: ...
    def explain(self, text: str, result: AnalysisResult) -> Explanation: ...
```

- [ ] **Step 4: Implement `api/schemas.py`** (pydantic request/response)

```python
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_ids: list[str] = Field(min_length=1)

class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_id: str
```

- [ ] **Step 5: Implement `api/models/registry.py`**

```python
from api.models.base import SentimentModel
from api.models.vader import VaderModel
from api.models.logreg import LogRegModel
from api.models.crf import CrfModel
from api.models.distilbert import DistilBertModel
from api.models.llm import LlmModel

REGISTRY: dict[str, SentimentModel] = {
    m.id: m for m in (VaderModel(), LogRegModel(), CrfModel(), DistilBertModel(), LlmModel())
}

def get_model(model_id: str) -> SentimentModel:
    if model_id not in REGISTRY:
        raise KeyError(model_id)
    return REGISTRY[model_id]

def list_models() -> list[dict]:
    return [
        {"id": m.id, "name": m.name, "family": m.family,
         "description": m.description, "available": m.available()}
        for m in REGISTRY.values()
    ]
```

Note: registry imports all 5 model classes — Tasks 3–7 create them. Until then this import fails; that's expected. Implement Task 3–7 model files before re-running this test, OR temporarily stub. To keep tasks independently testable, create minimal stub classes now in each model file (Tasks 3–7 replace bodies). Stub form for each (e.g. `api/models/vader.py`):

```python
from api.models.base import AnalysisResult, Explanation
class VaderModel:
    id="vader"; name="VADER"; family="Lexicon/rules"; description="Lexicon + rules baseline"
    def available(self): return True
    def analyze(self, text): return AnalysisResult(self.id, "neutral", 0.0)
    def explain(self, text, result): return Explanation(self.id, "native", "", [])
```
Create the analogous stub in `logreg.py` (id=logreg, name="Logistic Regression", family="Classical ML"), `crf.py` (id=crf, name="CRF", family="Sequence"), `distilbert.py` (id=distilbert, name="DistilBERT", family="Deep learning"), `llm.py` (id=llm, name="Claude LLM", family="LLM"). Replace bodies in their own tasks.

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add api/models/ api/schemas.py tests/test_registry.py
git commit -m "feat: backend core schemas, SentimentModel protocol, registry"
```

---

## Task 3: VADER model (lexicon, native explanation)

**Files:**
- Modify: `api/models/vader.py`
- Test: `tests/test_vader.py`

**Interfaces:**
- Consumes: `AnalysisResult`, `Explanation` from base.
- Produces: `VaderModel` with real `analyze`/`explain`.

- [ ] **Step 1: Write the failing test** — `tests/test_vader.py`

```python
from api.models.vader import VaderModel
m = VaderModel()

def test_positive_text():
    r = m.analyze("I absolutely love this, it is great!")
    assert r.label == "positive" and r.confidence > 0.5

def test_negative_text():
    r = m.analyze("This is terrible and I hate it.")
    assert r.label == "negative"

def test_explain_lists_word_contributions():
    r = m.analyze("great movie")
    e = m.explain("great movie", r)
    assert e.explanation_type == "native"
    assert any(ev.get("word") == "great" for ev in e.evidence)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_vader.py -v`
Expected: FAIL (stub returns neutral / empty evidence)

- [ ] **Step 3: Implement `api/models/vader.py`**

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api.models.base import AnalysisResult, Explanation

_analyzer = SentimentIntensityAnalyzer()

class VaderModel:
    id = "vader"; name = "VADER"; family = "Lexicon/rules"
    description = "Rule-based lexicon scorer (negation, intensifiers, punctuation)."

    def available(self) -> bool:
        return True

    def analyze(self, text: str) -> AnalysisResult:
        s = _analyzer.polarity_scores(text)
        compound = s["compound"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        return AnalysisResult(self.id, label, abs(compound), scores=s)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        lex = _analyzer.lexicon
        evidence = [
            {"word": w, "valence": lex[w.lower()]}
            for w in text.split() if w.lower() in lex
        ]
        return Explanation(
            self.id, "native",
            f"Compound score {result.scores.get('compound', 0):.2f} from summed word valences "
            f"(pos={result.scores.get('pos',0):.2f}, neg={result.scores.get('neg',0):.2f}).",
            evidence,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_vader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add api/models/vader.py tests/test_vader.py
git commit -m "feat: VADER model with native word-contribution explanation"
```

---

## Task 4: Train + LogReg model (classical ML, native explanation)

**Files:**
- Create: `scripts/train_models.py` (logreg portion), `artifacts/logreg.pkl`
- Modify: `api/models/logreg.py`
- Test: `tests/test_logreg.py`

**Interfaces:**
- Consumes: base dataclasses.
- Produces: `LogRegModel`; pickle at `artifacts/logreg.pkl` = a fitted sklearn `Pipeline(TfidfVectorizer, LogisticRegression)`.

- [ ] **Step 1: Write `scripts/train_models.py` (logreg part)**

```python
"""Train tiny sentiment models on a small labeled set; pickle to artifacts/.
ponytail: tiny built-in dataset keeps the repo self-contained and CPU-fast.
Swap CORPUS for an SST-2 subset for better accuracy."""
import pickle, pathlib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ART = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
ART.mkdir(exist_ok=True)

CORPUS = [
    ("I love this, it is wonderful and great", "positive"),
    ("Absolutely fantastic, highly recommend", "positive"),
    ("Best experience ever, so happy", "positive"),
    ("This is terrible and I hate it", "negative"),
    ("Awful, disappointing and bad", "negative"),
    ("Worst product, completely useless", "negative"),
    ("It is okay, nothing special", "neutral"),
    ("Average, neither good nor bad", "neutral"),
]  # ponytail: expand with a real dataset when accuracy matters

def train_logreg():
    X, y = zip(*CORPUS)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    with open(ART / "logreg.pkl", "wb") as f:
        pickle.dump(pipe, f)

if __name__ == "__main__":
    train_logreg()
    print("trained logreg ->", ART / "logreg.pkl")
```

- [ ] **Step 2: Generate the pickle**

Run: `python scripts/train_models.py`
Expected: prints `trained logreg -> .../artifacts/logreg.pkl`; file exists.

- [ ] **Step 3: Write the failing test** — `tests/test_logreg.py`

```python
from api.models.logreg import LogRegModel
m = LogRegModel()

def test_available_when_pickle_present():
    assert m.available() is True

def test_positive_text():
    r = m.analyze("I love this wonderful great experience")
    assert r.label == "positive"

def test_explain_lists_top_tokens():
    r = m.analyze("terrible awful bad")
    e = m.explain("terrible awful bad", r)
    assert e.explanation_type == "native"
    assert len(e.evidence) >= 1 and "token" in e.evidence[0]
```

- [ ] **Step 4: Run to verify fail**

Run: `pytest tests/test_logreg.py -v`
Expected: FAIL (stub)

- [ ] **Step 5: Implement `api/models/logreg.py`**

```python
import pickle, pathlib
import numpy as np
from api.models.base import AnalysisResult, Explanation

_PKL = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "logreg.pkl"

class LogRegModel:
    id = "logreg"; name = "Logistic Regression"; family = "Classical ML"
    description = "TF-IDF n-grams + logistic regression (discriminative baseline)."

    def __init__(self):
        self._pipe = None
        if _PKL.exists():
            with open(_PKL, "rb") as f:
                self._pipe = pickle.load(f)

    def available(self) -> bool:
        return self._pipe is not None

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="model artifact missing")
        proba = self._pipe.predict_proba([text])[0]
        classes = self._pipe.classes_
        idx = int(np.argmax(proba))
        return AnalysisResult(self.id, classes[idx], float(proba[idx]),
                              scores={c: float(p) for c, p in zip(classes, proba)})

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        if not self.available():
            return Explanation(self.id, "native", "Model unavailable.", [])
        vec = self._pipe.named_steps["tfidf"]
        clf = self._pipe.named_steps["clf"]
        feats = vec.transform([text])
        names = vec.get_feature_names_out()
        cls_idx = list(clf.classes_).index(result.label)
        # binary LR has shape (1,n); multiclass (k,n)
        coefs = clf.coef_[cls_idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        contribs = []
        for j in feats.nonzero()[1]:
            contribs.append({"token": names[j], "weight": float(coefs[j] * feats[0, j])})
        contribs.sort(key=lambda c: abs(c["weight"]), reverse=True)
        return Explanation(self.id, "native",
                           f"Predicted '{result.label}' ({result.confidence:.0%}). "
                           f"Top tokens by coefficient × TF-IDF:", contribs[:8])
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_logreg.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add scripts/train_models.py artifacts/logreg.pkl api/models/logreg.py tests/test_logreg.py
git commit -m "feat: LogReg model + training script with native token-weight explanation"
```

---

## Task 5: Train + CRF model (sequence, aspect extraction)

**Files:**
- Modify: `scripts/train_models.py` (add CRF), `api/models/crf.py`
- Create: `artifacts/crf.pkl`
- Test: `tests/test_crf.py`

**Interfaces:**
- Consumes: `AnalysisResult`, `Aspect`, `Explanation`.
- Produces: `CrfModel`. Result carries `aspects: list[Aspect]`; `label` = overall polarity (majority of aspect polarities, else neutral).

- [ ] **Step 1: Add CRF training to `scripts/train_models.py`**

```python
import sklearn_crfsuite

# BIO-tagged aspect data: tokens with (word, tag). ASP = aspect term.
# ponytail: handful of sentences; expand with SemEval ABSA for real use.
SENTS = [
    [("the","O"),("battery","ASP"),("is","O"),("great","O")],
    [("the","O"),("screen","ASP"),("is","O"),("terrible","O")],
    [("amazing","O"),("camera","ASP"),("but","O"),("poor","O"),("sound","ASP")],
    [("the","O"),("food","ASP"),("was","O"),("delicious","O")],
    [("slow","O"),("service","ASP"),("and","O"),("cold","O"),("coffee","ASP")],
]

def _feats(tokens, i):
    w = tokens[i][0]
    f = {"w.lower": w.lower(), "is_noun_like": w.isalpha(), "bias": 1.0}
    if i > 0: f["prev"] = tokens[i-1][0].lower()
    if i < len(tokens)-1: f["next"] = tokens[i+1][0].lower()
    return f

def train_crf():
    X = [[_feats(s, i) for i in range(len(s))] for s in SENTS]
    y = [[tag for _, tag in s] for s in SENTS]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=50,
                               c1=0.1, c2=0.1, all_possible_transitions=True)
    crf.fit(X, y)
    with open(ART / "crf.pkl", "wb") as f:
        pickle.dump(crf, f)
```

Add `train_crf()` call + print to `__main__`. The `_feats` function is reused by the model — import it from a shared module. To avoid duplication, move `_feats` into `api/models/crf_features.py` and import it in both `train_models.py` and `crf.py`:

```python
# api/models/crf_features.py
def token_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    f = {"w.lower": w.lower(), "is_noun_like": w.isalpha(), "bias": 1.0}
    if i > 0: f["prev"] = tokens[i-1].lower()
    if i < len(tokens)-1: f["next"] = tokens[i+1].lower()
    return f
```
Update `train_models.py` to use `token_features([w for w,_ in s], i)`.

- [ ] **Step 2: Generate pickle**

Run: `python scripts/train_models.py`
Expected: prints crf path; `artifacts/crf.pkl` exists.

- [ ] **Step 3: Write the failing test** — `tests/test_crf.py`

```python
from api.models.crf import CrfModel
m = CrfModel()

def test_available():
    assert m.available() is True

def test_extracts_aspect():
    r = m.analyze("the battery is great")
    assert r.aspects is not None
    assert any(a.term == "battery" for a in r.aspects)

def test_explain_native():
    r = m.analyze("the battery is great")
    e = m.explain("the battery is great", r)
    assert e.explanation_type == "native"
```

- [ ] **Step 4: Run to verify fail**

Run: `pytest tests/test_crf.py -v`
Expected: FAIL (stub)

- [ ] **Step 5: Implement `api/models/crf.py`**

```python
import pickle, pathlib
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api.models.base import AnalysisResult, Explanation, Aspect
from api.models.crf_features import token_features

_PKL = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "crf.pkl"
_vader = SentimentIntensityAnalyzer()

class CrfModel:
    id = "crf"; name = "CRF (aspect extraction)"; family = "Sequence"
    description = "Conditional Random Field tagging aspect terms (BIO); per-aspect polarity."

    def __init__(self):
        self._crf = None
        if _PKL.exists():
            with open(_PKL, "rb") as f:
                self._crf = pickle.load(f)

    def available(self) -> bool:
        return self._crf is not None

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="model artifact missing")
        tokens = text.split()
        feats = [token_features(tokens, i) for i in range(len(tokens))]
        tags = self._crf.predict_single(feats)
        aspects = []
        for tok, tag in zip(tokens, tags):
            if tag == "ASP":
                # polarity of the local window via VADER as a cheap proxy
                pol = _vader.polarity_scores(text)["compound"]
                aspects.append(Aspect(tok, "positive" if pol >= 0.05 else
                                           "negative" if pol <= -0.05 else "neutral"))
        labels = [a.polarity for a in aspects]
        overall = Counter(labels).most_common(1)[0][0] if labels else "neutral"
        return AnalysisResult(self.id, overall, 1.0 if aspects else 0.0,
                              aspects=aspects)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        ev = [{"aspect": a.term, "polarity": a.polarity} for a in (result.aspects or [])]
        return Explanation(self.id, "native",
                           "CRF tagged these aspect terms (BIO sequence labeling); "
                           "polarity assigned per aspect.", ev)
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_crf.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add scripts/train_models.py api/models/crf.py api/models/crf_features.py artifacts/crf.pkl tests/test_crf.py
git commit -m "feat: CRF aspect-extraction model + shared feature fn"
```

---

## Task 6: DistilBERT model (HF Inference API, Claude-narrated explanation)

**Files:**
- Modify: `api/models/distilbert.py`
- Test: `tests/test_distilbert.py`

**Interfaces:**
- Consumes: base dataclasses; env `HF_API_TOKEN`, `ANTHROPIC_API_KEY`.
- Produces: `DistilBertModel`. `available()` ⇔ `HF_API_TOKEN` set. Uses model `distilbert-base-uncased-finetuned-sst-2-english`.

- [ ] **Step 1: Write the failing test** — `tests/test_distilbert.py` (mock HTTP with respx)

```python
import respx, httpx, os
from api.models.distilbert import DistilBertModel, HF_URL

def test_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    assert DistilBertModel().available() is False

@respx.mock
def test_analyze_parses_hf_response(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "x")
    respx.post(HF_URL).mock(return_value=httpx.Response(
        200, json=[[{"label": "POSITIVE", "score": 0.99},
                    {"label": "NEGATIVE", "score": 0.01}]]))
    r = DistilBertModel().analyze("great")
    assert r.label == "positive" and r.confidence > 0.9
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_distilbert.py -v`
Expected: FAIL (stub / no `HF_URL`)

- [ ] **Step 3: Implement `api/models/distilbert.py`**

```python
import os, httpx
from api.models.base import AnalysisResult, Explanation

HF_URL = ("https://api-inference.huggingface.co/models/"
          "distilbert-base-uncased-finetuned-sst-2-english")

class DistilBertModel:
    id = "distilbert"; name = "DistilBERT"; family = "Deep learning"
    description = "Transformer fine-tuned on SST-2, served via Hugging Face Inference API."

    def available(self) -> bool:
        return bool(os.getenv("HF_API_TOKEN"))

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="HF_API_TOKEN not set")
        try:
            resp = httpx.post(HF_URL, headers={"Authorization": f"Bearer {os.environ['HF_API_TOKEN']}"},
                              json={"inputs": text}, timeout=30)
            resp.raise_for_status()
            preds = resp.json()[0]
        except Exception as e:
            return AnalysisResult(self.id, "neutral", 0.0, available=True, error=str(e))
        scores = {p["label"].lower(): p["score"] for p in preds}
        top = max(preds, key=lambda p: p["score"])
        return AnalysisResult(self.id, top["label"].lower(), top["score"], scores=scores)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        from api.models.llm import narrate
        summary = narrate(
            f"A DistilBERT SST-2 classifier predicted '{result.label}' "
            f"(confidence {result.confidence:.0%}) for: {text!r}. In 2-3 sentences, "
            f"explain in plain language what likely drove this, noting it is a transformer "
            f"using contextual embeddings.")
        return Explanation(self.id, "ai-narrated", summary, [])
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_distilbert.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add api/models/distilbert.py tests/test_distilbert.py
git commit -m "feat: DistilBERT via HF Inference API, Claude-narrated explanation"
```

---

## Task 7: LLM model (Claude API) + shared narrate helper

**Files:**
- Modify: `api/models/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: base dataclasses; env `ANTHROPIC_API_KEY`.
- Produces: `LlmModel`; module-level `narrate(prompt: str) -> str` (used by DistilBERT too). Uses model id `claude-opus-4-8`. `available()` ⇔ `ANTHROPIC_API_KEY` set.

- [ ] **Step 1: Write the failing test** — `tests/test_llm.py`

```python
from unittest.mock import patch, MagicMock
from api.models.llm import LlmModel

def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert LlmModel().available() is False

def test_analyze_parses_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake = MagicMock()
    fake.content = [MagicMock(text='{"label":"positive","confidence":0.92,"reason":"clear praise"}')]
    with patch("api.models.llm._client") as c:
        c.messages.create.return_value = fake
        r = LlmModel().analyze("I love it")
    assert r.label == "positive" and r.confidence == 0.92
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL (stub)

- [ ] **Step 3: Implement `api/models/llm.py`**

```python
import os, json
from functools import lru_cache
from api.models.base import AnalysisResult, Explanation

MODEL = "claude-opus-4-8"

@lru_cache(maxsize=1)
def _get_client():
    from anthropic import Anthropic
    return Anthropic()  # reads ANTHROPIC_API_KEY from env

# patch target for tests
_client = None
def _client_or_init():
    global _client
    if _client is None:
        _client = _get_client()
    return _client

def narrate(prompt: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "Explanation unavailable (ANTHROPIC_API_KEY not set)."
    try:
        msg = _client_or_init().messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception as e:
        return f"Explanation unavailable: {e}"

class LlmModel:
    id = "llm"; name = "Claude LLM"; family = "LLM"
    description = "Zero-shot sentiment via Claude with a natural-language rationale."

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="ANTHROPIC_API_KEY not set")
        prompt = ('Classify the sentiment of the text. Respond ONLY with JSON '
                  '{"label": "positive|negative|neutral", "confidence": 0..1, '
                  f'"reason": "short"}}. Text: {text!r}')
        try:
            msg = _client_or_init().messages.create(
                model=MODEL, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            data = json.loads(msg.content[0].text)
        except Exception as e:
            return AnalysisResult(self.id, "neutral", 0.0, available=True, error=str(e))
        return AnalysisResult(self.id, data["label"], float(data["confidence"]),
                              scores={"reason": 0.0}, error=None)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        summary = narrate(
            f"Explain in 2-3 sentences why the sentiment of {text!r} is "
            f"'{result.label}'. Note any nuance, negation, or sarcasm.")
        return Explanation(self.id, "native", summary, [])
```

Note: test patches `api.models.llm._client`; ensure `_client_or_init()` uses the module global so the patch takes effect. Adjust test/impl to agree (the impl above reads the global `_client`; in the test set `_client` via patch context — update `_client_or_init` to `return _client or _get_client()` and have tests patch the global directly). Keep them consistent.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run full backend suite + registry**

Run: `pytest -v`
Expected: all model + registry tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/models/llm.py tests/test_llm.py
git commit -m "feat: Claude LLM model + shared narrate() helper"
```

---

## Task 8: Concurrent runner + FastAPI endpoints

**Files:**
- Create: `api/runner.py`, `api/index.py`
- Test: `tests/test_runner.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `REGISTRY`, `get_model`, `list_models`, `AnalyzeRequest`, `ExplainRequest`.
- Produces: `analyze_many(text, model_ids) -> list[AnalysisResult]` (per-model error isolation); FastAPI app `app` with `GET /api/models`, `POST /api/analyze`, `POST /api/explain`.

- [ ] **Step 1: Write the failing test** — `tests/test_runner.py`

```python
from api.runner import analyze_many

def test_runs_selected_models_and_isolates_errors():
    results = analyze_many("I love this great thing", ["vader", "logreg"])
    ids = {r.model_id for r in results}
    assert ids == {"vader", "logreg"}

def test_unknown_model_id_yields_error_result():
    results = analyze_many("hi", ["vader", "nope"])
    bad = [r for r in results if r.model_id == "nope"][0]
    assert bad.available is False and bad.error
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL (no `api.runner`)

- [ ] **Step 3: Implement `api/runner.py`**

```python
from concurrent.futures import ThreadPoolExecutor
from api.models.registry import get_model
from api.models.base import AnalysisResult

def _one(text: str, model_id: str) -> AnalysisResult:
    try:
        return get_model(model_id).analyze(text)
    except KeyError:
        return AnalysisResult(model_id, "neutral", 0.0, available=False,
                              error="unknown model")
    except Exception as e:
        return AnalysisResult(model_id, "neutral", 0.0, available=True, error=str(e))

def analyze_many(text: str, model_ids: list[str]) -> list[AnalysisResult]:
    with ThreadPoolExecutor(max_workers=min(8, len(model_ids))) as ex:
        return list(ex.map(lambda mid: _one(text, mid), model_ids))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_runner.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing API test** — `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from api.index import app
client = TestClient(app)

def test_list_models():
    r = client.get("/api/models")
    assert r.status_code == 200 and len(r.json()) == 5

def test_analyze():
    r = client.post("/api/analyze", json={"text": "I love it", "model_ids": ["vader"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["model_id"] == "vader"

def test_analyze_validation():
    r = client.post("/api/analyze", json={"text": "", "model_ids": ["vader"]})
    assert r.status_code == 422

def test_explain():
    r = client.post("/api/explain", json={"text": "great movie", "model_id": "vader"})
    assert r.status_code == 200 and r.json()["explanation_type"] == "native"
```

- [ ] **Step 6: Implement `api/index.py`**

```python
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import AnalyzeRequest, ExplainRequest
from api.runner import analyze_many
from api.models.registry import list_models, get_model

app = FastAPI(title="Sentiment Comparison API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/models")
def models():
    return list_models()

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    results = analyze_many(req.text, req.model_ids)
    return {"results": [asdict(r) for r in results]}

@app.post("/api/explain")
def explain(req: ExplainRequest):
    try:
        m = get_model(req.model_id)
    except KeyError:
        raise HTTPException(404, "unknown model")
    result = m.analyze(req.text)
    return asdict(m.explain(req.text, result))
```

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add api/runner.py api/index.py tests/test_runner.py tests/test_api.py
git commit -m "feat: concurrent runner + FastAPI endpoints (models/analyze/explain)"
```

---

## Task 9: Frontend scaffold + Tailwind + types + API client

**Files:**
- Create: `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/index.css`, `frontend/src/types.ts`, `frontend/src/api.ts`
- Test: `frontend/src/__tests__/api.test.ts`

**Interfaces:**
- Produces: TS types mirroring backend (`ModelInfo`, `AnalysisResult`, `Explanation`); `api.ts` exports `listModels()`, `analyze(text, ids)`, `explain(text, id)`.

- [ ] **Step 1: Create config files**

`frontend/vite.config.ts`:
```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", setupFiles: [] },
});
```
`frontend/tailwind.config.js`:
```js
export default { content: ["./index.html", "./src/**/*.{ts,tsx}"], theme: { extend: {} }, plugins: [] };
```
`frontend/postcss.config.js`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```
`frontend/tsconfig.json`:
```json
{ "compilerOptions": { "target": "ES2020", "module": "ESNext", "moduleResolution": "bundler",
  "jsx": "react-jsx", "strict": true, "skipLibCheck": true, "types": ["vitest/globals"] },
  "include": ["src"] }
```
`frontend/index.html`:
```html
<!doctype html><html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sentiment Lab</title></head>
<body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>
```
`frontend/src/index.css`:
```css
@tailwind base; @tailwind components; @tailwind utilities;
```
`frontend/src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
ReactDOM.createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
```

- [ ] **Step 2: Create `frontend/src/types.ts`**

```ts
export interface ModelInfo { id: string; name: string; family: string; description: string; available: boolean; }
export interface Aspect { term: string; polarity: string; }
export interface AnalysisResult {
  model_id: string; label: string; confidence: number;
  scores: Record<string, number>; aspects: Aspect[] | null;
  available: boolean; error: string | null;
}
export interface Explanation {
  model_id: string; explanation_type: "native" | "ai-narrated";
  summary: string; evidence: Record<string, unknown>[];
}
```

- [ ] **Step 3: Write the failing test** — `frontend/src/__tests__/api.test.ts`

```ts
import { describe, it, expect, vi } from "vitest";
import { analyze } from "../api";

describe("api.analyze", () => {
  it("posts text and ids, returns results", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ results: [{ model_id: "vader", label: "positive" }] }),
    }) as any;
    const r = await analyze("hi", ["vader"]);
    expect(r[0].model_id).toBe("vader");
  });
});
```

- [ ] **Step 4: Run to verify fail**

Run: `cd frontend && npm install && npm test`
Expected: FAIL (no `../api`)

- [ ] **Step 5: Implement `frontend/src/api.ts`**

```ts
import type { ModelInfo, AnalysisResult, Explanation } from "./types";

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

export async function listModels(): Promise<ModelInfo[]> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error("failed to load models");
  return res.json();
}
export async function analyze(text: string, ids: string[]): Promise<AnalysisResult[]> {
  const data = await post<{ results: AnalysisResult[] }>("/api/analyze", { text, model_ids: ids });
  return data.results;
}
export async function explain(text: string, id: string): Promise<Explanation> {
  return post<Explanation>("/api/explain", { text, model_id: id });
}
```

- [ ] **Step 6: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/ && git commit -m "feat: frontend scaffold, tailwind, types, api client"
```

---

## Task 10: App shell + tab navigation + history hook

**Files:**
- Create: `frontend/src/App.tsx`, `frontend/src/hooks/useHistory.ts`
- Test: `frontend/src/__tests__/useHistory.test.ts`

**Interfaces:**
- Produces: `App` (3-tab shell); `useHistory()` → `{ history, addRun, clear }`. A run = `{ id, text, results, ts }`.

- [ ] **Step 1: Write the failing test** — `frontend/src/__tests__/useHistory.test.ts`

```ts
import { renderHook, act } from "@testing-library/react";
import { useHistory } from "../hooks/useHistory";

it("persists a run to localStorage", () => {
  localStorage.clear();
  const { result } = renderHook(() => useHistory());
  act(() => result.current.addRun({ text: "hi", results: [] }));
  expect(result.current.history).toHaveLength(1);
  expect(JSON.parse(localStorage.getItem("sentiment-history")!)).toHaveLength(1);
});
```

- [ ] **Step 2: Run to verify fail**

Run: `cd frontend && npm test`
Expected: FAIL (no `useHistory`)

- [ ] **Step 3: Implement `frontend/src/hooks/useHistory.ts`**

```ts
import { useState, useCallback } from "react";
import type { AnalysisResult } from "../types";

export interface Run { id: string; text: string; results: AnalysisResult[]; ts: number; }
const KEY = "sentiment-history";

function load(): Run[] {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
}

export function useHistory() {
  const [history, setHistory] = useState<Run[]>(load);
  const addRun = useCallback((r: { text: string; results: AnalysisResult[] }) => {
    setHistory((prev) => {
      const next = [{ ...r, id: crypto.randomUUID(), ts: Date.now() }, ...prev].slice(0, 100);
      try { localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* quota: skip */ }
      return next;
    });
  }, []);
  const clear = useCallback(() => { localStorage.removeItem(KEY); setHistory([]); }, []);
  return { history, addRun, clear };
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 5: Implement `frontend/src/App.tsx`** (tab shell; design pass happens in Task 14)

```tsx
import { useState } from "react";
import { useHistory } from "./hooks/useHistory";
import CompareTab from "./components/CompareTab";
import GuidesTab from "./components/GuidesTab";
import HistoryTab from "./components/HistoryTab";

type Tab = "compare" | "guides" | "history";

export default function App() {
  const [tab, setTab] = useState<Tab>("compare");
  const hist = useHistory();
  const tabs: [Tab, string][] = [["compare", "Compare"], ["guides", "Guides"], ["history", "History"]];
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <nav className="mx-auto flex max-w-5xl gap-1 px-4" role="tablist">
          {tabs.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`px-4 py-3 text-sm font-medium ${tab === id ? "border-b-2 border-indigo-600 text-indigo-700" : "text-slate-500"}`}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8">
        {tab === "compare" && <CompareTab onRun={hist.addRun} />}
        {tab === "guides" && <GuidesTab />}
        {tab === "history" && <HistoryTab history={hist.history} clear={hist.clear} />}
      </main>
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src && git commit -m "feat: app shell, tabs, history hook"
```

---

## Task 11: Compare tab — input, multi-select, analyze, cards

**Files:**
- Create: `frontend/src/components/CompareTab.tsx`, `frontend/src/components/MultiSelect.tsx`, `frontend/src/components/ModelCard.tsx`
- Test: `frontend/src/__tests__/CompareTab.test.tsx`

**Interfaces:**
- Consumes: `listModels`, `analyze`, `explain` from api; `ModelInfo`, `AnalysisResult`.
- Produces: `CompareTab({ onRun })`. Renders textarea, `MultiSelect`, Analyze button, card grid. Card click opens `ExplanationModal` (Task 12).

- [ ] **Step 1: Write the failing test** — `CompareTab.test.tsx`

```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import CompareTab from "../components/CompareTab";
import * as api from "../api";

it("analyzes and renders a card", async () => {
  vi.spyOn(api, "listModels").mockResolvedValue([
    { id: "vader", name: "VADER", family: "Lexicon", description: "", available: true }]);
  vi.spyOn(api, "analyze").mockResolvedValue([
    { model_id: "vader", label: "positive", confidence: 0.8, scores: {}, aspects: null, available: true, error: null }]);
  render(<CompareTab onRun={() => {}} />);
  await screen.findByText("VADER");
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "I love it" } });
  fireEvent.click(screen.getByText("VADER"));        // select model
  fireEvent.click(screen.getByRole("button", { name: /analyze/i }));
  await waitFor(() => expect(screen.getByText(/positive/i)).toBeInTheDocument());
});
```

- [ ] **Step 2: Run to verify fail**

Run: `cd frontend && npm test`
Expected: FAIL

- [ ] **Step 3: Implement `MultiSelect.tsx`**

```tsx
import type { ModelInfo } from "../types";

export default function MultiSelect(
  { models, selected, onToggle }:
  { models: ModelInfo[]; selected: Set<string>; onToggle: (id: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {models.map((m) => (
        <button key={m.id} type="button" disabled={!m.available}
          aria-pressed={selected.has(m.id)} onClick={() => onToggle(m.id)}
          title={m.available ? m.description : "Unavailable (API key not set)"}
          className={`rounded-full border px-3 py-1 text-sm ${
            selected.has(m.id) ? "border-indigo-600 bg-indigo-50 text-indigo-700" : "border-slate-300 text-slate-600"
          } ${!m.available ? "cursor-not-allowed opacity-40" : ""}`}>
          {m.name}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Implement `ModelCard.tsx`**

```tsx
import type { AnalysisResult } from "../types";

const COLOR: Record<string, string> = {
  positive: "bg-emerald-50 text-emerald-700 border-emerald-200",
  negative: "bg-rose-50 text-rose-700 border-rose-200",
  neutral: "bg-slate-50 text-slate-600 border-slate-200",
};

export default function ModelCard(
  { name, family, result, onClick }:
  { name: string; family: string; result: AnalysisResult; onClick: () => void }) {
  if (result.error && !result.available)
    return <div className="rounded-xl border p-4 opacity-60"><b>{name}</b><p className="text-sm text-slate-500">{result.error}</p></div>;
  return (
    <button onClick={onClick}
      className="rounded-xl border bg-white p-4 text-left shadow-sm transition hover:shadow-md">
      <div className="flex items-center justify-between">
        <b>{name}</b><span className="text-xs text-slate-400">{family}</span>
      </div>
      <span className={`mt-2 inline-block rounded-full border px-2 py-0.5 text-sm ${COLOR[result.label] || COLOR.neutral}`}>
        {result.label}
      </span>
      <div className="mt-3 h-2 w-full rounded bg-slate-100">
        <div className="h-2 rounded bg-indigo-500" style={{ width: `${Math.round(result.confidence * 100)}%` }} />
      </div>
      <p className="mt-1 text-xs text-slate-400">{Math.round(result.confidence * 100)}% confidence · click for why</p>
      {result.aspects && result.aspects.length > 0 && (
        <ul className="mt-2 text-xs text-slate-600">
          {result.aspects.map((a, i) => <li key={i}>{a.term}: {a.polarity}</li>)}
        </ul>
      )}
    </button>
  );
}
```

- [ ] **Step 5: Implement `CompareTab.tsx`**

```tsx
import { useEffect, useState } from "react";
import { listModels, analyze } from "../api";
import type { ModelInfo, AnalysisResult } from "../types";
import MultiSelect from "./MultiSelect";
import ModelCard from "./ModelCard";
import ExplanationModal from "./ExplanationModal";

export default function CompareTab({ onRun }: { onRun: (r: { text: string; results: AnalysisResult[] }) => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [text, setText] = useState("");
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState<AnalysisResult | null>(null);

  useEffect(() => { listModels().then(setModels).catch(() => setModels([])); }, []);
  const nameOf = (id: string) => models.find((m) => m.id === id)?.name ?? id;
  const familyOf = (id: string) => models.find((m) => m.id === id)?.family ?? "";

  const toggle = (id: string) =>
    setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });

  async function run() {
    if (!text.trim() || selected.size === 0) return;
    setLoading(true);
    try {
      const r = await analyze(text, [...selected]);
      setResults(r); onRun({ text, results: r });
    } finally { setLoading(false); }
  }

  return (
    <div className="space-y-6">
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4}
        placeholder="Enter text to analyze…"
        className="w-full rounded-lg border p-3 focus:border-indigo-500 focus:outline-none" />
      <MultiSelect models={models} selected={selected} onToggle={toggle} />
      <button onClick={run} disabled={loading || !text.trim() || selected.size === 0}
        className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white disabled:opacity-40">
        {loading ? "Analyzing…" : "Analyze"}
      </button>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {results.map((r) => (
          <ModelCard key={r.model_id} name={nameOf(r.model_id)} family={familyOf(r.model_id)}
            result={r} onClick={() => setActive(r)} />
        ))}
      </div>
      {active && <ExplanationModal text={text} result={active} name={nameOf(active.model_id)} onClose={() => setActive(null)} />}
    </div>
  );
}
```

- [ ] **Step 6: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src && git commit -m "feat: compare tab, multi-select, result cards"
```

---

## Task 12: Explanation modal (lazy-loaded, accessible)

**Files:**
- Create: `frontend/src/components/ExplanationModal.tsx`
- Test: `frontend/src/__tests__/ExplanationModal.test.tsx`

**Interfaces:**
- Consumes: `explain` from api; `AnalysisResult`, `Explanation`.
- Produces: `ExplanationModal({ text, result, name, onClose })`. Fetches explanation on mount; Esc/backdrop closes; focus-trapped dialog.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import ExplanationModal from "../components/ExplanationModal";
import * as api from "../api";

it("loads and shows explanation summary", async () => {
  vi.spyOn(api, "explain").mockResolvedValue({
    model_id: "vader", explanation_type: "native", summary: "because great is positive", evidence: [] });
  render(<ExplanationModal text="great" name="VADER"
    result={{ model_id: "vader", label: "positive", confidence: 0.8, scores: {}, aspects: null, available: true, error: null }}
    onClose={() => {}} />);
  await waitFor(() => expect(screen.getByText(/because great is positive/)).toBeInTheDocument());
  expect(screen.getByText(/native/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify fail**

Run: `cd frontend && npm test`
Expected: FAIL

- [ ] **Step 3: Implement `ExplanationModal.tsx`**

```tsx
import { useEffect, useState } from "react";
import { explain } from "../api";
import type { AnalysisResult, Explanation } from "../types";

export default function ExplanationModal(
  { text, result, name, onClose }:
  { text: string; result: AnalysisResult; name: string; onClose: () => void }) {
  const [exp, setExp] = useState<Explanation | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    explain(text, result.model_id).then(setExp).catch(() => setErr("Could not load explanation."));
  }, [text, result.model_id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose} role="presentation">
      <div role="dialog" aria-modal="true" aria-label={`Why ${name} predicted ${result.label}`}
        className="max-h-[80vh] w-full max-w-lg overflow-auto rounded-2xl bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <h2 className="text-lg font-semibold">{name} — {result.label}</h2>
          <button onClick={onClose} aria-label="Close" className="text-slate-400 hover:text-slate-700">✕</button>
        </div>
        {!exp && !err && <p className="mt-4 text-slate-500">Loading explanation…</p>}
        {err && <p className="mt-4 text-rose-600">{err}</p>}
        {exp && (
          <>
            <span className="mt-3 inline-block rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {exp.explanation_type === "native" ? "model-native evidence" : "AI-narrated"}
            </span>
            <p className="mt-3 text-slate-700">{exp.summary}</p>
            {exp.evidence.length > 0 && (
              <table className="mt-4 w-full text-sm">
                <tbody>
                  {exp.evidence.map((row, i) => (
                    <tr key={i} className="border-t">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="py-1 pr-4">{String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src && git commit -m "feat: accessible explanation modal, lazy-loaded"
```

---

## Task 13: History tab

**Files:**
- Create: `frontend/src/components/HistoryTab.tsx`
- Test: `frontend/src/__tests__/HistoryTab.test.tsx`

**Interfaces:**
- Consumes: `Run` from `useHistory`.
- Produces: `HistoryTab({ history, clear })`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import HistoryTab from "../components/HistoryTab";

it("renders runs and clears", () => {
  const clear = vi.fn();
  render(<HistoryTab clear={clear} history={[
    { id: "1", text: "I love it", ts: Date.now(),
      results: [{ model_id: "vader", label: "positive", confidence: 0.8, scores: {}, aspects: null, available: true, error: null }] }]} />);
  expect(screen.getByText(/I love it/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /clear/i }));
  expect(clear).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run to verify fail**

Run: `cd frontend && npm test`
Expected: FAIL

- [ ] **Step 3: Implement `HistoryTab.tsx`**

```tsx
import type { Run } from "../hooks/useHistory";

export default function HistoryTab({ history, clear }: { history: Run[]; clear: () => void }) {
  if (history.length === 0) return <p className="text-slate-500">No analyses yet.</p>;
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button onClick={clear} className="text-sm text-rose-600 hover:underline">Clear history</button>
      </div>
      <ul className="space-y-3">
        {history.map((run) => (
          <li key={run.id} className="rounded-lg border bg-white p-4">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{new Date(run.ts).toLocaleString()}</span>
              <span>{run.results.length} model(s)</span>
            </div>
            <p className="mt-1 truncate font-medium">{run.text}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {run.results.map((r) => (
                <span key={r.model_id} className="rounded-full bg-slate-100 px-2 py-0.5 text-xs">
                  {r.model_id}: {r.label}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src && git commit -m "feat: history tab"
```

---

## Task 14: Guides tab + content, then design polish pass

**Files:**
- Create: `frontend/src/components/GuidesTab.tsx`, `frontend/src/content/guides.ts`
- Test: `frontend/src/__tests__/GuidesTab.test.tsx`

**Interfaces:**
- Produces: `GuidesTab` rendering one expandable section per family, content sourced from `RESEARCH.md`/`THEORY.md`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import GuidesTab from "../components/GuidesTab";

it("renders a section per family", () => {
  render(<GuidesTab />);
  expect(screen.getByText(/Lexicon/i)).toBeInTheDocument();
  expect(screen.getByText(/Transformers|DistilBERT|LLM/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify fail**

Run: `cd frontend && npm test`
Expected: FAIL

- [ ] **Step 3: Implement `content/guides.ts`** — one entry per family

```ts
export interface Guide { id: string; title: string; era: string; body: string; }
export const GUIDES: Guide[] = [
  { id: "lexicon", title: "Lexicon / Rule-based (VADER)", era: "≈1997–2010",
    body: "Dictionary of word→polarity scores, summed with rules for negation, intensifiers, punctuation. Worked example: 'not very good' → +1.9 boosted to +2.193, negated ×-0.74 = -1.623, normalized to -0.39. Zero training, fully interpretable." },
  { id: "classical", title: "Classical ML (Logistic Regression)", era: "2002+",
    body: "TF-IDF n-grams → discriminative classifier. LR models P(class|text) directly, avoiding Naïve Bayes's independence assumption. Gradient is (prediction − truth)·features — the base case of every neural net." },
  { id: "sequence", title: "Sequence models (HMM & CRF)", era: "2004–2015",
    body: "Label tokens, not whole docs — used for aspect extraction. HMM is generative (Viterbi decoding). CRF is discriminative and globally normalized, so it admits arbitrary overlapping features and avoids label bias — why it beat HMMs." },
  { id: "deep", title: "Deep learning (CNN/LSTM → DistilBERT)", era: "2013+",
    body: "Learned embeddings replace hand features. LSTMs store a 'negation pending' signal in cell state. Transformers use self-attention: in 'not good', the word 'good' absorbs most of 'not' via softmax-weighted attention. DistilBERT = compressed BERT, ~97% accuracy at 40% size." },
  { id: "llm", title: "LLMs (Claude)", era: "2022+",
    body: "Zero-shot via prompting: no training data, handles sarcasm and aspect-based sentiment, returns structured output with a natural-language rationale. Best on nuanced text; costs an API call." },
];
```

- [ ] **Step 4: Implement `GuidesTab.tsx`** (native `<details>` — no accordion lib)

```tsx
import { GUIDES } from "../content/guides";

export default function GuidesTab() {
  return (
    <div className="space-y-3">
      <p className="text-slate-600">How each model family decides sentiment — the 20-year arc.</p>
      {GUIDES.map((g) => (
        <details key={g.id} className="rounded-lg border bg-white p-4 open:shadow-sm">
          <summary className="cursor-pointer font-medium">
            {g.title} <span className="ml-2 text-xs text-slate-400">{g.era}</span>
          </summary>
          <p className="mt-3 text-slate-700">{g.body}</p>
        </details>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Run to verify pass**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 6: Visual design polish (REQUIRED SUB-SKILL: frontend-design)**

Invoke the `frontend-design` skill and apply a distinctive, professional visual direction across `App.tsx`, cards, modal, and tabs: cohesive color system, typography scale, spacing rhythm, hover/active states, empty states, and responsive layout. Keep all existing component contracts and test IDs intact so tests stay green.

- [ ] **Step 7: Run full frontend suite**

Run: `cd frontend && npm test`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend/src && git commit -m "feat: guides tab + content; visual design polish"
```

---

## Task 15: Local end-to-end verification + deploy docs

**Files:**
- Modify: `README.md`, `docs/architecture.md`
- Test: manual E2E

**Interfaces:** none (integration + docs).

- [ ] **Step 1: Run the whole backend suite**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 2: Build the frontend**

Run: `cd frontend && npm run build`
Expected: `frontend/dist/` produced, no TS errors.

- [ ] **Step 3: Manual E2E**

Start backend `uvicorn api.index:app --reload` and frontend `npm run dev`. In the browser: enter text, select VADER + LogReg, Analyze → two cards appear; click a card → modal with native explanation; check History tab shows the run; reload → history persists. (DistilBERT/LLM show "unavailable" without keys — expected.)

- [ ] **Step 4: Finalize deploy docs in `README.md`**

Document: push repo to GitHub; import to Vercel; set `ANTHROPIC_API_KEY` and `HF_API_TOKEN` in Project Settings → Environment Variables; Vercel auto-detects `vercel.json`; the build runs `frontend` build + deploys `api/` as Python serverless. Note the graceful-degradation behavior when keys are absent.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/architecture.md && git commit -m "docs: deploy guide + e2e verification notes"
```

---

## Self-Review

**Spec coverage:**
- React frontend + Python backend → Tasks 2–8 (backend), 9–14 (frontend). ✓
- All families supported (hybrid hosting) → VADER (T3), LogReg (T4), CRF (T5), DistilBERT/HF (T6), LLM/Claude (T7). ✓
- Scaffolding incl. CLAUDE.md, settings, agents, hooks, README, architecture docs → Task 1. ✓
- 3 tabs (Compare/Guides/History) → T11, T14, T13; shell T10. ✓
- Multi-select dropdown + analyze + cards → T11. ✓
- Click card → explanation modal → T12. ✓
- History per browser (localStorage) → T10 hook + T13 tab. ✓
- SOLID (protocol + registry, Open/Closed) → T2; enforced by model-adder agent T1. ✓
- Uniform result shape, CRF aspects → T2 schema, used throughout. ✓
- Vercel deploy, env vars, graceful degradation → T1 config, T6/T7 availability, T15 docs. ✓
- Native vs AI-narrated explanations → T3/T4/T5 native, T6 ai-narrated, T7 native; labeled in modal T12. ✓
- Testing (pytest + Vitest) → every task is TDD. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; remote calls mocked in tests. Two `ponytail:` comments mark deliberate tiny-dataset simplifications with an upgrade path (real datasets) — intentional, not placeholders.

**Type consistency:** `AnalysisResult`/`Explanation`/`Aspect` fields identical across base.py (T2), all models (T3–7), runner (T8), and TS `types.ts` (T9). `narrate()` defined in T7, consumed in T6. `useHistory` shape (T10) matches `HistoryTab` props (T13) and `CompareTab onRun` (T11). Model ids canonical everywhere.

**Known caveat:** T7 has a noted client-patching consistency point between `_client_or_init()` and the test's patch target — the implementer must make the global-vs-helper agree (called out inline in T7 Step 3).
