from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import AnalyzeRequest, ExplainRequest
from api.runner import analyze_many
from api.models.registry import list_models, get_model

app = FastAPI(title="Sentiment Comparison API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
