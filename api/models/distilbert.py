import os
import httpx
from api.models.base import AnalysisResult, Explanation

HF_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)


class DistilBertModel:
    id = "distilbert"
    name = "DistilBERT"
    family = "Deep learning"
    description = "Transformer fine-tuned on SST-2, served via Hugging Face Inference API."

    def available(self) -> bool:
        return bool(os.getenv("HF_API_TOKEN"))

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="HF_API_TOKEN not set")
        try:
            resp = httpx.post(
                HF_URL,
                headers={"Authorization": f"Bearer {os.environ['HF_API_TOKEN']}"},
                json={"inputs": text},
                timeout=30,
            )
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
            f"using contextual embeddings."
        )
        return Explanation(self.id, "ai-narrated", summary, [])
