import os
import httpx
from api.models.base import AnalysisResult, Explanation

# HF retired api-inference.huggingface.co; serverless now lives behind the
# Inference Providers router and needs the org-prefixed model id.
HF_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


class DistilBertModel:
    id = "distilbert"
    name = "DistilBERT"
    family = "Deep learning"
    description = (
        "Transformer fine-tuned on SST-2, served via Hugging Face Inference API."
    )

    def available(self) -> bool:
        return bool(os.getenv("HF_API_TOKEN"))

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(
                self.id, "neutral", 0.0, available=False, error="HF_API_TOKEN not set"
            )
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
        return AnalysisResult(
            self.id, top["label"].lower(), top["score"], scores=scores
        )

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        from api.models.llm import narrate

        static = (
            f"DistilBERT (fine-tuned on SST-2) predicted '{result.label}' with "
            f"{result.confidence:.0%} confidence. The transformer reads the whole "
            f"sentence at once via self-attention, so its decision reflects contextual "
            f"phrasing rather than individual word lookups."
        )
        narrated = narrate(
            f"A DistilBERT SST-2 classifier predicted '{result.label}' "
            f"(confidence {result.confidence:.0%}) for: {text!r}. In 2-3 sentences, "
            f"explain in plain language what likely drove this, noting it is a transformer "
            f"using contextual embeddings."
        )
        # ponytail: narrate() returns an "unavailable" sentinel when the key is missing;
        # fall back to the static summary so the card is always informative.
        summary = static if narrated.startswith("Explanation unavailable") else narrated

        evidence = [
            {"label": lbl, "weight": float(sc) if lbl == "positive" else -float(sc)}
            for lbl, sc in result.scores.items()
            if lbl in ("positive", "negative")
        ]
        steps = [
            "Split the text into WordPiece subword tokens.",
            "Ran 6 transformer layers of self-attention to build contextual embeddings.",
            "Pooled the [CLS] token representation.",
            "Passed it through the linear SST-2 head and softmax over (negative, positive).",
            f"Took the higher probability, yielding '{result.label}' at {result.confidence:.0%}.",
        ]
        return Explanation(
            self.id,
            "ai-narrated",
            summary,
            evidence,
            method=(
                "Subword transformer encoder (6 self-attention layers) with a linear SST-2 "
                "classification head and softmax over negative/positive."
            ),
            steps=steps,
            biases=[
                "Fine-tuned on SST-2, a movie-review corpus; phrasing from other domains "
                "(medical, financial, casual chat) is out of distribution.",
                "The binary head has NO neutral class, so genuinely neutral text is forced "
                "into positive or negative.",
                "Sensitive to phrasing and length; small rewordings or very long inputs can "
                "swing the prediction.",
                "Inherits social and demographic biases from its pretraining and fine-tuning "
                "corpora.",
            ],
        )
