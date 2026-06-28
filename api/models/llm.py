import os
import json
from api.models.base import AnalysisResult, Explanation

MODEL = "claude-opus-4-8"

# Module-level client; tests patch this directly.
_client = None


def _client_or_init():
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def narrate(prompt: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "Explanation unavailable (ANTHROPIC_API_KEY not set)."
    try:
        msg = _client_or_init().messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"Explanation unavailable: {e}"


class LlmModel:
    id = "llm"
    name = "Claude LLM"
    family = "LLM"
    description = "Zero-shot sentiment via Claude with a natural-language rationale."

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(self.id, "neutral", 0.0, available=False,
                                  error="ANTHROPIC_API_KEY not set")
        prompt = (
            'Classify the sentiment of the text. Respond ONLY with JSON '
            '{"label": "positive|negative|neutral", "confidence": 0..1, '
            f'"reason": "short"}}. Text: {text!r}'
        )
        try:
            msg = _client_or_init().messages.create(
                model=MODEL, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(msg.content[0].text)
        except Exception as e:
            return AnalysisResult(self.id, "neutral", 0.0, available=True, error=str(e))
        return AnalysisResult(self.id, data["label"], float(data["confidence"]),
                              scores={})

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        summary = narrate(
            f"Explain in 2-3 sentences why the sentiment of {text!r} is "
            f"'{result.label}'. Note any nuance, negation, or sarcasm."
        )
        return Explanation(self.id, "native", summary, [])
