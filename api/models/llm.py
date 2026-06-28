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
            model=MODEL,
            max_tokens=300,
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
            return AnalysisResult(
                self.id,
                "neutral",
                0.0,
                available=False,
                error="ANTHROPIC_API_KEY not set",
            )
        prompt = (
            "Classify the sentiment of the text. Respond ONLY with JSON "
            '{"label": "positive|negative|neutral", "confidence": 0..1, '
            f'"reason": "short"}}. Text: {text!r}'
        )
        try:
            msg = _client_or_init().messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(msg.content[0].text)
        except Exception as e:
            return AnalysisResult(self.id, "neutral", 0.0, available=True, error=str(e))
        return AnalysisResult(
            self.id, data["label"], float(data["confidence"]), scores={}
        )

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        static = (
            f"Claude classified this text as '{result.label}' "
            f"(confidence {result.confidence:.0%}) by reading it holistically and weighing "
            f"its sentiment cues. Set ANTHROPIC_API_KEY for a generated, text-specific rationale."
        )
        narrated = narrate(
            f"You are explaining a zero-shot sentiment call. The text {text!r} was judged "
            f"'{result.label}' (confidence {result.confidence:.0%}). In 2-4 sentences name the "
            f"specific cues that drove this, how any negation or sarcasm was handled, and why "
            f"the confidence is at that level."
        )
        # ponytail: reuse narrate()'s unavailable sentinel rather than re-checking the key.
        summary = static if narrated.startswith("Explanation unavailable") else narrated

        steps = [
            f"Read the full text {text!r} as a single prompt (no training on this task).",
            "Weighed sentiment cues holistically using pretrained world knowledge, including "
            "negation and sarcasm in context.",
            f"Returned a structured judgment: label '{result.label}' with a self-reported "
            f"confidence of {result.confidence:.0%} and a short reason.",
        ]
        return Explanation(
            self.id,
            "ai-narrated",
            summary,
            [],
            method=(
                "Zero-shot prompting of a large language model that reads the text and returns "
                "a label, confidence, and natural-language rationale."
            ),
            steps=steps,
            biases=[
                "Prompt-sensitive: small wording changes in the prompt can change the label "
                "or confidence.",
                "May state a confidence it cannot actually justify (poorly calibrated, "
                "self-reported numbers).",
                "Knowledge is frozen at a training cutoff, so very recent slang or events may "
                "be misread.",
                "Tends toward verbosity and can rationalize after the fact rather than report "
                "true decision factors.",
                "Can reflect social biases present in its training data.",
            ],
        )
