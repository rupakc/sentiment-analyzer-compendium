from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer,
    NEGATE,
    BOOSTER_DICT,
)
from api.models.base import AnalysisResult, Explanation

_analyzer = SentimentIntensityAnalyzer()


class VaderModel:
    id = "vader"
    name = "VADER"
    family = "Lexicon/rules"
    description = "Rule-based lexicon scorer (negation, intensifiers, punctuation)."

    def available(self) -> bool:
        return True

    def analyze(self, text: str) -> AnalysisResult:
        s = _analyzer.polarity_scores(text)
        compound = s["compound"]
        label = (
            "positive"
            if compound >= 0.05
            else "negative"
            if compound <= -0.05
            else "neutral"
        )
        return AnalysisResult(self.id, label, abs(compound), scores=s)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        lex = _analyzer.lexicon
        words = text.split()
        evidence = []
        boosted, negated, capsed = [], [], []
        has_caps = any(w.isupper() and len(w) > 1 for w in words)
        for i, w in enumerate(words):
            wl = w.lower()
            prev = words[i - 1].lower() if i > 0 else ""
            if prev in NEGATE:
                negated.append(w)
            if prev in BOOSTER_DICT:
                boosted.append(f"{words[i - 1]} {w}")
            if w.isupper() and len(w) > 1 and wl in lex:
                capsed.append(w)
            if wl in lex:
                val = lex[wl]
                detail = (
                    "negator"
                    if wl in NEGATE
                    else (
                        "intensifier"
                        if wl in BOOSTER_DICT
                        else ("positive valence" if val > 0 else "negative valence")
                    )
                )
                evidence.append({"label": w, "detail": detail, "weight": float(val)})
            elif wl in NEGATE:
                evidence.append({"label": w, "detail": "negator", "weight": 0.0})

        compound = result.scores.get("compound", 0.0)
        label = result.label
        steps = [
            f"Looked up each token in the VADER lexicon; "
            f"{len(evidence)} of {len(words)} words carry valence.",
        ]
        adj = []
        if boosted:
            adj.append(f"intensifier boosts on {', '.join(boosted)}")
        if negated:
            adj.append(f"negation flips on {', '.join(negated)}")
        excl = text.count("!")
        if excl:
            adj.append(f"{excl} exclamation mark(s) amplify intensity")
        if capsed:
            adj.append(f"ALL-CAPS emphasis on {', '.join(capsed)}")
        steps.append(
            "Applied rule adjustments: "
            + ("; ".join(adj) if adj else "none triggered")
            + "."
        )
        steps.append(
            f"Summed valences (pos={result.scores.get('pos', 0):.2f}, "
            f"neg={result.scores.get('neg', 0):.2f}, neu={result.scores.get('neu', 0):.2f})."
        )
        steps.append(
            f"Normalized the raw sum via x/sqrt(x^2+15) to compound={compound:.2f}."
        )
        steps.append(
            f"Thresholded compound ({'>=0.05' if compound >= 0.05 else '<=-0.05' if compound <= -0.05 else 'in (-0.05,0.05)'}) "
            f"to label '{label}'."
        )
        return Explanation(
            self.id,
            "native",
            f"Compound score {compound:.2f} from summed word valences "
            f"(pos={result.scores.get('pos', 0):.2f}, neg={result.scores.get('neg', 0):.2f}).",
            evidence,
            method=(
                "Lexicon valence sum with rule-based negation/intensifier/punctuation/caps "
                "adjustments, normalized to [-1,1]."
            ),
            steps=steps,
            biases=[
                "Only scores words present in its fixed lexicon; novel slang, jargon, or "
                "misspellings contribute nothing (coverage gaps).",
                "No real syntactic parsing: negation/intensifier rules use a small fixed "
                "window, so long-range scope (e.g. 'not at all good') is often missed.",
                "Blind to sarcasm and irony ('oh great, another bug' scores positive).",
                "Tuned on social-media text; valences may not transfer to clinical, legal, "
                "or financial domains.",
                "Sensitive to surface cues like punctuation and capitalization, which can be "
                "gamed or absent in formal writing.",
            ],
        )
