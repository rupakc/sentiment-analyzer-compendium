from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
        evidence = [
            {"word": w, "valence": lex[w.lower()]}
            for w in text.split()
            if w.lower() in lex
        ]
        return Explanation(
            self.id,
            "native",
            f"Compound score {result.scores.get('compound', 0):.2f} from summed word valences "
            f"(pos={result.scores.get('pos', 0):.2f}, neg={result.scores.get('neg', 0):.2f}).",
            evidence,
        )
