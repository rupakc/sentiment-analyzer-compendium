import pickle
import pathlib
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from api.models.base import AnalysisResult, Explanation, Aspect
from api.models.crf_features import token_features

_PKL = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "crf.pkl"
_vader = SentimentIntensityAnalyzer()


class CrfModel:
    id = "crf"
    name = "CRF (aspect extraction)"
    family = "Sequence"
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
        if not tokens:
            return AnalysisResult(self.id, "neutral", 0.0, aspects=[])
        feats = [token_features(tokens, i) for i in range(len(tokens))]
        tags = self._crf.predict_single(feats)
        pol = _vader.polarity_scores(text)["compound"]
        polarity = ("positive" if pol >= 0.05 else "negative" if pol <= -0.05 else "neutral")
        aspects = [Aspect(tok, polarity) for tok, tag in zip(tokens, tags) if tag == "ASP"]
        labels = [a.polarity for a in aspects]
        overall = Counter(labels).most_common(1)[0][0] if labels else "neutral"
        return AnalysisResult(self.id, overall, 1.0 if aspects else 0.0, aspects=aspects)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        ev = [{"aspect": a.term, "polarity": a.polarity} for a in (result.aspects or [])]
        return Explanation(
            self.id, "native",
            "CRF tagged these aspect terms (BIO sequence labeling); polarity assigned per aspect.",
            ev,
        )
