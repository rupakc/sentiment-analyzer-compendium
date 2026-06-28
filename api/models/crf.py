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
    description = (
        "Conditional Random Field tagging aspect terms (BIO); per-aspect polarity."
    )

    def __init__(self):
        self._crf = None
        if _PKL.exists():
            with open(_PKL, "rb") as f:
                self._crf = pickle.load(f)

    def available(self) -> bool:
        return self._crf is not None

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(
                self.id, "neutral", 0.0, available=False, error="model artifact missing"
            )
        tokens = text.split()
        if not tokens:
            return AnalysisResult(self.id, "neutral", 0.0, aspects=[])
        feats = [token_features(tokens, i) for i in range(len(tokens))]
        tags = self._crf.predict_single(feats)
        pol = _vader.polarity_scores(text)["compound"]
        polarity = (
            "positive" if pol >= 0.05 else "negative" if pol <= -0.05 else "neutral"
        )
        aspects = [
            Aspect(tok, polarity) for tok, tag in zip(tokens, tags) if tag == "ASP"
        ]
        labels = [a.polarity for a in aspects]
        overall = Counter(labels).most_common(1)[0][0] if labels else "neutral"
        return AnalysisResult(
            self.id, overall, 1.0 if aspects else 0.0, aspects=aspects
        )

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        aspects = result.aspects or []
        ev = [
            {"label": a.term, "detail": "ASP", "weight": _aspect_weight(a.polarity)}
            for a in aspects
        ]
        tokens = text.split()
        compound = _vader.polarity_scores(text)["compound"]
        terms = ", ".join(a.term for a in aspects) or "none"
        steps = [
            f"Whitespace-tokenized the text into {len(tokens)} tokens.",
            "Extracted per-token features (word, prefix/suffix, position, neighbors).",
            f"CRF Viterbi-decoded the BIO/ASP tag sequence (globally normalized); "
            f"tagged aspect terms: {terms}.",
            f"Borrowed sentence-level polarity from VADER compound={compound:.2f} and "
            f"applied it to every aspect, giving overall '{result.label}'.",
        ]
        return Explanation(
            self.id,
            "native",
            "CRF tagged these aspect terms (BIO sequence labeling); polarity assigned per aspect.",
            ev,
            method=(
                "Linear-chain CRF Viterbi-decodes a BIO/ASP tag sequence over token features; "
                "polarity per aspect is borrowed from VADER's sentence compound."
            ),
            steps=steps,
            biases=[
                # ponytail: SENTS in scripts/train_models.py is 6 hand-labeled sentences
                "Trained on a TINY hand-labeled set (~6 sentences), so the tagger recognizes "
                "only aspect terms resembling those examples.",
                "Polarity is NOT learned by the CRF: it is VADER's single sentence-level "
                "compound copied onto every aspect, so all aspects share one sentiment.",
                "Aspect spans are only as good as the tiny label set; multi-word aspects and "
                "unseen terms are routinely missed.",
                "Cannot express conflicting per-aspect sentiment (e.g. 'great camera but poor "
                "sound' gets one polarity for both).",
            ],
        )


def _aspect_weight(polarity: str) -> float:
    return {"positive": 1.0, "negative": -1.0}.get(polarity, 0.0)
