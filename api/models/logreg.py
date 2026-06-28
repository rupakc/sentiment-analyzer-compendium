import pickle
import pathlib
import numpy as np
from api.models.base import AnalysisResult, Explanation

_PKL = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "logreg.pkl"


class LogRegModel:
    id = "logreg"
    name = "Logistic Regression"
    family = "Classical ML"
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
        # binary LR has shape (1, n); multiclass (k, n)
        coefs = clf.coef_[cls_idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        contribs = []
        for j in feats.nonzero()[1]:
            contribs.append({"token": names[j], "weight": float(coefs[j] * feats[0, j])})
        contribs.sort(key=lambda c: abs(c["weight"]), reverse=True)
        return Explanation(
            self.id, "native",
            f"Predicted '{result.label}' ({result.confidence:.0%}). "
            f"Top tokens by coefficient x TF-IDF:",
            contribs[:8],
        )
