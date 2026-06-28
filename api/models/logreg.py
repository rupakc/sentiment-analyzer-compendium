import json
import math
import pathlib
import re
from api.models.base import AnalysisResult, Explanation

_JSON = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "logreg.json"
)
# Matches scikit-learn TfidfVectorizer's default token_pattern.
_TOKEN_RE = re.compile(r"\b\w\w+\b")


def _ngrams(text: str) -> list[str]:
    """Replicate TfidfVectorizer(ngram_range=(1,2)): lowercase, unigrams + bigrams."""
    toks = _TOKEN_RE.findall(text.lower())
    grams = list(toks)
    grams += [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]
    return grams


class LogRegModel:
    id = "logreg"
    name = "Logistic Regression"
    family = "Classical ML"
    description = "TF-IDF n-grams + logistic regression (discriminative baseline)."

    def __init__(self):
        self._m = None
        if _JSON.exists():
            self._m = json.loads(_JSON.read_text())

    def available(self) -> bool:
        return self._m is not None

    def _vectorize(self, text: str) -> dict[int, float]:
        """TF-IDF sparse vector {feature_index: weight}, L2-normalized (norm='l2')."""
        vocab, idf = self._m["vocab"], self._m["idf"]
        counts: dict[int, int] = {}
        for g in _ngrams(text):
            j = vocab.get(g)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        vec = {j: c * idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm:
            vec = {j: v / norm for j, v in vec.items()}
        return vec

    def _proba(self, vec: dict[int, float]) -> dict[str, float]:
        coef, intercept, classes = (
            self._m["coef"],
            self._m["intercept"],
            self._m["classes"],
        )
        logits = [
            sum(vec[j] * row[j] for j in vec) + b for row, b in zip(coef, intercept)
        ]
        if len(classes) == 2 and len(logits) == 1:  # binary LR uses a sigmoid
            p = 1.0 / (1.0 + math.exp(-logits[0]))
            return {classes[0]: 1 - p, classes[1]: p}
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        s = sum(exps)
        return {c: e / s for c, e in zip(classes, exps)}

    def analyze(self, text: str) -> AnalysisResult:
        if not self.available():
            return AnalysisResult(
                self.id, "neutral", 0.0, available=False, error="model artifact missing"
            )
        proba = self._proba(self._vectorize(text))
        label = max(proba, key=proba.get)
        return AnalysisResult(self.id, label, proba[label], scores=proba)

    def explain(self, text: str, result: AnalysisResult) -> Explanation:
        if not self.available():
            return Explanation(self.id, "native", "Model unavailable.", [])
        vec = self._vectorize(text)
        classes, coef = self._m["classes"], self._m["coef"]
        names = {i: g for g, i in self._m["vocab"].items()}
        cls_idx = classes.index(result.label)
        # binary LR keeps a single coefficient row shared across both classes
        row = coef[cls_idx] if len(coef) > 1 else coef[0]
        contribs = [
            {
                "label": names[j],
                "detail": "bigram" if " " in names[j] else "unigram",
                "weight": row[j] * x,
            }
            for j, x in vec.items()
        ]
        contribs.sort(key=lambda c: abs(c["weight"]), reverse=True)
        contribs = contribs[:8]

        steps = [
            f"Vectorized the text into TF-IDF features over unigrams+bigrams; "
            f"{len(vec)} of the input's tokens matched the training vocabulary.",
            f"Took the dot product of that sparse vector with the '{result.label}' "
            f"class coefficient row.",
            "Repeated for every class and ran softmax over the resulting logits.",
            f"argmax selected '{result.label}' with probability {result.confidence:.0%}.",
        ]
        return Explanation(
            self.id,
            "native",
            f"Predicted '{result.label}' ({result.confidence:.0%}). "
            f"Top tokens by coefficient x TF-IDF:",
            contribs,
            method=(
                "TF-IDF (uni+bigram) bag-of-words features scored by a linear logistic "
                "regression, with softmax over class logits."
            ),
            steps=steps,
            biases=[
                # ponytail: corpus is 10 hand-written sentences in scripts/train_models.py
                "Trained on a TINY toy corpus (~10 hand-written sentences), so coefficients "
                "are noisy and overfit to those exact words.",
                "Bag-of-words: ignores word order entirely, so it cannot model negation scope "
                "('not good' looks like 'good' + 'not').",
                "Out-of-vocabulary words are invisible and contribute exactly zero, so unseen "
                "phrasing is silently dropped.",
                "Domain-bound to the training text; predictions degrade on any other style or topic.",
            ],
        )
