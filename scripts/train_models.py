"""Train tiny sentiment models on a small labeled set; export slim artifacts.

ponytail: tiny built-in datasets keep the repo self-contained and CPU-fast.
Swap CORPUS / SENTS for SST-2 and SemEval ABSA subsets when accuracy matters.

Training uses scikit-learn (logreg) but we EXPORT plain artifacts so the
serverless runtime needs neither scikit-learn nor scipy (the 148 MB that blew
the Vercel function limit):
  - logreg -> artifacts/logreg.json   (vocab + idf + coefficients; pure-python inference)
  - crf    -> artifacts/crf.crfsuite  (native python-crfsuite model; no sklearn wrapper)
"""

import json
import pathlib
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pycrfsuite

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from api.models.crf_features import token_features  # noqa: E402

ART = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
ART.mkdir(exist_ok=True)

CORPUS = [
    ("I love this, it is wonderful and great", "positive"),
    ("Absolutely fantastic, highly recommend", "positive"),
    ("Best experience ever, so happy", "positive"),
    ("A delightful and excellent product", "positive"),
    ("This is terrible and I hate it", "negative"),
    ("Awful, disappointing and bad", "negative"),
    ("Worst product, completely useless", "negative"),
    ("I regret buying this, very poor", "negative"),
    ("It is okay, nothing special", "neutral"),
    ("Average, neither good nor bad", "neutral"),
]

# BIO-ish aspect data: tokens with (word, tag). ASP = aspect term.
SENTS = [
    [("the", "O"), ("battery", "ASP"), ("is", "O"), ("great", "O")],
    [("the", "O"), ("screen", "ASP"), ("is", "O"), ("terrible", "O")],
    [
        ("amazing", "O"),
        ("camera", "ASP"),
        ("but", "O"),
        ("poor", "O"),
        ("sound", "ASP"),
    ],
    [("the", "O"), ("food", "ASP"), ("was", "O"), ("delicious", "O")],
    [("slow", "O"), ("service", "ASP"), ("and", "O"), ("cold", "O"), ("coffee", "ASP")],
    [("the", "O"), ("price", "ASP"), ("is", "O"), ("fair", "O")],
]


def train_logreg():
    X, y = zip(*CORPUS)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    # Export everything pure-python inference needs to replicate the pipeline.
    artifact = {
        "vocab": {term: int(i) for term, i in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": clf.coef_.tolist(),  # (n_classes, n_feat); (1, n) when binary
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
    }
    (ART / "logreg.json").write_text(json.dumps(artifact))
    print("trained logreg ->", ART / "logreg.json")


def train_crf():
    trainer = pycrfsuite.Trainer(verbose=False)
    for s in SENTS:
        words = [w for w, _ in s]
        xseq = [token_features(words, i) for i in range(len(s))]
        yseq = [tag for _, tag in s]
        trainer.append(xseq, yseq)
    trainer.set_params(
        {
            "c1": 0.1,
            "c2": 0.1,
            "max_iterations": 50,
            "feature.possible_transitions": True,
        }
    )
    trainer.train(str(ART / "crf.crfsuite"))
    print("trained crf ->", ART / "crf.crfsuite")


if __name__ == "__main__":
    train_logreg()
    train_crf()
