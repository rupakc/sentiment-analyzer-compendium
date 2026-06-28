"""Train tiny sentiment models on a small labeled set; pickle to artifacts/.

ponytail: tiny built-in datasets keep the repo self-contained and CPU-fast.
Swap CORPUS / SENTS for SST-2 and SemEval ABSA subsets when accuracy matters.
"""
import pickle
import pathlib
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn_crfsuite

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
    [("amazing", "O"), ("camera", "ASP"), ("but", "O"), ("poor", "O"), ("sound", "ASP")],
    [("the", "O"), ("food", "ASP"), ("was", "O"), ("delicious", "O")],
    [("slow", "O"), ("service", "ASP"), ("and", "O"), ("cold", "O"), ("coffee", "ASP")],
    [("the", "O"), ("price", "ASP"), ("is", "O"), ("fair", "O")],
]


def train_logreg():
    X, y = zip(*CORPUS)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    with open(ART / "logreg.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("trained logreg ->", ART / "logreg.pkl")


def train_crf():
    X = [[token_features([w for w, _ in s], i) for i in range(len(s))] for s in SENTS]
    y = [[tag for _, tag in s] for s in SENTS]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=50,
                               c1=0.1, c2=0.1, all_possible_transitions=True)
    crf.fit(X, y)
    with open(ART / "crf.pkl", "wb") as f:
        pickle.dump(crf, f)
    print("trained crf ->", ART / "crf.pkl")


if __name__ == "__main__":
    train_logreg()
    train_crf()
