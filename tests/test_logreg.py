from api.models.logreg import LogRegModel

m = LogRegModel()


def test_available_when_pickle_present():
    assert m.available() is True


def test_positive_text():
    r = m.analyze("I love this wonderful great experience")
    assert r.label == "positive"


def test_explain_lists_top_tokens():
    r = m.analyze("terrible awful bad")
    e = m.explain("terrible awful bad", r)
    assert e.explanation_type == "native"
    assert len(e.evidence) >= 1 and "token" in e.evidence[0]
