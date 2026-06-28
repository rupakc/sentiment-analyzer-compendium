from api.models.crf import CrfModel

m = CrfModel()


def test_available():
    assert m.available() is True


def test_extracts_aspect():
    r = m.analyze("the battery is great")
    assert r.aspects is not None
    assert any(a.term == "battery" for a in r.aspects)


def test_explain_native():
    r = m.analyze("the battery is great")
    e = m.explain("the battery is great", r)
    assert e.explanation_type == "native"
