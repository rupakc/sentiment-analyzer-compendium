from api.models.vader import VaderModel

m = VaderModel()


def test_positive_text():
    r = m.analyze("I absolutely love this, it is great!")
    assert r.label == "positive" and r.confidence > 0.5


def test_negative_text():
    r = m.analyze("This is terrible and I hate it.")
    assert r.label == "negative"


def test_explain_lists_word_contributions():
    r = m.analyze("great movie")
    e = m.explain("great movie", r)
    assert e.explanation_type == "native"
    assert any(ev.get("label") == "great" for ev in e.evidence)
    assert e.method
    assert e.steps
    assert e.biases
