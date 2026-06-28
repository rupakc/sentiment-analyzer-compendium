from api.runner import analyze_many


def test_runs_selected_models_and_isolates_errors():
    results = analyze_many("I love this great thing", ["vader", "logreg"])
    ids = {r.model_id for r in results}
    assert ids == {"vader", "logreg"}


def test_unknown_model_id_yields_error_result():
    results = analyze_many("hi", ["vader", "nope"])
    bad = [r for r in results if r.model_id == "nope"][0]
    assert bad.available is False and bad.error
