from api.models.registry import REGISTRY, list_models, get_model


def test_registry_has_five_models():
    assert set(REGISTRY) == {"vader", "logreg", "crf", "distilbert", "llm"}


def test_list_models_shape():
    items = list_models()
    assert all({"id", "name", "family", "description", "available"} <= set(m) for m in items)


def test_get_model_returns_matching_id():
    assert get_model("vader").id == "vader"
