from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)


def test_list_models():
    r = client.get("/api/models")
    assert r.status_code == 200 and len(r.json()) == 5


def test_analyze():
    r = client.post("/api/analyze", json={"text": "I love it", "model_ids": ["vader"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["model_id"] == "vader"


def test_analyze_validation():
    r = client.post("/api/analyze", json={"text": "", "model_ids": ["vader"]})
    assert r.status_code == 422


def test_explain():
    r = client.post("/api/explain", json={"text": "great movie", "model_id": "vader"})
    assert r.status_code == 200 and r.json()["explanation_type"] == "native"
