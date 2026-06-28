import respx
import httpx
from api.models.distilbert import DistilBertModel, HF_URL


def test_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    assert DistilBertModel().available() is False


@respx.mock
def test_analyze_parses_hf_response(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "x")
    respx.post(HF_URL).mock(return_value=httpx.Response(
        200, json=[[{"label": "POSITIVE", "score": 0.99},
                    {"label": "NEGATIVE", "score": 0.01}]]))
    r = DistilBertModel().analyze("great")
    assert r.label == "positive" and r.confidence > 0.9
