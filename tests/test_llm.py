from unittest.mock import patch, MagicMock
from api.models.llm import LlmModel


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert LlmModel().available() is False


def test_analyze_parses_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake = MagicMock()
    fake.content = [MagicMock(text='{"label":"positive","confidence":0.92,"reason":"clear praise"}')]
    with patch("api.models.llm._client") as c:
        c.messages.create.return_value = fake
        r = LlmModel().analyze("I love it")
    assert r.label == "positive" and r.confidence == 0.92
