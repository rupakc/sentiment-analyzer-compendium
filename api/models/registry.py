from api.models.base import SentimentModel
from api.models.vader import VaderModel
from api.models.logreg import LogRegModel
from api.models.crf import CrfModel
from api.models.distilbert import DistilBertModel
from api.models.llm import LlmModel

REGISTRY: dict[str, SentimentModel] = {
    m.id: m
    for m in (VaderModel(), LogRegModel(), CrfModel(), DistilBertModel(), LlmModel())
}


def get_model(model_id: str) -> SentimentModel:
    if model_id not in REGISTRY:
        raise KeyError(model_id)
    return REGISTRY[model_id]


def list_models() -> list[dict]:
    return [
        {
            "id": m.id,
            "name": m.name,
            "family": m.family,
            "description": m.description,
            "available": m.available(),
        }
        for m in REGISTRY.values()
    ]
