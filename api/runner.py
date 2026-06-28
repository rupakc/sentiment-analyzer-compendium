from concurrent.futures import ThreadPoolExecutor
from api.models.registry import get_model
from api.models.base import AnalysisResult


def _one(text: str, model_id: str) -> AnalysisResult:
    try:
        return get_model(model_id).analyze(text)
    except KeyError:
        return AnalysisResult(model_id, "neutral", 0.0, available=False,
                              error="unknown model")
    except Exception as e:
        return AnalysisResult(model_id, "neutral", 0.0, available=True, error=str(e))


def analyze_many(text: str, model_ids: list[str]) -> list[AnalysisResult]:
    with ThreadPoolExecutor(max_workers=min(8, len(model_ids))) as ex:
        return list(ex.map(lambda mid: _one(text, mid), model_ids))
