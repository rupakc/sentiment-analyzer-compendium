from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_ids: list[str] = Field(min_length=1)


class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_id: str
