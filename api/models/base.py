from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Aspect:
    term: str
    polarity: str


@dataclass
class AnalysisResult:
    model_id: str
    label: str  # "positive" | "negative" | "neutral"
    confidence: float  # 0..1
    scores: dict[str, float] = field(default_factory=dict)
    aspects: list[Aspect] | None = None
    available: bool = True
    error: str | None = None


@dataclass
class Explanation:
    model_id: str
    explanation_type: str  # "native" | "ai-narrated"
    summary: str
    evidence: list[dict] = field(default_factory=list)
    # evidence items: {"label": str, "detail": str (optional), "weight": float (optional)}
    method: str = ""  # one sentence naming the mechanism
    steps: list[str] = field(
        default_factory=list
    )  # ordered reasoning chain for THIS text
    biases: list[str] = field(default_factory=list)  # model-family-specific caveats


@runtime_checkable
class SentimentModel(Protocol):
    id: str
    name: str
    family: str
    description: str

    def available(self) -> bool: ...
    def analyze(self, text: str) -> AnalysisResult: ...
    def explain(self, text: str, result: AnalysisResult) -> Explanation: ...
