from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from memexp.agents.base import AnswerRecord
from memexp.core.dataset import Dataset, DatasetItem, DatasetQuestion


@dataclass(frozen=True)
class EvaluationRecord:
    item_id: str
    question_id: str
    evaluator_name: str
    score: float | None
    passed: bool | None
    reference: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        *,
        dataset: Dataset | None = None,
        item: DatasetItem | None = None,
    ) -> EvaluationRecord:
        ...
