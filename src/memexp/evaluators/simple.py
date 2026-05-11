from __future__ import annotations

from dataclasses import dataclass
import re

from memexp.agents.base import AnswerRecord
from memexp.core.dataset import Dataset, DatasetItem, DatasetQuestion
from memexp.evaluators.base import EvaluationRecord


@dataclass(frozen=True)
class ContainsEvaluatorConfig:
    policy: str = "normalized_contains_v1"
    tuple_reference_policy: str = "any"


class ContainsEvaluator:
    name = "contains"

    def __init__(self, config: ContainsEvaluatorConfig | None = None) -> None:
        self.config = config or ContainsEvaluatorConfig()
        if self.config.policy != "normalized_contains_v1":
            raise ValueError(
                f"Unsupported contains evaluator policy: {self.config.policy}")
        if self.config.tuple_reference_policy not in {"any", "all"}:
            raise ValueError("tuple_reference_policy must be one of: any, all")

    def evaluate(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        *,
        dataset: Dataset | None = None,
        item: DatasetItem | None = None,
    ) -> EvaluationRecord:
        reference = _question_reference(question)
        if reference is None:
            return EvaluationRecord(
                item_id=answer.item_id,
                question_id=answer.question_id,
                evaluator_name=self.name,
                score=None,
                passed=None,
                reference=None,
                metrics={
                    "evaluated": False,
                    "reason": "missing_reference",
                },
            )

        references = _reference_terms(reference)
        normalized_answer = _normalize(answer.answer)
        matches = tuple(reference for reference in references
                        if _normalize(reference) in normalized_answer)
        if self.config.tuple_reference_policy == "all":
            passed = len(matches) == len(references)
            score = len(matches) / len(references) if references else 0.0
        else:
            passed = bool(matches)
            score = 1.0 if passed else 0.0

        return EvaluationRecord(
            item_id=answer.item_id,
            question_id=answer.question_id,
            evaluator_name=self.name,
            score=score,
            passed=passed,
            reference=reference,
            metrics={
                "evaluated": True,
                "policy": self.config.policy,
                "tuple_reference_policy": self.config.tuple_reference_policy,
                "reference_count": len(references),
                "matched_count": len(matches),
                "matched_references": matches,
            },
        )


def _reference_terms(reference: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(reference, str):
        return (reference, )
    return tuple(reference)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _question_reference(
        question: DatasetQuestion) -> str | tuple[str, ...] | None:
    if question.label is not None:
        value = question.label.reference_answer
        if value is not None:
            return value if isinstance(value, tuple) else str(value)
    return None
