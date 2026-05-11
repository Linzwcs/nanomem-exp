from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from memexp.agents.base import AnswerRecord
from memexp.core.dataset import Dataset
from memexp.evaluators.base import EvaluationRecord, Evaluator
from memexp.runs.answer import AnswerRunResult
from memexp.runs.cache import (
    StageCache,
    cache_key,
    dataset_cache_spec,
    object_cache_spec,
    to_jsonable,
)
from memexp.runs.execution import RunExecutionConfig, RunTask, execute_run_tasks
from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger
from memexp.runs.serialization import (
    answer_record_to_dict,
    evaluation_record_from_dict,
    evaluation_record_to_dict,
)


@dataclass(frozen=True)
class EvaluationRunResult:
    dataset_name: str
    evaluator_name: str
    records: tuple[EvaluationRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    def record_for(self, item_id: str, question_id: str) -> EvaluationRecord:
        for record in self.records:
            if record.item_id == item_id and record.question_id == question_id:
                return record
        raise KeyError(f"Missing evaluation record for {item_id}/{question_id}")


class EvaluationRunner:
    """Evaluate answer records without calling memory or regenerating answers."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.evaluator_name = getattr(evaluator, "name", type(evaluator).__name__)

    def run(
        self,
        dataset: Dataset,
        answer_result: AnswerRunResult | Iterable[AnswerRecord],
        *,
        execution: RunExecutionConfig | None = None,
        logger: RunLogger | None = None,
        cache: StageCache | None = None,
    ) -> EvaluationRunResult:
        answers_by_key = _answer_map(answer_result)
        active_logger = logger or NullRunLogger()
        tasks: list[RunTask] = []

        def evaluate_question(
            item_index: int,
            question_index: int,
        ) -> tuple[EvaluationRecord, bool]:
            item = dataset.items[item_index]
            question = item.questions[question_index]
            key = (item.item_id, question.question_id)
            answer = answers_by_key.get(key)
            if answer is None:
                raise KeyError(
                    f"Missing answer record for {item.item_id}/{question.question_id}"
                )
            cache_record_key = _evaluation_cache_key(
                dataset,
                item.item_id,
                question,
                answer,
                self.evaluator,
            )
            if cache is not None:
                cached = cache.load("evaluate", cache_record_key)
                if cached is not None:
                    active_logger.emit(
                        RunEvent(
                            stage="evaluate",
                            event="cache_hit",
                            item_id=item.item_id,
                            question_id=question.question_id,
                            metrics={"cache_key": cache_record_key},
                        )
                    )
                    return evaluation_record_from_dict(cached), True
                active_logger.emit(
                    RunEvent(
                        stage="evaluate",
                        event="cache_miss",
                        item_id=item.item_id,
                        question_id=question.question_id,
                        metrics={"cache_key": cache_record_key},
                    )
                )

            record = self.evaluator.evaluate(
                answer,
                question,
                dataset=dataset,
                item=item,
            )
            if cache is not None:
                cache.store(
                    "evaluate",
                    cache_record_key,
                    evaluation_record_to_dict(record),
                    metadata={
                        "item_id": item.item_id,
                        "question_id": question.question_id,
                        "evaluator": self.evaluator_name,
                    },
                )
            return record, False

        task_index = 0
        for item_index, item in enumerate(dataset.items):
            for question_index, question in enumerate(item.questions):
                key = (item.item_id, question.question_id)
                tasks.append(
                    RunTask(
                        index=task_index,
                        item_id=key[0],
                        question_id=key[1],
                        run=lambda item_index=item_index, question_index=question_index: (
                            evaluate_question(item_index, question_index)
                        ),
                    )
                )
                task_index += 1

        batch = execute_run_tasks(
            stage="evaluate",
            tasks=tuple(tasks),
            execution=execution,
            logger=logger,
            completed_metrics=_evaluation_task_metrics,
        )
        records = tuple(result[0] for result in batch.results)
        cache_hit_count = sum(1 for result in batch.results if result[1])
        evaluated = tuple(record for record in records if record.passed is not None)
        passed_count = sum(1 for record in evaluated if record.passed)
        scored = tuple(record for record in records if record.score is not None)
        score_total = sum(record.score or 0.0 for record in scored)
        return EvaluationRunResult(
            dataset_name=dataset.name,
            evaluator_name=self.evaluator_name,
            records=records,
            summary={
                "dataset": dataset.name,
                "evaluator": self.evaluator_name,
                "question_count": sum(len(item.questions) for item in dataset.items),
                "evaluation_count": len(records),
                "failed_count": batch.failed_count,
                "cache_hit_count": cache_hit_count,
                "cache_miss_count": (
                    len(records) - cache_hit_count if cache is not None else 0
                ),
                "evaluated_count": len(evaluated),
                "passed_count": passed_count,
                "accuracy": (
                    passed_count / len(evaluated) if evaluated else None
                ),
                "avg_score": (
                    score_total / len(scored) if scored else None
                ),
            },
        )


def _answer_map(
    answer_result: AnswerRunResult | Iterable[AnswerRecord],
) -> dict[tuple[str, str], AnswerRecord]:
    if isinstance(answer_result, AnswerRunResult):
        records = answer_result.records
    else:
        records = tuple(answer_result)
    return {
        (record.item_id, record.question_id): record
        for record in records
    }


def _evaluation_record_metrics(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "score": record.score,
        "passed": record.passed,
        "evaluated": record.passed is not None,
    }


def _evaluation_task_metrics(
    result: tuple[EvaluationRecord, bool],
) -> dict[str, Any]:
    record, cache_hit = result
    metrics = _evaluation_record_metrics(record)
    metrics["cache_hit"] = cache_hit
    return metrics


def _evaluation_cache_key(
    dataset: Dataset,
    item_id: str,
    question: Any,
    answer: AnswerRecord,
    evaluator: Evaluator,
) -> str:
    return cache_key(
        "evaluate",
        {
            "dataset": dataset_cache_spec(dataset),
            "item_id": item_id,
            "question": {
                "question_id": question.question_id,
                "query": to_jsonable(question.query),
                "query_time": question.query_time,
                "label": to_jsonable(question.label),
                "metadata": to_jsonable(question.metadata),
            },
            "answer": answer_record_to_dict(answer),
            "evaluator": object_cache_spec(evaluator),
        },
    )
