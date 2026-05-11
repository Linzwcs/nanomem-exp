from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from memexp.agents.base import AgentSystem, AnswerRecord
from memexp.core.contracts import MemoryArtifact
from memexp.core.dataset import Dataset
from memexp.memsys.base import MemorySystem
from memexp.runs.build import BuildRunResult
from memexp.runs.cache import (
    StageCache,
    cache_key,
    dataset_cache_spec,
    object_cache_spec,
    to_jsonable,
)
from memexp.runs.execution import RunExecutionConfig, RunTask, execute_run_tasks
from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger
from memexp.runs.serialization import answer_record_from_dict, answer_record_to_dict


@dataclass(frozen=True)
class AnswerRunResult:
    dataset_name: str
    agent_name: str
    records: tuple[AnswerRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    def record_for(self, item_id: str, question_id: str) -> AnswerRecord:
        for record in self.records:
            if record.item_id == item_id and record.question_id == question_id:
                return record
        raise KeyError(f"Missing answer record for {item_id}/{question_id}")


class AnswerRunner:
    """Generate answers from prebuilt memory artifacts."""

    def __init__(
        self,
        memory_system: MemorySystem,
        agent: AgentSystem,
        *,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> None:
        self.memory_system = memory_system
        self.agent = agent
        self.top_k = top_k
        self.context_budget_tokens = context_budget_tokens
        self.agent_name = getattr(agent, "name", type(agent).__name__)

    def run(
        self,
        dataset: Dataset,
        build_result: BuildRunResult | Mapping[str, MemoryArtifact],
        *,
        execution: RunExecutionConfig | None = None,
        logger: RunLogger | None = None,
        cache: StageCache | None = None,
        record_sink: Callable[[AnswerRecord], None] | None = None,
    ) -> AnswerRunResult:
        artifacts_by_item_id = _artifact_map(build_result)
        active_logger = logger or NullRunLogger()
        tasks: list[RunTask] = []

        def answer_question(
            item_index: int,
            question_index: int,
        ) -> tuple[AnswerRecord, bool]:
            item = dataset.items[item_index]
            artifact = artifacts_by_item_id.get(item.item_id)
            if artifact is None:
                raise KeyError(f"Missing memory artifact for item_id={item.item_id}")
            question = item.questions[question_index]
            key = _answer_cache_key(
                dataset,
                item.item_id,
                question,
                artifact,
                self.memory_system,
                self.agent,
                top_k=self.top_k,
                context_budget_tokens=self.context_budget_tokens,
            )
            if cache is not None:
                cached = cache.load("answer", key)
                if cached is not None:
                    active_logger.emit(
                        RunEvent(
                            stage="answer",
                            event="cache_hit",
                            item_id=item.item_id,
                            question_id=question.question_id,
                            metrics={"cache_key": key},
                        )
                    )
                    record = answer_record_from_dict(cached)
                    if record_sink is not None:
                        record_sink(record)
                    return record, True
                active_logger.emit(
                    RunEvent(
                        stage="answer",
                        event="cache_miss",
                        item_id=item.item_id,
                        question_id=question.question_id,
                        metrics={"cache_key": key},
                    )
                )

            memory_runtime = self.memory_system.load(artifact)
            record = self.agent.answer(
                question,
                memory_runtime,
                item_id=item.item_id,
                top_k=self.top_k,
                context_budget_tokens=self.context_budget_tokens,
            )
            if cache is not None:
                cache.store(
                    "answer",
                    key,
                    answer_record_to_dict(record),
                    metadata={
                        "item_id": item.item_id,
                        "question_id": question.question_id,
                        "agent": self.agent_name,
                    },
                )
            if record_sink is not None:
                record_sink(record)
            return record, False

        task_index = 0
        for item_index, item in enumerate(dataset.items):
            for question_index, question in enumerate(item.questions):
                tasks.append(
                    RunTask(
                        index=task_index,
                        item_id=item.item_id,
                        question_id=question.question_id,
                        run=lambda item_index=item_index, question_index=question_index: (
                            answer_question(item_index, question_index)
                        ),
                    )
                )
                task_index += 1

        batch = execute_run_tasks(
            stage="answer",
            tasks=tuple(tasks),
            execution=execution,
            logger=logger,
            completed_metrics=_answer_task_metrics,
        )
        records = tuple(result[0] for result in batch.results)
        cache_hit_count = sum(1 for result in batch.results if result[1])
        memory_read_count = sum(len(record.memory_reads) for record in records)
        total_context_tokens = sum(
            read.context.token_count
            for record in records
            for read in record.memory_reads
        )

        return AnswerRunResult(
            dataset_name=dataset.name,
            agent_name=self.agent_name,
            records=records,
            summary={
                "dataset": dataset.name,
                "agent": self.agent_name,
                "question_count": sum(len(item.questions) for item in dataset.items),
                "answer_count": len(records),
                "failed_count": batch.failed_count,
                "cache_hit_count": cache_hit_count,
                "cache_miss_count": (
                    len(records) - cache_hit_count if cache is not None else 0
                ),
                "memory_read_count": memory_read_count,
                "total_context_tokens": total_context_tokens,
                "avg_context_tokens": (
                    total_context_tokens / memory_read_count
                    if memory_read_count
                    else 0.0
                ),
            },
        )


def _artifact_map(
    build_result: BuildRunResult | Mapping[str, MemoryArtifact],
) -> Mapping[str, MemoryArtifact]:
    if isinstance(build_result, BuildRunResult):
        return build_result.artifacts_by_item_id
    return build_result


def _answer_record_metrics(record: AnswerRecord) -> dict[str, Any]:
    return {
        "memory_artifact_id": record.memory_artifact_id,
        "memory_read_count": len(record.memory_reads),
        "context_tokens": sum(read.context.token_count for read in record.memory_reads),
    }


def _answer_task_metrics(result: tuple[AnswerRecord, bool]) -> dict[str, Any]:
    record, cache_hit = result
    metrics = _answer_record_metrics(record)
    metrics["cache_hit"] = cache_hit
    return metrics


def _answer_cache_key(
    dataset: Dataset,
    item_id: str,
    question: Any,
    artifact: MemoryArtifact,
    memory_system: MemorySystem,
    agent: AgentSystem,
    *,
    top_k: int | None,
    context_budget_tokens: int | None,
) -> str:
    return cache_key(
        "answer",
        {
            "dataset": dataset_cache_spec(dataset),
            "item_id": item_id,
            "question": {
                "question_id": question.question_id,
                "query": to_jsonable(question.query),
                "query_time": question.query_time,
                "metadata": to_jsonable(question.metadata),
            },
            "memory_artifact": {
                "artifact_id": artifact.artifact_id,
                "system_name": artifact.system_name,
            },
            "memory_system": object_cache_spec(memory_system),
            "agent": object_cache_spec(agent),
            "read": {
                "top_k": top_k,
                "context_budget_tokens": context_budget_tokens,
            },
        },
    )
