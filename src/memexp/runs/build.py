from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memexp.core.contracts import MemoryArtifact
from memexp.core.dataset import Dataset
from memexp.memsys.base import MemorySystem
from memexp.runs.cache import (
    StageCache,
    cache_key,
    dataset_cache_spec,
    object_cache_spec,
    to_jsonable,
)
from memexp.runs.execution import RunExecutionConfig, RunTask, execute_run_tasks
from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger
from memexp.runs.serialization import build_record_from_dict, build_record_to_dict


@dataclass(frozen=True)
class BuildRecord:
    item_id: str
    artifact: MemoryArtifact
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildRunResult:
    dataset_name: str
    memory_system_name: str
    records: tuple[BuildRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def artifacts_by_item_id(self) -> dict[str, MemoryArtifact]:
        return {record.item_id: record.artifact for record in self.records}


class MemoryBuildRunner:
    """Build one memory artifact per dataset item."""

    def __init__(self, memory_system: MemorySystem) -> None:
        self.memory_system = memory_system
        self.memory_system_name = getattr(
            memory_system,
            "name",
            type(memory_system).__name__,
        )

    def run(
        self,
        dataset: Dataset,
        *,
        execution: RunExecutionConfig | None = None,
        logger: RunLogger | None = None,
        cache: StageCache | None = None,
    ) -> BuildRunResult:
        active_logger = logger or NullRunLogger()

        def build_item(index: int) -> tuple[BuildRecord, bool]:
            item = dataset.items[index]
            key = _build_cache_key(dataset, item, self.memory_system)
            if cache is not None:
                cached = cache.load("build", key)
                if cached is not None:
                    record = build_record_from_dict(cached, BuildRecord)
                    _prepare_build_artifact(self.memory_system, record.artifact)
                    active_logger.emit(
                        RunEvent(
                            stage="build",
                            event="cache_hit",
                            item_id=item.item_id,
                            metrics={"cache_key": key},
                        )
                    )
                    return record, True
                active_logger.emit(
                    RunEvent(
                        stage="build",
                        event="cache_miss",
                        item_id=item.item_id,
                        metrics={"cache_key": key},
                    )
                )

            artifact = self.memory_system.build(
                [list(conversation) for conversation in item.conversations],
                scope=item.to_memory_scope(dataset_name=dataset.name),
            )
            unit_count = len(artifact.units)
            record = BuildRecord(
                item_id=item.item_id,
                artifact=artifact,
                stats={
                    "artifact_id": artifact.artifact_id,
                    "unit_count": unit_count,
                    "storage_token_stats": artifact.metadata.get(
                        "storage_token_stats"
                    ),
                },
            )
            if cache is not None:
                cache.store(
                    "build",
                    key,
                    build_record_to_dict(record),
                    metadata={
                        "item_id": item.item_id,
                        "memory_system": self.memory_system_name,
                    },
                )
            return record, False

        tasks = tuple(
            RunTask(
                index=index,
                item_id=item.item_id,
                question_id=None,
                run=lambda index=index: build_item(index),
            )
            for index, item in enumerate(dataset.items)
        )
        batch = execute_run_tasks(
            stage="build",
            tasks=tasks,
            execution=execution,
            logger=logger,
            completed_metrics=_build_task_metrics,
        )
        records = tuple(result[0] for result in batch.results)
        cache_hit_count = sum(1 for result in batch.results if result[1])
        total_units = sum(int(record.stats.get("unit_count") or 0) for record in records)
        return BuildRunResult(
            dataset_name=dataset.name,
            memory_system_name=self.memory_system_name,
            records=records,
            summary={
                "dataset": dataset.name,
                "memory_system": self.memory_system_name,
                "item_count": len(dataset.items),
                "artifact_count": len(records),
                "failed_count": batch.failed_count,
                "cache_hit_count": cache_hit_count,
                "cache_miss_count": (
                    len(records) - cache_hit_count if cache is not None else 0
                ),
                "total_units": total_units,
                "avg_units_per_item": (
                    total_units / len(records) if records else 0.0
                ),
            },
        )


def _build_record_metrics(record: BuildRecord) -> dict[str, Any]:
    return {
        "artifact_id": record.artifact.artifact_id,
        "unit_count": len(record.artifact.units),
    }


def _build_task_metrics(result: tuple[BuildRecord, bool]) -> dict[str, Any]:
    record, cache_hit = result
    metrics = _build_record_metrics(record)
    metrics["cache_hit"] = cache_hit
    return metrics


def _prepare_build_artifact(
    memory_system: MemorySystem,
    artifact: MemoryArtifact,
) -> None:
    prepare = getattr(memory_system, "prepare_build_artifact", None)
    if callable(prepare):
        prepare(artifact)


def _build_cache_key(dataset: Dataset, item: Any, memory_system: MemorySystem) -> str:
    return cache_key(
        "build",
        {
            "dataset": dataset_cache_spec(dataset),
            "item": {
                "item_id": item.item_id,
                "subject_id": item.subject_id,
                "metadata": to_jsonable(item.metadata),
                "conversations": to_jsonable(item.conversations),
            },
            "memory_system": object_cache_spec(memory_system),
        },
    )
