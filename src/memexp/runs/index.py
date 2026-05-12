from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from memexp.core.contracts import MemoryArtifact
from memexp.core.dataset import Dataset
from memexp.memsys.base import MemorySystem
from memexp.runs.build import BuildRunResult
from memexp.runs.execution import RunExecutionConfig, RunTask, execute_run_tasks
from memexp.runs.logging import RunLogger


@dataclass(frozen=True)
class IndexRecord:
    item_id: str
    memory_artifact_id: str
    system_name: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexRunResult:
    dataset_name: str
    memory_system_name: str
    records: tuple[IndexRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict)


class MemoryIndexRunner:
    """Materialize artifact-level indexes and storage embedding caches."""

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
        build_result: BuildRunResult | Mapping[str, MemoryArtifact],
        *,
        execution: RunExecutionConfig | None = None,
        logger: RunLogger | None = None,
    ) -> IndexRunResult:
        artifacts_by_item_id = _artifact_map(build_result)

        def index_item(index: int) -> IndexRecord:
            item = dataset.items[index]
            artifact = artifacts_by_item_id.get(item.item_id)
            if artifact is None:
                raise KeyError(f"Missing memory artifact for item_id={item.item_id}")
            stats = _index_artifact(self.memory_system, artifact)
            return IndexRecord(
                item_id=item.item_id,
                memory_artifact_id=artifact.artifact_id,
                system_name=artifact.system_name,
                stats=stats,
            )

        tasks = tuple(
            RunTask(
                index=index,
                item_id=item.item_id,
                question_id=None,
                run=lambda index=index: index_item(index),
            )
            for index, item in enumerate(dataset.items)
        )
        batch = execute_run_tasks(
            stage="index",
            tasks=tasks,
            execution=execution,
            logger=logger,
            completed_metrics=_index_record_metrics,
        )
        records = tuple(batch.results)
        indexed_count = sum(
            1 for record in records
            if record.stats.get("supported") is not False
        )
        return IndexRunResult(
            dataset_name=dataset.name,
            memory_system_name=self.memory_system_name,
            records=records,
            summary={
                "dataset": dataset.name,
                "memory_system": self.memory_system_name,
                "item_count": len(dataset.items),
                "artifact_count": len(records),
                "failed_count": batch.failed_count,
                "indexed_count": indexed_count,
            },
        )


def _artifact_map(
    build_result: BuildRunResult | Mapping[str, MemoryArtifact],
) -> Mapping[str, MemoryArtifact]:
    if isinstance(build_result, BuildRunResult):
        return build_result.artifacts_by_item_id
    return build_result


def _index_artifact(
    memory_system: MemorySystem,
    artifact: MemoryArtifact,
) -> dict[str, Any]:
    index = getattr(memory_system, "index_artifact", None)
    if not callable(index):
        return {
            "supported": False,
            "reason": "memory_system_has_no_index_artifact",
            "artifact_id": artifact.artifact_id,
            "unit_count": len(artifact.units),
        }
    stats = index(artifact)
    if not isinstance(stats, dict):
        raise TypeError("index_artifact() must return a dict")
    return {
        "supported": True,
        "artifact_id": artifact.artifact_id,
        "unit_count": len(artifact.units),
        **stats,
    }


def _index_record_metrics(record: IndexRecord) -> dict[str, Any]:
    storage_cache = record.stats.get("storage_embedding_cache")
    metrics = {
        "artifact_id": record.memory_artifact_id,
        "supported": record.stats.get("supported", True),
    }
    if isinstance(storage_cache, dict):
        metrics["storage_embedding_cache_enabled"] = storage_cache.get("enabled")
        metrics["storage_embedding_cache_hits"] = storage_cache.get("hits")
        metrics["storage_embedding_cache_misses"] = storage_cache.get("misses")
        metrics["storage_embedding_cache_writes"] = storage_cache.get("writes")
    return metrics
