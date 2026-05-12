from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from memexp.adapters import load_unified_dataset
from memexp.agents import (
    FixedQueryAgent,
    FixedQueryAgentConfig,
    ThinkStepByStepAgent,
    ThinkStepByStepAgentConfig,
)
from memexp.core.dataset import Dataset, DatasetItem
from memexp.evaluators import (
    ContainsEvaluator,
    ContainsEvaluatorConfig,
    DatasetPromptJudgeConfig,
    DatasetPromptJudgeEvaluator,
)
from memexp.memsys.baselines import (
    NullMemoryConfig,
    NullMemorySystem,
    RawMessageConfig,
    RawMessageMemorySystem,
)
from memexp.memsys.nanomem import (
    NanoMemConfig,
    NanoMemSystem,
    RenderConfig,
    RetrieveConfig,
    RetryConfig,
    StorageConfig,
)
from memexp.runs.cache import JsonStageCache
from memexp.runs.execution import RunExecutionConfig, StageExecutionConfig
from memexp.runs.experiment import ExperimentRunResult, ExperimentRunner
from memexp.runs.logging import CompositeRunLogger, JsonlRunLogger, TerminalRunLogger
from memexp.runs.manifest import write_run_manifest
from memexp.runs.records import JsonlRecordSink
from memexp.runs.serialization import answer_record_to_dict


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    format: str = "unified"
    max_items: int | None = None
    max_questions_per_item: int | None = None


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRunSpec:
    dataset: DatasetSpec
    memory_system: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(name="nanomem")
    )
    agent: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(name="fixed_query")
    )
    evaluator: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(name="contains")
    )
    run_id: str | None = None
    output_dir: str = "runs"
    cache_dir: str | None = None
    top_k: int | None = None
    context_budget_tokens: int | None = None
    execution: RunExecutionConfig = field(default_factory=RunExecutionConfig)
    stage_execution: StageExecutionConfig | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRunSpec":
        execution = _execution_config(payload.get("execution") or {})
        return cls(
            run_id=_optional_text(payload.get("run_id")),
            output_dir=str(payload.get("output_dir") or "runs"),
            cache_dir=_optional_text(payload.get("cache_dir")),
            dataset=_dataset_spec(payload.get("dataset")),
            memory_system=_component_spec(
                payload.get("memory_system"),
                default_name="nanomem",
            ),
            agent=_component_spec(payload.get("agent"), default_name="fixed_query"),
            evaluator=_component_spec(
                payload.get("evaluator"),
                default_name="contains",
            ),
            top_k=payload.get("top_k"),
            context_budget_tokens=payload.get("context_budget_tokens"),
            execution=execution,
            stage_execution=_stage_execution_config(
                payload.get("stage_execution"),
                fallback=execution,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRunOutput:
    run_dir: Path
    result: ExperimentRunResult
    manifest: dict[str, Any]


def load_experiment_run_spec(path: str | Path) -> ExperimentRunSpec:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Run spec must be a JSON object")
    return ExperimentRunSpec.from_dict(payload)


def execute_experiment_run_spec(spec: ExperimentRunSpec) -> ExperimentRunOutput:
    run_id = spec.run_id or _new_run_id()
    run_dir = Path(spec.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(spec.dataset)
    cache = JsonStageCache(spec.cache_dir) if spec.cache_dir else None
    events_path = run_dir / "events.jsonl"
    logger = CompositeRunLogger(JsonlRunLogger(events_path), TerminalRunLogger())
    answers_path = run_dir / "answers.jsonl"
    answer_record_sink = JsonlRecordSink(answers_path, answer_record_to_dict)

    runner = ExperimentRunner(
        _memory_system(spec.memory_system),
        _agent(spec.agent),
        _evaluator(spec.evaluator),
        top_k=spec.top_k,
        context_budget_tokens=spec.context_budget_tokens,
    )
    result = runner.run(
        dataset,
        execution=spec.execution,
        stage_execution=spec.stage_execution,
        logger=logger,
        cache=cache,
        answer_record_sink=answer_record_sink,
    )
    manifest = write_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        spec=spec.to_dict(),
        result=result,
        extra_artifacts={"events": events_path},
    )
    return ExperimentRunOutput(run_dir=run_dir, result=result, manifest=manifest)


def _load_dataset(spec: DatasetSpec) -> Dataset:
    if spec.format != "unified":
        raise ValueError(f"Unsupported dataset format: {spec.format}")
    dataset = load_unified_dataset(spec.path)
    return _limit_dataset(
        dataset,
        max_items=spec.max_items,
        max_questions_per_item=spec.max_questions_per_item,
    )


def _limit_dataset(
    dataset: Dataset,
    *,
    max_items: int | None,
    max_questions_per_item: int | None,
) -> Dataset:
    items = dataset.items[:max_items] if max_items is not None else dataset.items
    if max_questions_per_item is None:
        return Dataset(
            name=dataset.name,
            split=dataset.split,
            metadata=dataset.metadata,
            items=tuple(items),
        )
    limited_items = tuple(
        DatasetItem(
            item_id=item.item_id,
            conversations=item.conversations,
            questions=item.questions[:max_questions_per_item],
            subject_id=item.subject_id,
            metadata=item.metadata,
        )
        for item in items
    )
    return Dataset(
        name=dataset.name,
        split=dataset.split,
        metadata=dataset.metadata,
        items=limited_items,
    )


def _memory_system(spec: ComponentSpec) -> Any:
    name = spec.name
    if name == "nanomem":
        return NanoMemSystem(_nanomem_config(spec.config))
    if name == "raw_messages":
        config = dict(spec.config)
        if "target_roles" in config:
            config["target_roles"] = tuple(config["target_roles"])
        return RawMessageMemorySystem(RawMessageConfig(**config))
    if name == "null_memory":
        return NullMemorySystem(NullMemoryConfig(**spec.config))
    raise ValueError(f"Unsupported memory_system: {name}")


def _agent(spec: ComponentSpec) -> Any:
    if spec.name == "fixed_query":
        return FixedQueryAgent(FixedQueryAgentConfig(**spec.config))
    if spec.name == "think_step_by_step":
        return ThinkStepByStepAgent(ThinkStepByStepAgentConfig(**spec.config))
    raise ValueError(f"Unsupported agent: {spec.name}")


def _evaluator(spec: ComponentSpec) -> Any:
    if spec.name == "contains":
        return ContainsEvaluator(ContainsEvaluatorConfig(**spec.config))
    if spec.name == "dataset_prompt_judge":
        return DatasetPromptJudgeEvaluator(DatasetPromptJudgeConfig(**spec.config))
    raise ValueError(f"Unsupported evaluator: {spec.name}")


def _nanomem_config(payload: Mapping[str, Any]) -> NanoMemConfig:
    storage = dict(payload.get("storage") or {})
    retry = storage.get("retry")
    if isinstance(retry, dict):
        storage["retry"] = RetryConfig(**retry)

    retrieve = dict(payload.get("retrieve") or {})
    if "retrieval_fields" in retrieve:
        retrieve["retrieval_fields"] = tuple(retrieve["retrieval_fields"])

    render = dict(payload.get("render") or {})
    return NanoMemConfig(
        storage=StorageConfig(**storage),
        retrieve=RetrieveConfig(**retrieve),
        render=RenderConfig(**render),
        update_policy=str(payload.get("update_policy") or "append_only_v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _dataset_spec(value: Any) -> DatasetSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Run spec requires dataset object")
    return DatasetSpec(
        path=str(value["path"]),
        format=str(value.get("format") or "unified"),
        max_items=value.get("max_items"),
        max_questions_per_item=value.get("max_questions_per_item"),
    )


def _component_spec(value: Any, *, default_name: str) -> ComponentSpec:
    if value is None:
        return ComponentSpec(name=default_name)
    if isinstance(value, str):
        return ComponentSpec(name=value)
    if not isinstance(value, Mapping):
        raise ValueError("Component spec must be an object or name string")
    return ComponentSpec(
        name=str(value.get("name") or default_name),
        config=dict(value.get("config") or {}),
    )


def _execution_config(
    payload: Mapping[str, Any],
    *,
    fallback: RunExecutionConfig | None = None,
) -> RunExecutionConfig:
    return RunExecutionConfig(
        max_workers=int(
            payload.get(
                "max_workers",
                fallback.max_workers if fallback else 1,
            )
        ),
        fail_fast=bool(
            payload.get(
                "fail_fast",
                fallback.fail_fast if fallback else True,
            )
        ),
        preserve_order=bool(
            payload.get(
                "preserve_order",
                fallback.preserve_order if fallback else True,
            )
        ),
    )


def _stage_execution_config(
    payload: Any,
    *,
    fallback: RunExecutionConfig,
) -> StageExecutionConfig | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("stage_execution must be an object")
    return StageExecutionConfig(
        build=_execution_config(
            _stage_execution_payload(payload, "build"),
            fallback=fallback,
        ),
        index=_execution_config(
            _stage_execution_payload(payload, "index"),
            fallback=fallback,
        ),
        answer=_execution_config(
            _stage_execution_payload(payload, "answer"),
            fallback=fallback,
        ),
        evaluate=_execution_config(
            (
                _stage_execution_payload(payload, "evaluate")
                or _stage_execution_payload(payload, "evaluation")
            ),
            fallback=fallback,
        ),
    )


def _stage_execution_payload(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = payload.get(key) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"stage_execution.{key} must be an object")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"
