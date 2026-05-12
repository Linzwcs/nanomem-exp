from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger


@dataclass(frozen=True)
class RunExecutionConfig:
    max_workers: int = 1
    fail_fast: bool = True
    preserve_order: bool = True


@dataclass(frozen=True)
class StageExecutionConfig:
    build: RunExecutionConfig = field(default_factory=RunExecutionConfig)
    index: RunExecutionConfig = field(default_factory=RunExecutionConfig)
    answer: RunExecutionConfig = field(default_factory=RunExecutionConfig)
    evaluate: RunExecutionConfig = field(default_factory=RunExecutionConfig)


@dataclass(frozen=True)
class RunTask:
    index: int
    item_id: str | None
    question_id: str | None
    run: Callable[[], Any]


@dataclass(frozen=True)
class RunTaskBatchResult:
    results: tuple[Any, ...]
    failed_count: int


def execute_run_tasks(
    *,
    stage: str,
    tasks: tuple[RunTask, ...],
    execution: RunExecutionConfig | None = None,
    logger: RunLogger | None = None,
    completed_metrics: Callable[[Any], dict[str, Any]] | None = None,
) -> RunTaskBatchResult:
    config = execution or RunExecutionConfig()
    active_logger = logger or NullRunLogger()
    started_at = perf_counter()
    active_logger.emit(
        RunEvent(
            stage=stage,
            event="batch_started",
            metrics={
                "task_count": len(tasks),
                "max_workers": config.max_workers,
                "fail_fast": config.fail_fast,
                "preserve_order": config.preserve_order,
            },
        )
    )
    try:
        if config.max_workers <= 1 or len(tasks) <= 1:
            batch = _execute_serial(
                stage=stage,
                tasks=tasks,
                execution=config,
                logger=active_logger,
                completed_metrics=completed_metrics,
            )
        else:
            batch = _execute_parallel(
                stage=stage,
                tasks=tasks,
                execution=config,
                logger=active_logger,
                completed_metrics=completed_metrics,
            )
    except Exception as exc:
        active_logger.emit(
            RunEvent(
                stage=stage,
                event="batch_failed",
                message=str(exc),
                metrics={
                    "task_count": len(tasks),
                    "duration_ms": _elapsed_ms(started_at),
                    "error_type": type(exc).__name__,
                },
            )
        )
        raise

    active_logger.emit(
        RunEvent(
            stage=stage,
            event="batch_completed",
            metrics={
                "task_count": len(tasks),
                "completed_count": len(batch.results),
                "failed_count": batch.failed_count,
                "duration_ms": _elapsed_ms(started_at),
            },
        )
    )
    return batch


def _execute_serial(
    *,
    stage: str,
    tasks: tuple[RunTask, ...],
    execution: RunExecutionConfig,
    logger: RunLogger,
    completed_metrics: Callable[[Any], dict[str, Any]] | None,
) -> RunTaskBatchResult:
    indexed_results: dict[int, Any] = {}
    failed_count = 0
    for task in tasks:
        try:
            indexed_results[task.index] = _execute_one(
                stage=stage,
                task=task,
                logger=logger,
                completed_metrics=completed_metrics,
            )
        except Exception:
            failed_count += 1
            if execution.fail_fast:
                raise
    return RunTaskBatchResult(
        results=_ordered_results(
            indexed_results,
            preserve_order=execution.preserve_order,
        ),
        failed_count=failed_count,
    )


def _execute_parallel(
    *,
    stage: str,
    tasks: tuple[RunTask, ...],
    execution: RunExecutionConfig,
    logger: RunLogger,
    completed_metrics: Callable[[Any], dict[str, Any]] | None,
) -> RunTaskBatchResult:
    indexed_results: dict[int, Any] = {}
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max(1, execution.max_workers)) as executor:
        future_to_task = {
            executor.submit(
                _execute_one,
                stage=stage,
                task=task,
                logger=logger,
                completed_metrics=completed_metrics,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                indexed_results[task.index] = future.result()
            except Exception:
                failed_count += 1
                if execution.fail_fast:
                    raise
    return RunTaskBatchResult(
        results=_ordered_results(
            indexed_results,
            preserve_order=execution.preserve_order,
        ),
        failed_count=failed_count,
    )


def _execute_one(
    *,
    stage: str,
    task: RunTask,
    logger: RunLogger,
    completed_metrics: Callable[[Any], dict[str, Any]] | None,
) -> Any:
    started_at = perf_counter()
    logger.emit(
        RunEvent(
            stage=stage,
            event="started",
            item_id=task.item_id,
            question_id=task.question_id,
            metrics={"task_index": task.index},
        )
    )
    try:
        result = task.run()
    except Exception as exc:
        logger.emit(
            RunEvent(
                stage=stage,
                event="failed",
                item_id=task.item_id,
                question_id=task.question_id,
                message=str(exc),
                metrics={
                    "task_index": task.index,
                    "duration_ms": _elapsed_ms(started_at),
                    "error_type": type(exc).__name__,
                },
            )
        )
        raise

    metrics = completed_metrics(result) if completed_metrics else {}
    metrics = {
        "task_index": task.index,
        "duration_ms": _elapsed_ms(started_at),
        **metrics,
    }
    logger.emit(
        RunEvent(
            stage=stage,
            event="completed",
            item_id=task.item_id,
            question_id=task.question_id,
            metrics=metrics,
        )
    )
    return result


def _ordered_results(
    indexed_results: dict[int, Any],
    *,
    preserve_order: bool,
) -> tuple[Any, ...]:
    if preserve_order:
        return tuple(
            indexed_results[index]
            for index in sorted(indexed_results)
        )
    return tuple(indexed_results.values())


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
