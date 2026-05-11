from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger


@dataclass(frozen=True)
class RunExecutionConfig:
    max_workers: int = 1
    fail_fast: bool = True
    preserve_order: bool = True


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
    if config.max_workers <= 1 or len(tasks) <= 1:
        return _execute_serial(
            stage=stage,
            tasks=tasks,
            execution=config,
            logger=active_logger,
            completed_metrics=completed_metrics,
        )
    return _execute_parallel(
        stage=stage,
        tasks=tasks,
        execution=config,
        logger=active_logger,
        completed_metrics=completed_metrics,
    )


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
    logger.emit(
        RunEvent(
            stage=stage,
            event="started",
            item_id=task.item_id,
            question_id=task.question_id,
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
                metrics={"error_type": type(exc).__name__},
            )
        )
        raise

    metrics = completed_metrics(result) if completed_metrics else {}
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
