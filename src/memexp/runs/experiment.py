from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from memexp.agents.base import AgentSystem, AnswerRecord
from memexp.core.dataset import Dataset
from memexp.evaluators.base import Evaluator
from memexp.memsys.base import MemorySystem
from memexp.runs.answer import AnswerRunResult, AnswerRunner
from memexp.runs.build import BuildRunResult, MemoryBuildRunner
from memexp.runs.cache import StageCache
from memexp.runs.evaluate import EvaluationRunResult, EvaluationRunner
from memexp.runs.execution import RunExecutionConfig, StageExecutionConfig
from memexp.runs.index import IndexRunResult, MemoryIndexRunner
from memexp.runs.logging import NullRunLogger, RunEvent, RunLogger


@dataclass(frozen=True)
class ExperimentRunResult:
    dataset_name: str
    build: BuildRunResult
    index: IndexRunResult
    answer: AnswerRunResult
    evaluation: EvaluationRunResult
    summary: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Convenience wrapper that composes the independent experiment loops."""

    def __init__(
        self,
        memory_system: MemorySystem,
        agent: AgentSystem,
        evaluator: Evaluator,
        *,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> None:
        self.memory_system = memory_system
        self.agent = agent
        self.evaluator = evaluator
        self.top_k = top_k
        self.context_budget_tokens = context_budget_tokens

    def run(
        self,
        dataset: Dataset,
        *,
        execution: RunExecutionConfig | None = None,
        stage_execution: StageExecutionConfig | None = None,
        logger: RunLogger | None = None,
        cache: StageCache | None = None,
        answer_record_sink: Callable[[AnswerRecord], None] | None = None,
    ) -> ExperimentRunResult:
        active_logger = logger or NullRunLogger()
        active_logger.emit(
            RunEvent(
                stage="run",
                event="started",
                metrics={
                    "dataset": dataset.name,
                    "item_count": len(dataset.items),
                    "question_count": sum(len(item.questions) for item in dataset.items),
                },
            )
        )
        build = MemoryBuildRunner(self.memory_system).run(
            dataset,
            execution=stage_execution.build if stage_execution else execution,
            logger=active_logger,
            cache=cache,
        )
        active_logger.emit(RunEvent(stage="build", event="summary", metrics=build.summary))
        index = MemoryIndexRunner(self.memory_system).run(
            dataset,
            build,
            execution=stage_execution.index if stage_execution else execution,
            logger=active_logger,
        )
        active_logger.emit(RunEvent(stage="index", event="summary", metrics=index.summary))
        answer = AnswerRunner(
            self.memory_system,
            self.agent,
            top_k=self.top_k,
            context_budget_tokens=self.context_budget_tokens,
        ).run(
            dataset,
            build,
            execution=stage_execution.answer if stage_execution else execution,
            logger=active_logger,
            cache=cache,
            record_sink=answer_record_sink,
        )
        active_logger.emit(RunEvent(stage="answer", event="summary", metrics=answer.summary))
        evaluation = EvaluationRunner(self.evaluator).run(
            dataset,
            answer,
            execution=stage_execution.evaluate if stage_execution else execution,
            logger=active_logger,
            cache=cache,
        )
        active_logger.emit(
            RunEvent(stage="evaluate", event="summary", metrics=evaluation.summary)
        )
        result = ExperimentRunResult(
            dataset_name=dataset.name,
            build=build,
            index=index,
            answer=answer,
            evaluation=evaluation,
            summary={
                "dataset": dataset.name,
                "build": build.summary,
                "index": index.summary,
                "answer": answer.summary,
                "evaluation": evaluation.summary,
            },
        )
        active_logger.emit(
            RunEvent(stage="run", event="completed", metrics=result.summary)
        )
        return result
