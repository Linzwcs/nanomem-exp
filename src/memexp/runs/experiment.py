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
from memexp.runs.execution import RunExecutionConfig
from memexp.runs.logging import RunLogger


@dataclass(frozen=True)
class ExperimentRunResult:
    dataset_name: str
    build: BuildRunResult
    answer: AnswerRunResult
    evaluation: EvaluationRunResult
    summary: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Convenience wrapper that composes the three independent loops."""

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
        logger: RunLogger | None = None,
        cache: StageCache | None = None,
        answer_record_sink: Callable[[AnswerRecord], None] | None = None,
    ) -> ExperimentRunResult:
        build = MemoryBuildRunner(self.memory_system).run(
            dataset,
            execution=execution,
            logger=logger,
            cache=cache,
        )
        answer = AnswerRunner(
            self.memory_system,
            self.agent,
            top_k=self.top_k,
            context_budget_tokens=self.context_budget_tokens,
        ).run(
            dataset,
            build,
            execution=execution,
            logger=logger,
            cache=cache,
            record_sink=answer_record_sink,
        )
        evaluation = EvaluationRunner(self.evaluator).run(
            dataset,
            answer,
            execution=execution,
            logger=logger,
            cache=cache,
        )
        return ExperimentRunResult(
            dataset_name=dataset.name,
            build=build,
            answer=answer,
            evaluation=evaluation,
            summary={
                "dataset": dataset.name,
                "build": build.summary,
                "answer": answer.summary,
                "evaluation": evaluation.summary,
            },
        )
