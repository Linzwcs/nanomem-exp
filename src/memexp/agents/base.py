from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from memexp.core.contracts import MemoryReadRequest, MemoryReadResult
from memexp.core.dataset import DatasetQuestion


class MemoryReader(Protocol):
    """Example-bound memory runtime exposed to an agent."""

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        ...


@dataclass(frozen=True)
class AnswerRecord:
    item_id: str
    question_id: str
    query: str | dict[str, Any]
    answer: str
    agent_name: str
    query_time: str | None = None
    memory_artifact_id: str | None = None
    memory_reads: tuple[MemoryReadResult, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentSystem(Protocol):
    name: str

    def answer(
        self,
        question: DatasetQuestion,
        memory_runtime: MemoryReader,
        *,
        item_id: str,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> AnswerRecord:
        ...
