from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memexp.core.contracts import MemoryReadRequest, MemoryScope


@dataclass(frozen=True)
class QuestionLabel:
    reference_answer: Any = None
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetQuestion:
    question_id: str
    query: str | dict[str, Any]
    query_time: str | None = None
    label: QuestionLabel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_read_request(
        self,
        *,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> MemoryReadRequest:
        return MemoryReadRequest(
            query=self.query,
            query_id=self.question_id,
            query_time=self.query_time,
            top_k=top_k,
            context_budget_tokens=context_budget_tokens,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class DatasetItem:
    item_id: str
    conversations: tuple[tuple[dict[str, Any], ...], ...]
    questions: tuple[DatasetQuestion, ...]
    subject_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_scope(self, *, dataset_name: str) -> MemoryScope:
        return MemoryScope(
            scope_id=self.item_id,
            dataset=dataset_name,
            subject_id=self.subject_id,
            timeline_id=self.item_id,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class Dataset:
    name: str
    items: tuple[DatasetItem, ...]
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
