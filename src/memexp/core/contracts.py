from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryScope:
    """Library-level scope for one causally ordered memory timeline."""

    scope_id: str
    dataset: str | None = None
    subject_id: str | None = None
    timeline_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryUnit:
    unit_id: str
    text: str
    timestamp: str | None = None
    available_at: str | None = None
    source_time_start: str | None = None
    source_time_end: str | None = None
    source_ids: tuple[str, ...] = ()
    memory_type: str = "fact"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryArtifact:
    artifact_id: str
    system_name: str
    scope: MemoryScope
    units: tuple[MemoryUnit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryReadRequest:
    query: str | dict[str, Any]
    query_id: str | None = None
    query_time: str | None = None
    top_k: int | None = None
    context_budget_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedMemoryUnit:
    unit: MemoryUnit
    rank: int
    score: float
    retrieval_text: str
    score_breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedContext:
    text: str
    token_count: int
    block_count: int = 0
    timepoint_count: int | None = None


@dataclass(frozen=True)
class MemoryReadResult:
    request: MemoryReadRequest
    ranked_units: tuple[RankedMemoryUnit, ...]
    context: PackedContext
    stats: dict[str, Any] = field(default_factory=dict)
    trace_ref: str | None = None
