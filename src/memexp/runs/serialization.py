from __future__ import annotations

from typing import Any, TypeVar

from memexp.agents.base import AnswerRecord
from memexp.core.contracts import (
    MemoryArtifact,
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
    MemoryUnit,
    PackedContext,
    RankedMemoryUnit,
)
from memexp.evaluators.base import EvaluationRecord
from memexp.runs.cache import to_jsonable


T = TypeVar("T")


def memory_scope_to_dict(scope: MemoryScope) -> dict[str, Any]:
    return {
        "scope_id": scope.scope_id,
        "dataset": scope.dataset,
        "subject_id": scope.subject_id,
        "timeline_id": scope.timeline_id,
        "metadata": to_jsonable(scope.metadata),
    }


def memory_scope_from_dict(payload: dict[str, Any]) -> MemoryScope:
    return MemoryScope(
        scope_id=str(payload["scope_id"]),
        dataset=payload.get("dataset"),
        subject_id=payload.get("subject_id"),
        timeline_id=payload.get("timeline_id"),
        metadata=dict(payload.get("metadata") or {}),
    )


def memory_unit_to_dict(unit: MemoryUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "text": unit.text,
        "timestamp": unit.timestamp,
        "available_at": unit.available_at,
        "source_time_start": unit.source_time_start,
        "source_time_end": unit.source_time_end,
        "source_ids": list(unit.source_ids),
        "memory_type": unit.memory_type,
        "metadata": to_jsonable(unit.metadata),
    }


def memory_unit_from_dict(payload: dict[str, Any]) -> MemoryUnit:
    return MemoryUnit(
        unit_id=str(payload["unit_id"]),
        text=str(payload["text"]),
        timestamp=payload.get("timestamp"),
        available_at=payload.get("available_at"),
        source_time_start=payload.get("source_time_start"),
        source_time_end=payload.get("source_time_end"),
        source_ids=tuple(payload.get("source_ids") or ()),
        memory_type=str(payload.get("memory_type") or "fact"),
        metadata=dict(payload.get("metadata") or {}),
    )


def memory_artifact_to_dict(artifact: MemoryArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "system_name": artifact.system_name,
        "scope": memory_scope_to_dict(artifact.scope),
        "units": [memory_unit_to_dict(unit) for unit in artifact.units],
        "metadata": to_jsonable(artifact.metadata),
    }


def memory_artifact_from_dict(payload: dict[str, Any]) -> MemoryArtifact:
    return MemoryArtifact(
        artifact_id=str(payload["artifact_id"]),
        system_name=str(payload["system_name"]),
        scope=memory_scope_from_dict(payload["scope"]),
        units=tuple(memory_unit_from_dict(unit) for unit in payload.get("units") or ()),
        metadata=dict(payload.get("metadata") or {}),
    )


def memory_read_request_to_dict(request: MemoryReadRequest) -> dict[str, Any]:
    return {
        "query": to_jsonable(request.query),
        "query_id": request.query_id,
        "query_time": request.query_time,
        "top_k": request.top_k,
        "context_budget_tokens": request.context_budget_tokens,
        "metadata": to_jsonable(request.metadata),
    }


def memory_read_request_from_dict(payload: dict[str, Any]) -> MemoryReadRequest:
    return MemoryReadRequest(
        query=payload["query"],
        query_id=payload.get("query_id"),
        query_time=payload.get("query_time"),
        top_k=payload.get("top_k"),
        context_budget_tokens=payload.get("context_budget_tokens"),
        metadata=dict(payload.get("metadata") or {}),
    )


def ranked_memory_unit_to_dict(ranked: RankedMemoryUnit) -> dict[str, Any]:
    return {
        "unit": memory_unit_to_dict(ranked.unit),
        "rank": ranked.rank,
        "score": ranked.score,
        "retrieval_text": ranked.retrieval_text,
        "score_breakdown": to_jsonable(ranked.score_breakdown),
    }


def ranked_memory_unit_from_dict(payload: dict[str, Any]) -> RankedMemoryUnit:
    return RankedMemoryUnit(
        unit=memory_unit_from_dict(payload["unit"]),
        rank=int(payload["rank"]),
        score=float(payload["score"]),
        retrieval_text=str(payload["retrieval_text"]),
        score_breakdown=dict(payload.get("score_breakdown") or {}),
    )


def packed_context_to_dict(context: PackedContext) -> dict[str, Any]:
    return {
        "text": context.text,
        "token_count": context.token_count,
        "block_count": context.block_count,
        "timepoint_count": context.timepoint_count,
    }


def packed_context_from_dict(payload: dict[str, Any]) -> PackedContext:
    return PackedContext(
        text=str(payload["text"]),
        token_count=int(payload["token_count"]),
        block_count=int(payload.get("block_count") or 0),
        timepoint_count=payload.get("timepoint_count"),
    )


def memory_read_result_to_dict(read: MemoryReadResult) -> dict[str, Any]:
    return {
        "request": memory_read_request_to_dict(read.request),
        "ranked_units": [
            ranked_memory_unit_to_dict(ranked)
            for ranked in read.ranked_units
        ],
        "context": packed_context_to_dict(read.context),
        "stats": to_jsonable(read.stats),
        "trace_ref": read.trace_ref,
    }


def memory_read_result_from_dict(payload: dict[str, Any]) -> MemoryReadResult:
    return MemoryReadResult(
        request=memory_read_request_from_dict(payload["request"]),
        ranked_units=tuple(
            ranked_memory_unit_from_dict(ranked)
            for ranked in payload.get("ranked_units") or ()
        ),
        context=packed_context_from_dict(payload["context"]),
        stats=dict(payload.get("stats") or {}),
        trace_ref=payload.get("trace_ref"),
    )


def build_record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "item_id": record.item_id,
        "artifact": memory_artifact_to_dict(record.artifact),
        "stats": to_jsonable(record.stats),
    }


def build_record_from_dict(payload: dict[str, Any], record_type: type[T]) -> T:
    return record_type(
        item_id=str(payload["item_id"]),
        artifact=memory_artifact_from_dict(payload["artifact"]),
        stats=dict(payload.get("stats") or {}),
    )


def answer_record_to_dict(record: AnswerRecord) -> dict[str, Any]:
    return {
        "item_id": record.item_id,
        "question_id": record.question_id,
        "query": to_jsonable(record.query),
        "answer": record.answer,
        "agent_name": record.agent_name,
        "query_time": record.query_time,
        "memory_artifact_id": record.memory_artifact_id,
        "memory_reads": [
            memory_read_result_to_dict(read)
            for read in record.memory_reads
        ],
        "stats": to_jsonable(record.stats),
        "metadata": to_jsonable(record.metadata),
    }


def answer_record_from_dict(payload: dict[str, Any]) -> AnswerRecord:
    return AnswerRecord(
        item_id=str(payload["item_id"]),
        question_id=str(payload["question_id"]),
        query=payload["query"],
        answer=str(payload["answer"]),
        agent_name=str(payload["agent_name"]),
        query_time=payload.get("query_time"),
        memory_artifact_id=payload.get("memory_artifact_id"),
        memory_reads=tuple(
            memory_read_result_from_dict(read)
            for read in payload.get("memory_reads") or ()
        ),
        stats=dict(payload.get("stats") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )


def evaluation_record_to_dict(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "item_id": record.item_id,
        "question_id": record.question_id,
        "evaluator_name": record.evaluator_name,
        "score": record.score,
        "passed": record.passed,
        "reference": to_jsonable(record.reference),
        "metrics": to_jsonable(record.metrics),
        "metadata": to_jsonable(record.metadata),
    }


def evaluation_record_from_dict(payload: dict[str, Any]) -> EvaluationRecord:
    return EvaluationRecord(
        item_id=str(payload["item_id"]),
        question_id=str(payload["question_id"]),
        evaluator_name=str(payload["evaluator_name"]),
        score=payload.get("score"),
        passed=payload.get("passed"),
        reference=payload.get("reference"),
        metrics=dict(payload.get("metrics") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )
