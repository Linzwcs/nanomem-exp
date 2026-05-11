from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from memexp.core.contracts import (
    MemoryReadRequest,
    MemoryReadResult,
    MemoryUnit,
    PackedContext,
    RankedMemoryUnit,
)


CONTEXT_CACHE_SCHEMA_VERSION = "nanomem.context_cache.sqlite.v1"


class SqliteContextShardCache:
    """Artifact-sharded cache for rendered read contexts."""

    def __init__(self, root: str | Path, *, identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.identity = identity

    def load(
        self,
        *,
        shard_id: str,
        key: str,
    ) -> MemoryReadResult | None:
        path = self.path_for(shard_id=shard_id)
        if not path.exists():
            return None
        try:
            with sqlite3.connect(path, timeout=30) as connection:
                if not self._valid_meta(connection):
                    return None
                row = connection.execute(
                    "SELECT payload FROM contexts WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except (sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
            return _memory_read_result_from_dict(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def store(
        self,
        *,
        shard_id: str,
        key: str,
        result: MemoryReadResult,
    ) -> None:
        path = self.path_for(shard_id=shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            _jsonable(_memory_read_result_to_dict(result)),
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            with sqlite3.connect(path, timeout=30) as connection:
                self._ensure_schema(connection)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO contexts
                    (cache_key, payload)
                    VALUES (?, ?)
                    """,
                    (key, payload),
                )
                connection.commit()
        except sqlite3.Error:
            return

    def path_for(self, *, shard_id: str) -> Path:
        key = _cache_shard_key(identity=self.identity, shard_id=shard_id)
        return self.root / key[:2] / f"{key}.sqlite3"

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contexts (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        values = {
            "schema_version": CONTEXT_CACHE_SCHEMA_VERSION,
            "identity_hash": _identity_hash(self.identity),
            "namespace": str(self.identity.get("namespace") or ""),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            values.items(),
        )

    def _valid_meta(self, connection: sqlite3.Connection) -> bool:
        try:
            rows = connection.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.Error:
            return False
        meta = {str(key): str(value) for key, value in rows}
        return (
            meta.get("schema_version") == CONTEXT_CACHE_SCHEMA_VERSION
            and meta.get("identity_hash") == _identity_hash(self.identity)
        )


def context_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _cache_shard_key(*, identity: dict[str, Any], shard_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "identity_hash": _identity_hash(identity),
                "shard_id": shard_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(identity),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _memory_unit_to_dict(unit: MemoryUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "text": unit.text,
        "timestamp": unit.timestamp,
        "available_at": unit.available_at,
        "source_time_start": unit.source_time_start,
        "source_time_end": unit.source_time_end,
        "source_ids": list(unit.source_ids),
        "memory_type": unit.memory_type,
        "metadata": _jsonable(unit.metadata),
    }


def _memory_unit_from_dict(payload: dict[str, Any]) -> MemoryUnit:
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


def _memory_read_request_to_dict(request: MemoryReadRequest) -> dict[str, Any]:
    return {
        "query": _jsonable(request.query),
        "query_id": request.query_id,
        "query_time": request.query_time,
        "top_k": request.top_k,
        "context_budget_tokens": request.context_budget_tokens,
        "metadata": _jsonable(request.metadata),
    }


def _memory_read_request_from_dict(payload: dict[str, Any]) -> MemoryReadRequest:
    return MemoryReadRequest(
        query=payload["query"],
        query_id=payload.get("query_id"),
        query_time=payload.get("query_time"),
        top_k=payload.get("top_k"),
        context_budget_tokens=payload.get("context_budget_tokens"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _ranked_memory_unit_to_dict(ranked: RankedMemoryUnit) -> dict[str, Any]:
    return {
        "unit": _memory_unit_to_dict(ranked.unit),
        "rank": ranked.rank,
        "score": ranked.score,
        "retrieval_text": ranked.retrieval_text,
        "score_breakdown": _jsonable(ranked.score_breakdown),
    }


def _ranked_memory_unit_from_dict(payload: dict[str, Any]) -> RankedMemoryUnit:
    return RankedMemoryUnit(
        unit=_memory_unit_from_dict(payload["unit"]),
        rank=int(payload["rank"]),
        score=float(payload["score"]),
        retrieval_text=str(payload["retrieval_text"]),
        score_breakdown=dict(payload.get("score_breakdown") or {}),
    )


def _packed_context_to_dict(context: PackedContext) -> dict[str, Any]:
    return {
        "text": context.text,
        "token_count": context.token_count,
        "block_count": context.block_count,
        "timepoint_count": context.timepoint_count,
    }


def _packed_context_from_dict(payload: dict[str, Any]) -> PackedContext:
    return PackedContext(
        text=str(payload["text"]),
        token_count=int(payload["token_count"]),
        block_count=int(payload.get("block_count") or 0),
        timepoint_count=payload.get("timepoint_count"),
    )


def _memory_read_result_to_dict(read: MemoryReadResult) -> dict[str, Any]:
    return {
        "request": _memory_read_request_to_dict(read.request),
        "ranked_units": [
            _ranked_memory_unit_to_dict(ranked)
            for ranked in read.ranked_units
        ],
        "context": _packed_context_to_dict(read.context),
        "stats": _jsonable(read.stats),
        "trace_ref": read.trace_ref,
    }


def _memory_read_result_from_dict(payload: dict[str, Any]) -> MemoryReadResult:
    return MemoryReadResult(
        request=_memory_read_request_from_dict(payload["request"]),
        ranked_units=tuple(
            _ranked_memory_unit_from_dict(ranked)
            for ranked in payload.get("ranked_units") or ()
        ),
        context=_packed_context_from_dict(payload["context"]),
        stats=dict(payload.get("stats") or {}),
        trace_ref=payload.get("trace_ref"),
    )
