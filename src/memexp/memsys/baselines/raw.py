from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any

from memexp.core.contracts import (
    MemoryArtifact,
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
    MemoryUnit,
    PackedContext,
    RankedMemoryUnit,
)
from memexp.core.time import timestamp_lte
from memexp.core.tokenization import count_tokens


@dataclass(frozen=True)
class NullMemoryConfig:
    policy: str = "empty_v1"


class NullMemorySystem:
    name = "null_memory"

    def __init__(self, config: NullMemoryConfig | None = None) -> None:
        self.config = config or NullMemoryConfig()

    def build(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> MemoryArtifact:
        return MemoryArtifact(
            artifact_id=_artifact_id(self.name, scope, asdict(self.config), ()),
            system_name=self.name,
            scope=scope,
            units=(),
            metadata={
                "config": asdict(self.config),
                "unit_count": 0,
            },
        )

    def load(self, artifact: MemoryArtifact) -> "NullMemoryRuntime":
        _check_system(artifact, self.name)
        return NullMemoryRuntime(artifact)


class NullMemoryRuntime:
    def __init__(self, artifact: MemoryArtifact) -> None:
        self.artifact = artifact

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        return MemoryReadResult(
            request=request,
            ranked_units=(),
            context=PackedContext(text="", token_count=0, block_count=0),
            stats={
                "artifact_id": self.artifact.artifact_id,
                "unit_count": 0,
                "snapshot_unit_count": 0,
                "ranked_unit_count": 0,
                "context_tokens": 0,
                "context_blocks": 0,
                "baseline": "null_memory",
            },
        )


@dataclass(frozen=True)
class RawMessageConfig:
    policy: str = "raw_messages_v1"
    target_roles: tuple[str, ...] = ()
    top_k: int = 50
    context_tokens: int = 1000
    include_timestamps: bool = True


class RawMessageMemorySystem:
    name = "raw_messages"

    def __init__(self, config: RawMessageConfig | None = None) -> None:
        self.config = config or RawMessageConfig()

    def build(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> MemoryArtifact:
        units = _message_units(conversations, scope=scope, config=self.config)
        return MemoryArtifact(
            artifact_id=_artifact_id(self.name, scope, asdict(self.config), units),
            system_name=self.name,
            scope=scope,
            units=units,
            metadata={
                "config": asdict(self.config),
                "unit_count": len(units),
            },
        )

    def load(self, artifact: MemoryArtifact) -> "RawMessageRuntime":
        _check_system(artifact, self.name)
        return RawMessageRuntime(artifact=artifact, config=self.config)


class RawMessageRuntime:
    def __init__(self, *, artifact: MemoryArtifact,
                 config: RawMessageConfig) -> None:
        self.artifact = artifact
        self.config = config

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        units = self._snapshot_units(request.query_time)
        ranked_units = _rank_units(
            units,
            _query_text(request.query),
            top_k=request.top_k or self.config.top_k,
        )
        context = _render_context(
            ranked_units,
            budget_tokens=request.context_budget_tokens
            or self.config.context_tokens,
            include_timestamps=self.config.include_timestamps,
        )
        return MemoryReadResult(
            request=request,
            ranked_units=ranked_units,
            context=context,
            stats={
                "artifact_id": self.artifact.artifact_id,
                "unit_count": len(self.artifact.units),
                "snapshot_unit_count": len(units),
                "hidden_unit_count": len(self.artifact.units) - len(units),
                "ranked_unit_count": len(ranked_units),
                "context_tokens": context.token_count,
                "context_blocks": context.block_count,
                "baseline": "raw_messages",
                "retrieve_policy": "token_overlap_v1",
                "render_policy": "plain_lines_v1",
            },
        )

    def _snapshot_units(self,
                        query_time: str | None) -> tuple[MemoryUnit, ...]:
        if not query_time:
            return self.artifact.units
        return tuple(
            unit for unit in self.artifact.units
            if _unit_available_at(unit, query_time)
        )


def _message_units(
    conversations: list[list[dict[str, Any]]],
    *,
    scope: MemoryScope,
    config: RawMessageConfig,
) -> tuple[MemoryUnit, ...]:
    units: list[MemoryUnit] = []
    target_roles = {role.lower() for role in config.target_roles}
    for conversation_index, conversation in enumerate(conversations, start=1):
        for message_index, message in enumerate(conversation, start=1):
            role = _message_role(message)
            if target_roles and role not in target_roles:
                continue
            text = _message_text(message)
            if not text:
                continue
            source_id = _message_source_id(
                message,
                conversation_index=conversation_index,
                message_index=message_index,
            )
            timestamp = _message_timestamp(message)
            units.append(
                MemoryUnit(
                    unit_id=f"{scope.scope_id}:c{conversation_index}:m{message_index}",
                    text=text,
                    timestamp=timestamp,
                    available_at=timestamp,
                    source_time_start=timestamp,
                    source_time_end=timestamp,
                    source_ids=(source_id,),
                    memory_type="raw_message",
                    metadata={
                        "role": role,
                        "conversation_index": conversation_index,
                        "message_index": message_index,
                    },
                )
            )
    return tuple(units)


def _rank_units(
    units: tuple[MemoryUnit, ...],
    query: str,
    *,
    top_k: int,
) -> tuple[RankedMemoryUnit, ...]:
    query_tokens = set(_tokens(query))
    ranked: list[RankedMemoryUnit] = []
    for unit in units:
        unit_tokens = set(_tokens(unit.text))
        overlap = len(query_tokens & unit_tokens)
        denominator = math.sqrt(max(1, len(query_tokens)) * max(1, len(unit_tokens)))
        score = overlap / denominator
        ranked.append(
            RankedMemoryUnit(
                unit=unit,
                rank=0,
                score=float(score),
                retrieval_text=unit.text,
                score_breakdown={
                    "overlap": overlap,
                    "query_tokens": len(query_tokens),
                    "unit_tokens": len(unit_tokens),
                    "retrieve_policy": "token_overlap_v1",
                },
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.unit.unit_id))
    return tuple(
        RankedMemoryUnit(
            unit=item.unit,
            rank=index,
            score=item.score,
            retrieval_text=item.retrieval_text,
            score_breakdown=item.score_breakdown,
        )
        for index, item in enumerate(ranked[:max(0, top_k)], start=1)
    )


def _render_context(
    ranked_units: tuple[RankedMemoryUnit, ...],
    *,
    budget_tokens: int,
    include_timestamps: bool,
) -> PackedContext:
    lines: list[str] = []
    for ranked in ranked_units:
        line = _render_line(ranked.unit, include_timestamps=include_timestamps)
        candidate = "\n".join([*lines, line])
        if lines and count_tokens(candidate) > budget_tokens:
            break
        if count_tokens(line) > budget_tokens and not lines:
            break
        lines.append(line)
    text = "\n".join(lines)
    return PackedContext(
        text=text,
        token_count=count_tokens(text),
        block_count=len(lines),
    )


def _render_line(unit: MemoryUnit, *, include_timestamps: bool) -> str:
    if include_timestamps and unit.timestamp:
        return f"- {unit.timestamp}: {unit.text}"
    return f"- {unit.text}"


def _artifact_id(
    system_name: str,
    scope: MemoryScope,
    config: dict[str, Any],
    units: tuple[MemoryUnit, ...],
) -> str:
    payload = {
        'system_name': system_name,
        'scope': asdict(scope),
        'config': config,
        'units': [asdict(unit) for unit in units],
    }
    return f"{system_name}:{_stable_hash(payload)}"


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _check_system(artifact: MemoryArtifact, system_name: str) -> None:
    if artifact.system_name != system_name:
        raise ValueError(
            f"{system_name} cannot load artifact from system={artifact.system_name}"
        )


def _unit_available_at(unit: MemoryUnit, query_time: str) -> bool:
    available_at = unit.available_at or unit.timestamp
    return bool(available_at and timestamp_lte(available_at, query_time))


def _query_text(query: str | dict[str, Any]) -> str:
    if isinstance(query, dict):
        return str(
            query.get("question") or query.get("query") or query.get("text")
            or ""
        ).strip()
    return str(query).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_+\-']*", (text or "").lower())


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("speaker") or "").strip().lower()


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("content") or message.get("text") or "").strip()


def _message_timestamp(message: dict[str, Any]) -> str | None:
    value = message.get("timestamp") or message.get("time") or message.get("created_at")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _message_source_id(
    message: dict[str, Any],
    *,
    conversation_index: int,
    message_index: int,
) -> str:
    value = (
        message.get("message_id")
        or message.get("id")
        or message.get("turn_id")
        or message.get("record_id")
    )
    return str(value) if value is not None else f"c{conversation_index}:m{message_index}"
