from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from memexp.core.contracts import MemoryScope, MemoryUnit
from memexp.core.time import max_timestamp, min_timestamp
from memexp.memsys.nanomem.config import NanoMemConfig, StorageConfig, config_for_artifact
from memexp.memsys.nanomem.utils import (
    estimate_tokens,
    message_role,
    message_speaker,
    message_source_id,
    message_text,
    message_timestamp,
    stable_hash,
)

FACT_EXTRACTION_SYSTEM = """
Extract structured facts from one memory item.
Return JSON only:
{
  "facts": [{"text": "...", "tags": ["...", "..."]}]
}

Rules:
- You will receive:
  - <target_memory_text>: owner-focused text.
  - <full_dialogue_context>: full chunk dialogue context.
  - <speaker_reference>: the exact speaker phrase to use when provided.
- Extract facts about the target speaker from <target_memory_text>.
- Use <full_dialogue_context> only to resolve references (pronouns, Q/A links, omitted objects, temporal anchor).
- Do not invent facts.
- Keep facts short, retrieval-oriented, and faithful to the memory text.
- Every fact must be a standalone sentence fragment with an explicit third-person subject.
- If <speaker_reference> is non-empty, start each fact with that exact phrase whenever possible, for example "user said" or "assistant said".
- Otherwise, use the target speaker's name as the subject whenever possible; do not omit the subject.
- Do not start facts with bare verbs or verbless fragments such as "Went to...", "Joined...", or "At the support group...".
- Preserve answer-critical details when present: time expressions, causality, outcomes, counts, and plans.
- Preserve relative time expressions exactly when they appear, such as "yesterday", "last week", "next month", "two years ago", or "recently".
- Ignore filler, greetings, and generic encouragement.
- Prefer self-contained facts over vague thematic summaries.
- Return all facts.
""".strip()


@dataclass(frozen=True)
class GenerationUsage:
    backend: str
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    call_id: str | None = None
    attempt_count: int = 0
    last_error_type: str | None = None
    last_error_message: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    facts: list[dict[str, Any]]
    backend: str
    backend_reason: str
    generation: GenerationUsage


class _LLMExtractionFailure(Exception):

    def __init__(self, *, reason: str, generation: GenerationUsage) -> None:
        super().__init__(reason)
        self.reason = reason
        self.generation = generation


def _render_message(message: dict[str, Any]) -> str:
    speaker = message_speaker(message) or message_role(message) or "unknown"
    return f"speaker: {speaker}\ncontent: {message_text(message)}"


def _chunk_conversation(
    conversation: list[dict[str, Any]],
    *,
    chunk_tokens: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in conversation:
        text = message_text(message)
        if not text:
            continue
        candidate = current + [message]
        if current and estimate_tokens("\n".join(
                _render_message(item) for item in candidate)) > chunk_tokens:
            chunks.append(current)
            current = [message]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _sentence_candidates(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=[;:])\s+", normalized)
    return [part.strip(" ,;") for part in parts if part.strip(" ,;")]


def _split_to_token_bounded_segments(text: str,
                                     *,
                                     max_tokens: int = 24) -> list[str]:
    trimmed = re.sub(r"\s+", " ", text).strip(" ,;")
    if not trimmed:
        return []
    if estimate_tokens(trimmed) <= max_tokens:
        return [trimmed]

    words = trimmed.split()
    segments: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and estimate_tokens(candidate) > max_tokens:
            segments.append(" ".join(current).strip(" ,;"))
            current = [word]
        else:
            current.append(word)
    if current:
        segments.append(" ".join(current).strip(" ,;"))
    return [segment for segment in segments if segment]


def _heuristic_facts(
    target_messages: list[dict[str, Any]],
    *,
    speaker_reference: str,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for message in target_messages:
        message_reference = _speaker_reference_for_message(
            message,
            fallback=speaker_reference,
        )
        for sentence in _sentence_candidates(message_text(message)):
            for segment in _split_to_token_bounded_segments(sentence):
                lowered = segment.lower()
                if message_reference and not lowered.startswith(
                        message_reference.lower()):
                    segment = f"{message_reference} {segment}"
                facts.append({"text": segment, "tags": []})
    return facts


def _speaker_reference_for_message(
    message: dict[str, Any],
    *,
    fallback: str,
) -> str:
    speaker = message_speaker(message)
    if speaker:
        return f"{speaker} said"
    return fallback


def _speaker_reference_for_messages(
    messages: list[dict[str, Any]],
    *,
    fallback_speaker: str,
) -> str:
    speakers = tuple(
        dict.fromkeys(speaker for message in messages
                      if (speaker := message_speaker(message))))
    if speakers:
        return "; ".join(f"{speaker} said" for speaker in speakers)
    return f"{fallback_speaker} said"


def _message_speaker_key(message: dict[str, Any]) -> str:
    return message_speaker(message) or message_role(message) or "unknown"


def _messages_by_speaker(
    messages: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        grouped.setdefault(_message_speaker_key(message), []).append(message)
    return grouped


def _extract_json_object(text: str) -> dict[str, Any] | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(value[start:end + 1])
        except Exception:
            return None
    return None


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _source_tokens(conversations: list[list[dict[str, Any]]]) -> int:
    return sum(
        estimate_tokens(message_text(message))
        for conversation in conversations for message in conversation
        if message_text(message))


def storage_token_stats(
    conversations: list[list[dict[str, Any]]],
    units: tuple[MemoryUnit, ...],
) -> dict[str, Any]:
    unit_token_counts = [
        int(unit.metadata.get("unit_token_count", estimate_tokens(unit.text)))
        for unit in units
    ]
    unit_token_total = sum(unit_token_counts)
    source_tokens = _source_tokens(conversations)

    generation_by_call_id: dict[str, dict[str, Any]] = {}
    anonymous_generation: list[dict[str, Any]] = []
    for unit in units:
        generation = unit.metadata.get("generation")
        if not isinstance(generation, dict):
            continue
        call_count = int(generation.get("call_count") or 0)
        if call_count <= 0:
            continue
        call_id = str(generation.get("call_id") or "")
        if call_id:
            generation_by_call_id.setdefault(call_id, generation)
        else:
            anonymous_generation.append(generation)

    generation_calls = [*generation_by_call_id.values(), *anonymous_generation]
    prompt_tokens = sum(
        int(item.get("prompt_tokens") or 0) for item in generation_calls)
    completion_tokens = sum(
        int(item.get("completion_tokens") or 0) for item in generation_calls)
    total_tokens = sum(
        int(item.get("total_tokens") or 0) for item in generation_calls)
    call_count = sum(
        int(item.get("call_count") or 0) for item in generation_calls)

    return {
        "source_tokens": source_tokens,
        "unit_tokens": {
            "total":
            unit_token_total,
            "avg": (unit_token_total /
                    len(unit_token_counts)) if unit_token_counts else 0.0,
            "max":
            max(unit_token_counts) if unit_token_counts else 0,
        },
        "generation_tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
            "call_count": call_count,
        },
        "compression": {
            "unit_to_source_ratio":
            (unit_token_total / source_tokens) if source_tokens else None,
            "generation_to_unit_ratio":
            (total_tokens / unit_token_total) if unit_token_total else None,
        },
    }


class StoragePolicy(Protocol):

    def build_units(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> tuple[MemoryUnit, ...]:
        ...


def make_storage_policy(config: StorageConfig) -> StoragePolicy:
    if config.policy == "fact":
        return FactStoragePolicy(config)
    raise ValueError(f"Unsupported NanoMem storage policy: {config.policy}")


class FactStoragePolicy(StoragePolicy):

    def __init__(self, config: StorageConfig) -> None:
        if config.policy != "fact":
            raise ValueError(
                f"Unsupported NanoMem storage policy: {config.policy}")
        self.config = config

    def build_units(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> tuple[MemoryUnit, ...]:
        units: list[MemoryUnit] = []
        for conversation_index, conversation in enumerate(conversations,
                                                          start=1):
            conversation = self._with_stable_source_ids(
                conversation,
                conversation_index=conversation_index,
            )
            chunks = _chunk_conversation(
                conversation,
                chunk_tokens=max(1, self.config.chunk_tokens),
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                units.extend(
                    self._build_chunk_units(
                        chunk,
                        scope=scope,
                        conversation_index=conversation_index,
                        chunk_index=chunk_index,
                    ))
        return tuple(units)

    def _with_stable_source_ids(
        self,
        conversation: list[dict[str, Any]],
        *,
        conversation_index: int,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message_index, message in enumerate(conversation, start=1):
            item = dict(message)
            item.setdefault("__nanomem_source_id",
                            f"c{conversation_index}:m{message_index}")
            normalized.append(item)
        return normalized

    def _build_chunk_units(
        self,
        chunk: list[dict[str, Any]],
        *,
        scope: MemoryScope,
        conversation_index: int,
        chunk_index: int,
    ) -> list[MemoryUnit]:
        units: list[MemoryUnit] = []
        source_timestamps = tuple(
            message_timestamp(message) for message in chunk)
        source_time_start = min_timestamp(source_timestamps)
        source_time_end = max_timestamp(source_timestamps)
        for speaker, target_messages in _messages_by_speaker(chunk).items():
            if not target_messages:
                continue
            target_speakers = tuple(
                dict.fromkeys(_message_speaker_key(message)
                              for message in target_messages))
            available_at = source_time_end or message_timestamp(
                target_messages[-1])
            speaker_reference = _speaker_reference_for_messages(
                target_messages,
                fallback_speaker=speaker,
            )
            speaker_id = stable_hash(speaker, length=8)
            generation_call_id = (
                f"{scope.scope_id}:c{conversation_index}:chunk_{chunk_index}:speaker_{speaker_id}"
            )
            extraction = self._extract_facts(
                chunk=chunk,
                target_messages=target_messages,
                speaker_reference=speaker_reference,
                generation_call_id=generation_call_id,
            )
            generation = asdict(extraction.generation)
            for fact_index, fact in enumerate(extraction.facts, start=1):
                fact_text = str(fact.get("text", "")).strip()
                if not fact_text:
                    continue
                unit_token_count = estimate_tokens(fact_text)
                tags = tuple(
                    str(tag).strip() for tag in fact.get("tags", [])
                    if str(tag).strip())
                unit_id = (f"{scope.scope_id}:c{conversation_index}:"
                           f"fact_{chunk_index}_{fact_index}:speaker_{speaker_id}")
                source_ids = tuple(
                    message_source_id(
                        message,
                        conversation_index=conversation_index,
                        message_index=index,
                    ) for index, message in enumerate(chunk, start=1))
                units.append(
                    MemoryUnit(
                        unit_id=unit_id,
                        text=fact_text,
                        timestamp=message_timestamp(target_messages[-1]),
                        available_at=available_at,
                        source_time_start=source_time_start,
                        source_time_end=source_time_end,
                        source_ids=source_ids,
                        memory_type="fact",
                        metadata={
                            "scope_id": scope.scope_id,
                            "conversation_index": conversation_index,
                            "chunk_index": chunk_index,
                            "target_speakers": target_speakers,
                            "available_at": available_at,
                            "source_time_range": {
                                "start": source_time_start,
                                "end": source_time_end,
                            },
                            "storage_backend": extraction.backend,
                            "storage_backend_reason":
                            extraction.backend_reason,
                            "unit_token_count": unit_token_count,
                            "generation": generation,
                            "tags": tags,
                            "structured": {
                                "facts": [{
                                    "text": fact_text,
                                    "tags": list(tags),
                                }]
                            },
                        },
                    ))
        return units

    def _extract_facts(
        self,
        *,
        chunk: list[dict[str, Any]],
        target_messages: list[dict[str, Any]],
        speaker_reference: str,
        generation_call_id: str,
    ) -> ExtractionResult:
        fallback = _heuristic_facts(
            target_messages,
            speaker_reference=speaker_reference,
        )
        if self.config.backend != "llm":
            return ExtractionResult(
                facts=fallback,
                backend="heuristic",
                backend_reason="llm_disabled",
                generation=GenerationUsage(backend="heuristic"),
            )
        result = self._llm_facts(
            chunk=chunk,
            target_messages=target_messages,
            speaker_reference=speaker_reference,
            generation_call_id=generation_call_id,
        )
        if result.facts:
            return result
        return ExtractionResult(
            facts=fallback,
            backend="heuristic",
            backend_reason=result.backend_reason,
            generation=result.generation,
        )

    def _llm_facts(
        self,
        *,
        chunk: list[dict[str, Any]],
        target_messages: list[dict[str, Any]],
        speaker_reference: str,
        generation_call_id: str,
    ) -> ExtractionResult:
        model = self.config.llm_model or os.getenv("OPENAI_MODEL")
        api_key = self.config.llm_api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.llm_base_url or os.getenv("OPENAI_BASE_URL")
        if not model or not api_key:
            return ExtractionResult(
                facts=[],
                backend="llm",
                backend_reason="missing_llm_config",
                generation=GenerationUsage(backend="llm", model=model),
            )
        try:
            from openai import OpenAI
        except Exception:
            return ExtractionResult(
                facts=[],
                backend="llm",
                backend_reason="missing_openai_package",
                generation=GenerationUsage(backend="llm", model=model),
            )

        target_text = "\n".join(
            _render_message(message) for message in target_messages)
        full_context = "\n".join(_render_message(message) for message in chunk)
        prompt = "\n\n".join([
            f"<speaker_reference>\n{speaker_reference}\n</speaker_reference>",
            f"<target_messages>\n{target_text}\n</target_messages>",
            f"<full_dialogue_context>\n{full_context}\n</full_dialogue_context>",
        ])
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        try:
            response, attempt_count, last_error = self._complete_with_retry(
                client=client,
                model=model,
                prompt=prompt,
                generation_call_id=generation_call_id,
            )
        except _LLMExtractionFailure as failure:
            return ExtractionResult(
                facts=[],
                backend="llm",
                backend_reason=failure.reason,
                generation=failure.generation,
            )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        generation = GenerationUsage(
            backend="llm",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            call_count=attempt_count,
            call_id=generation_call_id,
            attempt_count=attempt_count,
            last_error_type=type(last_error).__name__ if last_error else None,
            last_error_message=str(last_error) if last_error else None,
        )
        payload = _extract_json_object(content)
        facts_payload = (payload or {}).get("facts", []) or []
        facts: list[dict[str, Any]] = []
        for item in facts_payload:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                tags = [
                    str(tag).strip() for tag in item.get("tags", [])
                    if str(tag).strip()
                ]
            else:
                text = str(item).strip()
                tags = []
            if text:
                facts.append({"text": text, "tags": tags[:6]})
        return ExtractionResult(
            facts=facts,
            backend="llm",
            backend_reason="ok" if facts else "empty_llm_facts",
            generation=generation,
        )

    def _complete_with_retry(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        generation_call_id: str,
    ) -> tuple[Any, int, Exception | None]:
        retry = self.config.retry
        max_attempts = max(1, retry.max_attempts)
        delay = max(0.0, retry.initial_delay_seconds)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=self.config.llm_max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": FACT_EXTRACTION_SYSTEM
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                )
                return response, attempt, last_error
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(
                        exc,
                        retry.retryable_errors) or attempt >= max_attempts:
                    if self.config.fail_on_error:
                        raise
                    raise _LLMExtractionFailure(
                        reason="llm_failed_after_retries",
                        generation=GenerationUsage(
                            backend="llm",
                            model=model,
                            call_count=attempt,
                            call_id=generation_call_id,
                            attempt_count=attempt,
                            last_error_type=type(exc).__name__,
                            last_error_message=str(exc),
                        ),
                    )
                if delay > 0:
                    time.sleep(delay)
                    delay *= max(1.0, retry.backoff_multiplier)
        raise RuntimeError("unreachable retry state")


def _is_retryable_error(exc: Exception, retryable_errors: tuple[str,
                                                                ...]) -> bool:
    if not retryable_errors:
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(str(value).lower() in haystack for value in retryable_errors)


def artifact_id_for_units(
    *,
    system_name: str,
    scope: MemoryScope,
    config: NanoMemConfig,
    units: tuple[MemoryUnit, ...],
) -> str:
    payload = {
        "system_name": system_name,
        "scope": asdict(scope),
        "config": config_for_artifact(config),
        "units": [asdict(unit) for unit in units],
    }
    return f"{system_name}:{stable_hash(payload)}"
