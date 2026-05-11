from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from memexp.adapters.unified import SCHEMA_VERSION, string_list, text, text_or_none


def longmemeval_records_to_unified(
    records: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    source_file: str | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    source_record_count = 0

    for record in records:
        source_record_count += 1
        question_id = text(record.get("question_id"))
        if not question_id:
            raise ValueError("LongMemEval record is missing question_id")
        items.append({
            "item_id": question_id,
            "conversations": _conversations_from_record(record),
            "questions": [_question_from_record(record)],
            "metadata": {
                "source_question_id": question_id,
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "metadata": {
            "source_dataset": "longmemeval",
            "source_file": Path(source_file).name if source_file else None,
            "source_record_count": source_record_count,
            "item_count": len(items),
            "question_count": source_record_count,
        },
        "items": items,
    }


def _conversations_from_record(
        record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    sessions = record.get("haystack_sessions") or []
    session_ids = string_list(record.get("haystack_session_ids"))
    session_dates = string_list(record.get("haystack_dates"))
    conversations: list[list[dict[str, Any]]] = []

    for session_index, session in enumerate(sessions, start=1):
        session_id = _session_value(
            session_ids,
            session_index=session_index,
            prefix="session",
        )
        session_date = _session_value(
            session_dates,
            session_index=session_index,
            prefix="",
            default=None,
        )
        conversations.append([
            _message_from_turn(
                turn,
                session_id=session_id,
                session_index=session_index,
                turn_index=turn_index,
                timestamp=session_date,
            ) for turn_index, turn in enumerate(session or [], start=1)
        ])
    return conversations


def _message_from_turn(
    turn: dict[str, Any],
    *,
    session_id: str,
    session_index: int,
    turn_index: int,
    timestamp: str | None,
) -> dict[str, Any]:
    role = text(turn.get("role")) or "unknown"
    speaker = text(turn.get("speaker")) or role
    return {
        "message_id": f"{session_id}:m{turn_index}",
        "role": role,
        "speaker": speaker,
        "content": text(turn.get("content")),
        "timestamp": timestamp,
        "metadata": {
            "session_id": session_id,
            "session_index": session_index,
            "turn_index": turn_index,
        },
    }


def _question_from_record(record: dict[str, Any]) -> dict[str, Any]:
    answer_session_ids = string_list(record.get("answer_session_ids"))
    return {
        "question_id": text(record.get("question_id")),
        "query": text(record.get("question")),
        "query_time": text_or_none(record.get("question_date")),
        "label": {
            "reference_answer": record.get("answer"),
            "evidence_ids": answer_session_ids,
            "metadata": {
                "evidence_level": "conversation",
            },
        },
        "metadata": {
            "question_type": text_or_none(record.get("question_type")),
        },
    }


def _session_value(
    values: list[str],
    *,
    session_index: int,
    prefix: str,
    default: str | None = "",
) -> str | None:
    try:
        value = values[session_index - 1]
    except IndexError:
        value = ""
    if value:
        return value
    if default is None:
        return None
    return f"{prefix}_{session_index}" if prefix else default
