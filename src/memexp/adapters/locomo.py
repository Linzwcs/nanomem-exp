from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from memexp.adapters.unified import (
    SCHEMA_VERSION,
    clean_metadata,
    string_list,
    text,
    text_or_none,
)


def locomo_records_to_unified(
    records: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    source_file: str | None = None,
) -> dict[str, Any]:
    items_by_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    source_record_count = 0

    for record in records:
        source_record_count += 1
        item_id = text(record.get("source_sample_id")) or text(
            record.get("example_id"))
        if not item_id:
            raise ValueError(
                "Locomo record is missing source_sample_id/example_id")

        if item_id not in items_by_id:
            items_by_id[item_id] = {
                "item_id": item_id,
                "conversations": _conversations_from_record(record),
                "questions": [],
                "metadata": _item_metadata(record, item_id=item_id),
            }

        item = items_by_id[item_id]
        item["questions"].append(_question_from_record(record))

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "metadata": {
            "source_dataset": "locomo",
            "source_file": Path(source_file).name if source_file else None,
            "source_record_count": source_record_count,
            "item_count": len(items_by_id),
            "question_count": source_record_count,
        },
        "items": list(items_by_id.values()),
    }


def _conversations_from_record(
        record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    turn_records = record.get("turn_records") or []
    if turn_records:
        return _conversations_from_turn_records(turn_records)
    return _conversations_from_session_records(
        record.get("session_records") or [])


def _conversations_from_turn_records(
    turn_records: Iterable[dict[str, Any]], ) -> list[list[dict[str, Any]]]:
    conversations_by_id: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for turn in turn_records:
        session_id = text(turn.get("session_id")) or "unknown_session"
        conversations_by_id.setdefault(session_id,
                                       []).append(_message_from_turn(turn))
    return list(conversations_by_id.values())


def _conversations_from_session_records(
    session_records: Iterable[dict[str, Any]], ) -> list[list[dict[str, Any]]]:
    conversations: list[list[dict[str, Any]]] = []
    for index, session in enumerate(session_records, start=1):
        session_id = text(session.get("record_id")) or f"session_{index}"
        conversations.append([{
            "message_id":
            session_id,
            "role":
            "conversation",
            "content":
            text(session.get("text")),
            "timestamp":
            text_or_none(session.get("timestamp")),
            "metadata": {
                "session_id": session_id,
                "session_index": session.get("session_index", index),
            },
        }])
    return conversations


def _message_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    speaker = text_or_none(turn.get("speaker"))
    metadata: dict[str, Any] = {
        "speaker": speaker,
        "session_id": text_or_none(turn.get("session_id")),
        "turn_index": turn.get("turn_index"),
    }
    source_metadata = turn.get("metadata")
    if isinstance(source_metadata, dict):
        for key in ("session_key", "dia_id", "img_url"):
            value = source_metadata.get(key)
            if value is not None:
                metadata[key] = value

    message = {
        "message_id": text(turn.get("record_id")) or text(turn.get("id")),
        "role": "participant",
        "content": text(turn.get("text")),
        "timestamp": text_or_none(turn.get("timestamp")),
        "metadata": clean_metadata(metadata),
    }
    if speaker:
        message["speaker"] = speaker
    return message


def _question_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id":
        text(record.get("question_id")),
        "query":
        text(record.get("question")),
        "query_time":
        text_or_none(record.get("question_date")),
        "label":
        _label_from_record(record),
        "metadata":
        clean_metadata({
            "question_type":
            text_or_none(record.get("question_type")),
            "source_example_id":
            text_or_none(record.get("example_id")),
        }),
    }


def _label_from_record(record: dict[str, Any]) -> dict[str, Any]:
    gold_turn_ids = string_list(record.get("gold_turn_ids"))
    gold_session_ids = string_list(record.get("gold_session_ids"))
    if gold_turn_ids:
        evidence_ids = gold_turn_ids
        evidence_level = "message"
    else:
        evidence_ids = gold_session_ids
        evidence_level = "conversation" if gold_session_ids else None

    return {
        "reference_answer":
        record.get("answer"),
        "evidence_ids":
        evidence_ids,
        "metadata":
        clean_metadata({
            "gold_session_ids": gold_session_ids,
            "evidence_level": evidence_level,
        }),
    }


def _item_metadata(record: dict[str, Any], *, item_id: str) -> dict[str, Any]:
    source_metadata = record.get("metadata")
    dataset_file = None
    category = None
    if isinstance(source_metadata, dict):
        dataset_file = source_metadata.get("dataset_file")
        category = source_metadata.get("category")

    return clean_metadata({
        "source_sample_id":
        text_or_none(record.get("source_sample_id")) or item_id,
        "participants":
        string_list(record.get("participants")),
        "dataset_file":
        dataset_file,
        "category":
        category,
    })
