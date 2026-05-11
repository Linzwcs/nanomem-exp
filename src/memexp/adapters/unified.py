from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from memexp.core.dataset import Dataset, DatasetItem, DatasetQuestion, QuestionLabel

SCHEMA_VERSION = "memexp.unified_dataset.v1"


def clean_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != [] and item != {}
    }


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def text_or_none(value: Any) -> str | None:
    normalized = text(value)
    return normalized or None


def stream_json_array(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    item_name: str = "JSON",
) -> Iterator[dict[str, Any]]:
    source = Path(path)
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    position = 0

    with source.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk and not buffer.strip():
                break
            buffer += chunk

            while True:
                position = _skip_ws(buffer, position)
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected JSON array in {source}")
                    started = True
                    position += 1
                    continue

                position = _skip_ws(buffer, position)
                if position >= len(buffer):
                    break
                if buffer[position] == ",":
                    position += 1
                    continue
                if buffer[position] == "]":
                    return

                try:
                    item, next_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                if not isinstance(item, dict):
                    raise ValueError(f"{item_name} array must contain objects")
                yield item
                position = next_position

            if position:
                buffer = buffer[position:]
                position = 0
            if not chunk and buffer.strip():
                raise ValueError(
                    f"Unexpected trailing JSON content in {source}")


def write_unified_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_unified_dataset(path: str | Path) -> Dataset:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported unified dataset schema: {payload.get('schema_version')}"
        )
    return unified_payload_to_dataset(payload)


def unified_payload_to_dataset(payload: dict[str, Any]) -> Dataset:
    return Dataset(
        name=text(payload.get("dataset_name")) or "unified",
        split=text_or_none(payload.get("split")),
        metadata=dict(payload.get("metadata") or {}),
        items=tuple(_item_from_payload(item) for item in payload.get("items") or ()),
    )


def export_summary(path: str | Path, payload: dict[str,
                                                   Any]) -> dict[str, Any]:
    return {
        "output": str(path),
        "dataset_name": payload["dataset_name"],
        "item_count": len(payload["items"]),
        "question_count": payload["metadata"]["question_count"],
    }


def _item_from_payload(payload: dict[str, Any]) -> DatasetItem:
    return DatasetItem(
        item_id=text(payload.get("item_id")),
        subject_id=text_or_none(payload.get("subject_id")),
        conversations=tuple(
            tuple(dict(message) for message in conversation)
            for conversation in payload.get("conversations") or ()
        ),
        questions=tuple(
            _question_from_payload(question)
            for question in payload.get("questions") or ()
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _question_from_payload(payload: dict[str, Any]) -> DatasetQuestion:
    label_payload = payload.get("label")
    return DatasetQuestion(
        question_id=text(payload.get("question_id")),
        query=payload.get("query") or "",
        query_time=text_or_none(payload.get("query_time")),
        label=(
            _label_from_payload(label_payload)
            if isinstance(label_payload, dict)
            else None
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _label_from_payload(payload: dict[str, Any]) -> QuestionLabel:
    return QuestionLabel(
        reference_answer=payload.get("reference_answer"),
        evidence_ids=tuple(string_list(payload.get("evidence_ids"))),
        metadata=dict(payload.get("metadata") or {}),
    )


def _skip_ws(value: str, position: int) -> int:
    while position < len(value) and value[position].isspace():
        position += 1
    return position
