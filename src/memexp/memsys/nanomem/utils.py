from __future__ import annotations

import hashlib
import json
from typing import Any

from memexp.core.tokenization import count_tokens


def stable_hash(payload: Any, *, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def estimate_tokens(text: str) -> int:
    return count_tokens(text)


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("speaker") or "").strip().lower()


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("content") or message.get("text") or "").strip()


def message_timestamp(message: dict[str, Any]) -> str | None:
    value = message.get("timestamp") or message.get("time") or message.get("created_at")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def message_source_id(
    message: dict[str, Any],
    *,
    conversation_index: int,
    message_index: int,
) -> str:
    value = (
        message.get("__nanomem_source_id")
        or message.get("id")
        or message.get("message_id")
        or message.get("turn_id")
        or message.get("record_id")
    )
    if value is not None:
        return str(value)
    return f"c{conversation_index}:m{message_index}"
