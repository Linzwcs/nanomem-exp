from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from memexp.core.dataset import Dataset


CACHE_SCHEMA_VERSION = "memexp.stage_cache.v1"


class StageCache(Protocol):
    def load(self, stage: str, key: str) -> dict[str, Any] | None:
        ...

    def store(
        self,
        stage: str,
        key: str,
        value: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ...


class JsonStageCache:
    """Small file-backed stage cache for runner records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = Lock()

    def load(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        with self._lock:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if payload.get("stage") != stage or payload.get("key") != key:
            return None
        value = payload.get("value")
        return value if isinstance(value, dict) else None

    def store(
        self,
        stage: str,
        key: str,
        value: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = self._path(stage, key)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "key": key,
            "metadata": to_jsonable(metadata or {}),
            "value": to_jsonable(value),
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / f"{key}.json"


def cache_key(stage: str, payload: Any) -> str:
    normalized = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "payload": to_jsonable(payload),
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def object_cache_spec(obj: Any) -> dict[str, Any]:
    cache_spec = getattr(obj, "cache_spec", None)
    if callable(cache_spec):
        spec = cache_spec()
        if not isinstance(spec, dict):
            raise TypeError("cache_spec() must return a dict")
        return redact_sensitive(to_jsonable(spec))

    spec: dict[str, Any] = {
        "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
    }
    name = getattr(obj, "name", None)
    if name is not None:
        spec["name"] = name
    config = getattr(obj, "config", None)
    if config is not None:
        spec["config"] = redact_sensitive(to_jsonable(config))
    return spec


def dataset_cache_spec(dataset: Dataset) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "split": dataset.split,
        "metadata": to_jsonable(dataset.metadata),
    }


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if _is_sensitive_key(key_text):
                continue
            redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(to_jsonable(item) for item in value)
    return value


def _is_sensitive_key(key: str) -> bool:
    return any(
        marker in key
        for marker in ("api_key", "base_url", "secret", "password", "token")
    )
