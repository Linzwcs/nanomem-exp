from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from threading import Lock
from typing import Any, Generic, TypeVar

from memexp.runs.cache import to_jsonable


T = TypeVar("T")


class JsonlRecordSink(Generic[T]):
    """Thread-safe immediate JSONL writer for long-running run stages."""

    def __init__(
        self,
        path: str | Path,
        serializer: Callable[[T], dict[str, Any]],
        *,
        reset: bool = True,
    ) -> None:
        self.path = Path(path)
        self.serializer = serializer
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def __call__(self, record: T) -> None:
        self.write(record)

    def write(self, record: T) -> None:
        payload = to_jsonable(self.serializer(record))
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
