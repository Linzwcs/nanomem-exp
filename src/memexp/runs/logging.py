from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


@dataclass(frozen=True)
class RunEvent:
    stage: str
    event: str
    item_id: str | None = None
    question_id: str | None = None
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RunLogger(Protocol):
    def emit(self, event: RunEvent) -> None:
        ...


class NullRunLogger:
    def emit(self, event: RunEvent) -> None:
        return None


class CompositeRunLogger:
    def __init__(self, *loggers: RunLogger) -> None:
        self.loggers = tuple(loggers)

    def emit(self, event: RunEvent) -> None:
        for logger in self.loggers:
            logger.emit(event)


class ListRunLogger:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []
        self._lock = Lock()

    def emit(self, event: RunEvent) -> None:
        with self._lock:
            self.events.append(event)


class TerminalRunLogger:
    def emit(self, event: RunEvent) -> None:
        target = event.item_id or "-"
        if event.question_id:
            target = f"{target}/{event.question_id}"
        suffix = f" {event.message}" if event.message else ""
        metrics = f" {event.metrics}" if event.metrics else ""
        print(f"[{event.stage}] {event.event} {target}{suffix}{metrics}")


class JsonlRunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: RunEvent) -> None:
        payload = {
            "timestamp": event.timestamp,
            "stage": event.stage,
            "event": event.event,
            "item_id": event.item_id,
            "question_id": event.question_id,
            "message": event.message,
            "metrics": event.metrics,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
