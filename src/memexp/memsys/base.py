from __future__ import annotations

from typing import Any, Protocol

from memexp.core.contracts import (
    MemoryArtifact,
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
)


class MemoryRuntime(Protocol):
    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        ...


class MemorySystem(Protocol):
    name: str

    def build(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> MemoryArtifact:
        ...

    def load(self, artifact: MemoryArtifact) -> MemoryRuntime:
        ...
