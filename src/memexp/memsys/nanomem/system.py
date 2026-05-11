from __future__ import annotations

from typing import Any

from memexp.core.contracts import (
    MemoryArtifact,
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
    MemoryUnit,
)
from memexp.core.time import timestamp_lte
from memexp.memsys.nanomem.config import (
    NanoMemConfig,
    config_for_artifact,
    config_for_cache,
)
from memexp.memsys.nanomem.render import RenderPolicy
from memexp.memsys.nanomem.retrieve import RetrievePolicy
from memexp.memsys.nanomem.storage import (
    StoragePolicy,
    artifact_id_for_units,
    make_storage_policy,
    storage_token_stats,
)


class NanoMemSystem:
    name = "nanomem"

    def __init__(self, config: NanoMemConfig | None = None) -> None:
        self.config = config or NanoMemConfig()
        self.storage_policy: StoragePolicy = make_storage_policy(
            self.config.storage)

    def build(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        scope: MemoryScope,
    ) -> MemoryArtifact:
        units = self.storage_policy.build_units(conversations, scope=scope)
        artifact_id = artifact_id_for_units(
            system_name=self.name,
            scope=scope,
            config=self.config,
            units=units,
        )
        return MemoryArtifact(
            artifact_id=artifact_id,
            system_name=self.name,
            scope=scope,
            units=units,
            metadata={
                "config": config_for_artifact(self.config),
                "unit_count": len(units),
                "storage_token_stats": storage_token_stats(conversations, units),
            },
        )

    def load(self, artifact: MemoryArtifact) -> "NanoMemRuntime":
        if artifact.system_name != self.name:
            raise ValueError(
                f"NanoMemSystem cannot load artifact from system={artifact.system_name}"
            )
        return NanoMemRuntime(artifact=artifact, config=self.config)

    def cache_spec(self) -> dict[str, Any]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "config": config_for_cache(self.config),
        }


class NanoMemRuntime:

    def __init__(self, *, artifact: MemoryArtifact,
                 config: NanoMemConfig) -> None:
        self.artifact = artifact
        self.config = config
        self.retrieve_policy = RetrievePolicy(config.retrieve)
        self.render_policy = RenderPolicy(config.render)
        self.storage_embedding_cache = self._warm_storage_embeddings()

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        snapshot_units = self._snapshot_units(request.query_time)
        query, ranked_units = self.retrieve_policy.retrieve(
            snapshot_units,
            request.query,
            top_k=request.top_k,
        )
        retrieve_stats = self.retrieve_policy.last_stats
        context = self.render_policy.render(
            ranked_units,
            budget_tokens=request.context_budget_tokens,
        )
        return MemoryReadResult(
            request=request,
            ranked_units=ranked_units,
            context=context,
            stats={
                "artifact_id": self.artifact.artifact_id,
                "query": query,
                "query_id": request.query_id,
                "query_time": request.query_time,
                "unit_count": len(self.artifact.units),
                "snapshot_unit_count": len(snapshot_units),
                "hidden_unit_count":
                len(self.artifact.units) - len(snapshot_units),
                "ranked_unit_count": len(ranked_units),
                "context_tokens": context.token_count,
                "context_blocks": context.block_count,
                "storage_policy": self.config.storage.policy,
                "retrieve_policy": self.config.retrieve.policy,
                "embedding_backend": retrieve_stats.get("embedding_backend"),
                "embedding_cache": retrieve_stats.get("embedding_cache"),
                "storage_embedding_cache": self.storage_embedding_cache,
                "render_policy": self.config.render.policy,
            },
        )

    def _warm_storage_embeddings(self) -> dict[str, Any]:
        if not self.config.retrieve.warm_storage_embeddings:
            return {
                "enabled": False,
                "reason": "disabled",
                "text_count": len(self.artifact.units),
            }
        return self.retrieve_policy.warm_storage_embeddings(self.artifact.units)

    def _snapshot_units(self,
                        query_time: str | None) -> tuple[MemoryUnit, ...]:
        if not query_time:
            return self.artifact.units
        return tuple(unit for unit in self.artifact.units
                     if _unit_available_at(unit, query_time))


def _unit_available_at(unit: MemoryUnit, query_time: str) -> bool:
    available_at = unit.available_at or unit.timestamp
    if not available_at:
        return False
    return timestamp_lte(available_at, query_time)
