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
from memexp.memsys.nanomem.context_cache import (
    CONTEXT_CACHE_SCHEMA_VERSION,
    SqliteContextShardCache,
    context_cache_key,
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
        artifact = MemoryArtifact(
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
        return artifact

    def index_artifact(self, artifact: MemoryArtifact) -> dict[str, Any]:
        if artifact.system_name != self.name:
            raise ValueError(
                f"NanoMemSystem cannot index artifact from system={artifact.system_name}"
            )
        return {
            "storage_embedding_cache": _warm_storage_embeddings_for_artifact(
                artifact=artifact,
                config=self.config,
            )
        }

    def prepare_build_artifact(self, artifact: MemoryArtifact) -> dict[str, Any]:
        return self.index_artifact(artifact)

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
        self.context_cache = self._context_cache()
        self.storage_embedding_cache = self._warm_storage_embeddings()

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        cache_key = self._read_cache_key(request)
        if self.context_cache is not None:
            cached = self.context_cache.load(
                shard_id=self.artifact.artifact_id,
                key=cache_key,
            )
            if cached is not None:
                return MemoryReadResult(
                    request=cached.request,
                    ranked_units=cached.ranked_units,
                    context=cached.context,
                    stats={
                        **cached.stats,
                        "context_cache": self._context_cache_stats(
                            hit=True,
                            key=cache_key,
                        ),
                    },
                    trace_ref=cached.trace_ref,
                )

        snapshot_units = self._snapshot_units(request.query_time)
        query, ranked_units = self.retrieve_policy.retrieve(
            snapshot_units,
            request.query,
            top_k=request.top_k,
            cache_shard_id=self.artifact.artifact_id,
        )
        retrieve_stats = self.retrieve_policy.last_stats
        context = self.render_policy.render(
            ranked_units,
            budget_tokens=request.context_budget_tokens,
        )
        result = MemoryReadResult(
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
                "context_cache": self._context_cache_stats(
                    hit=False,
                    key=cache_key,
                ),
            },
        )
        if self.context_cache is not None:
            self.context_cache.store(
                shard_id=self.artifact.artifact_id,
                key=cache_key,
                result=result,
            )
        return result

    def _context_cache(self) -> SqliteContextShardCache | None:
        if not self.config.retrieve.context_cache_path:
            return None
        config = config_for_cache(self.config)
        identity = {
            "schema_version": CONTEXT_CACHE_SCHEMA_VERSION,
            "namespace": self.config.retrieve.context_cache_namespace,
            "system_name": self.artifact.system_name,
            "retrieve": config["retrieve"],
            "render": config["render"],
        }
        return SqliteContextShardCache(
            self.config.retrieve.context_cache_path,
            identity=identity,
        )

    def _read_cache_key(self, request: MemoryReadRequest) -> str:
        return context_cache_key({
            "artifact_id": self.artifact.artifact_id,
            "request": {
                "query": request.query,
                "query_id": request.query_id,
                "query_time": request.query_time,
                "top_k": request.top_k,
                "context_budget_tokens": request.context_budget_tokens,
                "metadata": request.metadata,
            },
            "effective": {
                "top_k": request.top_k or self.config.retrieve.top_k,
                "context_budget_tokens": (
                    request.context_budget_tokens
                    or self.config.render.context_tokens
                ),
            },
        })

    def _context_cache_stats(self, *, hit: bool, key: str) -> dict[str, Any]:
        if self.context_cache is None:
            return {
                "enabled": False,
                "hit": False,
            }
        return {
            "enabled": True,
            "policy": CONTEXT_CACHE_SCHEMA_VERSION,
            "namespace": self.config.retrieve.context_cache_namespace,
            "hit": hit,
            "cache_key": key,
            "shard_id": self.artifact.artifact_id,
            "cache_file": str(
                self.context_cache.path_for(shard_id=self.artifact.artifact_id)
            ),
            "file_format": "sqlite3",
        }

    def _warm_storage_embeddings(self) -> dict[str, Any]:
        return _warm_storage_embeddings_for_artifact(
            artifact=self.artifact,
            config=self.config,
            retrieve_policy=self.retrieve_policy,
        )

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


def _warm_storage_embeddings_for_artifact(
    *,
    artifact: MemoryArtifact,
    config: NanoMemConfig,
    retrieve_policy: RetrievePolicy | None = None,
) -> dict[str, Any]:
    if not config.retrieve.warm_storage_embeddings:
        return {
            "enabled": False,
            "reason": "disabled",
            "text_count": len(artifact.units),
        }
    policy = retrieve_policy or RetrievePolicy(config.retrieve)
    return policy.warm_storage_embeddings(
        artifact.units,
        cache_shard_id=artifact.artifact_id,
    )
