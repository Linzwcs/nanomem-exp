from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    retryable_errors: tuple[str, ...] = (
        "timeout",
        "rate_limit",
        "ratelimit",
        "rate limit",
        "server_error",
        "server error",
        "connection",
        "temporarily",
        "429",
        "500",
        "502",
        "503",
        "504",
    )


@dataclass(frozen=True)
class StorageConfig:
    policy: str = "fact"
    chunk_tokens: int = 1024
    target_roles: tuple[str, ...] = ("user", )
    backend: str = "heuristic"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_max_tokens: int = 1024
    retry: RetryConfig = field(default_factory=RetryConfig)
    fail_on_error: bool = False


@dataclass(frozen=True)
class RetrieveConfig:
    """Read-side config. Dense cosine is the first implemented retrieve policy."""
    policy: str = "dense_cosine_v1"
    top_k: int = 50
    retrieval_fields: tuple[str, ...] = ("text", "tags")
    embedding_backend: str = "hashing_dense_v1"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_path: str | None = None
    embedding_cache_namespace: str = "default"
    warm_storage_embeddings: bool = False


@dataclass(frozen=True)
class RenderConfig:
    policy: str = "adaptive_markdown_temporal_v1"
    merge_policy: str = "temporal_metadata_merge_v1"
    context_tokens: int = 1000
    include_timestamps: bool = True
    merge_same_timestamp: bool = False
    min_group_size: int = 2
    sort_by_time: bool = False


@dataclass(frozen=True)
class NanoMemConfig:
    """Policy-grouped config for the NanoMem implementation."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    update_policy: str = "append_only_v1"
    metadata: dict[str, str] = field(default_factory=dict)


def config_for_artifact(config: NanoMemConfig) -> dict[str, Any]:
    """Return reproducibility config safe for artifact ids and metadata."""
    return {
        "storage": {
            "policy": config.storage.policy,
            "chunk_tokens": config.storage.chunk_tokens,
            "target_roles": config.storage.target_roles,
            "backend": config.storage.backend,
            "llm_model": config.storage.llm_model,
            "llm_max_tokens": config.storage.llm_max_tokens,
            "retry": {
                "max_attempts": config.storage.retry.max_attempts,
                "initial_delay_seconds":
                config.storage.retry.initial_delay_seconds,
                "backoff_multiplier": config.storage.retry.backoff_multiplier,
                "retryable_errors": config.storage.retry.retryable_errors,
            },
            "fail_on_error": config.storage.fail_on_error,
        },
        "retrieve": {
            "policy": config.retrieve.policy,
            "top_k": config.retrieve.top_k,
            "retrieval_fields": config.retrieve.retrieval_fields,
            "embedding_backend": config.retrieve.embedding_backend,
            "embedding_model": config.retrieve.embedding_model,
        },
        "render": {
            "policy": config.render.policy,
            "merge_policy": config.render.merge_policy,
            "context_tokens": config.render.context_tokens,
            "include_timestamps": config.render.include_timestamps,
            "merge_same_timestamp": config.render.merge_same_timestamp,
            "min_group_size": config.render.min_group_size,
            "sort_by_time": config.render.sort_by_time,
        },
        "update_policy": config.update_policy,
        "metadata": dict(config.metadata),
    }


def config_for_cache(config: NanoMemConfig) -> dict[str, Any]:
    """Return result-affecting config for runner cache keys without secrets."""
    payload = config_for_artifact(config)
    payload["storage"]["llm_model"] = resolved_llm_model(config.storage)
    payload["retrieve"]["embedding_model"] = resolved_embedding_model(
        config.retrieve)
    return payload


def resolved_llm_model(config: StorageConfig) -> str | None:
    if config.backend != "llm":
        return config.llm_model
    return config.llm_model or os.getenv("OPENAI_MODEL")


def resolved_embedding_model(config: RetrieveConfig) -> str | None:
    if config.embedding_backend != "openai_compatible":
        return config.embedding_model
    return (config.embedding_model or os.getenv("OPENAI_EMBED_MODEL")
            or os.getenv("EMBED_MODEL"))
