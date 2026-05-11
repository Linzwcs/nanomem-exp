from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
from typing import Any, Protocol

from memexp.core.contracts import MemoryUnit, RankedMemoryUnit
from memexp.memsys.nanomem.config import RetrieveConfig, resolved_embedding_model


EMBEDDING_CACHE_SCHEMA_VERSION = "nanomem.embedding_cache.sqlite.v1"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_+\-']*", (text or "").lower())


def _query_text(query: str | dict[str, Any]) -> str:
    if isinstance(query, dict):
        return str(
            query.get("question") or query.get("query") or query.get("text")
            or "").strip()
    return str(query).strip()


def build_query(query: str | dict[str, Any]) -> dict[str, Any]:
    text = _query_text(query)
    payload = dict(query) if isinstance(query, dict) else {}
    return {
        **payload,
        "text": text,
    }


def _unit_text(unit: MemoryUnit, fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    metadata = unit.metadata or {}
    for field in fields or ("text", ):
        if field == "text":
            parts.append(unit.text)
        elif field == "timestamp":
            parts.append(str(unit.timestamp or ""))
        elif field == "tags":
            parts.extend(str(tag) for tag in metadata.get("tags", []) or [])
        elif field.startswith("structured"):
            structured = metadata.get("structured") or {}
            for fact in structured.get("facts", []) or []:
                parts.append(str(fact.get("text", "")))
                parts.extend(str(tag) for tag in fact.get("tags", []) or [])
        else:
            value = metadata.get(field)
            if isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            elif value is not None:
                parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        l * r
        for l, r in zip(left, right, strict=False)) / (left_norm * right_norm)


class DenseEmbeddingBackend(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        ...


class HashingDenseEmbeddingBackend:
    name = "hashing_dense_v1"
    dimensions = 256

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return tuple(vector)


class OpenAICompatibleEmbeddingBackend:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str,
    ) -> None:
        if not model:
            raise ValueError(
                "OpenAI-compatible dense retrieval requires an embedding model."
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.client: Any | None = None

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        response = self._client().embeddings.create(model=self.model, input=texts)
        return [
            tuple(float(value) for value in item.embedding)
            for item in response.data
        ]

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise ValueError(
                "OpenAI-compatible dense retrieval requires an API key.")
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "The openai package is required for this embedding backend."
            ) from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client


class SqliteEmbeddingShardCache:
    """Artifact-sharded embedding cache; raw texts are never stored."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def load_many(
        self,
        *,
        identity: dict[str, Any],
        shard_id: str,
        texts: list[str],
    ) -> list[tuple[float, ...] | None]:
        if not texts:
            return []
        path = self._path(identity=identity, shard_id=shard_id)
        if not path.exists():
            return [None] * len(texts)
        keys = [_text_hash(text) for text in texts]
        try:
            with sqlite3.connect(path, timeout=30) as connection:
                if not self._valid_meta(connection, identity=identity):
                    return [None] * len(texts)
                rows = connection.execute(
                    f"""
                    SELECT text_hash, dimensions, vector
                    FROM embeddings
                    WHERE text_hash IN ({",".join("?" for _ in keys)})
                    """,
                    keys,
                ).fetchall()
        except sqlite3.Error:
            return [None] * len(texts)

        vectors_by_key: dict[str, tuple[float, ...]] = {}
        for key, dimensions, blob in rows:
            vector = _vector_from_blob(blob, int(dimensions))
            if vector is not None:
                vectors_by_key[str(key)] = vector
        return [vectors_by_key.get(key) for key in keys]

    def store_many(
        self,
        *,
        identity: dict[str, Any],
        shard_id: str,
        items: list[tuple[str, tuple[float, ...]]],
    ) -> None:
        if not items:
            return
        path = self._path(identity=identity, shard_id=shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (_text_hash(text), len(vector), _vector_to_blob(vector))
            for text, vector in items
        ]
        try:
            with sqlite3.connect(path, timeout=30) as connection:
                self._ensure_schema(connection, identity=identity)
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (text_hash, dimensions, vector)
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )
                connection.commit()
        except sqlite3.Error:
            return

    def path_for(self, *, identity: dict[str, Any], shard_id: str) -> Path:
        return self._path(identity=identity, shard_id=shard_id)

    def _path(self, *, identity: dict[str, Any], shard_id: str) -> Path:
        key = _embedding_cache_shard_key(identity=identity, shard_id=shard_id)
        return self.root / key[:2] / f"{key}.sqlite3"

    def _ensure_schema(
        self,
        connection: sqlite3.Connection,
        *,
        identity: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        values = {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "identity_hash": _identity_hash(identity),
            "namespace": str(identity.get("namespace") or ""),
            "backend": str(identity.get("backend") or ""),
            "model": str(identity.get("model") or ""),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            values.items(),
        )

    def _valid_meta(
        self,
        connection: sqlite3.Connection,
        *,
        identity: dict[str, Any],
    ) -> bool:
        try:
            rows = connection.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.Error:
            return False
        meta = {str(key): str(value) for key, value in rows}
        return (
            meta.get("schema_version") == EMBEDDING_CACHE_SCHEMA_VERSION
            and meta.get("identity_hash") == _identity_hash(identity)
        )


class CachedDenseEmbeddingBackend:
    def __init__(
        self,
        *,
        backend: DenseEmbeddingBackend,
        cache: SqliteEmbeddingShardCache,
        identity: dict[str, Any],
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.identity = identity
        self.name = backend.name
        self.last_stats: dict[str, Any] = {}

    def embed(
        self,
        texts: list[str],
        *,
        shard_id: str = "global",
    ) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...] | None] = [None] * len(texts)
        missing_indices_by_text: dict[str, list[int]] = {}
        cache_hits = 0
        batch_hits = 0

        cached_vectors = self.cache.load_many(
            identity=self.identity,
            shard_id=shard_id,
            texts=texts,
        )
        for index, (text, cached) in enumerate(zip(texts, cached_vectors, strict=True)):
            if cached is not None:
                vectors[index] = cached
                cache_hits += 1
                continue
            if text in missing_indices_by_text:
                batch_hits += 1
            missing_indices_by_text.setdefault(text, []).append(index)

        missing_texts = list(missing_indices_by_text)
        writes = 0
        if missing_texts:
            embedded = self.backend.embed(missing_texts)
            self.cache.store_many(
                identity=self.identity,
                shard_id=shard_id,
                items=list(zip(missing_texts, embedded, strict=True)),
            )
            for text, vector in zip(missing_texts, embedded, strict=True):
                writes += 1
                for index in missing_indices_by_text[text]:
                    vectors[index] = vector

        self.last_stats = {
            "enabled": True,
            "policy": EMBEDDING_CACHE_SCHEMA_VERSION,
            "namespace": self.identity["namespace"],
            "shard_id": shard_id,
            "cache_file": str(self.cache.path_for(
                identity=self.identity,
                shard_id=shard_id,
            )),
            "file_format": "sqlite3",
            "hits": cache_hits,
            "misses": len(missing_texts),
            "batch_hits": batch_hits,
            "writes": writes,
            "text_count": len(texts),
        }
        result: list[tuple[float, ...]] = []
        for vector in vectors:
            if vector is None:
                raise RuntimeError("embedding cache returned an incomplete batch")
            result.append(vector)
        return result


def _embedding_backend(config: RetrieveConfig) -> DenseEmbeddingBackend:
    if config.embedding_backend == "hashing_dense_v1":
        backend: DenseEmbeddingBackend = HashingDenseEmbeddingBackend()
        return _maybe_cached_backend(config, backend)
    if config.embedding_backend == "openai_compatible":
        backend = OpenAICompatibleEmbeddingBackend(
            model=resolved_embedding_model(config) or "",
            base_url=_resolved_embedding_base_url(config),
            api_key=_resolved_embedding_api_key(config),
        )
        return _maybe_cached_backend(config, backend)
    raise ValueError(
        f"Unsupported NanoMem embedding backend: {config.embedding_backend}")


def _maybe_cached_backend(
    config: RetrieveConfig,
    backend: DenseEmbeddingBackend,
) -> DenseEmbeddingBackend:
    if not config.embedding_cache_path:
        return backend
    return CachedDenseEmbeddingBackend(
        backend=backend,
        cache=SqliteEmbeddingShardCache(config.embedding_cache_path),
        identity=_embedding_cache_identity(config, backend),
    )


def _resolved_embedding_base_url(config: RetrieveConfig) -> str | None:
    return (config.embedding_base_url or os.getenv("OPENAI_EMBED_BASE_URL")
            or os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL"))


def _resolved_embedding_api_key(config: RetrieveConfig) -> str:
    return (config.embedding_api_key or os.getenv("OPENAI_EMBED_API_KEY")
            or os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or "")


def _embedding_cache_identity(
    config: RetrieveConfig,
    backend: DenseEmbeddingBackend,
) -> dict[str, Any]:
    return {
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "namespace": config.embedding_cache_namespace,
        "backend": backend.name,
        "model": (
            getattr(backend, "model", None)
            or resolved_embedding_model(config)
        ),
        "dimensions": getattr(backend, "dimensions", None),
    }


def _embedding_cache_shard_key(
    *,
    identity: dict[str, Any],
    shard_id: str,
) -> str:
    payload = {
        "identity_hash": _identity_hash(identity),
        "shard_id": shard_id,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_to_blob(vector: tuple[float, ...]) -> bytes:
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}d", *vector)


def _vector_from_blob(blob: bytes, dimensions: int) -> tuple[float, ...] | None:
    if dimensions < 0 or len(blob) != dimensions * 8:
        return None
    if dimensions == 0:
        return ()
    try:
        return tuple(float(value) for value in struct.unpack(f"<{dimensions}d", blob))
    except struct.error:
        return None


class RetrievePolicy:

    def __init__(self, config: RetrieveConfig) -> None:
        if config.policy != "dense_cosine_v1":
            raise ValueError(
                f"Unsupported NanoMem retrieve policy: {config.policy}")
        self.config = config
        self.backend = _embedding_backend(config)
        self.last_stats: dict[str, Any] = {}

    def warm_storage_embeddings(
        self,
        units: tuple[MemoryUnit, ...],
        *,
        cache_shard_id: str = "global",
    ) -> dict[str, Any]:
        texts = self.storage_texts(units)
        if not texts:
            return {
                "enabled": bool(self.config.embedding_cache_path),
                "scope": "storage",
                "text_count": 0,
                "hits": 0,
                "misses": 0,
                "writes": 0,
            }
        if not self.config.embedding_cache_path:
            return {
                "enabled": False,
                "reason": "missing_embedding_cache_path",
                "text_count": len(texts),
            }
        self._embed(texts, cache_shard_id=cache_shard_id)
        return {
            **getattr(self.backend, "last_stats", {}),
            "scope": "storage",
        }

    def retrieve(
        self,
        units: tuple[MemoryUnit, ...],
        query: str | dict[str, Any],
        *,
        top_k: int | None = None,
        cache_shard_id: str = "global",
    ) -> tuple[dict[str, Any], tuple[RankedMemoryUnit, ...]]:
        query_payload = build_query(query)
        query_text = str(query_payload["text"])
        limit = top_k or self.config.top_k
        texts = self.storage_texts(units)
        embeddings = self._embed(
            [query_text, *texts],
            cache_shard_id=cache_shard_id,
        )
        self.last_stats = {
            "embedding_backend": self.backend.name,
            "embedding_cache": getattr(
                self.backend,
                "last_stats",
                {
                    "enabled": False,
                    "text_count": 1 + len(texts),
                },
            ),
        }
        query_embedding = embeddings[0]
        unit_embeddings = embeddings[1:]

        ranked: list[RankedMemoryUnit] = []
        for unit, text, embedding in zip(units,
                                         texts,
                                         unit_embeddings,
                                         strict=False):
            score = _cosine(query_embedding, embedding)
            ranked.append(
                RankedMemoryUnit(
                    unit=unit,
                    rank=0,
                    score=float(score),
                    retrieval_text=text,
                    score_breakdown={
                        "similarity": score,
                        "retrieve_policy": self.config.policy,
                        "embedding_backend": self.backend.name,
                        "retrieval_fields": self.config.retrieval_fields,
                    },
                ))
        ranked.sort(key=lambda item: (-item.score, item.unit.unit_id))
        return query_payload, tuple(
            RankedMemoryUnit(
                unit=item.unit,
                rank=index,
                score=item.score,
                retrieval_text=item.retrieval_text,
                score_breakdown=item.score_breakdown,
            ) for index, item in enumerate(ranked[:limit], start=1))

    def storage_texts(self, units: tuple[MemoryUnit, ...]) -> list[str]:
        return [
            _unit_text(unit, self.config.retrieval_fields) for unit in units
        ]

    def _embed(
        self,
        texts: list[str],
        *,
        cache_shard_id: str,
    ) -> list[tuple[float, ...]]:
        if isinstance(self.backend, CachedDenseEmbeddingBackend):
            return self.backend.embed(texts, shard_id=cache_shard_id)
        return self.backend.embed(texts)
