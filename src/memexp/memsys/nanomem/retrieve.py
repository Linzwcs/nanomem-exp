from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from memexp.core.contracts import MemoryUnit, RankedMemoryUnit
from memexp.memsys.nanomem.config import RetrieveConfig, resolved_embedding_model


EMBEDDING_CACHE_SCHEMA_VERSION = "nanomem.embedding_cache.v1"


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


class FileEmbeddingVectorCache:
    """Persistent text-hash keyed embedding cache; raw texts are never stored."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._lock = Lock()

    def load(
        self,
        *,
        identity: dict[str, Any],
        text: str,
    ) -> tuple[float, ...] | None:
        key = _embedding_cache_key(identity, text)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with self._lock:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("schema_version") != EMBEDDING_CACHE_SCHEMA_VERSION:
            return None
        if payload.get("key") != key:
            return None
        if payload.get("identity") != identity:
            return None
        if payload.get("text_hash") != _text_hash(text):
            return None
        vector = payload.get("vector")
        if not isinstance(vector, list):
            return None
        try:
            return tuple(float(value) for value in vector)
        except (TypeError, ValueError):
            return None

    def store(
        self,
        *,
        identity: dict[str, Any],
        text: str,
        vector: tuple[float, ...],
    ) -> None:
        key = _embedding_cache_key(identity, text)
        path = self._path(key)
        payload = {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "key": key,
            "identity": identity,
            "text_hash": _text_hash(text),
            "vector": list(vector),
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"


class CachedDenseEmbeddingBackend:
    def __init__(
        self,
        *,
        backend: DenseEmbeddingBackend,
        cache: FileEmbeddingVectorCache,
        identity: dict[str, Any],
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.identity = identity
        self.name = backend.name
        self.last_stats: dict[str, Any] = {}

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...] | None] = [None] * len(texts)
        missing_indices_by_text: dict[str, list[int]] = {}
        cache_hits = 0
        batch_hits = 0

        for index, text in enumerate(texts):
            cached = self.cache.load(identity=self.identity, text=text)
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
            for text, vector in zip(missing_texts, embedded, strict=True):
                self.cache.store(identity=self.identity, text=text, vector=vector)
                writes += 1
                for index in missing_indices_by_text[text]:
                    vectors[index] = vector

        self.last_stats = {
            "enabled": True,
            "policy": EMBEDDING_CACHE_SCHEMA_VERSION,
            "namespace": self.identity["namespace"],
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
        cache=FileEmbeddingVectorCache(config.embedding_cache_path),
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


def _embedding_cache_key(identity: dict[str, Any], text: str) -> str:
    payload = {
        "identity": identity,
        "text_hash": _text_hash(text),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        self.backend.embed(texts)
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
    ) -> tuple[dict[str, Any], tuple[RankedMemoryUnit, ...]]:
        query_payload = build_query(query)
        query_text = str(query_payload["text"])
        limit = top_k or self.config.top_k
        texts = self.storage_texts(units)
        embeddings = self.backend.embed([query_text, *texts])
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
