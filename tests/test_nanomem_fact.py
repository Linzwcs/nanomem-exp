from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memexp import (
    MemoryReadRequest,
    MemoryScope,
    MemoryUnit,
    NanoMemConfig,
    RankedMemoryUnit,
    RenderConfig,
    NanoMemSystem,
    RetrieveConfig,
    RetryConfig,
    StorageConfig,
)
from memexp.memsys.nanomem.render import RenderPolicy
from memexp.memsys.nanomem.storage import FactStoragePolicy, make_storage_policy


class FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)


class NanoMemFactTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("memexp.core.tokenization.tokenizer", return_value=FakeEncoding())
        self.addCleanup(patcher.stop)
        patcher.start()

    def _ranked(
        self,
        unit_id: str,
        text: str,
        timestamp: str | None,
        rank: int,
        score: float = 1.0,
    ) -> RankedMemoryUnit:
        return RankedMemoryUnit(
            unit=MemoryUnit(
                unit_id=unit_id,
                text=text,
                timestamp=timestamp,
            ),
            rank=rank,
            score=score,
            retrieval_text=text,
        )

    def test_storage_policy_factory_selects_fact_policy(self) -> None:
        policy = make_storage_policy(StorageConfig(policy="fact"))

        self.assertIsInstance(policy, FactStoragePolicy)
        with self.assertRaisesRegex(ValueError, "Unsupported NanoMem storage policy"):
            make_storage_policy(StorageConfig(policy="unknown_v1"))

    def test_builds_fact_units_from_all_speaker_messages(self) -> None:
        system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(
                    backend="heuristic",
                    chunk_tokens=64,
                ),
            )
        )
        artifact = system.build(
            [
                [
                    {
                        "role": "user",
                        "content": "I moved to Seattle last year. I like quiet cafes.",
                        "timestamp": "2024-01-01",
                    },
                    {
                        "role": "assistant",
                        "content": "Thanks for sharing.",
                        "timestamp": "2024-01-01",
                    },
                ]
            ],
            scope=MemoryScope(scope_id="sample-1", dataset="toy", subject_id="user"),
        )

        self.assertEqual(artifact.system_name, "nanomem")
        self.assertGreaterEqual(len(artifact.units), 2)
        self.assertTrue(all(unit.memory_type == "fact" for unit in artifact.units))
        self.assertTrue(any("Seattle" in unit.text for unit in artifact.units))
        self.assertTrue(any("Thanks for sharing" in unit.text for unit in artifact.units))
        self.assertTrue(all(unit.source_ids for unit in artifact.units))
        self.assertTrue(all(unit.available_at == "2024-01-01" for unit in artifact.units))
        self.assertTrue(all(unit.source_time_start == "2024-01-01" for unit in artifact.units))
        self.assertTrue(all(unit.source_time_end == "2024-01-01" for unit in artifact.units))
        self.assertTrue(all(unit.metadata["unit_token_count"] > 0 for unit in artifact.units))
        self.assertTrue(all(unit.metadata["generation"]["total_tokens"] == 0 for unit in artifact.units))
        self.assertEqual(
            artifact.metadata["storage_token_stats"]["generation_tokens"],
            {
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "call_count": 0,
            },
        )
        self.assertGreater(artifact.metadata["storage_token_stats"]["unit_tokens"]["total"], 0)

    def test_read_retrieves_and_renders_matching_facts(self) -> None:
        system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(
                    backend="heuristic",
                ),
                retrieve=RetrieveConfig(
                    policy="dense_cosine_v1",
                    top_k=5,
                ),
                render=RenderConfig(
                    policy="timeline_v1",
                    context_tokens=40,
                ),
            )
        )
        artifact = system.build(
            [
                [
                    {
                        "role": "user",
                        "content": "I moved to Seattle last year. I dislike noisy offices.",
                        "timestamp": "2024-01-01",
                    },
                    {
                        "role": "user",
                        "content": "I plan to visit Boston next month.",
                        "timestamp": "2024-02-01",
                    },
                ]
            ],
            scope=MemoryScope(scope_id="sample-2", dataset="toy", subject_id="user"),
        )

        result = system.load(artifact).read(MemoryReadRequest(query="Where did the user move to Seattle?"))

        self.assertGreater(result.ranked_units[0].score, 0)
        self.assertIn("moved", result.context.text)
        self.assertLessEqual(result.context.token_count, 40)
        self.assertEqual(result.stats["retrieve_policy"], "dense_cosine_v1")

    def test_embedding_vector_cache_reuses_persisted_vectors_without_raw_text(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.embeddings = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=_fake_embedding(text))
                        for text in kwargs["input"]
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            system = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="heuristic",
                    ),
                    retrieve=RetrieveConfig(
                        embedding_backend="openai_compatible",
                        embedding_model="embed-a",
                        embedding_api_key="test-key",
                        embedding_cache_path=tmp,
                    ),
                    render=RenderConfig(context_tokens=80),
                )
            )
            artifact = system.build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle. I like quiet cafes.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-embedding-cache", dataset="toy"),
            )

            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                first = system.load(artifact).read(
                    MemoryReadRequest(query="Where did the user move?")
                )
                second = system.load(artifact).read(
                    MemoryReadRequest(query="Where did the user move?")
                )

            text_count = 1 + len(artifact.units)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.stats["embedding_cache"]["misses"], text_count)
            self.assertEqual(second.stats["embedding_cache"]["hits"], text_count)
            self.assertEqual(second.stats["embedding_cache"]["misses"], 0)
            self.assertNotIn(tmp, repr(artifact.metadata))

            cache_files = list(Path(tmp).rglob("*.sqlite3"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(second.stats["embedding_cache"]["file_format"], "sqlite3")

            cache_payload = b"".join(path.read_bytes() for path in cache_files)
            self.assertNotIn(b"Seattle", cache_payload)
            self.assertNotIn(b"Where did", cache_payload)
            self.assertNotIn(b"test-key", cache_payload)

    def test_storage_embedding_warmup_caches_unit_vectors_before_read(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.embeddings = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=_fake_embedding(text))
                        for text in kwargs["input"]
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            system = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="heuristic",
                    ),
                    retrieve=RetrieveConfig(
                        embedding_backend="openai_compatible",
                        embedding_model="embed-a",
                        embedding_api_key="test-key",
                        embedding_cache_path=tmp,
                        warm_storage_embeddings=True,
                    ),
                    render=RenderConfig(context_tokens=80),
                )
            )
            artifact = system.build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle. I like quiet cafes.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-storage-embedding-cache", dataset="toy"),
            )

            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                runtime = system.load(artifact)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(calls[0]["input"]), len(artifact.units))

                result = runtime.read(
                    MemoryReadRequest(query="Where did the user move?")
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["input"], ["Where did the user move?"])
            self.assertEqual(
                result.stats["storage_embedding_cache"]["scope"],
                "storage",
            )
            self.assertEqual(
                result.stats["storage_embedding_cache"]["misses"],
                len(artifact.units),
            )
            self.assertEqual(
                result.stats["embedding_cache"]["hits"],
                len(artifact.units),
            )
            self.assertEqual(result.stats["embedding_cache"]["misses"], 1)

    def test_context_cache_reuses_rendered_read_result_per_artifact(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.embeddings = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=_fake_embedding(text))
                        for text in kwargs["input"]
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            system = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="heuristic",
                    ),
                    retrieve=RetrieveConfig(
                        embedding_backend="openai_compatible",
                        embedding_model="embed-a",
                        embedding_api_key="test-key",
                        context_cache_path=tmp,
                    ),
                    render=RenderConfig(context_tokens=80),
                )
            )
            artifact = system.build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle. I like quiet cafes.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-context-cache", dataset="toy"),
            )

            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                first = system.load(artifact).read(
                    MemoryReadRequest(query="Where did the user move?")
                )
                second = system.load(artifact).read(
                    MemoryReadRequest(query="Where did the user move?")
                )

            self.assertEqual(len(calls), 1)
            self.assertFalse(first.stats["context_cache"]["hit"])
            self.assertTrue(second.stats["context_cache"]["hit"])
            self.assertEqual(first.context.text, second.context.text)
            self.assertEqual(len(list(Path(tmp).rglob("*.sqlite3"))), 1)

    def test_embedding_vector_cache_is_model_scoped(self) -> None:
        calls: list[str] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.embeddings = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                calls.append(str(kwargs["model"]))
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=_fake_embedding(text))
                        for text in kwargs["input"]
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            config_a = NanoMemConfig(
                storage=StorageConfig(backend="heuristic"),
                retrieve=RetrieveConfig(
                    embedding_backend="openai_compatible",
                    embedding_model="embed-a",
                    embedding_api_key="test-key",
                    embedding_cache_path=tmp,
                ),
            )
            system_a = NanoMemSystem(config_a)
            artifact = system_a.build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-embedding-model-cache", dataset="toy"),
            )
            system_b = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(backend="heuristic"),
                    retrieve=RetrieveConfig(
                        embedding_backend="openai_compatible",
                        embedding_model="embed-b",
                        embedding_api_key="test-key",
                        embedding_cache_path=tmp,
                    ),
                )
            )

            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                system_a.load(artifact).read(MemoryReadRequest(query="Where?"))
                system_a.load(artifact).read(MemoryReadRequest(query="Where?"))
                system_b.load(artifact).read(MemoryReadRequest(query="Where?"))
            self.assertEqual(len(list(Path(tmp).rglob("*.sqlite3"))), 2)

        self.assertEqual(calls, ["embed-a", "embed-b"])

    def test_artifact_config_hash_uses_model_names_without_secrets(self) -> None:
        conversations = [
            [
                {
                    "role": "user",
                    "content": "I moved to Seattle last year.",
                    "timestamp": "2024-01-01",
                }
            ]
        ]
        scope = MemoryScope(scope_id="sample-secret", dataset="toy", subject_id="user")
        base_config = NanoMemConfig(
            storage=StorageConfig(
                backend="heuristic",
                llm_model="extractor-model",
                llm_base_url="https://first.example",
                llm_api_key="first-secret",
            ),
            retrieve=RetrieveConfig(
                embedding_model="embedding-model",
                embedding_base_url="https://first-embed.example",
                embedding_api_key="first-embed-secret",
            ),
        )
        changed_secret_config = NanoMemConfig(
            storage=StorageConfig(
                backend="heuristic",
                llm_model="extractor-model",
                llm_base_url="https://second.example",
                llm_api_key="second-secret",
            ),
            retrieve=RetrieveConfig(
                embedding_model="embedding-model",
                embedding_base_url="https://second-embed.example",
                embedding_api_key="second-embed-secret",
            ),
        )

        first = NanoMemSystem(base_config).build(conversations, scope=scope)
        second = NanoMemSystem(changed_secret_config).build(conversations, scope=scope)

        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertEqual(first.metadata["config"]["storage"]["llm_model"], "extractor-model")
        self.assertEqual(first.metadata["config"]["retrieve"]["embedding_model"], "embedding-model")
        rendered_metadata = repr(first.metadata)
        self.assertNotIn("secret", rendered_metadata)
        self.assertNotIn("example", rendered_metadata)

    def test_llm_storage_generation_tokens_are_recorded_once_per_call(self) -> None:
        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"facts":['
                                    '{"text":"user said they moved to Seattle","tags":["move"]},'
                                    '{"text":"user said they like quiet cafes","tags":["preference"]}'
                                    ']}'
                                )
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=11,
                        completion_tokens=7,
                        total_tokens=18,
                    ),
                )

        fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            artifact = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="llm",
                        llm_model="fake-extractor",
                        llm_api_key="test-key",
                    )
                )
            ).build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle. I like quiet cafes.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-llm", dataset="toy", subject_id="user"),
            )

        self.assertEqual(len(artifact.units), 2)
        self.assertTrue(all(unit.metadata["storage_backend"] == "llm" for unit in artifact.units))
        self.assertTrue(all(unit.metadata["generation"]["model"] == "fake-extractor" for unit in artifact.units))
        self.assertTrue(all(unit.metadata["generation"]["total_tokens"] == 18 for unit in artifact.units))
        self.assertEqual(
            artifact.metadata["storage_token_stats"]["generation_tokens"],
            {
                "prompt": 11,
                "completion": 7,
                "total": 18,
                "call_count": 1,
            },
        )

    def test_llm_storage_prompt_renders_speaker_field(self) -> None:
        prompts: list[str] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                prompts.append(kwargs["messages"][1]["content"])
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"facts":[{"text":"Ava moved to Seattle","tags":[]}]}'
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=5,
                        completion_tokens=3,
                        total_tokens=8,
                    ),
                )

        fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            artifact = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="llm",
                        llm_model="fake-extractor",
                        llm_api_key="test-key",
                    )
                )
            ).build(
                [
                    [
                        {
                            "role": "participant",
                            "speaker": "Ava",
                            "content": "I moved to Seattle.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-speaker-prompt", dataset="toy"),
            )

        self.assertEqual(len(prompts), 1)
        self.assertIn("<speaker_reference>\nAva said\n</speaker_reference>", prompts[0])
        self.assertIn("speaker: Ava", prompts[0])
        self.assertIn("content: I moved to Seattle.", prompts[0])
        self.assertNotIn("<speaker_reference>\nparticipant said", prompts[0])
        self.assertNotIn("participant: I moved to Seattle.", prompts[0])
        self.assertNotIn("target_role", artifact.units[0].metadata)
        self.assertEqual(artifact.units[0].metadata["target_speakers"], ("Ava",))

    def test_llm_storage_retries_transient_generation_failures(self) -> None:
        attempts: list[int] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise TimeoutError("timeout while calling storage backend")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"facts":[{"text":"user moved to Seattle","tags":[]}]}'
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=5,
                        completion_tokens=3,
                        total_tokens=8,
                    ),
                )

        fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            artifact = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="llm",
                        llm_model="fake-extractor",
                        llm_api_key="test-key",
                        retry=RetryConfig(
                            max_attempts=2,
                            initial_delay_seconds=0,
                        ),
                    )
                )
            ).build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-llm-retry", dataset="toy", subject_id="user"),
            )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(artifact.units), 1)
        generation = artifact.units[0].metadata["generation"]
        self.assertEqual(generation["call_count"], 2)
        self.assertEqual(generation["attempt_count"], 2)
        self.assertEqual(generation["last_error_type"], "TimeoutError")
        self.assertEqual(
            artifact.metadata["storage_token_stats"]["generation_tokens"],
            {
                "prompt": 5,
                "completion": 3,
                "total": 8,
                "call_count": 2,
            },
        )

    def test_llm_storage_falls_back_after_retry_exhaustion(self) -> None:
        attempts: list[int] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                attempts.append(1)
                raise TimeoutError("timeout while extracting facts")

        fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            artifact = NanoMemSystem(
                NanoMemConfig(
                    storage=StorageConfig(
                        backend="llm",
                        llm_model="fake-extractor",
                        llm_api_key="test-key",
                        retry=RetryConfig(
                            max_attempts=2,
                            initial_delay_seconds=0,
                        ),
                    )
                )
            ).build(
                [
                    [
                        {
                            "role": "user",
                            "content": "I moved to Seattle.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                scope=MemoryScope(scope_id="sample-llm-fallback", dataset="toy", subject_id="user"),
            )

        self.assertEqual(len(attempts), 2)
        self.assertGreaterEqual(len(artifact.units), 1)
        self.assertTrue(
            all(unit.metadata["storage_backend"] == "heuristic" for unit in artifact.units)
        )
        self.assertTrue(
            all(
                unit.metadata["storage_backend_reason"] == "llm_failed_after_retries"
                for unit in artifact.units
            )
        )
        self.assertTrue(
            all(unit.metadata["generation"]["call_count"] == 2 for unit in artifact.units)
        )

    def test_read_materializes_query_time_snapshot_from_full_timeline(self) -> None:
        system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(
                    backend="heuristic",
                ),
                retrieve=RetrieveConfig(top_k=10),
                render=RenderConfig(
                    policy="timeline_v1",
                    context_tokens=100,
                ),
            )
        )
        artifact = system.build(
            [
                [
                    {
                        "role": "user",
                        "content": "I moved to Seattle in January.",
                        "timestamp": "2024-01-01",
                    },
                ],
                [
                    {
                        "role": "user",
                        "content": "I will visit Boston in March.",
                        "timestamp": "2024-03-01",
                    },
                ],
            ],
            scope=MemoryScope(
                scope_id="sample-causal",
                dataset="toy",
                subject_id="user",
                timeline_id="user-timeline",
            ),
        )

        result = system.load(artifact).read(
            MemoryReadRequest(
                query="What city is Boston relevant to?",
                query_id="q-before-boston",
                query_time="2024-02-01",
            )
        )

        self.assertEqual(result.stats["query_id"], "q-before-boston")
        self.assertEqual(result.stats["query_time"], "2024-02-01")
        self.assertGreater(result.stats["hidden_unit_count"], 0)
        self.assertLess(result.stats["snapshot_unit_count"], result.stats["unit_count"])
        self.assertTrue(
            all(ranked_unit.unit.available_at <= "2024-02-01" for ranked_unit in result.ranked_units)
        )
        self.assertNotIn("Boston", result.context.text)

    def test_render_budget_limits_context_blocks(self) -> None:
        system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(
                    backend="heuristic",
                ),
                retrieve=RetrieveConfig(top_k=10),
                render=RenderConfig(
                    policy="timeline_v1",
                    context_tokens=12,
                    include_timestamps=False,
                ),
            )
        )
        artifact = system.build(
            [
                [
                    {"role": "user", "content": "I like hiking in rain forests.", "timestamp": "2024-01-01"},
                    {"role": "user", "content": "I like baking sourdough on Sundays.", "timestamp": "2024-01-02"},
                ]
            ],
            scope=MemoryScope(scope_id="sample-3"),
        )

        result = system.load(artifact).read(MemoryReadRequest(query="What does the user like?"))

        self.assertLessEqual(result.context.token_count, 12)
        self.assertEqual(result.context.block_count, 1)

    def test_adaptive_markdown_temporal_render_uses_heading_metadata_merge(self) -> None:
        policy = RenderPolicy(
            RenderConfig(
                policy="adaptive_markdown_temporal_v1",
                context_tokens=200,
                sort_by_time=True,
            )
        )
        context = policy.render(
            (
                self._ranked("r1", "User moved to Seattle.", "2024-01-03", 1, 0.9),
                self._ranked("r2", "User likes quiet cafes.", "2024-01-03", 2, 0.8),
                self._ranked("r3", "User started a new job.", "2024-01-20", 3, 0.7),
            )
        )

        self.assertIn("# 2024", context.text)
        self.assertIn("## 01", context.text)
        self.assertIn("### 03", context.text)
        self.assertIn("- User moved to Seattle.", context.text)
        self.assertIn("- User likes quiet cafes.", context.text)
        self.assertNotIn("User moved to Seattle.;", context.text)
        self.assertEqual(context.block_count, 3)
        self.assertEqual(context.timepoint_count, 2)

    def test_adaptive_markdown_single_unit_stays_flat(self) -> None:
        policy = RenderPolicy(
            RenderConfig(
                policy="adaptive_markdown_temporal_v1",
                context_tokens=100,
            )
        )
        context = policy.render(
            (
                self._ranked("r1", "User moved to Seattle.", "2024-01-03", 1, 0.9),
            )
        )

        self.assertEqual(context.text, "- 2024-01-03: User moved to Seattle.")
        self.assertNotIn("# 2024", context.text)
        self.assertEqual(context.block_count, 1)

    def test_adaptive_markdown_prunes_to_budget(self) -> None:
        policy = RenderPolicy(
            RenderConfig(
                policy="adaptive_markdown_temporal_v1",
                context_tokens=14,
                sort_by_time=True,
            )
        )
        context = policy.render(
            (
                self._ranked("r1", "User moved to Seattle.", "2024-01-03", 1, 0.9),
                self._ranked("r2", "User likes quiet cafes.", "2024-01-03", 2, 0.8),
                self._ranked("r3", "User started a new job.", "2024-01-20", 3, 0.7),
            )
        )

        self.assertLessEqual(context.token_count, 14)
        self.assertLess(context.block_count, 3)
        self.assertIn("User moved to Seattle.", context.text)


def _fake_embedding(text: str) -> list[float]:
    checksum = sum(ord(character) for character in text)
    return [
        float(checksum % 17 + 1),
        float(len(text) % 13 + 1),
        float((checksum // 17) % 11 + 1),
    ]


if __name__ == "__main__":
    unittest.main()
