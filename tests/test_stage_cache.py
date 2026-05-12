from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memexp import (
    AnswerRecord,
    AnswerRunner,
    Dataset,
    DatasetItem,
    DatasetQuestion,
    EvaluationRecord,
    EvaluationRunner,
    JsonStageCache,
    ListRunLogger,
    MemoryArtifact,
    MemoryBuildRunner,
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
    MemoryUnit,
    NanoMemConfig,
    NanoMemSystem,
    PackedContext,
    QuestionLabel,
    RankedMemoryUnit,
    RetrieveConfig,
    object_cache_spec,
)


class CountingMemorySystem:
    name = "counting_memory"

    def __init__(self, version: str = "v1") -> None:
        self.config = {"version": version}
        self.build_count = 0
        self.load_count = 0

    def build(
        self,
        conversations: list[list[dict]],
        *,
        scope: MemoryScope,
    ) -> MemoryArtifact:
        self.build_count += 1
        return MemoryArtifact(
            artifact_id=f"artifact-{scope.scope_id}-{self.config['version']}",
            system_name=self.name,
            scope=scope,
            units=(
                MemoryUnit(
                    unit_id=f"unit-{scope.scope_id}",
                    text=f"cached fact {self.config['version']}",
                    timestamp="2024-01-01",
                    available_at="2024-01-01",
                    source_ids=("m1",),
                ),
            ),
            metadata={"config": self.config},
        )

    def load(self, artifact: MemoryArtifact) -> "CountingMemoryRuntime":
        self.load_count += 1
        return CountingMemoryRuntime(artifact)


class CountingMemoryRuntime:
    def __init__(self, artifact: MemoryArtifact) -> None:
        self.artifact = artifact

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        unit = self.artifact.units[0]
        return MemoryReadResult(
            request=request,
            ranked_units=(
                RankedMemoryUnit(
                    unit=unit,
                    rank=1,
                    score=1.0,
                    retrieval_text=unit.text,
                ),
            ),
            context=PackedContext(text=unit.text, token_count=3, block_count=1),
            stats={"artifact_id": self.artifact.artifact_id},
        )


class CountingAgent:
    name = "counting_agent"

    def __init__(self, version: str = "v1") -> None:
        self.config = {"version": version}
        self.answer_count = 0

    def answer(
        self,
        question: DatasetQuestion,
        memory_runtime: CountingMemoryRuntime,
        *,
        item_id: str,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> AnswerRecord:
        self.answer_count += 1
        read = memory_runtime.read(
            question.to_read_request(
                top_k=top_k,
                context_budget_tokens=context_budget_tokens,
            )
        )
        return AnswerRecord(
            item_id=item_id,
            question_id=question.question_id,
            query=question.query,
            answer=f"{self.config['version']}:{read.context.text}",
            agent_name=self.name,
            query_time=question.query_time,
            memory_artifact_id=read.stats["artifact_id"],
            memory_reads=(read,),
            stats={"version": self.config["version"]},
        )


class CountingEvaluator:
    name = "counting_eval"

    def __init__(self, version: str = "v1") -> None:
        self.config = {"version": version}
        self.evaluate_count = 0

    def evaluate(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        *,
        dataset: Dataset | None = None,
        item: DatasetItem | None = None,
    ) -> EvaluationRecord:
        self.evaluate_count += 1
        passed = self.config["version"] in answer.answer
        return EvaluationRecord(
            item_id=answer.item_id,
            question_id=answer.question_id,
            evaluator_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reference=question.label.reference_answer if question.label else None,
            metrics={"version": self.config["version"]},
        )


class StageCacheTest(unittest.TestCase):
    def test_build_cache_skips_rebuild_and_invalidates_on_config_change(self) -> None:
        dataset = toy_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonStageCache(tmp)
            memory = CountingMemorySystem()
            logger = ListRunLogger()
            runner = MemoryBuildRunner(memory)

            first = runner.run(dataset, cache=cache, logger=logger)
            second = runner.run(dataset, cache=cache, logger=logger)

            self.assertEqual(memory.build_count, 1)
            self.assertEqual(first.summary["cache_miss_count"], 1)
            self.assertEqual(second.summary["cache_hit_count"], 1)
            self.assertEqual(
                second.records[0].artifact.artifact_id,
                first.records[0].artifact.artifact_id,
            )
            self.assertTrue(
                any(event.stage == "build" and event.event == "cache_hit"
                    for event in logger.events)
            )

            changed_memory = CountingMemorySystem(version="v2")
            changed = MemoryBuildRunner(changed_memory).run(dataset, cache=cache)

            self.assertEqual(changed_memory.build_count, 1)
            self.assertEqual(changed.summary["cache_miss_count"], 1)

    def test_answer_cache_skips_load_and_agent_call(self) -> None:
        dataset = toy_dataset()
        memory = CountingMemorySystem()
        build = MemoryBuildRunner(memory).run(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonStageCache(tmp)
            agent = CountingAgent()
            runner = AnswerRunner(memory, agent, top_k=2, context_budget_tokens=20)

            first = runner.run(dataset, build, cache=cache)
            second = runner.run(dataset, build, cache=cache)

            self.assertEqual(memory.load_count, 1)
            self.assertEqual(agent.answer_count, 1)
            self.assertEqual(first.summary["cache_miss_count"], 1)
            self.assertEqual(second.summary["cache_hit_count"], 1)
            self.assertEqual(second.records[0].answer, first.records[0].answer)
            self.assertTrue((Path(tmp) / "answer.jsonl").exists())
            self.assertFalse((Path(tmp) / "answer").exists())

            changed_agent = CountingAgent(version="v2")
            changed = AnswerRunner(
                memory,
                changed_agent,
                top_k=2,
                context_budget_tokens=20,
            ).run(dataset, build, cache=cache)

            self.assertEqual(changed_agent.answer_count, 1)
            self.assertEqual(changed.summary["cache_miss_count"], 1)
            self.assertEqual(
                len((Path(tmp) / "answer.jsonl").read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_evaluation_cache_skips_evaluator_call(self) -> None:
        dataset = toy_dataset()
        memory = CountingMemorySystem()
        build = MemoryBuildRunner(memory).run(dataset)
        answer = AnswerRunner(memory, CountingAgent()).run(dataset, build)

        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonStageCache(tmp)
            evaluator = CountingEvaluator()
            runner = EvaluationRunner(evaluator)

            first = runner.run(dataset, answer, cache=cache)
            second = runner.run(dataset, answer, cache=cache)

            self.assertEqual(evaluator.evaluate_count, 1)
            self.assertEqual(first.summary["cache_miss_count"], 1)
            self.assertEqual(second.summary["cache_hit_count"], 1)
            self.assertEqual(second.records[0].score, first.records[0].score)
            self.assertTrue((Path(tmp) / "evaluate.jsonl").exists())
            self.assertFalse((Path(tmp) / "evaluate").exists())

            changed_evaluator = CountingEvaluator(version="v2")
            changed = EvaluationRunner(changed_evaluator).run(
                dataset,
                answer,
                cache=cache,
            )

            self.assertEqual(changed_evaluator.evaluate_count, 1)
            self.assertEqual(changed.summary["cache_miss_count"], 1)
            self.assertEqual(
                len((Path(tmp) / "evaluate.jsonl").read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_object_cache_spec_drops_secrets_and_urls(self) -> None:
        class ConfiguredObject:
            name = "configured"
            config = {
                "llm_model": "model-a",
                "llm_api_key": "sk-test-value",
                "embedding_base_url": "https://example.invalid",
            }

        encoded = json.dumps(object_cache_spec(ConfiguredObject()), sort_keys=True)

        self.assertIn("model-a", encoded)
        self.assertNotIn("sk-test-value", encoded)
        self.assertNotIn("example.invalid", encoded)

    def test_nanomem_cache_spec_uses_resolved_embedding_model_only(self) -> None:
        config = NanoMemConfig(
            retrieve=RetrieveConfig(embedding_backend="openai_compatible")
        )
        base_env = {
            "OPENAI_EMBED_MODEL": "embed-model-a",
            "OPENAI_EMBED_API_KEY": "embed-secret",
            "OPENAI_EMBED_BASE_URL": "https://embed.example.invalid",
        }
        with patch.dict(os.environ, base_env, clear=False):
            first = object_cache_spec(NanoMemSystem(config))
        with patch.dict(
            os.environ,
            {**base_env, "OPENAI_EMBED_MODEL": "embed-model-b"},
            clear=False,
        ):
            second = object_cache_spec(NanoMemSystem(config))

        encoded = json.dumps(first, sort_keys=True)

        self.assertIn("embed-model-a", encoded)
        self.assertNotIn("embed-secret", encoded)
        self.assertNotIn("embed.example.invalid", encoded)
        self.assertNotEqual(first, second)

    def test_nanomem_cache_spec_ignores_embedding_cache_path(self) -> None:
        first = object_cache_spec(
            NanoMemSystem(
                NanoMemConfig(
                    retrieve=RetrieveConfig(
                        embedding_model="embed-a",
                        embedding_cache_path="/tmp/cache-a",
                        warm_storage_embeddings=False,
                    )
                )
            )
        )
        second = object_cache_spec(
            NanoMemSystem(
                NanoMemConfig(
                    retrieve=RetrieveConfig(
                        embedding_model="embed-a",
                        embedding_cache_path="/tmp/cache-b",
                        warm_storage_embeddings=True,
                    )
                )
            )
        )

        self.assertEqual(first, second)


def toy_dataset() -> Dataset:
    return Dataset(
        name="toy",
        split="dev",
        items=(
            DatasetItem(
                item_id="item-1",
                subject_id="subject-1",
                conversations=(
                    (
                        {
                            "id": "m1",
                            "role": "user",
                            "content": "I like cached facts.",
                            "timestamp": "2024-01-01",
                        },
                    ),
                ),
                questions=(
                    DatasetQuestion(
                        question_id="q1",
                        query="What does the user like?",
                        query_time="2024-01-02",
                        label=QuestionLabel(reference_answer="cached facts"),
                    ),
                ),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
