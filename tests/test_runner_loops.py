from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from memexp import (
    ContainsEvaluator,
    Dataset,
    DatasetItem,
    DatasetQuestion,
    EvaluationRunner,
    ExperimentRunner,
    FixedQueryAgent,
    ListRunLogger,
    MemoryBuildRunner,
    NanoMemConfig,
    NanoMemSystem,
    RenderConfig,
    RetrieveConfig,
    RunExecutionConfig,
    StorageConfig,
    QuestionLabel,
)
from memexp.runs import AnswerRunner


class FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)


class RunnerLoopsTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("memexp.core.tokenization.tokenizer", return_value=FakeEncoding())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_build_answer_and_eval_are_independent_loops(self) -> None:
        dataset = Dataset(
            name="toy",
            split="dev",
            items=(
                DatasetItem(
                    item_id="item-1",
                    subject_id="user-1",
                    conversations=(
                        (
                            {
                                "id": "m1",
                                "role": "user",
                                "content": "I moved to Seattle in January.",
                                "timestamp": "2024-01-01",
                            },
                        ),
                        (
                            {
                                "id": "m2",
                                "role": "user",
                                "content": "I will visit Boston in March.",
                                "timestamp": "2024-03-01",
                            },
                        ),
                    ),
                    questions=(
                        DatasetQuestion(
                            question_id="q-seattle",
                            query="Where did the user move?",
                            query_time="2024-02-01",
                            label=QuestionLabel(reference_answer="Seattle"),
                        ),
                        DatasetQuestion(
                            question_id="q-boston-before",
                            query="What city will the user visit?",
                            query_time="2024-02-01",
                            label=QuestionLabel(reference_answer="Boston"),
                        ),
                    ),
                ),
            ),
        )
        memory_system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(backend="heuristic", target_roles=("user",)),
                retrieve=RetrieveConfig(top_k=5),
                render=RenderConfig(policy="timeline_v1", context_tokens=80),
            )
        )

        build = MemoryBuildRunner(memory_system).run(dataset)
        self.assertEqual(build.summary["artifact_count"], 1)
        self.assertEqual(build.summary["item_count"], 1)
        self.assertGreater(build.summary["total_units"], 0)

        answer = AnswerRunner(
            memory_system,
            FixedQueryAgent(),
            top_k=5,
            context_budget_tokens=80,
        ).run(dataset, build)
        seattle_answer = answer.record_for("item-1", "q-seattle")
        boston_answer = answer.record_for("item-1", "q-boston-before")

        self.assertEqual(answer.summary["answer_count"], 2)
        self.assertEqual(answer.summary["memory_read_count"], 2)
        self.assertEqual(seattle_answer.memory_artifact_id, build.records[0].artifact.artifact_id)
        self.assertIn("Seattle", seattle_answer.answer)
        self.assertNotIn("Boston", boston_answer.answer)
        self.assertGreater(boston_answer.memory_reads[0].stats["hidden_unit_count"], 0)

        evaluation = EvaluationRunner(ContainsEvaluator()).run(dataset, answer)
        self.assertTrue(evaluation.record_for("item-1", "q-seattle").passed)
        self.assertFalse(evaluation.record_for("item-1", "q-boston-before").passed)
        self.assertEqual(evaluation.summary["evaluated_count"], 2)
        self.assertEqual(evaluation.summary["passed_count"], 1)
        self.assertEqual(evaluation.summary["accuracy"], 0.5)

    def test_experiment_runner_composes_three_loops(self) -> None:
        dataset = Dataset(
            name="toy",
            items=(
                DatasetItem(
                    item_id="item-1",
                    conversations=(
                        (
                            {
                                "role": "user",
                                "content": "I like quiet cafes.",
                                "timestamp": "2024-01-01",
                            },
                        ),
                    ),
                    questions=(
                        DatasetQuestion(
                            question_id="q1",
                            query="What does the user like?",
                            query_time="2024-01-02",
                            label=QuestionLabel(reference_answer="quiet cafes"),
                        ),
                    ),
                ),
            ),
        )
        memory_system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(backend="heuristic", target_roles=("user",)),
                retrieve=RetrieveConfig(top_k=3),
                render=RenderConfig(policy="timeline_v1", context_tokens=40),
            )
        )

        result = ExperimentRunner(
            memory_system,
            FixedQueryAgent(),
            ContainsEvaluator(),
            top_k=3,
            context_budget_tokens=40,
        ).run(dataset)

        self.assertEqual(result.build.summary["artifact_count"], 1)
        self.assertEqual(result.answer.summary["answer_count"], 1)
        self.assertEqual(result.evaluation.summary["accuracy"], 1.0)
        self.assertEqual(result.summary["evaluation"]["passed_count"], 1)

    def test_experiment_runner_supports_parallel_execution_and_structured_logs(self) -> None:
        dataset = Dataset(
            name="toy",
            items=(
                DatasetItem(
                    item_id="item-1",
                    conversations=(
                        (
                            {
                                "role": "user",
                                "content": "I moved to Seattle.",
                                "timestamp": "2024-01-01",
                            },
                        ),
                    ),
                    questions=(
                        DatasetQuestion(
                            question_id="q1",
                            query="Where did the user move?",
                            query_time="2024-01-02",
                            label=QuestionLabel(reference_answer="Seattle"),
                        ),
                        DatasetQuestion(
                            question_id="q2",
                            query="What city is mentioned?",
                            query_time="2024-01-02",
                            label=QuestionLabel(reference_answer="Seattle"),
                        ),
                    ),
                ),
                DatasetItem(
                    item_id="item-2",
                    conversations=(
                        (
                            {
                                "role": "user",
                                "content": "I like quiet cafes.",
                                "timestamp": "2024-01-01",
                            },
                        ),
                    ),
                    questions=(
                        DatasetQuestion(
                            question_id="q3",
                            query="What does the user like?",
                            query_time="2024-01-02",
                            label=QuestionLabel(reference_answer="quiet cafes"),
                        ),
                    ),
                ),
            ),
        )
        memory_system = NanoMemSystem(
            NanoMemConfig(
                storage=StorageConfig(backend="heuristic", target_roles=("user",)),
                retrieve=RetrieveConfig(top_k=3),
                render=RenderConfig(policy="timeline_v1", context_tokens=40),
            )
        )
        logger = ListRunLogger()

        result = ExperimentRunner(
            memory_system,
            FixedQueryAgent(),
            ContainsEvaluator(),
            top_k=3,
            context_budget_tokens=40,
        ).run(
            dataset,
            execution=RunExecutionConfig(max_workers=2),
            logger=logger,
        )

        self.assertEqual(result.build.summary["artifact_count"], 2)
        self.assertEqual(result.answer.summary["answer_count"], 3)
        self.assertEqual(result.evaluation.summary["passed_count"], 3)
        self.assertEqual(result.build.records[0].item_id, "item-1")
        self.assertEqual(result.build.records[1].item_id, "item-2")
        self.assertEqual(
            [record.question_id for record in result.answer.records],
            ["q1", "q2", "q3"],
        )
        self.assertEqual(
            {
                event.stage
                for event in logger.events
                if event.event == "completed"
            },
            {"build", "answer", "evaluate"},
        )
        self.assertFalse(any(event.event == "failed" for event in logger.events))


if __name__ == "__main__":
    unittest.main()
