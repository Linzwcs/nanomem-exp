from __future__ import annotations

import unittest

from memexp import (
    AnswerRecord,
    Dataset,
    DatasetItem,
    DatasetPromptJudgeEvaluator,
    DatasetQuestion,
    EvaluationRunner,
    LOCOMO_ACCURACY_PROMPT,
    MBENCH_PROMPT_NAME,
    QuestionLabel,
    longmemeval_prompt,
    mbench_judge_prompt,
)


class FakeJudgeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class DatasetPromptJudgeEvaluatorTest(unittest.TestCase):
    def test_locomo_uses_exact_locomo_prompt_and_json_label_parser(self) -> None:
        backend = FakeJudgeBackend('{"label":"CORRECT"}')
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="locomo10_nonempty_answers",
            items=(
                DatasetItem(
                    item_id="conv-1",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="conv-1:q1",
                            query="When did Ava move?",
                            label=QuestionLabel(reference_answer="7 May 2023"),
                            metadata={"question_type": "2"},
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="conv-1",
            question_id="conv-1:q1",
            query="When did Ava move?",
            answer="Ava moved on May 7th.",
            agent_name="test-agent",
            metadata={
                "reasoning": (
                    "## STEP 1: RELEVANT MEMORIES EXTRACTION\n"
                    "- Ava moved on May 7th.\n\n"
                    "## FINAL ANSWER:\nAva moved on May 7th."
                )
            },
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        generated_answer = (
            "## STEP 1: RELEVANT MEMORIES EXTRACTION\n"
            "- Ava moved on May 7th.\n\n"
            "## FINAL ANSWER:\nAva moved on May 7th."
        )
        expected_prompt = LOCOMO_ACCURACY_PROMPT.format(
            question="When did Ava move?",
            gold_answer="7 May 2023",
            generated_answer=generated_answer,
        )
        self.assertEqual(backend.prompts, [expected_prompt])
        record = result.record_for("conv-1", "conv-1:q1")
        self.assertTrue(record.passed)
        self.assertEqual(record.score, 1.0)
        self.assertEqual(record.metrics["prompt_name"], "locomo_llm_judge_v1")
        self.assertEqual(result.summary["accuracy"], 1.0)
        self.assertEqual(record.metadata["judge_response"], generated_answer)
        self.assertEqual(record.metadata["question_category"], "2")
        self.assertEqual(result.summary["by_question_category"]["2"]["accuracy"], 1.0)

    def test_judge_uses_raw_response_without_internal_think_chain(self) -> None:
        backend = FakeJudgeBackend('{"label":"CORRECT"}')
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="locomo10_nonempty_answers",
            items=(
                DatasetItem(
                    item_id="conv-1",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="conv-1:q1",
                            query="When did Ava move?",
                            label=QuestionLabel(reference_answer="7 May 2023"),
                            metadata={"question_type": "2"},
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="conv-1",
            question_id="conv-1:q1",
            query="When did Ava move?",
            answer="Ava moved on May 7th.",
            agent_name="test-agent",
            metadata={
                "raw_response": (
                    "<think>private chain of thought</think>\n"
                    "<|message|>\n"
                    "## STEP 1: RELEVANT MEMORIES EXTRACTION\n"
                    "- Ava moved on May 7th.\n\n"
                    "## FINAL ANSWER:\nAva moved on May 7th."
                )
            },
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        self.assertNotIn("private chain of thought", backend.prompts[0])
        self.assertIn("## STEP 1: RELEVANT MEMORIES EXTRACTION", backend.prompts[0])
        self.assertEqual(
            result.record_for("conv-1", "conv-1:q1").metadata["judge_response"],
            (
                "## STEP 1: RELEVANT MEMORIES EXTRACTION\n"
                "- Ava moved on May 7th.\n\n"
                "## FINAL ANSWER:\nAva moved on May 7th."
            ),
        )

    def test_longmemeval_uses_prompt_selected_by_question_type(self) -> None:
        backend = FakeJudgeBackend("yes")
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="longmemeval_focus30",
            items=(
                DatasetItem(
                    item_id="0a995998",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="0a995998",
                            query="How many items should I return?",
                            label=QuestionLabel(reference_answer=3),
                            metadata={"question_type": "temporal-reasoning"},
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="0a995998",
            question_id="0a995998",
            query="How many items should I return?",
            answer="You need to pick up or return 3 items.",
            agent_name="test-agent",
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        expected_prompt = longmemeval_prompt(
            "temporal-reasoning",
            "How many items should I return?",
            "3",
            "You need to pick up or return 3 items.",
            abstention=False,
        )
        self.assertEqual(backend.prompts, [expected_prompt])
        record = result.record_for("0a995998", "0a995998")
        self.assertTrue(record.passed)
        self.assertEqual(record.score, 1.0)
        self.assertEqual(
            record.metrics["prompt_name"],
            "longmemeval_official_eval_qa_v1",
        )
        self.assertEqual(record.metadata["question_category"], "temporal-reasoning")
        self.assertEqual(
            result.summary["by_question_category"]["temporal-reasoning"]["accuracy"],
            1.0,
        )

    def test_locomo_category_5_is_skipped_without_judge_call(self) -> None:
        backend = FakeJudgeBackend('{"label":"CORRECT"}')
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="locomo10",
            items=(
                DatasetItem(
                    item_id="conv-1",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="conv-1:q5",
                            query="Unscored question",
                            label=QuestionLabel(reference_answer="something"),
                            metadata={"question_type": "5"},
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="conv-1",
            question_id="conv-1:q5",
            query="Unscored question",
            answer="something",
            agent_name="test-agent",
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        self.assertEqual(backend.prompts, [])
        record = result.record_for("conv-1", "conv-1:q5")
        self.assertIsNone(record.passed)
        self.assertEqual(record.metrics["skip_reason"], "category_5")
        self.assertEqual(
            result.summary["by_question_category"]["5"]["skipped_count"],
            1,
        )

    def test_mbench_exact_correct_reference_bypasses_judge_call(self) -> None:
        backend = FakeJudgeBackend('{"label":"WRONG"}')
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="mbench_persona_0",
            items=(
                DatasetItem(
                    item_id="persona_0",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="instance-1:q1",
                            query="Which class should I pick?",
                            label=QuestionLabel(
                                reference_answer=(
                                    "needs confirmation: live class, self-paced course",
                                ),
                                metadata={
                                    "incorrect_answers": ["Open Motion Academy"],
                                },
                            ),
                            metadata={
                                "relation_type": "contradictory",
                                "relation_subtype": "contradictory",
                                "question_category": "contradictory",
                            },
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="persona_0",
            question_id="instance-1:q1",
            query="Which class should I pick?",
            answer="needs confirmation: live class, self-paced course",
            agent_name="test-agent",
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        self.assertEqual(backend.prompts, [])
        record = result.record_for("persona_0", "instance-1:q1")
        self.assertTrue(record.passed)
        self.assertEqual(record.metrics["prompt_name"], MBENCH_PROMPT_NAME)
        self.assertEqual(
            record.metrics["judge_source"],
            "deterministic_reference_bypass",
        )
        self.assertEqual(
            result.summary["by_question_category"]["contradictory"]["accuracy"],
            1.0,
        )

    def test_mbench_prompt_uses_correct_incorrect_and_relation_guidance(self) -> None:
        backend = FakeJudgeBackend(
            '{"label":"CORRECT","reason":"It asks for clarification."}'
        )
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="mbench_persona_0",
            items=(
                DatasetItem(
                    item_id="persona_0",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="instance-2:q1",
                            query="Which Python signup should I choose?",
                            label=QuestionLabel(
                                reference_answer=(
                                    "Clarify live versus self-paced preference first.",
                                ),
                                evidence_ids=("session-a", "session-b"),
                                metadata={
                                    "incorrect_answers": [
                                        "Open Motion Academy",
                                    ],
                                    "facts": [
                                        "Amara both likes and dislikes self-paced coding courses.",
                                    ],
                                    "case": (
                                        "Amara gives conflicting accounts of "
                                        "self-paced coding courses."
                                    ),
                                },
                            ),
                            metadata={
                                "relation_type": "contradictory",
                                "relation_subtype": "contradictory",
                                "topic": "Technology",
                                "source": "user-related",
                                "question_category": "contradictory",
                            },
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="persona_0",
            question_id="instance-2:q1",
            query="Which Python signup should I choose?",
            answer="I would ask you to clarify the format first.",
            agent_name="test-agent",
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        expected_prompt = mbench_judge_prompt(
            question="Which Python signup should I choose?",
            generated_answer="I would ask you to clarify the format first.",
            correct_answers=["Clarify live versus self-paced preference first."],
            incorrect_answers=["Open Motion Academy"],
            metadata={
                "relation_type": "contradictory",
                "relation_subtype": "contradictory",
                "topic": "Technology",
                "source": "user-related",
                "question_category": "contradictory",
                "incorrect_answers": ["Open Motion Academy"],
                "facts": [
                    "Amara both likes and dislikes self-paced coding courses.",
                ],
                "case": (
                    "Amara gives conflicting accounts of "
                    "self-paced coding courses."
                ),
            },
        )
        self.assertEqual(backend.prompts, [expected_prompt])
        self.assertIn("Known incorrect answers:\n- Open Motion Academy", backend.prompts[0])
        self.assertIn("Relation semantics guidance:", backend.prompts[0])
        record = result.record_for("persona_0", "instance-2:q1")
        self.assertTrue(record.passed)
        self.assertEqual(record.score, 1.0)
        self.assertEqual(record.metrics["dataset_family"], "mbench")
        self.assertEqual(record.metrics["judge_reason"], "It asks for clarification.")
        self.assertEqual(record.metadata["judge_response"], answer.answer)

    def test_mbench_text_incorrect_label_is_wrong(self) -> None:
        backend = FakeJudgeBackend("INCORRECT")
        evaluator = DatasetPromptJudgeEvaluator(backend=backend)
        dataset = Dataset(
            name="mbench_persona_0",
            items=(
                DatasetItem(
                    item_id="persona_0",
                    conversations=(),
                    questions=(
                        DatasetQuestion(
                            question_id="instance-3:q1",
                            query="Which option?",
                            label=QuestionLabel(
                                reference_answer=("Ask for clarification.",),
                            ),
                            metadata={"relation_type": "contradictory"},
                        ),
                    ),
                ),
            ),
        )
        answer = AnswerRecord(
            item_id="persona_0",
            question_id="instance-3:q1",
            query="Which option?",
            answer="Choose Open Motion Academy.",
            agent_name="test-agent",
        )

        result = EvaluationRunner(evaluator).run(dataset, (answer,))

        record = result.record_for("persona_0", "instance-3:q1")
        self.assertFalse(record.passed)
        self.assertEqual(record.metrics["judge_label"], "WRONG")


if __name__ == "__main__":
    unittest.main()
