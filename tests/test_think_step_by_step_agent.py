from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memexp import (
    DatasetQuestion,
    MemoryReadRequest,
    MemoryReadResult,
    PackedContext,
    ThinkStepByStepAgent,
    ThinkStepByStepAgentConfig,
    render_think_step_by_step_prompt,
)


class FakeMemoryRuntime:
    def __init__(self) -> None:
        self.requests: list[MemoryReadRequest] = []

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        self.requests.append(request)
        return MemoryReadResult(
            request=request,
            ranked_units=(),
            context=PackedContext(
                text="- 2024-01-01: Ava moved to Seattle.",
                token_count=8,
                block_count=1,
            ),
            stats={"artifact_id": "artifact-1"},
        )


class ThinkStepByStepAgentTest(unittest.TestCase):
    def test_prompt_renders_memories_question_and_question_time(self) -> None:
        prompt = render_think_step_by_step_prompt(
            memories="- 2024-01-01: Ava moved to Seattle.",
            question="Where did Ava move?",
            question_time="2024-02-01",
            include_question_time=True,
        )

        self.assertIn("Memories:\n- 2024-01-01: Ava moved to Seattle.", prompt)
        self.assertIn("Question time: 2024-02-01", prompt)
        self.assertIn("Question: Where did Ava move?", prompt)
        self.assertIn("## FINAL ANSWER:", prompt)

    def test_prompt_omits_question_time_when_disabled(self) -> None:
        prompt = render_think_step_by_step_prompt(
            memories="memory",
            question="question",
            question_time=None,
            include_question_time=False,
        )

        self.assertNotIn("Question time:", prompt)
        self.assertIn("Question: question", prompt)

    def test_agent_uses_think_template_for_qa_completion(self) -> None:
        calls: list[dict] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="## FINAL ANSWER:\nAva moved to Seattle."
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        runtime = FakeMemoryRuntime()
        agent = ThinkStepByStepAgent(
            ThinkStepByStepAgentConfig(
                model="qa-model",
                base_url="https://qa.example/v1",
                api_key="test-key",
                max_tokens=123,
            )
        )

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            record = agent.answer(
                DatasetQuestion(
                    question_id="q1",
                    query="Where did Ava move?",
                    query_time="2024-02-01",
                ),
                runtime,
                item_id="item-1",
                top_k=3,
                context_budget_tokens=80,
            )

        prompt = calls[0]["messages"][0]["content"]
        self.assertEqual(calls[0]["model"], "qa-model")
        self.assertEqual(calls[0]["max_tokens"], 123)
        self.assertIn("Ava moved to Seattle", prompt)
        self.assertIn("Question time: 2024-02-01", prompt)
        self.assertEqual(runtime.requests[0].top_k, 3)
        self.assertEqual(runtime.requests[0].context_budget_tokens, 80)
        self.assertEqual(record.answer, "## FINAL ANSWER:\nAva moved to Seattle.")
        self.assertEqual(record.agent_name, "think_step_by_step")
        self.assertEqual(record.memory_artifact_id, "artifact-1")
        self.assertEqual(
            record.stats["qa_generation_tokens"],
            {"prompt": 10, "completion": 5, "total": 15},
        )
        self.assertIn("prompt", record.metadata)


if __name__ == "__main__":
    unittest.main()
