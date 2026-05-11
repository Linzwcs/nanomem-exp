from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from memexp import (
    MemoryReadRequest,
    MemoryScope,
    NullMemorySystem,
    RawMessageConfig,
    RawMessageMemorySystem,
)


class FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)


class BaselineMemorySystemsTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("memexp.core.tokenization.tokenizer", return_value=FakeEncoding())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_raw_message_baseline_builds_and_reads_messages(self) -> None:
        system = RawMessageMemorySystem(
            RawMessageConfig(target_roles=("user",), top_k=2, context_tokens=40)
        )
        artifact = system.build(
            [
                [
                    {
                        "message_id": "m1",
                        "role": "user",
                        "content": "I moved to Seattle.",
                        "timestamp": "2024-01-01",
                    },
                    {
                        "message_id": "m2",
                        "role": "assistant",
                        "content": "Thanks.",
                        "timestamp": "2024-01-01",
                    },
                ]
            ],
            scope=MemoryScope(scope_id="sample", dataset="toy"),
        )

        result = system.load(artifact).read(
            MemoryReadRequest(
                query="Where did the user move?",
                query_time="2024-02-01",
            )
        )

        self.assertEqual(artifact.system_name, "raw_messages")
        self.assertEqual(len(artifact.units), 1)
        self.assertIn("Seattle", result.context.text)
        self.assertEqual(result.stats["baseline"], "raw_messages")

    def test_null_memory_baseline_returns_empty_context(self) -> None:
        system = NullMemorySystem()
        artifact = system.build(
            [[{"role": "user", "content": "I moved to Seattle."}]],
            scope=MemoryScope(scope_id="sample", dataset="toy"),
        )

        result = system.load(artifact).read(
            MemoryReadRequest(query="Where did the user move?")
        )

        self.assertEqual(artifact.units, ())
        self.assertEqual(result.context.text, "")
        self.assertEqual(result.stats["baseline"], "null_memory")


if __name__ == "__main__":
    unittest.main()
