from __future__ import annotations

import unittest

from memexp.adapters.longmemeval import longmemeval_records_to_unified
from memexp.adapters.unified import SCHEMA_VERSION


class LongMemEvalAdapterTest(unittest.TestCase):
    def test_longmemeval_records_are_exported_as_question_items(self) -> None:
        records = [
            {
                "question_id": "q-1",
                "question_type": "multi-session",
                "question": "How many items should I return?",
                "question_date": "2023/02/15 (Wed) 23:50",
                "answer": 3,
                "answer_session_ids": ["answer-session-1"],
                "haystack_dates": [
                    "2023/02/15 (Wed) 01:41",
                    "2023/02/15 (Wed) 02:06",
                ],
                "haystack_session_ids": [
                    "session-1",
                    "answer-session-1",
                ],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "Can you help me plan a trip?",
                        },
                        {
                            "role": "assistant",
                            "content": "Sure.",
                        },
                    ],
                    [
                        {
                            "role": "user",
                            "content": "I need to return three shirts.",
                        },
                    ],
                ],
            }
        ]

        unified = longmemeval_records_to_unified(
            records,
            dataset_name="longmemeval-test",
            source_file="/tmp/longmemeval-test.json",
        )

        self.assertEqual(unified["schema_version"], SCHEMA_VERSION)
        self.assertEqual(unified["dataset_name"], "longmemeval-test")
        self.assertEqual(unified["metadata"]["source_dataset"], "longmemeval")
        self.assertEqual(unified["metadata"]["question_count"], 1)
        self.assertEqual(len(unified["items"]), 1)

        item = unified["items"][0]
        self.assertEqual(item["item_id"], "q-1")
        self.assertEqual(len(item["conversations"]), 2)
        self.assertEqual(len(item["questions"]), 1)

        message = item["conversations"][0][0]
        self.assertEqual(message["message_id"], "session-1:m1")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["speaker"], "user")
        self.assertEqual(message["content"], "Can you help me plan a trip?")
        self.assertEqual(message["timestamp"], "2023/02/15 (Wed) 01:41")
        self.assertEqual(message["metadata"]["session_id"], "session-1")

        question = item["questions"][0]
        self.assertEqual(question["question_id"], "q-1")
        self.assertEqual(question["query"], "How many items should I return?")
        self.assertEqual(question["query_time"], "2023/02/15 (Wed) 23:50")
        self.assertEqual(question["metadata"]["question_type"], "multi-session")
        self.assertNotIn("reference_answer", question)
        self.assertNotIn("evidence_ids", question)

        label = question["label"]
        self.assertEqual(label["reference_answer"], 3)
        self.assertEqual(label["evidence_ids"], ["answer-session-1"])
        self.assertEqual(label["metadata"]["evidence_level"], "conversation")


if __name__ == "__main__":
    unittest.main()
