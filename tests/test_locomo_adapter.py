from __future__ import annotations

import unittest

from memexp.adapters.locomo import locomo_records_to_unified
from memexp.adapters.unified import SCHEMA_VERSION


class LocomoAdapterTest(unittest.TestCase):
    def test_locomo_records_are_grouped_into_unified_items(self) -> None:
        turn_records = [
            {
                "record_id": "D1:1",
                "session_id": "conv-1:session_1",
                "turn_index": 1,
                "speaker": "Ava",
                "text": "I moved to Seattle.",
                "timestamp": "1:00 pm on 1 January, 2024",
                "metadata": {
                    "session_key": "session_1",
                    "dia_id": "D1:1",
                    "img_url": None,
                },
            },
            {
                "record_id": "D1:2",
                "session_id": "conv-1:session_1",
                "turn_index": 2,
                "speaker": "Ben",
                "text": "That sounds exciting.",
                "timestamp": "1:00 pm on 1 January, 2024",
                "metadata": {
                    "session_key": "session_1",
                    "dia_id": "D1:2",
                },
            },
        ]
        records = [
            {
                "dataset": "locomo",
                "example_id": "locomo:conv-1:q1",
                "source_sample_id": "conv-1",
                "question_id": "conv-1:q1",
                "question_type": "2",
                "question": "Where did Ava move?",
                "answer": "Seattle",
                "question_date": None,
                "participants": ["Ava", "Ben"],
                "turn_records": turn_records,
                "gold_session_ids": ["conv-1:session_1"],
                "gold_turn_ids": ["D1:1"],
                "metadata": {
                    "dataset_file": "locomo10.json",
                    "category": 2,
                    "event_summary": {"oracle": "must not be copied"},
                },
            },
            {
                "dataset": "locomo",
                "example_id": "locomo:conv-1:q2",
                "source_sample_id": "conv-1",
                "question_id": "conv-1:q2",
                "question_type": "3",
                "question": "Who responded to Ava?",
                "answer": "Ben",
                "question_date": None,
                "participants": ["Ava", "Ben"],
                "turn_records": turn_records,
                "gold_session_ids": ["conv-1:session_1"],
                "gold_turn_ids": ["D1:2"],
                "metadata": {
                    "dataset_file": "locomo10.json",
                    "category": 3,
                },
            },
        ]

        unified = locomo_records_to_unified(
            records,
            dataset_name="locomo-test",
            source_file="/tmp/locomo-test.json",
        )

        self.assertEqual(unified["schema_version"], SCHEMA_VERSION)
        self.assertEqual(unified["dataset_name"], "locomo-test")
        self.assertEqual(unified["metadata"]["source_record_count"], 2)
        self.assertEqual(len(unified["items"]), 1)

        item = unified["items"][0]
        self.assertEqual(item["item_id"], "conv-1")
        self.assertEqual(len(item["conversations"]), 1)
        self.assertEqual(len(item["questions"]), 2)
        self.assertEqual(item["metadata"]["participants"], ["Ava", "Ben"])
        self.assertNotIn("event_summary", repr(item))

        message = item["conversations"][0][0]
        self.assertEqual(message["message_id"], "D1:1")
        self.assertEqual(message["role"], "participant")
        self.assertEqual(message["content"], "I moved to Seattle.")
        self.assertEqual(message["metadata"]["speaker"], "Ava")

        question = item["questions"][0]
        self.assertEqual(question["question_id"], "conv-1:q1")
        self.assertEqual(question["query"], "Where did Ava move?")
        self.assertNotIn("reference_answer", question)
        self.assertNotIn("evidence_ids", question)

        label = question["label"]
        self.assertEqual(label["reference_answer"], "Seattle")
        self.assertEqual(label["evidence_ids"], ["D1:1"])
        self.assertEqual(label["metadata"]["evidence_level"], "message")


if __name__ == "__main__":
    unittest.main()
