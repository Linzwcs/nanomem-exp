from __future__ import annotations

import json
import unittest

from memexp.adapters.mbench import mbench_records_to_unified
from memexp.adapters.unified import SCHEMA_VERSION, unified_payload_to_dataset


class MBenchAdapterTest(unittest.TestCase):
    def test_mbench_persona_records_are_exported_as_one_memory_item(self) -> None:
        persona = {
            "persona_id": "0",
            "name": "Amara Nwosu",
            "sensitive_information": {
                "llm_api_key": "should-not-be-exported",
            },
        }
        history_sessions = [{
            "case_id": "case-1",
            "conversation_type": "troubleshooting",
            "history": [
                {
                    "role": "user",
                    "content": "I like weekend stargazing.",
                },
                {
                    "role": "assistant",
                    "content": "Weekends sound easier.",
                },
            ],
            "order": 0,
            "persona_id": "0",
            "persona_signal_level": "high",
            "session_id": "persona-0-s1",
            "source": "user-related",
            "timestamp": "2025-04-01T10:01:36+08:00",
        }]
        bench_instances = [{
            "case": "Amara prefers weekend sky watches over late weeknights.",
            "case_id": "case-1",
            "facts": [
                "Amara is usually game for clear weekend sky watches.",
            ],
            "instance_id": "instance-1",
            "persona_id": "0",
            "persona_str": json.dumps(persona),
            "qas": [{
                "correct_answers": ["Weekend sky watches are safest."],
                "incorrect_answers": ["Tuesday late-night rooftop watches."],
                "query": "Which sky-watch plan should I prefer?",
            }],
            "relation_subtype": "Context",
            "relation_type": "nuanced",
            "session_ids": ["persona-0-s1"],
            "source": "user-related",
            "topic": "Lifestyle",
        }]

        unified = mbench_records_to_unified(
            history_sessions,
            bench_instances,
            dataset_name="mbench_persona_0",
            source_dir="/tmp/persona_0",
        )

        self.assertEqual(unified["schema_version"], SCHEMA_VERSION)
        self.assertEqual(unified["metadata"]["source_dataset"], "mbench")
        self.assertEqual(unified["metadata"]["session_count"], 1)
        self.assertEqual(unified["metadata"]["question_count"], 1)
        self.assertEqual(len(unified["items"]), 1)

        item = unified["items"][0]
        self.assertEqual(item["item_id"], "persona_0")
        self.assertEqual(item["subject_id"], "0")
        self.assertEqual(item["metadata"]["persona_name"], "Amara Nwosu")
        self.assertNotIn("persona_str", json.dumps(item))
        self.assertNotIn("should-not-be-exported", json.dumps(item))

        message = item["conversations"][0][0]
        self.assertEqual(message["message_id"], "persona-0-s1:m1")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["speaker"], "Amara Nwosu")
        self.assertEqual(message["timestamp"], "2025-04-01T10:01:36+08:00")
        self.assertEqual(message["metadata"]["session_id"], "persona-0-s1")

        question = item["questions"][0]
        self.assertEqual(question["question_id"], "instance-1:q1")
        self.assertEqual(question["metadata"]["question_category"], "nuanced")
        self.assertEqual(question["metadata"]["question_type"], "Context")
        self.assertEqual(
            question["label"]["reference_answer"],
            ["Weekend sky watches are safest."],
        )
        self.assertEqual(question["label"]["evidence_ids"], ["persona-0-s1"])
        self.assertEqual(
            question["label"]["metadata"]["incorrect_answers"],
            ["Tuesday late-night rooftop watches."],
        )

        dataset = unified_payload_to_dataset(unified)
        self.assertEqual(dataset.name, "mbench_persona_0")
        self.assertEqual(dataset.items[0].questions[0].label.evidence_ids, ("persona-0-s1",))


if __name__ == "__main__":
    unittest.main()
