from __future__ import annotations

import unittest

from memexp import Dataset, DatasetItem, DatasetQuestion, QuestionLabel


class DatasetContractTest(unittest.TestCase):
    def test_dataset_item_build_scope_and_question_read_request(self) -> None:
        question = DatasetQuestion(
            question_id="q1",
            query="Where did the user move?",
            label=QuestionLabel(
                reference_answer="Seattle",
                evidence_ids=("m1",),
                metadata={"source": "gold"},
            ),
            query_time="2024-02-01",
            metadata={"category": "location"},
        )
        item = DatasetItem(
            item_id="item-1",
            subject_id="user-1",
            conversations=(
                (
                    {
                        "id": "m1",
                        "role": "user",
                        "content": "I moved to Seattle.",
                        "timestamp": "2024-01-01",
                    },
                ),
            ),
            questions=(question,),
            metadata={"source": "toy"},
        )
        dataset = Dataset(name="toy", split="dev", items=(item,))

        scope = dataset.items[0].to_memory_scope(dataset_name=dataset.name)
        request = dataset.items[0].questions[0].to_read_request(
            top_k=5,
            context_budget_tokens=100,
        )

        self.assertEqual(scope.scope_id, "item-1")
        self.assertEqual(scope.dataset, "toy")
        self.assertEqual(scope.subject_id, "user-1")
        self.assertEqual(scope.timeline_id, "item-1")
        self.assertEqual(request.query_id, "q1")
        self.assertEqual(request.query_time, "2024-02-01")
        self.assertEqual(request.top_k, 5)
        self.assertEqual(request.context_budget_tokens, 100)
        self.assertEqual(request.metadata, {"category": "location"})
        self.assertNotIn("label", request.metadata)


if __name__ == "__main__":
    unittest.main()
