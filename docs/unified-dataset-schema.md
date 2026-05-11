# Unified Dataset Schema

Status: draft

The unified dataset format keeps each question and its evaluation label together
while preserving a clear read boundary for the three experiment loops.

```text
Build loop      uses items[].conversations
Answer loop     uses items[].questions without question.label
Evaluation loop uses items[].questions[].label
```

## v1 Shape

```json
{
  "schema_version": "memexp.unified_dataset.v1",
  "dataset_name": "dataset-name",
  "metadata": {},
  "items": [
    {
      "item_id": "example-or-subject-id",
      "conversations": [
        [
          {
            "message_id": "message-id",
            "role": "user|assistant|system|participant|conversation",
            "content": "message text",
            "timestamp": "timestamp or null",
            "metadata": {}
          }
        ]
      ],
      "questions": [
        {
          "question_id": "question-id",
          "query": "question text",
          "query_time": "timestamp or null",
          "label": {
            "reference_answer": "gold answer",
            "evidence_ids": ["message-or-session-id"],
            "metadata": {}
          },
          "metadata": {}
        }
      ],
      "metadata": {}
    }
  ]
}
```

Only fields shown outside `metadata` are unified fields. Dataset-specific fields,
source identifiers, categories, speaker names, and provenance details belong in
`metadata` at the closest relevant level.

Gold answers and evidence identifiers belong in `question.label`. Agent code
must consume the question input view without `label`.

Dataset-aware LLM evaluation uses the official prompt selected from
`dataset_name`: Locomo datasets use `locomo_llm_judge_v1`; LongMemEval datasets
use `longmemeval_official_eval_qa_v1`.

LongMemEval exports use one item per question. Each haystack session becomes one
conversation, and `answer_session_ids` are stored as
`question.label.evidence_ids` with `evidence_level = "conversation"`.
