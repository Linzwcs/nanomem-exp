# Configs

Declarative specs for memory-system builds and evaluations.

Configs should describe the evaluated system, not incidental CLI arguments. A config should be hashable and sufficient to identify:

- dataset version and example set;
- `MemorySystem` implementation and parameters;
- `AgentSystem` implementation and parameters;
- evaluator and metric set;
- budgets, models, prompts, and other result-affecting variables;
- comparability label where applicable.

Retriever choices belong inside the `MemorySystem` read policy. Fixed-query QA
belongs inside an `AgentSystem`, not in a separate top-level retrieval pipeline.
Artifact-level index materialization, such as storage embedding cache warmup,
belongs in the `index` stage between build and answer.

LLM configs are explicit JSON files. Do not rely on `.env` loading inside the
runner. Fill `REPLACE_WITH_*` values directly in a private copy before running:

- `nanomem_gpt_oss_120b_locomo_judge.json`
- `nanomem_gpt_oss_120b_longmemeval_judge.json`

Both use:

- fact extraction model: `gpt-oss-120B`
- QA model: `gpt-oss-120B`
- evaluation judge model: `gpt-oss-120B`
- embedding model: `qwen3-0.6b-embedding`
