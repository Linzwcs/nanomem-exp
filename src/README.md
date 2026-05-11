# Source

Implementation for the long-term personal memory evaluation framework.

Planned modules:

```text
memexp/
  core/        Stable schemas and protocol contracts.
  adapters/    Dataset and task adapters.
  memsys/      Memory-system methods, including nanomem and baselines.
  agents/      Fixed-query QA agents, memory-tool agents, and task agents.
  evaluators/  Official and deployment-oriented evaluators.
  metrics/     Metric contracts, aggregation, and paired comparisons.
  runs/        Independent build, answer, and evaluation loops.
  reports/     Terminal summaries, result cards, and public exports.
  cli/         Thin command-line entry points.
```

`MemorySystem` is the primary research object. Retrieval is modeled as an
internal read policy of a memory system, not as a top-level platform stage.
`AgentSystem` owns answer or task control flow and consumes an example-bound
memory runtime provided by the runner.

Memory artifacts are modeled as causal timelines: offline builds may process a
full subject history, but reads with `query_time` only expose units whose
`available_at` is not in the future.

Datasets use the unified `Dataset -> DatasetItem -> DatasetQuestion` contract.
Each `DatasetItem.conversations` is an independent memory build unit; its
questions are evaluated against the artifact built from those conversations.
External benchmark exports use the JSON shape documented in
`docs/unified-dataset-schema.md`, with `conversations`, `questions`, and
question-level `label` fields aligned to the build, answer, and evaluation
loops.

See `docs/code-architecture.md` for the package architecture. The platform
package is `memexp`; memory-system implementations live under `memexp.memsys`.

Current implementation starts with `memexp.memsys.nanomem`:

- storage policy that currently emits fact units;
- dense cosine retrieve policy;
- optional file-backed embedding vector cache under the retrieve backend,
  including storage-side memory-unit warmup;
- adaptive Markdown temporal render policy with metadata-only time grouping;
- unified `NanoMemSystem.build(...).load(...).read(...)` boundary.

The initial platform runner layer is split into three reusable loops:

- `MemoryBuildRunner`: `DatasetItem.conversations -> MemoryArtifact`;
- `AnswerRunner`: `MemoryArtifact + DatasetQuestion + AgentSystem -> AnswerRecord`;
- `EvaluationRunner`: `AnswerRecord + reference + Evaluator -> EvaluationRecord`.

`ExperimentRunner` composes these loops for smoke runs, but caching and reruns
should target the individual loop outputs.

Runners accept `RunExecutionConfig` for stable parallel execution and a
`RunLogger` for structured stage events. The default execution remains serial.
They also accept `JsonStageCache` for explicit stage reuse across the build,
answer, and evaluation loops. Cache hits are visible in runner summaries and
structured log events; memory systems, agents, and evaluators do not own this
platform-level cache.

Evaluation includes a dataset-aware prompt judge that selects the Locomo or
LongMemEval official QA prompt from the dataset name, so different memory
methods can be judged with the same prompt protocol.

Experiment runs can now be launched from a JSON run spec:

```bash
PYTHONPATH=src python -m memexp.cli.run configs/run.json
```

Minimal spec shape:

```json
{
  "run_id": "toy-raw",
  "output_dir": "runs",
  "cache_dir": "runs/cache",
  "dataset": {"path": "data/toy.unified.json", "format": "unified"},
  "memory_system": {"name": "raw_messages", "config": {"target_roles": ["user"]}},
  "agent": {"name": "fixed_query"},
  "evaluator": {"name": "contains"},
  "top_k": 5,
  "context_budget_tokens": 100,
  "execution": {"max_workers": 1}
}
```

Each run writes a manifest and flat artifacts under `runs/{run_id}/`:
`build.jsonl`, `answers.jsonl`, `evaluations.jsonl`, `events.jsonl`,
`summary.json`, and report tables in JSON/CSV/Markdown.

Baseline memory systems are available under `memexp.memsys.baselines`:

- `null_memory`: empty-memory control;
- `raw_messages`: raw message storage with token-overlap retrieval.
