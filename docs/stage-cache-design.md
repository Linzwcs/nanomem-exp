# Stage Cache Design

Status: implemented for build, answer, and evaluation runner records

This document defines how staged caching should work in the experiment platform.

## 1. Core Principle

Cache is artifact reuse, not hidden memoization.

A cache hit means the platform found an existing artifact whose input hash, spec hash, schema version, and relevant provenance match the requested stage. The reused artifact must still be recorded in the current run manifest.

Caching must never be an untracked optimization inside a method implementation.

## 2. Ownership

Stage caching belongs to:

```text
memexp.runs
```

and the artifact/manifest layer in:

```text
memexp.core
```

Concrete memory systems, agents, evaluators, and metrics may expose stable inputs and outputs, but they should not silently own platform-level cache decisions.

Concrete methods may use local computation caches for private implementation
efficiency, such as compiled indexes or tokenizer helpers, but those caches must
not change platform-visible results or replace artifact-level stage reuse.

## 3. Stage Executor

The platform should provide one explicit executor for cacheable stages.

Conceptual interface:

```python
class StageExecutor:
    def run(
        self,
        stage_name: str,
        input_refs: list[ArtifactRef],
        input_hash: str,
        spec_hash: str,
        schema_version: str,
        compute_fn: Callable[[], StageOutput],
    ) -> StageResult:
        ...
```

Responsibilities:

- compute or receive the stage cache key;
- find matching prior artifacts;
- validate schema version and provenance;
- return cache hits without recomputing;
- execute `compute_fn` on cache miss;
- persist outputs as artifacts;
- write stage records to the run manifest;
- notify lifecycle hooks;
- provide structured data for terminal summaries.

## 4. Cacheable Stages

Initial cacheable stages:

| Stage | Input | Output |
| --- | --- | --- |
| `dataset_adaptation` | dataset spec, dataset files | examples artifact |
| `memory_build` | scoped conversations, `memsys` config | memory artifact |
| `memory_read` | memory artifact, read requests, read config | memory read results |
| `agent_run` | examples, bound memory artifacts, agent config | agent results and traces |
| `evaluation` | agent outputs, gold/rubrics, evaluator config | evaluation results |
| `metric_aggregation` | example results, metric config | aggregate metrics |
| `reporting` | metrics and result refs, report config | report artifacts |

Not every run must enable caching for every stage. The cache policy should be explicit in the run spec or runtime settings.

## 5. Cache Key Requirements

The cache key must include all result-affecting inputs.

Minimum components:

- stage name;
- input artifact ids or input content hashes;
- stage spec hash;
- schema version;
- dataset version where applicable;
- implementation id;
- implementation version or code provenance;
- prompt version where applicable;
- model and provider identity where applicable;
- tokenizer and budget where applicable;
- random seed and sampling parameters where applicable.

For hosted model calls, provider identity should be recorded clearly enough to interpret comparability. If provider behavior cannot be fully controlled, the result may still be reusable locally but should carry an appropriate comparability label.

## 6. Stage Record

Every stage execution or cache reuse should append a manifest record.

Conceptual schema:

```python
StageRecord:
  stage_name: str
  cache_key: str
  cache_hit: bool
  current_run_id: str
  producer_run_id: str | None
  input_refs: list[ArtifactRef]
  output_refs: list[ArtifactRef]
  spec_hash: str
  schema_version: str
  started_at: str
  ended_at: str
  stats: dict
  error: dict | None
```

This lets the terminal and reports show which stages were recomputed and which were reused.

## 7. Lifecycle Hooks

Hooks are allowed, but they must be formal lifecycle hooks, not hidden control-flow hacks.

Conceptual interface:

```python
class RunHook:
    def before_stage(self, stage_context): ...
    def after_stage(self, stage_context, stage_result): ...
    def on_cache_hit(self, stage_context, artifact_refs): ...
    def on_cache_miss(self, stage_context): ...
    def on_artifact_written(self, stage_context, artifact_ref): ...
```

Allowed hook responsibilities:

- terminal progress;
- logging;
- tracing;
- cost accounting;
- debug instrumentation;
- external observability export.

Hooks must not:

- mutate stage inputs;
- mutate stage outputs;
- change cache keys;
- skip required artifact writes;
- hide errors;
- alter metric values;
- make results incomparable without recording it.

## 8. Terminal Output

Every run summary should include stage cache status.

Example:

```text
Stage                Cache  Outputs
dataset_adaptation   HIT    examples:locomo-dev@...
memory_build         HIT    memory_artifact:nanomem@...
memory_read          MISS   memory_reads:200 examples
agent_run            MISS   agent_results:200 examples
evaluation           HIT    eval:gpt-judge@...
metric_aggregation   MISS   metrics:accuracy/cost/latency
```

The terminal view should be rendered from `StageRecord` and artifact metadata.

## 9. Failure Modes

The platform should detect and report:

- missing input artifacts;
- stale schema versions;
- cache key collisions;
- partial artifact writes;
- cache hit with missing output files;
- cache reuse across incompatible specs;
- unsupported cache reuse for nondeterministic stages;
- hosted-model outputs reused without clear provenance.

## 10. Current Implementation

The current implementation provides an explicit runner-level JSON cache:

```python
from memexp import JsonStageCache

cache = JsonStageCache("runs/cache")
result = ExperimentRunner(...).run(dataset, cache=cache)
```

Implemented cacheable loops:

| Stage | Cache key includes | Cached output |
| --- | --- | --- |
| `build` | dataset identity, item conversations, memory-system spec | `BuildRecord` |
| `answer` | dataset identity, question, artifact id, memory-system spec, agent spec, read budget | `AnswerRecord` |
| `evaluate` | dataset identity, question label, answer record, evaluator spec | `EvaluationRecord` |

Runner summaries include `cache_hit_count` and `cache_miss_count`. Structured
loggers receive `cache_hit` and `cache_miss` events with the cache key.

Implementation specs are derived from object class/name and `config` when
present. Sensitive config fields such as API keys and base URLs are omitted
from cache keys. For NanoMem, hosted LLM and embedding specs record the resolved
model name, including model names supplied by environment variables, while still
omitting API keys and base URLs.

## 11. Acceptance Criteria

The next artifact/manifest layer should support:

- explicit cache policy in run settings;
- deterministic cache keys for at least `dataset_adaptation`, `memory_build`, and `evaluation`;
- manifest records for cache hits and misses;
- terminal cache summary;
- tests that verify result-affecting spec changes invalidate cache;
- tests that verify cache hits do not bypass manifest recording.
