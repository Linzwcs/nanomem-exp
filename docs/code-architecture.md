# Code Architecture

Status: draft

This document defines the intended Python package architecture for the experiment platform.

## 1. Package Boundary

The top-level package is:

```text
memexp
```

`memexp` is the experiment platform. It owns dataset adaptation, experiment specs, run orchestration, artifact management, evaluation, metrics, reporting, and CLI entry points.

Memory-system implementations live under:

```text
memexp.memsys
```

`memsys` is the method layer for memory systems. Each subpackage under `memsys` represents a memory-system family or adapter that can be compared by the platform.

## 2. Proposed Layout

```text
src/
  memexp/
    __init__.py

    core/
      contracts.py
      time.py
      schemas.py
      specs.py
      artifacts.py
      manifests.py
      hashing.py
      errors.py

    adapters/
      base.py
      locomo.py
      longmemeval.py

    memsys/
      base.py
      registry.py

      nanomem/
        __init__.py
        system.py
        config.py
        storage.py
        retrieve.py
        render.py
        update.py
        defaults.yaml

      baselines/
        raw_context/
          system.py
          config.py
        chunk_rag/
          system.py
          config.py
        summary_memory/
          system.py
          config.py
        oracle_memory/
          system.py
          config.py

      external/
        mem0/
          system.py
          adapter.py
        zep/
          system.py
          adapter.py
        langmem/
          system.py
          adapter.py

    agents/
      base.py
      registry.py
      fixed_query_qa/
        system.py
        config.py
        prompts/
      memory_tool_agent/
        system.py
        config.py
        prompts/

    evaluators/
      base.py
      official.py
      judge.py

    metrics/
      base.py
      aggregate.py
      compare.py
      failures.py

    runs/
      build.py
      answer.py
      evaluate.py
      experiment.py
      binding.py
      stage.py
      cache.py
      registry.py

    reports/
      terminal.py
      markdown.py
      export.py

    cli/
      build.py
      eval.py
      compare.py
      inspect.py
```

## 3. Layer Responsibilities

### 3.1 `memexp.core`

Stable contracts and platform-level schemas.

This layer should not import concrete memory-system or agent implementations.

Core objects include:

- `Dataset`;
- `DatasetItem`;
- `DatasetQuestion`;
- `MemoryScope`;
- `MemoryUnit`;
- `MemoryArtifact`;
- `MemoryReadRequest`;
- `MemoryReadResult`;
- `AgentRunResult`;
- `RunManifest`;
- `ArtifactRef`;
- experiment specs and config hashing.

### 3.2 `memexp.adapters`

Dataset and task adapters.

Adapters normalize benchmark-specific formats into unified datasets. Each
`DatasetItem` is an independent memory build unit, and its
`DatasetQuestion` values are evaluated against that item's memory artifact.

They should not know how any concrete memory system works.

### 3.3 `memexp.memsys`

Memory-system method layer.

Every memory-system implementation must satisfy the same platform contract:

```python
class MemorySystem:
    def build(self, conversations, scope, config) -> MemoryArtifact:
        ...

    def load(self, artifact: MemoryArtifact):
        ...
```

The loaded runtime exposes:

```python
class MemoryRuntime:
    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        ...
```

Retrieval, filtering, rendering, and context-budget assembly are internal to `MemorySystem.read` unless a later design explicitly separates them for a controlled comparison.

Build implementations may process a full subject timeline for efficiency, but
the resulting `MemoryArtifact` must be streaming-equivalent. Each
`MemoryUnit` carries causal availability metadata such as `available_at` and
source time bounds. A `MemoryReadRequest.query_time` materializes the visible
snapshot before retrieval.

### 3.4 `memexp.memsys.nanomem`

Our method.

`nanomem` is a concrete memory-system family. Its internal policies are configurable but remain inside the method package:

- storage policy;
- retrieve policy;
- render policy;
- update policy;
- budget policy.

Example configuration shape:

```yaml
memory_system:
  name: nanomem
  params:
    storage:
      policy: fact
      backend: heuristic
      chunk_tokens: 1024
    retrieve:
      policy: dense_cosine_v1
      embedding_backend: hashing_dense_v1
      embedding_cache_path: null
      warm_storage_embeddings: false
      top_k: 50
    render:
      policy: adaptive_markdown_temporal_v1
      merge_policy: temporal_metadata_merge_v1
      context_tokens: 1000
    update_policy: append_with_conflict_marking_v1
```

The platform treats the whole configured `nanomem` instance as one `MemorySystem`.

Storage backend retries are method-internal because they sit at the backend call
boundary. For NanoMem fact storage, LLM extraction retries are configured in
`StorageConfig.retry`; exhausted retries fall back to heuristic fact extraction
unless `fail_on_error` is enabled. Retry attempts and final error metadata are
recorded in each generated memory unit's storage generation metadata.

Embedding vector caching is also method-internal because it sits at the dense
embedding backend boundary. `RetrieveConfig.embedding_cache_path` enables a
file-backed text-hash cache keyed by backend, resolved model name, namespace,
and text hash. The cache stores vectors only, not raw texts, API keys, or base
URLs. Cache path and namespace do not affect memory artifact ids or stage cache
keys because they change performance, not retrieval semantics.
`RetrieveConfig.warm_storage_embeddings` can precompute and cache memory-unit
embeddings when a memory artifact is loaded; query embeddings are still computed
at read time.

### 3.5 `memexp.memsys.baselines`

Controlled baseline memory systems.

These are implemented inside the platform so they can be compared under the same dataset, agent, evaluator, artifact, metric, and reporting protocol.

Current baselines:

- `null_memory`: builds an empty artifact and returns empty context, useful as
  a lower-bound control.
- `raw_messages`: stores raw conversation messages as memory units, retrieves
  with deterministic token overlap, and renders plain timestamped lines.

## 4. Run Specs, Manifests, and Reports

Experiments can be launched from a JSON run spec through:

```bash
PYTHONPATH=src python -m memexp.cli.run configs/run.json
```

The run spec selects:

- unified dataset file;
- memory system and config;
- agent and config;
- evaluator and config;
- execution settings;
- optional stage cache directory.

The runner writes a self-contained run directory:

```text
runs/{run_id}/
  manifest.json
  build.jsonl
  answers.jsonl
  evaluations.jsonl
  events.jsonl
  summary.json
  report.json
  report.csv
  report.md
```

The manifest records the redacted run spec, stage summaries, report row, and
artifact paths. Report rows aggregate task accuracy, score, unit counts, context
tokens, storage generation tokens, and stage cache hit counts.

Examples:

- full or truncated raw context;
- chunked RAG;
- summary memory;
- oracle memory.

### 3.6 `memexp.memsys.external`

Adapters for external memory systems.

External systems must still expose the platform `MemorySystem` contract. Their internal APIs are hidden behind adapters.

External runs must carry comparability labels because implementation details, hosted state, or vendor behavior may not be fully controlled.

### 3.7 `memexp.agents`

Answer and task execution systems.

Agents consume an example-bound memory runtime. They must not load memory artifacts directly.

Initial agents:

- `fixed_query_qa`: deterministic question-to-memory-read-to-answer pipeline;
- `memory_tool_agent`: dynamic agent loop that chooses memory reads.

### 3.8 `memexp.runs`

Run orchestration.

Runners bind dataset items to memory artifacts, enforce isolation, call systems,
write artifacts, and record manifests.

Runners should not contain method-specific logic.

The initial runner design is three independent loops:

```text
MemoryBuildRunner
  DatasetItem.conversations
  -> MemoryArtifact

AnswerRunner
  MemoryArtifact + DatasetQuestion + AgentSystem
  -> AnswerRecord

EvaluationRunner
  AnswerRecord + DatasetQuestion.reference + Evaluator
  -> EvaluationRecord
```

`ExperimentRunner` may compose the three loops for a full smoke run, but the
platform should still expose each loop independently so a later cache layer can
reuse build artifacts, regenerate answers with a different agent, or reevaluate
old answers with a different evaluator.

Parallelism and logging belong to the runner layer. Runners accept an execution
config for max worker count and fail-fast behavior, and emit structured
`RunEvent` values through a `RunLogger`. Concrete memory systems should not hide
run-level parallelism or terminal reporting inside method code.

Stage caching also belongs in this layer. Cache hits are artifact reuse events,
not hidden memoization inside concrete methods. See `docs/stage-cache-design.md`.

### 3.9 `memexp.metrics` and `memexp.reports`

Metrics compute structured observations.

Reports render structured observations into terminal views, Markdown, CSV, or public result cards.

Terminal output must read from artifacts or registry records, not from recomputing experiment logic.

## 4. Dependency Direction

Allowed dependency direction:

```text
cli -> runs -> adapters / memsys / agents / evaluators / metrics / reports -> core
```

Method packages may depend on `memexp.core`, but `memexp.core` must not depend on method packages.

`memexp.agents` may call memory through core contracts only. Agents must not import concrete `memexp.memsys.nanomem` internals.

## 5. Naming Rules

- Use `memexp` for the platform package.
- Use `memexp.memsys` for memory-system implementations.
- Use `memexp.memsys.nanomem` for our method.
- Use `memexp.agents` for answer or task systems.
- Do not create a top-level `memsys` package.
- Do not use `memory_systems` as a package name.
