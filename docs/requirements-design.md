# Requirements Design

Status: draft

This document captures the first requirements pass for the long-term personal memory experiment platform. It intentionally focuses on concepts, boundaries, and platform requirements before implementation details.

## 1. Platform Goal

Build a clean experiment platform for evaluating long-term personal memory systems.

The platform should support controlled, reproducible, artifact-first comparisons of memory systems across benchmark and agent-oriented tasks. It should make experiment results easy to inspect from the terminal and easy to explain through preserved artifacts.

This is an experiment platform, not a one-off benchmark script.

## 2. Research Object

The main research object is `MemorySystem`.

A `MemorySystem` receives scoped multi-session conversation histories and builds an isolated memory library. It exposes standard memory operations, especially `read`, for downstream answer or agent systems.

At the simplest boundary:

```python
conversations: list[list[dict]]
```

where:

```text
list[conversation]
conversation = list[message]
message = dict
```

Example message fields may include:

```text
role
content
timestamp
speaker
session_id
metadata
```

The platform should normalize benchmark-specific history formats into this scoped conversation form before passing them to a memory system.

Build implementations may receive full subject-level conversation timelines for
offline parallelism, but their outputs must satisfy a causal offline replay
contract: for any query time `t`, the materialized memory snapshot must match
what an online streaming implementation would have produced after observing
only conversations available at or before `t`.

## 3. System Boundary

The top-level evaluation boundary is:

```text
DatasetAdapter
  -> ExampleSet
  -> MemorySystem
  -> AgentSystem
  -> Evaluator
  -> Metrics / Reporter
```

### 3.1 MemorySystem

`MemorySystem` owns memory behavior.

Responsibilities:

- build a causally ordered memory timeline from scoped conversations;
- store or index memory;
- expose memory read behavior;
- return standardized memory read results;
- optionally expose write and update behavior;
- produce memory artifacts and build manifests;
- record memory read traces and diagnostics.

Retriever implementations are internal read policies of `MemorySystem`. They are not top-level platform components.

Examples of internal read policies:

- lexical read;
- dense read;
- hybrid read;
- temporal-aware read;
- reranked read;
- multi-query read.

The `read` contract should return a `MemoryReadResult`, not an implementation-specific object.

Conceptual object:

```python
MemoryReadResult:
  request_id: str
  query: str | dict
  items: list[dict]
  rendered_context: str | None
  stats: dict
  trace_ref: str | None
```

Rendering, filtering, and budgeted context assembly are considered part of
`MemorySystem.read` in the initial design. This keeps the research object as the
complete memory behavior exposed to agents. A later architecture document may
split this only if there is a concrete comparability reason.

When a read request includes `query_time`, `MemorySystem.read` materializes the
snapshot whose units are causally available at that time before retrieval and
rendering. This is not a future-leakage repair step; it is the read view of a
streaming-equivalent timeline artifact.

### 3.2 AgentSystem

`AgentSystem` owns answering or task execution.

Responsibilities:

- receive an `Example`;
- receive a platform-bound memory runtime;
- decide how to use memory;
- produce answer, action, or trace;
- record calls, observations, and final outputs.

Fixed query QA is a deterministic `AgentSystem`:

```text
question
  -> fixed memory read
  -> answer prompt
  -> final answer
```

Dynamic memory use is another `AgentSystem`:

```text
question/task
  -> agent loop
  -> memory read calls
  -> optional tool calls
  -> final answer/action
```

The important distinction is that `MemorySystem` implements memory capability, while `AgentSystem` controls how that capability is used.

## 4. Example-Bound Memory Runtime

Every `AgentSystem` run must use a memory runtime bound to the current example.

The agent must not select or load arbitrary memory artifacts. The platform runner owns binding.

A memory binding must enforce:

- dataset scope;
- subject scope, such as user or conversation;
- timeline scope;
- query-time snapshot selection;
- read/write isolation;
- artifact identity and provenance.

Conceptual object:

```python
MemoryBinding:
  example_id: str
  dataset: str
  subject_id: str
  history_scope: str
  query_time: str | None
  artifact_id: str
  isolation_mode: "read_only" | "sandbox_write" | "persistent_update"
```

This prevents cross-example leakage, future information leakage, and write pollution across runs.

## 5. Core Workflows

### 5.1 Dataset Adaptation

Input:

- benchmark data;
- internal task data;
- external result imports where allowed.

Output:

- normalized examples;
- scoped conversations;
- gold answers or rubrics;
- category and metadata fields.

### 5.2 Build Run

Input:

- examples or subject-level history scopes;
- `MemorySystem` spec;
- dataset version.

Output:

- memory artifact;
- build manifest;
- build stats;
- artifact hashes.

The build run is responsible for constructing memory from allowed history only.

### 5.3 Answer Run

Input:

- examples;
- bound memory artifacts;
- `AgentSystem` spec;

Output:

- agent results;
- memory read traces;
- answer outputs;
- answer manifest.

The answer run must not rebuild memory. It consumes memory artifacts produced by
the build run, loads an example-bound memory runtime, and lets the agent answer
each question.

### 5.4 Evaluation Run

Input:

- answer records;
- gold answers or rubrics;
- evaluator spec.

Output:

- judge outputs where applicable;
- metrics;
- eval manifest.

The evaluation run must not call memory or regenerate answers. This allows
evaluator changes to be isolated from memory-system and agent changes.

### 5.5 Analysis and Reporting

Input:

- structured run outputs;
- metrics;
- artifact references.

Output:

- terminal summaries;
- paired comparisons;
- failure reports;
- shareable result tables;
- result cards.

## 6. Controlled Variables

The platform must make result-affecting variables explicit.

### 6.1 First-Class Experiment Variables

These belong in the experiment spec:

- dataset;
- split or example set;
- memory system implementation;
- memory build parameters;
- memory read policy;
- memory read budget;
- agent system implementation;
- agent prompt;
- agent model;
- agent step budget;
- answer token budget;
- evaluator;
- judge model;
- metric set.

### 6.2 Runtime and Provenance Variables

These belong in the run manifest:

- config hash;
- git SHA when available;
- dataset version;
- schema version;
- prompt version;
- random seed;
- temperature;
- max tokens;
- tokenizer;
- provider or base URL identity;
- runtime parallelism;
- runner fail-fast behavior;
- cache policy;
- created timestamp;
- environment summary.

### 6.3 Diagnostic Variables

These belong in artifacts, metrics, or the result registry:

- memory unit count;
- memory token count;
- storage generation prompt/completion token count;
- storage generation-to-memory token ratio;
- index size;
- read recall at k where evidence labels exist;
- hit rate and MRR where applicable;
- returned context token count;
- answer accuracy;
- category-level score;
- failure type;
- latency;
- cost;
- examples fixed or broken in paired comparison;
- stale or conflicting memory rate where measurable.

## 7. Extensibility Requirements

The platform should be extensible through stable interfaces, not informal hooks.

Top-level extension points:

- `DatasetAdapter`;
- `MemorySystem`;
- `AgentSystem`;
- `Evaluator`;
- `Metric`;
- `Reporter`.

Every extension must define:

- purpose;
- input contract;
- output contract;
- config fields;
- artifact outputs;
- failure modes;
- comparability impact;
- smoke test expectation;
- terminal summary behavior.

## 8. Artifact Requirements

The platform must preserve intermediate outputs needed for failure attribution.

Minimum artifact categories:

- build manifest;
- memory artifact;
- memory stats;
- memory read trace;
- agent trace;
- answers or final outputs;
- judge outputs;
- example-level results;
- aggregate metrics;
- report outputs.

Artifacts should record:

- artifact id;
- run id;
- type;
- path;
- content hash;
- schema version;
- created timestamp;
- parent artifact ids.

## 9. Terminal Observability Requirements

Terminal reading is a first-class user workflow.

Every major run should support compact terminal views for:

- overall summary;
- category breakdown;
- example-level results;
- top failures;
- cost and latency;
- artifact paths;
- paired run comparison;
- metric deltas.

Terminal views must be rendered from structured outputs. They should not recompute experiment logic.

Runner logs should be structured events, not ad hoc prints. At minimum, build,
answer, and evaluation tasks should emit started, completed, failed, and retrying
events where applicable.

## 10. Comparability Requirements

Each reported result should state its comparability level.

Initial labels:

- `fully_controlled`;
- `protocol_aligned`;
- `reported_baseline`;
- `external_reference`;
- `internal_agent_eval`.

The platform should make it clear when two runs differ by a controlled variable versus when they are not directly comparable.

## 11. Non-Goals

Initial platform versions should not optimize for:

- generic ML training workflows;
- a full dashboard before terminal and artifact workflows are stable;
- compatibility with unrelated experimental scripts;
- deeply nested plugin systems;
- hiding experimental protocol inside command-line flags;
- letting agent systems bypass memory binding.

## 12. Software Engineering Research Claim

The platform should preserve a publishable software engineering contribution:

> Long-term AI memory evaluation can be made reproducible, comparable, extensible, observable, and auditable by treating memory systems as isolated, example-bound, artifact-producing software systems consumed by explicit agent systems.

This claim should guide future requirements, architecture, implementation, and evaluation design.

## 13. Initial Acceptance Criteria

The first usable version should demonstrate:

- one dataset adapter;
- at least two `MemorySystem` implementations or configurations;
- at least one fixed-query `AgentSystem`;
- example-bound memory runtime enforcement;
- structured build and eval manifests;
- structured metrics;
- terminal summary output;
- paired comparison between two runs;
- smoke tests for schema validity and run reproducibility.
