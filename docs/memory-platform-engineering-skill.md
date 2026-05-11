# Memory Platform Engineering Skill Draft

Status: draft

This document defines the project-specific Codex skill we should install after the requirements and architecture documents stabilize. Until then, this file is the source of truth for how the skill should guide work in this repository.

## Purpose

The skill should make Codex behave like an engineer building a publishable long-term memory experiment platform, not like an assistant patching ad hoc experiment scripts.

It should enforce five project disciplines:

- platform-first design;
- controlled experimental variables;
- example-bound memory isolation;
- artifact-first reproducibility;
- terminal-native metric inspection.

## Proposed Skill Name

```text
memory-platform-engineering
```

## Trigger Scope

Use this skill when working on:

- requirements design;
- architecture documents;
- schema and contract design;
- experiment protocol design;
- run orchestration design;
- memory-system and agent-system boundaries;
- artifact, metric, registry, and reporting design;
- implementation plans and acceptance criteria for this platform.

Do not use it for unrelated coding tasks or generic software engineering advice.

## Proposed `SKILL.md`

```markdown
---
name: memory-platform-engineering
description: Use when working on the long-term personal memory experiment platform, including requirements, architecture, schemas, experiment protocols, memory systems, agent systems, artifacts, metrics, terminal observability, evaluation, reporting, and implementation planning. Enforce project-specific discipline: platform-first design, controlled variables, example-bound memory runtimes, artifact-first reproducibility, extensible metrics and methods, and no reliance on parent-directory experimental code unless explicitly requested.
---

# Memory Platform Engineering

## Mission

Treat this repository as a clean long-term personal memory experiment platform.

The platform evaluates `MemorySystem` implementations under controlled `AgentSystem` consumers. It is not a collection of one-off experiment scripts.

## Source Boundary

- Use only the current `memory/` project as source of truth unless the user explicitly allows outside references.
- Do not infer platform requirements from parent-directory experimental code.
- Preserve decisions in project documents before implementing broad behavior.

## Core Objects

Use these terms consistently:

- `MemorySystem`: the main research object. It receives scoped multi-session conversations, builds an isolated memory library, and exposes standard memory operations such as `read`.
- `AgentSystem`: the answer or task-execution system. It consumes a bound memory runtime and produces an answer, action, or trace.
- `FixedQueryQAAgent`: a deterministic `AgentSystem` that converts a question to a fixed memory read, then answers from the returned memory context.
- `MemoryToolAgent`: an `AgentSystem` that decides when and how to call memory during an agent loop.
- `Example`: one benchmark or task unit with allowed history, question/task, question time, gold answer or rubric, category, and metadata.
- `BoundMemoryRuntime`: the memory runtime bound by the platform to a specific example, subject, history scope, and time cutoff.
- `MemoryArtifact`: persisted build output for one allowed memory scope.
- `MemoryReadResult`: structured output of memory reads, including returned memory items or rendered memory context, stats, and trace references.
- `RunManifest`: reproducibility record for a build, eval, agent, analysis, or reporting run.
- `Metric`: structured measurement written to artifacts or registry, not just printed.
- `Reporter`: terminal, CSV, Markdown, paper-table, or result-card view over structured outputs.

## System Boundary Rules

- `MemorySystem` owns memory construction, storage, indexing, retrieval, and memory read behavior.
- Retriever implementations are internal read policies of `MemorySystem`, not top-level platform components.
- Memory rendering, filtering, and budgeted context assembly are part of the `MemorySystem.read` contract unless a later document explicitly separates them.
- `AgentSystem` owns the control flow that decides how memory is used to answer or act.
- Fixed-query QA is a deterministic `AgentSystem`, not a separate top-level retrieval pipeline.
- `AgentSystem` must not read memory artifact internals directly. It must use the bound memory API.
- `AgentSystem` must receive an example-bound memory runtime from the platform runner. It must not choose arbitrary memory artifacts.

## Design Workflow

When designing a new platform area:

1. Define the user need.
2. Define non-goals.
3. Define core objects.
4. Define inputs and outputs.
5. Define lifecycle and ownership.
6. Define controlled variables.
7. Define failure modes.
8. Define artifact outputs.
9. Define metrics and terminal views.
10. Define extension boundaries.
11. Define tests and acceptance criteria.

## Controlled Variables

Every result-affecting variable must be explicit.

- First-class experiment variables belong in the experiment spec.
- Runtime and provenance variables belong in the run manifest.
- Diagnostic observations belong in artifacts or the result registry.

When a new method, metric, prompt, model, budget, read policy, agent policy, or evaluator is added, document whether it changes comparability.

## Extensibility Rules

Design extension points deliberately. Prefer a small number of stable interfaces over many informal hooks.

Every new method or metric must define:

- purpose;
- input contract;
- output contract;
- config fields;
- artifact outputs;
- failure modes;
- comparability impact;
- smoke test expectation;
- terminal summary behavior.

## Terminal Observability

Design every run so important metrics can be inspected from the terminal without opening notebooks or dashboards.

Prefer:

- compact tables;
- sorted deltas;
- category breakdowns;
- top failing examples;
- paired comparison summaries;
- artifact path references;
- machine-readable JSON/JSONL/CSV/SQLite plus human-readable terminal views.

Every major run type should support:

- `summary`: aggregate scores, cost, latency, and key diagnostics;
- `examples`: question-level or task-level results;
- `compare`: paired comparison between two runs;
- `failures`: grouped failure analysis;
- `artifacts`: generated paths, hashes, schema versions, and parent links.

## Documentation Standard

Every major design document should answer:

- What problem does this solve?
- Who uses it?
- What are the core workflows?
- What are the stable contracts?
- What artifacts are produced?
- Which variables are controlled?
- What changes make results incomparable?
- How can metrics be read from the terminal?
- How is the system extended?
- How is it tested?
- What is intentionally out of scope?

## Publishable Software Engineering Claim

Preserve a software engineering research claim in major design choices.

The platform should make AI memory evaluation more:

- reproducible;
- comparable;
- auditable;
- extensible;
- observable;
- useful for failure attribution.
```

## Installation Plan

After this draft stabilizes:

1. Create a real skill directory named `memory-platform-engineering`.
2. Copy the proposed `SKILL.md` into that directory.
3. Validate the skill with the Codex skill validation workflow.
4. Use the installed skill for future requirements, architecture, and implementation work.
