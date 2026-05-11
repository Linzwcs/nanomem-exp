# Code Verifier Skill Draft

Status: draft

This document defines a project-specific code verifier skill. The skill should be installed after we agree on the verification standard.

## Purpose

The verifier skill makes Codex validate code changes with evidence. It should prevent unverified architectural drift, broken imports, stale contracts, weak tests, hidden behavior changes, and metric or artifact regressions.

This is not a generic code review skill. It is a verification workflow for the `memexp` experiment platform.

## Proposed Skill Name

```text
memexp-code-verifier
```

## Trigger Scope

Use this skill when:

- verifying newly written code;
- checking a refactor;
- reviewing whether a change matches the platform architecture;
- deciding what tests or smoke checks are required;
- validating memory-system, agent-system, metric, artifact, cache, or reporter behavior;
- preparing a change for commit or PR.

Do not use it for pure requirements brainstorming unless code-level acceptance criteria are being defined.

## Verification Philosophy

Verification must produce evidence.

Acceptable evidence includes:

- passing command output;
- focused unit tests;
- smoke-run artifacts;
- import checks;
- contract checks;
- small deterministic examples;
- terminal metric output;
- manifest or artifact inspection.

Statements like "looks good" are not verification.

## Proposed `SKILL.md`

```markdown
---
name: memexp-code-verifier
description: Use when verifying code changes in the memexp long-term memory experiment platform, especially refactors or implementations involving MemorySystem, memexp.memsys.nanomem, AgentSystem, artifacts, metrics, stage cache, terminal reporters, configs, or run orchestration. Enforce evidence-based validation: architecture checks, import checks, contract checks, focused tests, smoke runs, artifact inspection, and concise reporting of residual risk.
---

# Memexp Code Verifier

## Mission

Verify that code changes are correct, scoped, testable, and aligned with the memexp architecture.

Verification must produce evidence. Do not rely on inspection alone when a command, test, import check, or smoke example can verify the behavior.

## Source Boundary

- Use the current project as source of truth.
- Do not compare against parent-directory experimental code unless the user explicitly asks.
- Protect user changes. Do not revert unrelated edits.

## Architecture Checks

Verify these boundaries:

- Top-level package is `memexp`.
- Memory-system implementations live under `memexp.memsys`.
- Our method lives under `memexp.memsys.nanomem`.
- Do not introduce a top-level `memsys` package.
- Do not use `memory_systems` as a package name.
- Fact storage, retrieve, render, and update policies for our method belong under `memexp.memsys.nanomem`.
- Platform runners may orchestrate systems, but should not contain method-specific logic.
- Agents must call memory through core contracts, not concrete NanoMem internals.
- Stage cache belongs in runs/artifact layers, not hidden inside methods.

## Verification Levels

Choose the lightest level that gives real evidence for the risk.

### Level 0: Static Read

Use for documentation-only or naming-only changes.

Evidence:
- files inspected;
- no runtime behavior changed.

### Level 1: Import and Syntax

Use for new modules, package moves, or public API changes.

Evidence:
- import command;
- syntax compilation where useful.

Examples:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m compileall -q src
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -c "from memexp.memsys.nanomem import NanoMemSystem"
```

### Level 2: Focused Unit Tests

Use for any behavior implementation.

Evidence:
- focused test command;
- tests cover the changed contract or failure mode.

Example:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests
```

### Level 3: Smoke Run

Use for changes touching runners, artifact writing, configs, terminal reports, or stage cache.

Evidence:
- small deterministic run;
- generated artifact paths;
- inspected manifest or terminal output.

### Level 4: Regression/Comparison

Use for changes touching metrics, ranking, rendering budgets, evaluation, cache reuse, or report exports.

Evidence:
- before/after comparison;
- paired example deltas where applicable;
- explanation for expected metric changes.

## Required Checks by Change Type

For `memexp.memsys.nanomem`:

- Verify `NanoMemSystem.build(...).load(...).read(...)` still works.
- Verify policy config changes affect artifact id or stats when result-affecting.
- Verify fact units preserve source ids and timestamps when present.
- Verify retrieval returns stable ranks for ties.
- Verify rendering respects context budget.

For contracts:

- Verify imports from public package entry points.
- Verify downstream code still type-shapes against the changed contract.
- Add or update tests for any required field or changed invariant.

For configs:

- Verify config keys map to actual code paths.
- Verify result-affecting config appears in hashes/manifests once that layer exists.
- Verify defaults are explicit and documented.

For stage cache:

- Verify cache key invalidates when result-affecting spec changes.
- Verify cache hits still write manifest records.
- Verify hooks cannot mutate inputs, outputs, cache keys, or metrics.

For metrics/reporters:

- Verify metrics are computed from structured artifacts, not hidden recomputation.
- Verify terminal views show enough evidence to inspect results.
- Verify missing data is reported explicitly.

## Blockers

Do not mark verification as passed if:

- changed code cannot be imported;
- a public contract is changed without updating callers or tests;
- a result-affecting variable is hidden outside config/spec/manifest;
- `nanomem` policies are placed outside `memexp.memsys.nanomem` without a documented reason;
- an agent bypasses the memory runtime API;
- cache reuse is hidden from manifests;
- tests fail for reasons related to the change;
- generated artifacts contradict terminal summaries.

## Reporting Format

Final verification reports should be concise:

- Scope verified.
- Commands run and result.
- Findings or blockers.
- Residual risk.
- Files most relevant to the verification.

If verification is incomplete, say exactly what was not run and why.
```

## Initial Standard For This Project

Until the skill is installed, use this practical standard:

1. New Python modules must pass import or test execution with `PYTHONPATH=src`.
2. New memory-system behavior must have focused tests.
3. Refactors must preserve documented package boundaries.
4. Fact-related NanoMem policy code must stay under `src/memexp/memsys/nanomem`.
5. Verification output must include exact commands run.
6. Generated `__pycache__` files should not remain in the workspace after tests.

## Installation Plan

After the standard stabilizes:

1. Create a skill directory named `memexp-code-verifier`.
2. Install the proposed `SKILL.md`.
3. Optionally add small helper scripts only if repeated verifier commands become tedious.
4. Validate the skill on a real code change and revise the checklist.
