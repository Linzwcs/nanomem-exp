# Long-Term Personal Memory Evaluation

This directory is the clean project workspace for the long-term personal memory evaluation framework.

The project should stay focused on long-term personal memory:

- multi-session personal histories;
- personal fact, preference, plan, relationship, and update memory;
- retrieval and rendering under answer-time budgets;
- official benchmark evaluation on LoCoMo, LongMemEval, BEAM, and related tasks;
- deployment-oriented evaluation for agents that read, write, and update personal memory.

## Directory Layout

```text
memory/
  configs/      Declarative experiment and system specs.
  docs/         Framework design, protocols, result cards, and release notes.
  src/          Framework implementation.
  scripts/      CLI entry points and operational scripts.
  tests/        Unit, integration, and protocol tests.
  artifacts/   Local generated artifacts. Large files should not be committed by default.
  results/     Public or shareable result tables and reports.
```

## Initial Build Order

1. Define core schemas for examples, memory units, run manifests, artifacts, and metrics.
2. Add dataset adapters for LoCoMo and LongMemEval.
3. Add build/eval run separation.
4. Add artifact logging and a small run registry.
5. Add paired comparison and summary reporting.
6. Add BEAM and agent-oriented personal-memory evaluation.
