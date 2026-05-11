# Scripts

Command-line entry points for running builds, evaluations, comparisons, and public result exports.

Scripts should stay thin. Core behavior belongs in `memory/src/`.

Available exports:

- `export_longmemeval_unified.py`: converts LongMemEval JSON arrays to the
  unified dataset JSON shape.
- `export_locomo_unified.py`: converts Locomo JSON arrays to the unified dataset
  JSON shape.
