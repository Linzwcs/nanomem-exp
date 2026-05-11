from __future__ import annotations

import argparse
import json
from typing import Sequence

from memexp.runs import execute_experiment_run_spec, load_experiment_run_spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a memexp experiment spec.")
    parser.add_argument("config", help="Path to a JSON experiment run spec.")
    args = parser.parse_args(argv)

    output = execute_experiment_run_spec(load_experiment_run_spec(args.config))
    payload = {
        "run_dir": str(output.run_dir),
        "manifest": output.manifest["artifacts"].get("manifest", str(output.run_dir / "manifest.json")),
        "summary": output.result.summary,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
