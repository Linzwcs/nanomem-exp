#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memexp.adapters.mbench import mbench_records_to_unified
from memexp.adapters.unified import export_summary, stream_json_array, write_unified_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MBench persona files to memexp unified dataset JSON."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing history_sessions.json and bench_instances.json.",
    )
    parser.add_argument("--output", required=True, help="Path to write unified JSON.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Unified dataset_name. Defaults to mbench_<input-dir-name>.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    dataset_name = args.dataset_name or f"mbench_{input_dir.name}"
    unified = mbench_records_to_unified(
        stream_json_array(input_dir / "history_sessions.json", item_name="MBench history"),
        stream_json_array(input_dir / "bench_instances.json", item_name="MBench bench"),
        dataset_name=dataset_name,
        source_dir=str(input_dir),
    )

    write_unified_json(output_path, unified)
    print(
        json.dumps(
            export_summary(output_path, unified),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
