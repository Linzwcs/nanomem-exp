#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memexp.adapters.locomo import locomo_records_to_unified
from memexp.adapters.unified import export_summary, stream_json_array, write_unified_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Locomo JSON records to memexp unified dataset JSON."
    )
    parser.add_argument("--input", required=True, help="Path to Locomo JSON array.")
    parser.add_argument("--output", required=True, help="Path to write unified JSON.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Unified dataset_name. Defaults to the input file stem.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    dataset_name = args.dataset_name or input_path.stem
    unified = locomo_records_to_unified(
        stream_json_array(input_path),
        dataset_name=dataset_name,
        source_file=str(input_path),
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
