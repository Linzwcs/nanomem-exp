from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from memexp.reports.summary import experiment_report_row, write_report_table
from memexp.runs.cache import redact_sensitive, to_jsonable
from memexp.runs.experiment import ExperimentRunResult
from memexp.runs.serialization import (
    answer_record_to_dict,
    build_record_to_dict,
    evaluation_record_to_dict,
    index_record_to_dict,
)


def write_run_manifest(
    *,
    run_dir: str | Path,
    run_id: str,
    spec: dict[str, Any],
    result: ExperimentRunResult,
    extra_artifacts: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)

    paths = {
        "build_records": target / "build.jsonl",
        "index_records": target / "index.jsonl",
        "answer_records": target / "answers.jsonl",
        "evaluation_records": target / "evaluations.jsonl",
        "evaluation_by_category_json": target / "evaluation_by_category.json",
        "evaluation_by_category_csv": target / "evaluation_by_category.csv",
        "summary": target / "summary.json",
        "report_json": target / "report.json",
        "report_csv": target / "report.csv",
        "report_md": target / "report.md",
        "manifest": target / "manifest.json",
    }

    _write_jsonl(paths["build_records"], [
        build_record_to_dict(record) for record in result.build.records
    ])
    _write_jsonl(paths["index_records"], [
        index_record_to_dict(record) for record in result.index.records
    ])
    _write_jsonl(paths["answer_records"], [
        answer_record_to_dict(record) for record in result.answer.records
    ])
    _write_jsonl(paths["evaluation_records"], [
        evaluation_record_to_dict(record)
        for record in result.evaluation.records
    ])
    category_scores = result.evaluation.summary.get("by_question_category") or {}
    _write_json(paths["evaluation_by_category_json"], category_scores)
    _write_category_scores_csv(
        paths["evaluation_by_category_csv"],
        category_scores,
    )
    _write_json(paths["summary"], result.summary)

    row = experiment_report_row(result, run_id=run_id)
    write_report_table((row,), paths["report_json"])
    write_report_table((row,), paths["report_csv"])
    write_report_table((row,), paths["report_md"])

    manifest = {
        "schema_version": "memexp.run_manifest.v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": result.dataset_name,
        "spec": redact_sensitive(to_jsonable(spec)),
        "summary": result.summary,
        "report": row,
        "artifacts": {
            name: str(path)
            for name, path in paths.items()
        },
    }
    if extra_artifacts:
        manifest["artifacts"].update({
            name: str(path) for name, path in extra_artifacts.items()
        })
    _write_json(paths["manifest"], manifest)
    return manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_category_scores_csv(path: Path, scores: dict[str, Any]) -> None:
    fields = (
        "question_category",
        "question_count",
        "evaluation_count",
        "evaluated_count",
        "skipped_count",
        "passed_count",
        "accuracy",
        "avg_score",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for category, stats in sorted(scores.items()):
            row = {"question_category": category}
            if isinstance(stats, dict):
                row.update(stats)
            writer.writerow({field: row.get(field) for field in fields})
