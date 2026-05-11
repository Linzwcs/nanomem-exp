from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from memexp.runs.experiment import ExperimentRunResult


REPORT_FIELDS = (
    "run_id",
    "dataset",
    "memory_system",
    "agent",
    "evaluator",
    "item_count",
    "question_count",
    "accuracy",
    "avg_score",
    "passed_count",
    "evaluated_count",
    "total_units",
    "avg_units_per_item",
    "memory_read_count",
    "total_context_tokens",
    "avg_context_tokens",
    "storage_generation_tokens",
    "build_cache_hit_count",
    "answer_cache_hit_count",
    "evaluation_cache_hit_count",
)


def experiment_report_row(
    result: "ExperimentRunResult",
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    build = result.build.summary
    answer = result.answer.summary
    evaluation = result.evaluation.summary
    return {
        "run_id": run_id,
        "dataset": result.dataset_name,
        "memory_system": build.get("memory_system"),
        "agent": answer.get("agent"),
        "evaluator": evaluation.get("evaluator"),
        "item_count": build.get("item_count"),
        "question_count": answer.get("question_count"),
        "accuracy": evaluation.get("accuracy"),
        "avg_score": evaluation.get("avg_score"),
        "passed_count": evaluation.get("passed_count"),
        "evaluated_count": evaluation.get("evaluated_count"),
        "total_units": build.get("total_units"),
        "avg_units_per_item": build.get("avg_units_per_item"),
        "memory_read_count": answer.get("memory_read_count"),
        "total_context_tokens": answer.get("total_context_tokens"),
        "avg_context_tokens": answer.get("avg_context_tokens"),
        "storage_generation_tokens": _storage_generation_tokens(result),
        "build_cache_hit_count": build.get("cache_hit_count"),
        "answer_cache_hit_count": answer.get("cache_hit_count"),
        "evaluation_cache_hit_count": evaluation.get("cache_hit_count"),
    }


def write_report_table(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = tuple(rows)
    if target.suffix == ".json":
        with target.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return
    if target.suffix == ".md":
        target.write_text(markdown_report_table(rows), encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in REPORT_FIELDS})


def markdown_report_table(rows: Iterable[dict[str, Any]]) -> str:
    rows = tuple(rows)
    header = "| " + " | ".join(REPORT_FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in REPORT_FIELDS) + " |"
    body = [
        "| " + " | ".join(_markdown_value(row.get(field)) for field in REPORT_FIELDS) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _storage_generation_tokens(result: ExperimentRunResult) -> int:
    total = 0
    for record in result.build.records:
        stats = record.stats.get("storage_token_stats") or {}
        generation = stats.get("generation_tokens") or {}
        total += int(generation.get("total") or 0)
    return total


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
