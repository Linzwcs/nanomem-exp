from __future__ import annotations

import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from memexp import (
    execute_experiment_run_spec,
    load_experiment_run_spec,
    load_unified_dataset,
)
from memexp.cli.run import main as cli_main


class FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)


class RunSpecManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("memexp.core.tokenization.tokenizer", return_value=FakeEncoding())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_run_spec_executes_and_writes_manifest_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "toy.unified.json"
            spec_path = root / "run.json"
            _write_json(dataset_path, _toy_unified_payload())
            _write_json(
                spec_path,
                {
                    "run_id": "toy-run",
                    "output_dir": str(root / "runs"),
                    "cache_dir": str(root / "cache"),
                    "dataset": {"path": str(dataset_path), "format": "unified"},
                    "memory_system": {
                        "name": "raw_messages",
                        "config": {
                            "target_roles": ["user"],
                            "context_tokens": 60,
                        },
                    },
                    "agent": {"name": "fixed_query"},
                    "evaluator": {"name": "contains"},
                    "top_k": 3,
                    "context_budget_tokens": 60,
                    "execution": {"max_workers": 1},
                },
            )

            dataset = load_unified_dataset(dataset_path)
            live_answer_line_counts: list[int] = []

            def assert_live_answers_before_manifest(**kwargs):
                run_dir = Path(kwargs["run_dir"])
                live_answer_line_counts.append(
                    len((run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines())
                )
                from memexp.runs.manifest import write_run_manifest

                return write_run_manifest(**kwargs)

            with patch(
                "memexp.runs.spec.write_run_manifest",
                side_effect=assert_live_answers_before_manifest,
            ):
                output = execute_experiment_run_spec(load_experiment_run_spec(spec_path))

            run_dir = root / "runs" / "toy-run"
            manifest = json.loads((run_dir / "manifest.json").read_text())
            report = json.loads((run_dir / "report.json").read_text())

            self.assertEqual(dataset.name, "toy")
            self.assertEqual(output.run_dir, run_dir)
            self.assertEqual(manifest["run_id"], "toy-run")
            self.assertEqual(manifest["report"]["accuracy"], 1.0)
            self.assertEqual(live_answer_line_counts, [1])
            self.assertEqual(report[0]["memory_system"], "raw_messages")
            self.assertTrue((run_dir / "build.jsonl").exists())
            self.assertTrue((run_dir / "answers.jsonl").exists())
            self.assertTrue((run_dir / "evaluations.jsonl").exists())
            self.assertTrue((run_dir / "events.jsonl").exists())
            self.assertIn("accuracy", (run_dir / "report.md").read_text())

    def test_cli_runs_json_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "toy.unified.json"
            spec_path = root / "run.json"
            _write_json(dataset_path, _toy_unified_payload())
            _write_json(
                spec_path,
                {
                    "run_id": "cli-run",
                    "output_dir": str(root / "runs"),
                    "dataset": {"path": str(dataset_path)},
                    "memory_system": {"name": "raw_messages"},
                    "evaluator": {"name": "contains"},
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(cli_main([str(spec_path)]), 0)
            self.assertIn("cli-run", stdout.getvalue())
            self.assertTrue((root / "runs" / "cli-run" / "manifest.json").exists())

    def test_llm_config_templates_parse_with_explicit_models(self) -> None:
        for path in (
            "configs/nanomem_gpt_oss_120b_locomo_judge.json",
            "configs/nanomem_gpt_oss_120b_longmemeval_judge.json",
        ):
            spec = load_experiment_run_spec(path)

            memory_config = spec.memory_system.config
            agent_config = spec.agent.config
            evaluator_config = spec.evaluator.config

            self.assertTrue(memory_config["storage"]["llm_model"])
            self.assertEqual(spec.agent.name, "think_step_by_step")
            self.assertIn("model", agent_config)
            self.assertTrue(memory_config["retrieve"]["embedding_model"])
            self.assertTrue(agent_config["model"])
            self.assertTrue(evaluator_config["model"])
            self.assertNotIn("env_file", spec.to_dict())


def _toy_unified_payload() -> dict:
    return {
        "schema_version": "memexp.unified_dataset.v1",
        "dataset_name": "toy",
        "metadata": {},
        "items": [
            {
                "item_id": "item-1",
                "conversations": [
                    [
                        {
                            "message_id": "m1",
                            "role": "user",
                            "content": "I moved to Seattle.",
                            "timestamp": "2024-01-01",
                        }
                    ]
                ],
                "questions": [
                    {
                        "question_id": "q1",
                        "query": "Where did the user move?",
                        "query_time": "2024-02-01",
                        "label": {
                            "reference_answer": "Seattle",
                            "evidence_ids": ["m1"],
                            "metadata": {},
                        },
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
