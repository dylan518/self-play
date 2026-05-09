import json
import tempfile
import unittest
from pathlib import Path

import yaml

from grpo_math.self_play.run_self_play_iteration import run_iteration


class TestRunSelfPlayIteration(unittest.TestCase):
    def _write_config(self, tmp: Path) -> Path:
        config = {
            "iteration": 3,
            "rollout": {
                "enabled": True,
                "module": "example.rollout",
                "config": "configs/rollout.yaml",
            },
            "solver_train": {
                "enabled": True,
                "module": "example.solver_train",
                "config": "configs/solver.yaml",
            },
            "proposer_train": {
                "enabled": True,
                "module": "example.proposer_train",
                "config": "configs/proposer.yaml",
            },
            "evaluation": {
                "enabled": True,
                "module": "example.eval",
                "config": "configs/eval.yaml",
            },
        }
        path = tmp / "self_play.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_dry_run_writes_report_commands_and_next_config(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "iter"

            result = run_iteration(
                config_path=str(config),
                output_dir=out,
                dry_run=True,
                overwrite=True,
            )

            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual([r["status"] for r in result["records"]], ["planned"] * 4)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Self-Play Iteration 3", report)
            self.assertIn("Exact Wash U Commands", report)
            self.assertIn("example.rollout", report)

            next_cfg = yaml.safe_load((out / "next_iteration.yaml").read_text(encoding="utf-8"))
            self.assertEqual(next_cfg["iteration"], 4)
            self.assertEqual(next_cfg["resume_from"], str(out))
            self.assertEqual(next_cfg["initial_checkpoints"]["solver"], str(out / "solver"))

            rows = [
                json.loads(line)
                for line in (out / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["stage"] for row in rows], ["rollout", "solver_grpo", "proposer_grpo", "evaluation"])
            self.assertTrue(all(row["status"] == "planned" for row in rows))
            self.assertIn("--overwrite", rows[0]["command"])
            self.assertNotIn("--output-dir", rows[0]["command"])
            self.assertIn("--output_dir", rows[1]["command"])
            self.assertIn("--output_dir", rows[2]["command"])
            self.assertNotIn("--output-dir", rows[3]["command"])

    def test_skip_train_marks_train_stages_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "iter"

            result = run_iteration(
                config_path=str(config),
                output_dir=out,
                dry_run=True,
                skip_train=True,
                overwrite=True,
            )

            statuses = {record["stage"]: record["status"] for record in result["records"]}
            self.assertEqual(statuses["rollout"], "planned")
            self.assertEqual(statuses["solver_grpo"], "skipped")
            self.assertEqual(statuses["proposer_grpo"], "skipped")
            self.assertEqual(statuses["evaluation"], "planned")

    def test_existing_output_requires_overwrite_or_resume(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "iter"
            out.mkdir()
            (out / "existing.txt").write_text("keep me", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_iteration(config_path=str(config), output_dir=out, dry_run=True)

            result = run_iteration(
                config_path=str(config),
                output_dir=out,
                dry_run=True,
                resume=True,
            )
            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual((out / "existing.txt").read_text(encoding="utf-8"), "keep me")

    def test_validate_only_plans_without_requiring_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "iter"
            out.mkdir()
            (out / "existing.txt").write_text("keep me", encoding="utf-8")

            result = run_iteration(config_path=str(config), output_dir=out, validate_only=True)

            self.assertEqual(result["mode"], "validate-only")
            self.assertEqual([r["status"] for r in result["records"]], ["planned"] * 4)


if __name__ == "__main__":
    unittest.main()
