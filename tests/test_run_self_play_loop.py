import tempfile
import unittest
from pathlib import Path

import yaml

from grpo_math.self_play.run_self_play_loop import run_loop


class TestRunSelfPlayLoop(unittest.TestCase):
    def _write_config(self, tmp: Path) -> Path:
        config = {
            "iteration": 5,
            "loop": {"iterations": 2},
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

    def test_loop_chains_next_iteration_configs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "loop"

            result = run_loop(
                config_path=str(config),
                output_dir=out,
                dry_run=True,
                overwrite=True,
            )

            self.assertEqual(result["iterations"], 2)
            self.assertEqual([record["iteration"] for record in result["records"]], [5, 6])
            self.assertTrue((out / "iteration_005" / "report.md").exists())
            self.assertTrue((out / "iteration_006" / "report.md").exists())

            first_next = yaml.safe_load(
                (out / "iteration_005" / "next_iteration.yaml").read_text(encoding="utf-8")
            )
            second_next = yaml.safe_load(
                (out / "iteration_006" / "next_iteration.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(first_next["iteration"], 6)
            self.assertEqual(second_next["iteration"], 7)

            loop_report = (out / "loop_report.md").read_text(encoding="utf-8")
            self.assertIn("Self-Play Loop", loop_report)
            self.assertIn("iteration_005", loop_report)
            self.assertIn("iteration_006", loop_report)

    def test_iterations_override_config(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            out = tmp / "loop"

            result = run_loop(
                config_path=str(config),
                output_dir=out,
                iterations=1,
                dry_run=True,
                overwrite=True,
            )

            self.assertEqual(result["iterations"], 1)
            self.assertEqual(len(result["records"]), 1)
            self.assertTrue((out / "iteration_005").exists())
            self.assertFalse((out / "iteration_006").exists())

    def test_rejects_non_positive_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = self._write_config(tmp)
            with self.assertRaises(ValueError):
                run_loop(
                    config_path=str(config),
                    output_dir=tmp / "loop",
                    iterations=0,
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
