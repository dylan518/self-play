import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from grpo_math.self_play import run_self_play_grpo_loop as loop


class TestRunSelfPlayGrpoLoop(unittest.TestCase):
    def test_rollout_reliability_stats_include_question_reward_components(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "samples.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"question_reward": 0.5, "question_rewards": {"solver_correctness_probability": 0.25, "difficulty_balance": 0.5, "verifiability": 0.8, "lambda_verifiability": 0.35}}',
                        '{"question_reward": 1.0, "question_rewards": {"solver_correctness_probability": 0.5, "difficulty_balance": 1.0, "verifiability": 1.0, "lambda_verifiability": 0.35}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stats = loop._infer_rollout_reliability_stats(path)

            self.assertAlmostEqual(stats["rollout/question_reward/mean"], 0.75)
            self.assertAlmostEqual(stats["rollout/question_reward/min"], 0.5)
            self.assertAlmostEqual(stats["rollout/question_reward/max"], 1.0)
            self.assertAlmostEqual(stats["rollout/question_reward/nonzero_frac"], 1.0)
            self.assertAlmostEqual(
                stats["rollout/question_reward/solver_correctness_probability_mean"],
                0.375,
            )
            self.assertAlmostEqual(stats["rollout/question_reward/difficulty_balance_mean"], 0.75)
            self.assertAlmostEqual(stats["rollout/question_reward/verifiability_mean"], 0.9)
            self.assertAlmostEqual(stats["rollout/question_reward/lambda_verifiability_mean"], 0.35)

    def test_train_rollout_max_new_tokens_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rollout_cfg = tmp / "rollout.yaml"
            train_cfg = tmp / "train.yaml"

            rollout_cfg.write_text(
                yaml.safe_dump(
                    {
                        "solver": {"max_new_tokens": 1024},
                        "output": {"jsonl_path": "unused.jsonl"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            train_cfg.write_text(
                yaml.safe_dump(
                    {
                        "data": {"eval_fraction": 0.1},
                        "rollout": {"max_new_tokens": 64},
                        "train": {"wandb": {"enabled": False}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            commands: list[list[str]] = []

            def fake_run_timed(cmd, *, cwd, env):
                commands.append(list(cmd))
                return 0.0

            def fake_run_timed_with_gpu(cmd, *, cwd, env, gpu_indices, sample_every_s=2.0):
                commands.append(list(cmd))
                if "generate_pairwise_data" in cmd:
                    cycle_jsonl = tmp / "outputs" / "self_play_grpo_loop" / "unit" / "cycle_001_samples.jsonl"
                    cycle_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    cycle_jsonl.write_text('{"question": "What is 1+1?", "solutions": []}\n', encoding="utf-8")
                return 0.0, {}

            with mock.patch.object(loop, "_repo_root", return_value=tmp), mock.patch.object(
                loop, "_run_timed", side_effect=fake_run_timed
            ), mock.patch.object(
                loop, "_run_timed_with_gpu_sampling", side_effect=fake_run_timed_with_gpu
            ):
                loop.main_args = None
                with mock.patch(
                    "sys.argv",
                    [
                        "run_self_play_grpo_loop",
                        "--rollout_config",
                        str(rollout_cfg),
                        "--train_config",
                        str(train_cfg),
                        "--cycles",
                        "1",
                        "--run_tag",
                        "unit",
                        "--max_steps",
                        "1",
                    ],
                ):
                    loop.main()

            emitted_train_cfg = yaml.safe_load(
                (tmp / "outputs" / "self_play_grpo_loop" / "unit" / "configs" / "cycle_001_train.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(emitted_train_cfg["rollout"]["max_new_tokens"], 64)
            self.assertTrue(any("grpo_math.trl.train_grpo_trl" in cmd for cmd in commands))

    def test_train_rollout_max_new_tokens_can_inherit_solver_value(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rollout_cfg = tmp / "rollout.yaml"
            train_cfg = tmp / "train.yaml"

            rollout_cfg.write_text(
                yaml.safe_dump(
                    {
                        "solver": {"max_new_tokens": 512},
                        "output": {"jsonl_path": "unused.jsonl"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            train_cfg.write_text(
                yaml.safe_dump(
                    {
                        "data": {"eval_fraction": 0.1},
                        "rollout": {"max_new_tokens": 64, "inherit_solver_max_new_tokens": True},
                        "train": {"wandb": {"enabled": False}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            def fake_run_timed(*args, **kwargs):
                return 0.0

            def fake_run_timed_with_gpu(cmd, *, cwd, env, gpu_indices, sample_every_s=2.0):
                if "generate_pairwise_data" in cmd:
                    cycle_jsonl = tmp / "outputs" / "self_play_grpo_loop" / "unit" / "cycle_001_samples.jsonl"
                    cycle_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    cycle_jsonl.write_text('{"question": "What is 1+1?", "solutions": []}\n', encoding="utf-8")
                return 0.0, {}

            with mock.patch.object(loop, "_repo_root", return_value=tmp), mock.patch.object(
                loop, "_run_timed", side_effect=fake_run_timed
            ), mock.patch.object(
                loop, "_run_timed_with_gpu_sampling", side_effect=fake_run_timed_with_gpu
            ), mock.patch(
                "sys.argv",
                [
                    "run_self_play_grpo_loop",
                    "--rollout_config",
                    str(rollout_cfg),
                    "--train_config",
                    str(train_cfg),
                    "--cycles",
                    "1",
                    "--run_tag",
                    "unit",
                    "--max_steps",
                    "1",
                ],
            ):
                loop.main()

            emitted_train_cfg = yaml.safe_load(
                (tmp / "outputs" / "self_play_grpo_loop" / "unit" / "configs" / "cycle_001_train.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(emitted_train_cfg["rollout"]["max_new_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
