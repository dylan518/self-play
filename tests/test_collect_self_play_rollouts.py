from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo_math.self_play.collect_self_play_rollouts import (
    FixtureProposerProvider,
    FilterResult,
    ProposedQuestion,
    SolverCompletion,
    VerificationResult,
    collect_rollouts,
    deterministic_question_filter,
    run_from_config,
)


class CountingSolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def solve(self, question: ProposedQuestion, *, num_completions: int) -> list[SolverCompletion]:
        self.calls.append(question.question_id)
        return [
            SolverCompletion(f"{question.question_id}_c000", "work\nFINAL_ANSWER: 4"),
            SolverCompletion(f"{question.question_id}_c001", "work\nFINAL_ANSWER: 5"),
        ][:num_completions]


class CountingVerifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify(self, question: str, completions: list[SolverCompletion]) -> list[VerificationResult]:
        self.calls.append(question)
        return [
            VerificationResult("CORRECT"),
            VerificationResult("INCORRECT"),
        ][: len(completions)]


class CollectSelfPlayRolloutsTest(unittest.TestCase):
    def test_filter_rejects_duplicates_and_answer_leaks(self) -> None:
        seen: set[str] = set()
        first = deterministic_question_filter("What is 2 plus 2?", seen)
        self.assertEqual(first, FilterResult(True, "accepted"))
        seen.add("what is 2 plus 2?")

        duplicate = deterministic_question_filter(" What is 2 plus 2? ", seen)
        self.assertEqual(duplicate, FilterResult(False, "duplicate"))

        leaked = deterministic_question_filter("ANSWER: 7. What is 3 plus 4?", set())
        self.assertEqual(leaked, FilterResult(False, "answer_leak"))

    def test_collects_rollouts_and_skips_solver_for_filtered_questions(self) -> None:
        proposer = FixtureProposerProvider(
            {
                "prompts": [
                    {
                        "id": "prompt_a",
                        "prompt": "make questions",
                        "questions": [
                            "What is 2 plus 2?",
                            "What is 2 plus 2?",
                            "ANSWER: 6. What is 3 plus 3?",
                        ],
                    }
                ]
            }
        )
        solver = CountingSolver()
        verifier = CountingVerifier()

        rows, rejected = collect_rollouts(
            {
                "rollout": {"iteration_id": "test_iter", "num_solver_completions": 2},
                "output": {"include_summary": False},
            },
            proposer,
            solver,
            verifier,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(solver.calls, ["prompt_a_q000"])
        self.assertEqual(len(verifier.calls), 1)

        accepted = rows[0]
        self.assertTrue(accepted["trainable_for_solver"])
        self.assertTrue(accepted["trainable_for_proposer"])
        self.assertEqual(accepted["question_reward"], 1.0)
        self.assertEqual([c["reward"] for c in accepted["solver_completions"]], [1.0, 0.0])
        self.assertEqual([c["advantage"] for c in accepted["solver_completions"]], [0.5, -0.5])

        duplicate = rows[1]
        self.assertFalse(duplicate["trainable_for_solver"])
        self.assertTrue(duplicate["trainable_for_proposer"])
        self.assertEqual(duplicate["question_reward"], 0.0)
        self.assertEqual(duplicate["solver_completions"], [])

        prompt_advantages = [row["question_advantage"] for row in rows]
        for actual, expected in zip(prompt_advantages, [2 / 3, -1 / 3, -1 / 3], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_fixture_config_writes_rollout_and_rejected_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "fixture.yaml"
            rollout_path = tmp_path / "rollouts.jsonl"
            rejected_path = tmp_path / "rejected_questions.jsonl"
            config_path.write_text(
                f"""
provider_type: fixture
rollout:
  iteration_id: test_fixture
  num_solver_completions: 2
fixture:
  proposer:
    prompts:
      - id: p
        prompt: make questions
        questions:
          - What is 10 minus 6?
          - What is 10 minus 6?
  solver:
    default_completions:
      - FINAL_ANSWER: 4
      - FINAL_ANSWER: 5
  verifier:
    default_verdicts:
      - CORRECT
      - INCORRECT
output:
  rollout_jsonl_path: {rollout_path}
  rejected_jsonl_path: {rejected_path}
  overwrite: true
  include_summary: false
""",
                encoding="utf-8",
            )

            written_rollouts, written_rejected = run_from_config(config_path)

            self.assertEqual(written_rollouts, rollout_path)
            self.assertEqual(written_rejected, rejected_path)
            rollout_rows = [
                json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines()
            ]
            rejected_rows = [
                json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rollout_rows), 2)
            self.assertEqual(len(rejected_rows), 1)
            self.assertEqual(rollout_rows[0]["schema_version"], "self_play_verdict_grpo_v1")
            self.assertEqual(rejected_rows[0]["question_reward"], 0.0)

    def test_output_dir_overrides_configured_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "fixture.yaml"
            configured_path = tmp_path / "configured" / "rollouts.jsonl"
            output_dir = tmp_path / "override"
            config_path.write_text(
                f"""
provider_type: fixture
rollout:
  iteration_id: test_fixture
  num_solver_completions: 1
fixture:
  proposer:
    prompts:
      - id: p
        prompt: make questions
        questions:
          - What is 2 plus 3?
  solver:
    default_completions:
      - FINAL_ANSWER: 5
  verifier:
    default_verdicts:
      - CORRECT
output:
  rollout_jsonl_path: {configured_path}
  overwrite: true
  include_summary: false
""",
                encoding="utf-8",
            )

            written_rollouts, written_rejected = run_from_config(config_path, output_dir=output_dir)

            self.assertEqual(written_rollouts, output_dir / "rollouts.jsonl")
            self.assertEqual(written_rejected, output_dir / "rejected_questions.jsonl")
            self.assertTrue(written_rollouts.exists())
            self.assertFalse(configured_path.exists())


if __name__ == "__main__":
    unittest.main()
