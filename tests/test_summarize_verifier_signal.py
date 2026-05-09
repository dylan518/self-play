from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo_math.self_play.summarize_verifier_signal import summarize


class TestSummarizeVerifierSignal(unittest.TestCase):
    def test_summary_flags_zero_signal_and_majority_oracle_accuracy(self) -> None:
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "rollout.jsonl"
            oracle = Path(tmp) / "oracle.json"
            row = {
                "question_index": 0,
                "question_reward": 0.0,
                "trainable_for_proposer": True,
                "verdict_summary": {"note": "all_incorrect"},
                "solutions": [
                    {
                        "solution_index": 0,
                        "parsed_final_answer": "12",
                        "solver_advantage": 0.0,
                        "trainable_for_solver": True,
                    },
                    {
                        "solution_index": 1,
                        "parsed_final_answer": "13",
                        "solver_advantage": 0.0,
                        "trainable_for_solver": True,
                    },
                ],
                "solution_verifications": [
                    {"solution_index": 0, "verdicts": ["INCORRECT", "INCORRECT", "INCORRECT"]},
                    {"solution_index": 1, "verdicts": ["INCORRECT", "INCORRECT", "INCORRECT"]},
                ],
            }
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            oracle.write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "question_index": 0,
                                "solution_index": 0,
                                "agrees": False,
                            },
                            {
                                "question_index": 0,
                                "solution_index": 1,
                                "agrees": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize(jsonl, oracle)

        self.assertEqual(summary["majority_buckets"]["3/3"]["n"], 2)
        self.assertEqual(summary["majority_buckets"]["3/3"]["oracle_accuracy"], 0.5)
        self.assertTrue(summary["red_flags"]["all_questions_zero_reward"])
        self.assertTrue(summary["red_flags"]["all_solver_advantages_zero"])


if __name__ == "__main__":
    unittest.main()
