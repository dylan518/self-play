import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from grpo_math.self_play.verifier import (
    CachedVerifier,
    VerificationResult,
    VerifierCache,
    VerifierVerdict,
)
from grpo_math.trl.train_solver_verdict_grpo import (
    build_reward_functions,
    format_reward,
    load_generated_questions,
    load_yaml,
    parse_verdict,
    validate_config,
)


PROMPT_TEMPLATE = "Question:\n{question}\nFINAL_ANSWER required."


class CountingVerifier:
    model = "fixture"
    prompt_version = "prompt-v1"

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, question: str, completion: str) -> VerificationResult:
        del question
        self.calls += 1
        verdict = VerifierVerdict.CORRECT if completion == "FINAL_ANSWER: 4" else VerifierVerdict.INCORRECT
        return VerificationResult(
            verdict=verdict,
            raw_text=f"VERDICT: {verdict.value}",
            model=self.model,
            prompt_version=self.prompt_version,
        )

    def verify_batch(self, requests):
        return [self.verify(req.question, req.completion) for req in requests]


class TestTrainSolverVerdictGrpo(unittest.TestCase):
    def test_load_generated_questions_keeps_only_solver_trainable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            rows = [
                {"question_index": 7, "question": "What is 2 + 2?", "trainable_for_solver": True},
                {"question_index": 8, "question": "What is 3 + 3?", "trainable_for_solver": False},
                {"question_index": 9, "question": "What is 4 + 4?", "filter_result": {"accepted": False}},
                {"question_index": 10, "question": "What is 5 + 5?"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            examples = load_generated_questions([path], prompt_template=PROMPT_TEMPLATE)

        self.assertEqual([example.question for example in examples], ["What is 2 + 2?", "What is 5 + 5?"])
        self.assertEqual(examples[0].source_index, 7)
        self.assertIn("What is 2 + 2?", examples[0].prompt)

    def test_verdict_parser_and_format_reward(self) -> None:
        self.assertEqual(parse_verdict("VERDICT: CORRECT"), "CORRECT")
        self.assertEqual(parse_verdict("cannot tell"), "UNCLEAR")
        self.assertEqual(format_reward("work\nFINAL_ANSWER: 4"), 1.0)
        self.assertEqual(format_reward("answer is 4"), 0.0)

    def test_reward_functions_use_cached_verifier_and_ignore_format_metric_weighting(self) -> None:
        inner = CountingVerifier()
        verifier = CachedVerifier(inner, VerifierCache())
        reward_verdict, reward_format = build_reward_functions(verifier)

        first = reward_verdict(
            prompts=["p", "p"],
            completions=["FINAL_ANSWER: 4", "FINAL_ANSWER: 5"],
            question=["What is 2 + 2?", "What is 2 + 2?"],
        )
        second = reward_verdict(
            prompts=["p"],
            completions=["FINAL_ANSWER: 4"],
            question=["What is 2 + 2?"],
        )
        format_scores = reward_format(
            prompts=["p", "p"],
            completions=["FINAL_ANSWER: 4", "no final answer"],
        )

        self.assertEqual(first, [1.0, 0.0])
        self.assertEqual(second, [1.0])
        self.assertEqual(format_scores, [1.0, 0.0])
        self.assertEqual(inner.calls, 2)

    def test_file_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.jsonl"
            inner = CountingVerifier()
            verifier = CachedVerifier(inner, VerifierCache(cache_path))
            self.assertEqual(
                verifier.verify("What is 2 + 2?", "FINAL_ANSWER: 4").verdict,
                VerifierVerdict.CORRECT,
            )

            verifier_reloaded = CachedVerifier(inner, VerifierCache(cache_path))
            cached = verifier_reloaded.verify("What is 2 + 2?", "FINAL_ANSWER: 4")

        self.assertTrue(cached.cached)
        self.assertEqual(inner.calls, 1)

    def test_validate_smoke_config(self) -> None:
        cfg = load_yaml("grpo_math/configs/train_solver_verdict_grpo_smoke.yaml")
        validate_config(cfg)

    def test_validate_only_cli_generates_server_command(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpo_math.trl.train_solver_verdict_grpo",
                "--config",
                "grpo_math/configs/train_solver_verdict_grpo_smoke.yaml",
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Config valid. Server command:", result.stdout)
        self.assertIn("python -m grpo_math.trl.train_solver_verdict_grpo", result.stdout)


if __name__ == "__main__":
    unittest.main()
