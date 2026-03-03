import json
import unittest
from unittest.mock import patch

from grpo_math.self_play.generate_pairwise_data import _openai_oracle_verify_batch
from grpo_math.trl.train_grpo_trl import _build_verdict_teacher_prompts


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestUnparseableHandling(unittest.TestCase):
    def test_oracle_verify_batch_skips_unparseable_solutions_in_prompt(self) -> None:
        captured_user_prompt = {"text": ""}

        def _fake_urlopen(req, timeout=0):  # noqa: ANN001
            body = json.loads(req.data.decode("utf-8"))
            captured_user_prompt["text"] = body["messages"][1]["content"]
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": "SOLUTION_0: CORRECT\nSOLUTION_2: INCORRECT",
                        }
                    }
                ]
            }
            return _FakeResponse(response_payload)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            verdicts, raw, err = _openai_oracle_verify_batch(
                question="What is 2+2?",
                solutions=["... FINAL_ANSWER: 4", "no parseable answer here", "... FINAL_ANSWER: 5"],
                candidate_answers=[4, None, 5],
                model="dummy-model",
                api_key="dummy-key",
                prompt_template=(
                    "Question:\n{question}\n\nSolutions:\n{solutions_block}\n\n"
                    "Candidates:\n{candidate_answers_block}\n"
                ),
            )

        self.assertIsNone(err)
        self.assertIsNotNone(raw)
        self.assertEqual(verdicts, {0: True, 2: False})
        self.assertIn("SOLUTION_0", captured_user_prompt["text"])
        self.assertIn("SOLUTION_2", captured_user_prompt["text"])
        self.assertNotIn("SOLUTION_1", captured_user_prompt["text"])

    def test_oracle_verify_batch_short_circuits_when_all_unparseable(self) -> None:
        with patch("urllib.request.urlopen") as mocked_urlopen:
            verdicts, raw, err = _openai_oracle_verify_batch(
                question="Anything",
                solutions=["bad", "still bad"],
                candidate_answers=[None, None],
                model="dummy-model",
                api_key="dummy-key",
            )
        self.assertEqual(verdicts, {})
        self.assertIsNone(raw)
        self.assertIsNone(err)
        mocked_urlopen.assert_not_called()

    def test_build_verdict_teacher_prompts_filters_unparseable(self) -> None:
        indices, prompts = _build_verdict_teacher_prompts(
            questions=["Q1", "Q2", "Q3"],
            completions=[
                "analysis...\nFINAL_ANSWER: 11",
                "analysis only, no final tag",
                "FINAL_ANSWER: -3",
            ],
            teacher_template="Question={question}\nCandidate={candidate_answer}",
        )
        self.assertEqual(indices, [0, 2])
        self.assertEqual(len(prompts), 2)
        self.assertIn("Question=Q1", prompts[0])
        self.assertIn("Candidate=analysis...\nFINAL_ANSWER: 11", prompts[0])
        self.assertIn("Question=Q3", prompts[1])
        self.assertIn("Candidate=FINAL_ANSWER: -3", prompts[1])


if __name__ == "__main__":
    unittest.main()
