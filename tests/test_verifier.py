import urllib.error
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from grpo_math.self_play.verifier import (
    CachedVerifier,
    FixtureVerifier,
    OpenAICompatibleVerifier,
    PythonAssistedVerifier,
    VerificationRequest,
    VerificationResult,
    VerifierCache,
    VerifierVerdict,
    cache_key,
    extract_candidate_final_answer,
    parse_verifier_response,
    render_verifier_prompt,
    run_python_verification_probe,
)


class CountingVerifier:
    model = "counting-model"
    prompt_version = "prompt-v1"

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, question: str, completion: str) -> VerificationResult:
        self.calls += 1
        return VerificationResult(
            verdict=VerifierVerdict.CORRECT,
            raw_text="VERDICT: CORRECT",
            model=self.model,
            prompt_version=self.prompt_version,
        )

    def verify_batch(self, requests: list[VerificationRequest]) -> list[VerificationResult]:
        return [self.verify(req.question, req.completion) for req in requests]


class TestVerifier(unittest.TestCase):
    def test_parse_verifier_response(self) -> None:
        self.assertEqual(
            parse_verifier_response("VERDICT: CORRECT\nCONFIDENCE: 0.75"),
            (VerifierVerdict.CORRECT, 0.75),
        )
        self.assertEqual(parse_verifier_response("incorrect"), (VerifierVerdict.UNCLEAR, None))
        self.assertEqual(
            parse_verifier_response("VERDICT: INVALID"),
            (VerifierVerdict.INVALID, None),
        )
        self.assertEqual(
            parse_verifier_response("VERDICT: UNCLEAR"),
            (VerifierVerdict.UNCLEAR, None),
        )
        self.assertEqual(
            parse_verifier_response("I cannot tell from this answer."),
            (VerifierVerdict.UNCLEAR, None),
        )
        self.assertEqual(
            parse_verifier_response(
                "The candidate appears incorrect at first.\n"
                "VERDICT: CORRECT\n"
                "CONFIDENCE: 1.0"
            ),
            (VerifierVerdict.CORRECT, 1.0),
        )

    def test_fixture_verifier_and_batch_api(self) -> None:
        verifier = FixtureVerifier(
            {
                ("q1", "a1"): "VERDICT: CORRECT\nCONFIDENCE: 1",
                ("q1", "a2"): VerifierVerdict.INCORRECT,
            }
        )
        result = verifier.verify("q1", "a1")
        self.assertEqual(result.verdict, VerifierVerdict.CORRECT)
        self.assertEqual(result.confidence, 1.0)

        batch = verifier.verify_batch(
            [
                VerificationRequest(question="q1", completion="a1"),
                VerificationRequest(question="q1", completion="a2"),
            ]
        )
        self.assertEqual([item.verdict for item in batch], [VerifierVerdict.CORRECT, VerifierVerdict.INCORRECT])

    def test_render_verifier_prompt_supports_candidate_answer(self) -> None:
        prompt = render_verifier_prompt(
            "Question: {question}\nSolution: {solution}\nAnswer: {candidate_answer}",
            "What is 2+2?",
            "work\nFINAL_ANSWER: 4",
        )

        self.assertIn("Answer: 4", prompt)

    def test_render_verifier_prompt_supports_non_integer_candidate_answer(self) -> None:
        prompt = render_verifier_prompt(
            "Question: {question}\nSolution: {solution}\nAnswer: {candidate_answer}",
            "What is the probability?",
            "work\nFINAL_ANSWER: 35/396",
        )

        self.assertIn("Answer: 35/396", prompt)

    def test_render_verifier_prompt_uses_last_full_final_answer(self) -> None:
        prompt = render_verifier_prompt(
            "Question: {question}\nAnswer: {candidate_answer}",
            "What is the probability?",
            "bad draft\nFINAL_ANSWER: 35\nfixed draft\nFINAL_ANSWER: 35/396",
        )

        self.assertIn("Answer: 35/396", prompt)
        self.assertNotIn("Answer: 35\n", prompt)

    def test_answer_prompt_includes_full_candidate_solution(self) -> None:
        template = Path("grpo_math/prompts/single_solution_verify_prompt.txt").read_text(encoding="utf-8")
        prompt = render_verifier_prompt(
            template,
            "What is 2 plus 2?",
            "This reasoning should not be shown.\nFINAL_ANSWER: 4",
        )

        self.assertIn("Candidate answer:\n4", prompt)
        self.assertIn("Full candidate solution:", prompt)
        self.assertIn("This reasoning should not be shown.", prompt)

    def test_extract_candidate_final_answer_uses_last_full_text_answer(self) -> None:
        self.assertEqual(
            extract_candidate_final_answer("draft\nFINAL_ANSWER: 12\nrevision\nFINAL_ANSWER: 35/396"),
            "35/396",
        )
        self.assertIsNone(extract_candidate_final_answer("no final answer"))

    def test_extract_candidate_final_answer_prefers_final_section(self) -> None:
        self.assertEqual(
            extract_candidate_final_answer(
                "[Thinking]\nTemplate says FINAL_ANSWER: <final answer>.\n\n"
                "[Final]\nFINAL_ANSWER: 3"
            ),
            "3",
        )
        self.assertIsNone(extract_candidate_final_answer("[Final]\nFINAL_ANSWER: <final answer>."))

    def test_python_probe_preserves_fraction_candidate_answer(self) -> None:
        output = run_python_verification_probe(
            "A probability question with answer 35/396.",
            "work\nFINAL_ANSWER: 35/396",
        )

        self.assertIn("candidate_final_answer = 35/396", output)
        self.assertIn("candidate_as_fraction = 35/396", output)

    def test_python_probe_enumerates_digit_sum_divisibility_question(self) -> None:
        output = run_python_verification_probe(
            "Let S be the set of all positive integers n less than 100 such that n is "
            "divisible by 7 and the sum of the decimal digits of n is also divisible by 7. "
            "How many elements are in S?",
            "work\nFINAL_ANSWER: 3",
        )

        self.assertIn("python_enumerated_values = [7, 70, 77]", output)
        self.assertIn("python_enumerated_count = 3", output)
        self.assertIn("exact_expected_answer = 3", output)
        self.assertIn("candidate_matches_exact = True", output)

    def test_cache_key_depends_on_all_required_fields(self) -> None:
        base = cache_key(question="q", completion="a", prompt_version="v1", model="m1")
        self.assertNotEqual(
            base,
            cache_key(question="q2", completion="a", prompt_version="v1", model="m1"),
        )
        self.assertNotEqual(
            base,
            cache_key(question="q", completion="a2", prompt_version="v1", model="m1"),
        )
        self.assertNotEqual(
            base,
            cache_key(question="q", completion="a", prompt_version="v2", model="m1"),
        )
        self.assertNotEqual(
            base,
            cache_key(question="q", completion="a", prompt_version="v1", model="m2"),
        )

    def test_cached_verifier_avoids_duplicate_calls(self) -> None:
        inner = CountingVerifier()
        verifier = CachedVerifier(inner, VerifierCache())

        first = verifier.verify("q", "a")
        second = verifier.verify("q", "a")

        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(second.verdict, VerifierVerdict.CORRECT)

    def test_file_cache_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "verifier_cache.jsonl"
            cache = VerifierCache(cache_path)
            key = cache_key(question="q", completion="a", prompt_version="v1", model="m1")
            cache.set(
                key,
                VerificationResult(
                    verdict=VerifierVerdict.INVALID,
                    raw_text="VERDICT: INVALID",
                    model="m1",
                    prompt_version="v1",
                ),
            )

            loaded = VerifierCache(cache_path).get(key)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.verdict, VerifierVerdict.INVALID)

    def test_openai_compatible_verifier_request_and_parse(self) -> None:
        seen: dict[str, Any] = {}

        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            seen["url"] = url
            seen["payload"] = payload
            seen["headers"] = headers
            seen["timeout_s"] = timeout_s
            return {"choices": [{"message": {"content": "VERDICT: INCORRECT\nCONFIDENCE: 0.25"}}]}

        verifier = OpenAICompatibleVerifier(
            model="gpt-test",
            api_key="test-key",
            base_url="https://example.test/v1",
            prompt_template="Question: {question}\nSolution: {solution}",
            prompt_version="prompt-v1",
            request_fn=fake_request,
        )
        result = verifier.verify("What is 2+2?", "FINAL_ANSWER: 5")

        self.assertEqual(result.verdict, VerifierVerdict.INCORRECT)
        self.assertEqual(result.confidence, 0.25)
        self.assertEqual(seen["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(seen["payload"]["max_completion_tokens"], 32)
        self.assertEqual(seen["payload"]["messages"][0]["content"], "Question: What is 2+2?\nSolution: FINAL_ANSWER: 5")

    def test_verifier_does_not_parse_hidden_reasoning_as_content(self) -> None:
        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            del url, payload, headers, timeout_s
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning": "VERDICT: CORRECT\nCONFIDENCE: 1.0",
                        }
                    }
                ]
            }

        verifier = OpenAICompatibleVerifier(
            model="qwen-test",
            api_key="test-key",
            prompt_template="{question}\n{candidate_answer}",
            prompt_version="prompt-v1",
            request_fn=fake_request,
        )

        with self.assertRaises(KeyError):
            verifier.verify("q", "FINAL_ANSWER: 1")

    def test_openai_compatible_verifier_supports_max_tokens_param_override(self) -> None:
        seen: dict[str, Any] = {}

        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            del url, headers, timeout_s
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "VERDICT: CORRECT"}}]}

        verifier = OpenAICompatibleVerifier(
            model="gemini-test",
            api_key="test-key",
            prompt_template="{question}\n{completion}",
            prompt_version="prompt-v1",
            max_tokens_param="max_tokens",
            request_fn=fake_request,
        )

        verifier.verify("q", "FINAL_ANSWER: 1")

        self.assertIn("max_tokens", seen["payload"])
        self.assertNotIn("max_completion_tokens", seen["payload"])

    def test_openai_compatible_verifier_supports_extra_body(self) -> None:
        seen: dict[str, Any] = {}

        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            del url, headers, timeout_s
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "VERDICT: CORRECT"}}]}

        verifier = OpenAICompatibleVerifier(
            model="qwen-test",
            api_key="test-key",
            prompt_template="{question}\n{candidate_answer}",
            prompt_version="prompt-v1",
            max_tokens_param="max_tokens",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            request_fn=fake_request,
        )

        verifier.verify("q", "FINAL_ANSWER: 1")

        self.assertEqual(
            seen["payload"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_openai_compatible_verifier_retries_transient_errors(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            del url, payload, headers, timeout_s
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise urllib.error.HTTPError(
                    url="https://example.test/v1/chat/completions",
                    code=429,
                    msg="rate limited",
                    hdrs=None,
                    fp=None,
                )
            return {"choices": [{"message": {"content": "VERDICT: CORRECT"}}]}

        verifier = OpenAICompatibleVerifier(
            model="gpt-test",
            api_key="test-key",
            prompt_template="{question}\n{completion}",
            prompt_version="prompt-v1",
            request_fn=fake_request,
            sleep_fn=sleeps.append,
            initial_backoff_s=0.5,
        )
        result = verifier.verify("q", "a")

        self.assertEqual(result.verdict, VerifierVerdict.CORRECT)
        self.assertEqual(attempts, 2)
        self.assertEqual(sleeps, [0.5])

    def test_python_assisted_verifier_adds_tool_result_to_completion(self) -> None:
        seen: dict[str, str] = {}

        class RecordingVerifier:
            model = "recording"
            prompt_version = "v1"

            def verify(self, question: str, completion: str) -> VerificationResult:
                seen["question"] = question
                seen["completion"] = completion
                return VerificationResult(verdict=VerifierVerdict.CORRECT, raw_text="VERDICT: CORRECT")

            def verify_batch(self, requests):
                return [self.verify(req.question, req.completion) for req in requests]

        verifier = PythonAssistedVerifier(RecordingVerifier())
        result = verifier.verify("What is 2 plus 2?", "FINAL_ANSWER: 4")

        self.assertEqual(result.verdict, VerifierVerdict.CORRECT)
        self.assertIn("Python tool result", seen["question"])
        self.assertIn("candidate_final_answer = 4", seen["question"])
        self.assertEqual(seen["completion"], "FINAL_ANSWER: 4")

    def test_integrated_answer_only_python_assisted_prompt_parses_fraction(self) -> None:
        seen: dict[str, Any] = {}
        template = Path("grpo_math/prompts/single_solution_verify_prompt.txt").read_text(encoding="utf-8")

        def fake_request(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            del url, headers, timeout_s
            seen["prompt"] = payload["messages"][0]["content"]
            return {"choices": [{"message": {"content": "VERDICT: CORRECT\nCONFIDENCE: 1.0"}}]}

        base = OpenAICompatibleVerifier(
            model="qwen/qwen3.5-9b",
            api_key="test-key",
            base_url="https://api.together.xyz/v1",
            prompt_template=template,
            prompt_version="answer-only-v1",
            max_tokens_param="max_tokens",
            request_fn=fake_request,
        )
        verifier = PythonAssistedVerifier(base)
        completion = (
            "Wrong intermediate draft.\n"
            "FINAL_ANSWER: 35\n"
            "Corrected final line.\n"
            "FINAL_ANSWER: 35/396"
        )

        result = verifier.verify("A bag has 5 red and 7 blue marbles. What is the probability?", completion)

        self.assertEqual(result.verdict, VerifierVerdict.CORRECT)
        prompt = str(seen["prompt"])
        self.assertIn("Candidate answer:\n35/396", prompt)
        self.assertIn("candidate_final_answer = 35/396", prompt)
        self.assertIn("candidate_as_fraction = 35/396", prompt)
        self.assertIn("Full candidate solution:", prompt)
        self.assertIn("Wrong intermediate draft.", prompt)


if __name__ == "__main__":
    unittest.main()

