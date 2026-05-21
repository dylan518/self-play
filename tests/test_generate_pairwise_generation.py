import random
import unittest

from grpo_math.self_play.generate_pairwise_data import (
    ModelBundle,
    _canonicalize_solver_output,
    _generate_texts,
    _openai_generate_texts,
    _parse_question,
    _parse_question_confirm,
    _select_verify_indices,
)


class TestGeneratePairwiseGeneration(unittest.TestCase):
    def test_select_verify_indices_samples_only_parseable_solutions(self) -> None:
        indices = _select_verify_indices(
            [1, None, 2, 3, None, 4, 5, 6],
            3,
            rng=random.Random(1234),
        )

        self.assertEqual(len(indices), 3)
        self.assertEqual(indices, sorted(indices))
        self.assertTrue(set(indices).issubset({0, 2, 3, 5, 6, 7}))

    def test_canonicalize_solver_output_stops_at_final_answer(self) -> None:
        raw = "reasoning\nFINAL_ANSWER: 46\nmore tokens 123"

        self.assertEqual(_canonicalize_solver_output(raw), "reasoning\nFINAL_ANSWER: 46")

    def test_parse_question_strips_confirm_and_reads_confirmation(self) -> None:
        raw = (
            "QUESTION: Find the smallest n divisible by 7.\n"
            "CONFIRM: different_from_samples=yes; integer_answer=yes; why=uses modular arithmetic."
        )

        self.assertEqual(_parse_question(raw), "Find the smallest n divisible by 7.")
        self.assertEqual(
            _parse_question_confirm(raw),
            (True, "different_from_samples=yes; integer_answer=yes; why=uses modular arithmetic."),
        )

    def test_parse_question_confirm_rejects_self_correction(self) -> None:
        raw = (
            "QUESTION: What is 1+1?\n"
            "CONFIRM: different_from_samples=yes; integer_answer=yes; why=ok. "
            "Wait, better question: What is 2+2?"
        )

        self.assertEqual(_parse_question_confirm(raw)[0], False)

    def test_zero_temperature_uses_greedy_decoding(self) -> None:
        class FakeTensor:
            shape = (1, 2)

            def to(self, _device):
                return self

            def __getitem__(self, _key):
                return self

            def __iter__(self):
                return iter([self])

        class FakeParameter:
            device = "cpu"

        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = None

            def parameters(self):
                return iter([FakeParameter()])

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeTensor()

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, _prompts, return_tensors, padding, truncation):
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

            def decode(self, _row, skip_special_tokens):
                return "VERDICT: CORRECT"

        model = FakeModel()

        outputs = _generate_texts(
            ModelBundle(model=model, tokenizer=FakeTokenizer()),
            ["prompt"],
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=8,
            batch_size=1,
        )

        self.assertEqual(outputs, ["VERDICT: CORRECT"])
        self.assertFalse(model.kwargs["do_sample"])
        self.assertNotIn("temperature", model.kwargs)
        self.assertNotIn("top_p", model.kwargs)

    def test_qwen_two_step_generation_closes_thinking_before_final_answer(self) -> None:
        import json
        import urllib.request

        calls: list[tuple[str, dict]] = []

        def fake_urlopen(req, timeout):  # noqa: ANN001
            del timeout
            payload = json.loads(req.data.decode("utf-8"))
            calls.append((req.full_url, payload))

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    if req.full_url.endswith("/chat/completions"):
                        return json.dumps(
                            {
                                "choices": [
                                    {
                                        "message": {
                                            "reasoning": "Compute 1 + 1.",
                                            "content": "",
                                        }
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    return json.dumps({"choices": [{"text": "2"}]}).encode("utf-8")

            return Response()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            outputs = _openai_generate_texts(
                prompts=["Question: 1+1?"],
                model="Qwen/Qwen3-4B",
                api_key="test",
                base_url="https://example.test/v1",
                temperature=0.2,
                top_p=1.0,
                max_completion_tokens=4096,
                timeout_s=10,
                max_tokens_param="max_tokens",
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}},
                include_reasoning=True,
                tokenizer_name_or_path="Qwen/Qwen3-4B",
                final_max_tokens=1024,
                final_prefill="FINAL_ANSWER: ",
            )
        finally:
            urllib.request.urlopen = original

        self.assertEqual(outputs, ["[Thinking]\nCompute 1 + 1.\n\n[Final]\nFINAL_ANSWER: 2"])
        self.assertEqual(calls[0][1]["max_tokens"], 512)
        self.assertEqual(calls[0][1]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertTrue(calls[1][0].endswith("/completions"))
        self.assertEqual(calls[1][1]["max_tokens"], 1024)
        self.assertIn("</think>\n\nFINAL_ANSWER:", calls[1][1]["prompt"])
        self.assertEqual(calls[1][1]["stop"], ["\n"])

    def test_qwen_two_step_generation_uses_content_as_reasoning_for_vllm(self) -> None:
        import json
        import urllib.request

        calls: list[tuple[str, dict]] = []

        def fake_urlopen(req, timeout):  # noqa: ANN001
            del timeout
            payload = json.loads(req.data.decode("utf-8"))
            calls.append((req.full_url, payload))

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    if req.full_url.endswith("/chat/completions"):
                        return json.dumps(
                            {
                                "choices": [
                                    {
                                        "message": {
                                            "content": "Work out 3 + 4 carefully.",
                                        }
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    return json.dumps({"choices": [{"text": "7"}]}).encode("utf-8")

            return Response()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            outputs = _openai_generate_texts(
                prompts=["Question: 3+4?"],
                model="Qwen/Qwen3.5-9B",
                api_key="test",
                base_url="https://example.test/v1",
                temperature=0.2,
                top_p=1.0,
                max_completion_tokens=4096,
                timeout_s=10,
                max_tokens_param="max_tokens",
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}},
                include_reasoning=True,
                tokenizer_name_or_path="Qwen/Qwen3.5-9B",
                final_max_tokens=32,
                final_prefill="FINAL_ANSWER:",
            )
        finally:
            urllib.request.urlopen = original

        self.assertEqual(outputs, ["[Thinking]\nWork out 3 + 4 carefully.\n\n[Final]\nFINAL_ANSWER: 7"])
        self.assertEqual(len(calls), 2)
        self.assertIn("<think>\nWork out 3 + 4 carefully.\n</think>", calls[1][1]["prompt"])
        self.assertEqual(calls[1][1]["stop"], ["\n"])


if __name__ == "__main__":
    unittest.main()
