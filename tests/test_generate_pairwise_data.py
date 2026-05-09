from __future__ import annotations

import unittest

from grpo_math.self_play.generate_pairwise_data import (
    _append_python_tool_result,
    _build_qwen_continuation_prompt,
    _extract_chat_reasoning,
    _extract_chat_content,
    _extract_completion_text,
    _extract_configured_thinking_budget,
    _execute_python_tool_code,
    _openai_generate_texts,
    _PYTHON_VERIFY_TOOL,
    _parse_question,
    _parse_solver_final_answer,
    _parse_verdict,
)


class TestGeneratePairwiseDataHelpers(unittest.TestCase):
    def test_parse_question_strips_leaked_verification_idea(self) -> None:
        self.assertEqual(
            _parse_question(
                "QUESTION: How many integers n satisfy n^2 < 100?\n"
                "VERIFICATION IDEA: Enumerate n with Python."
            ),
            "How many integers n satisfy n^2 < 100?",
        )

    def test_parse_solver_final_answer_accepts_non_integer_text(self) -> None:
        self.assertEqual(_parse_solver_final_answer("work\nFINAL_ANSWER: 85/1728"), "85/1728")
        self.assertEqual(
            _parse_solver_final_answer("work\nFINAL_ANSWER: Alice, Dana, Bob, Charlie"),
            "Alice, Dana, Bob, Charlie",
        )

    def test_parse_solver_final_answer_uses_final_section_not_thinking_placeholder(self) -> None:
        self.assertEqual(
            _parse_solver_final_answer(
                "[Thinking]\nMust end with FINAL_ANSWER: <final answer>.\n\n"
                "[Final]\nWork.\nFINAL_ANSWER: 1367044"
            ),
            "1367044",
        )
        self.assertIsNone(_parse_solver_final_answer("[Final]\nFINAL_ANSWER: <final answer>."))

    def test_parse_solver_final_answer_recovers_simple_final_answer_sentence(self) -> None:
        self.assertEqual(
            _parse_solver_final_answer(
                "[Thinking]\nMaybe the answer is 99.\n\n"
                "[Final]\nSince n=1 works, the answer is 1."
            ),
            "1",
        )

    def test_parse_verdict_missing_label_is_unclear(self) -> None:
        self.assertEqual(_parse_verdict("REASONING: truncated before label"), "UNCLEAR")
        self.assertEqual(
            _parse_verdict("This seems incorrect in the reasoning.\nVERDICT: CORRECT"),
            "CORRECT",
        )

    def test_append_python_tool_result(self) -> None:
        prompt = _append_python_tool_result(
            "Verifier prompt",
            question="What is 2 plus 2?",
            solution="FINAL_ANSWER: 4",
            enabled=True,
            timeout_s=5,
        )

        self.assertIn("Python tool result:", prompt)
        self.assertIn("candidate_final_answer = 4", prompt)

    def test_append_python_tool_result_preserves_fraction_answer(self) -> None:
        prompt = _append_python_tool_result(
            "Verifier prompt\nCandidate answer:\n35/396",
            question="A probability question.",
            solution="work\nFINAL_ANSWER: 35/396",
            enabled=True,
            timeout_s=5,
        )

        self.assertIn("Candidate answer:\n35/396", prompt)
        self.assertIn("candidate_final_answer = 35/396", prompt)
        self.assertIn("candidate_as_fraction = 35/396", prompt)

    def test_python_exec_tool_schema_accepts_code(self) -> None:
        self.assertEqual(_PYTHON_VERIFY_TOOL["function"]["name"], "python_exec")
        params = _PYTHON_VERIFY_TOOL["function"]["parameters"]
        self.assertEqual(params["required"], ["code"])
        self.assertIn("code", params["properties"])

    def test_execute_python_tool_code_returns_stdout(self) -> None:
        result = _execute_python_tool_code("print(2 + 2)", timeout_s=5)

        self.assertIn("exit_code = 0", result)
        self.assertIn("stdout:\n4", result)

    def test_extract_chat_content_does_not_use_hidden_reasoning_by_default(self) -> None:
        with self.assertRaises(KeyError):
            _extract_chat_content(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "QUESTION: hidden reasoning is not final content",
                            }
                        }
                    ]
                }
            )

    def test_extract_chat_content_can_include_reasoning_for_solver_outputs(self) -> None:
        text = _extract_chat_content(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning": "I compute the value.",
                            "content": "Work summary.\nFINAL_ANSWER: 42",
                        }
                    }
                ]
            },
            include_reasoning=True,
        )

        self.assertIn("[Thinking]\nI compute the value.", text)
        self.assertIn("[Final]\nWork summary.", text)
        self.assertEqual(_parse_solver_final_answer(text), "42")

    def test_extract_chat_content_can_store_reasoning_only_for_inspection(self) -> None:
        text = _extract_chat_content(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning": "Thinking without final content.",
                            "content": "",
                        }
                    }
                ]
            },
            include_reasoning=True,
        )

        self.assertEqual(text, "[Thinking]\nThinking without final content.")
        self.assertIsNone(_parse_solver_final_answer(text))

    def test_extract_chat_reasoning_supports_qwen_field_names(self) -> None:
        self.assertEqual(
            _extract_chat_reasoning({"choices": [{"message": {"reasoning_content": "first"}}]}),
            "first",
        )
        self.assertEqual(
            _extract_chat_reasoning({"choices": [{"message": {"reasoning": "second"}}]}),
            "second",
        )

    def test_extract_completion_text_for_qwen_continuation(self) -> None:
        self.assertEqual(
            _extract_completion_text({"choices": [{"text": "FINAL_ANSWER: 42\n"}]}),
            "FINAL_ANSWER: 42",
        )

    def test_extract_configured_thinking_budget(self) -> None:
        self.assertEqual(
            _extract_configured_thinking_budget(
                {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": "512"}}
            ),
            512,
        )
        self.assertEqual(
            _extract_configured_thinking_budget({"chat_template_kwargs": {"max_thinking_tokens": 256}}),
            256,
        )
        self.assertEqual(_extract_configured_thinking_budget({"chat_template_kwargs": {}}), 0)

    def test_qwen_continuation_prompt_places_reasoning_inside_think_block(self) -> None:
        prompt = _build_qwen_continuation_prompt(
            prompt="Solve 1+1.",
            reasoning="The answer is 2.",
            tokenizer_name_or_path="Qwen/Qwen3.5-9B",
        )

        self.assertIn("<think>\nThe answer is 2.\n</think>", prompt)
        self.assertTrue(prompt.rstrip().endswith("</think>"))

    def test_qwen_continuation_prompt_can_prefill_final_answer_tag(self) -> None:
        prompt = _build_qwen_continuation_prompt(
            prompt="Solve 1+1.",
            reasoning="The answer is 2.",
            tokenizer_name_or_path="Qwen/Qwen3.5-9B",
            final_prefill="FINAL_ANSWER: ",
        )

        self.assertIn("<think>\nThe answer is 2.\n</think>\n\nFINAL_ANSWER:", prompt)
        self.assertTrue(prompt.endswith("FINAL_ANSWER:"))

    def test_two_step_qwen_final_tokens_can_be_capped(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_urlopen(req, timeout):  # noqa: ANN001
            import json

            calls.append((req.full_url, json.loads(req.data.decode("utf-8"))))

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
                                            "reasoning": "Compute directly.",
                                            "content": "",
                                        }
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    return json.dumps(
                        {"choices": [{"text": "FINAL_ANSWER: 2"}]}
                    ).encode("utf-8")

            return Response()

        import grpo_math.self_play.generate_pairwise_data as mod

        original = mod.urllib.request.urlopen
        mod.urllib.request.urlopen = fake_urlopen
        try:
            outputs = _openai_generate_texts(
                prompts=["What is 1+1?"],
                model="Qwen/Qwen3.5-9B",
                api_key="test",
                base_url="https://example.test/v1",
                temperature=0.1,
                top_p=0.9,
                max_completion_tokens=4096,
                timeout_s=10,
                max_tokens_param="max_tokens",
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}},
                include_reasoning=True,
                tokenizer_name_or_path="Qwen/Qwen3.5-9B",
                final_max_tokens=2048,
                final_prefill="FINAL_ANSWER: ",
            )
        finally:
            mod.urllib.request.urlopen = original

        self.assertEqual(outputs[0], "[Thinking]\nCompute directly.\n\n[Final]\nFINAL_ANSWER: 2")
        self.assertEqual(calls[0][1]["max_tokens"], 512)
        self.assertEqual(calls[1][1]["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
