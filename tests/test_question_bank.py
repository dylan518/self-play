from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo_math.self_play.generate_example_question_bank import (
    _build_prompt,
    _make_diversity_nonce,
    _normalize_row,
    _row_to_question_bank_example,
)
from grpo_math.self_play.question_bank import (
    is_current_loop_compatible,
    load_question_bank,
    load_recent_generated_questions,
    render_question_bank_block,
    render_question_bank_block_from_config,
    select_question_bank_examples,
)


class TestQuestionBank(unittest.TestCase):
    def test_loads_categories_and_extended_tasks(self) -> None:
        examples = load_question_bank("examples.json")

        self.assertGreaterEqual(len(examples), 20)
        self.assertIn("algebra", {example.category for example in examples})
        self.assertTrue(any(example.question.startswith("Find all real solutions") for example in examples))

    def test_filters_to_current_integer_answer_contract(self) -> None:
        examples = load_question_bank("examples.json")
        compatible = [example for example in examples if is_current_loop_compatible(example)]

        self.assertTrue(compatible)
        self.assertFalse(any("Write a function" in example.question for example in compatible))
        self.assertFalse(any(example.category == "constrained_poetry" for example in compatible))

    def test_selects_deterministically_from_full_bank(self) -> None:
        examples = load_question_bank("examples.json")

        first = select_question_bank_examples(
            examples, count=5, seed=1234, compatible_only=False
        )
        second = select_question_bank_examples(
            examples, count=5, seed=1234, compatible_only=False
        )

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 5)
        self.assertEqual(len({example.question for example in first}), len(first))

    def test_renders_prompt_block(self) -> None:
        selected = select_question_bank_examples(
            load_question_bank("examples.json"), count=3, seed=1, compatible_only=False
        )
        block = render_question_bank_block(selected)

        self.assertIn("Question bank examples", block)
        self.assertIn("Generate something novel", block)
        self.assertIn('"question":', block)
        self.assertIn('"verification":', block)
        self.assertEqual(block.count('"question":'), 3)
        self.assertNotIn("Category:", block)
        self.assertNotIn("Task:", block)

    def test_loads_recent_generated_questions_from_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"question": "How many subsets of {1,2,3} have size 2?"}),
                        json.dumps(
                            {
                                "question": (
                                    "Find the smallest n such that n^2 > 50. "
                                    "VERIFICATION IDEA: brute force"
                                )
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            examples = load_recent_generated_questions(path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].category, "recent_generated")
        self.assertIn("Avoid repeating", examples[0].verification)
        self.assertIn("different_from_samples=no", examples[0].confirm)
        self.assertNotIn("VERIFICATION IDEA", examples[1].question)

    def test_renders_from_config(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            recent_path = Path(tmp) / "recent.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "tasks_extended": [
                            {
                                "category": "algebra",
                                "question": "How many integers x from 1 through 5 are odd?",
                                "verification": "Enumerate with Python",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            recent_path.write_text(
                json.dumps({"question": "Avoid this old generated question."}) + "\n",
                encoding="utf-8",
            )

            block = render_question_bank_block_from_config(
                {
                    "question_bank": {
                        "enabled": True,
                        "path": str(path),
                        "num_examples": 1,
                        "recent_jsonl_path": str(recent_path),
                        "recent_num_examples": 1,
                    }
                },
                seed=0,
            )

        self.assertIn("How many integers", block)
        self.assertIn("Avoid this old generated question", block)
        self.assertNotIn("Recent generated questions to avoid", block)
        self.assertNotIn("Original inspiration examples", block)

    def test_generation_prompt_prioritizes_diversity(self) -> None:
        selected = select_question_bank_examples(
            load_question_bank("examples.json"), count=2, seed=3, compatible_only=False
        )
        recent = [
            _row_to_question_bank_example(
                {
                    "question": "Find the smallest positive integer with a repeated digit pattern.",
                    "verification": "Brute force over integers.",
                }
            )
        ]

        prompt = _build_prompt(selected, recent, diversity_nonce="abc123xy")

        self.assertTrue(prompt.startswith("Diversity nonce: abc123xy"))
        self.assertIn("infer the broad dataset goal", prompt)
        self.assertIn("recent generated examples as things to move away from immediately", prompt)
        self.assertIn("completely different from the examples below", prompt)
        self.assertIn("Seed examples:", prompt)
        self.assertIn("Recent generated examples:", prompt)
        self.assertIn(
            '{"question":"<question text>","verification":"<brief Python verification plan>",',
            prompt,
        )
        self.assertIn('"confirm":"different_from_samples=<yes/no>; why=<short reason>"', prompt)
        self.assertNotIn("Category:", prompt)

    def test_active_question_prompt_requests_confirmation(self) -> None:
        prompt = Path("grpo_math/prompts/question_generator_prompt.txt").read_text(encoding="utf-8")

        self.assertIn("Output exactly two lines", prompt)
        self.assertIn("CONFIRM: different_from_samples=<yes/no>", prompt)
        self.assertIn("integer_answer=<yes/no>", prompt)
        self.assertIn("single integer final answer", prompt)

    def test_normalizes_optional_generation_confirmation(self) -> None:
        row = _normalize_row(
            {
                "question": "Find the smallest n divisible by 7.",
                "verification": "Brute force in Python.",
                "confirm": "different_from_samples=yes; why=uses a new divisibility pattern",
            }
        )

        self.assertNotIn("category", row)
        self.assertIn("different_from_samples=yes", row["confirm"])

    def test_converts_generated_row_for_feedback_sampling(self) -> None:
        example = _row_to_question_bank_example(
            {
                "question": "Count valid strings.",
                "verification": "Brute force with itertools.product.",
            }
        )

        self.assertEqual(example.category, "generated")
        self.assertEqual(example.task, "generated_feedback")
        self.assertEqual(example.question, "Count valid strings.")

    def test_make_diversity_nonce(self) -> None:
        import random

        nonce = _make_diversity_nonce(random.Random(0), 8)

        self.assertEqual(len(nonce), 8)
        self.assertRegex(nonce, r"^[a-z0-9]+$")


if __name__ == "__main__":
    unittest.main()
