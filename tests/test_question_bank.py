from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
        self.assertIn("VERIFICATION IDEA:", block)
        self.assertEqual(block.count("QUESTION:"), 3)
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


if __name__ == "__main__":
    unittest.main()
