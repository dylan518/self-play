import unittest

from grpo_math.data.reward import (
    binary_reward,
    extract_final_answer_int,
    extract_final_answer_int_strict,
    extract_ground_truth_int,
)


class TestRewardParsing(unittest.TestCase):
    def test_extract_ground_truth_int(self) -> None:
        self.assertEqual(extract_ground_truth_int("foo #### 42"), 42)
        self.assertEqual(extract_ground_truth_int("#### -7"), -7)
        self.assertEqual(extract_ground_truth_int("no ints here"), None)
        self.assertEqual(extract_ground_truth_int("#### 12\nsome text 99"), 99)  # last int wins

    def test_extract_final_answer_int_prefers_tag(self) -> None:
        self.assertEqual(extract_final_answer_int("FINAL_ANSWER: 123"), 123)
        self.assertEqual(extract_final_answer_int("blah\nFINAL_ANSWER: -9\n"), -9)
        # If tag exists but has no int, fall back to last int overall
        self.assertEqual(extract_final_answer_int("FINAL_ANSWER:\n... 55"), 55)

    def test_extract_final_answer_int_strict(self) -> None:
        self.assertEqual(extract_final_answer_int_strict("FINAL_ANSWER: 123"), 123)
        self.assertEqual(extract_final_answer_int_strict("blah\nFINAL_ANSWER: -9\n"), -9)
        self.assertIsNone(extract_final_answer_int_strict("no tag but 55"))
        self.assertIsNone(extract_final_answer_int_strict("FINAL_ANSWER:\n(no int here)"))
        # Tolerate floats that are mathematically integers
        self.assertEqual(extract_final_answer_int_strict("FINAL_ANSWER: 30.00"), 30)
        self.assertIsNone(extract_final_answer_int_strict("FINAL_ANSWER: 30.5"))
        # Allow some models to continue without inserting a newline
        self.assertEqual(extract_final_answer_int_strict("FINAL_ANSWER: 12Human: blah"), 12)

    def test_binary_reward(self) -> None:
        r, pred, gt = binary_reward("FINAL_ANSWER: 5", "#### 5")
        self.assertEqual(r, 1.0)
        self.assertEqual(pred, 5)
        self.assertEqual(gt, 5)

        r, pred, gt = binary_reward("FINAL_ANSWER: 6", "#### 5")
        self.assertEqual(r, 0.0)
        self.assertEqual(pred, 6)
        self.assertEqual(gt, 5)

        # Strict format: no tag => no reward, even if some integer appears elsewhere.
        r, pred, gt = binary_reward("work...\nanswer is 5\n", "#### 5")
        self.assertEqual(r, 0.0)
        self.assertIsNone(pred)
        self.assertEqual(gt, 5)


if __name__ == "__main__":
    unittest.main()

