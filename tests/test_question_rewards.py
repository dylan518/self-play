"""Unit tests for grpo_math.self_play.question_rewards."""

from __future__ import annotations

import unittest

from grpo_math.self_play.question_rewards import (
    QuestionRewardBreakdown,
    compute_proposer_group_advantages,
    compute_question_reward,
    score_questions,
)
from grpo_math.self_play.rewards import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INVALID,
    VERDICT_UNCLEAR,
)


class TestRejectedQuestions(unittest.TestCase):
    def test_filtered_out_has_zero_reward_and_default_proposer_trainable(self) -> None:
        b = compute_question_reward(accepted=False, verdicts=None)
        self.assertEqual(b.reward, 0.0)
        self.assertEqual(b.accepted, False)
        # Per spec: filtered-out questions are zero-reward but still in proposer group.
        self.assertTrue(b.trainable_for_proposer)
        self.assertEqual(b.note, "filtered_out")
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.judgeability, 0.0)
        # All verdict counts are zero (no compute spent).
        self.assertEqual(
            b.verdict_counts,
            {
                VERDICT_CORRECT: 0,
                VERDICT_INCORRECT: 0,
                VERDICT_INVALID: 0,
                VERDICT_UNCLEAR: 0,
            },
        )

    def test_filtered_out_can_be_excluded_from_proposer_groups(self) -> None:
        b = compute_question_reward(
            accepted=False, verdicts=None, train_filtered_for_proposer=False
        )
        self.assertEqual(b.reward, 0.0)
        self.assertFalse(b.trainable_for_proposer)
        self.assertEqual(b.note, "filtered_out")

    def test_filtered_out_ignores_verdict_input(self) -> None:
        # Even if (somehow) verdicts are passed for a filtered-out question,
        # the reward must still be exactly 0.0 (no compute should have been spent).
        b = compute_question_reward(
            accepted=False,
            verdicts=["CORRECT", "INCORRECT"],
            duplicate=True,
            filter_score=1.0,
        )
        self.assertEqual(b.reward, 0.0)


class TestAcceptedQuestions(unittest.TestCase):
    def test_mixed_verdicts_yield_positive_reward(self) -> None:
        b = compute_question_reward(
            accepted=True, verdicts=["CORRECT", "INCORRECT", "INCORRECT", "CORRECT"]
        )
        self.assertEqual(b.note, "mixed")
        self.assertEqual(b.verdict_variance, 1.0)  # 50/50 split
        self.assertEqual(b.judgeability, 1.0)  # no UNCLEAR
        # default weights: variance only, filter_score=1
        self.assertEqual(b.reward, 1.0)
        self.assertTrue(b.trainable_for_proposer)
        self.assertTrue(b.accepted)

    def test_all_correct_collapses_to_zero_reward(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=["CORRECT", "CORRECT"])
        self.assertEqual(b.note, "all_correct")
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.reward, 0.0)
        self.assertTrue(b.trainable_for_proposer)

    def test_all_incorrect_collapses_to_zero_reward(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=["INCORRECT", "INCORRECT"])
        self.assertEqual(b.note, "all_incorrect")
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.reward, 0.0)
        self.assertTrue(b.trainable_for_proposer)

    def test_all_invalid_collapses_to_zero_reward(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=["INVALID", "INVALID"])
        # No CORRECT/INCORRECT scoring -> variance 0, but judgeability is still 1
        # (INVALID is not UNCLEAR).
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.judgeability, 1.0)
        self.assertEqual(b.reward, 0.0)
        self.assertEqual(b.note, "all_unscored")

    def test_all_unclear_collapses_to_zero_reward_and_low_judgeability(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=["UNCLEAR", "UNCLEAR"])
        self.assertEqual(b.judgeability, 0.0)
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.reward, 0.0)
        self.assertEqual(b.note, "all_unscored")

    def test_judgeability_term(self) -> None:
        # Two CORRECT, two INCORRECT, one UNCLEAR -> judgeability = 4/5
        # variance = 4 * 0.5 * 0.5 = 1.0
        b = compute_question_reward(
            accepted=True,
            verdicts=["CORRECT", "INCORRECT", "CORRECT", "INCORRECT", "UNCLEAR"],
            weight_judgeability=1.0,
        )
        self.assertAlmostEqual(b.verdict_variance, 1.0)
        self.assertAlmostEqual(b.judgeability, 4 / 5)
        self.assertAlmostEqual(b.reward, 1.0 * 1.0 * (4 / 5))

    def test_filter_score_multiplies_reward(self) -> None:
        b = compute_question_reward(
            accepted=True,
            verdicts=["CORRECT", "INCORRECT"],
            filter_score=0.5,
        )
        # variance = 1.0, filter_score = 0.5 -> 0.5
        self.assertAlmostEqual(b.reward, 0.5)
        # Negative filter scores get clamped to 0.
        b2 = compute_question_reward(
            accepted=True,
            verdicts=["CORRECT", "INCORRECT"],
            filter_score=-2.0,
        )
        self.assertEqual(b2.filter_score, 0.0)
        self.assertEqual(b2.reward, 0.0)

    def test_duplicate_accepted_question_has_zero_reward_but_trainable(self) -> None:
        b = compute_question_reward(
            accepted=True,
            verdicts=["CORRECT", "INCORRECT"],
            duplicate=True,
        )
        self.assertEqual(b.reward, 0.0)
        self.assertEqual(b.note, "duplicate")
        self.assertTrue(b.trainable_for_proposer)
        self.assertTrue(b.accepted)
        self.assertTrue(b.duplicate)

    def test_malformed_verdicts_become_unclear(self) -> None:
        b = compute_question_reward(
            accepted=True, verdicts=["???", "", None, "CORRECT"]
        )
        self.assertEqual(b.verdict_counts[VERDICT_UNCLEAR], 3)
        self.assertEqual(b.verdict_counts[VERDICT_CORRECT], 1)
        # Only one scored completion -> p=1 -> variance=0
        self.assertEqual(b.verdict_variance, 0.0)
        self.assertEqual(b.reward, 0.0)

    def test_no_verdicts_is_accepted_no_verdicts(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=[])
        self.assertEqual(b.note, "accepted_no_verdicts")
        self.assertEqual(b.reward, 0.0)
        self.assertTrue(b.trainable_for_proposer)


class TestProposerGroupAdvantages(unittest.TestCase):
    def test_mean_centered(self) -> None:
        adv = compute_proposer_group_advantages([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(adv, [0.5, -0.5, -0.5, 0.5])

    def test_uniform_group_zero(self) -> None:
        adv = compute_proposer_group_advantages([0.7, 0.7, 0.7])
        self.assertEqual(len(adv), 3)
        for a in adv:
            self.assertAlmostEqual(a, 0.0)

    def test_zero_reward_filtered_questions_drag_mean(self) -> None:
        # 3 accepted (reward 1.0) + 1 filtered (0.0) -> mean = 0.75
        adv = compute_proposer_group_advantages([1.0, 1.0, 1.0, 0.0])
        self.assertEqual(adv, [0.25, 0.25, 0.25, -0.75])

    def test_empty(self) -> None:
        self.assertEqual(compute_proposer_group_advantages([]), [])


class TestScoreQuestions(unittest.TestCase):
    def test_batch_scoring(self) -> None:
        records = [
            {"accepted": True, "verdicts": ["CORRECT", "INCORRECT"]},
            {"accepted": False},
            {"accepted": True, "verdicts": ["CORRECT", "CORRECT"]},
            {"accepted": True, "verdicts": ["CORRECT", "INCORRECT"], "duplicate": True},
        ]
        out = score_questions(records)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0].note, "mixed")
        self.assertEqual(out[1].note, "filtered_out")
        self.assertEqual(out[2].note, "all_correct")
        self.assertEqual(out[3].note, "duplicate")
        self.assertEqual(out[1].reward, 0.0)
        self.assertEqual(out[2].reward, 0.0)
        self.assertEqual(out[3].reward, 0.0)
        self.assertGreater(out[0].reward, 0.0)

    def test_breakdown_is_serializable(self) -> None:
        b = compute_question_reward(accepted=True, verdicts=["CORRECT", "INCORRECT"])
        d = b.to_dict()
        self.assertIn("reward", d)
        self.assertIn("verdict_counts", d)
        self.assertIn("note", d)
        self.assertIsInstance(b, QuestionRewardBreakdown)
        # verdict_counts dict should round-trip.
        self.assertEqual(d["verdict_counts"][VERDICT_CORRECT], 1)
        self.assertEqual(d["verdict_counts"][VERDICT_INCORRECT], 1)
        self.assertEqual(d["verdict_counts"][VERDICT_INVALID], 0)
        self.assertEqual(d["verdict_counts"][VERDICT_UNCLEAR], 0)


if __name__ == "__main__":
    unittest.main()
