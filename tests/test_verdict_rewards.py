"""Unit tests for grpo_math.self_play.rewards."""

from __future__ import annotations

import unittest

from grpo_math.self_play.rewards import (
    DEFAULT_VERDICT_SCORES,
    SCHEMA_VERSION,
    SolverCompletionReward,
    VALID_VERDICTS,
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INVALID,
    VERDICT_UNCLEAR,
    compute_within_question_advantages,
    normalize_verdict,
    score_solver_completions,
    verdict_distribution,
    verdict_to_score,
)


class TestSchemaConstants(unittest.TestCase):
    def test_schema_version_is_v1(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "self_play_verdict_grpo_v1")

    def test_valid_verdicts_set(self) -> None:
        self.assertEqual(
            VALID_VERDICTS,
            frozenset({VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_INVALID, VERDICT_UNCLEAR}),
        )

    def test_default_verdict_scores(self) -> None:
        self.assertEqual(DEFAULT_VERDICT_SCORES[VERDICT_CORRECT], 1.0)
        self.assertEqual(DEFAULT_VERDICT_SCORES[VERDICT_INCORRECT], 0.0)
        self.assertEqual(DEFAULT_VERDICT_SCORES[VERDICT_INVALID], 0.0)
        self.assertIsNone(DEFAULT_VERDICT_SCORES[VERDICT_UNCLEAR])


class TestNormalizeVerdict(unittest.TestCase):
    def test_passthrough_canonical_labels(self) -> None:
        self.assertEqual(normalize_verdict("CORRECT"), VERDICT_CORRECT)
        self.assertEqual(normalize_verdict("INCORRECT"), VERDICT_INCORRECT)
        self.assertEqual(normalize_verdict("INVALID"), VERDICT_INVALID)
        self.assertEqual(normalize_verdict("UNCLEAR"), VERDICT_UNCLEAR)

    def test_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(normalize_verdict("  correct  "), VERDICT_CORRECT)
        self.assertEqual(normalize_verdict("Incorrect\n"), VERDICT_INCORRECT)

    def test_aliases(self) -> None:
        self.assertEqual(normalize_verdict("true"), VERDICT_CORRECT)
        self.assertEqual(normalize_verdict("YES"), VERDICT_CORRECT)
        self.assertEqual(normalize_verdict("right"), VERDICT_CORRECT)
        self.assertEqual(normalize_verdict("false"), VERDICT_INCORRECT)
        self.assertEqual(normalize_verdict("NO"), VERDICT_INCORRECT)
        self.assertEqual(normalize_verdict("wrong"), VERDICT_INCORRECT)

    def test_unknown_or_empty_to_unclear(self) -> None:
        self.assertEqual(normalize_verdict(None), VERDICT_UNCLEAR)
        self.assertEqual(normalize_verdict(""), VERDICT_UNCLEAR)
        self.assertEqual(normalize_verdict("   "), VERDICT_UNCLEAR)
        self.assertEqual(normalize_verdict("maybe?"), VERDICT_UNCLEAR)
        self.assertEqual(normalize_verdict(42), VERDICT_UNCLEAR)


class TestVerdictToScore(unittest.TestCase):
    def test_default_mapping(self) -> None:
        self.assertEqual(verdict_to_score(VERDICT_CORRECT), 1.0)
        self.assertEqual(verdict_to_score(VERDICT_INCORRECT), 0.0)
        self.assertEqual(verdict_to_score(VERDICT_INVALID), 0.0)
        self.assertIsNone(verdict_to_score(VERDICT_UNCLEAR))
        self.assertIsNone(verdict_to_score("not_a_verdict"))

    def test_custom_mapping(self) -> None:
        custom = {VERDICT_CORRECT: 2.0, VERDICT_INCORRECT: -1.0}
        self.assertEqual(verdict_to_score(VERDICT_CORRECT, custom), 2.0)
        self.assertEqual(verdict_to_score(VERDICT_INCORRECT, custom), -1.0)
        self.assertIsNone(verdict_to_score(VERDICT_INVALID, custom))


class TestComputeWithinQuestionAdvantages(unittest.TestCase):
    def test_mixed_group(self) -> None:
        adv = compute_within_question_advantages([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(adv, [0.5, -0.5, -0.5, 0.5])

    def test_all_correct_yields_zero_advantage(self) -> None:
        adv = compute_within_question_advantages([1.0, 1.0, 1.0])
        self.assertEqual(adv, [0.0, 0.0, 0.0])

    def test_all_incorrect_yields_zero_advantage(self) -> None:
        adv = compute_within_question_advantages([0.0, 0.0, 0.0])
        self.assertEqual(adv, [0.0, 0.0, 0.0])

    def test_unclear_excluded_from_mean(self) -> None:
        # Two correct (1.0) + one ignored -> mean = 1.0, advantages 0 / 0 / None
        adv = compute_within_question_advantages([1.0, 1.0, None])
        self.assertEqual(adv, [0.0, 0.0, None])

    def test_all_none_returns_all_none(self) -> None:
        self.assertEqual(compute_within_question_advantages([None, None, None]), [None, None, None])

    def test_empty(self) -> None:
        self.assertEqual(compute_within_question_advantages([]), [])


class TestScoreSolverCompletions(unittest.TestCase):
    def _verdicts(self, rs):
        return [r.verdict for r in rs]

    def _scores(self, rs):
        return [r.score for r in rs]

    def _advs(self, rs):
        return [r.advantage for r in rs]

    def _trainable(self, rs):
        return [r.trainable_for_solver for r in rs]

    def test_accepted_mixed(self) -> None:
        rs = score_solver_completions(["CORRECT", "INCORRECT", "INCORRECT", "CORRECT"])
        self.assertEqual(self._verdicts(rs), [VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_INCORRECT, VERDICT_CORRECT])
        self.assertEqual(self._scores(rs), [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(self._advs(rs), [0.5, -0.5, -0.5, 0.5])
        self.assertEqual(self._trainable(rs), [True, True, True, True])

    def test_accepted_all_correct(self) -> None:
        rs = score_solver_completions(["CORRECT", "CORRECT", "CORRECT"])
        self.assertEqual(self._scores(rs), [1.0, 1.0, 1.0])
        self.assertEqual(self._advs(rs), [0.0, 0.0, 0.0])
        # Still trainable -- they just contribute zero gradient.
        self.assertEqual(self._trainable(rs), [True, True, True])

    def test_accepted_all_incorrect(self) -> None:
        rs = score_solver_completions(["INCORRECT", "INCORRECT"])
        self.assertEqual(self._scores(rs), [0.0, 0.0])
        self.assertEqual(self._advs(rs), [0.0, 0.0])
        self.assertEqual(self._trainable(rs), [True, True])

    def test_invalid_treated_as_zero_score(self) -> None:
        rs = score_solver_completions(["CORRECT", "INVALID", "INCORRECT"])
        # mean = (1 + 0 + 0)/3 = 1/3
        self.assertEqual(self._scores(rs), [1.0, 0.0, 0.0])
        self.assertEqual(self._trainable(rs), [True, True, True])
        self.assertAlmostEqual(rs[0].advantage, 2.0 / 3.0)
        self.assertAlmostEqual(rs[1].advantage, -1.0 / 3.0)
        self.assertAlmostEqual(rs[2].advantage, -1.0 / 3.0)

    def test_unclear_skipped_from_advantage_but_logged(self) -> None:
        rs = score_solver_completions(["CORRECT", "UNCLEAR", "INCORRECT"])
        # UNCLEAR has score=None, excluded from the mean -> mean=0.5
        self.assertEqual(rs[0].verdict, VERDICT_CORRECT)
        self.assertEqual(rs[1].verdict, VERDICT_UNCLEAR)
        self.assertEqual(rs[2].verdict, VERDICT_INCORRECT)
        self.assertEqual([rs[0].score, rs[1].score, rs[2].score], [1.0, None, 0.0])
        self.assertEqual(rs[1].trainable_for_solver, False)
        self.assertIsNone(rs[1].advantage)
        self.assertAlmostEqual(rs[0].advantage, 0.5)
        self.assertAlmostEqual(rs[2].advantage, -0.5)
        self.assertEqual(rs[0].trainable_for_solver, True)
        self.assertEqual(rs[2].trainable_for_solver, True)

    def test_all_unclear_yields_no_trainable_completions(self) -> None:
        rs = score_solver_completions(["UNCLEAR", "UNCLEAR"])
        self.assertEqual(self._trainable(rs), [False, False])
        self.assertEqual(self._advs(rs), [None, None])
        self.assertEqual(self._scores(rs), [None, None])

    def test_malformed_verdicts_become_unclear(self) -> None:
        rs = score_solver_completions(["???", "", None, "CORRECT"])
        verdicts = self._verdicts(rs)
        self.assertEqual(
            verdicts,
            [VERDICT_UNCLEAR, VERDICT_UNCLEAR, VERDICT_UNCLEAR, VERDICT_CORRECT],
        )
        # Only the CORRECT one is trainable.
        self.assertEqual(self._trainable(rs), [False, False, False, True])
        # Mean of the single scored completion is 1.0 -> advantage 0.0.
        self.assertEqual(rs[3].advantage, 0.0)

    def test_rejected_question_disables_solver_training(self) -> None:
        rs = score_solver_completions(
            ["CORRECT", "INCORRECT"], question_accepted=False
        )
        # Verdicts are still preserved for logging.
        self.assertEqual(self._verdicts(rs), [VERDICT_CORRECT, VERDICT_INCORRECT])
        # But nothing is trainable and there are no advantages.
        self.assertEqual(self._trainable(rs), [False, False])
        self.assertEqual(self._advs(rs), [None, None])

    def test_custom_verdict_score_mapping(self) -> None:
        mapping = {VERDICT_CORRECT: 1.0, VERDICT_INCORRECT: -1.0}
        rs = score_solver_completions(
            ["CORRECT", "INCORRECT"], verdict_scores=mapping
        )
        self.assertEqual(self._scores(rs), [1.0, -1.0])
        # mean = 0 -> advantages = [1.0, -1.0]
        self.assertEqual(self._advs(rs), [1.0, -1.0])

    def test_records_are_serializable(self) -> None:
        rs = score_solver_completions(["CORRECT", "INCORRECT"])
        d = rs[0].to_dict()
        self.assertEqual(
            set(d.keys()),
            {"verdict", "score", "trainable_for_solver", "advantage"},
        )
        self.assertIsInstance(rs[0], SolverCompletionReward)


class TestVerdictDistribution(unittest.TestCase):
    def test_all_labels_always_present(self) -> None:
        counts = verdict_distribution([])
        self.assertEqual(
            set(counts.keys()),
            {VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_INVALID, VERDICT_UNCLEAR},
        )
        self.assertTrue(all(v == 0 for v in counts.values()))

    def test_counts_with_normalization(self) -> None:
        counts = verdict_distribution(
            ["correct", "TRUE", "incorrect", "no", "invalid", "???", None]
        )
        self.assertEqual(counts[VERDICT_CORRECT], 2)
        self.assertEqual(counts[VERDICT_INCORRECT], 2)
        self.assertEqual(counts[VERDICT_INVALID], 1)
        self.assertEqual(counts[VERDICT_UNCLEAR], 2)


if __name__ == "__main__":
    unittest.main()
