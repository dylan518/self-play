"""Proposer-side question reward math for the verdict-only self-play loop.

A "question reward" is the scalar that the proposer (a.k.a. question generator)
GRPO update sees for each candidate question it produces.

Hard contracts (per ``LOCAL_SETUP_AGENT_PLAN.md`` Agent 1):

* Filtered-out questions get exactly ``question_reward = 0.0``. No solver or
  verifier compute is spent, so the verdict distribution is empty.
* Filtered-out questions have ``trainable_for_solver = False``.
* Filtered-out questions can remain ``trainable_for_proposer = True`` -- they
  stay in the proposer GRPO group as zero-reward samples (which gives the
  proposer a learning signal to stop emitting unusable questions).

For accepted questions, the reward rewards mixed verdicts (questions that the
solver finds genuinely interesting) and -- optionally -- judgeability and a
filter-supplied usefulness score, multiplicatively, following the design in
``VERDICT_RL_DESIGN.md``::

    proposer_reward(q) = filter_score(q) * verdict_variance(q) * judgeability(q)

We expose individual weights so the loop can toggle terms on/off without
touching the math.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from .rewards import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INVALID,
    VERDICT_UNCLEAR,
    normalize_verdict,
    verdict_distribution,
)


@dataclass(frozen=True)
class QuestionRewardBreakdown:
    """Structured breakdown of a single question's proposer reward.

    Keeping the components separate (rather than only the final scalar) makes
    debugging proposer training much easier: it is the only way to tell apart
    "filtered out", "all-correct", "all-incorrect", and "all-unclear", which
    can all collapse to ``reward=0`` for very different reasons.
    """

    accepted: bool
    duplicate: bool
    reward: float
    verdict_variance: float
    judgeability: float
    filter_score: float
    trainable_for_proposer: bool
    note: str
    verdict_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _verdict_variance(n_correct: int, n_incorrect: int) -> float:
    """4 * p * (1 - p) over the *scored* completions.

    Returns 0.0 when there are no scored completions, 1.0 when the verdicts
    split exactly 50/50 between CORRECT and INCORRECT.
    """
    n_scored = n_correct + n_incorrect
    if n_scored <= 0:
        return 0.0
    p = n_correct / n_scored
    return 4.0 * p * (1.0 - p)


def _judgeability(n_total: int, n_unclear: int) -> float:
    """Fraction of completions that received a non-unclear verdict."""
    if n_total <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (n_unclear / n_total)))


def compute_question_reward(
    *,
    accepted: bool,
    verdicts: Optional[Sequence[Any]] = None,
    duplicate: bool = False,
    filter_score: float = 1.0,
    weight_variance: float = 1.0,
    weight_judgeability: float = 0.0,
    train_filtered_for_proposer: bool = True,
) -> QuestionRewardBreakdown:
    """Compute a proposer/question reward.

    Args:
        accepted: Did the question filter accept this question? If ``False``,
            ``reward`` is hard-pinned to 0.0 and the verdict distribution is
            assumed to be empty (since no solver/verifier compute was spent).
        verdicts: Raw verdict labels for solver completions on this question.
            Required for accepted questions, ignored for rejected ones.
        duplicate: If True (and the question is accepted), reward is forced
            to 0.0. Duplicates waste compute and we don't want the proposer
            to learn that emitting near-copies is fine.
        filter_score: Optional ``[0.0, 1.0]`` usefulness score from the
            question filter. Multiplied into the reward.
        weight_variance: Exponent / weight on the verdict-variance term. Set
            to 0 to ignore variance.
        weight_judgeability: Exponent / weight on the judgeability term. Off
            (0.0) by default in the minimum viable loop.
        train_filtered_for_proposer: Should filtered-out questions remain in
            the proposer GRPO group as zero-reward samples? Default True (per
            spec): including them lets the proposer learn to stop generating
            unusable questions.

    Returns:
        A ``QuestionRewardBreakdown`` with the final scalar reward and all
        component pieces (useful for logging and tests).
    """
    if filter_score < 0.0:
        filter_score = 0.0

    if not accepted:
        return QuestionRewardBreakdown(
            accepted=False,
            duplicate=duplicate,
            reward=0.0,
            verdict_variance=0.0,
            judgeability=0.0,
            filter_score=filter_score,
            trainable_for_proposer=bool(train_filtered_for_proposer),
            note="filtered_out",
            verdict_counts={
                VERDICT_CORRECT: 0,
                VERDICT_INCORRECT: 0,
                VERDICT_INVALID: 0,
                VERDICT_UNCLEAR: 0,
            },
        )

    counts = verdict_distribution(verdicts or [])
    n_correct = counts[VERDICT_CORRECT]
    n_incorrect = counts[VERDICT_INCORRECT]
    n_invalid = counts[VERDICT_INVALID]
    n_unclear = counts[VERDICT_UNCLEAR]
    n_total = n_correct + n_incorrect + n_invalid + n_unclear

    variance = _verdict_variance(n_correct, n_incorrect)
    judgeability = _judgeability(n_total, n_unclear)

    if duplicate:
        return QuestionRewardBreakdown(
            accepted=True,
            duplicate=True,
            reward=0.0,
            verdict_variance=variance,
            judgeability=judgeability,
            filter_score=filter_score,
            trainable_for_proposer=True,
            note="duplicate",
            verdict_counts=counts,
        )

    raw = filter_score
    if weight_variance != 0.0:
        raw = raw * (variance ** float(weight_variance))
    if weight_judgeability != 0.0:
        raw = raw * (judgeability ** float(weight_judgeability))

    reward = max(0.0, float(raw))

    if n_total == 0:
        note = "accepted_no_verdicts"
    elif n_correct + n_incorrect == 0:
        note = "all_unscored"
    elif n_correct == 0:
        note = "all_incorrect"
    elif n_incorrect == 0:
        note = "all_correct"
    else:
        note = "mixed"

    return QuestionRewardBreakdown(
        accepted=True,
        duplicate=False,
        reward=reward,
        verdict_variance=variance,
        judgeability=judgeability,
        filter_score=filter_score,
        trainable_for_proposer=True,
        note=note,
        verdict_counts=counts,
    )


def compute_proposer_group_advantages(
    rewards: Sequence[float],
) -> List[float]:
    """Mean-center a proposer GRPO group's question rewards.

    A "proposer group" is the set of candidate questions sampled from the
    same proposer prompt: each one gets one scalar question reward, and we
    normalize within the group exactly the way the solver does within its
    answer group.

    Empty groups return an empty list. Uniform groups (all rewards equal)
    return all zeros, which yields no gradient.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(r - mean) for r in rewards]


def score_questions(
    questions: Sequence[Dict[str, Any]],
    *,
    weight_variance: float = 1.0,
    weight_judgeability: float = 0.0,
    train_filtered_for_proposer: bool = True,
) -> List[QuestionRewardBreakdown]:
    """Convenience: score a list of question records.

    Each input dict may have keys: ``accepted`` (bool, default True),
    ``verdicts`` (sequence of verdict labels, default empty),
    ``duplicate`` (bool, default False), ``filter_score`` (float, default 1.0).
    Unknown keys are ignored. Useful for batch scoring inside the rollout
    collector or proposer trainer.
    """
    out: List[QuestionRewardBreakdown] = []
    for q in questions:
        out.append(
            compute_question_reward(
                accepted=bool(q.get("accepted", True)),
                verdicts=q.get("verdicts"),
                duplicate=bool(q.get("duplicate", False)),
                filter_score=float(q.get("filter_score", 1.0)),
                weight_variance=weight_variance,
                weight_judgeability=weight_judgeability,
                train_filtered_for_proposer=train_filtered_for_proposer,
            )
        )
    return out


__all__ = [
    "QuestionRewardBreakdown",
    "compute_question_reward",
    "compute_proposer_group_advantages",
    "score_questions",
    # re-exported for convenience
    "normalize_verdict",
]
