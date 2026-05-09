"""Solver-side verdict reward math for the verdict-only self-play loop.

This module implements the canonical pieces shared between rollout collection
and the solver GRPO trainer:

* a stable schema version string for verdict-only self-play rollouts
* the verdict label vocabulary and a forgiving normalizer
* a verdict-to-scalar score mapping
* per-completion solver reward records (score, advantage, trainable flag)
* within-question (a.k.a. within-group) advantage computation

The functions here are pure, deterministic, and have no I/O. They are designed
to be unit-tested without GPUs, models, or API access.

See ``VERDICT_RL_DESIGN.md`` and ``LOCAL_SETUP_AGENT_PLAN.md`` (Agent 1) for the
broader context.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION: str = "self_play_verdict_grpo_v1"
"""Canonical schema version for rollout JSONL rows produced by the verdict-only
self-play loop. Bump when rollout fields change in a backwards-incompatible way.
"""


VERDICT_CORRECT: str = "CORRECT"
VERDICT_INCORRECT: str = "INCORRECT"
VERDICT_INVALID: str = "INVALID"
VERDICT_UNCLEAR: str = "UNCLEAR"

VALID_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_INVALID, VERDICT_UNCLEAR}
)


DEFAULT_VERDICT_SCORES: Mapping[str, Optional[float]] = {
    VERDICT_CORRECT: 1.0,
    VERDICT_INCORRECT: 0.0,
    VERDICT_INVALID: 0.0,
    VERDICT_UNCLEAR: None,
}
"""Default mapping from verdict label to scalar score. ``None`` means the
completion should be ignored when computing within-question advantages
(but is still recorded for logging).
"""


_TRUE_ALIASES: frozenset[str] = frozenset({"TRUE", "T", "YES", "Y", "RIGHT", "PASS"})
_FALSE_ALIASES: frozenset[str] = frozenset({"FALSE", "F", "NO", "N", "WRONG", "FAIL"})


def normalize_verdict(raw: Any) -> str:
    """Normalize a model-emitted verdict to one of ``VALID_VERDICTS``.

    Anything we cannot confidently map (None, empty string, parse failures,
    arbitrary strings) becomes ``UNCLEAR``. ``TRUE/YES/RIGHT`` -> ``CORRECT``,
    ``FALSE/NO/WRONG`` -> ``INCORRECT``. This mirrors the tolerant parsing used
    by the existing single-verify pipeline.
    """
    if raw is None:
        return VERDICT_UNCLEAR
    s = str(raw).strip().upper()
    if not s:
        return VERDICT_UNCLEAR
    if s in VALID_VERDICTS:
        return s
    if s in _TRUE_ALIASES:
        return VERDICT_CORRECT
    if s in _FALSE_ALIASES:
        return VERDICT_INCORRECT
    return VERDICT_UNCLEAR


def verdict_to_score(
    verdict: str,
    mapping: Mapping[str, Optional[float]] = DEFAULT_VERDICT_SCORES,
) -> Optional[float]:
    """Look up the scalar score for a (already-normalized) verdict.

    Returns ``None`` when the verdict should be ignored (typically ``UNCLEAR``).
    Unknown verdicts also return ``None`` so callers cannot accidentally train
    on garbage labels.
    """
    return mapping.get(verdict)


@dataclass(frozen=True)
class SolverCompletionReward:
    """Per-completion reward record for a single solver answer.

    Attributes:
        verdict: Normalized verdict label.
        score: Scalar score, or ``None`` if the completion should be ignored.
        trainable_for_solver: Whether this completion contributes to the GRPO
            update for the solver.
        advantage: Within-question advantage (``score - mean(group_scores)``)
            or ``None`` when not trainable.
    """

    verdict: str
    score: Optional[float]
    trainable_for_solver: bool
    advantage: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_within_question_advantages(
    scores: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Compute mean-centered advantages over a group of (possibly partial) scores.

    Entries with ``score=None`` (e.g. unclear verdicts) are excluded from the
    mean and receive ``advantage=None`` in the output.

    With all-correct or all-incorrect groups the mean equals every score, so
    every advantage is exactly ``0.0`` -- which yields no gradient under GRPO.
    That is the desired behavior: the solver only learns from groups that
    contain disagreement.
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return [None for _ in scores]
    mean = sum(valid) / len(valid)
    out: List[Optional[float]] = []
    for s in scores:
        if s is None:
            out.append(None)
        else:
            out.append(float(s - mean))
    return out


def score_solver_completions(
    verdicts: Sequence[Any],
    *,
    question_accepted: bool = True,
    verdict_scores: Mapping[str, Optional[float]] = DEFAULT_VERDICT_SCORES,
) -> List[SolverCompletionReward]:
    """Score a group of solver completions for a single question.

    Args:
        verdicts: Raw verdict labels (one per completion). Will be normalized.
        question_accepted: If ``False`` (question filtered out), every
            completion is marked non-trainable for the solver and receives
            ``advantage=None``. The verdict + score are still preserved for
            logging if they were present.
        verdict_scores: Verdict -> scalar score mapping. Override to change the
            reward scheme (e.g. partial credit) without touching the rest of
            the pipeline.

    Returns:
        One ``SolverCompletionReward`` per input verdict, in the same order.
    """
    normalized = [normalize_verdict(v) for v in verdicts]
    scores: List[Optional[float]] = [verdict_scores.get(v) for v in normalized]

    if not question_accepted:
        return [
            SolverCompletionReward(
                verdict=v,
                score=s,
                trainable_for_solver=False,
                advantage=None,
            )
            for v, s in zip(normalized, scores)
        ]

    advantages = compute_within_question_advantages(scores)
    out: List[SolverCompletionReward] = []
    for v, s, adv in zip(normalized, scores, advantages):
        trainable = s is not None
        out.append(
            SolverCompletionReward(
                verdict=v,
                score=s,
                trainable_for_solver=trainable,
                advantage=adv if trainable else None,
            )
        )
    return out


def verdict_distribution(verdicts: Sequence[Any]) -> Dict[str, int]:
    """Return a count of each canonical verdict label in ``verdicts``.

    All four labels are always present in the returned dict (with value 0 if
    absent) so downstream code can rely on stable keys.
    """
    counts: Dict[str, int] = {label: 0 for label in VALID_VERDICTS}
    for raw in verdicts:
        counts[normalize_verdict(raw)] += 1
    return counts


__all__ = [
    "SCHEMA_VERSION",
    "VERDICT_CORRECT",
    "VERDICT_INCORRECT",
    "VERDICT_INVALID",
    "VERDICT_UNCLEAR",
    "VALID_VERDICTS",
    "DEFAULT_VERDICT_SCORES",
    "SolverCompletionReward",
    "normalize_verdict",
    "verdict_to_score",
    "compute_within_question_advantages",
    "score_solver_completions",
    "verdict_distribution",
]
