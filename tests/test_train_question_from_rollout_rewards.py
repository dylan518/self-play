import pytest

from grpo_math.trl.train_question_from_rollout_rewards import (
    build_preference_rows,
    split_preference_pairs,
)


def test_build_preference_rows_pairs_questions_by_rollout_reward() -> None:
    rows = [
        {
            "run_id": "cycle-a",
            "proposer_prompt": "make a question",
            "question": "easy",
            "generator_raw_output": "QUESTION: easy",
            "question_reward": 0.1,
            "trainable_for_proposer": True,
        },
        {
            "run_id": "cycle-a",
            "proposer_prompt": "make a question",
            "question": "mixed",
            "generator_raw_output": "QUESTION: mixed",
            "question_reward": 0.8,
            "trainable_for_proposer": True,
        },
        {
            "run_id": "cycle-b",
            "question": "solo",
            "generator_raw_output": "QUESTION: solo",
            "question_reward": 1.0,
            "trainable_for_proposer": True,
        },
    ]

    pairs = build_preference_rows(
        rows,
        prompt_fallback="fallback prompt",
        min_reward_gap=0.05,
        max_pairs_per_group=10,
        seed=1,
    )

    assert pairs == [
        {
            "prompt": "make a question",
            "chosen": "QUESTION: mixed",
            "rejected": "QUESTION: easy",
        }
    ]


def test_build_preference_rows_uses_any_positive_reward_difference() -> None:
    pairs = build_preference_rows(
        [
            {"run_id": "g", "question": "a", "question_reward": 0.40},
            {"run_id": "g", "question": "b", "question_reward": 0.45},
        ],
        prompt_fallback="prompt",
        min_reward_gap=0.05,
        max_pairs_per_group=10,
        seed=1,
    )

    assert pairs == [
        {
            "prompt": "prompt",
            "chosen": "QUESTION: b",
            "rejected": "QUESTION: a",
        }
    ]


def test_split_preference_pairs_is_reproducible_and_disjoint() -> None:
    pairs = [{"prompt": "p", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(10)]
    train_a, eval_a = split_preference_pairs(pairs, eval_fraction=0.2, seed=7)
    train_b, eval_b = split_preference_pairs(pairs, eval_fraction=0.2, seed=7)
    assert train_a == train_b
    assert eval_a == eval_b
    assert len(eval_a) == 2
    assert len(train_a) == 8
    assert {id(x) for x in train_a}.isdisjoint({id(x) for x in eval_a})
