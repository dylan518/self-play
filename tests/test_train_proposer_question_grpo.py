import json
from pathlib import Path

import pytest

from grpo_math.trl.train_proposer_question_grpo import (
    compute_question_reward,
    examples_with_group_advantages,
    load_offline_examples,
    normalize_group_advantages,
    summarize_examples,
    validate_config,
)


def _base_cfg(path: Path) -> dict:
    return {
        "model": {"name_or_path": "sshleifer/tiny-gpt2", "torch_dtype": "float32"},
        "data": {"mode": "offline", "rollout_jsonl_paths": [str(path)]},
        "prompt": {"template": "QUESTION: make a problem"},
        "rollout": {"k": 2, "temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32},
        "train": {"steps": 0, "output_dir": "unused", "lr": 1e-5, "kl_beta": 0.01},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compute_question_reward_keeps_filtered_questions_at_zero() -> None:
    row = {
        "question": "bad",
        "trainable_for_solver": False,
        "trainable_for_proposer": True,
        "question_reward": 0.9,
        "solution_verifications": [{"verdicts": ["CORRECT"]}],
    }

    assert compute_question_reward(row) == 0.0


def test_compute_question_reward_from_mixed_verdicts() -> None:
    row = {
        "question": "ok",
        "solution_verifications": [
            {"verdicts": ["CORRECT", "CORRECT"]},
            {"verdicts": ["INCORRECT", "INCORRECT"]},
        ],
        "reliability": {"preference_stability": 0.8},
    }

    assert compute_question_reward(row) == pytest.approx(0.7)


def test_normalize_group_advantages_are_centered() -> None:
    advantages = normalize_group_advantages([0.0, 0.5, 1.0])

    assert advantages == pytest.approx([-0.5, 0.0, 0.5])
    assert sum(advantages) == pytest.approx(0.0)


def test_load_offline_examples_and_group_advantages(tmp_path: Path) -> None:
    jsonl = tmp_path / "rollouts.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "question_index": 0,
                "question": "What is 2+2?",
                "question_reward": 0.75,
                "trainable_for_solver": True,
                "solution_verifications": [{"verdicts": ["CORRECT"]}, {"verdicts": ["INCORRECT"]}],
            },
            {
                "question_index": 0,
                "question": "What is 3+3?",
                "trainable_for_solver": False,
                "trainable_for_proposer": True,
                "solution_verifications": [],
            },
            {
                "question_index": 0,
                "question": "What is 2+2?",
                "question_reward": 0.25,
                "trainable_for_solver": True,
                "solution_verifications": [{"verdicts": ["CORRECT"]}],
            },
        ],
    )

    examples = load_offline_examples(_base_cfg(jsonl))
    rows = examples_with_group_advantages(examples)
    summary = summarize_examples(examples)

    assert [ex.question_reward for ex in examples] == [0.75, 0.0, 0.25]
    assert examples[1].accepted is False
    assert examples[2].duplicate is True
    assert [row["question_advantage"] for row in rows] == pytest.approx([0.4166667, -0.3333333, -0.0833333])
    assert summary["acceptance_rate"] == pytest.approx(2 / 3)
    assert summary["zero_reward_rate"] == pytest.approx(1 / 3)
    assert summary["duplicate_rate"] == pytest.approx(1 / 3)
    assert summary["mixed_verdict_rate"] == pytest.approx(1 / 3)


def test_validate_config_rejects_online_until_provider_interfaces_exist(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "x.jsonl")
    cfg["data"]["mode"] = "online"

    with pytest.raises(ValueError, match="online proposer GRPO"):
        validate_config(cfg)


def test_validate_config_requires_rollout_paths(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "x.jsonl")
    cfg["data"].pop("rollout_jsonl_paths")

    with pytest.raises(ValueError, match="rollout_jsonl_paths"):
        validate_config(cfg)
