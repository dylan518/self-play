from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import yaml

try:
    import torch
except Exception:  # pragma: no cover - exercised only in minimal CPU envs
    torch = None  # type: ignore[assignment]

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - import validation should still work without datasets
    Dataset = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:  # pragma: no cover
    GRPOConfig = None  # type: ignore[assignment]
    GRPOTrainer = None  # type: ignore[assignment]


SCHEMA_VERSION = "self_play_verdict_grpo_v1"
_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ProposerExample:
    prompt: str
    question: str
    group_id: str
    question_reward: float
    trainable_for_proposer: bool
    accepted: bool
    duplicate: bool
    mixed_verdict: bool


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            yield row


def _torch_dtype(name: str) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for training.")
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _parse_question(text: str) -> str:
    match = _QUESTION_RE.search(text or "")
    candidate = match.group(1) if match else text
    candidate = re.sub(
        r"\n?\s*(ANSWER|FINAL_ANSWER|SOLUTION)\s*:.*$",
        "",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return candidate.strip()


def _canonical_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _row_prompt(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if isinstance(row.get("proposer_prompt"), str):
        return str(row["proposer_prompt"])
    if isinstance(row.get("prompt"), str):
        return str(row["prompt"])
    template = str(cfg.get("prompt", {}).get("template", "Generate one math question.\n"))
    return template


def _row_group_id(row: Dict[str, Any], fallback: int) -> str:
    for key in ("proposer_group_id", "group_id", "prompt_id", "question_index"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(fallback)


def _row_trainable_for_proposer(row: Dict[str, Any]) -> bool:
    return bool(row.get("trainable_for_proposer", True))


def _row_accepted(row: Dict[str, Any]) -> bool:
    if "accepted" in row:
        return bool(row["accepted"])
    if "trainable_for_solver" in row:
        return bool(row["trainable_for_solver"])
    result = row.get("filter_result")
    if isinstance(result, dict) and "accepted" in result:
        return bool(result["accepted"])
    return True


def _solution_verdict_scores(row: Dict[str, Any]) -> List[float]:
    scores: List[float] = []
    for item in row.get("solution_verifications", []) or []:
        if not isinstance(item, dict):
            continue
        counts = item.get("counts")
        if isinstance(counts, dict):
            total = float(sum(float(v) for v in counts.values()))
            if total > 0:
                scores.append(float(counts.get("CORRECT", 0.0)) / total)
                continue
        verdicts = item.get("verdicts")
        if isinstance(verdicts, list) and verdicts:
            scores.append(sum(1.0 for v in verdicts if str(v).upper() == "CORRECT") / len(verdicts))
    if scores:
        return scores

    solution_scores: List[float] = []
    for sol in row.get("solutions", []) or []:
        if not isinstance(sol, dict):
            continue
        if "verdict_score" in sol:
            solution_scores.append(float(sol["verdict_score"]))
        elif "oracle_correct" in sol:
            solution_scores.append(1.0 if sol["oracle_correct"] else 0.0)
    return solution_scores


def _mixed_verdict(scores: Sequence[float]) -> bool:
    if not scores:
        return False
    return any(score > 0.0 for score in scores) and any(score < 1.0 for score in scores)


def compute_question_reward(row: Dict[str, Any]) -> float:
    if not _row_accepted(row):
        return 0.0
    if "question_reward" in row:
        return float(row["question_reward"])
    rewards = row.get("question_rewards")
    if isinstance(rewards, dict) and "question_reward" in rewards:
        return float(rewards["question_reward"])

    scores = _solution_verdict_scores(row)
    if not scores:
        return 0.0
    correct_rate = sum(scores) / len(scores)
    mixed_bonus = 1.0 if _mixed_verdict(scores) else 0.0
    reliability = row.get("reliability")
    stability = 1.0
    if isinstance(reliability, dict):
        stability = float(reliability.get("preference_stability", 1.0))
    # Reward questions that are accepted, verifier-stable, and induce non-degenerate solver outcomes.
    return max(0.0, min(1.0, stability * (0.75 * mixed_bonus + 0.25 * correct_rate)))


def normalize_group_advantages(rewards: Sequence[float]) -> List[float]:
    if not rewards:
        return []
    group_mean = sum(float(x) for x in rewards) / len(rewards)
    return [float(x) - group_mean for x in rewards]


def load_offline_examples(cfg: Dict[str, Any]) -> List[ProposerExample]:
    data_cfg = cfg.get("data", {})
    paths = data_cfg.get("rollout_jsonl_paths") or data_cfg.get("rollout_jsonl_path")
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("data.rollout_jsonl_paths must contain at least one JSONL path")

    seen: set[str] = set()
    examples: List[ProposerExample] = []
    for path_str in paths:
        path = Path(path_str)
        for row_idx, row in enumerate(_read_jsonl(path)):
            question = str(row.get("question") or _parse_question(str(row.get("generator_raw_output", ""))))
            canonical = _canonical_question(question)
            duplicate = canonical in seen
            seen.add(canonical)
            accepted = _row_accepted(row)
            reward = 0.0 if not accepted else compute_question_reward(row)
            scores = _solution_verdict_scores(row)
            examples.append(
                ProposerExample(
                    prompt=_row_prompt(row, cfg),
                    question=question,
                    group_id=_row_group_id(row, len(examples) if row_idx == 0 else row_idx),
                    question_reward=reward,
                    trainable_for_proposer=_row_trainable_for_proposer(row),
                    accepted=accepted,
                    duplicate=duplicate,
                    mixed_verdict=_mixed_verdict(scores),
                )
            )

    usable = [ex for ex in examples if ex.trainable_for_proposer]
    if not usable:
        raise ValueError("No trainable proposer examples found")
    return usable


def examples_with_group_advantages(examples: Sequence[ProposerExample]) -> List[Dict[str, Any]]:
    rewards_by_group: Dict[str, List[float]] = {}
    for ex in examples:
        rewards_by_group.setdefault(ex.group_id, []).append(ex.question_reward)
    offsets: Dict[str, int] = {group_id: 0 for group_id in rewards_by_group}
    advantages_by_group = {
        group_id: normalize_group_advantages(rewards) for group_id, rewards in rewards_by_group.items()
    }

    rows: List[Dict[str, Any]] = []
    for ex in examples:
        idx = offsets[ex.group_id]
        offsets[ex.group_id] += 1
        rows.append(
            {
                "prompt": ex.prompt,
                "question": ex.question,
                "group_id": ex.group_id,
                "question_reward": ex.question_reward,
                "question_advantage": advantages_by_group[ex.group_id][idx],
                "accepted": ex.accepted,
                "duplicate": ex.duplicate,
                "mixed_verdict": ex.mixed_verdict,
            }
        )
    return rows


def summarize_examples(examples: Sequence[ProposerExample]) -> Dict[str, float]:
    n = max(1, len(examples))
    rewards = [ex.question_reward for ex in examples]
    return {
        "num_examples": float(len(examples)),
        "acceptance_rate": sum(1.0 for ex in examples if ex.accepted) / n,
        "zero_reward_rate": sum(1.0 for ex in examples if ex.question_reward == 0.0) / n,
        "duplicate_rate": sum(1.0 for ex in examples if ex.duplicate) / n,
        "mixed_verdict_rate": sum(1.0 for ex in examples if ex.mixed_verdict) / n,
        "mean_question_reward": mean(rewards) if rewards else 0.0,
    }


def validate_config(cfg: Dict[str, Any]) -> None:
    mode = str(cfg.get("data", {}).get("mode", "offline"))
    if mode not in {"offline", "fixture", "online"}:
        raise ValueError("data.mode must be one of: offline, fixture, online")
    if mode == "online":
        raise ValueError("online proposer GRPO requires Agent 2-4 provider interfaces; use offline mode locally")
    for section in ("model", "data", "prompt", "rollout", "train"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing required config section: {section}")
    if int(cfg["rollout"].get("k", 0)) < 2:
        raise ValueError("rollout.k must be >= 2 for proposer GRPO")
    if int(cfg["train"].get("steps", 0)) < 0:
        raise ValueError("train.steps must be >= 0")
    if not cfg["data"].get("rollout_jsonl_paths") and not cfg["data"].get("rollout_jsonl_path"):
        raise ValueError("data.rollout_jsonl_paths is required for offline proposer GRPO")


def make_dataset(rows: Sequence[Dict[str, Any]]) -> Any:
    if Dataset is None:
        raise RuntimeError("datasets is required for training")
    return Dataset.from_list(list(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    validate_config(cfg)
    examples = load_offline_examples(cfg)
    if args.max_train_samples is not None:
        examples = examples[: args.max_train_samples]
    rows = examples_with_group_advantages(examples)
    summary = summarize_examples(examples)

    if args.validate_only:
        print(json.dumps({"schema_version": SCHEMA_VERSION, **summary}, sort_keys=True))
        return

    if AutoTokenizer is None or GRPOConfig is None or GRPOTrainer is None:
        raise RuntimeError("transformers and trl are required for proposer GRPO training")

    model_name = str(cfg["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds = make_dataset(rows)

    reward_by_question = {row["question"]: float(row["question_reward"]) for row in rows}

    def reward_question(*, completions: List[str], **_: Any) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            question = _parse_question(completion)
            rewards.append(float(reward_by_question.get(question, 0.0)))
        return rewards

    out_dir = str(args.output_dir or cfg["train"]["output_dir"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    prompts_per_step = int(cfg["train"].get("prompts_per_step", 1))
    per_device_bsz = max(1, prompts_per_step // max(1, world_size))
    wandb_cfg = cfg.get("train", {}).get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "grpo-math")))
        if wandb_cfg.get("run_name"):
            os.environ.setdefault("WANDB_NAME", str(wandb_cfg["run_name"]))

    grpo_args = GRPOConfig(
        output_dir=out_dir,
        do_train=True,
        learning_rate=float(cfg["train"]["lr"]),
        per_device_train_batch_size=per_device_bsz,
        gradient_accumulation_steps=int(cfg["train"].get("grad_accum_steps", 1)),
        bf16=True if cfg["model"].get("torch_dtype", "bfloat16") in ("bf16", "bfloat16") else None,
        num_train_epochs=1,
        max_steps=int(args.max_steps) if args.max_steps is not None else int(cfg["train"]["steps"]),
        logging_steps=int(cfg["train"].get("log_every", 1)),
        save_steps=int(cfg["train"].get("save_every", 200)),
        save_strategy="steps",
        report_to=["wandb"] if wandb_enabled else [],
        run_name=wandb_cfg.get("run_name"),
        num_generations=int(cfg["rollout"]["k"]),
        max_completion_length=int(cfg["rollout"]["max_new_tokens"]),
        temperature=float(cfg["rollout"]["temperature"]),
        top_p=float(cfg["rollout"]["top_p"]),
        beta=float(cfg["train"]["kl_beta"]),
        model_init_kwargs={"torch_dtype": _torch_dtype(cfg["model"].get("torch_dtype", "bfloat16"))},
        reward_weights=[1.0],
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_question],
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
