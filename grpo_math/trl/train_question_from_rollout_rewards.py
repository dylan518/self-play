from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            rows.append(obj)
    return rows


def _question_text(row: dict[str, Any]) -> str:
    raw = str(row.get("generator_raw_output") or "").strip()
    if raw:
        return raw
    question = str(row.get("question") or "").strip()
    if question.upper().startswith("QUESTION:"):
        return question
    return f"QUESTION: {question}"


def _reward(row: dict[str, Any]) -> float:
    val = row.get("question_reward")
    if isinstance(val, (int, float)):
        return float(val)
    rewards = row.get("question_rewards")
    if isinstance(rewards, dict):
        val = rewards.get("reward")
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _prompt(row: dict[str, Any], fallback: str) -> str:
    prompt = row.get("proposer_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return fallback


def _group_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("proposer_group_id", "group_id", "prompt_id", "run_id"):
        val = row.get(key)
        if val is not None:
            return str(val)
    return fallback


def build_preference_rows(
    rollout_rows: list[dict[str, Any]],
    *,
    prompt_fallback: str,
    min_reward_gap: float,
    max_pairs_per_group: int,
    seed: int,
) -> list[dict[str, str]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rollout_rows):
        if not row.get("trainable_for_proposer", True):
            continue
        if not str(row.get("question") or "").strip():
            continue
        by_group[_group_id(row, f"row-{idx}")].append(row)

    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    for group_rows in by_group.values():
        ordered = sorted(group_rows, key=_reward)
        group_pairs: list[dict[str, str]] = []
        for low_idx, rejected in enumerate(ordered):
            for chosen in ordered[low_idx + 1 :]:
                if _reward(chosen) - _reward(rejected) + 1e-12 < min_reward_gap:
                    continue
                group_pairs.append(
                    {
                        "prompt": _prompt(chosen, prompt_fallback),
                        "chosen": _question_text(chosen),
                        "rejected": _question_text(rejected),
                    }
                )
        rng.shuffle(group_pairs)
        if max_pairs_per_group > 0:
            group_pairs = group_pairs[:max_pairs_per_group]
        pairs.extend(group_pairs)
    rng.shuffle(pairs)
    return pairs


def split_preference_pairs(
    pairs: list[dict[str, str]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hold out a reproducible eval subset; remainder is train."""
    if not pairs:
        return [], []
    frac = float(eval_fraction)
    if frac <= 0.0:
        return list(pairs), []
    if frac >= 1.0:
        return [], list(pairs)
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_eval = max(1, int(round(len(pairs) * frac)))
    n_eval = min(n_eval, len(pairs) - 1) if len(pairs) > 1 else 1
    eval_idx = set(indices[:n_eval])
    train_pairs = [pairs[i] for i in indices[n_eval:]]
    eval_pairs = [pairs[i] for i in indices if i in eval_idx]
    return train_pairs, eval_pairs


def save_preference_pairs_jsonl(pairs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("trl with DPOTrainer is required for rollout-reward question training.") from exc

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    jsonl_path = Path(str(data_cfg.get("jsonl_path") or data_cfg.get("rollout_jsonl_path") or ""))
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Rollout JSONL not found: {jsonl_path}")

    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt"), dict) else {}
    prompt_fallback = str(prompt_cfg.get("template", "Output exactly one line:\nQUESTION: <question text>"))
    template_path = str(prompt_cfg.get("template_path", "")).strip()
    if template_path:
        prompt_fallback = Path(template_path).read_text(encoding="utf-8")

    pair_seed = int(cfg.get("seed", 1234))
    pairs = build_preference_rows(
        _read_jsonl(jsonl_path),
        prompt_fallback=prompt_fallback,
        min_reward_gap=float(data_cfg.get("min_reward_gap", 0.05)),
        max_pairs_per_group=int(data_cfg.get("max_pairs_per_group", 256)),
        seed=pair_seed,
    )
    if args.max_train_samples is not None:
        pairs = pairs[: max(0, int(args.max_train_samples))]
    if not pairs:
        raise ValueError(f"No preference pairs could be built from {jsonl_path}")

    eval_fraction = float(data_cfg.get("eval_fraction", 0.0))
    train_pairs, eval_pairs = split_preference_pairs(pairs, eval_fraction=eval_fraction, seed=pair_seed + 1)
    if eval_pairs and args.output_dir:
        out_root = Path(args.output_dir)
    elif eval_pairs:
        out_root = Path(str(train_cfg.get("output_dir", "outputs/question_dpo")))
    else:
        out_root = None
    if out_root is not None and eval_pairs:
        save_preference_pairs_jsonl(train_pairs, out_root / "preference_pairs_train.jsonl")
        save_preference_pairs_jsonl(eval_pairs, out_root / "preference_pairs_eval.jsonl")
    pairs = train_pairs if eval_pairs else pairs

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    from grpo_math.trl.train_grpo_trl import (
        _load_adapter_model_if_needed,
        _maybe_build_lora_config,
        _torch_dtype,
    )

    model_name = str(model_cfg["name_or_path"])
    attn_impl = "sdpa"
    if bool(model_cfg.get("use_flash_attn", False)):
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"
    model_for_trainer, tokenizer_source, loaded_adapter = _load_adapter_model_if_needed(
        model_name,
        torch_dtype=_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
        attn_impl=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = _maybe_build_lora_config(cfg)
    if loaded_adapter:
        peft_config = None

    wandb_cfg = train_cfg.get("wandb", {}) if isinstance(train_cfg.get("wandb"), dict) else {}
    if bool(wandb_cfg.get("enabled", False)):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "grpo-math")))
        if wandb_cfg.get("run_name"):
            os.environ.setdefault("WANDB_NAME", str(wandb_cfg["run_name"]))
        if wandb_cfg.get("group"):
            os.environ.setdefault("WANDB_RUN_GROUP", str(wandb_cfg["group"]))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch_target = int(train_cfg.get("prompts_per_step", 8))
    per_device_bsz = max(1, min(batch_target // max(1, world_size), len(pairs)))
    out_dir = str(args.output_dir or train_cfg["output_dir"])
    dpo_kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        "do_train": True,
        "learning_rate": float(train_cfg.get("lr", 5.0e-5)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "per_device_train_batch_size": per_device_bsz,
        "gradient_accumulation_steps": int(train_cfg.get("grad_accum_steps", 1)),
        "bf16": True if model_cfg.get("torch_dtype", "bfloat16") in ("bf16", "bfloat16") else False,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "max_steps": int(args.max_steps) if args.max_steps is not None else int(train_cfg.get("steps", 25)),
        "logging_steps": int(train_cfg.get("log_every", 1)),
        "save_steps": int(train_cfg.get("save_every", 25)),
        "save_strategy": "steps",
        "report_to": ["wandb"] if bool(wandb_cfg.get("enabled", False)) else [],
        "run_name": wandb_cfg.get("run_name"),
        "beta": float(train_cfg.get("dpo_beta", 0.1)),
        "max_prompt_length": int(train_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 512)),
        "model_init_kwargs": {"torch_dtype": _torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))},
    }
    import inspect

    supported = set(inspect.signature(DPOConfig.__init__).parameters)
    dpo_args = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in supported})
    eval_dataset = Dataset.from_list(eval_pairs) if eval_pairs else None
    if eval_dataset is not None:
        dpo_eval_kwargs = dict(dpo_kwargs)
        dpo_eval_kwargs["do_eval"] = True
        dpo_eval_kwargs["eval_strategy"] = "steps"
        dpo_eval_kwargs["eval_steps"] = int(train_cfg.get("eval_every", dpo_kwargs["save_steps"]))
        dpo_args = DPOConfig(**{k: v for k, v in dpo_eval_kwargs.items() if k in supported})

    trainer = DPOTrainer(
        model=model_for_trainer,
        ref_model=None,
        args=dpo_args,
        train_dataset=Dataset.from_list(pairs),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if peft_config is not None and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    print(
        json.dumps(
            {
                "rollout_jsonl": str(jsonl_path),
                "preference_pairs": len(pairs),
                "preference_pairs_eval": len(eval_pairs),
                "eval_fraction": eval_fraction,
                "output_dir": out_dir,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()
