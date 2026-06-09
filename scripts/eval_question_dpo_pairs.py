#!/usr/bin/env python3
"""Rebuild DPO preference pairs from a rollout JSONL and evaluate a checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from grpo_math.trl.train_question_from_rollout_rewards import (
    _read_jsonl,
    build_preference_rows,
    save_preference_pairs_jsonl,
    split_preference_pairs,
)


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _prompt_fallback(cfg: dict[str, Any]) -> str:
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt"), dict) else {}
    prompt_fallback = str(prompt_cfg.get("template", "Output exactly one line:\nQUESTION: <question text>"))
    template_path = str(prompt_cfg.get("template_path", "")).strip()
    if template_path:
        prompt_fallback = Path(template_path).read_text(encoding="utf-8")
    return prompt_fallback


def build_pairs_from_rollout(cfg: dict[str, Any], rollout_jsonl: Path) -> list[dict[str, str]]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    return build_preference_rows(
        _read_jsonl(rollout_jsonl),
        prompt_fallback=_prompt_fallback(cfg),
        min_reward_gap=float(data_cfg.get("min_reward_gap", 0.05)),
        max_pairs_per_group=int(data_cfg.get("max_pairs_per_group", 256)),
        seed=int(cfg.get("seed", 1234)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Question DPO YAML (pair-building hyperparams).")
    ap.add_argument("--rollout-jsonl", default=None, help="cycle_*_samples.jsonl used to build pairs.")
    ap.add_argument("--pairs-jsonl", default=None, help="Pre-built eval pairs (skips rollout rebuild).")
    ap.add_argument("--checkpoint", required=True, help="LoRA / full checkpoint directory.")
    ap.add_argument(
        "--ref-checkpoint",
        default=None,
        help="Policy LoRA checkpoint to load into frozen ref adapter (DPO train-start reference).",
    )
    ap.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of pairs for eval; use 1.0 for a full held-out rollout (e.g. another cycle).",
    )
    ap.add_argument("--eval-seed-offset", type=int, default=1, help="Must match training split (seed+offset).")
    ap.add_argument("--save-pairs-dir", default=None, help="Write preference_pairs_{train,eval}.jsonl here.")
    ap.add_argument("--output-json", default=None, help="Write metrics JSON to this path.")
    ap.add_argument("--per-device-eval-batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}

    if args.pairs_jsonl:
        eval_pairs = _read_jsonl(Path(args.pairs_jsonl))
        train_pairs: list[dict[str, str]] = []
    else:
        if not args.rollout_jsonl:
            raise SystemExit("Provide --rollout-jsonl or --pairs-jsonl")
        all_pairs = build_pairs_from_rollout(cfg, Path(args.rollout_jsonl))
        pair_seed = int(cfg.get("seed", 1234))
        train_pairs, eval_pairs = split_preference_pairs(
            all_pairs,
            eval_fraction=float(args.eval_fraction),
            seed=pair_seed + int(args.eval_seed_offset),
        )

    if not eval_pairs:
        raise ValueError("No eval pairs (increase --eval-fraction or check rollout JSONL).")

    if args.save_pairs_dir:
        out_dir = Path(args.save_pairs_dir)
        if train_pairs:
            save_preference_pairs_jsonl(train_pairs, out_dir / "preference_pairs_train.jsonl")
        save_preference_pairs_jsonl(eval_pairs, out_dir / "preference_pairs_eval.jsonl")

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("trl with DPOTrainer is required.") from exc

    from grpo_math.trl.train_grpo_trl import (
        _freeze_ref_adapter_for_dpo_eval,
        _load_adapter_model_if_needed,
        _load_policy_adapter_into_ref_adapter,
        _torch_dtype,
        _unwrap_trainer_model,
    )

    model_for_trainer, tokenizer_source, loaded_adapter = _load_adapter_model_if_needed(
        str(args.checkpoint),
        torch_dtype=_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
        attn_impl="sdpa",
        is_trainable=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    import inspect

    dpo_kwargs: dict[str, Any] = {
        "output_dir": str(Path(args.checkpoint) / "dpo_eval_tmp"),
        "do_train": False,
        "do_eval": True,
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "bf16": model_cfg.get("torch_dtype", "bfloat16") in ("bf16", "bfloat16"),
        "beta": float(train_cfg.get("dpo_beta", 0.1)),
        "max_prompt_length": int(train_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 512)),
        "report_to": [],
    }
    supported = set(inspect.signature(DPOConfig.__init__).parameters)
    dpo_args = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in supported})
    trainer = DPOTrainer(
        model=model_for_trainer,
        ref_model=None,
        args=dpo_args,
        train_dataset=Dataset.from_list(eval_pairs[:1]),
        eval_dataset=Dataset.from_list(eval_pairs),
        processing_class=tokenizer,
        peft_config=None,
    )
    ref_load_stats: dict[str, int] | None = None
    if loaded_adapter and args.ref_checkpoint:
        eval_model = _unwrap_trainer_model(trainer)
        ref_load_stats = _load_policy_adapter_into_ref_adapter(
            eval_model, Path(args.ref_checkpoint)
        )
        _freeze_ref_adapter_for_dpo_eval(eval_model)
    metrics = trainer.evaluate()
    summary = {
        "checkpoint": str(args.checkpoint),
        "ref_checkpoint": args.ref_checkpoint,
        "rollout_jsonl": args.rollout_jsonl,
        "pairs_jsonl": args.pairs_jsonl,
        "eval_fraction": float(args.eval_fraction),
        "eval_mode": "all_pairs" if float(args.eval_fraction) >= 1.0 else "split",
        "eval_pairs": len(eval_pairs),
        "train_pairs": len(train_pairs),
        "min_reward_gap": float(data_cfg.get("min_reward_gap", 0.05)),
        "ref_adapter_load": ref_load_stats,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
