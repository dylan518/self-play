from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from grpo_math.data.gsm8k import load_gsm8k
from grpo_math.data.reward import extract_final_answer_int_strict, extract_ground_truth_int


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_dtype(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _make_dataset(cfg: Dict[str, Any], split: str, max_samples: int | None) -> Dataset:
    ex = load_gsm8k(
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        split=split,
        max_samples=max_samples,
    )
    template = cfg["prompt"]["template"]

    rows = []
    for r in ex:
        prompt = template.format(question=r.question)
        rows.append({"prompt": prompt, "answer_text": r.answer_text})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=None, help="Optional override for train.max_steps")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    model_name = cfg["model"]["name_or_path"]
    use_flash = bool(cfg.get("model", {}).get("use_flash_attn", False))
    if use_flash:
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            # Fall back if flash-attn isn't installed in this env.
            attn_impl = "sdpa"
    else:
        attn_impl = "sdpa"

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    train_ds = _make_dataset(cfg, split=cfg["data"]["split_train"], max_samples=args.max_train_samples)
    eval_ds = _make_dataset(cfg, split=cfg["data"]["split_eval"], max_samples=args.max_eval_samples)

    # TRL GRPO calls reward funcs as:
    #   reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs)
    # where reward_kwargs contains the other dataset columns repeated to match num_generations.
    def reward_correct(*, prompts: List[str], completions: List[str], answer_text: List[str], **_: Any) -> List[float]:
        out: List[float] = []
        for c, gt_text in zip(completions, answer_text, strict=True):
            pred = extract_final_answer_int_strict(c)
            gt = extract_ground_truth_int(gt_text)
            out.append(1.0 if (pred is not None and gt is not None and pred == gt) else 0.0)
        return out

    def reward_format(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        # Format-only metric: did the model produce a parseable FINAL_ANSWER?
        out: List[float] = []
        for c in completions:
            out.append(1.0 if extract_final_answer_int_strict(c) is not None else 0.0)
        return out

    class _WandbMetricAliasesCallback(TrainerCallback):
        """
        TRL/Transformers logs per-reward-function eval metrics like:
          eval/rewards/reward_correct/mean
          eval/rewards/reward_format/mean
        Many people look for a single 'eval/mean_reward' scalar, so we alias it.
        """

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if not logs:
                return
            # Alias the "real" binary reward mean as eval/mean_reward for convenience.
            rc_mean = logs.get("eval/rewards/reward_correct/mean")
            if rc_mean is not None and "eval/mean_reward" not in logs:
                logs["eval/mean_reward"] = rc_mean
            # Alias format mean to eval/format_rate (it's a 0/1 rate).
            rf_mean = logs.get("eval/rewards/reward_format/mean")
            if rf_mean is not None and "eval/format_rate" not in logs:
                logs["eval/format_rate"] = rf_mean

    out_dir = str(args.output_dir or cfg["train"]["output_dir"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    prompts_per_step = int(cfg["train"]["prompts_per_step"])
    # Keep per-device batch size sane for tiny debug runs as well.
    per_device_bsz_target = max(1, prompts_per_step // max(1, world_size))
    per_device_bsz_cap = max(1, len(train_ds) // max(1, world_size))
    per_device_bsz = max(1, min(per_device_bsz_target, per_device_bsz_cap))

    wandb_cfg = cfg.get("train", {}).get("wandb", {}) if isinstance(cfg.get("train"), dict) else {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_project = str(wandb_cfg.get("project", "grpo-math"))
    wandb_run_name = wandb_cfg.get("run_name", None)
    # TRL/Transformers uses wandb.init defaults unless WANDB_PROJECT is set.
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", str(wandb_run_name))
    grpo_args = GRPOConfig(
        output_dir=out_dir,
        do_train=True,
        do_eval=True if int(cfg["train"].get("eval_every", 0)) > 0 else False,
        learning_rate=float(cfg["train"]["lr"]),
        lr_scheduler_type="cosine",
        warmup_steps=int(cfg["train"].get("warmup_steps", 0)),
        per_device_train_batch_size=per_device_bsz,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg["train"].get("grad_accum_steps", 1)),
        bf16=True if cfg["model"].get("torch_dtype", "bfloat16") in ("bf16", "bfloat16") else None,
        gradient_checkpointing=bool(cfg["train"].get("gradient_checkpointing", True)),
        num_train_epochs=1,  # we drive by max_steps
        max_steps=int(args.max_steps) if args.max_steps is not None else int(cfg["train"]["steps"]),
        logging_steps=1,
        save_steps=int(cfg["train"].get("save_every", 200)),
        eval_steps=int(cfg["train"].get("eval_every", 200)),
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["wandb"] if wandb_enabled else [],
        run_name=wandb_run_name,
        num_generations=int(cfg["rollout"]["k"]),
        # Keep eval cheap and avoid divisibility constraints on small world sizes.
        num_generations_eval=1,
        max_completion_length=int(cfg["rollout"]["max_new_tokens"]),
        temperature=float(cfg["rollout"]["temperature"]),
        top_p=float(cfg["rollout"]["top_p"]),
        beta=float(cfg["train"]["kl_beta"]),
        model_init_kwargs={"torch_dtype": _torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")), "attn_implementation": attn_impl},
        disable_dropout=True,
        # Helpful debugging: print a few completions periodically so we can see formatting issues.
        log_completions=bool(cfg.get("train", {}).get("debug_rollouts", {}).get("enabled", False)),
        num_completions_to_print=int(cfg.get("train", {}).get("debug_rollouts", {}).get("max_prompts", 4)),
        # Ensure the auxiliary format metric does not affect training reward.
        reward_weights=[1.0, 0.0],
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_correct, reward_format],
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )
    trainer.add_callback(_WandbMetricAliasesCallback())

    trainer.train()


if __name__ == "__main__":
    main()

