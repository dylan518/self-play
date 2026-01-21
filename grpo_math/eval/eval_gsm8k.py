from __future__ import annotations

import argparse
import random
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from grpo_math.data.gsm8k import batch_prompts, load_gsm8k
from grpo_math.models.policy import generate_completions, load_policy_and_ref
from grpo_math.data.reward import binary_reward, extract_final_answer_int_strict


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional HF checkpoint dir to evaluate.")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    set_seed(int(cfg.get("seed", 1234)))

    eval_ex = load_gsm8k(
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        split=cfg["data"]["split_eval"],
        max_samples=min(int(args.max_samples), int(cfg["data"].get("max_eval_samples", 10**9))),
    )
    # Shard evaluation examples across ranks to avoid duplicated work.
    # For exact prompt count + clean gathers, prefer max_samples divisible by num_processes.
    eval_ex = eval_ex[accelerator.process_index :: accelerator.num_processes]

    model_name = args.checkpoint or cfg["model"]["name_or_path"]
    bundle = load_policy_and_ref(
        name_or_path=model_name,
        torch_dtype=_torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")),
        gradient_checkpointing=False,
        load_ref=False,
    )
    model = bundle.policy
    tokenizer = bundle.tokenizer
    model.eval()

    model = accelerator.prepare(model)

    k = int(args.k or cfg["rollout"]["k"])
    temperature = float(cfg["rollout"]["temperature"])
    top_p = float(cfg["rollout"]["top_p"])
    max_new_tokens = int(cfg["rollout"]["max_new_tokens"])

    template = cfg["prompt"]["template"]
    # Evaluate in batches of prompts per rank
    bs = int(args.batch_size)
    total_prompts_local = 0
    total_rewards = []
    total_format_ok = []

    it = range(0, len(eval_ex), bs)
    it = tqdm(it, total=(len(eval_ex) + bs - 1) // bs, disable=not is_main, desc="eval batches")
    for i in it:
        batch = eval_ex[i : i + bs]
        prompts = batch_prompts(batch, template=template)
        gt_texts = [ex.answer_text for ex in batch]

        texts, _, _, _ = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            k=k,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        # Compute rewards locally (strict FINAL_ANSWER parsing).
        gt_rep = [gt for gt in gt_texts for _ in range(k)]
        rs = []
        fmt = []
        for t, gt in zip(texts, gt_rep, strict=True):
            r, _, _ = binary_reward(model_text=t, gsm8k_answer_text=gt)
            rs.append(float(r))
            fmt.append(1.0 if extract_final_answer_int_strict(t) is not None else 0.0)
        total_rewards.append(torch.tensor(rs, dtype=torch.float32))
        total_format_ok.append(torch.tensor(fmt, dtype=torch.float32))
        total_prompts_local += len(batch)

    rewards_cat = torch.cat(total_rewards, dim=0).to(accelerator.device)
    fmt_cat = torch.cat(total_format_ok, dim=0).to(accelerator.device)
    mean_reward = accelerator.gather_for_metrics(rewards_cat).float().mean().item()
    format_rate = accelerator.gather_for_metrics(fmt_cat).float().mean().item()
    # Compute pass@1/pass@k from tensors (exact across ranks since prompts are sharded).
    # rewards_cat is per-completion; reshape into [num_prompts_local, k] before gather.
    if total_prompts_local > 0:
        rewards_bk_local = rewards_cat.view(total_prompts_local, k)
        pass1_local = (rewards_bk_local[:, 0] > 0).to(torch.float32)
        passk_local = (rewards_bk_local.max(dim=1).values > 0).to(torch.float32)
    else:
        pass1_local = torch.zeros((0,), device=accelerator.device, dtype=torch.float32)
        passk_local = torch.zeros((0,), device=accelerator.device, dtype=torch.float32)

    pass1 = accelerator.gather_for_metrics(pass1_local).float().mean().item()
    passk = accelerator.gather_for_metrics(passk_local).float().mean().item()
    prompts_total = int(accelerator.gather_for_metrics(torch.tensor([total_prompts_local], device=accelerator.device)).sum().item())

    if is_main:
        print(
            f"mean_reward={mean_reward:.4f} format_rate={format_rate:.4f} "
            f"pass@1={pass1:.4f} pass@{k}={passk:.4f} prompts={prompts_total}"
        )

    # Avoid NCCL warning on exit
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

