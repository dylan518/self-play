from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from grpo_math.data.gsm8k import load_gsm8k
from grpo_math.data.reward import extract_final_answer_int_strict, extract_ground_truth_int

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", flags=re.IGNORECASE)
_BOOL_VERDICT_RE = re.compile(r"\b(TRUE|FALSE)\b", flags=re.IGNORECASE)


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
    data_source = str(cfg.get("data", {}).get("source", "gsm8k")).strip().lower()
    if data_source == "pairwise_jsonl":
        return _make_pairwise_jsonl_dataset(cfg, split=split, max_samples=max_samples)

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


def _make_pairwise_jsonl_dataset(cfg: Dict[str, Any], split: str, max_samples: int | None) -> Dataset:
    data_cfg = cfg.get("data", {})
    jsonl_path = Path(str(data_cfg["jsonl_path"]))
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Pairwise JSONL not found: {jsonl_path}")

    prompt_template = str(cfg.get("prompt", {}).get("template", "{question}"))
    seed = int(cfg.get("seed", 1234))
    eval_fraction = float(data_cfg.get("eval_fraction", 0.1))
    eval_fraction = min(max(eval_fraction, 0.0), 0.95)

    questions: List[str] = []
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            q = str(row.get("question", "")).strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            questions.append(q)

    if not questions:
        raise ValueError(f"No usable questions found in {jsonl_path}")

    rng = random.Random(seed)
    rng.shuffle(questions)
    split_idx = max(1, int(len(questions) * (1.0 - eval_fraction)))
    split_idx = min(split_idx, max(1, len(questions) - 1))
    if split == "train":
        selected = questions[:split_idx]
    elif split == "eval":
        selected = questions[split_idx:]
    else:
        raise ValueError(f"Unsupported split for pairwise_jsonl: {split!r} (use 'train' or 'eval')")

    if max_samples is not None:
        selected = selected[: max(0, int(max_samples))]

    rows = [{"prompt": prompt_template.format(question=q), "question": q} for q in selected]
    return Dataset.from_list(rows)


def _extract_chat_content(body: Dict[str, Any]) -> str:
    try:
        msg = body["choices"][0]["message"]
    except Exception as e:
        raise KeyError(f"Missing choices/message in response: {e}")
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "\n".join(parts).strip()
    alt = msg.get("text")
    if isinstance(alt, str):
        return alt.strip()
    out_txt = body.get("output_text")
    if isinstance(out_txt, str):
        return out_txt.strip()
    raise KeyError("Could not parse textual content from chat completion response.")


def _read_env_var_from_dotenv(var_name: str, dotenv_path: Path = Path(".env")) -> str | None:
    if not dotenv_path.exists():
        return None
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key != var_name:
                continue
            value = value.strip().strip('"').strip("'")
            return value if value else None
    except Exception:
        return None
    return None


def _resolve_api_key(role_cfg: Dict[str, Any]) -> str | None:
    explicit = str(role_cfg.get("api_key", "")).strip()
    if explicit:
        return explicit
    key_env = str(role_cfg.get("api_key_env", "")).strip()
    if key_env:
        return os.environ.get(key_env) or _read_env_var_from_dotenv(key_env)
    for env_name in ("OPENAI_API_KEY", "OPENAI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_name) or _read_env_var_from_dotenv(env_name)
        if val:
            return val
    return None


def _parse_verdict(text: str) -> str:
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper()
    m2 = _BOOL_VERDICT_RE.search(text)
    if m2:
        return "CORRECT" if m2.group(1).upper() == "TRUE" else "INCORRECT"
    return "INCORRECT"


def _build_verdict_teacher_prompts(
    *,
    questions: List[str],
    completions: List[str],
    teacher_template: str,
) -> tuple[List[int], List[str]]:
    valid_indices: List[int] = []
    verify_prompts: List[str] = []
    for idx, (q, c) in enumerate(zip(questions, completions, strict=True)):
        if extract_final_answer_int_strict(c) is None:
            continue
        valid_indices.append(idx)
        verify_prompts.append(teacher_template.format(question=q, candidate_answer=c))
    return valid_indices, verify_prompts


def _openai_generate_texts(
    *,
    prompts: List[str],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    timeout_s: float,
    max_tokens_param: str = "max_completion_tokens",
    max_retries: int = 6,
    initial_backoff_s: float = 1.0,
) -> List[str]:
    outputs: List[str] = []
    for prompt in prompts:
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload[max_tokens_param] = max_completion_tokens
        backoff_s = max(0.0, initial_backoff_s)
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(
                url=base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                outputs.append(_extract_chat_content(body))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt <= max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2.0, 30.0)
                    continue
                raise
            except Exception:
                if attempt <= max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2.0, 30.0)
                    continue
                raise
    return outputs


def _maybe_build_lora_config(cfg: Dict[str, Any]) -> Any | None:
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    lora_cfg = train_cfg.get("lora", {}) if isinstance(train_cfg.get("lora"), dict) else {}
    enabled = bool(lora_cfg.get("enabled", False))
    if not enabled:
        return None
    try:
        from peft import LoraConfig, TaskType
    except Exception as e:
        raise RuntimeError(
            "LoRA is enabled but `peft` is not installed. Install with `pip install peft`."
        ) from e

    target_modules = lora_cfg.get("target_modules", "all-linear")
    if isinstance(target_modules, str) and "," in target_modules:
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    modules_to_save = lora_cfg.get("modules_to_save", None)
    if isinstance(modules_to_save, str) and "," in modules_to_save:
        modules_to_save = [x.strip() for x in modules_to_save.split(",") if x.strip()]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def _load_adapter_model_if_needed(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    attn_impl: str,
) -> tuple[Any, str, bool]:
    """
    Returns:
      - model argument for GRPOTrainer (str model id/path or loaded model object)
      - tokenizer source path/model id
      - whether model_name_or_path was an adapter checkpoint
    """
    path = Path(str(model_name_or_path))
    if not path.exists():
        return model_name_or_path, model_name_or_path, False
    adapter_cfg_path = path / "adapter_config.json"
    model_cfg_path = path / "config.json"
    # LoRA checkpoints saved by PEFT commonly have adapter_config.json but no full model config.
    if not adapter_cfg_path.exists() or model_cfg_path.exists():
        return model_name_or_path, model_name_or_path, False

    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_model_name = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
    if not base_model_name:
        raise ValueError(
            f"Adapter checkpoint {path} is missing base_model_name_or_path in adapter_config.json"
        )
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "Adapter checkpoint detected but `peft` is not installed. Install with `pip install peft`."
        ) from e

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    adapter_model = PeftModel.from_pretrained(base_model, str(path), is_trainable=True)
    return adapter_model, base_model_name, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=None, help="Optional override for train.max_steps")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    model_name = str(cfg["model"]["name_or_path"])
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

    model_for_trainer, tokenizer_source, loaded_adapter_checkpoint = _load_adapter_model_if_needed(
        model_name,
        torch_dtype=_torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")),
        attn_impl=attn_impl,
    )
    if loaded_adapter_checkpoint:
        print(
            f"[train] loading LoRA adapter checkpoint for continued training: {model_name}",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    train_ds = _make_dataset(cfg, split=cfg["data"]["split_train"], max_samples=args.max_train_samples)
    eval_ds = _make_dataset(cfg, split=cfg["data"]["split_eval"], max_samples=args.max_eval_samples)
    reward_mode = str(cfg.get("reward", {}).get("mode", "correctness")).strip().lower()

    teacher_cfg = cfg.get("reward", {}).get("teacher", {})
    teacher_template = str(
        teacher_cfg.get(
            "verify_prompt_template",
            "Question:\n{question}\n\nCandidate answer:\n{candidate_answer}\n\nVERDICT: CORRECT or INCORRECT",
        )
    )
    teacher_template_path = str(teacher_cfg.get("verify_prompt_template_path", "")).strip()
    if teacher_template_path:
        with open(teacher_template_path, "r", encoding="utf-8") as f:
            teacher_template = f.read()
    teacher_api_model = str(teacher_cfg.get("api_model", "gpt-4.1"))
    teacher_base_url = str(teacher_cfg.get("api_base_url", "https://api.openai.com/v1"))
    teacher_api_key = _resolve_api_key(teacher_cfg)

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

    def reward_verdict(*, prompts: List[str], completions: List[str], question: List[str], **_: Any) -> List[float]:
        if not teacher_api_key:
            raise RuntimeError(
                "reward.mode=verdict requires reward.teacher.api_key/api_key_env (or OPENAI_API_KEY) to be set."
            )
        # Directly assign 0 reward to unparseable completions and skip teacher API calls.
        rewards: List[float] = [0.0 for _ in completions]
        valid_indices, verify_prompts = _build_verdict_teacher_prompts(
            questions=question,
            completions=completions,
            teacher_template=teacher_template,
        )
        if verify_prompts:
            judge_outputs = _openai_generate_texts(
                prompts=verify_prompts,
                model=teacher_api_model,
                api_key=teacher_api_key,
                base_url=teacher_base_url,
                temperature=float(teacher_cfg.get("temperature", 0.0)),
                top_p=float(teacher_cfg.get("top_p", 1.0)),
                max_completion_tokens=int(teacher_cfg.get("max_new_tokens", 256)),
                timeout_s=float(teacher_cfg.get("api_timeout_s", 120.0)),
                max_tokens_param=str(teacher_cfg.get("api_max_tokens_param", "max_completion_tokens")),
                max_retries=int(teacher_cfg.get("api_max_retries", 6)),
                initial_backoff_s=float(teacher_cfg.get("api_backoff_initial_s", 1.0)),
            )
            for idx, out in zip(valid_indices, judge_outputs, strict=True):
                rewards[idx] = 1.0 if _parse_verdict(out) == "CORRECT" else 0.0
        return rewards

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
            # Alias the primary reward mean as eval/mean_reward for convenience.
            for key in ("eval/rewards/reward_correct/mean", "eval/rewards/reward_verdict/mean"):
                val = logs.get(key)
                if val is not None and "eval/mean_reward" not in logs:
                    logs["eval/mean_reward"] = val
                    break
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
    wandb_group = wandb_cfg.get("group", None)
    wandb_tags = wandb_cfg.get("tags", None)
    # TRL/Transformers uses wandb.init defaults unless WANDB_PROJECT is set.
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", str(wandb_run_name))
        if wandb_group:
            os.environ.setdefault("WANDB_RUN_GROUP", str(wandb_group))
        if isinstance(wandb_tags, list) and wandb_tags:
            os.environ.setdefault("WANDB_TAGS", ",".join(str(x) for x in wandb_tags))
    if reward_mode == "verdict":
        reward_funcs = [reward_verdict, reward_format]
    elif reward_mode == "correctness":
        reward_funcs = [reward_correct, reward_format]
    else:
        raise ValueError(f"Unsupported reward.mode: {reward_mode} (use 'correctness' or 'verdict')")
    peft_config = _maybe_build_lora_config(cfg)
    # If model is already a loaded PEFT adapter checkpoint, do not wrap with a new peft_config.
    if loaded_adapter_checkpoint:
        peft_config = None
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
        bf16=True if cfg["model"].get("torch_dtype", "bfloat16") in ("bf16", "bfloat16") else False,
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
        model=model_for_trainer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    if peft_config is not None and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    trainer.add_callback(_WandbMetricAliasesCallback())

    trainer.train()


if __name__ == "__main__":
    main()

