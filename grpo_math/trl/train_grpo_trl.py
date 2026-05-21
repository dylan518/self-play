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
from grpo_math.data.reward import (
    canonicalize_final_answer_text,
    extract_final_answer_int,
    extract_final_answer_int_strict,
    extract_ground_truth_int,
    final_answer_tail_char_count,
)
from grpo_math.self_play.generate_pairwise_data import _openai_generate_texts as _rollout_openai_generate_texts
from grpo_math.self_play.question_rewards import compute_question_reward
from grpo_math.self_play.question_bank import render_question_bank_block_from_config

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", flags=re.IGNORECASE)
_BOOL_VERDICT_RE = re.compile(r"\b(TRUE|FALSE)\b", flags=re.IGNORECASE)
_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_CONFIRM_LINE_RE = re.compile(r"CONFIRM\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
_CONFIRM_RE = _CONFIRM_LINE_RE
_DIFFERENT_FROM_SAMPLES_RE = re.compile(r"different_from_samples\s*=\s*(yes|no)", flags=re.IGNORECASE)


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
    if data_source == "question_generation":
        return _make_question_generation_dataset(cfg, split=split, max_samples=max_samples)

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
            filter_result = row.get("filter_result")
            if row.get("trainable_for_solver") is False or row.get("accepted") is False:
                continue
            if isinstance(filter_result, dict) and filter_result.get("accepted") is False:
                continue
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


def _trim_dataset_for_generation_groups(ds: Dataset, num_generations: int) -> Dataset:
    """Keep train/eval dataset lengths compatible with TRL GRPO grouping."""
    k = max(1, int(num_generations))
    n = len(ds)
    if n == 0 or n % k == 0:
        return ds
    trimmed = n - (n % k)
    if trimmed <= 0:
        # With fewer prompts than k, keep the data and let batch sizing below use
        # a single group. This is mainly for tiny debug/eval calls.
        return ds
    return ds.select(range(trimmed))


def _make_question_generation_dataset(cfg: Dict[str, Any], split: str, max_samples: int | None) -> Dataset:
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split for question_generation: {split!r}")
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt"), dict) else {}
    seed = int(cfg.get("seed", 1234))
    prompt_template = str(prompt_cfg.get("template", "Output exactly one line:\nQUESTION: <question text>"))
    prompt_template_path = str(prompt_cfg.get("template_path", "")).strip()
    if prompt_template_path:
        prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
    question_bank_examples = render_question_bank_block_from_config(
        {
            "question_bank": {
                "enabled": True,
                "path": str(data_cfg.get("question_bank_path", "examples.json")),
                "num_examples": int(data_cfg.get("num_examples", 3)),
                "seed": int(data_cfg.get("question_bank_seed", seed + (0 if split == "train" else 10_000))),
                "compatible_only": bool(data_cfg.get("compatible_only", False)),
            }
        },
        seed=seed + (0 if split == "train" else 10_000),
    )
    num_prompts = int(data_cfg.get("num_prompts", 64))
    if split == "eval":
        num_prompts = max(1, int(data_cfg.get("num_eval_prompts", max(1, num_prompts // 10))))
    if max_samples is not None:
        num_prompts = min(num_prompts, max(0, int(max_samples)))
    rows = []
    for idx in range(num_prompts):
        rows.append(
            {
                "prompt": prompt_template.format(question_bank_examples=question_bank_examples),
                "question_seed": str(seed + idx),
            }
        )
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


def _extract_generated_question(text: str) -> str | None:
    raw = str(text or "").strip()
    m = _QUESTION_RE.search(raw)
    q = m.group(1).strip() if m else raw
    if not q:
        return None
    nonempty_lines = [line.strip() for line in q.splitlines() if line.strip()]
    if not nonempty_lines:
        return None
    q = nonempty_lines[0]
    q = re.sub(r"^\s*[-*\d.)]+\s*", "", q).strip()
    q = re.sub(r"^QUESTION\s*:\s*", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s*CONFIRM\s*:.*$", "", q, flags=re.IGNORECASE | re.DOTALL).strip()
    return q if len(q) >= 12 else None


def _confirm_different_from_samples(text: str) -> bool:
    m = _CONFIRM_LINE_RE.search(str(text or ""))
    if not m:
        return False
    confirm = m.group(1).strip()
    dm = _DIFFERENT_FROM_SAMPLES_RE.search(confirm)
    integer_ok = re.search(r"integer_answer\s*=\s*yes", confirm, flags=re.IGNORECASE) is not None
    self_correction = re.search(
        r"\b(wait|revised\s+question|better\s+question|answer\s+is|final_answer)\b",
        confirm,
        flags=re.IGNORECASE,
    )
    concise = len(confirm) <= 400 and "\n" not in confirm
    return bool(dm and dm.group(1).lower() == "yes" and integer_ok and concise and not self_correction)


def _build_verdict_teacher_prompts(
    *,
    questions: List[str],
    completions: List[str],
    teacher_template: str,
    allow_lenient_answer: bool = False,
) -> tuple[List[int], List[str]]:
    valid_indices: List[int] = []
    verify_prompts: List[str] = []
    for idx, (q, c) in enumerate(zip(questions, completions, strict=True)):
        candidate_answer = canonicalize_final_answer_text(c)
        if candidate_answer is None:
            if not allow_lenient_answer:
                continue
            pred = extract_final_answer_int(c)
            if pred is None:
                continue
            candidate_answer = f"{c.rstrip()}\n\nFINAL_ANSWER: {pred}"
        valid_indices.append(idx)
        verify_prompts.append(teacher_template.format(question=q, candidate_answer=candidate_answer))
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
    extra_body: Dict[str, Any] | None = None,
    max_parallel: int = 1,
) -> List[str]:
    return _rollout_openai_generate_texts(
        prompts=prompts,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        timeout_s=timeout_s,
        max_tokens_param=max_tokens_param,
        extra_body=extra_body,
        include_reasoning=False,
        max_retries=max_retries,
        initial_backoff_s=initial_backoff_s,
        max_parallel=max_parallel,
    )


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
            canonical = canonicalize_final_answer_text(c)
            pred = extract_final_answer_int_strict(canonical or c)
            gt = extract_ground_truth_int(gt_text)
            out.append(1.0 if (pred is not None and gt is not None and pred == gt) else 0.0)
        return out

    def reward_format(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        # Format-only metric: did the model produce a parseable FINAL_ANSWER?
        allow_lenient = bool(cfg.get("reward", {}).get("format_lenient_final_answer", False))
        out: List[float] = []
        for c in completions:
            tail_chars = final_answer_tail_char_count(c)
            if tail_chars is not None:
                out.append(1.0 if tail_chars == 0 else 0.5)
            elif allow_lenient and extract_final_answer_int(c) is not None:
                out.append(0.25)
            else:
                out.append(0.0)
        return out

    def reward_answer_boundary(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        del prompts
        rewards: List[float] = []
        for c in completions:
            tail_chars = final_answer_tail_char_count(c)
            if tail_chars is None:
                rewards.append(0.0)
            elif tail_chars == 0:
                rewards.append(1.0)
            else:
                rewards.append(max(0.0, 1.0 - min(tail_chars, 400) / 400.0))
        return rewards

    def reward_question_format(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        del prompts
        return [
            1.0 if (_extract_generated_question(c) is not None and _confirm_different_from_samples(c)) else 0.0
            for c in completions
        ]

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
            allow_lenient_answer=bool(cfg.get("reward", {}).get("verdict_lenient_final_answer", False)),
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
                extra_body=teacher_cfg.get("api_extra_body", None),
                max_parallel=int(teacher_cfg.get("api_max_parallel", 1)),
            )
            for idx, out in zip(valid_indices, judge_outputs, strict=True):
                rewards[idx] = 1.0 if _parse_verdict(out) == "CORRECT" else 0.0
        return rewards

    question_reward_cfg = cfg.get("reward", {}).get("question", {})

    def reward_question(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        del prompts
        solver_cfg = question_reward_cfg.get("solver", {})
        judge_cfg = question_reward_cfg.get("judge", {})
        solver_api_key = _resolve_api_key(solver_cfg)
        judge_api_key = _resolve_api_key(judge_cfg)
        if not solver_api_key or not judge_api_key:
            raise RuntimeError("reward.mode=question requires reward.question.solver/judge API keys.")

        extracted_questions = [_extract_generated_question(c) for c in completions]
        confirm_ok = [_confirm_different_from_samples(c) for c in completions]
        rewards = [0.0 for _ in completions]
        confirm_soft_reward = float(question_reward_cfg.get("confirm_soft_reward", 0.2))
        for idx, q in enumerate(extracted_questions):
            if q and not confirm_ok[idx]:
                rewards[idx] = max(0.0, min(1.0, confirm_soft_reward))
        valid_items = [(idx, q) for idx, q in enumerate(extracted_questions) if q and confirm_ok[idx]]
        if not valid_items:
            return rewards

        num_solutions = int(question_reward_cfg.get("num_solutions_per_question", 4))
        repeats = int(question_reward_cfg.get("judge_repeats_per_question", 3))
        solver_prompts: list[str] = []
        solver_owner: list[int] = []
        solver_prompt_template = str(
            question_reward_cfg.get(
                "solver_prompt_template",
                "Question:\n{question}\n\nPut the final answer first.\nFINAL_ANSWER: <integer>",
            )
        )
        for idx, question_text in valid_items:
            for _rep in range(num_solutions):
                solver_owner.append(idx)
                solver_prompts.append(solver_prompt_template.format(question=question_text))

        solver_outputs = _rollout_openai_generate_texts(
            prompts=solver_prompts,
            model=str(solver_cfg.get("api_model", "Qwen/Qwen3.5-9B")),
            api_key=solver_api_key,
            base_url=str(solver_cfg.get("api_base_url", "http://127.0.0.1:8001/v1")),
            temperature=float(solver_cfg.get("temperature", 1.2)),
            top_p=float(solver_cfg.get("top_p", 0.98)),
            max_completion_tokens=int(solver_cfg.get("max_new_tokens", 2048)),
            timeout_s=float(solver_cfg.get("api_timeout_s", 180.0)),
            max_tokens_param=str(solver_cfg.get("api_max_tokens_param", "max_tokens")),
            reasoning_effort=str(solver_cfg.get("api_reasoning_effort", "")),
            extra_body=solver_cfg.get("api_extra_body", None),
            include_reasoning=bool(solver_cfg.get("include_reasoning_in_output", False)),
            tokenizer_name_or_path=str(solver_cfg.get("api_tokenizer_name_or_path", "")),
            final_max_tokens=int(solver_cfg["api_final_max_tokens"]) if solver_cfg.get("api_final_max_tokens") is not None else None,
            final_prefill=str(solver_cfg.get("api_final_prefill", "FINAL_ANSWER: ")),
            min_interval_s=float(solver_cfg.get("api_min_interval_s", 0.0)),
            max_retries=int(solver_cfg.get("api_max_retries", 4)),
            initial_backoff_s=float(solver_cfg.get("api_backoff_initial_s", 0.5)),
            max_parallel=int(solver_cfg.get("api_max_parallel", 4)),
        )

        judge_prompts: list[str] = []
        judge_owner: list[int] = []
        verify_template_path = str(judge_cfg.get("verify_prompt_template_path", "")).strip()
        if verify_template_path:
            verify_template = Path(verify_template_path).read_text(encoding="utf-8")
        else:
            verify_template = str(
                judge_cfg.get(
                    "verify_prompt_template",
                    "Question:\n{question}\n\nCandidate answer:\n{candidate_answer}\n\nVERDICT: CORRECT or INCORRECT",
                )
            )
        for owner, solution in zip(solver_owner, solver_outputs, strict=True):
            if extract_final_answer_int_strict(solution) is None:
                continue
            q = extracted_questions[owner]
            if not q:
                continue
            for _rep in range(repeats):
                judge_owner.append(owner)
                judge_prompts.append(verify_template.format(question=q, candidate_answer=solution))

        verdicts_by_idx: dict[int, list[str]] = {idx: [] for idx, _q in valid_items}
        if judge_prompts:
            judge_outputs = _rollout_openai_generate_texts(
                prompts=judge_prompts,
                model=str(judge_cfg.get("api_model", "Qwen/Qwen3.5-9B")),
                api_key=judge_api_key,
                base_url=str(judge_cfg.get("api_base_url", "http://127.0.0.1:8001/v1")),
                temperature=float(judge_cfg.get("temperature", 0.0)),
                top_p=float(judge_cfg.get("top_p", 1.0)),
                max_completion_tokens=int(judge_cfg.get("max_new_tokens", 256)),
                timeout_s=float(judge_cfg.get("api_timeout_s", 180.0)),
                max_tokens_param=str(judge_cfg.get("api_max_tokens_param", "max_tokens")),
                reasoning_effort=str(judge_cfg.get("api_reasoning_effort", "")),
                extra_body=judge_cfg.get("api_extra_body", None),
                max_retries=int(judge_cfg.get("api_max_retries", 4)),
                initial_backoff_s=float(judge_cfg.get("api_backoff_initial_s", 0.5)),
                max_parallel=int(judge_cfg.get("api_max_parallel", 4)),
            )
            for owner, judge_output in zip(judge_owner, judge_outputs, strict=True):
                verdicts_by_idx.setdefault(owner, []).append(_parse_verdict(judge_output))

        for idx, question_text in valid_items:
            del question_text
            breakdown = compute_question_reward(
                accepted=True,
                verdicts=verdicts_by_idx.get(idx, []),
                lambda_verifiability=float(question_reward_cfg.get("lambda_verifiability", 0.35)),
            )
            rewards[idx] = float(breakdown.reward)
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
    num_generations = int(cfg["rollout"]["k"])
    if reward_mode in {"verdict", "correctness"}:
        train_ds = _trim_dataset_for_generation_groups(train_ds, num_generations)
    # Keep per-device batch size sane for tiny debug runs as well.
    per_device_bsz_target = max(1, prompts_per_step // max(1, world_size))
    per_device_bsz_cap = max(1, len(train_ds) // max(1, world_size))
    per_device_bsz = max(1, min(per_device_bsz_target, per_device_bsz_cap))
    if reward_mode in {"verdict", "correctness"} and per_device_bsz >= num_generations:
        per_device_bsz -= per_device_bsz % num_generations
        per_device_bsz = max(num_generations, per_device_bsz)

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
        reward_funcs = [reward_verdict, reward_format, reward_answer_boundary]
        reward_weights = [
            float(cfg["reward"].get("verdict_weight", 1.0)),
            float(cfg["reward"].get("format_weight", 0.0)),
            float(cfg["reward"].get("answer_boundary_weight", 0.0)),
        ]
    elif reward_mode == "correctness":
        reward_funcs = [reward_correct, reward_format, reward_answer_boundary]
        reward_weights = [
            float(cfg["reward"].get("correctness_weight", 1.0)),
            float(cfg["reward"].get("format_weight", 0.0)),
            float(cfg["reward"].get("answer_boundary_weight", 0.0)),
        ]
    elif reward_mode == "question":
        reward_funcs = [reward_question, reward_question_format]
        reward_weights = [
            float(cfg["reward"].get("question_weight", 1.0)),
            float(cfg["reward"].get("question_format_weight", 0.1)),
        ]
    else:
        raise ValueError(
            f"Unsupported reward.mode: {reward_mode} (use 'correctness', 'verdict', or 'question')"
        )
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
        num_generations=num_generations,
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
        reward_weights=reward_weights,
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

