from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import yaml

from grpo_math.data.reward import extract_final_answer_int_strict
from grpo_math.self_play.verifier import (
    CachedVerifier,
    FixtureVerifier,
    OpenAICompatibleVerifier,
    PythonAssistedVerifier,
    VerificationResult,
    Verifier,
    VerifierCache,
    VerifierVerdict,
    parse_verifier_response,
)


@dataclass(frozen=True)
class GeneratedQuestionExample:
    prompt: str
    question: str
    source_path: str
    source_index: int


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    required_sections = ("model", "data", "prompt", "rollout", "verifier", "train")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    data = cfg["data"]
    if not data.get("train_jsonl_paths"):
        raise ValueError("data.train_jsonl_paths must contain at least one rollout JSONL path")
    if not data.get("eval_jsonl_paths"):
        raise ValueError("data.eval_jsonl_paths must contain at least one rollout JSONL path")

    prompt_template = cfg["prompt"].get("template")
    if not isinstance(prompt_template, str) or "{question}" not in prompt_template:
        raise ValueError("prompt.template must be a string containing {question}")

    k = int(cfg["rollout"].get("k", 0))
    if k < 2:
        raise ValueError("rollout.k must be at least 2 for GRPO")

    provider = str(cfg["verifier"].get("provider", "fixture"))
    if provider not in {"fixture", "openai_compatible"}:
        raise ValueError("verifier.provider must be fixture or openai_compatible")
    if provider == "fixture" and not cfg["verifier"].get("fixture_map"):
        raise ValueError("fixture verifier requires verifier.fixture_map")

    steps = int(cfg["train"].get("steps", 0))
    if steps < 1:
        raise ValueError("train.steps must be positive")


def parse_verdict(text: str) -> str:
    return parse_verifier_response(text)[0].value


def verdict_score(verdict: str | VerifierVerdict) -> float:
    return 1.0 if VerifierVerdict(verdict) == VerifierVerdict.CORRECT else 0.0


def format_reward(completion: str) -> float:
    return 1.0 if extract_final_answer_int_strict(completion) is not None else 0.0


def _is_accepted_row(row: Dict[str, Any]) -> bool:
    if row.get("trainable_for_solver") is False:
        return False
    if row.get("accepted") is False:
        return False
    filter_result = row.get("filter_result")
    if isinstance(filter_result, dict) and filter_result.get("accepted") is False:
        return False
    return bool(row.get("question"))


def load_generated_questions(
    paths: Iterable[str | Path],
    *,
    prompt_template: str,
    max_samples: int | None = None,
) -> List[GeneratedQuestionExample]:
    examples: List[GeneratedQuestionExample] = []
    for pathish in paths:
        path = Path(pathish)
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not _is_accepted_row(row):
                    continue
                question = str(row["question"]).strip()
                examples.append(
                    GeneratedQuestionExample(
                        prompt=prompt_template.format(question=question),
                        question=question,
                        source_path=str(path),
                        source_index=int(row.get("question_index", line_idx)),
                    )
                )
                if max_samples is not None and len(examples) >= max_samples:
                    return examples
    return examples


def _fixture_items(fixture_map: Dict[str, str]) -> Dict[tuple[str, str], str]:
    items: Dict[tuple[str, str], str] = {}
    for key, value in fixture_map.items():
        if "\n" in key:
            question, completion = key.split("\n", 1)
        else:
            question, completion = "", key
        items[(question, completion)] = value
    return items


class CompletionFallbackVerifier:
    def __init__(self, inner: Verifier, fixture_map: Dict[str, str]) -> None:
        self.inner = inner
        self.completion_fixtures = {key: value for key, value in fixture_map.items() if "\n" not in key}
        self.model = inner.model
        self.prompt_version = inner.prompt_version

    def verify(self, question: str, completion: str) -> VerificationResult:
        try:
            return self.inner.verify(question, completion)
        except KeyError:
            raw_text = self.completion_fixtures.get(completion, "VERDICT: UNCLEAR")
            verdict, confidence = parse_verifier_response(raw_text)
            return VerificationResult(
                verdict=verdict,
                raw_text=raw_text,
                confidence=confidence,
                model=self.model,
                prompt_version=self.prompt_version,
            )

    def verify_batch(self, requests):
        return [self.verify(req.question, req.completion) for req in requests]


def make_verifier(cfg: Dict[str, Any]) -> Verifier:
    provider = str(cfg.get("provider", "fixture"))
    if provider == "fixture":
        fixture_map = dict(cfg.get("fixture_map", {}))
        inner: Verifier = CompletionFallbackVerifier(
            FixtureVerifier(
                _fixture_items(fixture_map),
                model="fixture",
                prompt_version=str(cfg.get("prompt_version", "solver_verdict_grpo_v1")),
            ),
            fixture_map,
        )
    elif provider == "openai_compatible":
        api_key_env = str(cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set for verifier calls")
        inner = OpenAICompatibleVerifier(
            model=str(cfg["model"]),
            api_key=api_key,
            base_url=str(cfg.get("base_url", "https://api.openai.com/v1")),
            prompt_template=str(cfg["prompt_template"]),
            prompt_version=str(cfg.get("prompt_version", "solver_verdict_grpo_v1")),
            temperature=float(cfg.get("temperature", 0.0)),
            max_completion_tokens=int(cfg.get("max_tokens", cfg.get("max_completion_tokens", 32))),
            max_tokens_param=str(cfg.get("max_tokens_param", "max_completion_tokens")),
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            max_retries=int(cfg.get("max_retries", 6)),
        )
    else:
        raise ValueError(f"Unsupported verifier.provider: {provider}")
    if bool(cfg.get("python_assisted", False)):
        inner = PythonAssistedVerifier(inner, timeout_s=float(cfg.get("python_timeout_s", 5.0)))
    return CachedVerifier(inner, VerifierCache(cfg.get("cache_path")))


def build_reward_functions(
    verifier: Verifier,
) -> tuple[Callable[..., List[float]], Callable[..., List[float]]]:
    def reward_verdict(
        *,
        prompts: List[str],
        completions: List[str],
        question: List[str],
        **_: Any,
    ) -> List[float]:
        del prompts
        return [
            verdict_score(verifier.verify(q, completion).verdict)
            for q, completion in zip(question, completions, strict=True)
        ]

    def reward_format(*, prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        del prompts
        return [format_reward(completion) for completion in completions]

    reward_verdict.__name__ = "reward_verdict"
    reward_format.__name__ = "reward_format"
    return reward_verdict, reward_format


def build_datasets(cfg: Dict[str, Any]) -> tuple[Any, Any]:
    from datasets import Dataset

    prompt_template = str(cfg["prompt"]["template"])
    train_examples = load_generated_questions(
        cfg["data"]["train_jsonl_paths"],
        prompt_template=prompt_template,
        max_samples=cfg["data"].get("max_train_samples"),
    )
    eval_examples = load_generated_questions(
        cfg["data"]["eval_jsonl_paths"],
        prompt_template=prompt_template,
        max_samples=cfg["data"].get("max_eval_samples"),
    )
    if not train_examples:
        raise ValueError("No accepted training questions found")
    if not eval_examples:
        raise ValueError("No accepted eval questions found")
    return Dataset.from_list([e.__dict__ for e in train_examples]), Dataset.from_list(
        [e.__dict__ for e in eval_examples]
    )


def _torch_dtype(name: str) -> Any:
    import torch

    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def train(cfg: Dict[str, Any], *, output_dir: str | None = None, max_steps: int | None = None) -> None:
    import torch
    from transformers import AutoTokenizer
    from transformers.trainer_callback import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    train_ds, eval_ds = build_datasets(cfg)
    model_name = str(cfg["model"]["name_or_path"])
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    reward_verdict, reward_format = build_reward_functions(make_verifier(cfg["verifier"]))

    class _MetricAliasesCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            del args, state, control, kwargs
            if not logs:
                return
            verdict_mean = logs.get("eval/rewards/reward_verdict/mean")
            if verdict_mean is not None and "eval/mean_reward" not in logs:
                logs["eval/mean_reward"] = verdict_mean
            format_mean = logs.get("eval/rewards/reward_format/mean")
            if format_mean is not None and "eval/format_rate" not in logs:
                logs["eval/format_rate"] = format_mean

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    prompts_per_step = int(cfg["train"]["prompts_per_step"])
    per_device_bsz = max(1, min(prompts_per_step // max(1, world_size), len(train_ds)))
    wandb_cfg = cfg.get("train", {}).get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "grpo-math")))
        if wandb_cfg.get("run_name"):
            os.environ.setdefault("WANDB_NAME", str(wandb_cfg["run_name"]))

    use_flash = bool(cfg.get("model", {}).get("use_flash_attn", False))
    attn_impl = "sdpa"
    if use_flash:
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"

    grpo_args = GRPOConfig(
        output_dir=str(output_dir or cfg["train"]["output_dir"]),
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
        num_train_epochs=1,
        max_steps=int(max_steps) if max_steps is not None else int(cfg["train"]["steps"]),
        logging_steps=int(cfg["train"].get("log_every", 1)),
        save_steps=int(cfg["train"].get("save_every", 200)),
        eval_steps=int(cfg["train"].get("eval_every", 200)),
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["wandb"] if wandb_enabled else [],
        run_name=wandb_cfg.get("run_name"),
        num_generations=int(cfg["rollout"]["k"]),
        num_generations_eval=1,
        max_completion_length=int(cfg["rollout"]["max_new_tokens"]),
        temperature=float(cfg["rollout"]["temperature"]),
        top_p=float(cfg["rollout"]["top_p"]),
        beta=float(cfg["train"]["kl_beta"]),
        model_init_kwargs={
            "torch_dtype": _torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")),
            "attn_implementation": attn_impl,
        },
        disable_dropout=True,
        log_completions=bool(cfg.get("train", {}).get("debug_rollouts", {}).get("enabled", False)),
        num_completions_to_print=int(cfg.get("train", {}).get("debug_rollouts", {}).get("max_prompts", 4)),
        reward_weights=[1.0, 0.0],
    )

    del torch
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_verdict, reward_format],
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )
    trainer.add_callback(_MetricAliasesCallback())
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    validate_config(cfg)
    if args.validate_only:
        print("Config valid. Server command:")
        print(f"python -m grpo_math.trl.train_solver_verdict_grpo --config {args.config}")
        return
    train(cfg, output_dir=args.output_dir, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
