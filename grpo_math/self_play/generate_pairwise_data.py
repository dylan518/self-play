from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_math.data.reward import extract_final_answer_int_strict


_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", flags=re.DOTALL)
_PREF_RE = re.compile(r"PREFERENCE:\s*(A|B)", flags=re.IGNORECASE)
_BOXED_INT_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_FINAL_TEXT_INT_RE = re.compile(r"(?:Final Answer|Final answer)\s*[:\-]?\s*(-?\d+)\b")
_ANY_INT_RE = re.compile(r"-?\d+")


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


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


def _attn_impl(use_flash_attn: bool) -> str:
    if not use_flash_attn:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _load_model_bundle(name_or_path: str, torch_dtype: torch.dtype, use_flash_attn: bool) -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs: Dict[str, Any] = {"dtype": torch_dtype, "attn_implementation": _attn_impl(use_flash_attn)}
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return ModelBundle(model=model, tokenizer=tok)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _apply_chat_template(tokenizer: Any, prompt: str, use_chat_template: bool, enable_thinking: bool | None) -> str:
    if not use_chat_template:
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        if enable_thinking is None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _generate_texts(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    device = next(bundle.model.parameters()).device
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        tok = bundle.tokenizer(list(chunk), return_tensors="pt", padding=True, truncation=True)
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        out = bundle.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
        completion_ids = out[:, input_ids.shape[1] :]
        outputs.extend(bundle.tokenizer.decode(row, skip_special_tokens=True).strip() for row in completion_ids)
    return outputs


def _parse_question(text: str) -> str:
    m = _QUESTION_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip().splitlines()[0].strip()
    # Enforce question-only output even if the model leaks answer fields.
    candidate = re.sub(r"\n?\s*(ANSWER|FINAL_ANSWER)\s*:.*$", "", candidate, flags=re.IGNORECASE | re.DOTALL)
    return candidate.strip()


def _parse_preference(text: str) -> str:
    m = _PREF_RE.search(text)
    if not m:
        # Avoid systematic positional bias on malformed judge output.
        return random.choice(["A", "B"])
    return m.group(1).upper()


def _parse_solver_final_answer(text: str) -> int | None:
    # Prefer strict training format when present.
    strict = extract_final_answer_int_strict(text)
    if strict is not None:
        return strict
    # Fallbacks for analysis-time metadata in pairwise rollouts.
    boxed_matches = _BOXED_INT_RE.findall(text)
    if boxed_matches:
        return int(boxed_matches[-1])
    final_text_matches = _FINAL_TEXT_INT_RE.findall(text)
    if final_text_matches:
        return int(final_text_matches[-1])
    return None


def _looks_like_question_only(raw_text: str, parsed_question: str) -> bool:
    if not parsed_question:
        return False
    if re.search(r"\b(ANSWER|FINAL_ANSWER)\s*:", raw_text, flags=re.IGNORECASE):
        return False
    return True


def _pair_indices(n: int, max_pairs: int | None) -> List[tuple[int, int]]:
    pairs = list(itertools.combinations(range(n), 2))
    if max_pairs is not None and max_pairs > 0 and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _elo_update(r_a: float, r_b: float, s_a: float, k: float) -> tuple[float, float]:
    e_a = _elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    new_a = r_a + k * (s_a - e_a)
    new_b = r_b + k * ((1.0 - s_a) - e_b)
    return new_a, new_b


def _parse_int_from_text(text: str) -> int | None:
    strict = extract_final_answer_int_strict(text)
    if strict is not None:
        return strict
    boxed_matches = _BOXED_INT_RE.findall(text)
    if boxed_matches:
        return int(boxed_matches[-1])
    final_text_matches = _FINAL_TEXT_INT_RE.findall(text)
    if final_text_matches:
        return int(final_text_matches[-1])
    int_matches = _ANY_INT_RE.findall(text)
    if int_matches:
        return int(int_matches[-1])
    return None


def _openai_oracle_answer(
    *,
    question: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    timeout_s: float = 60.0,
) -> tuple[int | None, str | None, str | None]:
    prompt = (
        "Solve the following math question.\n"
        "Return only one line in this exact format:\n"
        "FINAL_ANSWER: <integer>\n\n"
        f"Question: {question}\n"
    )
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a precise math solver."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
    }
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
            raw = resp.read().decode("utf-8")
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"]
        parsed = _parse_int_from_text(content)
        return parsed, content, None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        return None, None, f"HTTPError: {e.code} {detail}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="grpo_math/configs/pairwise_rollouts_smoke.yaml")
    args = ap.parse_args()
    cfg = _load_yaml(args.config)

    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)

    pair_cfg = cfg["pairwise"]
    gen_cfg = cfg["generator"]
    sol_cfg = cfg["solver"]
    judge_cfg = cfg["judge"]
    out_cfg = cfg["output"]
    strong_cfg = cfg.get("strong_verifier", {})

    gen_bundle = _load_model_bundle(
        gen_cfg["model_name_or_path"],
        _torch_dtype(gen_cfg.get("torch_dtype", "bfloat16")),
        bool(gen_cfg.get("use_flash_attn", False)),
    )
    sol_bundle = _load_model_bundle(
        sol_cfg["model_name_or_path"],
        _torch_dtype(sol_cfg.get("torch_dtype", "bfloat16")),
        bool(sol_cfg.get("use_flash_attn", False)),
    )
    judge_bundle = _load_model_bundle(
        judge_cfg["model_name_or_path"],
        _torch_dtype(judge_cfg.get("torch_dtype", "bfloat16")),
        bool(judge_cfg.get("use_flash_attn", False)),
    )

    question_template = _load_text(gen_cfg["prompt_template_path"])
    judge_template = _load_text(judge_cfg["prompt_template_path"])
    solver_template = str(sol_cfg["prompt_template"])

    out_path = Path(out_cfg["jsonl_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    include_judge_traces = bool(out_cfg.get("include_judge_traces", False))
    strong_enabled = bool(strong_cfg.get("enabled", False))
    strong_provider = str(strong_cfg.get("provider", "openai")).lower()
    strong_model = str(strong_cfg.get("model", "gpt-4.1"))
    strong_base_url = str(strong_cfg.get("base_url", "https://api.openai.com/v1"))

    num_questions = int(pair_cfg.get("num_questions", 1))
    num_solutions = int(sol_cfg.get("num_solutions_per_question", 4))
    repeats = int(judge_cfg.get("repeats_per_pair", 3))
    randomize_judge_order = bool(judge_cfg.get("randomize_a_b_order", True))
    balanced_judge_order = bool(judge_cfg.get("balanced_a_b_order", True))
    elo_k = float(pair_cfg.get("elo_k_factor", 24.0))
    elo_initial = float(pair_cfg.get("elo_initial_rating", 1000.0))
    max_pairs_per_question = pair_cfg.get("max_pairs_per_question", None)
    max_pairs_per_question = int(max_pairs_per_question) if max_pairs_per_question is not None else None
    max_retries = int(gen_cfg.get("max_retries_per_question", 3))
    sampling_groups_cfg = sol_cfg.get("sampling_groups")
    if sampling_groups_cfg:
        sampling_groups: List[Dict[str, float | int]] = []
        for grp in sampling_groups_cfg:
            count = int(grp["count"])
            if count <= 0:
                continue
            sampling_groups.append(
                {
                    "count": count,
                    "temperature": float(grp.get("temperature", sol_cfg.get("temperature", 0.8))),
                    "top_p": float(grp.get("top_p", sol_cfg.get("top_p", 0.9))),
                }
            )
        total_group_count = sum(int(g["count"]) for g in sampling_groups)
        if total_group_count != num_solutions:
            raise ValueError(
                f"solver.sampling_groups total count ({total_group_count}) must equal "
                f"solver.num_solutions_per_question ({num_solutions})."
            )
    else:
        sampling_groups = [
            {
                "count": num_solutions,
                "temperature": float(sol_cfg.get("temperature", 0.8)),
                "top_p": float(sol_cfg.get("top_p", 0.9)),
            }
        ]
    batch_solver_across_questions = bool(sol_cfg.get("batch_across_questions", True))

    raw_gen_prompt = question_template
    generator_prompt = _apply_chat_template(
        gen_bundle.tokenizer,
        raw_gen_prompt,
        bool(gen_cfg.get("use_chat_template", False)),
        gen_cfg.get("enable_thinking", None),
    )
    generator_prompts = [generator_prompt] * num_questions
    question_generations = _generate_texts(
        gen_bundle,
        generator_prompts,
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
        batch_size=int(gen_cfg.get("batch_size", 8)),
    )
    questions: List[str] = []
    cleaned_question_generations: List[str] = []
    for raw in question_generations:
        parsed = _parse_question(raw)
        tries = 0
        while (not _looks_like_question_only(raw, parsed)) and tries < max_retries:
            retry_outputs = _generate_texts(
                gen_bundle,
                [generator_prompt],
                temperature=float(gen_cfg.get("temperature", 1.0)),
                top_p=float(gen_cfg.get("top_p", 0.95)),
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
                batch_size=1,
            )
            raw = retry_outputs[0]
            parsed = _parse_question(raw)
            tries += 1
        questions.append(parsed)
        cleaned_question_generations.append(raw)

    solver_batch_size = int(sol_cfg.get("batch_size", 16))
    solver_max_new_tokens = int(sol_cfg.get("max_new_tokens", 512))

    def _generate_solver_outputs_for_prompt(solver_prompt: str) -> List[str]:
        outputs: List[str] = []
        for grp in sampling_groups:
            group_count = int(grp["count"])
            group_outputs = _generate_texts(
                sol_bundle,
                [solver_prompt] * group_count,
                temperature=float(grp["temperature"]),
                top_p=float(grp["top_p"]),
                max_new_tokens=solver_max_new_tokens,
                batch_size=solver_batch_size,
            )
            outputs.extend(group_outputs)
        return outputs

    # Optional global batching mode for throughput on short generations.
    solver_outputs_by_question: List[List[str]] = []
    if batch_solver_across_questions:
        raw_solver_prompts = [solver_template.format(question=q) for q in questions]
        solver_prompts = [
            _apply_chat_template(
                sol_bundle.tokenizer,
                raw_prompt,
                bool(sol_cfg.get("use_chat_template", False)),
                sol_cfg.get("enable_thinking", None),
            )
            for raw_prompt in raw_solver_prompts
        ]
        solver_outputs_by_question = [[] for _ in range(len(questions))]
        for grp in sampling_groups:
            prompts_for_group: List[str] = []
            prompt_question_indices: List[int] = []
            group_count = int(grp["count"])
            for q_idx, prompt in enumerate(solver_prompts):
                prompts_for_group.extend([prompt] * group_count)
                prompt_question_indices.extend([q_idx] * group_count)
            group_outputs = _generate_texts(
                sol_bundle,
                prompts_for_group,
                temperature=float(grp["temperature"]),
                top_p=float(grp["top_p"]),
                max_new_tokens=solver_max_new_tokens,
                batch_size=solver_batch_size,
            )
            for q_idx, output in zip(prompt_question_indices, group_outputs):
                solver_outputs_by_question[q_idx].append(output)

    with out_path.open("w", encoding="utf-8") as f:
        for q_idx, question in enumerate(tqdm(questions, desc="pairwise_rollouts", unit="q"), start=1):
            if batch_solver_across_questions:
                solver_outputs = solver_outputs_by_question[q_idx - 1]
            else:
                raw_solver_prompt = solver_template.format(question=question)
                solver_prompt = _apply_chat_template(
                    sol_bundle.tokenizer,
                    raw_solver_prompt,
                    bool(sol_cfg.get("use_chat_template", False)),
                    sol_cfg.get("enable_thinking", None),
                )
                solver_outputs = _generate_solver_outputs_for_prompt(solver_prompt)
            if len(solver_outputs) != num_solutions:
                raise RuntimeError(
                    f"Expected {num_solutions} solver outputs for question index {q_idx - 1}, "
                    f"got {len(solver_outputs)}."
                )

            pairs = _pair_indices(num_solutions, max_pairs_per_question)
            pairwise_rows = []
            score_points = [0.0 for _ in range(num_solutions)]
            score_counts = [0 for _ in range(num_solutions)]
            consistency_values: List[float] = []

            for i, j in pairs:
                judge_prompts: List[str] = []
                raw_judge_prompt_for_pair: str | None = None
                judge_presentations: List[Dict[str, int]] = []
                swap_offset = int(bool(random.getrandbits(1)))
                for rep_idx in range(repeats):
                    if not randomize_judge_order:
                        a_idx, b_idx = i, j
                    elif balanced_judge_order:
                        # Ensure near-equal A/B exposure per pair, which cancels position bias.
                        use_swapped = ((rep_idx + swap_offset) % 2 == 1)
                        a_idx, b_idx = (j, i) if use_swapped else (i, j)
                    else:
                        a_idx, b_idx = (j, i) if bool(random.getrandbits(1)) else (i, j)
                    raw_judge_prompt = judge_template.format(
                        question=question,
                        answer_a=solver_outputs[a_idx],
                        answer_b=solver_outputs[b_idx],
                    )
                    if raw_judge_prompt_for_pair is None:
                        raw_judge_prompt_for_pair = raw_judge_prompt
                    judge_presentations.append({"a_solution_index": a_idx, "b_solution_index": b_idx})
                    judge_prompt = _apply_chat_template(
                        judge_bundle.tokenizer,
                        raw_judge_prompt,
                        bool(judge_cfg.get("use_chat_template", False)),
                        judge_cfg.get("enable_thinking", None),
                    )
                    judge_prompts.append(judge_prompt)

                judge_outputs = _generate_texts(
                    judge_bundle,
                    judge_prompts,
                    temperature=float(judge_cfg.get("temperature", 0.4)),
                    top_p=float(judge_cfg.get("top_p", 0.9)),
                    max_new_tokens=int(judge_cfg.get("max_new_tokens", 64)),
                    batch_size=int(judge_cfg.get("batch_size", 32)),
                )
                raw_prefs = [_parse_preference(x) for x in judge_outputs]
                prefs: List[str] = []
                for raw_pref, presentation in zip(raw_prefs, judge_presentations):
                    if raw_pref == "A":
                        winner_idx = int(presentation["a_solution_index"])
                    else:
                        winner_idx = int(presentation["b_solution_index"])
                    # Keep legacy encoding: A means i won, B means j won.
                    prefs.append("A" if winner_idx == i else "B")
                n_a = prefs.count("A")
                n_b = prefs.count("B")
                n_t = 0
                consistency = max(n_a, n_b) / max(1, repeats)
                consistency_values.append(consistency)

                # Keep average fractional points for diagnostic metrics.
                a_points = (n_a + 0.5 * n_t) / max(1, repeats)
                b_points = (n_b + 0.5 * n_t) / max(1, repeats)
                score_points[i] += a_points
                score_points[j] += b_points
                score_counts[i] += 1
                score_counts[j] += 1

                pairwise_rows.append(
                    {
                        "i": i,
                        "j": j,
                        "prefs": prefs,
                        "counts": {"A": n_a, "B": n_b, "TIE": n_t},
                        "consistency": consistency,
                        "a_points": a_points,
                        "b_points": b_points,
                        **(
                            {
                                "judge_trace": {
                                    "raw_prompt": raw_judge_prompt_for_pair,
                                    "raw_outputs": judge_outputs,
                                    "presentations": judge_presentations,
                                    "raw_prefs": raw_prefs,
                                    "mapped_prefs": prefs,
                                }
                            }
                            if include_judge_traces
                            else {}
                        ),
                    }
                )

            avg_pairwise_scores = [
                (score_points[k] / score_counts[k]) if score_counts[k] > 0 else 0.5 for k in range(num_solutions)
            ]
            elo_ratings = [elo_initial for _ in range(num_solutions)]
            for pair in pairwise_rows:
                i = int(pair["i"])
                j = int(pair["j"])
                for pref in pair["prefs"]:
                    if pref == "A":
                        s_i = 1.0
                    elif pref == "B":
                        s_i = 0.0
                    else:
                        s_i = 0.5
                    elo_ratings[i], elo_ratings[j] = _elo_update(elo_ratings[i], elo_ratings[j], s_i, elo_k)

            oracle_answer: int | None = None
            oracle_raw_response: str | None = None
            oracle_error: str | None = None
            if strong_enabled and strong_provider == "openai":
                api_key = (
                    os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("OPENAI_KEY")
                    or _read_env_var_from_dotenv("OPENAI_API_KEY")
                    or _read_env_var_from_dotenv("OPENAI_KEY")
                )
                if not api_key:
                    oracle_error = "OPENAI_API_KEY (or OPENAI_KEY) is not set."
                else:
                    oracle_answer, oracle_raw_response, oracle_error = _openai_oracle_answer(
                        question=question,
                        model=strong_model,
                        api_key=api_key,
                        base_url=strong_base_url,
                        timeout_s=float(strong_cfg.get("timeout_s", 60.0)),
                    )

            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question_index": q_idx - 1,
                "question": question,
                "generator_raw_output": cleaned_question_generations[q_idx - 1],
                "solutions": [
                    {
                        "solution_index": s_idx,
                        "text": s_text,
                        "parsed_final_answer": (parsed := _parse_solver_final_answer(s_text)),
                        "pairwise_score": avg_pairwise_scores[s_idx],
                        "elo_rating": elo_ratings[s_idx],
                        **(
                            {
                                "oracle_correct": (
                                    (parsed is not None and oracle_answer is not None and parsed == oracle_answer)
                                )
                            }
                            if strong_enabled
                            else {}
                        ),
                    }
                    for s_idx, s_text in enumerate(solver_outputs)
                ],
                "pairwise_comparisons": pairwise_rows,
                "reliability": {
                    "preference_stability": (
                        sum(consistency_values) / len(consistency_values) if consistency_values else 0.0
                    ),
                    "num_pairs": len(pairwise_rows),
                    "repeats_per_pair": repeats,
                },
                "ranking": {
                    "method": "elo",
                    "k_factor": elo_k,
                    "initial_rating": elo_initial,
                },
                **(
                    {
                        "oracle": {
                            "enabled": True,
                            "provider": strong_provider,
                            "model": strong_model,
                            "answer": oracle_answer,
                            "error": oracle_error,
                            **({"raw_response": oracle_raw_response} if include_judge_traces else {}),
                        }
                    }
                    if strong_enabled
                    else {}
                ),
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
