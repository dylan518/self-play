from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from grpo_math.data.reward import extract_final_answer_int_strict


_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", flags=re.DOTALL)
_PREF_RE = re.compile(r"PREFERENCE:\s*(A|B)", flags=re.IGNORECASE)
_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", flags=re.IGNORECASE)
_BOOL_VERDICT_RE = re.compile(r"\b(TRUE|FALSE)\b", flags=re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(0(?:\.\d+)?|1(?:\.0+)?)", flags=re.IGNORECASE)
_BATCH_VERIFY_LINE_RE = re.compile(
    r"SOLUTION_(\d+)\s*:\s*(CORRECT|INCORRECT|TRUE|FALSE)(?:\s*\|\s*CONFIDENCE\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?))?",
    flags=re.IGNORECASE,
)
_BOXED_INT_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_FINAL_TEXT_INT_RE = re.compile(r"(?:Final Answer|Final answer)\s*[:\-]?\s*(-?\d+)\b")
_ANY_INT_RE = re.compile(r"-?\d+")


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


def _no_grad(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_last_jsonl_row(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _torch_dtype(name: str) -> torch.dtype:
    if torch is None:
        raise RuntimeError("PyTorch is required for local HF model loading.")
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
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required for local HF model loading.")
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
    if tokenizer is None:
        return prompt
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


@_no_grad
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


def _parse_verdict(text: str) -> str:
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper()
    m2 = _BOOL_VERDICT_RE.search(text)
    if m2:
        return "CORRECT" if m2.group(1).upper() == "TRUE" else "INCORRECT"
    return "INCORRECT"


def _parse_confidence(text: str) -> float:
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return 0.5
    try:
        val = float(m.group(1))
    except Exception:
        return 0.5
    return max(0.0, min(1.0, val))


def _parse_batch_verify_output(text: str, num_solutions: int) -> Dict[int, tuple[str, float]]:
    out: Dict[int, tuple[str, float]] = {}
    for m in _BATCH_VERIFY_LINE_RE.finditer(text):
        idx = int(m.group(1))
        if 0 <= idx < num_solutions:
            raw_verdict = m.group(2).upper()
            verdict = "CORRECT" if raw_verdict in {"CORRECT", "TRUE"} else "INCORRECT"
            conf = 0.5 if m.group(3) is None else max(0.0, min(1.0, float(m.group(3))))
            out[idx] = (verdict, conf)
    return out


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
    q = (parsed_question or "").strip()
    if not q:
        return False
    if q.upper() in {"QUESTION:", "QUESTION"}:
        return False
    # Prevent degenerate empty stubs that sometimes pass surface formatting.
    if len(q) < 16:
        return False
    if not re.search(r"[A-Za-z]", q):
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
    max_tokens_param: str = "max_completion_tokens",
) -> tuple[int | None, str | None, str | None]:
    prompt = (
        "Solve the following math question.\n"
        "Return only one line in this exact format:\n"
        "FINAL_ANSWER: <integer>\n\n"
        f"Question: {question}\n"
    )
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a precise math solver."},
            {"role": "user", "content": prompt},
        ],
    }
    payload[max_tokens_param] = 64
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
        content = _extract_chat_content(body)
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
    # Fallbacks for alternate OpenAI-compatible payloads.
    alt = msg.get("text")
    if isinstance(alt, str):
        return alt.strip()
    out_txt = body.get("output_text")
    if isinstance(out_txt, str):
        return out_txt.strip()
    raise KeyError("Could not parse textual content from chat completion response.")


def _openai_generate_texts(
    *,
    prompts: Sequence[str],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    timeout_s: float,
    max_tokens_param: str = "max_completion_tokens",
    reasoning_effort: str = "",
    min_interval_s: float = 0.0,
    max_retries: int = 6,
    initial_backoff_s: float = 1.0,
    max_parallel: int = 1,
) -> List[str]:
    def _request_once(prompt: str, backoff_s: float) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload[max_tokens_param] = max_completion_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
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
                return _extract_chat_content(body)
            except urllib.error.HTTPError as e:
                # Retry on server/transient throttling errors.
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

    # Parallel path for independent prompts (best for verifier mode).
    parallel = max(1, int(max_parallel))
    if parallel > 1 and len(prompts) > 1 and min_interval_s <= 0:
        ordered_outputs: List[str | None] = [None] * len(prompts)

        def _worker(idx_prompt: tuple[int, str]) -> tuple[int, str]:
            idx, prompt = idx_prompt
            text = _request_once(prompt, max(0.0, initial_backoff_s))
            return idx, text

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_worker, (idx, prompt)) for idx, prompt in enumerate(prompts)]
            for fut in concurrent.futures.as_completed(futures):
                idx, text = fut.result()
                ordered_outputs[idx] = text
        return [str(x) for x in ordered_outputs]

    # Sequential path with optional throttle.
    outputs: List[str] = []
    last_request_at = 0.0
    for prompt in prompts:
        if min_interval_s > 0:
            now = time.monotonic()
            wait_s = (last_request_at + min_interval_s) - now
            if wait_s > 0:
                time.sleep(wait_s)
            last_request_at = time.monotonic()
        outputs.append(_request_once(prompt, max(0.0, initial_backoff_s)))
    return outputs


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


def _cfg_uses_openai(role_cfg: Dict[str, Any]) -> bool:
    provider = str(role_cfg.get("api_provider", "")).lower()
    model_name = str(role_cfg.get("model_name_or_path", ""))
    return provider == "openai" or model_name.startswith("openai:")


def _cfg_openai_model(role_cfg: Dict[str, Any]) -> str:
    explicit = str(role_cfg.get("api_model", "")).strip()
    if explicit:
        return explicit
    model_name = str(role_cfg.get("model_name_or_path", ""))
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1].strip()
    return "gpt-4.1"


def _resolve_api_key(role_cfg: Dict[str, Any]) -> str | None:
    explicit = str(role_cfg.get("api_key", "")).strip()
    if explicit:
        return explicit
    key_env = str(role_cfg.get("api_key_env", "")).strip()
    if key_env:
        return (
            os.environ.get(key_env)
            or _read_env_var_from_dotenv(key_env)
        )
    for env_name in ("OPENAI_API_KEY", "OPENAI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_name) or _read_env_var_from_dotenv(env_name)
        if val:
            return val
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="grpo_math/configs/pairwise_rollouts_smoke.yaml")
    args = ap.parse_args()
    cfg = _load_yaml(args.config)
    debug_timing = bool(cfg.get("debug_timing", False))

    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    pair_cfg = cfg["pairwise"]
    gen_cfg = cfg["generator"]
    sol_cfg = cfg["solver"]
    judge_cfg = cfg["judge"]
    gen_cfg["_role_name"] = "generator"
    sol_cfg["_role_name"] = "solver"
    judge_cfg["_role_name"] = "judge"
    out_cfg = cfg["output"]
    strong_cfg = cfg.get("strong_verifier", {})

    gen_is_openai = _cfg_uses_openai(gen_cfg)
    sol_is_openai = _cfg_uses_openai(sol_cfg)
    judge_is_openai = _cfg_uses_openai(judge_cfg)

    gen_bundle = (
        ModelBundle(model=None, tokenizer=None)
        if gen_is_openai
        else _load_model_bundle(
            gen_cfg["model_name_or_path"],
            _torch_dtype(gen_cfg.get("torch_dtype", "bfloat16")),
            bool(gen_cfg.get("use_flash_attn", False)),
        )
    )
    sol_bundle = (
        ModelBundle(model=None, tokenizer=None)
        if sol_is_openai
        else _load_model_bundle(
            sol_cfg["model_name_or_path"],
            _torch_dtype(sol_cfg.get("torch_dtype", "bfloat16")),
            bool(sol_cfg.get("use_flash_attn", False)),
        )
    )
    judge_bundle = (
        ModelBundle(model=None, tokenizer=None)
        if judge_is_openai
        else _load_model_bundle(
            judge_cfg["model_name_or_path"],
            _torch_dtype(judge_cfg.get("torch_dtype", "bfloat16")),
            bool(judge_cfg.get("use_flash_attn", False)),
        )
    )

    question_template = _load_text(gen_cfg["prompt_template_path"])
    judge_template = _load_text(judge_cfg["prompt_template_path"])
    verify_template_default = (
        "You are a strict math verifier.\n"
        "Determine if the candidate solution's FINAL_ANSWER is mathematically correct for the question.\n"
        "Return exactly two lines:\n"
        "VERDICT: CORRECT or VERDICT: INCORRECT\n"
        "CONFIDENCE: <number between 0 and 1>\n\n"
        "Question:\n{question}\n\n"
        "Candidate solution:\n{solution}\n"
    )
    verify_template_path = str(judge_cfg.get("verify_prompt_template_path", "")).strip()
    if verify_template_path:
        verify_template = _load_text(verify_template_path)
    else:
        verify_template = str(judge_cfg.get("verify_prompt_template", verify_template_default))
    verify_batch_template = str(
        judge_cfg.get(
            "verify_batch_prompt_template",
            (
                "You are a strict math verifier.\n"
                "For each candidate solution, determine whether the FINAL_ANSWER is correct.\n"
                "Output exactly one line per solution in this format:\n"
                "SOLUTION_<index>: CORRECT|INCORRECT | CONFIDENCE: <0..1>\n\n"
                "Question:\n{question}\n\n"
                "Candidate solutions:\n{solutions_block}\n"
            ),
        )
    )
    solver_template = str(sol_cfg["prompt_template"])

    out_path = Path(out_cfg["jsonl_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = str(out_cfg.get("write_mode", "append")).lower()
    if write_mode not in {"append", "overwrite"}:
        raise ValueError(f"Unsupported output.write_mode: {write_mode}. Use 'append' or 'overwrite'.")
    file_open_mode = "a" if write_mode == "append" else "w"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    question_index_base = 0
    if file_open_mode == "a":
        last_row = _read_last_jsonl_row(out_path)
        if isinstance(last_row, dict):
            try:
                question_index_base = int(last_row.get("question_index", -1)) + 1
            except Exception:
                question_index_base = 0
    include_judge_traces = bool(out_cfg.get("include_judge_traces", False))
    strong_enabled = bool(strong_cfg.get("enabled", False))
    strong_provider = str(strong_cfg.get("provider", "openai")).lower()
    strong_model = str(strong_cfg.get("model", "gpt-4.1"))
    strong_base_url = str(strong_cfg.get("base_url", "https://api.openai.com/v1"))

    num_questions = int(pair_cfg.get("num_questions", 1))
    num_solutions = int(sol_cfg.get("num_solutions_per_question", 4))
    repeats = int(judge_cfg.get("repeats_per_pair", 3))
    verify_repeats = int(judge_cfg.get("repeats_per_solution", repeats))
    randomize_judge_order = bool(judge_cfg.get("randomize_a_b_order", True))
    balanced_judge_order = bool(judge_cfg.get("balanced_a_b_order", True))
    judge_mode = str(judge_cfg.get("mode", "pairwise")).strip().lower()
    if judge_mode not in {"pairwise", "single_verify"}:
        raise ValueError(f"Unsupported judge.mode: {judge_mode}. Use 'pairwise' or 'single_verify'.")
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
    openai_api_key = _resolve_api_key({})
    openai_base_url = "https://api.openai.com/v1"

    def _role_generate_texts(
        role_cfg: Dict[str, Any],
        role_bundle: ModelBundle,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> List[str]:
        role_name = str(role_cfg.get("_role_name", "role"))
        t0 = time.perf_counter()
        if debug_timing:
            print(
                f"[timing] start {role_name}: prompts={len(prompts)} max_new_tokens={max_new_tokens} "
                f"batch_size={batch_size}",
                flush=True,
            )
        if _cfg_uses_openai(role_cfg):
            role_api_key = _resolve_api_key(role_cfg) or openai_api_key
            if not role_api_key:
                raise RuntimeError(
                    "No API key found for API-backed role. Set api_key_env in config or export "
                    "OPENAI_API_KEY / GEMINI_API_KEY."
                )
            outputs = _openai_generate_texts(
                prompts=prompts,
                model=_cfg_openai_model(role_cfg),
                api_key=role_api_key,
                base_url=str(role_cfg.get("api_base_url", openai_base_url)),
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_new_tokens,
                timeout_s=float(role_cfg.get("api_timeout_s", 120.0)),
                max_tokens_param=str(role_cfg.get("api_max_tokens_param", "max_completion_tokens")),
                reasoning_effort=str(role_cfg.get("api_reasoning_effort", "")).strip(),
                min_interval_s=float(role_cfg.get("api_min_interval_s", 0.0)),
                max_retries=int(role_cfg.get("api_max_retries", 6)),
                initial_backoff_s=float(role_cfg.get("api_backoff_initial_s", 1.0)),
                max_parallel=int(role_cfg.get("api_max_parallel", 1)),
            )
        else:
            outputs = _generate_texts(
                role_bundle,
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
        if debug_timing:
            dt = time.perf_counter() - t0
            print(f"[timing] done  {role_name}: {dt:.2f}s", flush=True)
        return outputs

    raw_gen_prompt = question_template
    generator_prompt = _apply_chat_template(
        gen_bundle.tokenizer,
        raw_gen_prompt,
        bool(gen_cfg.get("use_chat_template", False)),
        gen_cfg.get("enable_thinking", None),
    )
    generator_prompts = [generator_prompt] * num_questions
    question_generations = _role_generate_texts(
        gen_cfg,
        gen_bundle,
        generator_prompts,
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
        batch_size=int(gen_cfg.get("batch_size", 8)),
    )
    # Retry malformed generations in parallel rounds instead of one-by-one.
    cleaned_question_generations = list(question_generations)
    questions = [_parse_question(raw) for raw in cleaned_question_generations]
    invalid_indices = [
        idx
        for idx, (raw, parsed) in enumerate(zip(cleaned_question_generations, questions))
        if not _looks_like_question_only(raw, parsed)
    ]
    retry_round = 0
    while invalid_indices and retry_round < max_retries:
        retry_outputs = _role_generate_texts(
            gen_cfg,
            gen_bundle,
            [generator_prompt] * len(invalid_indices),
            temperature=float(gen_cfg.get("temperature", 1.0)),
            top_p=float(gen_cfg.get("top_p", 0.95)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
            batch_size=max(1, int(gen_cfg.get("batch_size", 8))),
        )
        for idx, new_raw in zip(invalid_indices, retry_outputs):
            cleaned_question_generations[idx] = new_raw
            questions[idx] = _parse_question(new_raw)
        invalid_indices = [
            idx
            for idx in invalid_indices
            if not _looks_like_question_only(cleaned_question_generations[idx], questions[idx])
        ]
        retry_round += 1

    if invalid_indices:
        # Last-chance repair pass for truncated/malformed question drafts.
        repair_prompts = [
            _apply_chat_template(
                gen_bundle.tokenizer,
                (
                    "Rewrite the draft below into ONE complete, self-contained math question with a single "
                    "integer answer.\n"
                    "Do not provide a solution.\n"
                    "Output exactly one line in this format:\n"
                    "QUESTION: <complete question text ending with ?>\n\n"
                    f"DRAFT:\n{cleaned_question_generations[idx]}"
                ),
                bool(gen_cfg.get("use_chat_template", False)),
                gen_cfg.get("enable_thinking", None),
            )
            for idx in invalid_indices
        ]
        repaired = _role_generate_texts(
            gen_cfg,
            gen_bundle,
            repair_prompts,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=max(256, int(gen_cfg.get("max_new_tokens", 128))),
            batch_size=max(1, int(gen_cfg.get("batch_size", 8))),
        )
        for idx, new_raw in zip(invalid_indices, repaired):
            cleaned_question_generations[idx] = new_raw
            questions[idx] = _parse_question(new_raw)
        invalid_indices = [
            idx
            for idx in invalid_indices
            if not _looks_like_question_only(cleaned_question_generations[idx], questions[idx])
        ]

    if invalid_indices:
        examples = "\n".join(
            f"- idx={idx}: {cleaned_question_generations[idx][:200]!r}" for idx in invalid_indices[:3]
        )
        raise RuntimeError(
            "Question generation failed to produce complete questions after retries.\n"
            f"Invalid count: {len(invalid_indices)}\n"
            f"Examples:\n{examples}"
        )

    solver_batch_size = int(sol_cfg.get("batch_size", 16))
    solver_max_new_tokens = int(sol_cfg.get("max_new_tokens", 512))

    def _generate_solver_outputs_for_prompt(solver_prompt: str) -> List[str]:
        outputs: List[str] = []
        for grp in sampling_groups:
            group_count = int(grp["count"])
            group_outputs = _role_generate_texts(
                sol_cfg,
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
            group_outputs = _role_generate_texts(
                sol_cfg,
                sol_bundle,
                prompts_for_group,
                temperature=float(grp["temperature"]),
                top_p=float(grp["top_p"]),
                max_new_tokens=solver_max_new_tokens,
                batch_size=solver_batch_size,
            )
            for q_idx, output in zip(prompt_question_indices, group_outputs):
                solver_outputs_by_question[q_idx].append(output)

    # In single-verify mode, precompute verifier outputs for all questions in one call.
    precomputed_verify_raw_prompts_by_question: List[str] = []
    precomputed_verify_outputs_by_question: List[List[str]] = []
    if (
        judge_mode == "single_verify"
        and bool(judge_cfg.get("batch_verify_all_solutions", False))
        and batch_solver_across_questions
    ):
        verify_jobs: List[Dict[str, Any]] = []
        precomputed_verify_raw_prompts_by_question = ["" for _ in range(len(questions))]
        for q_idx, question in enumerate(questions):
            solver_outputs = solver_outputs_by_question[q_idx]
            solver_parsed_answers = [_parse_solver_final_answer(s_text) for s_text in solver_outputs]
            candidate_answers_block = "\n".join(
                f"SOLUTION_{s_idx}: "
                f"{solver_parsed_answers[s_idx] if solver_parsed_answers[s_idx] is not None else 'NONE'}"
                for s_idx in range(num_solutions)
            )
            solutions_block = "\n\n".join(
                f"SOLUTION_{s_idx}:\n{s_text}" for s_idx, s_text in enumerate(solver_outputs)
            )
            raw_prompt = verify_batch_template.format(
                question=question,
                solutions_block=solutions_block,
                candidate_answers_block=candidate_answers_block,
            )
            precomputed_verify_raw_prompts_by_question[q_idx] = raw_prompt
            formatted_prompt = _apply_chat_template(
                judge_bundle.tokenizer,
                raw_prompt,
                bool(judge_cfg.get("use_chat_template", False)),
                judge_cfg.get("enable_thinking", None),
            )
            for _ in range(verify_repeats):
                verify_jobs.append(
                    {
                        "question_index": q_idx,
                        "formatted_prompt": formatted_prompt,
                    }
                )

        verify_outputs_all = _role_generate_texts(
            judge_cfg,
            judge_bundle,
            [j["formatted_prompt"] for j in verify_jobs],
            temperature=float(judge_cfg.get("temperature", 0.0)),
            top_p=float(judge_cfg.get("top_p", 0.9)),
            max_new_tokens=int(judge_cfg.get("max_new_tokens", 64)),
            batch_size=int(judge_cfg.get("batch_size", 32)),
        )
        precomputed_verify_outputs_by_question = [[] for _ in range(len(questions))]
        for job, output in zip(verify_jobs, verify_outputs_all):
            q_idx = int(job["question_index"])
            precomputed_verify_outputs_by_question[q_idx].append(output)

    with out_path.open(file_open_mode, encoding="utf-8") as f:
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
            solver_parsed_answers = [_parse_solver_final_answer(s_text) for s_text in solver_outputs]

            pairwise_rows = []
            score_points = [0.0 for _ in range(num_solutions)]
            score_counts = [0 for _ in range(num_solutions)]
            consistency_values: List[float] = []

            verification_rows = []
            if judge_mode == "pairwise":
                pairs = _pair_indices(num_solutions, max_pairs_per_question)
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

                    judge_outputs = _role_generate_texts(
                        judge_cfg,
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
            else:
                verify_batch_all = bool(judge_cfg.get("batch_verify_all_solutions", False))
                per_solution: Dict[int, Dict[str, Any]] = {
                    s_idx: {"raw_prompt": None, "raw_outputs": [], "verdicts": [], "model_confidences": []}
                    for s_idx in range(num_solutions)
                }
                if verify_batch_all:
                    if precomputed_verify_outputs_by_question:
                        raw_prompt = precomputed_verify_raw_prompts_by_question[q_idx - 1]
                        verify_outputs = precomputed_verify_outputs_by_question[q_idx - 1]
                    else:
                        candidate_answers_block = "\n".join(
                            f"SOLUTION_{s_idx}: "
                            f"{solver_parsed_answers[s_idx] if solver_parsed_answers[s_idx] is not None else 'NONE'}"
                            for s_idx in range(num_solutions)
                        )
                        solutions_block = "\n\n".join(
                            f"SOLUTION_{s_idx}:\n{s_text}" for s_idx, s_text in enumerate(solver_outputs)
                        )
                        raw_prompt = verify_batch_template.format(
                            question=question,
                            solutions_block=solutions_block,
                            candidate_answers_block=candidate_answers_block,
                        )
                        formatted_prompt = _apply_chat_template(
                            judge_bundle.tokenizer,
                            raw_prompt,
                            bool(judge_cfg.get("use_chat_template", False)),
                            judge_cfg.get("enable_thinking", None),
                        )
                        verify_outputs = _role_generate_texts(
                            judge_cfg,
                            judge_bundle,
                            [formatted_prompt] * verify_repeats,
                            temperature=float(judge_cfg.get("temperature", 0.0)),
                            top_p=float(judge_cfg.get("top_p", 0.9)),
                            max_new_tokens=int(judge_cfg.get("max_new_tokens", 64)),
                            batch_size=int(judge_cfg.get("batch_size", 32)),
                        )
                    for output in verify_outputs:
                        parsed_rows = _parse_batch_verify_output(output, num_solutions)
                        for s_idx in range(num_solutions):
                            verdict, conf = parsed_rows.get(s_idx, ("INCORRECT", 0.5))
                            per_solution[s_idx]["raw_prompt"] = raw_prompt
                            per_solution[s_idx]["raw_outputs"].append(output)
                            per_solution[s_idx]["verdicts"].append(verdict)
                            per_solution[s_idx]["model_confidences"].append(conf)
                else:
                    verify_jobs: List[Dict[str, Any]] = []
                    for s_idx, s_text in enumerate(solver_outputs):
                        for _ in range(verify_repeats):
                            candidate_answer = (
                                str(solver_parsed_answers[s_idx])
                                if solver_parsed_answers[s_idx] is not None
                                else "NONE"
                            )
                            raw_prompt = verify_template.format(
                                question=question,
                                solution=s_text,
                                candidate_answer=candidate_answer,
                            )
                            formatted_prompt = _apply_chat_template(
                                judge_bundle.tokenizer,
                                raw_prompt,
                                bool(judge_cfg.get("use_chat_template", False)),
                                judge_cfg.get("enable_thinking", None),
                            )
                            verify_jobs.append(
                                {
                                    "solution_index": s_idx,
                                    "raw_prompt": raw_prompt,
                                    "formatted_prompt": formatted_prompt,
                                }
                            )

                    verify_outputs = _role_generate_texts(
                        judge_cfg,
                        judge_bundle,
                        [j["formatted_prompt"] for j in verify_jobs],
                        temperature=float(judge_cfg.get("temperature", 0.0)),
                        top_p=float(judge_cfg.get("top_p", 0.9)),
                        max_new_tokens=int(judge_cfg.get("max_new_tokens", 64)),
                        batch_size=int(judge_cfg.get("batch_size", 32)),
                    )
                    for job, output in zip(verify_jobs, verify_outputs):
                        s_idx = int(job["solution_index"])
                        row = per_solution[s_idx]
                        if row["raw_prompt"] is None:
                            row["raw_prompt"] = job["raw_prompt"]
                        row["raw_outputs"].append(output)
                        row["verdicts"].append(_parse_verdict(output))
                        row["model_confidences"].append(_parse_confidence(output))

                for s_idx in range(num_solutions):
                    verdicts = list(per_solution[s_idx]["verdicts"])
                    model_confidences = list(per_solution[s_idx]["model_confidences"])
                    n_correct = verdicts.count("CORRECT")
                    n_incorrect = verdicts.count("INCORRECT")
                    confidence = max(n_correct, n_incorrect) / max(1, verify_repeats)
                    consistency_values.append(confidence)
                    score_points[s_idx] += n_correct / max(1, verify_repeats)
                    score_counts[s_idx] += 1
                    verification_rows.append(
                        {
                            "solution_index": s_idx,
                            "verdicts": verdicts,
                            "model_confidences": model_confidences,
                            "counts": {"CORRECT": n_correct, "INCORRECT": n_incorrect},
                            "confidence": confidence,
                            "model_confidence_mean": (
                                sum(model_confidences) / len(model_confidences) if model_confidences else 0.5
                            ),
                            **(
                                {
                                    "judge_trace": {
                                        "raw_prompt": per_solution[s_idx]["raw_prompt"],
                                        "raw_outputs": per_solution[s_idx]["raw_outputs"],
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
            if judge_mode == "pairwise":
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
                        max_tokens_param=str(
                            strong_cfg.get("api_max_tokens_param", "max_completion_tokens")
                        ),
                    )

            row = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question_index": question_index_base + (q_idx - 1),
                "question": question,
                "generator_raw_output": cleaned_question_generations[q_idx - 1],
                "solutions": [
                    {
                        "solution_index": s_idx,
                        "text": s_text,
                        "parsed_final_answer": (parsed := solver_parsed_answers[s_idx]),
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
                **({"solution_verifications": verification_rows} if judge_mode == "single_verify" else {}),
                "reliability": {
                    "preference_stability": (
                        sum(consistency_values) / len(consistency_values) if consistency_values else 0.0
                    ),
                    "num_pairs": len(pairwise_rows),
                    "repeats_per_pair": repeats,
                    **(
                        {
                            "num_solutions_verified": len(verification_rows),
                            "repeats_per_solution": verify_repeats,
                        }
                        if judge_mode == "single_verify"
                        else {}
                    ),
                },
                "ranking": {
                    "method": "elo" if judge_mode == "pairwise" else "verify_score",
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
