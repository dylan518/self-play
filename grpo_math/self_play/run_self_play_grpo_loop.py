from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> None:
    print(f"[loop] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _start_async(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> tuple[subprocess.Popen[Any], float]:
    print(f"[loop] starting async: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=cwd, env=env), time.perf_counter()


def _finish_async(proc: subprocess.Popen[Any], cmd: list[str], t0: float) -> float:
    rc = proc.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return elapsed


def _cfg_uses_openai_api(role_cfg: Dict[str, Any]) -> bool:
    provider = str(role_cfg.get("api_provider", "")).strip().lower()
    model_name = str(role_cfg.get("model_name_or_path", "")).strip()
    return provider == "openai" or model_name.startswith("openai:")


def _safe_vllm_lora_name(role: str, cycle_tag: str, adapter_path: str) -> str:
    raw = f"{role}_{cycle_tag}_{Path(adapter_path).parent.name}_{Path(adapter_path).name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _vllm_load_lora_adapter(
    *,
    base_url: str,
    api_key: str,
    lora_name: str,
    lora_path: str,
    timeout_s: float = 120.0,
) -> None:
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    v1_url = re.sub(r"/?$", "", base_url.rstrip("/"))
    if not v1_url.endswith("/v1"):
        v1_url += "/v1"
    req = urllib.request.Request(
        url=v1_url + "/load_lora_adapter",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        print(f"[loop] loaded vLLM LoRA adapter {lora_name}: {body[:500]}", flush=True)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 400 and ("already" in body.lower() or "exist" in body.lower()):
            print(f"[loop] vLLM LoRA adapter already loaded: {lora_name}", flush=True)
            return
        raise RuntimeError(
            f"Failed to load vLLM LoRA adapter {lora_name} from {lora_path}: "
            f"HTTP {e.code}: {body[:1000]}"
        ) from e


def _run_timed(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> float:
    t0 = time.perf_counter()
    _run(cmd, cwd=cwd, env=env)
    return time.perf_counter() - t0


def _build_train_command(
    *,
    python: str,
    train_module_args: list[str],
    launcher: str,
    num_processes: int,
) -> list[str]:
    launcher = str(launcher).strip().lower()
    num_processes = max(1, int(num_processes))
    if launcher == "python" or num_processes == 1:
        return [python, "-m", "grpo_math.trl.train_grpo_trl", *train_module_args]
    if launcher == "accelerate":
        return [
            python,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "--mixed_precision",
            "bf16",
            "-m",
            "grpo_math.trl.train_grpo_trl",
            *train_module_args,
        ]
    raise ValueError(f"Unsupported train launcher: {launcher!r}")


def _query_gpu_snapshot(indices: list[int] | None = None) -> Dict[int, Dict[str, float]]:
    """
    Returns keyed stats by GPU index:
      util_gpu_pct, util_mem_pct, mem_used_gb, mem_total_gb
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    keep = set(indices) if indices else None
    parsed: Dict[int, Dict[str, float]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            idx = int(parts[0])
            if keep is not None and idx not in keep:
                continue
            util_gpu = float(parts[1])
            util_mem = float(parts[2])
            mem_used_gb = float(parts[3]) / 1024.0
            mem_total_gb = float(parts[4]) / 1024.0
        except Exception:
            continue
        parsed[idx] = {
            "util_gpu_pct": util_gpu,
            "util_mem_pct": util_mem,
            "mem_used_gb": mem_used_gb,
            "mem_total_gb": mem_total_gb,
        }
    return parsed


def _run_timed_with_gpu_sampling(
    cmd: list[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    gpu_indices: list[int],
    sample_every_s: float = 2.0,
    wandb_run: Any | None = None,
    heartbeat_prefix: str = "heartbeat",
    heartbeat_payload: Dict[str, Any] | None = None,
    heartbeat_every_s: float = 30.0,
    progress_jsonl: Path | None = None,
) -> tuple[float, Dict[str, float]]:
    """
    Runs command and returns:
      - elapsed seconds
      - aggregated GPU metrics with keys perf/gpu_<idx>/<metric>_{mean|max}
    """
    print(f"[loop] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    last_heartbeat_t = t0
    heartbeat_count = 0
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    series: Dict[int, Dict[str, list[float]]] = {
        i: {
            "util_gpu_pct": [],
            "util_mem_pct": [],
            "mem_used_gb": [],
            "mem_total_gb": [],
        }
        for i in gpu_indices
    }
    sample_every_s = max(0.5, float(sample_every_s))
    while True:
        rc = proc.poll()
        snap = _query_gpu_snapshot(gpu_indices)
        for i in gpu_indices:
            s = snap.get(i)
            if not s:
                continue
            for k in ("util_gpu_pct", "util_mem_pct", "mem_used_gb", "mem_total_gb"):
                series[i][k].append(float(s[k]))
        now = time.perf_counter()
        if wandb_run is not None and now - last_heartbeat_t >= max(5.0, float(heartbeat_every_s)):
            heartbeat_count += 1
            elapsed_s = now - t0
            payload: Dict[str, Any] = {
                "loop/heartbeat": heartbeat_count,
                f"{heartbeat_prefix}/heartbeat": heartbeat_count,
                f"{heartbeat_prefix}/elapsed_s": elapsed_s,
            }
            if heartbeat_payload:
                payload.update(heartbeat_payload)
            if progress_jsonl is not None:
                rows_so_far = float(_count_jsonl_rows(progress_jsonl))
                payload[f"{heartbeat_prefix}/jsonl_rows_so_far"] = rows_so_far
                payload["rollout/progress_rows_so_far"] = rows_so_far
            for i in gpu_indices:
                s = snap.get(i)
                if not s:
                    continue
                for k, v in s.items():
                    payload[f"{heartbeat_prefix}/gpu_{i}/{k}"] = float(v)
                    if heartbeat_prefix == "heartbeat/rollout_generate":
                        payload[f"rollout/progress_gpu_{i}_{k}"] = float(v)
            if heartbeat_prefix == "heartbeat/rollout_generate":
                payload["rollout/progress_heartbeat"] = heartbeat_count
                payload["rollout/progress_elapsed_s"] = elapsed_s
            elif heartbeat_prefix == "heartbeat/question_train":
                payload["question_grpo/progress_heartbeat"] = heartbeat_count
                payload["question_grpo/progress_elapsed_s"] = elapsed_s
            elif heartbeat_prefix == "heartbeat/solver_train":
                payload["grpo/progress_heartbeat"] = heartbeat_count
                payload["grpo/progress_elapsed_s"] = elapsed_s
            try:
                _wandb_log(wandb_run, payload)
            except Exception as e:
                print(f"[loop] failed to log heartbeat ({type(e).__name__}: {e})", flush=True)
            last_heartbeat_t = now
        if rc is not None:
            break
        time.sleep(sample_every_s)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    out: Dict[str, float] = {}
    for i in gpu_indices:
        for metric in ("util_gpu_pct", "util_mem_pct", "mem_used_gb", "mem_total_gb"):
            vals = series[i][metric]
            if not vals:
                continue
            out[f"perf/gpu_{i}/{metric}_mean"] = float(sum(vals) / len(vals))
            out[f"perf/gpu_{i}/{metric}_max"] = float(max(vals))
    return elapsed, out


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _infer_rollout_stats(path: Path) -> tuple[int, int]:
    """
    Returns:
      - number of questions (rows)
      - mean number of solutions per question (integer floor)
    """
    rows = 0
    total_solutions = 0
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows += 1
            try:
                obj = json.loads(raw)
                total_solutions += len(obj.get("solutions", []))
            except Exception:
                pass
    mean_solutions = (total_solutions // rows) if rows else 0
    return rows, mean_solutions


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _distinct_ngram_ratio(texts: list[str], n: int) -> float:
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        toks = _tokens(text)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return float(len(unique)) / float(total) if total else 0.0


def _unique_text_frac(texts: list[str]) -> float:
    nonempty = [t.strip() for t in texts if t.strip()]
    return float(len(set(nonempty))) / float(len(nonempty)) if nonempty else 0.0


def _prefix_unique_frac(texts: list[str], token_count: int) -> float:
    prefixes: list[str] = []
    for text in texts:
        toks = _tokens(text)
        if not toks:
            continue
        prefixes.append(" ".join(toks[:token_count]))
    return float(len(set(prefixes))) / float(len(prefixes)) if prefixes else 0.0


def _numeric_signature(text: str) -> str:
    # Keep the shape of numbers without making exact numeric values dominate.
    return re.sub(r"\d+", "<N>", text.lower())


def _pairwise_token_jaccard_distance_mean(texts: list[str]) -> float:
    token_sets = [set(_tokens(t)) for t in texts if t.strip()]
    vals: list[float] = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if not union:
                continue
            inter = token_sets[i] & token_sets[j]
            vals.append(1.0 - (float(len(inter)) / float(len(union))))
    return _mean(vals)


def _infer_rollout_diversity_stats(path: Path) -> Dict[str, float]:
    rows = 0
    questions: list[str] = []
    question_token_counts: list[float] = []
    question_char_counts: list[float] = []
    question_numeric_signatures: list[str] = []
    solution_texts: list[str] = []
    raw_solution_texts: list[str] = []
    per_question_solution_unique: list[float] = []
    per_question_answer_unique: list[float] = []
    per_question_solution_jaccard_distance: list[float] = []
    solution_count = 0
    parsed_count = 0
    truncated_count = 0
    tail_chars: list[float] = []
    solution_chars: list[float] = []
    raw_solution_chars: list[float] = []

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            rows += 1
            question = str(obj.get("question", "") or "")
            questions.append(question)
            q_toks = _tokens(question)
            question_token_counts.append(float(len(q_toks)))
            question_char_counts.append(float(len(question)))
            question_numeric_signatures.append(_numeric_signature(question))
            sols = obj.get("solutions", [])
            if not isinstance(sols, list):
                continue

            q_solution_texts: list[str] = []
            q_answers: list[str] = []
            for s in sols:
                if not isinstance(s, dict):
                    continue
                solution_count += 1
                text = str(s.get("text", "") or "")
                raw_text = str(s.get("raw_text", text) or "")
                solution_texts.append(text)
                raw_solution_texts.append(raw_text)
                q_solution_texts.append(text)
                solution_chars.append(float(len(text)))
                raw_solution_chars.append(float(len(raw_text)))

                parsed = s.get("parsed_final_answer")
                if parsed is not None:
                    parsed_count += 1
                    q_answers.append(str(parsed))
                if s.get("truncated_at_final_answer") is True:
                    truncated_count += 1
                tail = s.get("post_final_answer_tail_chars")
                if isinstance(tail, (int, float)):
                    tail_chars.append(float(tail))

            if q_solution_texts:
                per_question_solution_unique.append(_unique_text_frac(q_solution_texts))
                per_question_solution_jaccard_distance.append(
                    _pairwise_token_jaccard_distance_mean(q_solution_texts)
                )
            if q_answers:
                per_question_answer_unique.append(float(len(set(q_answers))) / float(len(q_answers)))

    out: Dict[str, float] = {}
    if rows > 0:
        out["rollout/diversity/questions"] = float(rows)
        out["rollout/diversity/question_unique_frac"] = _unique_text_frac(questions)
        out["rollout/diversity/question_prefix5_unique_frac"] = _prefix_unique_frac(questions, 5)
        out["rollout/diversity/question_prefix10_unique_frac"] = _prefix_unique_frac(questions, 10)
        out["rollout/diversity/question_numeric_signature_unique_frac"] = _unique_text_frac(
            question_numeric_signatures
        )
        out["rollout/diversity/question_distinct_1"] = _distinct_ngram_ratio(questions, 1)
        out["rollout/diversity/question_distinct_2"] = _distinct_ngram_ratio(questions, 2)
        out["rollout/diversity/question_pairwise_jaccard_distance_mean"] = (
            _pairwise_token_jaccard_distance_mean(questions)
        )
        out["rollout/diversity/question_tokens_mean"] = _mean(question_token_counts)
        out["rollout/diversity/question_chars_mean"] = _mean(question_char_counts)
        clock_like = sum(
            1
            for q in questions
            if re.search(r"\b(clock|time|hour|minute|second|utc|timestamp|station)\b", q, flags=re.IGNORECASE)
        )
        out["rollout/diversity/question_clock_time_frac"] = float(clock_like) / float(rows)
    if solution_count > 0:
        out["rollout/diversity/solutions"] = float(solution_count)
        out["rollout/diversity/solution_unique_frac"] = _unique_text_frac(solution_texts)
        out["rollout/diversity/solution_distinct_1"] = _distinct_ngram_ratio(solution_texts, 1)
        out["rollout/diversity/solution_distinct_2"] = _distinct_ngram_ratio(solution_texts, 2)
        out["rollout/diversity/raw_solution_distinct_1"] = _distinct_ngram_ratio(raw_solution_texts, 1)
        out["rollout/diversity/raw_solution_distinct_2"] = _distinct_ngram_ratio(raw_solution_texts, 2)
        out["rollout/diversity/solution_parse_rate"] = float(parsed_count) / float(solution_count)
        out["rollout/diversity/solution_truncated_frac"] = float(truncated_count) / float(solution_count)
        out["rollout/diversity/solution_chars_mean"] = _mean(solution_chars)
        out["rollout/diversity/raw_solution_chars_mean"] = _mean(raw_solution_chars)
    if per_question_solution_unique:
        out["rollout/diversity/solution_unique_frac_per_question_mean"] = _mean(
            per_question_solution_unique
        )
    if per_question_solution_jaccard_distance:
        out["rollout/diversity/solution_pairwise_jaccard_distance_mean"] = _mean(
            per_question_solution_jaccard_distance
        )
    if per_question_answer_unique:
        out["rollout/diversity/answer_unique_frac_per_question_mean"] = _mean(per_question_answer_unique)
    if tail_chars:
        out["rollout/diversity/post_final_answer_tail_chars_mean"] = _mean(tail_chars)
        out["rollout/diversity/post_final_answer_tail_chars_max"] = max(tail_chars)
    return out


def _infer_rollout_reliability_stats(path: Path) -> Dict[str, float]:
    """
    Aggregates oracle/R_sep style metrics from rollout JSONL rows.
    """
    rows = 0
    rsep_vals: list[float] = []
    pref_stab_vals: list[float] = []
    group_gap_present = 0
    question_reward_vals: list[float] = []
    question_reward_p_vals: list[float] = []
    question_reward_d_vals: list[float] = []
    question_reward_v_vals: list[float] = []
    question_reward_lambda_vals: list[float] = []
    confirm_total = 0
    confirm_passed = 0

    oracle_enabled_rows = 0
    oracle_answer_rows = 0
    oracle_error_rows = 0
    oracle_solution_total = 0
    oracle_solution_correct = 0
    oracle_question_check_enabled_rows = 0
    oracle_question_check_valid_rows = 0
    oracle_question_check_invalid_rows = 0
    oracle_question_check_unknown_rows = 0
    oracle_question_check_error_rows = 0

    verify_rows_total = 0
    verify_rows_unanimous = 0
    verify_rows_parsed = 0
    verify_expected_votes_total = 0
    # Verifier-vs-oracle solution-level diagnostics (single-verify mode).
    verify_oracle_compared = 0
    verify_oracle_majority_compared = 0
    verify_oracle_agree = 0
    verify_oracle_agree_correct = 0
    verify_oracle_agree_incorrect = 0
    verify_oracle_disagree = 0
    verify_oracle_disagree_marked_correct = 0
    verify_oracle_disagree_marked_incorrect = 0
    verify_oracle_pred_correct = 0
    verify_oracle_pred_correct_true = 0
    verify_oracle_pred_correct_false = 0
    verify_oracle_pred_incorrect = 0
    verify_oracle_pred_incorrect_true = 0
    verify_oracle_pred_incorrect_false = 0
    verify_oracle_unanimous_compared = 0
    verify_oracle_unanimous_agree = 0
    verify_oracle_non_unanimous_compared = 0
    verify_oracle_non_unanimous_majority_compared = 0
    verify_oracle_non_unanimous_agree = 0

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows += 1
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            reliability = obj.get("reliability", {}) if isinstance(obj.get("reliability"), dict) else {}
            pref = reliability.get("preference_stability")
            if isinstance(pref, (int, float)):
                pref_stab_vals.append(float(pref))
            rsep = reliability.get("r_sep")
            if isinstance(rsep, (int, float)):
                rsep_vals.append(float(rsep))
            if isinstance(reliability.get("group_verify_means"), list):
                group_gap_present += 1

            qr = obj.get("question_reward")
            if isinstance(qr, (int, float)):
                question_reward_vals.append(float(qr))
            qrs = obj.get("question_rewards", {})
            if isinstance(qrs, dict):
                p = qrs.get("solver_correctness_probability")
                d = qrs.get("difficulty_balance")
                v = qrs.get("verifiability")
                lam = qrs.get("lambda_verifiability")
                if isinstance(p, (int, float)):
                    question_reward_p_vals.append(float(p))
                if isinstance(d, (int, float)):
                    question_reward_d_vals.append(float(d))
                if isinstance(v, (int, float)):
                    question_reward_v_vals.append(float(v))
                if isinstance(lam, (int, float)):
                    question_reward_lambda_vals.append(float(lam))
            confirm = obj.get("generator_confirm")
            if isinstance(confirm, dict):
                confirm_total += 1
                if confirm.get("different_from_samples") is True:
                    confirm_passed += 1

            oracle = obj.get("oracle", {}) if isinstance(obj.get("oracle"), dict) else {}
            if oracle.get("enabled") is True:
                oracle_enabled_rows += 1
                if oracle.get("answer") is not None:
                    oracle_answer_rows += 1
                if oracle.get("error"):
                    oracle_error_rows += 1
                if oracle.get("question_check_enabled") is True:
                    oracle_question_check_enabled_rows += 1
                    qv = oracle.get("question_check_valid")
                    if qv is True:
                        oracle_question_check_valid_rows += 1
                    elif qv is False:
                        oracle_question_check_invalid_rows += 1
                    else:
                        oracle_question_check_unknown_rows += 1
                    if oracle.get("question_check_error"):
                        oracle_question_check_error_rows += 1

            sols = obj.get("solutions", [])
            if isinstance(sols, list):
                for s in sols:
                    if not isinstance(s, dict):
                        continue
                    if "oracle_correct" in s:
                        oracle_solution_total += 1
                        if s.get("oracle_correct") is True:
                            oracle_solution_correct += 1

            verifs = obj.get("solution_verifications", [])
            if isinstance(verifs, list):
                ver_by_idx: Dict[int, Dict[str, Any]] = {}
                repeats_per_solution = None
                reliability = obj.get("reliability", {})
                if isinstance(reliability, dict):
                    rps = reliability.get("repeats_per_solution")
                    if isinstance(rps, (int, float)) and int(rps) > 0:
                        repeats_per_solution = int(rps)
                for v in verifs:
                    if not isinstance(v, dict):
                        continue
                    sidx_raw = v.get("solution_index")
                    if isinstance(sidx_raw, (int, float)):
                        ver_by_idx[int(sidx_raw)] = v
                    counts = v.get("counts", {})
                    if not isinstance(counts, dict):
                        continue
                    n_c = int(counts.get("CORRECT", 0) or 0)
                    n_i = int(counts.get("INCORRECT", 0) or 0)
                    n = n_c + n_i
                    if n <= 0:
                        continue
                    verify_rows_total += 1
                    verify_rows_parsed += n
                    if repeats_per_solution is not None:
                        verify_expected_votes_total += repeats_per_solution
                    if n_c == n or n_i == n:
                        verify_rows_unanimous += 1

                # Oracle/verifier agreement metrics at solution level.
                sols = obj.get("solutions", [])
                oracle_answer = obj.get("oracle", {}).get("answer") if isinstance(obj.get("oracle"), dict) else None
                if isinstance(sols, list):
                    for s in sols:
                        if not isinstance(s, dict):
                            continue
                        sidx_raw = s.get("solution_index")
                        if not isinstance(sidx_raw, (int, float)):
                            continue
                        v = ver_by_idx.get(int(sidx_raw))
                        if not isinstance(v, dict):
                            continue
                        counts = v.get("counts", {})
                        if not isinstance(counts, dict):
                            continue
                        n_c = int(counts.get("CORRECT", 0) or 0)
                        n_i = int(counts.get("INCORRECT", 0) or 0)
                        is_unanimous = (n_c > 0 and n_i == 0) or (n_i > 0 and n_c == 0)
                        if (n_c + n_i) <= 0:
                            continue

                        oracle_correct_label = s.get("oracle_correct")
                        parsed_ans = s.get("parsed_final_answer")
                        oracle_is_correct: bool | None = None
                        if isinstance(oracle_correct_label, bool):
                            oracle_is_correct = oracle_correct_label
                        elif isinstance(oracle_answer, int) and parsed_ans is not None:
                            oracle_is_correct = int(parsed_ans) == int(oracle_answer)
                        if oracle_is_correct is None:
                            continue

                        verify_oracle_compared += 1
                        if is_unanimous:
                            verify_oracle_unanimous_compared += 1
                        else:
                            verify_oracle_non_unanimous_compared += 1
                        if n_c == n_i:
                            continue
                        verify_oracle_majority_compared += 1
                        if not is_unanimous:
                            verify_oracle_non_unanimous_majority_compared += 1
                        verifier_pred_correct = n_c > n_i

                        if verifier_pred_correct:
                            verify_oracle_pred_correct += 1
                            if oracle_is_correct:
                                verify_oracle_pred_correct_true += 1
                            else:
                                verify_oracle_pred_correct_false += 1
                        else:
                            verify_oracle_pred_incorrect += 1
                            if oracle_is_correct:
                                verify_oracle_pred_incorrect_false += 1
                            else:
                                verify_oracle_pred_incorrect_true += 1

                        if verifier_pred_correct == oracle_is_correct:
                            verify_oracle_agree += 1
                            if is_unanimous:
                                verify_oracle_unanimous_agree += 1
                            else:
                                verify_oracle_non_unanimous_agree += 1
                            if oracle_is_correct:
                                verify_oracle_agree_correct += 1
                            else:
                                verify_oracle_agree_incorrect += 1
                        else:
                            verify_oracle_disagree += 1
                            if verifier_pred_correct:
                                verify_oracle_disagree_marked_correct += 1
                            else:
                                verify_oracle_disagree_marked_incorrect += 1

    out: Dict[str, float] = {}
    if rows > 0:
        out["rollout/rows"] = float(rows)
    if pref_stab_vals:
        out["rollout/reliability/preference_stability_mean"] = sum(pref_stab_vals) / len(pref_stab_vals)
    if rsep_vals:
        out["rollout/reliability/r_sep_mean"] = sum(rsep_vals) / len(rsep_vals)
        out["rollout/reliability/r_sep_positive_frac"] = (
            sum(1 for x in rsep_vals if x > 0.0) / len(rsep_vals)
        )
        out["rollout/reliability/r_sep_ge_0_15_frac"] = (
            sum(1 for x in rsep_vals if x >= 0.15) / len(rsep_vals)
        )
        out["rollout/reliability/r_sep_count"] = float(len(rsep_vals))
    out["rollout/reliability/group_verify_means_rows"] = float(group_gap_present)

    if question_reward_vals:
        out["rollout/question_reward/mean"] = _mean(question_reward_vals)
        out["rollout/question_reward/min"] = min(question_reward_vals)
        out["rollout/question_reward/max"] = max(question_reward_vals)
        out["rollout/question_reward/nonzero_frac"] = (
            sum(1 for x in question_reward_vals if x > 0.0) / len(question_reward_vals)
        )
    if question_reward_p_vals:
        out["rollout/question_reward/solver_correctness_probability_mean"] = _mean(question_reward_p_vals)
    if question_reward_d_vals:
        out["rollout/question_reward/difficulty_balance_mean"] = _mean(question_reward_d_vals)
    if question_reward_v_vals:
        out["rollout/question_reward/verifiability_mean"] = _mean(question_reward_v_vals)
    if question_reward_lambda_vals:
        out["rollout/question_reward/lambda_verifiability_mean"] = _mean(question_reward_lambda_vals)
    out["rollout/question_confirm/rows"] = float(confirm_total)
    if confirm_total > 0:
        out["rollout/question_confirm/different_from_samples_rate"] = float(confirm_passed) / float(confirm_total)

    out["rollout/oracle/enabled_rows"] = float(oracle_enabled_rows)
    out["rollout/oracle/answer_rows"] = float(oracle_answer_rows)
    out["rollout/oracle/error_rows"] = float(oracle_error_rows)
    if oracle_enabled_rows > 0:
        out["rollout/oracle/answer_rate"] = float(oracle_answer_rows) / float(oracle_enabled_rows)
        out["rollout/oracle/error_rate"] = float(oracle_error_rows) / float(oracle_enabled_rows)
    if oracle_solution_total > 0:
        out["rollout/oracle/solution_correct_rate"] = float(oracle_solution_correct) / float(oracle_solution_total)
        out["rollout/oracle/solution_scored"] = float(oracle_solution_total)
    out["rollout/oracle/question_check_enabled_rows"] = float(oracle_question_check_enabled_rows)
    out["rollout/oracle/question_check_valid_rows"] = float(oracle_question_check_valid_rows)
    out["rollout/oracle/question_check_invalid_rows"] = float(oracle_question_check_invalid_rows)
    out["rollout/oracle/question_check_unknown_rows"] = float(oracle_question_check_unknown_rows)
    out["rollout/oracle/question_check_error_rows"] = float(oracle_question_check_error_rows)
    if oracle_question_check_enabled_rows > 0:
        out["rollout/oracle/question_check_valid_rate"] = (
            float(oracle_question_check_valid_rows) / float(oracle_question_check_enabled_rows)
        )
        out["rollout/oracle/question_check_invalid_rate"] = (
            float(oracle_question_check_invalid_rows) / float(oracle_question_check_enabled_rows)
        )
        out["rollout/oracle/question_check_unknown_rate"] = (
            float(oracle_question_check_unknown_rows) / float(oracle_question_check_enabled_rows)
        )

    out["rollout/verify/rows"] = float(verify_rows_total)
    if verify_rows_total > 0:
        out["rollout/verify/unanimous_frac"] = float(verify_rows_unanimous) / float(verify_rows_total)
    if verify_rows_parsed > 0 and verify_rows_total > 0:
        out["rollout/verify/parsed_votes_per_solution_mean"] = float(verify_rows_parsed) / float(verify_rows_total)
    if verify_expected_votes_total > 0:
        out["rollout/verify/parse_rate"] = float(verify_rows_parsed) / float(verify_expected_votes_total)

    out["rollout/verify_oracle/compared"] = float(verify_oracle_compared)
    out["rollout/verify_oracle/majority_compared"] = float(verify_oracle_majority_compared)
    out["rollout/verify_oracle/unanimous_compared"] = float(verify_oracle_unanimous_compared)
    out["rollout/verify_oracle/non_unanimous_compared"] = float(verify_oracle_non_unanimous_compared)
    out["rollout/verify_oracle/non_unanimous_majority_compared"] = float(
        verify_oracle_non_unanimous_majority_compared
    )
    if verify_oracle_majority_compared > 0:
        out["rollout/verify_oracle/accuracy"] = float(verify_oracle_agree) / float(verify_oracle_majority_compared)
        out["rollout/verify_oracle/agreement_correct_frac"] = (
            float(verify_oracle_agree_correct) / float(verify_oracle_majority_compared)
        )
        out["rollout/verify_oracle/agreement_incorrect_frac"] = (
            float(verify_oracle_agree_incorrect) / float(verify_oracle_majority_compared)
        )
        out["rollout/verify_oracle/disagreement_frac"] = (
            float(verify_oracle_disagree) / float(verify_oracle_majority_compared)
        )
        out["rollout/verify_oracle/disagree_marked_correct_frac"] = (
            float(verify_oracle_disagree_marked_correct) / float(verify_oracle_majority_compared)
        )
        out["rollout/verify_oracle/disagree_marked_incorrect_frac"] = (
            float(verify_oracle_disagree_marked_incorrect) / float(verify_oracle_majority_compared)
        )
    if verify_oracle_unanimous_compared > 0:
        out["rollout/verify_oracle/unanimous_accuracy"] = (
            float(verify_oracle_unanimous_agree) / float(verify_oracle_unanimous_compared)
        )
    if verify_oracle_non_unanimous_majority_compared > 0:
        out["rollout/verify_oracle/non_unanimous_accuracy"] = (
            float(verify_oracle_non_unanimous_agree) / float(verify_oracle_non_unanimous_majority_compared)
        )
    if verify_oracle_pred_correct > 0:
        out["rollout/verify_oracle/precision_when_marked_correct"] = (
            float(verify_oracle_pred_correct_true) / float(verify_oracle_pred_correct)
        )
        out["rollout/verify_oracle/marked_correct_but_oracle_diff_frac"] = (
            float(verify_oracle_pred_correct_false) / float(verify_oracle_pred_correct)
        )
    if verify_oracle_pred_incorrect > 0:
        out["rollout/verify_oracle/precision_when_marked_incorrect"] = (
            float(verify_oracle_pred_incorrect_true) / float(verify_oracle_pred_incorrect)
        )
        out["rollout/verify_oracle/marked_incorrect_but_oracle_correct_frac"] = (
            float(verify_oracle_pred_incorrect_false) / float(verify_oracle_pred_incorrect)
        )
    return out


def _latest_checkpoint(train_output_dir: Path) -> Path | None:
    checkpoints = [p for p in train_output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _key(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except Exception:
            return -1

    return sorted(checkpoints, key=_key)[-1]


def _extract_train_perf_metrics(train_output_dir: Path) -> Dict[str, float]:
    """
    Extracts speed-oriented training metrics from TRL trainer_state.json if present.
    Returns keys suitable for direct W&B logging.
    """
    out: Dict[str, float] = {}
    ckpt = _latest_checkpoint(train_output_dir)
    if ckpt is None:
        return out
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return out
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    log_history = payload.get("log_history", [])
    if not isinstance(log_history, list):
        return out
    for row in log_history:
        if not isinstance(row, dict):
            continue
        step_time = row.get("step_time")
        num_tokens = row.get("num_tokens")
        if isinstance(step_time, (int, float)):
            out["train/step_time_s"] = float(step_time)
        if isinstance(num_tokens, (int, float)):
            out["train/num_tokens"] = float(num_tokens)
        if (
            isinstance(step_time, (int, float))
            and isinstance(num_tokens, (int, float))
            and float(step_time) > 0
        ):
            out["train/tokens_per_s"] = float(num_tokens) / float(step_time)
        eval_runtime = row.get("eval_runtime")
        if isinstance(eval_runtime, (int, float)):
            out["train/eval_runtime_s"] = float(eval_runtime)
        eval_num_tokens = row.get("eval_num_tokens")
        if isinstance(eval_num_tokens, (int, float)):
            out["train/eval_num_tokens"] = float(eval_num_tokens)
        if (
            isinstance(eval_runtime, (int, float))
            and isinstance(eval_num_tokens, (int, float))
            and float(eval_runtime) > 0
        ):
            out["train/eval_tokens_per_s"] = float(eval_num_tokens) / float(eval_runtime)
    return out


def _extract_train_scalar_metrics(train_output_dir: Path) -> Dict[str, float]:
    """
    Extract key scalar metrics from trainer_state.json and prefix for unified loop logging.
    """
    out: Dict[str, float] = {}
    ckpt = _latest_checkpoint(train_output_dir)
    if ckpt is None:
        return out
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return out
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    log_history = payload.get("log_history", [])
    if not isinstance(log_history, list):
        return out

    train_row: Dict[str, Any] | None = None
    eval_row: Dict[str, Any] | None = None
    for row in log_history:
        if not isinstance(row, dict):
            continue
        if "loss" in row and "eval_loss" not in row:
            train_row = row
        if any(k.startswith("eval_") for k in row.keys()):
            eval_row = row

    train_keys = [
        "loss",
        "grad_norm",
        "learning_rate",
        "num_tokens",
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "entropy",
        "step_time",
        "train_runtime",
        "train_steps_per_second",
        "train_samples_per_second",
    ]
    eval_keys = [
        "eval_loss",
        "eval_reward",
        "eval_reward_std",
        "eval_frac_reward_zero_std",
        "eval_kl",
        "eval_entropy",
        "eval_runtime",
        "eval_steps_per_second",
        "eval_samples_per_second",
        "eval_num_tokens",
    ]

    if train_row is not None:
        for k in train_keys:
            v = train_row.get(k)
            if isinstance(v, (int, float)):
                out[f"grpo/train/{k}"] = float(v)
        # Keep the most useful per-reward means if present.
        for k, v in train_row.items():
            if k.startswith("rewards/") and k.endswith("/mean") and isinstance(v, (int, float)):
                out[f"grpo/train/{k}"] = float(v)

    if eval_row is not None:
        for k in eval_keys:
            v = eval_row.get(k)
            if isinstance(v, (int, float)):
                out[f"grpo/eval/{k}"] = float(v)
        for k, v in eval_row.items():
            if k.startswith("eval_rewards/") and k.endswith("/mean") and isinstance(v, (int, float)):
                out[f"grpo/eval/{k}"] = float(v)
    return out


def _wandb_log(wandb_run: Any, payload: Dict[str, Any]) -> None:
    wandb_run.log(payload)
    summary_payload = {
        k: v
        for k, v in payload.items()
        if isinstance(v, (int, float, str, bool)) and not str(k).startswith("_")
    }
    if summary_payload:
        wandb_run.summary.update(summary_payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rollout_config",
        type=str,
        default="grpo_math/configs/pairwise_rollouts_llama31_8b_vllm_single_verify_fast_20q.yaml",
        help="Self-play rollout config template.",
    )
    ap.add_argument(
        "--train_config",
        type=str,
        default="grpo_math/configs/train_pairwise_verdict_llama31_8b_trl.yaml",
        help="GRPO train config template.",
    )
    ap.add_argument(
        "--question_train_config",
        type=str,
        default="",
        help="Optional GRPO train config for the question generator.",
    )
    ap.add_argument("--cycles", type=int, default=1, help="Number of rollout->train iterations.")
    ap.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional run tag. Defaults to UTC timestamp.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional override passed to train_grpo_trl.",
    )
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional override passed to train_grpo_trl.",
    )
    ap.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional override passed to train_grpo_trl.",
    )
    ap.add_argument(
        "--log_rollouts_to_wandb",
        action="store_true",
        help="If set, uploads per-cycle rollout JSONL and markdown exports as W&B artifacts.",
    )
    ap.add_argument(
        "--use_rollout_strong_verifier_for_train",
        action="store_true",
        help=(
            "If set and rollout config has `strong_verifier.enabled: true`, copy that verifier "
            "into train.reward.teacher so GRPO reward uses an external teacher endpoint."
        ),
    )
    ap.add_argument(
        "--separate_train_wandb",
        action="store_true",
        help=(
            "If set, keep separate W&B runs for each train cycle. "
            "Default behavior logs train metrics into the single loop run."
        ),
    )
    ap.add_argument(
        "--gpu_sample_indices",
        type=str,
        default="0,1",
        help="Comma-separated GPU indices to sample for utilization/memory metrics.",
    )
    ap.add_argument(
        "--train_launcher",
        type=str,
        choices=["python", "accelerate"],
        default="python",
        help="Launcher for question/solver GRPO training subprocesses.",
    )
    ap.add_argument(
        "--train_num_processes",
        type=int,
        default=1,
        help="Number of processes to use when --train_launcher=accelerate.",
    )
    ap.add_argument(
        "--stream_question_train",
        action="store_true",
        help=(
            "Start question-generator GRPO for the current generator checkpoint while rollout runs. "
            "This overlaps the independent question-update phase with rollout; solver GRPO still waits "
            "for the rollout JSONL."
        ),
    )
    ap.add_argument(
        "--initial_solver_model",
        type=str,
        default="",
        help="Optional solver model/checkpoint to use for the first cycle instead of the train config model.",
    )
    ap.add_argument(
        "--initial_generator_model",
        type=str,
        default="",
        help="Optional question-generator model/checkpoint to use for the first cycle instead of the rollout config model.",
    )
    ap.add_argument(
        "--cycle_offset",
        type=int,
        default=0,
        help="Add this offset to cycle numbering for output files and W&B cycle metrics.",
    )
    args = ap.parse_args()

    root = _repo_root()
    rollout_template_path = (root / args.rollout_config).resolve()
    train_template_path = (root / args.train_config).resolve()
    question_train_template_path = (
        (root / args.question_train_config).resolve() if str(args.question_train_config).strip() else None
    )
    if not rollout_template_path.exists():
        raise FileNotFoundError(f"Rollout config not found: {rollout_template_path}")
    if not train_template_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_template_path}")
    if question_train_template_path is not None and not question_train_template_path.exists():
        raise FileNotFoundError(f"Question train config not found: {question_train_template_path}")

    tag = args.run_tag.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wandb_group = f"self-play-grpo-{tag}"
    run_dir = root / "outputs" / "self_play_grpo_loop" / tag
    cfg_dir = run_dir / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python = sys.executable
    current_solver_model: str | None = args.initial_solver_model.strip() or None
    current_generator_model: str | None = args.initial_generator_model.strip() or None
    wandb_run = None
    gpu_indices: list[int] = []
    for tok in str(args.gpu_sample_indices).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            gpu_indices.append(int(tok))
        except Exception:
            pass

    print(f"[loop] run_dir={run_dir}", flush=True)
    if args.log_rollouts_to_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "grpo-math"),
                name=f"self-play-rollouts-{tag}",
                group=wandb_group,
                job_type="self-play-rollout-loop",
                tags=["self-play", "rollout-loop", tag],
                config={
                    "cycles": args.cycles,
                    "cycle_offset": args.cycle_offset,
                    "initial_solver_model": current_solver_model or "",
                    "initial_generator_model": current_generator_model or "",
                    "rollout_config": str(rollout_template_path),
                    "train_config": str(train_template_path),
                    "question_train_config": str(question_train_template_path) if question_train_template_path else "",
                    "run_dir": str(run_dir),
                },
                reinit=True,
            )
            print("[loop] W&B rollout artifact logging enabled", flush=True)
            _wandb_log(
                wandb_run,
                {
                    "loop/status_code": 0,
                    "loop/status": "initialized",
                    "loop/cycles_total": args.cycles,
                    "rollout/cycle": 0,
                    "perf/elapsed_s": 0.0,
                },
            )
        except Exception as e:
            print(f"[loop] W&B artifact logging disabled ({type(e).__name__}: {e})", flush=True)
            wandb_run = None

    exit_code = 0
    try:
        for local_cycle in range(1, args.cycles + 1):
            cycle = int(args.cycle_offset) + local_cycle
            cycle_tag = f"cycle_{cycle:03d}"
            print(f"[loop] ===== {cycle_tag} =====", flush=True)
            cycle_t0 = time.perf_counter()

            if wandb_run is not None:
                _wandb_log(wandb_run, {"loop/status_code": 1, "loop/status": "cycle_start", "rollout/cycle": cycle})

            rollout_cfg = _load_yaml(rollout_template_path)
            cycle_jsonl = run_dir / f"{cycle_tag}_samples.jsonl"
            rel_cycle_jsonl = cycle_jsonl.relative_to(root)
            rollout_cfg.setdefault("output", {})
            rollout_cfg["output"]["jsonl_path"] = str(rel_cycle_jsonl)
            rollout_cfg["output"]["write_mode"] = "overwrite"
            if current_generator_model:
                rollout_cfg.setdefault("generator", {})
                if _cfg_uses_openai_api(rollout_cfg["generator"]):
                    generator_lora_name = _safe_vllm_lora_name("generator", cycle_tag, current_generator_model)
                    _vllm_load_lora_adapter(
                        base_url=str(rollout_cfg["generator"].get("api_base_url", "http://127.0.0.1:8001/v1")),
                        api_key=str(rollout_cfg["generator"].get("api_key", "EMPTY")),
                        lora_name=generator_lora_name,
                        lora_path=current_generator_model,
                        timeout_s=float(rollout_cfg["generator"].get("api_timeout_s", 120.0)),
                    )
                    rollout_cfg["generator"]["api_model"] = generator_lora_name
                    print(
                        f"[loop] rollout generator vLLM LoRA: {generator_lora_name} -> {current_generator_model}",
                        flush=True,
                    )
                else:
                    rollout_cfg["generator"]["model_name_or_path"] = current_generator_model
                    print(f"[loop] rollout generator model: {current_generator_model}", flush=True)
            if current_solver_model:
                rollout_cfg.setdefault("solver", {})
                if _cfg_uses_openai_api(rollout_cfg["solver"]):
                    solver_lora_name = _safe_vllm_lora_name("solver", cycle_tag, current_solver_model)
                    _vllm_load_lora_adapter(
                        base_url=str(rollout_cfg["solver"].get("api_base_url", "http://127.0.0.1:8001/v1")),
                        api_key=str(rollout_cfg["solver"].get("api_key", "EMPTY")),
                        lora_name=solver_lora_name,
                        lora_path=current_solver_model,
                        timeout_s=float(rollout_cfg["solver"].get("api_timeout_s", 120.0)),
                    )
                    rollout_cfg["solver"]["api_model"] = solver_lora_name
                    for grp in rollout_cfg["solver"].get("sampling_groups", []) or []:
                        if isinstance(grp, dict) and any(
                            key in grp for key in ("api_model", "api_base_url", "api_key_env")
                        ):
                            grp["api_model"] = solver_lora_name
                    print(
                        f"[loop] rollout solver vLLM LoRA: {solver_lora_name} -> {current_solver_model}",
                        flush=True,
                    )
                else:
                    rollout_cfg["solver"]["model_name_or_path"] = current_solver_model
                    print(f"[loop] rollout solver model: {current_solver_model}", flush=True)
            rollout_cfg_path = cfg_dir / f"{cycle_tag}_rollout.yaml"
            _dump_yaml(rollout_cfg_path, rollout_cfg)

            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    {
                        "loop/status_code": 2,
                        "loop/status": "rollout_generate_start",
                        "rollout/cycle": cycle,
                        "rollout/config_num_questions": int(rollout_cfg.get("pairwise", {}).get("num_questions", 0) or 0),
                        "rollout/config_num_solutions_per_question": int(rollout_cfg.get("solver", {}).get("num_solutions_per_question", 0) or 0),
                        "rollout/config_judge_verify_sample_size": int(rollout_cfg.get("judge", {}).get("verify_sample_size", 0) or 0),
                    },
                )

            question_train_s = 0.0
            question_train_gpu: Dict[str, float] = {}
            question_train_out: Path | None = None
            question_train_cmd: list[str] | None = None
            question_train_proc: subprocess.Popen[Any] | None = None
            question_train_t0 = 0.0
            if question_train_template_path is not None:
                question_train_cfg = _load_yaml(question_train_template_path)
                if current_generator_model:
                    question_train_cfg.setdefault("model", {})
                    question_train_cfg["model"]["name_or_path"] = current_generator_model

                question_train_out = run_dir / f"{cycle_tag}_question_grpo"
                rel_question_train_out = question_train_out.relative_to(root)
                question_train_cfg.setdefault("train", {})
                question_train_cfg["train"]["output_dir"] = str(rel_question_train_out)
                question_wandb_cfg = question_train_cfg["train"].setdefault("wandb", {})
                base_question_run_name = str(question_wandb_cfg.get("run_name", "self-play-question-grpo"))
                question_wandb_cfg["run_name"] = f"{base_question_run_name}-{tag}-{cycle_tag}"
                if args.separate_train_wandb:
                    question_wandb_cfg["enabled"] = True
                    question_wandb_cfg["group"] = wandb_group
                    base_tags = question_wandb_cfg.get("tags", [])
                    if not isinstance(base_tags, list):
                        base_tags = [str(base_tags)] if base_tags else []
                    question_wandb_cfg["tags"] = list(
                        dict.fromkeys([*base_tags, "self-play", "question-grpo", tag, cycle_tag])
                    )
                else:
                    question_wandb_cfg["enabled"] = False

                question_train_cfg_path = cfg_dir / f"{cycle_tag}_question_train.yaml"
                _dump_yaml(question_train_cfg_path, question_train_cfg)
                question_train_cmd = _build_train_command(
                    python=python,
                    launcher=args.train_launcher,
                    num_processes=args.train_num_processes,
                    train_module_args=["--config", str(question_train_cfg_path)],
                )
                if wandb_run is not None:
                    _wandb_log(
                        wandb_run,
                        {
                            "loop/status_code": 4.5,
                            "loop/status": "question_train_start",
                            "rollout/cycle": cycle,
                            "question_train/streamed": bool(args.stream_question_train),
                        },
                    )
                if args.stream_question_train:
                    question_train_proc, question_train_t0 = _start_async(question_train_cmd, cwd=root, env=env)

            rollout_generate_s, rollout_generate_gpu = _run_timed_with_gpu_sampling(
                [
                    python,
                    "-m",
                    "grpo_math.self_play.generate_pairwise_data",
                    "--config",
                    str(rollout_cfg_path),
                ],
                cwd=root,
                env=env,
                gpu_indices=gpu_indices,
                **(
                    {
                        "wandb_run": wandb_run,
                        "heartbeat_prefix": "heartbeat/rollout_generate",
                        "heartbeat_payload": {
                            "loop/status": "rollout_generate_running",
                            "rollout/cycle": cycle,
                            "loop/cycle_index": cycle,
                        },
                        "heartbeat_every_s": 30.0,
                        "progress_jsonl": cycle_jsonl,
                    }
                    if wandb_run is not None
                    else {}
                ),
            )

            if wandb_run is not None:
                _wandb_log(wandb_run, {"loop/status_code": 3, "loop/status": "rollout_generate_done", "rollout/cycle": cycle})

            rollout_export_s = _run_timed(
                [
                    python,
                    "-m",
                    "grpo_math.self_play.export_rollout_readmes",
                    "--jsonl",
                    str(rel_cycle_jsonl),
                    "--out_dir",
                    "outputs/readme_exports",
                    "--config",
                    str(rollout_cfg_path),
                ],
                cwd=root,
                env=env,
            )
            md_dir = root / "outputs" / "readme_exports" / rel_cycle_jsonl.stem
            if wandb_run is not None:
                try:
                    import wandb

                    rows, mean_solutions = _infer_rollout_stats(cycle_jsonl)
                    artifact = wandb.Artifact(
                        name=f"self-play-rollout-{tag}-{cycle_tag}",
                        type="rollout",
                        description="Self-play rollout outputs: JSONL + markdown question exports + cycle configs",
                        metadata={
                            "cycle": cycle,
                            "run_tag": tag,
                            "jsonl_rows": rows,
                            "mean_solutions_per_question": mean_solutions,
                        },
                    )
                    artifact.add_file(str(cycle_jsonl), name=f"{cycle_tag}/samples.jsonl")
                    if md_dir.exists():
                        artifact.add_dir(str(md_dir), name=f"{cycle_tag}/markdown_exports")
                    artifact.add_file(str(rollout_cfg_path), name=f"{cycle_tag}/configs/rollout.yaml")
                    wandb_run.log_artifact(artifact, aliases=[cycle_tag, "latest"])
                    _wandb_log(
                        wandb_run,
                        {
                            "loop/status_code": 4,
                            "loop/status": "rollout_export_done",
                            "rollout/cycle": cycle,
                            "rollout/questions": rows,
                            "rollout/mean_solutions_per_question": mean_solutions,
                            "perf/rollout_generate_s": rollout_generate_s,
                            "perf/rollout_export_s": rollout_export_s,
                            "perf/questions_per_s": (float(rows) / rollout_generate_s) if rollout_generate_s > 0 else 0.0,
                            "perf/solutions_per_s": (float(rows * mean_solutions) / rollout_generate_s) if rollout_generate_s > 0 else 0.0,
                            **rollout_generate_gpu,
                            **_infer_rollout_diversity_stats(cycle_jsonl),
                            **_infer_rollout_reliability_stats(cycle_jsonl),
                        },
                    )
                    print("[loop] uploaded rollout artifacts to W&B", flush=True)
                except Exception as e:
                    print(f"[loop] failed to upload rollout artifacts ({type(e).__name__}: {e})", flush=True)

            if question_train_cmd is not None:
                if question_train_proc is not None:
                    question_train_s = _finish_async(question_train_proc, question_train_cmd, question_train_t0)
                else:
                    question_train_s, question_train_gpu = _run_timed_with_gpu_sampling(
                        question_train_cmd,
                        cwd=root,
                        env=env,
                        gpu_indices=gpu_indices,
                        **(
                            {
                                "wandb_run": wandb_run,
                                "heartbeat_prefix": "heartbeat/question_train",
                                "heartbeat_payload": {
                                    "loop/status": "question_train_running",
                                    "rollout/cycle": cycle,
                                    "loop/cycle_index": cycle,
                                },
                                "heartbeat_every_s": 30.0,
                            }
                            if wandb_run is not None
                            else {}
                        ),
                    )
                latest_question_ckpt = _latest_checkpoint(question_train_out)
                if latest_question_ckpt is not None:
                    current_generator_model = str(latest_question_ckpt)
                    print(f"[loop] next cycle generator model: {current_generator_model}", flush=True)
                else:
                    print("[loop] no question checkpoint found; next cycle keeps template generator", flush=True)

            train_cfg = _load_yaml(train_template_path)
            train_cfg.setdefault("data", {})
            train_cfg["data"]["source"] = "pairwise_jsonl"
            train_cfg["data"]["jsonl_path"] = str(rel_cycle_jsonl)
            train_cfg["data"]["split_train"] = "train"
            train_cfg["data"]["split_eval"] = "eval"
            train_cfg["data"]["eval_fraction"] = float(train_cfg["data"].get("eval_fraction", 0.2))
            train_cfg.setdefault("rollout", {})
            solver_cfg = rollout_cfg.get("solver", {}) if isinstance(rollout_cfg, dict) else {}
            solver_max_new_tokens = int(solver_cfg.get("max_new_tokens", 0) or 0)
            inherit_solver_max_new_tokens = bool(train_cfg["rollout"].get("inherit_solver_max_new_tokens", False))
            if inherit_solver_max_new_tokens and solver_max_new_tokens > 0:
                train_cfg["rollout"]["max_new_tokens"] = solver_max_new_tokens
            if args.use_rollout_strong_verifier_for_train:
                strong_cfg = rollout_cfg.get("strong_verifier", {}) if isinstance(rollout_cfg, dict) else {}
                if bool(strong_cfg.get("enabled", False)):
                    train_cfg.setdefault("reward", {})
                    train_cfg["reward"]["mode"] = "verdict"
                    train_teacher_cfg = train_cfg["reward"].setdefault("teacher", {})
                    provider = str(strong_cfg.get("provider", "")).strip().lower()
                    if provider == "openai":
                        train_teacher_cfg["api_base_url"] = str(strong_cfg.get("base_url", "https://api.openai.com/v1"))
                        train_teacher_cfg["api_model"] = str(strong_cfg.get("model", "gpt-4.1"))
                        train_teacher_cfg["api_timeout_s"] = float(strong_cfg.get("timeout_s", 120.0))
                        train_teacher_cfg.setdefault("api_max_tokens_param", "max_completion_tokens")
                    verify_path = str(rollout_cfg.get("judge", {}).get("verify_prompt_template_path", "")).strip()
                    if verify_path:
                        train_teacher_cfg["verify_prompt_template_path"] = verify_path
                    print(
                        f"[loop] train reward teacher sourced from rollout strong_verifier: "
                        f"{train_teacher_cfg.get('api_model')} @ {train_teacher_cfg.get('api_base_url')}",
                        flush=True,
                    )

            if current_solver_model:
                train_cfg.setdefault("model", {})
                train_cfg["model"]["name_or_path"] = current_solver_model

            cycle_train_out = run_dir / f"{cycle_tag}_grpo"
            rel_cycle_train_out = cycle_train_out.relative_to(root)
            train_cfg.setdefault("train", {})
            train_cfg["train"]["output_dir"] = str(rel_cycle_train_out)
            wandb_cfg = train_cfg["train"].setdefault("wandb", {})
            base_run_name = str(wandb_cfg.get("run_name", "self-play-grpo"))
            wandb_cfg["run_name"] = f"{base_run_name}-{tag}-{cycle_tag}"
            if args.separate_train_wandb:
                wandb_cfg["enabled"] = True
                wandb_cfg["group"] = wandb_group
                base_tags = wandb_cfg.get("tags", [])
                if not isinstance(base_tags, list):
                    base_tags = [str(base_tags)] if base_tags else []
                wandb_cfg["tags"] = list(dict.fromkeys([*base_tags, "self-play", "grpo", tag, cycle_tag]))
            else:
                wandb_cfg["enabled"] = False

            train_cfg_path = cfg_dir / f"{cycle_tag}_train.yaml"
            _dump_yaml(train_cfg_path, train_cfg)

            if wandb_run is not None:
                _wandb_log(wandb_run, {"loop/status_code": 5, "loop/status": "train_start", "rollout/cycle": cycle})

            train_module_args = ["--config", str(train_cfg_path)]
            if args.max_steps is not None:
                train_module_args += ["--max_steps", str(args.max_steps)]
            if args.max_train_samples is not None:
                train_module_args += ["--max_train_samples", str(args.max_train_samples)]
            if args.max_eval_samples is not None:
                train_module_args += ["--max_eval_samples", str(args.max_eval_samples)]
            train_cmd = _build_train_command(
                python=python,
                launcher=args.train_launcher,
                num_processes=args.train_num_processes,
                train_module_args=train_module_args,
            )
            train_s, train_gpu = _run_timed_with_gpu_sampling(
                train_cmd,
                cwd=root,
                env=env,
                gpu_indices=gpu_indices,
                **(
                    {
                        "wandb_run": wandb_run,
                        "heartbeat_prefix": "heartbeat/solver_train",
                        "heartbeat_payload": {
                            "loop/status": "solver_train_running",
                            "rollout/cycle": cycle,
                            "loop/cycle_index": cycle,
                        },
                        "heartbeat_every_s": 30.0,
                    }
                    if wandb_run is not None
                    else {}
                ),
            )

            latest_ckpt = _latest_checkpoint(cycle_train_out)
            if latest_ckpt is not None:
                current_solver_model = str(latest_ckpt)
                print(f"[loop] next cycle solver model: {current_solver_model}", flush=True)
            else:
                print("[loop] no checkpoint found; next cycle keeps template model", flush=True)

            cycle_s = time.perf_counter() - cycle_t0
            if wandb_run is not None:
                try:
                    perf_payload: Dict[str, float | str] = {
                        "loop/status_code": 6,
                        "loop/status": "cycle_done",
                        "rollout/cycle": cycle,
                        "perf/question_train_s": question_train_s,
                        "perf/train_s": train_s,
                        "perf/cycle_s": cycle_s,
                    }
                    if question_train_out is not None:
                        perf_payload.update(
                            {
                                f"question_grpo/{k}": v
                                for k, v in _extract_train_scalar_metrics(question_train_out).items()
                            }
                        )
                    perf_payload.update(
                        {f"question_train_gpu/{k}": v for k, v in question_train_gpu.items()}
                    )
                    perf_payload.update(_extract_train_perf_metrics(cycle_train_out))
                    perf_payload.update(_extract_train_scalar_metrics(cycle_train_out))
                    perf_payload.update(train_gpu)
                    util_keys = [k for k in train_gpu.keys() if k.endswith("/util_gpu_pct_mean")]
                    if util_keys:
                        perf_payload["perf/mfu_proxy_gpu_util_mean"] = float(sum(float(train_gpu[k]) for k in util_keys) / len(util_keys))
                    _wandb_log(wandb_run, perf_payload)
                except Exception as e:
                    print(f"[loop] failed to log perf metrics ({type(e).__name__}: {e})", flush=True)
    except Exception as e:
        exit_code = 1
        if wandb_run is not None:
            try:
                _wandb_log(wandb_run, {"loop/status_code": -1, "loop/status": "failed", "loop/error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
        raise
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=exit_code)
            except TypeError:
                wandb_run.finish()
    print("[loop] complete", flush=True)
    print(f"[loop] artifacts in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

