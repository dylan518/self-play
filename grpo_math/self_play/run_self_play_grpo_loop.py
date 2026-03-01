from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
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


def _run_timed(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> float:
    t0 = time.perf_counter()
    _run(cmd, cwd=cwd, env=env)
    return time.perf_counter() - t0


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
) -> tuple[float, Dict[str, float]]:
    """
    Runs command and returns:
      - elapsed seconds
      - aggregated GPU metrics with keys perf/gpu_<idx>/<metric>_{mean|max}
    """
    print(f"[loop] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
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


def _infer_rollout_reliability_stats(path: Path) -> Dict[str, float]:
    """
    Aggregates oracle/R_sep style metrics from rollout JSONL rows.
    """
    rows = 0
    rsep_vals: list[float] = []
    pref_stab_vals: list[float] = []
    group_gap_present = 0

    oracle_enabled_rows = 0
    oracle_answer_rows = 0
    oracle_error_rows = 0
    oracle_solution_total = 0
    oracle_solution_correct = 0

    verify_rows_total = 0
    verify_rows_unanimous = 0
    verify_rows_parsed = 0
    # Verifier-vs-oracle solution-level diagnostics (single-verify mode).
    verify_oracle_compared = 0
    verify_oracle_majority_compared = 0
    verify_oracle_tie = 0
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

            oracle = obj.get("oracle", {}) if isinstance(obj.get("oracle"), dict) else {}
            if oracle.get("enabled") is True:
                oracle_enabled_rows += 1
                if oracle.get("answer") is not None:
                    oracle_answer_rows += 1
                if oracle.get("error"):
                    oracle_error_rows += 1

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
                    if n_c == n or n_i == n:
                        verify_rows_unanimous += 1

                # Oracle/verifier agreement metrics at solution level.
                oracle_answer = obj.get("oracle", {}).get("answer") if isinstance(obj.get("oracle"), dict) else None
                sols = obj.get("solutions", [])
                if isinstance(sols, list) and isinstance(oracle_answer, int):
                    for s in sols:
                        if not isinstance(s, dict):
                            continue
                        sidx_raw = s.get("solution_index")
                        parsed_ans = s.get("parsed_final_answer")
                        if not isinstance(sidx_raw, (int, float)) or parsed_ans is None:
                            continue
                        v = ver_by_idx.get(int(sidx_raw))
                        if not isinstance(v, dict):
                            continue
                        counts = v.get("counts", {})
                        if not isinstance(counts, dict):
                            continue
                        n_c = int(counts.get("CORRECT", 0) or 0)
                        n_i = int(counts.get("INCORRECT", 0) or 0)
                        if (n_c + n_i) <= 0:
                            continue

                        verify_oracle_compared += 1
                        if n_c == n_i:
                            verify_oracle_tie += 1
                            continue
                        verify_oracle_majority_compared += 1
                        verifier_pred_correct = n_c > n_i
                        oracle_is_correct = int(parsed_ans) == int(oracle_answer)

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

    out["rollout/oracle/enabled_rows"] = float(oracle_enabled_rows)
    out["rollout/oracle/answer_rows"] = float(oracle_answer_rows)
    out["rollout/oracle/error_rows"] = float(oracle_error_rows)
    if oracle_enabled_rows > 0:
        out["rollout/oracle/answer_rate"] = float(oracle_answer_rows) / float(oracle_enabled_rows)
        out["rollout/oracle/error_rate"] = float(oracle_error_rows) / float(oracle_enabled_rows)
    if oracle_solution_total > 0:
        out["rollout/oracle/solution_correct_rate"] = float(oracle_solution_correct) / float(oracle_solution_total)
        out["rollout/oracle/solution_scored"] = float(oracle_solution_total)

    out["rollout/verify/rows"] = float(verify_rows_total)
    if verify_rows_total > 0:
        out["rollout/verify/unanimous_frac"] = float(verify_rows_unanimous) / float(verify_rows_total)
    if verify_rows_parsed > 0 and verify_rows_total > 0:
        out["rollout/verify/parsed_votes_per_solution_mean"] = float(verify_rows_parsed) / float(verify_rows_total)

    out["rollout/verify_oracle/compared"] = float(verify_oracle_compared)
    out["rollout/verify_oracle/majority_compared"] = float(verify_oracle_majority_compared)
    out["rollout/verify_oracle/tie"] = float(verify_oracle_tie)
    if verify_oracle_compared > 0:
        out["rollout/verify_oracle/tie_frac"] = float(verify_oracle_tie) / float(verify_oracle_compared)
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
    args = ap.parse_args()

    root = _repo_root()
    rollout_template_path = (root / args.rollout_config).resolve()
    train_template_path = (root / args.train_config).resolve()
    if not rollout_template_path.exists():
        raise FileNotFoundError(f"Rollout config not found: {rollout_template_path}")
    if not train_template_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_template_path}")

    tag = args.run_tag.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wandb_group = f"self-play-grpo-{tag}"
    run_dir = root / "outputs" / "self_play_grpo_loop" / tag
    cfg_dir = run_dir / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python = sys.executable
    current_policy_model: str | None = None
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
                    "rollout_config": str(rollout_template_path),
                    "train_config": str(train_template_path),
                    "run_dir": str(run_dir),
                },
                reinit=True,
            )
            print("[loop] W&B rollout artifact logging enabled", flush=True)
        except Exception as e:
            print(f"[loop] W&B artifact logging disabled ({type(e).__name__}: {e})", flush=True)
            wandb_run = None

    for cycle in range(1, args.cycles + 1):
        cycle_tag = f"cycle_{cycle:03d}"
        print(f"[loop] ===== {cycle_tag} =====", flush=True)
        cycle_t0 = time.perf_counter()

        rollout_cfg = _load_yaml(rollout_template_path)
        cycle_jsonl = run_dir / f"{cycle_tag}_samples.jsonl"
        rel_cycle_jsonl = cycle_jsonl.relative_to(root)
        rollout_cfg.setdefault("output", {})
        rollout_cfg["output"]["jsonl_path"] = str(rel_cycle_jsonl)
        rollout_cfg["output"]["write_mode"] = "overwrite"
        rollout_cfg_path = cfg_dir / f"{cycle_tag}_rollout.yaml"
        _dump_yaml(rollout_cfg_path, rollout_cfg)

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
        )

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
                wandb_run.log(
                    {
                        "rollout/cycle": cycle,
                        "rollout/questions": rows,
                        "rollout/mean_solutions_per_question": mean_solutions,
                        "perf/rollout_generate_s": rollout_generate_s,
                        "perf/rollout_export_s": rollout_export_s,
                        "perf/questions_per_s": (float(rows) / rollout_generate_s) if rollout_generate_s > 0 else 0.0,
                        "perf/solutions_per_s": (
                            float(rows * mean_solutions) / rollout_generate_s
                        )
                        if rollout_generate_s > 0
                        else 0.0,
                        **rollout_generate_gpu,
                        **_infer_rollout_reliability_stats(cycle_jsonl),
                    }
                )
                print("[loop] uploaded rollout artifacts to W&B", flush=True)
            except Exception as e:
                print(f"[loop] failed to upload rollout artifacts ({type(e).__name__}: {e})", flush=True)

        train_cfg = _load_yaml(train_template_path)
        train_cfg.setdefault("data", {})
        train_cfg["data"]["source"] = "pairwise_jsonl"
        train_cfg["data"]["jsonl_path"] = str(rel_cycle_jsonl)
        train_cfg["data"]["split_train"] = "train"
        train_cfg["data"]["split_eval"] = "eval"
        train_cfg["data"]["eval_fraction"] = float(train_cfg["data"].get("eval_fraction", 0.2))
        if args.use_rollout_strong_verifier_for_train:
            strong_cfg = rollout_cfg.get("strong_verifier", {}) if isinstance(rollout_cfg, dict) else {}
            if bool(strong_cfg.get("enabled", False)):
                train_cfg.setdefault("reward", {})
                train_cfg["reward"]["mode"] = "verdict"
                train_teacher_cfg = train_cfg["reward"].setdefault("teacher", {})
                provider = str(strong_cfg.get("provider", "")).strip().lower()
                if provider == "openai":
                    train_teacher_cfg["api_base_url"] = str(
                        strong_cfg.get("base_url", "https://api.openai.com/v1")
                    )
                    train_teacher_cfg["api_model"] = str(strong_cfg.get("model", "gpt-4.1"))
                    train_teacher_cfg["api_timeout_s"] = float(strong_cfg.get("timeout_s", 120.0))
                    train_teacher_cfg.setdefault("api_max_tokens_param", "max_completion_tokens")
                # Keep prompt format anchored to the verifier prompt used by rollout judge.
                verify_path = str(
                    rollout_cfg.get("judge", {}).get("verify_prompt_template_path", "")
                ).strip()
                if verify_path:
                    train_teacher_cfg["verify_prompt_template_path"] = verify_path
                print(
                    f"[loop] train reward teacher sourced from rollout strong_verifier: "
                    f"{train_teacher_cfg.get('api_model')} @ {train_teacher_cfg.get('api_base_url')}",
                    flush=True,
                )

        if current_policy_model:
            train_cfg.setdefault("model", {})
            train_cfg["model"]["name_or_path"] = current_policy_model

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
            # Single-run mode: disable per-cycle trainer W&B runs.
            wandb_cfg["enabled"] = False

        train_cfg_path = cfg_dir / f"{cycle_tag}_train.yaml"
        _dump_yaml(train_cfg_path, train_cfg)

        train_cmd = [
            python,
            "-m",
            "grpo_math.trl.train_grpo_trl",
            "--config",
            str(train_cfg_path),
        ]
        if args.max_steps is not None:
            train_cmd += ["--max_steps", str(args.max_steps)]
        if args.max_train_samples is not None:
            train_cmd += ["--max_train_samples", str(args.max_train_samples)]
        if args.max_eval_samples is not None:
            train_cmd += ["--max_eval_samples", str(args.max_eval_samples)]
        train_s, train_gpu = _run_timed_with_gpu_sampling(
            train_cmd,
            cwd=root,
            env=env,
            gpu_indices=gpu_indices,
        )

        latest_ckpt = _latest_checkpoint(cycle_train_out)
        if latest_ckpt is not None:
            current_policy_model = str(latest_ckpt)
            print(f"[loop] next cycle model: {current_policy_model}", flush=True)
        else:
            print("[loop] no checkpoint found; next cycle keeps template model", flush=True)

        cycle_s = time.perf_counter() - cycle_t0
        if wandb_run is not None:
            try:
                perf_payload: Dict[str, float] = {
                    "rollout/cycle": cycle,
                    "perf/train_s": train_s,
                    "perf/cycle_s": cycle_s,
                }
                perf_payload.update(_extract_train_perf_metrics(cycle_train_out))
                perf_payload.update(_extract_train_scalar_metrics(cycle_train_out))
                perf_payload.update(train_gpu)
                # Heuristic utilization proxy to track "MFU-like" behavior across cycles.
                util_keys = [k for k in train_gpu.keys() if k.endswith("/util_gpu_pct_mean")]
                if util_keys:
                    perf_payload["perf/mfu_proxy_gpu_util_mean"] = float(
                        sum(float(train_gpu[k]) for k in util_keys) / len(util_keys)
                    )
                wandb_run.log(perf_payload)
            except Exception as e:
                print(f"[loop] failed to log perf metrics ({type(e).__name__}: {e})", flush=True)

    if wandb_run is not None:
        wandb_run.finish()
    print("[loop] complete", flush=True)
    print(f"[loop] artifacts in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

