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

        rollout_generate_s = _run_timed(
            [
                python,
                "-m",
                "grpo_math.self_play.generate_pairwise_data",
                "--config",
                str(rollout_cfg_path),
            ],
            cwd=root,
            env=env,
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
        wandb_cfg["group"] = wandb_group
        base_tags = wandb_cfg.get("tags", [])
        if not isinstance(base_tags, list):
            base_tags = [str(base_tags)] if base_tags else []
        wandb_cfg["tags"] = list(dict.fromkeys([*base_tags, "self-play", "grpo", tag, cycle_tag]))

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
        train_s = _run_timed(train_cmd, cwd=root, env=env)

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
                wandb_run.log(perf_payload)
            except Exception as e:
                print(f"[loop] failed to log perf metrics ({type(e).__name__}: {e})", flush=True)

    if wandb_run is not None:
        wandb_run.finish()
    print("[loop] complete", flush=True)
    print(f"[loop] artifacts in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

