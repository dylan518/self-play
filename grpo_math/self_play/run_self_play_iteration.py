from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "grpo_math/configs/self_play_verdict_grpo_orchestrator_fixture.yaml"
COMMAND_LOG = "commands.jsonl"
REPORT_NAME = "report.md"
NEXT_CONFIG_NAME = "next_iteration.yaml"


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    enabled: bool = True
    train_stage: bool = False
    produces_checkpoint: bool = False


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _cfg_bool(cfg: dict[str, Any], section: str, key: str, default: bool = True) -> bool:
    value = cfg.get(section, {})
    if not isinstance(value, dict):
        return default
    return bool(value.get(key, default))


def _cfg_str(cfg: dict[str, Any], section: str, key: str, default: str) -> str:
    value = cfg.get(section, {})
    if not isinstance(value, dict):
        return default
    raw = value.get(key, default)
    return str(raw)


def build_iteration_stages(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    iteration: int,
    skip_train: bool = False,
) -> list[Stage]:
    rollout_config = _cfg_str(cfg, "rollout", "config", "grpo_math/configs/pairwise_rollouts_smoke.yaml")
    solver_config = _cfg_str(
        cfg,
        "solver_train",
        "config",
        "grpo_math/configs/train_solver_verdict_grpo_smoke.yaml",
    )
    proposer_config = _cfg_str(
        cfg,
        "proposer_train",
        "config",
        "grpo_math/configs/train_proposer_question_grpo_smoke.yaml",
    )
    eval_config = _cfg_str(cfg, "evaluation", "config", "grpo_math/configs/train_smoke_trl.yaml")

    rollout_module = _cfg_str(cfg, "rollout", "module", "grpo_math.self_play.collect_self_play_rollouts")
    solver_module = _cfg_str(cfg, "solver_train", "module", "grpo_math.trl.train_solver_verdict_grpo")
    proposer_module = _cfg_str(cfg, "proposer_train", "module", "grpo_math.trl.train_proposer_question_grpo")
    eval_module = _cfg_str(cfg, "evaluation", "module", "grpo_math.eval.eval_gsm8k")

    solver_dir = output_dir / "solver"
    proposer_dir = output_dir / "proposer"

    return [
        Stage(
            name="rollout",
            command=[
                sys.executable,
                "-m",
                rollout_module,
                "--config",
                rollout_config,
                "--overwrite",
            ],
            enabled=_cfg_bool(cfg, "rollout", "enabled", True),
        ),
        Stage(
            name="solver_grpo",
            command=[
                sys.executable,
                "-m",
                solver_module,
                "--config",
                solver_config,
                "--output_dir",
                str(solver_dir),
            ],
            enabled=_cfg_bool(cfg, "solver_train", "enabled", True) and not skip_train,
            train_stage=True,
            produces_checkpoint=True,
        ),
        Stage(
            name="proposer_grpo",
            command=[
                sys.executable,
                "-m",
                proposer_module,
                "--config",
                proposer_config,
                "--output_dir",
                str(proposer_dir),
            ],
            enabled=_cfg_bool(cfg, "proposer_train", "enabled", True) and not skip_train,
            train_stage=True,
            produces_checkpoint=True,
        ),
        Stage(
            name="evaluation",
            command=[
                sys.executable,
                "-m",
                eval_module,
                "--config",
                eval_config,
                "--checkpoint",
                str(solver_dir),
            ],
            enabled=_cfg_bool(cfg, "evaluation", "enabled", True),
        ),
    ]


def _command_record(stage: Stage, *, status: str, returncode: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage.name,
        "status": status,
        "command": stage.command,
        "command_text": _quote_command(stage.command),
    }
    if returncode is not None:
        record["returncode"] = returncode
    return record


def _append_command_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _run_stage(stage: Stage, *, command_log: Path, execute: bool) -> dict[str, Any]:
    if not stage.enabled:
        record = _command_record(stage, status="skipped")
        _append_command_record(command_log, record)
        return record

    if not execute:
        record = _command_record(stage, status="planned")
        _append_command_record(command_log, record)
        return record

    completed = subprocess.run(stage.command, check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    record = _command_record(stage, status=status, returncode=completed.returncode)
    _append_command_record(command_log, record)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, stage.command)
    return record


def _next_iteration_config(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    iteration: int,
) -> dict[str, Any]:
    next_cfg = dict(cfg)
    next_cfg["iteration"] = int(iteration) + 1
    next_cfg["resume_from"] = str(output_dir)
    next_cfg["initial_checkpoints"] = {
        "solver": str(output_dir / "solver"),
        "proposer": str(output_dir / "proposer"),
    }
    return next_cfg


def _washu_commands(stages: list[Stage], *, dry_run: bool) -> list[str]:
    commands: list[str] = []
    prefix = "DRY_RUN=1 " if dry_run else ""
    for stage in stages:
        if stage.enabled:
            commands.append(prefix + _quote_command(stage.command))
    return commands


def write_report(
    path: Path,
    *,
    cfg_path: str,
    output_dir: Path,
    iteration: int,
    mode: str,
    records: list[dict[str, Any]],
    washu_commands: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Self-Play Iteration {iteration}",
        "",
        f"- Config: `{cfg_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Mode: `{mode}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Stage Status",
        "",
    ]
    for record in records:
        lines.append(f"- `{record['stage']}`: {record['status']}")
    lines.extend(["", "## Commands", ""])
    for record in records:
        lines.append(f"- `{record['stage']}`: `{record['command_text']}`")
    lines.extend(["", "## Exact Wash U Commands", ""])
    if washu_commands:
        lines.append("```bash")
        lines.extend(washu_commands)
        lines.append("```")
    else:
        lines.append("No enabled stages.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_iteration(
    *,
    config_path: str,
    output_dir: Path,
    iteration: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    skip_train: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    iteration_num = int(iteration if iteration is not None else cfg.get("iteration", 0))
    if output_dir.exists() and any(output_dir.iterdir()) and not (overwrite or resume or validate_only):
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite or --resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    command_log = output_dir / COMMAND_LOG
    if command_log.exists() and overwrite and not resume:
        command_log.unlink()

    stages = build_iteration_stages(cfg, output_dir=output_dir, iteration=iteration_num, skip_train=skip_train)
    execute = not dry_run and not validate_only
    records = [_run_stage(stage, command_log=command_log, execute=execute) for stage in stages]

    next_cfg = _next_iteration_config(cfg, output_dir=output_dir, iteration=iteration_num)
    _write_yaml(output_dir / NEXT_CONFIG_NAME, next_cfg)

    mode = "validate-only" if validate_only else "dry-run" if dry_run else "execute"
    washu_commands = _washu_commands(stages, dry_run=dry_run or validate_only)
    write_report(
        output_dir / REPORT_NAME,
        cfg_path=config_path,
        output_dir=output_dir,
        iteration=iteration_num,
        mode=mode,
        records=records,
        washu_commands=washu_commands,
    )
    return {
        "config": config_path,
        "output_dir": str(output_dir),
        "iteration": iteration_num,
        "mode": mode,
        "records": records,
        "next_iteration_config": str(output_dir / NEXT_CONFIG_NAME),
        "report": str(output_dir / REPORT_NAME),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    result = run_iteration(
        config_path=args.config,
        output_dir=args.output_dir,
        iteration=args.iteration,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        skip_train=args.skip_train,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
