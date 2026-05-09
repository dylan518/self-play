from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from grpo_math.self_play.run_self_play_iteration import DEFAULT_CONFIG, run_iteration


LOOP_REPORT_NAME = "loop_report.md"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _write_loop_report(path: Path, *, config_path: str, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Self-Play Loop",
        "",
        f"- Config: `{config_path}`",
        f"- Iterations: {len(records)}",
        "",
        "## Iteration Outputs",
        "",
    ]
    for record in records:
        lines.append(
            f"- Iteration {record['iteration']}: "
            f"`{record['output_dir']}` (`{record['mode']}`)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_loop(
    *,
    config_path: str,
    output_dir: Path,
    iterations: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    skip_train: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    loop_cfg = cfg.get("loop", {}) if isinstance(cfg.get("loop"), dict) else {}
    total_iterations = int(iterations if iterations is not None else loop_cfg.get("iterations", 1))
    if total_iterations < 1:
        raise ValueError("iterations must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    current_config = config_path
    start_iteration = int(cfg.get("iteration", 0))

    for offset in range(total_iterations):
        iteration_num = start_iteration + offset
        iteration_dir = output_dir / f"iteration_{iteration_num:03d}"
        record = run_iteration(
            config_path=current_config,
            output_dir=iteration_dir,
            iteration=iteration_num,
            dry_run=dry_run,
            validate_only=validate_only,
            skip_train=skip_train,
            resume=resume,
            overwrite=overwrite,
        )
        records.append(record)
        current_config = str(iteration_dir / "next_iteration.yaml")

    report_path = output_dir / LOOP_REPORT_NAME
    _write_loop_report(report_path, config_path=config_path, records=records)
    return {
        "config": config_path,
        "output_dir": str(output_dir),
        "iterations": total_iterations,
        "records": records,
        "report": str(report_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    result = run_loop(
        config_path=args.config,
        output_dir=args.output_dir,
        iterations=args.iterations,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        skip_train=args.skip_train,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
