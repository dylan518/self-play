from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="grpo_math/configs/pairwise_rollouts_debug.yaml",
        help="Path to pairwise rollout config.",
    )
    ap.add_argument(
        "--reset-output",
        action="store_true",
        help="If set, delete existing output file before running.",
    )
    args = ap.parse_args()

    root = _repo_root()
    cfg_path = root / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Read output path from config so we can show the single-row result.
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_path = root / cfg["output"]["jsonl_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_output and out_path.exists():
        out_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "grpo_math.self_play.generate_pairwise_data",
        "--config",
        str(cfg_path),
    ]
    env = os.environ.copy()

    print("Running single rollout...")
    subprocess.run(cmd, cwd=root, env=env, check=True)

    if not out_path.exists():
        raise RuntimeError(f"Expected output not found: {out_path}")
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Output file is empty.")

    row = json.loads(lines[-1])
    print("\n=== Debug Single Rollout Result ===")
    print(f"Question: {row['question']}")
    print(f"Num solutions: {len(row['solutions'])}")
    print(f"Num pairwise comparisons: {len(row['pairwise_comparisons'])}")
    print(f"Preference stability: {row['reliability']['preference_stability']:.3f}")
    oracle = row.get("oracle")
    if oracle:
        print(
            "Oracle: "
            f"provider={oracle.get('provider')} "
            f"model={oracle.get('model')} "
            f"answer={oracle.get('answer')} "
            f"error={oracle.get('error')}"
        )
    print(
        "Ranking config: "
        f"method={row['ranking']['method']} "
        f"k={row['ranking']['k_factor']} "
        f"initial={row['ranking']['initial_rating']}"
    )
    print("\nAll solutions (ranked by Elo):")
    ranked = sorted(
        row["solutions"],
        key=lambda s: float(s.get("elo_rating", 0.0)),
        reverse=True,
    )
    for idx, sol in enumerate(ranked, start=1):
        text_preview = sol["text"].replace("\n", " ")
        if len(text_preview) > 220:
            text_preview = text_preview[:220] + "..."
        print(
            f"{idx}. solution_index={sol['solution_index']} "
            f"elo={sol.get('elo_rating', 0.0):.2f} "
            f"pairwise={sol.get('pairwise_score', 0.0):.3f} "
            f"final={sol.get('parsed_final_answer')} "
            f"oracle_correct={sol.get('oracle_correct')} "
            f"text='{text_preview}'"
        )

    print("\nPairwise voting details:")
    total_counts = collections.Counter()
    total_raw_counts = collections.Counter()
    for p_idx, pair in enumerate(row["pairwise_comparisons"], start=1):
        prefs = pair.get("prefs", [])
        counts = pair.get("counts", {})
        total_counts.update(prefs)
        print(
            f"{p_idx}. ({pair['i']} vs {pair['j']}) "
            f"prefs={prefs} counts={counts} consistency={pair.get('consistency', 0.0):.3f}"
        )
        trace = pair.get("judge_trace", {})
        raw_prefs = trace.get("raw_prefs", [])
        presentations = trace.get("presentations", [])
        if raw_prefs:
            total_raw_counts.update(raw_prefs)
            print(f"   raw_prefs={raw_prefs}")
        if presentations:
            print(f"   presentations={presentations}")

    if total_counts:
        print(f"\nMapped preference totals: {dict(total_counts)}")
    if total_raw_counts:
        print(f"Raw judge A/B totals: {dict(total_raw_counts)}")

    print(f"\nRaw JSONL row saved at: {out_path}")


if __name__ == "__main__":
    main()
