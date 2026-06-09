#!/usr/bin/env python3
"""Quantify preference-label signal in rollout -> DPO pair pipelines (no GPU)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from grpo_math.trl.train_question_from_rollout_rewards import (
    _question_text,
    _read_jsonl,
    _reward,
    build_preference_rows,
    split_preference_pairs,
)


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping config: {path}")
    return data


def _prompt_fallback(cfg: dict[str, Any]) -> str:
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt"), dict) else {}
    out = str(prompt_cfg.get("template", ""))
    template_path = str(prompt_cfg.get("template_path", "")).strip()
    if template_path:
        out = Path(template_path).read_text(encoding="utf-8")
    return out


def _reward_by_question_text(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Map DPO completion text -> reward (matches build_preference_rows chosen/rejected)."""
    out: dict[str, float] = {}
    for row in rows:
        if not row.get("trainable_for_proposer", True):
            continue
        q = _question_text(row).strip()
        if not q:
            continue
        out[q] = _reward(row)
    return out


def _pair_margins_from_pairs(
    pairs: list[dict[str, str]],
    reward_by_q: dict[str, float],
) -> list[float]:
    margins: list[float] = []
    for pair in pairs:
        chosen = str(pair.get("chosen", "")).strip()
        rejected = str(pair.get("rejected", "")).strip()
        rc = reward_by_q.get(chosen)
        rr = reward_by_q.get(rejected)
        if rc is None or rr is None:
            continue
        margins.append(float(rc) - float(rr))
    return margins


def _cohens_d(margins: list[float]) -> float:
    if len(margins) < 2:
        return 0.0
    mean = sum(margins) / len(margins)
    var = sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return mean / std if std > 1e-12 else (float("inf") if mean > 0 else 0.0)


def _histogram(margins: list[float], edges: list[float]) -> dict[str, int]:
    bins = {f"[{edges[i]:.2f},{edges[i + 1]:.2f})": 0 for i in range(len(edges) - 1)}
    bins[f"[{edges[-1]:.2f},inf)"] = 0
    for m in margins:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= m < edges[i + 1]:
                bins[f"[{edges[i]:.2f},{edges[i + 1]:.2f})"] += 1
                placed = True
                break
        if not placed and m >= edges[-1]:
            bins[f"[{edges[-1]:.2f},inf)"] += 1
    return bins


def _question_reward_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards: list[float] = []
    for row in rows:
        if not row.get("trainable_for_proposer", True):
            continue
        if not str(row.get("question") or "").strip():
            continue
        rewards.append(_reward(row))
    if not rewards:
        return {"n_questions": 0}
    rewards_sorted = sorted(rewards)
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / max(1, n - 1)
    std = math.sqrt(var)
    unique = len(set(round(r, 6) for r in rewards))
    return {
        "n_questions": n,
        "reward_mean": mean,
        "reward_std": std,
        "reward_min": rewards_sorted[0],
        "reward_max": rewards_sorted[-1],
        "reward_range": rewards_sorted[-1] - rewards_sorted[0],
        "reward_unique_rounded_1e6": unique,
        "reward_unique_fraction": unique / n,
    }


def analyze_rollout(
    cfg: dict[str, Any],
    rollout_jsonl: Path,
    *,
    eval_fraction: float,
    eval_seed_offset: int,
) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    min_gap = float(data_cfg.get("min_reward_gap", 0.05))
    max_pairs = int(data_cfg.get("max_pairs_per_group", 256))
    seed = int(cfg.get("seed", 1234))

    rows = _read_jsonl(rollout_jsonl)
    qstats = _question_reward_stats(rows)
    pairs = build_preference_rows(
        rows,
        prompt_fallback=_prompt_fallback(cfg),
        min_reward_gap=min_gap,
        max_pairs_per_group=max_pairs,
        seed=seed,
    )
    train_pairs, eval_pairs = split_preference_pairs(
        pairs,
        eval_fraction=float(eval_fraction),
        seed=seed + int(eval_seed_offset),
    )
    reward_by_q = _reward_by_question_text(rows)
    all_margins = _pair_margins_from_pairs(pairs, reward_by_q)
    eval_margins = _pair_margins_from_pairs(eval_pairs if eval_pairs else pairs, reward_by_q)

    n_q = max(1, int(qstats.get("n_questions", 0)))
    max_possible = n_q * (n_q - 1) // 2

    weak_cut = min_gap + 0.03
    strong_cut = 0.15

    def _margin_summary(margins: list[float]) -> dict[str, Any]:
        if not margins:
            return {"n_pairs": 0}
        mean = sum(margins) / len(margins)
        med = sorted(margins)[len(margins) // 2]
        return {
            "n_pairs": len(margins),
            "margin_mean": mean,
            "margin_median": med,
            "margin_min": min(margins),
            "margin_max": max(margins),
            "fraction_at_min_gap": sum(1 for m in margins if m < min_gap + 1e-9) / len(margins),
            "fraction_weak": sum(1 for m in margins if m < weak_cut) / len(margins),
            "fraction_strong": sum(1 for m in margins if m >= strong_cut) / len(margins),
            "cohens_d_on_margins": _cohens_d(margins),
            "label_snr_proxy": (qstats.get("reward_std", 0.0) or 0.0) / mean if mean > 1e-12 else 0.0,
            "margin_histogram": _histogram(
                margins, [min_gap, 0.08, 0.12, 0.15, 0.25, 0.40]
            ),
        }

    all_summary = _margin_summary(all_margins)
    eval_summary = _margin_summary(eval_margins)

    return {
        "rollout_jsonl": str(rollout_jsonl),
        "min_reward_gap": min_gap,
        "max_pairs_per_group": max_pairs,
        "question_stats": qstats,
        "pairs_all": {
            **all_summary,
            "pair_yield_fraction": len(pairs) / max(1, max_possible),
            "pairs_per_question": len(pairs) / n_q,
            "redundancy_factor": len(pairs) / max(1, n_q * (n_q - 1) // 2),
        },
        "pairs_train": {"n_pairs": len(train_pairs)},
        "pairs_eval": eval_summary,
        "eval_fraction": eval_fraction,
    }


def _dpo_loss_to_implied_acc(loss: float) -> float:
    """Rough chance model prefers chosen when per-example loss ≈ loss (i.i.d.)."""
    return float(math.exp(-loss))


def analyze_eval_metrics(metrics_path: Path) -> dict[str, Any]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss = float(data.get("metrics", {}).get("eval_loss", float("nan")))
    ln2 = math.log(2.0)
    return {
        "checkpoint": data.get("checkpoint"),
        "ref_checkpoint": data.get("ref_checkpoint"),
        "eval_pairs": data.get("eval_pairs"),
        "eval_loss": loss,
        "ln2": ln2,
        "delta_vs_ln2": loss - ln2 if not math.isnan(loss) else None,
        "implied_pairwise_acc_proxy": _dpo_loss_to_implied_acc(loss) if not math.isnan(loss) else None,
        "beats_random": loss < ln2 if not math.isnan(loss) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="grpo_math/configs/train_question_qwen35_overnight.yaml")
    ap.add_argument("--rollout-jsonl", action="append", default=[], help="One or more cycle_*_samples.jsonl")
    ap.add_argument("--eval-metrics", action="append", default=[], help="metrics.json from dpo_pair_eval")
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    report: dict[str, Any] = {"label_signal": [], "eval_signal": []}

    for path in args.rollout_jsonl:
        report["label_signal"].append(
            analyze_rollout(
                cfg,
                Path(path),
                eval_fraction=float(args.eval_fraction),
                eval_seed_offset=1,
            )
        )

    for path in args.eval_metrics:
        report["eval_signal"].append(analyze_eval_metrics(Path(path)))

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
