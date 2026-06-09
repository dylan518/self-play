#!/usr/bin/env python3
"""Measure text/reward diversity under a single pinned generator prompt (GRPO group sanity check)."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _reward_std(rewards: list[float]) -> float:
    if len(rewards) < 2:
        return 0.0
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    return math.sqrt(var)


def _jaccard_tokens(a: str, b: str) -> float:
    ta = set(_canon(a).split())
    tb = set(_canon(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _pairwise_mean_jaccard(questions: list[str]) -> float:
    if len(questions) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            total += _jaccard_tokens(questions[i], questions[j])
            n += 1
    return total / n if n else 0.0


def analyze_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trainable = [
        r
        for r in rows
        if r.get("trainable_for_proposer", True) and str(r.get("question") or "").strip()
    ]
    prompts = [str(r.get("proposer_prompt") or "") for r in trainable]
    unique_prompts = {p for p in prompts if p}
    questions = [str(r["question"]) for r in trainable]
    canonical = [_canon(q) for q in questions]
    rewards = [float(r.get("question_reward", 0.0)) for r in trainable]
    dup_rate = 1.0 - (len(set(canonical)) / max(1, len(canonical)))

    group_sizes = Counter(str(r.get("proposer_group_id") or "unknown") for r in trainable)

    return {
        "path": str(path),
        "n_rows": len(trainable),
        "unique_prompts": len(unique_prompts),
        "prompt_has_bank_placeholder": any("{question_bank_examples}" in p for p in unique_prompts),
        "unique_questions": len(set(canonical)),
        "duplicate_question_rate": dup_rate,
        "unique_rewards_rounded_1e4": len({round(r, 4) for r in rewards}),
        "reward_std": _reward_std(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "mean_pairwise_jaccard": _pairwise_mean_jaccard(questions),
        "proposer_group_ids": len(group_sizes),
        "largest_group_size": max(group_sizes.values()) if group_sizes else 0,
        "grpo_k_sufficient_signal": _reward_std(rewards) >= 0.05
        and len(set(canonical)) >= max(2, len(canonical) // 2),
        "grpo_k_sufficient_text_diversity": len(set(canonical)) >= max(2, len(canonical) // 2)
        and _pairwise_mean_jaccard(questions) <= 0.55,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", help="Rollout JSONL(s) or prompt-sample JSONL(s)")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    reports = [analyze_jsonl(Path(p)) for p in args.jsonl]
    out = {"reports": reports}
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
