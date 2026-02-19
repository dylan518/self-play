#!/usr/bin/env python3
"""Analyze single-verify rollout results from JSONL output files.

Usage:
    python tests/analyze_rollout_results.py outputs/.../file.jsonl [file2.jsonl ...]
"""
import argparse
import collections
import json
import math
import sys


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(records: list[dict], label: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {label}  ({len(records)} questions)")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------
    questions = [r.get("question", "") for r in records]
    unique_qs = {q.strip().lower() for q in questions if q.strip()}
    dup_rate = 1.0 - len(unique_qs) / len(questions) if questions else 0.0

    oracle_none = sum(
        1 for r in records if r.get("oracle", {}).get("answer") is None
    )
    oracle_error = sum(
        1 for r in records if r.get("oracle", {}).get("error") is not None
    )

    print("\n--- QUESTIONS ---")
    print(f"  total={len(questions)}  unique={len(unique_qs)}  duplicate_rate={dup_rate:.0%}")
    print(f"  oracle answer=None: {oracle_none}  oracle errors: {oracle_error}")

    # ------------------------------------------------------------------
    # Solution parse rate
    # ------------------------------------------------------------------
    all_solutions = [s for r in records for s in r.get("solutions", [])]
    parsed_count = sum(1 for s in all_solutions if s.get("parsed_final_answer") is not None)
    parse_pct = 100 * parsed_count / len(all_solutions) if all_solutions else 0.0

    print(f"\n--- PARSE RATE ---")
    print(f"  {parsed_count}/{len(all_solutions)} solutions have parsed_final_answer ({parse_pct:.0f}%)")

    # ------------------------------------------------------------------
    # Verifier accuracy vs oracle
    # ------------------------------------------------------------------
    tp = fp = tn = fn = skipped = 0
    # Only count solutions from questions where the oracle actually answered.
    questions_with_oracle = {
        r["question_index"]
        for r in records
        if r.get("oracle", {}).get("answer") is not None
    }
    for r in records:
        if r["question_index"] not in questions_with_oracle:
            skipped += len(r.get("solution_verifications", []))
            continue
        sol_by_idx = {s["solution_index"]: s for s in r.get("solutions", [])}
        for sv in r.get("solution_verifications", []):
            s_idx = sv["solution_index"]
            sol = sol_by_idx.get(s_idx, {})
            oracle_correct = sol.get("oracle_correct")
            if oracle_correct is None:
                skipped += 1
                continue
            counts = sv.get("counts", {})
            n_c = counts.get("CORRECT", 0)
            n_i = counts.get("INCORRECT", 0)
            verdict_correct = n_c > n_i
            if verdict_correct and oracle_correct:
                tp += 1
            elif verdict_correct and not oracle_correct:
                fp += 1
            elif not verdict_correct and not oracle_correct:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"\n--- VERIFIER vs ORACLE ---")
    if total == 0:
        print(f"  No oracle labels available (all skipped={skipped}) — oracle likely failing")
    else:
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}  (total={total}, skipped={skipped})")
        if tp + tn == 0:
            print("  WARNING: oracle_correct=False for all solutions — oracle errors skewing results")
        print(f"  Accuracy={acc:.1%}  Precision={prec:.1%}  Recall={rec:.1%}")

    # ------------------------------------------------------------------
    # Confidence calibration
    # ------------------------------------------------------------------
    conf_data: list[tuple[float, float, int]] = []  # (model_conf, agg_conf, correct)
    for r in records:
        if r["question_index"] not in questions_with_oracle:
            continue
        sol_by_idx = {s["solution_index"]: s for s in r.get("solutions", [])}
        for sv in r.get("solution_verifications", []):
            s_idx = sv["solution_index"]
            sol = sol_by_idx.get(s_idx, {})
            oracle_correct = sol.get("oracle_correct")
            if oracle_correct is None:
                continue
            counts = sv.get("counts", {})
            n_c = counts.get("CORRECT", 0)
            n_i = counts.get("INCORRECT", 0)
            verdict_correct = int(n_c > n_i)
            mc = sv.get("model_confidence_mean", 0.5)
            agg = sv.get("confidence", 1.0)
            conf_data.append((mc, agg, verdict_correct == int(oracle_correct)))

    print(f"\n--- CONFIDENCE CALIBRATION ---")
    if not conf_data:
        print("  no data")
    else:
        mc_vals = [d[0] for d in conf_data]
        agg_vals = [d[1] for d in conf_data]
        correct_vals = [d[2] for d in conf_data]

        mc_unique = sorted(set(mc_vals))
        agg_unique = sorted(set(agg_vals))
        print(f"  model_confidence_mean unique values: {mc_unique}")
        print(f"  agg confidence unique values:        {agg_unique}")
        if len(agg_unique) == 1:
            print(f"  NOTE: agg confidence is always {agg_unique[0]:.2f} — "
                  f"increase repeats_per_solution for meaningful values")

        # Bucket table
        buckets: dict[float, list[int]] = collections.defaultdict(list)
        for mc, _, correct in conf_data:
            b = round(mc * 10) / 10  # round to nearest 0.1
            buckets[b].append(correct)

        print(f"\n  model_confidence_mean bucket → verifier accuracy:")
        print(f"  {'conf':>6}  {'n':>5}  {'acc':>6}")
        for b in sorted(buckets):
            vals = buckets[b]
            bucket_acc = sum(vals) / len(vals)
            print(f"  {b:>6.1f}  {len(vals):>5}  {bucket_acc:>6.1%}")

        # Pearson r
        n = len(mc_vals)
        if n > 2:
            mx = sum(mc_vals) / n
            my = sum(correct_vals) / n
            num = sum((x - mx) * (y - my) for x, y in zip(mc_vals, correct_vals))
            den = math.sqrt(
                sum((x - mx) ** 2 for x in mc_vals)
                * sum((y - my) ** 2 for y in correct_vals)
            )
            r = num / den if den else float("nan")
            sign = "positive" if r > 0.1 else ("negative" if r < -0.1 else "near-zero")
            print(f"\n  Pearson r (model_confidence_mean vs verdict_correct): {r:+.3f}  [{sign}]  n={n}")
            if r < -0.05:
                print("  WARNING: confidence is negatively correlated with accuracy — "
                      "model is overconfident when wrong")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze single-verify rollout JSONL results."
    )
    parser.add_argument("files", nargs="+", help="JSONL output file(s) to analyze")
    args = parser.parse_args()

    for path in args.files:
        try:
            records = load_jsonl(path)
        except FileNotFoundError:
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            continue
        analyze(records, path)

    print()


if __name__ == "__main__":
    main()
