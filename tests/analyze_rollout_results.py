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
    # Oracle solve rate: what fraction of solutions / questions are correct
    # ------------------------------------------------------------------
    oracle_solutions = [s for r in records for s in r.get("solutions", []) if s.get("oracle_correct") is not None]
    if oracle_solutions:
        n_correct_sols = sum(1 for s in oracle_solutions if s.get("oracle_correct") is True)
        overall_solve_rate = n_correct_sols / len(oracle_solutions)

        # Per-group solve rates
        groups: dict[int, list[int]] = {}
        for s in oracle_solutions:
            g = s.get("sampling_group", 0)
            if g not in groups:
                groups[g] = []
            groups[g].append(1 if s.get("oracle_correct") else 0)

        # Question-level: fraction of questions where at least one solution is correct
        qs_any_correct = sum(
            1 for r in records
            if any(s.get("oracle_correct") is True for s in r.get("solutions", []))
            and r.get("oracle", {}).get("answer") is not None
        )
        qs_with_oracle = sum(1 for r in records if r.get("oracle", {}).get("answer") is not None)
        # Fraction of questions where ALL solutions are correct
        qs_all_correct = sum(
            1 for r in records
            if r.get("oracle", {}).get("answer") is not None
            and r.get("solutions")
            and all(
                s.get("oracle_correct") is True
                for s in r["solutions"]
                if s.get("oracle_correct") is not None
            )
        )

        print(f"\n--- ORACLE SOLVE RATE ---")
        print(f"  Overall: {n_correct_sols}/{len(oracle_solutions)} solutions correct ({overall_solve_rate:.1%})")
        for g in sorted(groups):
            vals = groups[g]
            label = "strong" if g == 0 else "weak" if g == 1 else f"group{g}"
            print(f"  Group {g} ({label}): {sum(vals)}/{len(vals)} correct ({sum(vals)/len(vals):.1%})")
        print(f"  Questions with ≥1 correct solution: {qs_any_correct}/{qs_with_oracle} ({qs_any_correct/max(1,qs_with_oracle):.1%})")
        print(f"  Questions where ALL solutions correct: {qs_all_correct}/{qs_with_oracle} ({qs_all_correct/max(1,qs_with_oracle):.1%})")

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

    # ------------------------------------------------------------------
    # distinct_correct as verifiability signal
    # ------------------------------------------------------------------
    print(f"\n--- VERIFIABILITY: distinct_correct per question ---")
    print(f"  {'Q':>3}  {'oracle':>8}  {'d_correct':>9}  {'v_correct':>9}  {'fp_rate':>7}  {'note'}")

    # per-question stats
    dc_fp_pairs: list[tuple[int, float]] = []  # (distinct_correct, fp_rate) for oracle questions
    bucket_stats: dict[str, dict] = {"0": {"fp": 0, "total": 0}, "1": {"fp": 0, "total": 0}, "2+": {"fp": 0, "total": 0}}

    for r in records:
        oracle_ans = r.get("oracle", {}).get("answer")
        sol_by_idx = {s["solution_index"]: s for s in r.get("solutions", [])}

        correct_answers: set = set()
        tp_q = fp_q = tn_q = fn_q = 0

        for sv in r.get("solution_verifications", []):
            sol = sol_by_idx.get(sv["solution_index"], {})
            parsed = sol.get("parsed_final_answer")
            oracle_correct = sol.get("oracle_correct")
            counts = sv.get("counts", {})
            n_c = counts.get("CORRECT", 0)
            n_i = counts.get("INCORRECT", 0)
            verdict = n_c > n_i
            if verdict:
                correct_answers.add(parsed)
            if oracle_ans is not None and oracle_correct is not None:
                if verdict and oracle_correct:       tp_q += 1
                elif verdict and not oracle_correct: fp_q += 1
                elif not verdict and oracle_correct: fn_q += 1
                else:                                tn_q += 1

        dc = len(correct_answers)
        v_correct = tp_q + fp_q
        total_q = tp_q + fp_q + tn_q + fn_q
        fp_rate = fp_q / max(1, tp_q + fp_q) if (tp_q + fp_q) > 0 else float("nan")

        if oracle_ans is not None and total_q > 0:
            note = "no_correct_verdicts" if v_correct == 0 else ("✓ verifiable" if dc == 1 and fp_q == 0 else ("✗ multi-correct" if dc > 1 else ""))
            fp_str = f"{fp_rate:.0%}"
            dc_fp_pairs.append((dc, fp_q / max(1, v_correct) if v_correct > 0 else 0.0))
            bk = "0" if dc == 0 else ("1" if dc == 1 else "2+")
            bucket_stats[bk]["fp"] += fp_q
            bucket_stats[bk]["total"] += max(1, v_correct)
        else:
            note = "oracle=None"
            fp_str = "n/a"

        print(f"  {r['question_index']:>3}  {str(oracle_ans):>8}  {dc:>9}  {v_correct:>9}  {fp_str:>7}  {note}")

    # bucket summary
    print(f"\n  distinct_correct bucket → avg FP rate (oracle-answered questions):")
    print(f"  {'bucket':>8}  {'questions':>9}  {'fp_rate':>7}")
    for bk in ["0", "1", "2+"]:
        s = bucket_stats[bk]
        n_q = sum(1 for dc, _ in dc_fp_pairs if (bk == "0" and dc == 0) or (bk == "1" and dc == 1) or (bk == "2+" and dc > 1))
        rate = s["fp"] / s["total"] if s["total"] > 0 else float("nan")
        print(f"  {bk:>8}  {n_q:>9}  {rate:>7.0%}")

    # Pearson r: distinct_correct vs fp_rate
    if len(dc_fp_pairs) > 2:
        xs = [d[0] for d in dc_fp_pairs]
        ys = [d[1] for d in dc_fp_pairs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        r = num / den if den else float("nan")
        sign = "positive" if r > 0.1 else ("negative" if r < -0.1 else "near-zero")
        print(f"\n  Pearson r (distinct_correct vs fp_rate): {r:+.3f}  [{sign}]  n={n} questions")
        if r > 0.3:
            print("  ✓ distinct_correct is a meaningful verifiability signal")
        elif r > 0.1:
            print("  ~ weak signal — needs more data")
        else:
            print("  ✗ no signal — distinct_correct does not predict fp_rate here")

    # ------------------------------------------------------------------
    # R_sep: strong group verify score minus weak group verify score
    # ------------------------------------------------------------------
    print(f"\n--- R_SEP (strong - weak sampling separation) ---")
    r_sep_vals: list[float] = []
    r_sep_oracle_acc: list[float] = []  # oracle accuracy per question (for correlation)

    has_groups = any(
        any(s.get("sampling_group") is not None for s in r.get("solutions", []))
        for r in records
    )
    if not has_groups:
        print("  no sampling_group field in solutions — run with sampling_groups config to enable R_sep")
    else:
        print(f"  {'Q':>3}  {'oracle':>8}  {'g0_mean':>8}  {'g1_mean':>8}  {'r_sep':>7}  {'note'}")
        for r in records:
            rel = r.get("reliability", {})
            r_sep = rel.get("r_sep")
            group_means = rel.get("group_verify_means", [])
            oracle_ans = r.get("oracle", {}).get("answer")

            # Compute oracle accuracy for this question (fraction of solutions matching oracle)
            oracle_acc: float | None = None
            sols = r.get("solutions", [])
            if oracle_ans is not None and sols:
                oracle_hits = sum(1 for s in sols if s.get("oracle_correct") is True)
                oracle_acc = oracle_hits / len(sols)

            if r_sep is not None:
                r_sep_vals.append(r_sep)
                if oracle_acc is not None:
                    r_sep_oracle_acc.append(oracle_acc)

            g0_str = f"{group_means[0]:.2f}" if len(group_means) > 0 else "n/a"
            g1_str = f"{group_means[1]:.2f}" if len(group_means) > 1 else "n/a"
            rsep_str = f"{r_sep:+.2f}" if r_sep is not None else "n/a"
            note = ""
            if r_sep is not None:
                if r_sep > 0.15:
                    note = "✓ strong>weak"
                elif r_sep < -0.15:
                    note = "✗ weak>strong (verifier noise)"
                else:
                    note = "~ no separation"
            print(
                f"  {r['question_index']:>3}  {str(oracle_ans):>8}  "
                f"{g0_str:>8}  {g1_str:>8}  {rsep_str:>7}  {note}"
            )

        if r_sep_vals:
            mean_r_sep = sum(r_sep_vals) / len(r_sep_vals)
            frac_positive = sum(1 for v in r_sep_vals if v > 0.0) / len(r_sep_vals)
            print(f"\n  mean R_sep={mean_r_sep:+.3f}  frac_positive={frac_positive:.0%}  n={len(r_sep_vals)}")
            if mean_r_sep > 0.1:
                print("  ✓ strong group reliably outscores weak — verifier is discriminative")
            elif mean_r_sep > 0.0:
                print("  ~ weak positive separation — needs more data or larger temperature gap")
            else:
                print("  ✗ no separation — verifier cannot distinguish strong vs weak sampling")

        # Pearson r: r_sep vs oracle accuracy (do questions with high r_sep have higher oracle accuracy?)
        if len(r_sep_vals) > 2 and len(r_sep_oracle_acc) == len(r_sep_vals):
            xs = r_sep_vals
            ys = r_sep_oracle_acc
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
            r = num / den if den else float("nan")
            print(f"  Pearson r (r_sep vs oracle_acc): {r:+.3f}  n={n} questions")
            if r > 0.3:
                print("  ✓ R_sep is correlated with oracle accuracy — good training signal")

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
