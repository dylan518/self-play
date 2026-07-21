#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINAL_ANSWER_RE = r"FINAL_ANSWER\s*:\s*-?\d+"
OUTPUT_PATH = Path("outputs/diagnostics/signal_biases.md")


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, pred_correct: bool, oracle_correct: bool) -> None:
        if pred_correct and oracle_correct:
            self.tp += 1
        elif pred_correct and not oracle_correct:
            self.fp += 1
        elif not pred_correct and not oracle_correct:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float | None:
        return ratio(self.tp + self.tn, self.total)

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)


@dataclass
class GroupAccumulator:
    parse_hits: int = 0
    parse_total: int = 0
    oracle_hits: int = 0
    oracle_total: int = 0
    verifier_hits: int = 0
    verifier_total: int = 0
    rsep_sum: float = 0.0
    rsep_total: int = 0


def ratio(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return num / den


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "correct"}:
            return True
        if lowered in {"false", "0", "no", "n", "incorrect"}:
            return False
    return None


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-") and stripped[1:].isdigit():
            return int(stripped)
        if stripped.isdigit():
            return int(stripped)
    return None


def extract_group_index(solution: dict[str, Any]) -> int | None:
    candidates = [
        solution.get("sampling_group"),
        solution.get("sampling_group_index"),
        solution.get("group_index"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("group_index", "index", "id"):
                maybe = to_int(candidate.get(key))
                if maybe is not None:
                    return maybe
        else:
            maybe = to_int(candidate)
            if maybe is not None:
                return maybe
    return None


def infer_parsed_final_answer(solution: dict[str, Any]) -> bool:
    parsed_value = solution.get("parsed_final_answer")
    if parsed_value is not None:
        parsed_text = str(parsed_value).strip()
        return parsed_text != ""
    text = solution.get("text")
    if not isinstance(text, str):
        return False
    return bool(re.search(FINAL_ANSWER_RE, text))


def counts_from_verification(verification: dict[str, Any]) -> tuple[int, int]:
    counts = verification.get("counts")
    if isinstance(counts, dict):
        n_correct = int(to_float(counts.get("CORRECT")) or 0)
        n_incorrect = int(to_float(counts.get("INCORRECT")) or 0)
        return n_correct, n_incorrect

    verdicts = verification.get("verdicts")
    n_correct = 0
    n_incorrect = 0
    if isinstance(verdicts, list):
        for verdict in verdicts:
            if not isinstance(verdict, str):
                continue
            if verdict.strip().upper() == "CORRECT":
                n_correct += 1
            elif verdict.strip().upper() == "INCORRECT":
                n_incorrect += 1
    return n_correct, n_incorrect


def infer_verifier_pred_correct(
    verification: dict[str, Any], n_correct: int, n_incorrect: int
) -> bool | None:
    for key in ("verifier_pred_correct", "predicted_correct", "is_correct"):
        if key in verification:
            coerced = as_bool(verification.get(key))
            if coerced is not None:
                return coerced
    if n_correct == 0 and n_incorrect == 0:
        return None
    return n_correct > n_incorrect


def classify_regime(n_correct: int, n_incorrect: int) -> str | None:
    if n_correct == 0 and n_incorrect == 0:
        return None
    if n_correct > 0 and n_incorrect == 0:
        return "unanimous_CORRECT"
    if n_incorrect > 0 and n_correct == 0:
        return "unanimous_INCORRECT"
    return "split_votes"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def analyze_paths(paths: list[Path]) -> tuple[str, str]:
    overall_correct_votes = 0
    overall_total_votes = 0
    overall_oracle_vote_hits = 0
    overall_oracle_vote_total = 0
    overall_oracle_solution_hits = 0
    overall_oracle_solution_total = 0

    question_rows: list[tuple[str, str, float | None, float | None, float | None]] = []
    file_rows: list[tuple[str, int, float | None, float | None, float | None]] = []

    regime_confusions = {
        "unanimous_CORRECT": Confusion(),
        "unanimous_INCORRECT": Confusion(),
        "split_votes": Confusion(),
    }

    group_stats: dict[int, GroupAccumulator] = defaultdict(GroupAccumulator)
    position_votes_a = 0
    position_votes_b = 0

    for path in paths:
        records = load_jsonl(path)
        file_correct_votes = 0
        file_total_votes = 0
        file_oracle_vote_hits = 0
        file_oracle_vote_total = 0

        for row_idx, record in enumerate(records):
            question_name = f"{path.name}#q{record.get('question_index', row_idx)}"
            solutions = record.get("solutions")
            if not isinstance(solutions, list):
                solutions = []

            solution_by_idx: dict[int, dict[str, Any]] = {}
            for pos, solution in enumerate(solutions):
                if not isinstance(solution, dict):
                    continue
                raw_solution_index = solution.get("solution_index", pos)
                solution_index = to_int(raw_solution_index)
                if solution_index is None:
                    solution_index = pos
                solution_by_idx[solution_index] = solution

                group_idx = extract_group_index(solution)
                if group_idx is not None:
                    parsed = infer_parsed_final_answer(solution)
                    group_stats[group_idx].parse_total += 1
                    if parsed:
                        group_stats[group_idx].parse_hits += 1

                oracle_correct = as_bool(solution.get("oracle_correct"))
                if oracle_correct is not None:
                    overall_oracle_solution_total += 1
                    if oracle_correct:
                        overall_oracle_solution_hits += 1
                    if group_idx is not None:
                        group_stats[group_idx].oracle_total += 1
                        if oracle_correct:
                            group_stats[group_idx].oracle_hits += 1

            verifications = record.get("solution_verifications")
            if not isinstance(verifications, list):
                verifications = []

            question_correct_votes = 0
            question_total_votes = 0
            question_oracle_vote_hits = 0
            question_oracle_vote_total = 0
            per_question_group_preds: dict[int, list[int]] = defaultdict(list)

            for verification in verifications:
                if not isinstance(verification, dict):
                    continue
                n_correct, n_incorrect = counts_from_verification(verification)
                n_votes = n_correct + n_incorrect
                if n_votes > 0:
                    overall_correct_votes += n_correct
                    overall_total_votes += n_votes
                    file_correct_votes += n_correct
                    file_total_votes += n_votes
                    question_correct_votes += n_correct
                    question_total_votes += n_votes

                solution_index = to_int(verification.get("solution_index"))
                if solution_index is None:
                    solution_index = -1
                solution = solution_by_idx.get(solution_index, {})
                oracle_correct = as_bool(solution.get("oracle_correct"))
                if n_votes > 0 and oracle_correct is not None:
                    overall_oracle_vote_total += n_votes
                    file_oracle_vote_total += n_votes
                    question_oracle_vote_total += n_votes
                    if oracle_correct:
                        overall_oracle_vote_hits += n_votes
                        file_oracle_vote_hits += n_votes
                        question_oracle_vote_hits += n_votes

                pred_correct = infer_verifier_pred_correct(
                    verification, n_correct, n_incorrect
                )
                regime = classify_regime(n_correct, n_incorrect)
                if (
                    regime is not None
                    and pred_correct is not None
                    and oracle_correct is not None
                ):
                    regime_confusions[regime].update(pred_correct, oracle_correct)

                group_idx = extract_group_index(solution) if solution else None
                if group_idx is not None and pred_correct is not None:
                    group_stats[group_idx].verifier_total += 1
                    if pred_correct:
                        group_stats[group_idx].verifier_hits += 1
                    per_question_group_preds[group_idx].append(int(pred_correct))

            question_verdict_rate = ratio(question_correct_votes, question_total_votes)
            question_oracle_vote_rate = ratio(
                question_oracle_vote_hits, question_oracle_vote_total
            )
            question_gap = None
            if (
                question_verdict_rate is not None
                and question_oracle_vote_rate is not None
            ):
                question_gap = question_verdict_rate - question_oracle_vote_rate
            question_rows.append(
                (
                    path.name,
                    question_name,
                    question_verdict_rate,
                    question_oracle_vote_rate,
                    question_gap,
                )
            )

            group_means: dict[int, float] = {}
            if per_question_group_preds:
                for group_idx, preds in per_question_group_preds.items():
                    maybe_mean = ratio(sum(preds), len(preds))
                    if maybe_mean is not None:
                        group_means[group_idx] = maybe_mean

            if len(group_means) < 2:
                reliability = record.get("reliability")
                if isinstance(reliability, dict):
                    raw_means = reliability.get("group_verify_means")
                    if isinstance(raw_means, list):
                        for group_idx, mean in enumerate(raw_means):
                            maybe = to_float(mean)
                            if maybe is not None:
                                group_means[group_idx] = maybe

            if len(group_means) >= 2:
                group_ids = sorted(group_means.keys())
                for group_idx in group_ids:
                    others = [
                        group_means[other_idx]
                        for other_idx in group_ids
                        if other_idx != group_idx
                    ]
                    if not others:
                        continue
                    contribution = group_means[group_idx] - (sum(others) / len(others))
                    group_stats[group_idx].rsep_sum += contribution
                    group_stats[group_idx].rsep_total += 1

            pairwise_rows = record.get("pairwise_judgments")
            if not isinstance(pairwise_rows, list):
                pairwise_rows = record.get("pairwise_comparisons")
            if not isinstance(pairwise_rows, list):
                pairwise_rows = []
            for pairwise in pairwise_rows:
                if not isinstance(pairwise, dict):
                    continue
                judge_trace = pairwise.get("judge_trace")
                if isinstance(judge_trace, dict) and isinstance(
                    judge_trace.get("raw_prefs"), list
                ):
                    pref_seq = judge_trace["raw_prefs"]
                else:
                    pref_seq = pairwise.get("prefs")
                if not isinstance(pref_seq, list):
                    continue
                for pref in pref_seq:
                    if not isinstance(pref, str):
                        continue
                    upper_pref = pref.strip().upper()
                    if upper_pref == "A":
                        position_votes_a += 1
                    elif upper_pref == "B":
                        position_votes_b += 1

        file_verdict_rate = ratio(file_correct_votes, file_total_votes)
        file_oracle_vote_rate = ratio(file_oracle_vote_hits, file_oracle_vote_total)
        file_gap = None
        if file_verdict_rate is not None and file_oracle_vote_rate is not None:
            file_gap = file_verdict_rate - file_oracle_vote_rate
        file_rows.append(
            (path.name, len(records), file_verdict_rate, file_oracle_vote_rate, file_gap)
        )

    overall_verdict_rate = ratio(overall_correct_votes, overall_total_votes)
    overall_oracle_vote_rate = ratio(overall_oracle_vote_hits, overall_oracle_vote_total)
    overall_oracle_solution_rate = ratio(
        overall_oracle_solution_hits, overall_oracle_solution_total
    )
    overall_gap_vote = None
    if overall_verdict_rate is not None and overall_oracle_vote_rate is not None:
        overall_gap_vote = overall_verdict_rate - overall_oracle_vote_rate
    overall_gap_solution = None
    if overall_verdict_rate is not None and overall_oracle_solution_rate is not None:
        overall_gap_solution = overall_verdict_rate - overall_oracle_solution_rate

    group_rows: list[tuple[int, GroupAccumulator, float | None, float | None, float | None, float | None]] = []
    for group_idx in sorted(group_stats.keys()):
        stats = group_stats[group_idx]
        parse_rate = ratio(stats.parse_hits, stats.parse_total)
        oracle_rate = ratio(stats.oracle_hits, stats.oracle_total)
        verifier_rate = ratio(stats.verifier_hits, stats.verifier_total)
        rsep_rate = ratio(stats.rsep_sum, stats.rsep_total)
        group_rows.append(
            (group_idx, stats, parse_rate, oracle_rate, verifier_rate, rsep_rate)
        )

    ranked_group_rows = sorted(
        group_rows,
        key=lambda item: (
            -item[5] if item[5] is not None else math.inf,
            -item[3] if item[3] is not None else math.inf,
        ),
    )

    parse_spread = None
    if len(group_rows) >= 2:
        valid_parse = [row[2] for row in group_rows if row[2] is not None]
        if len(valid_parse) >= 2:
            parse_spread = max(valid_parse) - min(valid_parse)

    position_total = position_votes_a + position_votes_b
    position_a_rate = ratio(position_votes_a, position_total)
    position_gap = None
    if position_a_rate is not None:
        position_gap = position_a_rate - 0.5

    text_lines: list[str] = []
    md_lines: list[str] = []

    text_lines.append("Signal Bias Diagnostics")
    text_lines.append("=" * 80)
    text_lines.append("Inputs:")
    for path in paths:
        text_lines.append(f"- {path}")
    text_lines.append("")

    md_lines.append("# Signal Bias Diagnostics")
    md_lines.append("")
    md_lines.append("## Inputs")
    for path in paths:
        md_lines.append(f"- `{path}`")
    md_lines.append("")

    text_lines.append("[1] verdict_bias")
    text_lines.append(
        f"- overall P(verdict=CORRECT): {pct(overall_verdict_rate)} "
        f"({overall_correct_votes}/{overall_total_votes})"
    )
    text_lines.append(
        f"- oracle correct rate (vote-weighted): {pct(overall_oracle_vote_rate)} "
        f"({overall_oracle_vote_hits}/{overall_oracle_vote_total})"
    )
    text_lines.append(
        f"- oracle correct rate (solution-level): {pct(overall_oracle_solution_rate)} "
        f"({overall_oracle_solution_hits}/{overall_oracle_solution_total})"
    )
    text_lines.append(f"- bias gap vs oracle (vote-weighted): {pct(overall_gap_vote)}")
    text_lines.append(f"- bias gap vs oracle (solution-level): {pct(overall_gap_solution)}")
    text_lines.append("")

    md_lines.append("## 1) verdict_bias")
    md_lines.append(
        f"- Overall **P(verdict=CORRECT)**: **{pct(overall_verdict_rate)}** "
        f"({overall_correct_votes}/{overall_total_votes} votes)"
    )
    md_lines.append(
        f"- Oracle correct rate (vote-weighted): **{pct(overall_oracle_vote_rate)}** "
        f"({overall_oracle_vote_hits}/{overall_oracle_vote_total} vote-weighted labels)"
    )
    md_lines.append(
        f"- Oracle correct rate (solution-level): **{pct(overall_oracle_solution_rate)}** "
        f"({overall_oracle_solution_hits}/{overall_oracle_solution_total} solutions)"
    )
    md_lines.append(
        f"- Gap vs oracle (vote-weighted): **{pct(overall_gap_vote)}** "
        "(positive means judge over-grants CORRECT)"
    )
    md_lines.append(
        f"- Gap vs oracle (solution-level): **{pct(overall_gap_solution)}**"
    )
    md_lines.append("")
    md_lines.append("### Per-file verdict bias")
    md_lines.append(
        "| file | questions | P(verdict=CORRECT) | oracle(vote-weighted) | gap |"
    )
    md_lines.append("|---|---:|---:|---:|---:|")
    for file_name, n_questions, verdict_rate, oracle_vote_rate, gap in file_rows:
        md_lines.append(
            f"| `{file_name}` | {n_questions} | {pct(verdict_rate)} | "
            f"{pct(oracle_vote_rate)} | {pct(gap)} |"
        )
    md_lines.append("")

    text_lines.append("per-file:")
    for file_name, n_questions, verdict_rate, oracle_vote_rate, gap in file_rows:
        text_lines.append(
            f"- {file_name}: questions={n_questions} verdict={pct(verdict_rate)} "
            f"oracle={pct(oracle_vote_rate)} gap={pct(gap)}"
        )
    text_lines.append("")

    md_lines.append("### Per-question verdict bias")
    md_lines.append(
        "| question | P(verdict=CORRECT) | oracle(vote-weighted) | gap |"
    )
    md_lines.append("|---|---:|---:|---:|")
    for _, question_name, verdict_rate, oracle_vote_rate, gap in question_rows:
        md_lines.append(
            f"| `{question_name}` | {pct(verdict_rate)} | {pct(oracle_vote_rate)} | {pct(gap)} |"
        )
    md_lines.append("")

    text_lines.append("[2] parse_bias")
    if parse_spread is not None:
        text_lines.append(
            f"- parse-rate spread across groups: {pct(parse_spread)} "
            "(max - min parse rate)"
        )
    else:
        text_lines.append("- parse-rate spread across groups: n/a")
    text_lines.append("- per-group parse/oracle rates:")
    for group_idx, _, parse_rate, oracle_rate, _, _ in group_rows:
        text_lines.append(
            f"  - group {group_idx}: parse={pct(parse_rate)} oracle={pct(oracle_rate)}"
        )
    text_lines.append("")

    md_lines.append("## 2) parse_bias")
    if parse_spread is not None:
        md_lines.append(
            f"- Parse-rate spread across groups (max-min): **{pct(parse_spread)}**"
        )
    else:
        md_lines.append("- Parse-rate spread across groups (max-min): n/a")
    md_lines.append(
        "- Inspect whether weak-temperature groups have lower parse rates and lower oracle solve rates."
    )
    md_lines.append("")
    md_lines.append(
        "| group | parse rate | oracle solve rate | verifier-correct rate |"
    )
    md_lines.append("|---:|---:|---:|---:|")
    for group_idx, _, parse_rate, oracle_rate, verifier_rate, _ in group_rows:
        md_lines.append(
            f"| {group_idx} | {pct(parse_rate)} | {pct(oracle_rate)} | {pct(verifier_rate)} |"
        )
    md_lines.append("")

    text_lines.append("[3] unanimity_bias")
    for regime in ("unanimous_CORRECT", "unanimous_INCORRECT", "split_votes"):
        conf = regime_confusions[regime]
        text_lines.append(
            f"- {regime}: n={conf.total} acc={pct(conf.accuracy)} "
            f"prec={pct(conf.precision)} rec={pct(conf.recall)} "
            f"(TP={conf.tp} FP={conf.fp} TN={conf.tn} FN={conf.fn})"
        )
    text_lines.append("")

    md_lines.append("## 3) unanimity_bias")
    md_lines.append(
        "Regimes compare `verifier_pred_correct` (derived from counts when absent) vs `oracle_correct`."
    )
    md_lines.append("")
    md_lines.append("| regime | n | accuracy | precision | recall | TP | FP | TN | FN |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for regime in ("unanimous_CORRECT", "unanimous_INCORRECT", "split_votes"):
        conf = regime_confusions[regime]
        md_lines.append(
            f"| {regime} | {conf.total} | {pct(conf.accuracy)} | {pct(conf.precision)} | "
            f"{pct(conf.recall)} | {conf.tp} | {conf.fp} | {conf.tn} | {conf.fn} |"
        )
    md_lines.append("")

    text_lines.append("[4] group_effects")
    text_lines.append("ranked groups by R_sep contribution:")
    for rank, (group_idx, _, parse_rate, oracle_rate, verifier_rate, rsep_rate) in enumerate(
        ranked_group_rows, start=1
    ):
        text_lines.append(
            f"- #{rank} group {group_idx}: r_sep={pct(rsep_rate)} "
            f"oracle={pct(oracle_rate)} verifier={pct(verifier_rate)} parse={pct(parse_rate)}"
        )
    text_lines.append("")

    md_lines.append("## 4) group_effects")
    md_lines.append(
        "R_sep per group is computed as each group's per-question verifier mean minus the mean of other groups on the same question, then averaged."
    )
    md_lines.append("")
    md_lines.append(
        "| rank (by R_sep) | group | mean oracle_correct | mean verifier-correct | parse rate | mean R_sep |"
    )
    md_lines.append("|---:|---:|---:|---:|---:|---:|")
    for rank, (
        group_idx,
        _,
        parse_rate,
        oracle_rate,
        verifier_rate,
        rsep_rate,
    ) in enumerate(ranked_group_rows, start=1):
        md_lines.append(
            f"| {rank} | {group_idx} | {pct(oracle_rate)} | {pct(verifier_rate)} | "
            f"{pct(parse_rate)} | {pct(rsep_rate)} |"
        )
    md_lines.append("")

    text_lines.append("[5] position_bias (optional)")
    if position_total == 0:
        text_lines.append("- no pairwise A/B vote rows found in provided files")
    else:
        text_lines.append(
            f"- P(prefer A position)={pct(position_a_rate)} "
            f"(A={position_votes_a}, B={position_votes_b}, gap-from-50/50={pct(position_gap)})"
        )
    text_lines.append("")

    md_lines.append("## 5) position_bias (optional)")
    if position_total == 0:
        md_lines.append(
            "- No pairwise A/B preference rows were available in the provided files."
        )
    else:
        md_lines.append(
            f"- P(prefer A position): **{pct(position_a_rate)}** "
            f"(A={position_votes_a}, B={position_votes_b})"
        )
        md_lines.append(f"- Gap from 50/50: **{pct(position_gap)}**")
    md_lines.append("")

    return "\n".join(text_lines).rstrip() + "\n", "\n".join(md_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose verifier signal/stability biases from rollout JSONLs."
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="One or more rollout JSONL files to analyze.",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.jsonl]
    text_report, md_report = analyze_paths(paths)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(md_report, encoding="utf-8")
    print(text_report, end="")
    print(f"Markdown summary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
