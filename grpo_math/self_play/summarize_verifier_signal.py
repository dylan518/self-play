from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_oracle(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for item in payload.get("comparisons", []):
        try:
            key = (int(item["question_index"]), int(item["solution_index"]))
        except Exception:
            continue
        out[key] = item
    return out


def _majority(verdicts: list[str]) -> tuple[str, int, int]:
    counts = Counter(verdicts)
    total = len(verdicts)
    if counts["CORRECT"] > counts["INCORRECT"]:
        return "CORRECT", counts["CORRECT"] - counts["INCORRECT"], total
    if counts["INCORRECT"] > counts["CORRECT"]:
        return "INCORRECT", counts["INCORRECT"] - counts["CORRECT"], total
    return "UNCLEAR", 0, total


def _bucket(margin: int, total: int) -> str:
    return f"{margin}/{total}" if total else "0/0"


def _nonzero(value: Any) -> bool:
    try:
        return abs(float(value)) > 1e-12
    except Exception:
        return False


def summarize(jsonl_path: Path, oracle_path: Path | None = None) -> dict[str, Any]:
    rows = _read_jsonl(jsonl_path)
    oracle = _read_oracle(oracle_path)
    majority_buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "oracle_agree": 0, "oracle_binary_agree": 0, "oracle_binary_n": 0}
    )
    canonical_counts: Counter[str] = Counter()
    question_notes: Counter[str] = Counter()
    trainable_questions = 0
    nonzero_question_rewards = 0
    nonzero_solver_advantages = 0
    trainable_solver_completions = 0
    high_agreement_total = 0
    high_agreement_oracle_agree = 0
    distinct_correct_by_question: list[dict[str, Any]] = []

    for row in rows:
        qidx = int(row.get("question_index", len(distinct_correct_by_question)))
        if row.get("trainable_for_proposer"):
            trainable_questions += 1
        if _nonzero(row.get("question_reward")):
            nonzero_question_rewards += 1
        summary = row.get("verdict_summary") or {}
        note = str(summary.get("note", "missing"))
        question_notes[note] += 1

        ver_by_idx = {
            int(v.get("solution_index", -1)): v
            for v in row.get("solution_verifications", []) or []
            if isinstance(v, dict)
        }
        correct_answers: set[str] = set()
        for sol in row.get("solutions", []) or []:
            if not isinstance(sol, dict):
                continue
            sidx = int(sol.get("solution_index", -1))
            if sol.get("trainable_for_solver"):
                trainable_solver_completions += 1
            if _nonzero(sol.get("solver_advantage")):
                nonzero_solver_advantages += 1
            ver = ver_by_idx.get(sidx, {})
            verdicts = [str(v) for v in ver.get("verdicts", [])]
            label, margin, total = _majority(verdicts)
            canonical_counts[label] += 1
            bucket = _bucket(margin, total)
            majority_buckets[bucket]["n"] += 1
            oracle_item = oracle.get((qidx, sidx))
            if oracle_item is not None:
                agrees = bool(oracle_item.get("agrees"))
                majority_buckets[bucket]["oracle_agree"] += int(agrees)
                if oracle_item.get("binary_agrees") is not None:
                    majority_buckets[bucket]["oracle_binary_n"] += 1
                    majority_buckets[bucket]["oracle_binary_agree"] += int(
                        bool(oracle_item.get("binary_agrees"))
                    )
                if total and margin == total:
                    high_agreement_total += 1
                    high_agreement_oracle_agree += int(
                        bool(oracle_item.get("binary_agrees"))
                        if oracle_item.get("binary_agrees") is not None
                        else agrees
                    )
            if label == "CORRECT":
                ans = sol.get("parsed_final_answer")
                correct_answers.add(str(ans) if ans is not None else "NONE")
        distinct_correct_by_question.append(
            {
                "question_index": qidx,
                "distinct_correct_answers": sorted(correct_answers),
                "distinct_correct_count": len(correct_answers),
                "verdict_note": note,
                "question_reward": row.get("question_reward"),
            }
        )

    majority_summary = {
        key: {
            "n": val["n"],
            "oracle_accuracy": (
                val["oracle_agree"] / val["n"]
                if oracle and val["n"]
                else None
            ),
            "oracle_binary_accuracy": (
                val["oracle_binary_agree"] / val["oracle_binary_n"]
                if oracle and val["oracle_binary_n"]
                else None
            ),
        }
        for key, val in sorted(majority_buckets.items())
    }
    return {
        "jsonl": str(jsonl_path),
        "oracle_json": str(oracle_path) if oracle_path else None,
        "num_questions": len(rows),
        "num_solutions": sum(len(row.get("solutions", []) or []) for row in rows),
        "canonical_verdict_counts": dict(canonical_counts),
        "question_note_counts": dict(question_notes),
        "majority_buckets": majority_summary,
        "high_agreement_oracle_accuracy": (
            high_agreement_oracle_agree / high_agreement_total
            if high_agreement_total
            else None
        ),
        "high_agreement_n": high_agreement_total,
        "trainable_questions": trainable_questions,
        "nonzero_question_rewards": nonzero_question_rewards,
        "trainable_solver_completions": trainable_solver_completions,
        "nonzero_solver_advantages": nonzero_solver_advantages,
        "distinct_correct_by_question": distinct_correct_by_question,
        "red_flags": {
            "all_questions_zero_reward": bool(rows) and nonzero_question_rewards == 0,
            "all_solver_advantages_zero": trainable_solver_completions > 0 and nonzero_solver_advantages == 0,
            "multi_distinct_correct_questions": sum(
                1 for item in distinct_correct_by_question if item["distinct_correct_count"] > 1
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize verifier-only rollout learning signal.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--oracle-json", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    oracle_path = Path(args.oracle_json) if args.oracle_json else None
    summary = summarize(Path(args.jsonl), oracle_path)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
