"""Separate judge rollout: program-consensus labels for the band-filtered subset.

Runs AFTER upload.py's difficulty filter, on the filtered parquet only (the
subsample — a few hundred questions, not the full generation batch). For each
question the judge model writes K Python programs; subprocess execution +
mathruler-equivalence voting produce a verified label (see verify.py).

Output parquet differs from the input in exactly one column: `answer` becomes
the program-consensus label (questions with no consensus are dropped). The
majority-vote `score` (difficulty / band semantics) is UNTOUCHED, so difficulty
selection and label verifiability stay separated.

A sidecar JSONL records majority-vs-verified agreement per question for the
label-error analysis.
"""

import multiprocessing as _mp
import argparse
import json

import vllm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from verify import _answers_match, verify_questions, self_critique_keep, SELF_FILTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Judge model (HF path); typically the current solver checkpoint.")
    ap.add_argument("--in_parquet", type=str, required=True)
    ap.add_argument("--out_parquet", type=str, required=True)
    ap.add_argument("--report_jsonl", type=str, default="")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    import os
    if not os.path.exists(args.in_parquet) or os.path.getsize(args.in_parquet) == 0:
        raise SystemExit(f"[judge] FATAL: input parquet missing or empty: {args.in_parquet} "
                         "(did the band filter keep zero questions, or did evaluation fail upstream?)")
    ds = load_dataset("parquet", data_files=args.in_parquet, split="train")
    questions = list(ds["problem"])
    if not questions:
        raise SystemExit(f"[judge] FATAL: zero rows in {args.in_parquet}")
    print(f"[judge] {len(questions)} band-filtered questions to verify")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=(os.getenv("WORKER_ENFORCE_EAGER","0")=="1"),
        dtype="bfloat16",
        max_model_len=int(os.getenv("JUDGE_MAX_MODEL_LEN", "8192")),
    )
    verdicts = verify_questions(model, tokenizer, questions)

    kept, report = [], []
    for row, (verified_answer, prog_outs) in zip(ds, verdicts):
        majority_answer = row["answer"]
        entry = {
            "problem": row["problem"],
            "majority_answer": majority_answer,
            "score": row["score"],
            "program_outputs": prog_outs,
            "verified_answer": verified_answer,
        }
        if verified_answer is not None:
            entry["majority_agrees"] = bool(_answers_match(majority_answer, verified_answer))
            kept.append({"problem": row["problem"], "answer": verified_answer, "score": row["score"]})
        report.append(entry)

    n = len(report)
    n_verified = sum(1 for e in report if e["verified_answer"] is not None)
    n_agree = sum(1 for e in report if e.get("majority_agrees"))
    print(f"[judge] consensus on {n_verified}/{n}; majority label agreed on {n_agree}/{n_verified} of those")

    if SELF_FILTER and kept:
        mask = self_critique_keep(model, tokenizer, [(k["problem"], k["answer"]) for k in kept])
        survivors = {k["problem"] for k, m in zip(kept, mask) if m}
        before = len(kept)
        kept = [k for k, m in zip(kept, mask) if m]
        for e in report:
            if e["verified_answer"] is not None:
                e["self_filter_kept"] = e["problem"] in survivors
        print(f"[judge] self-filter (model sanity gate) kept {len(kept)}/{before} verified questions")

    Dataset.from_list(kept).to_parquet(args.out_parquet)
    print(f"[judge] wrote {len(kept)} verified rows to {args.out_parquet}")
    if args.report_jsonl:
        with open(args.report_jsonl, "w", encoding="utf-8") as f:
            for e in report:
                f.write(json.dumps(e) + "\n")
        print(f"[judge] report: {args.report_jsonl}")


if __name__ == "__main__" and _mp.parent_process() is None:
    main()
