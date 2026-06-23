#!/usr/bin/env python3
"""Convert R-Zero pipeline output (problem/answer[/score/verified]) into a verl 0.8
RLHFDataset parquet with PRE-RENDERED solver prompts (verl 0.8 has no runtime format_prompt).
Schema: prompt=[system,user], data_source, ability, reward_model{style,ground_truth}, extra_info.
Usage: to_verl08_parquet.py --in_parquet IN --out_parquet OUT [--data_source solver_train]
"""
import argparse, os
import pandas as pd

SOLVER_SYS = r"Please reason step by step, and put your final answer within \boxed{}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--data_source", default="solver_train")
    ap.add_argument("--problem_key", default="problem")
    ap.add_argument("--answer_key", default="answer")
    args = ap.parse_args()
    df = pd.read_parquet(args.in_parquet)
    cols = set(df.columns)
    pk = args.problem_key if args.problem_key in cols else ("question" if "question" in cols else None)
    ak = args.answer_key
    if len(df) == 0 or pk is None or ak not in cols:
        # empty / no usable rows (e.g. judge verified 0 questions) -> write empty schema, exit 0
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.out_parquet)), exist_ok=True)
        pd.DataFrame(columns=["data_source", "prompt", "ability", "reward_model", "extra_info"]).to_parquet(args.out_parquet)
        print(f"WROTE {args.out_parquet} rows=0 (empty input: {len(df)} rows, cols={cols})")
        return
    rows = []
    for i, r in df.iterrows():
        q = str(r[pk]).strip(); a = str(r[ak]).strip()
        if not q or a in ("", "None"):
            continue
        rows.append({
            "data_source": args.data_source,
            "prompt": [{"role": "system", "content": SOLVER_SYS},
                       {"role": "user", "content": q}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": a},
            "extra_info": {"index": int(i), "score": float(r["score"]) if "score" in cols else -1.0},
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out_parquet)
    print(f"WROTE {args.out_parquet} rows={len(rows)} (from {len(df)} input)")

if __name__ == "__main__":
    main()
