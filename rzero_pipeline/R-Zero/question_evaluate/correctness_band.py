import json, argparse, pandas as pd
from mathruler.grader import grade_answer
ap = argparse.ArgumentParser()
ap.add_argument("--verified_parquet", required=True)
ap.add_argument("--all_questions", required=True)
ap.add_argument("--out_parquet", required=True)
ap.add_argument("--lo", type=float, default=0.2)
ap.add_argument("--hi", type=float, default=0.8)
a = ap.parse_args()
qmap = {}
for l in open(a.all_questions):
    try:
        r = json.loads(l); qmap[str(r.get("question","")).strip()] = r.get("results", [])
    except Exception: pass
d = pd.read_parquet(a.verified_parquet)
keep, matched = [], 0
for _, row in d.iterrows():
    label = str(row["answer"]); res = qmap.get(str(row["problem"]).strip(), [])
    if res:
        matched += 1
        sr = sum(1 for x in res if grade_answer(str(x), label)) / len(res)
        keep.append(a.lo < sr < a.hi)
    else:
        keep.append(False)
out = d[pd.Series(keep, index=d.index)].reset_index(drop=True)
out.to_parquet(a.out_parquet)
print(f"CVBAND: in={len(d)} matched_to_dist={matched} kept={len(out)} (solve_rate in ({a.lo},{a.hi})) -> {a.out_parquet}")
