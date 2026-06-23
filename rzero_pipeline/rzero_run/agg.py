import json, glob, sys, statistics
tag = sys.argv[1]
rows = []
for f in glob.glob(f"/data/selfplay/oly_{tag}_shard*.jsonl"):
    rows += [json.loads(l) for l in open(f)]
n = len(rows)
fmt = sum(1 for r in rows if r["fmt"])
cf = sum(1 for r in rows if r["corr_first"])
mr = statistics.mean(r["resp_len"] for r in rows) if rows else 0
trunc = sum(1 for r in rows if r["resp_len"] >= 31000)
af = cf / fmt if fmt else 0.0
print(f"RESULT_{tag} n={n} format_rate={fmt/n:.4f} acc_given_fmt={af:.4f} pass={cf/n:.4f} mean_resp_chars={mr:.0f} truncated~={trunc}")
