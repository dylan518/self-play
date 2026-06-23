import re, glob, json, os
# parse challenger_batches.md batch headers for verifiability (mean_verified) + diversity (vendi)
# and compute difficulty = mean uncertainty = mean(min(s,1-s)) from per-question solver_score
files = sorted(glob.glob("/home/nvidia/rzero_run/artifacts/cont_it*_verified/challenger_batches.md"))
for f in files[-3:]:
    tag = f.split("/")[-2]
    txt = open(f).read()
    # only "real" batches (n>=100; small n are eval/probe batches)
    hdrs = re.findall(r"## batch @ \S+ \| n=(\d+) \| mean_verified=([\-0-9.]+).*?vendi ([0-9.]+)->([0-9.]+)", txt)
    # split into batch sections to compute difficulty per batch
    sections = txt.split("## batch @")[1:]
    print(f"\n=== {tag} ({len(sections)} batches) ===")
    print(f"{'batch':>5} {'n':>4} {'verifiability':>13} {'diversity(vendi)':>16} {'difficulty(unc)':>15}")
    bi = 0
    for sec in sections:
        m = re.search(r"n=(\d+) \| mean_verified=([\-0-9.]+).*?vendi ([0-9.]+)->([0-9.]+)", sec)
        if not m: continue
        n = int(m.group(1))
        if n < 100: continue   # skip tiny probe batches
        verif = float(m.group(2)); vendi = float(m.group(4))
        # per-question solver_score from table rows: | 0.90 | +1.00 | 3 | ...
        scores = [float(x) for x in re.findall(r"^\| ([0-9.]+) \|", sec, re.M)]
        diff = sum(min(s,1-s) for s in scores)/len(scores) if scores else float('nan')
        bi += 1
        print(f"{bi:>5} {n:>4} {verif:>13.3f} {vendi:>16.2f} {diff:>15.3f}")
