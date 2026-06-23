"""Paired significance test on held-out acc|formatted (the reasoning metric).
Reads two RESULT json files (from eval_accfmt.py), compares per-question acc|formatted
(base vs trained) with a paired t-test. Prints SIGNIFICANT_GAIN / NO_GAIN + stats.

Usage: python scripts/paired_test.py <base_result.json> <trained_result.json>
Exit code 0 = significant positive gain, 1 = not.
"""
import json, sys, statistics, math


def load(p):
    for line in open(p):
        line = line.strip()
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no RESULT in " + p)


def main():
    base = load(sys.argv[1])
    trained = load(sys.argv[2])
    b = base["per_q_accf"]
    t = trained["per_q_accf"]
    n = min(len(b), len(t))
    d = [t[i] - b[i] for i in range(n)]
    m = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 1e-9
    tval = m / se
    up = sum(1 for x in d if x > 0.01)
    dn = sum(1 for x in d if x < -0.01)
    # df=n-1; ~p<0.05 two-sided for n>=8 needs |t|>~2.3
    crit = 2.3
    sig = (m > 0) and (tval > crit)
    print(f"base acc|fmt={base['acc_formatted']:.3f} (fmt {base.get('format_rate','?')}) | "
          f"trained acc|fmt={trained['acc_formatted']:.3f} (fmt {trained.get('format_rate','?')})")
    print(f"paired Δacc|fmt={m:+.4f}  t={tval:.2f} (df={n-1})  {up} up / {dn} down  -> "
          f"{'SIGNIFICANT_GAIN' if sig else 'NO_GAIN'}")
    sys.exit(0 if sig else 1)


if __name__ == "__main__":
    main()
