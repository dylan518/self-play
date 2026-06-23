"""External-oracle verifiability check: have gpt-5.4-nano independently solve the run's
questions and compare its answer to our tool-judge-certified answer.

Agreement = our certified answer is corroborated by an independent strong model.
Disagreements flag questions where our judge may be wrong (or the question ambiguous).
"""
import argparse, json, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ.get("ORACLE_MODEL", "gpt-5.4-nano")
FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
PROMPT = ("Solve this problem by careful reasoning (you may reason step by step). "
          "It has a single integer answer.\n\n%s\n\nEnd with exactly: FINAL_ANSWER: <integer>")

def nano(q):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT % q}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    for attempt in range(4):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            txt = r["choices"][0]["message"]["content"] or ""
            m = FA.findall(txt)
            return m[-1] if m else None
        except Exception:
            import time; time.sleep(5 * (attempt + 1))
    return None

def certified_answer(r):
    for a, v in (r.get("artifact_verdicts") or {}).items():
        if v.get("verdict"):
            return a
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_jsonl", default="outputs/online_selfplay_run1/steps.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="outputs/online_selfplay_run1/oracle_nano.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.steps_jsonl) if l.strip()]
    cand = [r for r in rows if certified_answer(r) is not None]
    # stratified sample across pass rates
    cand.sort(key=lambda r: (r.get("pass_count", 0), r["step"]))
    sample = cand[:: max(1, len(cand) // args.n)][:args.n]
    print("oracle=%s | comparing %d certified questions" % (MODEL, len(sample)), flush=True)

    def one(r):
        return r, certified_answer(r), nano(r["question"])
    agree = disagree = nano_none = 0
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, ours, theirs in ex.map(one, sample):
            if theirs is None: nano_none += 1; tag = "NANO_NO_ANS"
            elif str(theirs) == str(ours): agree += 1; tag = "AGREE"
            else: disagree += 1; tag = "DISAGREE"
            results.append({"step": r["step"], "pass": r.get("pass_count"), "ours": ours, "nano": theirs, "tag": tag, "q": r["question"]})
            print("[%s] ours=%s nano=%s pass=%s | %s" % (tag, ours, theirs, r.get("pass_count"), r["question"][:80]), flush=True)
    n_cmp = agree + disagree
    print("\n==== ORACLE AGREEMENT: %d/%d (%.0f%%) | disagree %d | nano no-answer %d ===="
          % (agree, n_cmp, 100 * agree / max(1, n_cmp), disagree, nano_none), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out, flush=True)

if __name__ == "__main__":
    main()
