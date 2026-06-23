"""Select REASONING-hard questions (model finishes but is often WRONG) from a hard
math dataset, via a vLLM endpoint. Unlike pass-band selection (which on GSM-hard
just picks length/truncation-hard questions), this selects on acc|formatted so the
training signal is real reasoning, not conciseness.

Filters to INTEGER answers (our reward is strict FINAL_ANSWER:<int>). For each
question: k samples at `budget`, compute format_rate (% parseable FINAL_ANSWER) and
acc|formatted (correct among formatted). Keep questions with format_rate >= min_fmt
AND acc_lo <= acc|formatted <= acc_hi  -> genuine reasoning-hard band.

Usage: python scripts/select_reasoning_hard.py <base_url> <model> <out_prefix> [n_pool] [k] [budget]
Writes <out_prefix>_train.jsonl/.json and <out_prefix>_heldout.json.
"""
import json, re, sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
BOXED = re.compile(r"\\boxed\{(-?\d+)\}")
P = ("Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
     "concisely.\nEnd your response with one final line that is exactly:\n"
     "FINAL_ANSWER: <integer>")


def int_answer(ex):
    """Return int ground-truth if this competition problem has an integer answer, else None."""
    a = ex.get("answer", "")
    if isinstance(a, (int,)):
        return int(a)
    s = str(a).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = BOXED.search(s)
    if m:
        return int(m.group(1))
    return None


def main():
    base_url, model, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    n_pool = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    budget = int(sys.argv[6]) if len(sys.argv) > 6 else 3072
    min_fmt, acc_lo, acc_hi = 0.5, 0.2, 0.8
    url = base_url.rstrip("/") + "/v1/chat/completions"

    ds = load_dataset("hiyouga/math12k", split="train")
    pool = []
    for ex in ds:
        gt = int_answer(ex)
        if gt is not None:
            pool.append({"question": ex["problem"], "answer": gt})
        if len(pool) >= n_pool:
            break
    print(f"integer-answer pool: {len(pool)} (of first scanned)", flush=True)

    def sample(q):
        body = {"model": model, "messages": [{"role": "user", "content": P.format(q=q)}],
                "temperature": 1.1, "top_p": 0.98, "max_tokens": budget}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        for _ in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
            except Exception:
                continue
        return ""

    def measure(it):
        with ThreadPoolExecutor(max_workers=k) as e:
            outs = list(e.map(lambda _: sample(it["question"]), range(k)))
        ms = [FA.findall(g) for g in outs]
        f = sum(1 for m in ms if m)
        c = sum(1 for m in ms if m and int(m[-1]) == it["answer"])
        it["format_rate"] = f / k
        it["acc_formatted"] = (c / f) if f else 0.0
        it["pass"] = c / k
        return it

    with ThreadPoolExecutor(max_workers=8) as e:
        scored = list(e.map(measure, pool))

    rh = [x for x in scored if x["format_rate"] >= min_fmt and acc_lo <= x["acc_formatted"] <= acc_hi]
    print(f"reasoning-hard (fmt>={min_fmt}, acc|fmt in [{acc_lo},{acc_hi}]): {len(rh)} / {len(scored)}", flush=True)
    json.dump(scored, open(out_prefix + "_scored.json", "w"))
    # split: every 5th to heldout
    train = [r for i, r in enumerate(rh) if i % 5 != 0]
    held = [r for i, r in enumerate(rh) if i % 5 == 0]
    json.dump(train, open(out_prefix + "_train.json", "w"))
    json.dump(held, open(out_prefix + "_heldout.json", "w"))
    with open(out_prefix + "_train.jsonl", "w") as fp:
        for r in train:
            fp.write(json.dumps({"question": r["question"], "reference_answer": str(r["answer"])}) + "\n")
    print(f"train={len(train)} heldout={len(held)} written to {out_prefix}_*", flush=True)


if __name__ == "__main__":
    main()
