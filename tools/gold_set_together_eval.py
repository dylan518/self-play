#!/usr/bin/env python3
"""Difficulty probe of the v3 gold set (43 Claude-written V(s)->bool problems)
against Qwen3.5 on Together, mirroring the R-Zero band measurement.

Modes per problem:
  direct — model sees V, must emit the answer string s in <answer>...</answer>
  code   — model writes a Python program that prints s; we execute it (subprocess,
           timeout) and feed stdout to V
n samples per (problem, mode) at temp 1.0 (same n=9 as the R-Zero band eval).
All API calls fan out at once (user ask: max workers 1000+; total is 43*2*9=774).
Verification runs locally in a small subprocess pool (V may eval/loop/call lean).

Usage:
  TOGETHER_API_KEY=... python3 tools/gold_set_together_eval.py [--smoke] [--n 9]
Outputs (Desktop):
  ~/Desktop/gold_set/together_eval_samples.jsonl   every sample
  ~/Desktop/gold_set/together_eval_report.md       per-problem table + band read
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, tempfile, threading, time, urllib.request

CATALOG = os.path.expanduser("~/Desktop/gold_set/catalog_canonical.json")
OUT_DIR = os.path.expanduser("~/Desktop/gold_set")
MODEL = os.environ.get("GOLD_MODEL", "Qwen/Qwen3.5-9B")
API = "https://api.together.xyz/v1/chat/completions"
KEY = os.environ["TOGETHER_API_KEY"]
MAX_TOK = int(os.environ.get("GOLD_MAX_TOK", 32000))  # reasoning model: thinking counts against this
PY = sys.executable

DIRECT_TMPL = """You are given a Python verifier function. There is no problem statement: the verifier IS the problem.

```python
{verifier}
```

Your task: produce a string s such that V(s) returns True.

Reason as needed, then give ONLY the exact answer string between <answer> and </answer> tags. The tags must wrap the literal string that will be passed to V — no quotes added, no commentary inside."""

CODE_TMPL = """You are given a Python verifier function. There is no problem statement: the verifier IS the problem.

```python
{verifier}
```

Your task: write a standalone Python program that COMPUTES a string s such that V(s) returns True, and prints exactly s to stdout (nothing else). Your program may search/construct/solve however you like, but must finish within ~25 seconds. Do not print anything except the answer string.

Give your final program in a single ```python fenced code block."""

VERIFY_RUNNER = r"""
import json, sys
spec = json.load(open(sys.argv[1]))
ns = {}
try:
    exec(spec["verifier"], ns)
    ok = bool(ns["V"](spec["s"]))
    print(json.dumps({"ok": ok}))
except Exception as e:
    print(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"[:300]}))
"""

_print_lock = threading.Lock()

def log(msg):
    with _print_lock:
        print(msg, flush=True)

def api_call(prompt, temperature, seed):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK, "temperature": temperature, "seed": seed,
    }).encode()
    for attempt in range(10):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                "User-Agent": "curl/8.7.1"})  # Cloudflare 1010-blocks the urllib UA
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            return ch["message"].get("content") or "", ch.get("finish_reason")
        except Exception as e:
            code = getattr(e, "code", None)
            if attempt == 9:
                return f"__API_ERROR__ {type(e).__name__} {code}: {e}"[:300], "error"
            retry_after = None
            try:
                retry_after = float(e.headers.get("Retry-After"))
            except Exception:
                pass
            time.sleep(retry_after or min(120, 2 ** attempt * (4 if code == 429 else 1)))

def extract_direct(text):
    i = text.rfind("<answer>")
    if i < 0: return None
    j = text.find("</answer>", i)
    if j < 0: return None
    return text[i + len("<answer>"):j].strip()

def extract_code(text):
    blocks, i = [], 0
    while True:
        i = text.find("```", i)
        if i < 0: break
        j = text.find("```", i + 3)
        if j < 0: break
        b = text[i + 3:j]
        if b.startswith("python"): b = b[len("python"):]
        blocks.append(b.strip("\n"))
        i = j + 3
    return blocks[-1] if blocks else None

def run_program(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([PY, path], capture_output=True, text=True, timeout=30)
        if p.returncode != 0:
            return None, f"exit {p.returncode}: {p.stderr.strip()[-200:]}"
        return p.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "program timeout 30s"
    finally:
        os.unlink(path)

def verify(verifier, s, uses_lean):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"verifier": verifier, "s": s}, f); spath = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(VERIFY_RUNNER); rpath = f.name
    try:
        p = subprocess.run([PY, rpath, spath], capture_output=True, text=True,
                           timeout=300 if uses_lean else 60)
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return False, f"verifier crash: {p.stderr.strip()[-200:]}"
        return d["ok"], d.get("err")
    except subprocess.TimeoutExpired:
        return False, "verifier timeout"
    finally:
        os.unlink(spath); os.unlink(rpath)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--retry", action="store_true",
                    help="re-run only samples that failed with __API_ERROR__ in the existing jsonl; merge")
    ap.add_argument("--workers", type=int, default=0, help="0 = all-at-once")
    args = ap.parse_args()

    probs = json.load(open(CATALOG))["problems"]
    if args.smoke:
        probs = probs[:2]; args.n = 2
    pmap = {p["id"]: p for p in probs}
    kept = []
    if args.retry:
        old = [json.loads(l) for l in open(os.path.join(OUT_DIR, "together_eval_samples.jsonl"))]
        kept = [r for r in old if not (r.get("err") or "").startswith("__API_ERROR__")]
        jobs = [(pmap[r["id"]], r["mode"], r["k"]) for r in old
                if (r.get("err") or "").startswith("__API_ERROR__")]
        log(f"[eval] retry mode: {len(kept)} kept, {len(jobs)} to re-run")
    else:
        jobs = [(p, mode, k) for p in probs for mode in ("direct", "code") for k in range(args.n)]
    log(f"[eval] {len(probs)} problems x 2 modes x n={args.n} = {len(jobs)} calls -> {MODEL}")

    verify_pool = cf.ThreadPoolExecutor(max_workers=6)
    done = [0]

    def run_job(job):
        p, mode, k = job
        prompt = (DIRECT_TMPL if mode == "direct" else CODE_TMPL).format(verifier=p["verifier_code"])
        text, finish = api_call(prompt, args.temp, seed=1000 * k + 7)
        rec = {"id": p["id"], "domain": p["domain"], "mode": mode, "k": k,
               "ok": False, "err": None, "s": None, "raw_len": len(text), "finish": finish}
        if text.startswith("__API_ERROR__"):
            rec["err"] = text
        else:
            if mode == "direct":
                s = extract_direct(text)
                if s is None: rec["err"] = "no <answer> tag"
            else:
                code = extract_code(text)
                if code is None:
                    s, rec["err"] = None, "no code block"
                else:
                    s, err = verify_pool.submit(run_program, code).result()
                    if err: rec["err"] = err
            if s is not None:
                rec["s"] = s[:2000]
                uses_lean = "subprocess" in p["verifier_code"]
                ok, verr = verify_pool.submit(verify, p["verifier_code"], s, uses_lean).result()
                rec["ok"], rec["err"] = ok, verr
        done[0] += 1
        if done[0] % 50 == 0: log(f"[eval] {done[0]}/{len(jobs)} done")
        return rec

    nw = args.workers or max(1000, len(jobs))
    with cf.ThreadPoolExecutor(max_workers=nw) as ex:
        recs = list(ex.map(run_job, jobs))
    verify_pool.shutdown()
    recs = kept + recs

    os.makedirs(OUT_DIR, exist_ok=True)
    samples_path = os.path.join(OUT_DIR, "together_eval_samples.jsonl")
    with open(samples_path, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")

    # ---- report ----
    by = {}
    for r in recs: by.setdefault(r["id"], {}).setdefault(r["mode"], []).append(r)
    rows = []
    for p in probs:
        d = by[p["id"]].get("direct", []); c = by[p["id"]].get("code", [])
        rows.append({"id": p["id"], "domain": p["domain"],
                     "direct": sum(r["ok"] for r in d), "code": sum(r["ok"] for r in c),
                     "nd": len(d), "nc": len(c)})
    n = args.n
    def band(k, nn): return "in-band" if nn and 0.3 <= k / nn <= 0.8 else ("too hard" if nn and k / nn < 0.3 else "too easy")
    lines = [f"# Gold set difficulty probe — {MODEL} via Together, n={n}/mode, temp={args.temp}",
             "", f"{len(probs)} problems. `direct` = emit answer string; `code` = write+run a program.",
             "", "| problem | domain | direct | code | best mode band (0.3-0.8) |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(max(r["direct"], r["code"]))):
        k = max(r["direct"], r["code"])
        lines.append(f"| {r['id']} | {r['domain']} | {r['direct']}/{r['nd']} | {r['code']}/{r['nc']} | {band(k, n)} |")
    trunc = sum(1 for r in recs if r.get("finish") == "length")
    lines.append("")
    lines.append(f"_Truncated at max_tokens={MAX_TOK}: {trunc}/{len(recs)} samples (these count as fails)._")
    solved_any = sum(1 for r in rows if r["direct"] + r["code"] > 0)
    inband = sum(1 for r in rows if band(max(r["direct"], r["code"]), n) == "in-band")
    hard = sum(1 for r in rows if max(r["direct"], r["code"]) == 0)
    easy = sum(1 for r in rows if band(max(r["direct"], r["code"]), n) == "too easy")
    lines += ["", f"**Summary:** solved-by-any-sample {solved_any}/{len(rows)} | in-band {inband} | "
              f"too hard (0 passes) {hard} | too easy (>0.8) {easy}",
              "", f"direct-mode total pass {sum(r['direct'] for r in rows)}/{sum(r['nd'] for r in rows)}; "
              f"code-mode total pass {sum(r['code'] for r in rows)}/{sum(r['nc'] for r in rows)}"]
    report_path = os.path.join(OUT_DIR, "together_eval_report.md")
    open(report_path, "w").write("\n".join(lines) + "\n")
    log(f"[eval] wrote {samples_path} and {report_path}")
    log("\n".join(lines[-4:]))

if __name__ == "__main__":
    main()
