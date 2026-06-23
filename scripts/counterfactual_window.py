"""Counterfactual: does the new pinned-golden + difficulty-weighted window change the
difficulty band of proposed questions, given the SAME (drifted) v3 history?

ARM A: raw last-80 window (what v3 actually used).
ARM B: select_window() difficulty-weighted + pinned-golden framing (the fix).
Both generate 8 questions; difficulty measured by k solver rollouts scored on modal-answer
consensus (fast proxy, identical for both arms -> comparison is valid).
"""
import json, os, sys, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_question_gen as AQG

FA = AQG.FA
SOLVER = AQG.SOLVER_TMPL
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")

def solve_band(questions, k=8, max_tok=8192):
    def one(q):
        with ThreadPoolExecutor(max_workers=k) as ex:
            outs = list(ex.map(lambda _: AQG.chat(SOLVER % q, 1.1, 0.98, max_tok, False), range(k)))
        ans = [(FA.findall(o) or [None])[-1] for o in outs]
        valid = [a for a in ans if a is not None]
        if not valid:
            return 0, ans
        mode = max(set(valid), key=valid.count)
        return sum(1 for a in ans if a == mode), ans  # consensus "pass" proxy
    res = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, questions))
    return res

def main():
    rows = [json.loads(l) for l in open("/tmp/run1_steps_full.jsonl") if l.strip()] if os.path.exists("/tmp/run1_steps_full.jsonl") \
        else [json.loads(l) for l in open("outputs/online_selfplay_run1/steps.jsonl") if l.strip()]
    g = [r for r in rows if r.get("pass_count") is not None]
    history = [(r["question"], r["pass_count"], "x", "structure: %s" % r.get("structure_label", "?")) for r in g]
    labels = [r.get("structure_label", "?") for r in g]
    AQG._all_labels = list(labels)
    focus = AQG.CHAIN_FOCUS[0]

    arms = {
        "A_raw_window": history[-80:],
        "B_weighted_window": AQG.select_window(history, n=60),
    }
    out = {}
    for name, window in arms.items():
        wp = [p for _, p, _, _ in window]
        print("[%s] window demo mix: mean %.2f, mid2-6 %.0f%%, 8/8 %.0f%%" % (
            name, sum(wp)/len(wp), 100*sum(1 for p in wp if 2<=p<=6)/len(wp), 100*sum(1 for p in wp if p==8)/len(wp)), flush=True)
        prompt = AQG.build_prompt(window, focus, labels)
        qs = []
        for attempt in range(4):
            gen = AQG.chat(prompt, 0.9, 0.95, 16000, True)
            qs = AQG.parse_questions(gen)[:8]
            if len(qs) >= 6: break
            print("[%s] attempt %d parsed %d (thinking truncation), retrying" % (name, attempt+1, len(qs)), flush=True)
        print("[%s] generated %d questions; solving..." % (name, len(qs)), flush=True)
        band = solve_band(qs)
        passes = [b for b, _ in band]
        mid = sum(1 for p in passes if 2 <= p <= 6)
        print("[%s] PROPOSED difficulty: passes=%s | mean %.2f | mid2-6 %d/%d (%.0f%%)" % (
            name, passes, sum(passes)/max(1,len(passes)), mid, len(passes), 100*mid/max(1,len(passes))), flush=True)
        out[name] = {"passes": passes, "questions": qs}
    json.dump(out, open("outputs/counterfactual_window.json", "w"), indent=1)
    print("\n=== SUMMARY (v3 actual late-run mid band was ~15%) ===", flush=True)
    for name, d in out.items():
        p = d["passes"]; mid = sum(1 for x in p if 2 <= x <= 6)
        print("  %-18s mean %.2f | mid %d/%d (%.0f%%)" % (name, sum(p)/max(1,len(p)), mid, len(p), 100*mid/max(1,len(p))), flush=True)

if __name__ == "__main__":
    main()
