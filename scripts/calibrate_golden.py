"""Quick-calibrate candidate GOLDEN questions: solve each k times on the base model,
score by tool-judge consensus, report pass rate. Keep the 2-6/8 ones for GOLDEN."""
import json, os, sys, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_question_gen as AQG
import python_tool_judge as PTJ
FA = AQG.FA

def main():
    cands = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
        "/tmp/diverse_golden_candidates.json"))
    print("calibrating %d candidates (base model, k=8)\n" % len(cands), flush=True)
    def measure(q):
        with ThreadPoolExecutor(max_workers=8) as ex:
            comps = list(ex.map(lambda _: AQG.chat(AQG.SOLVER_TMPL % q, 1.1, 0.98, 6000, False), range(8)))
        ans = [(FA.findall(c) or [None])[-1] for c in comps]
        # reference via tool judge on modal answer
        valid = [a for a in ans if a is not None]
        if not valid:
            return q, 0, None
        ref, _, _ = PTJ.judge_reference(q)
        if ref is None:
            mode = max(set(valid), key=valid.count); ref = int(mode)
        passes = sum(1 for a in ans if a is not None and int(a) == ref)
        return q, passes, ref
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for q, p, ref in ex.map(measure, cands):
            band = "MID-KEEP" if 2 <= p <= 6 else ("too-easy" if p > 6 else "too-hard")
            print("[%s] %d/8 ans=%s | %s" % (band, p, ref, q[:80]), flush=True)
            out.append({"question": q, "pass": p, "ref": ref})
    keep = [o for o in out if 2 <= o["pass"] <= 6]
    print("\n=== KEEP (2-6/8): %d ===" % len(keep), flush=True)
    json.dump(out, open("/tmp/golden_calibrated.json", "w"), indent=1)

if __name__ == "__main__":
    main()
