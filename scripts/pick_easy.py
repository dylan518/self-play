"""Pick GSM-hard questions the model solves ~30-70% (reliable method + slips = sharpenable).
Saves easy set for the 'can RL sharpen anything' overfit test."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_self_play_grpo as OSP
from datasets import load_dataset
FA = OSP.FA
MT = 4096

def main():
    n_scan = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ds = load_dataset("reasoning-machines/gsm-hard", split="train").select(range(n_scan))
    def measure(ex):
        q = ex["input"]; ref = int(round(float(ex["target"])))
        with ThreadPoolExecutor(max_workers=8) as e:
            outs = list(e.map(lambda _: OSP.solver_rollout(q, "Qwen/Qwen3.5-9B", 1.1, 0.98, MT), range(8)))
        ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        p = sum(1 for a in ans if a is not None and int(a) == ref) / 8
        return {"question": q, "answer": ref, "pass": p, "structure": "gsm-hard"}
    with ThreadPoolExecutor(max_workers=6) as ex:
        scored = list(ex.map(measure, ds))
    mid = [x for x in scored if 0.25 <= x["pass"] <= 0.75]
    mid.sort(key=lambda x: abs(x["pass"] - 0.5))
    picked = mid[:want]
    print("scanned %d, %d in [0.25,0.75], picked %d:" % (len(scored), len(mid), len(picked)), flush=True)
    for x in picked:
        print("  base %.2f | ans=%s | %s" % (x["pass"], x["answer"], x["question"][:70]), flush=True)
    # pad to >= want+2 so harness heldout split works
    json.dump(picked, open("outputs/easy_gsmhard.json", "w"), indent=1)
    print("saved %d to outputs/easy_gsmhard.json" % len(picked), flush=True)

if __name__ == "__main__":
    main()
