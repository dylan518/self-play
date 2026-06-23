"""Measure pass rate on the easy_gsmhard questions for a given vLLM-served model."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_self_play_grpo as OSP
FA = OSP.FA

def main():
    model = sys.argv[1]
    items = json.load(open("outputs/easy_gsmhard.json"))
    def one(x):
        with ThreadPoolExecutor(max_workers=8) as e:
            o = list(e.map(lambda _: OSP.solver_rollout(x["question"], model, 1.1, 0.98, 4096), range(8)))
        a = [(FA.findall(t) or [None])[-1] for t, _ in o]
        return sum(1 for v in a if v is not None and int(v) == x["answer"]) / 8
    with ThreadPoolExecutor(max_workers=10) as e:
        r = list(e.map(one, items))
    print("easy-Q pass [%s]: %.3f  per-Q=%s" % (model, sum(r) / len(r), [round(x, 2) for x in r]), flush=True)

if __name__ == "__main__":
    main()
