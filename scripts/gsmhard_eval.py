"""Eval a vLLM-served model on GSM-hard (harder, integer answers, less contaminated than GSM8K).
Usage: gsmhard_eval.py <model_name> <n> [k]"""
import sys, os, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_self_play_grpo as OSP
from datasets import load_dataset
FA = OSP.FA

def main():
    model = sys.argv[1]; n = int(sys.argv[2]); k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ds = load_dataset("reasoning-machines/gsm-hard", split="train").select(range(n))
    def one(ex):
        gold = int(round(float(ex["target"])))
        with ThreadPoolExecutor(max_workers=k) as e:
            outs = list(e.map(lambda _: OSP.solver_rollout(ex["input"], model, 1.1, 0.98, 8192), range(k)))
        ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        p1 = 1.0 if (ans[0] is not None and int(ans[0]) == gold) else 0.0
        pk = 1.0 if any(a is not None and int(a) == gold for a in ans) else 0.0
        return p1, pk
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, ds))
    print("GSM-HARD [%s] n=%d k=%d: pass@1 %.3f | pass@%d %.3f" % (
        model, n, k, sum(p for p, _ in res)/n, k, sum(pk for _, pk in res)/n), flush=True)

if __name__ == "__main__":
    main()
