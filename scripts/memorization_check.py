"""Did training improve the model on the questions it TRAINED on (memorization)?
Compares base vs final-adapter pass rate on the train split. If trained-Q pass rises
but held-out didn't -> training works, doesn't generalize in few steps. If trained-Q
also flat -> training genuinely not moving the policy."""
import json, os, sys, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_self_play_grpo as OSP
FA = OSP.FA

def rate(qs, refs, model, k, mt):
    def one(qr):
        q, r = qr
        with ThreadPoolExecutor(max_workers=k) as ex:
            outs = list(ex.map(lambda _: OSP.solver_rollout(q, model, 1.1, 0.98, mt), range(k)))
        ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        return sum(1 for a in ans if a is not None and int(a) == r) / k
    with ThreadPoolExecutor(max_workers=16) as ex:
        return sum(ex.map(one, zip(qs, refs))) / len(qs)

def main():
    adapter = sys.argv[1]; tag = sys.argv[2]
    items = json.load(open("outputs/fixed_train_questions.json"))
    random.Random(0).shuffle(items)
    train = items[14:]  # same split as harness (heldout=14)
    q = [x["question"] for x in train]; r = [x["answer"] for x in train]
    base = rate(q, r, "Qwen/Qwen3.5-9B", 6, 8192)
    name = "memchk_%s" % tag
    OSP.http_json("/v1/load_lora_adapter", {"lora_name": name, "lora_path": os.path.abspath(adapter)}, timeout=120)
    aft = rate(q, r, name, 6, 8192)
    print("[%s] TRAINED-Q pass: base %.3f -> trained %.3f (delta %+.3f) over %d train questions" % (
        tag, base, aft, aft - base, len(train)), flush=True)

if __name__ == "__main__":
    main()
