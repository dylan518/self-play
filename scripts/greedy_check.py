"""Rule out temp-1.1 masking: eval base vs adapter at LOW temp on the golden questions.
If the adapter beats base greedily, learning was happening but masked by high-temp sampling."""
import json, os, sys, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_self_play_grpo as OSP
FA = OSP.FA
URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
SOLVER = OSP.SOLVER_TMPL

def chat_temp(q, model, temp, mt=6000):
    body = {"model": model, "messages": [{"role": "user", "content": SOLVER % q}],
            "temperature": temp, "top_p": 0.95, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception:
            import time; time.sleep(5)
    return ""

def rate(qs, refs, model, temp, k):
    def one(qr):
        q, ref = qr
        with ThreadPoolExecutor(max_workers=k) as ex:
            outs = list(ex.map(lambda _: chat_temp(q, model, temp), range(k)))
        ans = [(FA.findall(t) or [None])[-1] for t in outs]
        return sum(1 for a in ans if a is not None and int(a) == ref) / k
    with ThreadPoolExecutor(max_workers=12) as ex:
        per = list(ex.map(one, zip(qs, refs)))
    return per, sum(per) / len(per)

def main():
    adapter = sys.argv[1]
    items = json.load(open("outputs/golden8.json"))
    qs = [x["question"] for x in items]; refs = [x["answer"] for x in items]
    OSP.http_json("/v1/load_lora_adapter", {"lora_name": "ck", "lora_path": os.path.abspath(adapter)}, timeout=120)
    for temp, lbl in [(0.2, "near-greedy"), (1.1, "train-temp")]:
        bp, bm = rate(qs, refs, "Qwen/Qwen3.5-9B", temp, 4)
        ap, am = rate(qs, refs, "ck", temp, 4)
        print("[%s temp %.1f] BASE mean %.3f -> ADAPTER mean %.3f (delta %+.3f)" % (lbl, temp, bm, am, am - bm), flush=True)
        print("   per-Q base:    %s" % [round(x, 2) for x in bp], flush=True)
        print("   per-Q adapter: %s" % [round(x, 2) for x in ap], flush=True)

if __name__ == "__main__":
    main()
