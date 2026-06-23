"""Fast vLLM-backed pass@1 measurement (parallel HTTP). High-n capable.

Usage: python scripts/vllm_measure.py <base_url> <model> <questions.json> [k] [max_tokens]
Prints overall mean pass (k samples/question, strict last-FINAL_ANSWER int) + per-Q + 95% CI.
"""
import json, re, sys, math
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
PROMPT = (
    "Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
    "concisely.\nEnd your response with one final line that is exactly:\n"
    "FINAL_ANSWER: <integer>"
)


def main():
    base_url, model, qfile = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    items = json.load(open(qfile))
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def one_sample(question):
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(q=question)}],
                "temperature": 1.1, "top_p": 0.98, "max_tokens": max_tokens}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        for _ in range(4):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=600))
                return d["choices"][0]["message"]["content"]
            except Exception:
                continue
        return ""

    def measure_q(it):
        q, ans = it["question"], int(it["answer"])
        with ThreadPoolExecutor(max_workers=k) as e:
            outs = list(e.map(lambda _: one_sample(q), range(k)))
        hits = sum(1 for g in outs if (FA.findall(g) or [None])[-1] is not None and int(FA.findall(g)[-1]) == ans)
        return hits / k

    with ThreadPoolExecutor(max_workers=8) as e:
        per_q = list(e.map(measure_q, items))
    n = len(per_q) * k
    mean = sum(per_q) / len(per_q)
    se = math.sqrt(mean * (1 - mean) / n)
    print("pass [%s | %s | k=%d n=%d]: %.4f  95%%CI=[%.4f,%.4f]  per-Q=%s" % (
        model, qfile, k, n, mean, mean - 1.96 * se, mean + 1.96 * se, [round(x, 2) for x in per_q]), flush=True)


if __name__ == "__main__":
    main()
