"""Scan a dataset, measure base pass@k under the STRICT HF-generate scorer, and
write per-question pass rates. Used to build a mid-band (2-6/8) training set that
actually carries GRPO gradient signal (all-correct and all-wrong groups give zero
advantage). Shardable: pass start:end to split across GPUs.

Usage: python scripts/select_midband.py <dataset> <start> <end> <out.jsonl> [k] [max_new]
  dataset: gsm8k | gsmhard
"""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-9B"
FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
PROMPT = (
    "Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
    "concisely.\nEnd your response with one final line that is exactly:\n"
    "FINAL_ANSWER: <integer>"
)


def load_pool(dataset):
    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        out = []
        for r in ds:
            m = re.findall(r"####\s*(-?[\d,]+)", r["answer"])
            if m:
                out.append((r["question"], int(m[-1].replace(",", ""))))
        return out
    else:  # gsmhard
        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        out = []
        for r in ds:
            try:
                ans = int(round(float(r["target"])))
            except Exception:
                continue
            out.append((r["input"], ans))
        return out


def main():
    dataset, start, end, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    max_new = int(sys.argv[6]) if len(sys.argv) > 6 else 2048
    pool = load_pool(dataset)[start:end]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    with open(out_path, "w") as f:
        for q, ans in pool:
            text = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False, add_generation_prompt=True
            )
            enc = tok([text] * k, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=True, temperature=1.1,
                    top_p=0.98, pad_token_id=tok.pad_token_id,
                )
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hits = sum(1 for g in outs if (FA.findall(g) or [None])[-1] is not None and int(FA.findall(g)[-1]) == ans)
            rec = {"question": q, "answer": ans, "pass": hits, "k": k}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  pass={hits}/{k}  ans={ans}", flush=True)


if __name__ == "__main__":
    main()
