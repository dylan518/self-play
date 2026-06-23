"""HF-generate eval reporting the format/accuracy decomposition (per CLAUDE.md:
never report pass alone). Prints format_rate, acc|formatted, pass + per-question
arrays so a paired test can be run. Used as the sweep-loop's gate metric.

Usage: python scripts/eval_accfmt.py <model_path> <questions.json> [k] [budget]
Emits a JSON line: {"model":..., "format_rate":..., "acc_formatted":..., "pass":...,
                    "per_q_pass":[...], "per_q_accf":[...], "k":..., "budget":...}
"""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
P = ("Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
     "concisely.\nEnd your response with one final line that is exactly:\n"
     "FINAL_ANSWER: <integer>")


def main():
    model_path = sys.argv[1]
    qfile = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    items = json.load(open(qfile))
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()

    per_q_pass, per_q_accf, per_q_fmt = [], [], []
    for it in items:
        ans = int(it.get("answer", it.get("reference_answer")))
        text = tok.apply_chat_template([{"role": "user", "content": P.format(q=it["question"])}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok([text] * k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=1.1,
                                 top_p=0.98, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ms = [FA.findall(g) for g in gen]
        f = sum(1 for m in ms if m)
        c = sum(1 for m in ms if m and int(m[-1]) == ans)
        per_q_fmt.append(f / k)
        per_q_pass.append(c / k)
        per_q_accf.append((c / f) if f else 0.0)
    n = len(items)
    res = {"model": model_path, "questions": qfile, "k": k, "budget": budget,
           "format_rate": round(sum(per_q_fmt) / n, 4),
           "acc_formatted": round(sum(per_q_accf) / n, 4),
           "pass": round(sum(per_q_pass) / n, 4),
           "per_q_pass": [round(x, 3) for x in per_q_pass],
           "per_q_accf": [round(x, 3) for x in per_q_accf]}
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
