"""HF-generate easy-Q pass measurement (no vLLM). Provably uses the given weights.

Usage: python scripts/hf_measure_easy.py <model_path_or_name> [k]
Measures pass@1 over the easy_gsmhard set: k samples/question, temp 1.1 top_p 0.98,
max_new 4096, strict last-FINAL_ANSWER integer match. Prints overall + per-Q.
"""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
PROMPT = (
    "Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
    "concisely.\nEnd your response with one final line that is exactly:\n"
    "FINAL_ANSWER: <integer>"
)


def main():
    model_path = sys.argv[1]
    questions_file = sys.argv[2] if len(sys.argv) > 2 else "outputs/easy_gsmhard.json"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    items = json.load(open(questions_file))
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    per_q = []
    for it in items:
        msgs = [{"role": "user", "content": PROMPT.format(q=it["question"])}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True, temperature=1.1,
                top_p=0.98, pad_token_id=tok.pad_token_id,
            )
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        hits = 0
        for g in gen:
            m = FA.findall(g)
            if m and int(m[-1]) == int(it["answer"]):
                hits += 1
        per_q.append(hits / k)
        print(f"  Q ans={it['answer']:<8} pass={hits}/{k}", flush=True)
    print("pass [%s | %s]: %.3f  per-Q=%s" % (model_path, questions_file, sum(per_q) / len(per_q), [round(x, 2) for x in per_q]), flush=True)


if __name__ == "__main__":
    main()
