"""HF-generate version of reasoning-hard selection (no vLLM — loads the text stack
via AutoModelForCausalLM, which avoids the multimodal vision-tower init that hangs
vLLM serve on some setups). Reliable fallback.

Selects on acc|formatted (real reasoning signal), integer answers only.

Usage: python scripts/select_reasoning_hard_hf.py <out_prefix> [n_pool] [k] [budget]
"""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
BOXED = re.compile(r"\\boxed\{(-?\d+)\}")
MODEL = "Qwen/Qwen3.5-9B"
P = ("Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
     "concisely.\nEnd your response with one final line that is exactly:\n"
     "FINAL_ANSWER: <integer>")


def int_answer(ex):
    a = ex.get("answer", "")
    s = str(a).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = BOXED.search(s)
    return int(m.group(1)) if m else None


def main():
    out_prefix = sys.argv[1]
    n_pool = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    min_fmt, acc_lo, acc_hi = 0.5, 0.2, 0.8

    ds = load_dataset("hiyouga/math12k", split="train")
    pool = []
    for ex in ds:
        gt = int_answer(ex)
        if gt is not None:
            pool.append({"question": ex["problem"], "answer": gt})
        if len(pool) >= n_pool:
            break
    print(f"integer-answer pool: {len(pool)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    scored = []
    for i, it in enumerate(pool):
        text = tok.apply_chat_template([{"role": "user", "content": P.format(q=it["question"])}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok([text] * k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=1.1,
                                 top_p=0.98, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ms = [FA.findall(g) for g in gen]
        f = sum(1 for m in ms if m)
        c = sum(1 for m in ms if m and int(m[-1]) == it["answer"])
        it["format_rate"] = f / k
        it["acc_formatted"] = (c / f) if f else 0.0
        it["pass"] = c / k
        scored.append(it)
        if i % 20 == 0:
            print(f"  {i}/{len(pool)} fmt={it['format_rate']:.2f} accf={it['acc_formatted']:.2f}", flush=True)

    rh = [x for x in scored if x["format_rate"] >= min_fmt and acc_lo <= x["acc_formatted"] <= acc_hi]
    import statistics
    accf_all = [x["acc_formatted"] for x in scored if x["format_rate"] >= min_fmt]
    print(f"mean fmt={statistics.mean([x['format_rate'] for x in scored]):.3f} "
          f"mean acc|fmt(finishers)={statistics.mean(accf_all) if accf_all else 0:.3f}", flush=True)
    print(f"reasoning-hard (fmt>={min_fmt}, acc|fmt in [{acc_lo},{acc_hi}]): {len(rh)} / {len(scored)}", flush=True)
    json.dump(scored, open(out_prefix + "_scored.json", "w"))
    train = [r for i, r in enumerate(rh) if i % 5 != 0]
    held = [r for i, r in enumerate(rh) if i % 5 == 0]
    json.dump(train, open(out_prefix + "_train.json", "w"))
    json.dump(held, open(out_prefix + "_heldout.json", "w"))
    with open(out_prefix + "_train.jsonl", "w") as fp:
        for r in train:
            fp.write(json.dumps({"question": r["question"], "reference_answer": str(r["answer"])}) + "\n")
    print(f"train={len(train)} heldout={len(held)} -> {out_prefix}_*", flush=True)


if __name__ == "__main__":
    main()
