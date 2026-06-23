"""vLLM IN-PROCESS (offline LLM) scorer — avoids the `vllm serve` HTTP server
(fastapi/starlette bug on WashU) and is faster (one big batched generate).

Two modes:
  scan  : load math12k, filter integer answers, score format_rate/acc|formatted,
          select the reasoning-hard band (acc|formatted in [lo,hi], fmt>=min_fmt),
          write <out>_train.jsonl/.json + <out>_heldout.json + <out>_scored.json.
  eval  : load a questions json, score, print RESULT json (format_rate, acc_formatted,
          pass, per-question arrays) for paired tests.

Usage:
  python scripts/vllm_score.py scan <out_prefix> <n_pool> <k> <budget>
  python scripts/vllm_score.py eval <model_path> <questions.json> <k> <budget> [out.json]
First positional after mode that looks like a path is the model for eval; scan uses base.
"""
import json, re, sys, statistics
from vllm import LLM, SamplingParams

FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")
BOX = re.compile(r"\\boxed\{(-?\d+)\}")
PROMPT = ("Question:\n{q}\n\nSolve the problem step by step, showing your reasoning "
          "concisely.\nEnd your response with one final line that is exactly:\n"
          "FINAL_ANSWER: <integer>")


def int_answer(a):
    s = str(a).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = BOX.search(s)
    return int(m.group(1)) if m else None


def score(llm, tok_chat, items, k, budget):
    """Return per-question (format_rate, acc_formatted, pass). One batched generate."""
    prompts = []
    for it in items:
        msg = [{"role": "user", "content": PROMPT.format(q=it["question"])}]
        prompts.append(tok_chat(msg, tokenize=False, add_generation_prompt=True))
    sp = SamplingParams(n=k, temperature=1.1, top_p=0.98, max_tokens=budget)
    outs = llm.generate(prompts, sp)
    res = []
    for it, o in zip(items, outs):
        gens = [c.text for c in o.outputs]
        ms = [FA.findall(g) for g in gens]
        f = sum(1 for m in ms if m)
        c = sum(1 for m in ms if m and int(m[-1]) == int(it["answer"]))
        res.append((f / k, (c / f) if f else 0.0, c / k))
    return res


def main():
    mode = sys.argv[1]
    model = "Qwen/Qwen3.5-9B"
    if mode == "eval":
        model = sys.argv[2]
    llm = LLM(model=model, gpu_memory_utilization=0.9, max_model_len=8192,
              dtype="bfloat16", enforce_eager=False)
    tok = llm.get_tokenizer()
    chat = lambda msg, **kw: tok.apply_chat_template(msg, **kw)

    if mode == "scan":
        out_prefix, n_pool, k, budget = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        from datasets import load_dataset
        ds = load_dataset("hiyouga/math12k", split="train")
        pool = []
        for ex in ds:
            gt = int_answer(ex.get("answer", ""))
            if gt is not None:
                pool.append({"question": ex["problem"], "answer": gt})
            if len(pool) >= n_pool:
                break
        print(f"integer-answer pool: {len(pool)}", flush=True)
        r = score(llm, chat, pool, k, budget)
        for it, (fm, af, ps) in zip(pool, r):
            it.update(format_rate=fm, acc_formatted=af, **{"pass": ps})
        min_fmt, lo, hi = 0.5, 0.2, 0.8
        rh = [it for it in pool if it["format_rate"] >= min_fmt and lo <= it["acc_formatted"] <= hi]
        accf = [it["acc_formatted"] for it in pool if it["format_rate"] >= min_fmt]
        print(f"mean fmt={statistics.mean([it['format_rate'] for it in pool]):.3f} "
              f"mean acc|fmt(finishers)={statistics.mean(accf) if accf else 0:.3f}", flush=True)
        print(f"REASONING-HARD (fmt>={min_fmt}, acc|fmt in [{lo},{hi}]): {len(rh)} / {len(pool)}", flush=True)
        json.dump(pool, open(out_prefix + "_scored.json", "w"))
        train = [r for i, r in enumerate(rh) if i % 5 != 0]
        held = [r for i, r in enumerate(rh) if i % 5 == 0]
        json.dump(train, open(out_prefix + "_train.json", "w"))
        json.dump(held, open(out_prefix + "_heldout.json", "w"))
        with open(out_prefix + "_train.jsonl", "w") as fp:
            for it in train:
                fp.write(json.dumps({"question": it["question"], "reference_answer": str(it["answer"])}) + "\n")
        print(f"train={len(train)} heldout={len(held)} -> {out_prefix}_*", flush=True)
    elif mode == "eval":
        qfile, k, budget = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        outp = sys.argv[6] if len(sys.argv) > 6 else None
        items = json.load(open(qfile))
        for it in items:
            it["answer"] = int(it.get("answer", it.get("reference_answer")))
        r = score(llm, chat, items, k, budget)
        n = len(r)
        res = {"model": model, "questions": qfile, "k": k, "budget": budget,
               "format_rate": round(sum(x[0] for x in r) / n, 4),
               "acc_formatted": round(sum(x[1] for x in r) / n, 4),
               "pass": round(sum(x[2] for x in r) / n, 4),
               "per_q_accf": [round(x[1], 3) for x in r],
               "per_q_pass": [round(x[2], 3) for x in r]}
        line = "RESULT " + json.dumps(res)
        print(line, flush=True)
        if outp:
            open(outp, "w").write(line + "\n")


if __name__ == "__main__":
    main()
