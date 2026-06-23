import os, json, argparse, vllm, pandas as pd
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet(a.parquet)
    if a.max_rows: d = d.iloc[:a.max_rows]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts=[]; gts=[]; qs=[]; bands=[]
    for _,row in d.iterrows():
        msgs=[{"role":m["role"],"content":m["content"]} for m in row["prompt"]]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        gts.append(row["reward_model"]["ground_truth"]); qs.append(msgs[-1]["content"])
        try: bands.append(float(row["extra_info"].get("score")))
        except Exception: bands.append(None)

    llm = vllm.LLM(model=a.model, gpu_memory_utilization=0.75, max_model_len=int(os.getenv("MAXLEN","8192")), enforce_eager=True, trust_remote_code=True)
    sp = vllm.SamplingParams(n=a.n, temperature=1.0, top_p=0.99, max_tokens=int(os.getenv("MAXTOK","4096")))
    outs = llm.generate(prompts, sp)

    n_corr=n_tot=0
    with open(a.out,"w") as f:
        for q,o,gt,band in zip(qs,outs,gts,bands):
            sols=[s.text for s in o.outputs]
            ans=[extract_boxed_content(t) for t in sols]
            def ok(x):
                try: return gt is not None and grade_answer(str(x),str(gt))
                except Exception: return False
            corr=[bool(ok(x)) for x in ans]; n_corr+=sum(corr); n_tot+=len(corr)
            f.write(json.dumps({"question":q,"label":gt,"band_score":band,"solutions":sols,
                                "answers":[str(x) for x in ans],"correct":corr,
                                "solve_rate":sum(corr)/len(corr)})+"\n")
    print(f"DONE {a.out} rows={len(qs)} n={a.n} solve_rate={n_corr/max(1,n_tot):.3f}")

if __name__ == "__main__":
    main()
