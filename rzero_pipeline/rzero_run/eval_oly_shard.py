import os, json, argparse, vllm
from transformers import AutoTokenizer
from datasets import load_dataset
from mathruler.grader import extract_boxed_content, grade_answer
SYS = r"Please reason step by step, and put your final answer within \boxed{}."

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=4)
    ap.add_argument("--max_tokens",type=int,default=8192)
    a=ap.parse_args()
    ds=load_dataset("zwhe99/simplerl-OlympiadBench",split="test")
    rows=[ds[i] for i in range(len(ds)) if i % a.nshards == a.shard]
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    prompts=[tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":r["question"]}],
             tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    mml=max(4096,a.max_tokens+2048)
    m=vllm.LLM(model=a.model,enforce_eager=True,dtype="bfloat16",max_model_len=mml,
               gpu_memory_utilization=0.80,trust_remote_code=True)
    sp=vllm.SamplingParams(n=1,temperature=0.0,top_p=1.0,max_tokens=a.max_tokens)
    outs=m.generate(prompts,sp)
    with open(a.out,"w") as f:
        for r,o in zip(rows,outs):
            txt=o.outputs[0].text
            boxed=extract_boxed_content(txt)
            golds=[str(g).strip() for g in (r["final_answer"] if isinstance(r["final_answer"],list) else [r["final_answer"]])]
            fmt = boxed not in (None,"")
            def g(b,gg):
                try: return bool(grade_answer(str(b),str(gg)))
                except Exception: return False
            corr_first = fmt and g(boxed,golds[0])
            corr_any   = fmt and any(g(boxed,gg) for gg in golds)
            f.write(json.dumps({"id":r.get("id"),"q":r["question"][:400],"golds":golds,
                "is_multiple":bool(r.get("is_multiple_answer")),"answer_type":r.get("answer_type"),
                "boxed":boxed,"fmt":fmt,"corr_first":corr_first,"corr_any":corr_any,
                "resp_len":len(txt),"solution":txt[:3000]})+"\n")
    print(f"SHARD {a.shard} DONE n={len(rows)}")

if __name__=="__main__": main()
