#!/usr/bin/env python3
"""Compare solvers across math benchmarks with format/accuracy decomposition (CLAUDE.md).
Reports format_rate (fraction emitting parseable \\boxed{}), acc|formatted, pass (overall), at given k/max_tokens.
Benchmarks: gsm8k, math500, amc23, minerva, olympiad, aime24, aime25."""
import os, sys, argparse, json
import vllm
from mathruler.grader import extract_boxed_content, grade_answer
from datasets import load_dataset

SYS = r"Please reason step by step, and put your final answer within \boxed{}."

def load_bench(b):
    if b == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [x["problem"] for x in ds], [str(x["answer"]).strip() for x in ds]
    if b == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [x["question"] for x in ds], [x["answer"].split("####")[-1].strip().replace(",", "") for x in ds]
    if b == "amc23":
        ds = load_dataset("zwhe99/amc23", split="test")
        return [x["question"] for x in ds], [str(x["answer"]).strip() for x in ds]
    if b == "minerva":
        ds = load_dataset("zwhe99/simplerl-minerva-math", split="test")
        return [x["problem"] for x in ds], [str(x["answer"]).strip() for x in ds]
    if b == "olympiad":
        ds = load_dataset("zwhe99/simplerl-OlympiadBench", split="test")
        return [x["question"] for x in ds], [str(x["final_answer"][0]).strip() for x in ds]
    if b == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [x["problem"] for x in ds], [str(x["answer"]).strip() for x in ds]
    if b == "aime25":
        ds = load_dataset("yentinglin/aime_2025", "default")["train"]
        return [x["problem"] for x in ds], [str(x["answer"]).strip() for x in ds]
    raise ValueError(f"unknown benchmark {b}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", default="gsm8k")
    a = ap.parse_args()
    probs, golds = load_bench(a.benchmark)
    n = min(a.n, len(probs)); probs = probs[:n]; golds = golds[:n]
    mml = max(4096, a.max_tokens + 2048)
    m = vllm.LLM(model=a.model, enforce_eager=True, dtype="bfloat16", max_model_len=mml,
                 gpu_memory_utilization=0.85, tokenizer=a.model)
    tok = m.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in probs]
    sp = vllm.SamplingParams(n=a.k, temperature=(0.0 if a.k == 1 else 0.8), top_p=0.95, max_tokens=a.max_tokens)
    outs = m.generate(prompts, sp)
    total = len(probs); fmt = 0; correct = 0
    for o, gold in zip(outs, golds):
        boxed = [extract_boxed_content(s.text) for s in o.outputs]
        boxed = [b for b in boxed if b not in (None, "")]
        if boxed: fmt += 1
        if any(grade_answer(b, gold) for b in boxed): correct += 1
    res = {"tag": a.tag, "benchmark": a.benchmark, "model": a.model, "n": total, "k": a.k, "max_tokens": a.max_tokens,
           "format_rate": round(fmt/total, 4),
           "acc_given_formatted": round(correct/fmt, 4) if fmt else 0.0,
           "pass": round(correct/total, 4)}
    print("RESULT", json.dumps(res))
    with open(a.out, "a") as f: f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
