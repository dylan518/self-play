"""GRPO learning diagnostics. Answers: is the update mechanism BROKEN or just WEAK?

Tier 1 (offline, instant): tokenization round-trip + logprob sanity on real rollouts.
Tier 2 (overfit): freeze ONE question's k graded completions, run N optimizer steps,
track mean-token-logprob of CORRECT vs WRONG completions. Correct should rise above
wrong if the gradient is doing the intended thing. Compare configs side by side.

Generation uses an existing vLLM (default port 8002). Trainer on this process's GPU.
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentic_question_gen as AQG  # noqa: E402
import online_self_play_grpo as OSP  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from grpo_math.models.policy import sequence_logprobs  # noqa: E402

FA = AQG.FA
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")


def per_token_logp(policy, tokenizer, device, question, comps):
    ids, attn, pe = OSP.build_training_batch(tokenizer, question, comps)
    ids, attn, pe = ids.to(device), attn.to(device), pe.to(device)
    out = []
    with torch.no_grad():
        for i in range(len(comps)):
            sl, gm = sequence_logprobs(policy, ids[i:i+1], attn[i:i+1], pe[i:i+1])
            out.append(float(sl) / max(1, int(gm.sum())))
    return out


def tier1(policy, tokenizer, device, question, comps, answers, ref):
    print("\n=== TIER 1: tokenization round-trip + logprob sanity ===", flush=True)
    msgs = [{"role": "user", "content": OSP.SOLVER_TMPL % question}]
    bad = 0
    for c in comps:
        ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        rt = tokenizer.decode(ids, skip_special_tokens=True)
        if rt.strip() != c.strip():
            bad += 1
    print("round-trip mismatches: %d/%d (text != retokenize->decode)" % (bad, len(comps)), flush=True)
    lps = per_token_logp(policy, tokenizer, device, question, comps)
    print("mean per-token logprob: %.3f (sane ~ -0.3..-2.5; ~-12 = uniform/garbage)" % (sum(lps)/len(lps)), flush=True)
    print("  per completion:", ["%.2f" % x for x in lps], flush=True)


def overfit(question, comps, rewards, answers, ref, device, lr, full_weight, steps, kl_beta=0.0):
    tag = "%s lr=%.0e" % ("FULL" if full_weight else "LoRA-r8", lr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    if full_weight:
        policy = base.to(device)
        for p in policy.parameters():
            p.requires_grad_(True)
    else:
        policy = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules="all-linear", task_type="CAUSAL_LM")).to(device)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)

    corr = [i for i, a in enumerate(answers) if a is not None and ref is not None and int(a) == ref]
    wrong = [i for i in range(len(comps)) if i not in corr]
    lp0 = per_token_logp(policy, tokenizer, device, question, comps)
    base_corr = sum(lp0[i] for i in corr)/max(1, len(corr))
    base_wrong = sum(lp0[i] for i in wrong)/max(1, len(wrong))
    print("\n=== TIER 2 OVERFIT [%s] | %d correct, %d wrong | start: corr %.3f wrong %.3f gap %.3f ==="
          % (tag, len(corr), len(wrong), base_corr, base_wrong, base_corr - base_wrong), flush=True)
    if not corr or not wrong:
        print("  (need both correct and wrong completions for a signal; skipping)", flush=True); return

    for step in range(1, steps + 1):
        m = OSP.grpo_step(policy, opt, tokenizer, device, [(question, comps, rewards)], kl_beta, microbatch=1)
        if step % 10 == 0 or step == 1:
            lp = per_token_logp(policy, tokenizer, device, question, comps)
            mc = sum(lp[i] for i in corr)/len(corr)
            mw = sum(lp[i] for i in wrong)/len(wrong)
            print("  step %3d | corr %.3f wrong %.3f gap %+.3f (start %+.3f) | grad_norm? loss %.4f"
                  % (step, mc, mw, mc - mw, base_corr - base_wrong, m["loss"]), flush=True)
    del policy, base
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="outputs/online_selfplay_run1")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--gen_model", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()
    device = torch.device("cuda")

    # pick an early question that had BOTH correct and wrong rollouts (mid pass rate)
    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "steps.jsonl")) if l.strip()]
    cand = [r for r in rows if r.get("pass_count") not in (None, 0, 8)]
    cand.sort(key=lambda r: abs((r.get("pass_count") or 0) - 4))
    r = cand[0]
    q, ref = r["question"], None
    for a, v in (r.get("artifact_verdicts") or {}).items():
        if v.get("verdict"):
            ref = int(a); break
    print("question (step %d, was %d/8): %s" % (r["step"], r["pass_count"], q[:120]), flush=True)
    print("reference answer:", ref, flush=True)

    # fresh rollouts from base gen model
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(lambda _: OSP.solver_rollout(q, args.gen_model, 1.1, 0.98, args.max_new_tokens), range(8)))
    comps = [t for t, _ in outs]
    answers = [(FA.findall(c) or [None])[-1] for c in comps]
    rewards = [1.0 if (a is not None and ref is not None and int(a) == ref) else 0.0 for a in answers]
    print("rollout answers:", answers, "| rewards:", rewards, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    b = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    tier1(b, tok, device, q, comps, answers, ref)
    del b
    torch.cuda.empty_cache()

    # Tier 2: compare configs. If correct-vs-wrong gap doesn't open at lr1e-5/LoRA
    # but does at lr1e-4 or full-weight, the original recipe was just too weak.
    overfit(q, comps, rewards, answers, ref, device, lr=1e-5, full_weight=False, steps=args.steps)
    overfit(q, comps, rewards, answers, ref, device, lr=1e-4, full_weight=False, steps=args.steps)
    overfit(q, comps, rewards, answers, ref, device, lr=1e-5, full_weight=True, steps=args.steps)


if __name__ == "__main__":
    main()
