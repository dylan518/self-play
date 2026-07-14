#!/usr/bin/env python3
"""Replay-GRPO solver update from harvested Stage-A rollouts. ZERO new rollouts.

Reads solver_harvest.jsonl (written by the patched vLLM service: per question, the n
solution texts + extracted answers + K-program verified_answer/votes). Builds GRPO
groups from VERIFIED questions (votes >= MIN_AGREE), reward = solution answer matches
the program-consensus label. Only groups with 0 < hits < n carry gradient (live).

Update: one full-batch optimizer step via gradient accumulation over sequence
micro-batches — batch size can be ANY integer (110 is fine), nothing is dropped.
A_i = r_i - mean(r) within group; loss = -sum_i A_i * mean_token_logprob_i / N_seqs.

Usage:
  python solver_replay_grpo.py --harvest a.jsonl [b.jsonl ...] --model <hf path/name>
      --out <ckpt dir> [--min_live 96] [--lr 1e-6] [--max_sol_tokens 3072] [--micro 4]
Exit codes: 0 = stepped + saved; 3 = not enough live groups (keep accumulating).
"""
import argparse, json, math, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MIN_AGREE = int(os.getenv("VERIFY_MIN_AGREE", "2"))
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


def load_groups(paths, tok, max_sol_tokens):
    groups, skipped_long = [], 0
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            va = r.get("verified_answer")
            if va is None or (r.get("votes") or 0) < MIN_AGREE:
                continue
            texts = r.get("sol_texts") or []
            answs = r.get("sol_answers") or []
            if not texts or len(texts) != len(answs):
                continue
            rewards = [1.0 if (a is not None and str(a).strip() == str(va).strip()) else 0.0
                       for a in answs]
            if not (0 < sum(rewards) < len(rewards)):
                continue  # zero-variance group: no gradient
            kept_t, kept_r = [], []
            for t, rw in zip(texts, rewards):
                if len(tok(t, add_special_tokens=False).input_ids) <= max_sol_tokens:
                    kept_t.append(t); kept_r.append(rw)
                else:
                    skipped_long += 1
            if len(kept_t) >= 2 and 0 < sum(kept_r) < len(kept_r):
                groups.append({"question": r["question"], "texts": kept_t, "rewards": kept_r})
    return groups, skipped_long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_live", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--max_sol_tokens", type=int, default=3072)
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    groups, skipped_long = load_groups(args.harvest, tok, args.max_sol_tokens)
    n_seq = sum(len(g["texts"]) for g in groups)
    print(f"[replay] live groups {len(groups)} (threshold {args.min_live}) | sequences {n_seq} "
          f"| skipped-too-long {skipped_long}", flush=True)
    if len(groups) < args.min_live:
        print(f"[replay] NOT_ENOUGH_LIVE ({len(groups)}/{args.min_live}) — keep accumulating", flush=True)
        sys.exit(3)
    if args.check_only:
        print("[replay] CHECK_ONLY_OK", flush=True)
        sys.exit(0)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Build (input_ids, completion_mask, advantage) per sequence
    seqs = []
    for g in groups:
        mu = sum(g["rewards"]) / len(g["rewards"])
        chat = [{"role": "system", "content": SYS}, {"role": "user", "content": g["question"]}]
        try:
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        for t, rw in zip(g["texts"], g["rewards"]):
            c_ids = tok(t, add_special_tokens=False).input_ids
            adv = rw - mu
            if abs(adv) < 1e-8:
                continue
            seqs.append((p_ids, c_ids, adv))
    print(f"[replay] {len(seqs)} sequences with nonzero advantage; one full-batch step "
          f"(grad accumulation, micro={args.micro})", flush=True)

    opt.zero_grad(set_to_none=True)
    total_loss, n_done = 0.0, 0
    for i in range(0, len(seqs), args.micro):
        chunk = seqs[i:i + args.micro]
        maxlen = max(len(p) + len(c) for p, c, _ in chunk)
        input_ids = torch.full((len(chunk), maxlen), tok.pad_token_id or tok.eos_token_id,
                               dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        cmask = torch.zeros((len(chunk), maxlen), dtype=torch.bool)
        advs = torch.tensor([a for _, _, a in chunk], dtype=torch.float32)
        for j, (p, c, _) in enumerate(chunk):
            L = len(p) + len(c)
            input_ids[j, :L] = torch.tensor(p + c)
            attn[j, :L] = 1
            cmask[j, len(p):L] = True
        input_ids, attn, cmask, advs = (x.cuda() for x in (input_ids, attn, cmask, advs))
        out = model(input_ids=input_ids, attention_mask=attn)
        logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
        tgt = input_ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = cmask[:, 1:].float()
        seq_lp = (tok_lp * m).sum(1) / m.sum(1).clamp(min=1)
        loss = -(advs * seq_lp).sum() / len(seqs)
        loss.backward()
        total_loss += float(loss) * len(seqs)
        n_done += len(chunk)
        if (i // args.micro) % 20 == 0:
            print(f"[replay] {n_done}/{len(seqs)} seqs | running loss {total_loss:.4f}", flush=True)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"[replay] STEP DONE | loss {total_loss:.4f} | grad_norm {float(gn):.3f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"[replay] SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
