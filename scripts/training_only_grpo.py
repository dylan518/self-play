"""Fast training-only GRPO harness to find a config that LEARNS.

Strips the self-play loop to its core question: does GRPO training on fixed known-good
mid-difficulty questions improve the solver? No proposer (no thinking call), no judge
(answers are known -> exact-match reward). Measures in-distribution learning (held-out
questions from the same pool) and OOD (GSM8K) before vs after.
"""
import argparse, json, os, sys, re, random
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import online_self_play_grpo as OSP
from peft import LoraConfig, get_peft_model
from grpo_math.models.policy import load_policy_and_ref
FA = OSP.FA
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")

def pass_rate(questions, refs, solver_model, k, max_tok, workers=16):
    def one(qr):
        q, ref = qr
        with ThreadPoolExecutor(max_workers=k) as ex:
            outs = list(ex.map(lambda _: OSP.solver_rollout(q, solver_model, 1.1, 0.98, max_tok), range(k)))
        ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        return sum(1 for a in ans if a is not None and int(a) == ref) / k
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rates = list(ex.map(one, zip(questions, refs)))
    return sum(rates) / max(1, len(rates))

def gsm8k_eval(solver_model, n, k, max_tok):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    def one(ex):
        q = ex["question"]; gold = int(ex["answer"].split("####")[-1].replace(",", "").strip())
        with ThreadPoolExecutor(max_workers=k) as e:
            outs = list(e.map(lambda _: OSP.solver_rollout(q, solver_model, 1.1, 0.98, max_tok), range(k)))
        ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        return 1.0 if any(a is not None and int(a) == gold for a in ans) else 0.0  # pass@k
    with ThreadPoolExecutor(max_workers=16) as ex:
        return sum(ex.map(one, ds)) / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="outputs/fixed_train_questions.json")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--grad_epochs", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--gsm8k_n", type=int, default=150)
    ap.add_argument("--out_dir", default="outputs/train_only")
    ap.add_argument("--tag", default="A")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(0)

    items = json.load(open(args.questions))
    rng.shuffle(items)
    held = items[:args.heldout]; train = items[args.heldout:]
    h_q = [x["question"] for x in held]; h_r = [x["answer"] for x in held]
    print("[%s] train=%d heldout=%d | lr=%.1e epochs=%d r=%d batch=%d" % (
        args.tag, len(train), len(held), args.lr, args.grad_epochs, args.lora_r, args.batch), flush=True)

    bundle = load_policy_and_ref(MODEL, torch.bfloat16, gradient_checkpointing=True, load_ref=False)
    tok = bundle.tokenizer
    policy = get_peft_model(bundle.policy, LoraConfig(
        r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=0.05, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM")).to("cuda")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    device = torch.device("cuda")

    t_q = [x["question"] for x in train]; t_r = [x["answer"] for x in train]
    solver_model = MODEL
    # BEFORE
    base_tr = pass_rate(t_q, t_r, solver_model, args.k, args.max_new_tokens)
    base_in = pass_rate(h_q, h_r, solver_model, args.k, args.max_new_tokens)
    base_ood = gsm8k_eval(solver_model, args.gsm8k_n, 1, args.max_new_tokens)
    print("[%s] BEFORE: trained-Q %.3f | held-out %.3f | GSM8K pass@1 %.3f" % (args.tag, base_tr, base_in, base_ood), flush=True)

    for step in range(1, args.steps + 1):
        batch = rng.sample(train, args.batch)
        jobs = [(it["question"], it["answer"], qi) for qi, it in enumerate(batch) for _ in range(args.k)]
        with ThreadPoolExecutor(max_workers=min(48, len(jobs))) as ex:
            outs = list(ex.map(lambda j: OSP.solver_rollout(j[0], solver_model, 1.1, 0.98, args.max_new_tokens), jobs))
        byq = {}
        for (q, ref, qi), (text, _) in zip(jobs, outs):
            byq.setdefault(qi, [q, ref, []])[2].append(text)
        groups = []
        for qi in sorted(byq):
            q, ref, comps = byq[qi]
            ans = [(FA.findall(c) or [None])[-1] for c in comps]
            rewards = [1.0 if (a is not None and int(a) == ref) else 0.0 for a in ans]
            groups.append((q, comps, rewards))
        m = {"n_trained_groups": 0, "loss": 0.0, "kl": 0.0}
        for _ in range(args.grad_epochs):
            m = OSP.grpo_step(policy, opt, tok, device, groups, float(os.environ.get('KL_BETA','0.01')), pg_norm=(float(os.environ['PG_NORM']) if os.environ.get('PG_NORM') else None))
            if m["n_trained_groups"] == 0: break
        passes = [int(sum(r)) for _, _, r in groups]
        name = OSP.push_adapter(policy, step, args.out_dir, prev_name=None if solver_model == MODEL else solver_model)
        if name: solver_model = name
        print("[%s step %d] passes=%s trained=%d loss=%.4f solver=%s" % (
            args.tag, step, passes, m["n_trained_groups"], m["loss"], solver_model), flush=True)

    # AFTER
    aft_tr = pass_rate(t_q, t_r, solver_model, args.k, args.max_new_tokens)
    aft_in = pass_rate(h_q, h_r, solver_model, args.k, args.max_new_tokens)
    aft_ood = gsm8k_eval(solver_model, args.gsm8k_n, 1, args.max_new_tokens)
    print("\n[%s] ==== RESULT ====" % args.tag, flush=True)
    print("[%s] TRAINED-Q (memoriz): %.3f -> %.3f  (delta %+.3f)" % (args.tag, base_tr, aft_tr, aft_tr - base_tr), flush=True)
    print("[%s] HELD-OUT (generaliz): %.3f -> %.3f  (delta %+.3f)" % (args.tag, base_in, aft_in, aft_in - base_in), flush=True)
    print("[%s] OOD GSM8K pass@1:    %.3f -> %.3f  (delta %+.3f)" % (args.tag, base_ood, aft_ood, aft_ood - base_ood), flush=True)
    json.dump({"tag": args.tag, "base_in": base_in, "aft_in": aft_in, "base_ood": base_ood, "aft_ood": aft_ood,
               "lr": args.lr, "epochs": args.grad_epochs, "r": args.lora_r, "batch": args.batch, "steps": args.steps},
              open(os.path.join(args.out_dir, "result_%s.json" % args.tag), "w"), indent=1)

if __name__ == "__main__":
    main()
