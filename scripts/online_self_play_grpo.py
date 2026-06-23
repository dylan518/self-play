"""Online self-play GRPO: ONE set of solver rollouts per question serves both
training (GRPO group advantages) and measurement (pass-rate feedback to the
in-context proposer). Judges roll out AFTER the solver; every question gets 2
independent tool-judge runs, and a subportion (plus all disagreements) gets a
3rd — judge agreement is the verifiability signal in the proposer feedback.

Per step:
  1. proposer (in-context, feedback-conditioned) keeps a question queue filled
  2. judge: 2 tool-loop runs/question (+3rd for subportion/disagreement) -> reference
  3. solver: k rollouts per question via vLLM with the CURRENT LoRA adapter
  4. GRPO update on the training GPU (micro-batch 1, memory-safe logprob gather)
  5. push updated adapter to vLLM (runtime LoRA swap) -> next rollouts on-policy
  6. feedback line [pass n/k | judge_agreement | structure] -> proposer history

Requires the vLLM server started with:
  --enable-lora --max-lora-rank 16 --enable-auto-tool-choice --tool-call-parser qwen3_xml
and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentic_question_gen as AQG  # noqa: E402
import python_tool_judge as PTJ  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from grpo_math.models.policy import load_policy_and_ref, sequence_logprobs  # noqa: E402

BASE_URL = os.environ.get("VLLM_BASE", "http://127.0.0.1:8001")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
SOLVER_TMPL = AQG.SOLVER_TMPL
FA = AQG.FA


def http_json(path, body, timeout=1800, retries=5):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except Exception:
                # /v1/load_lora_adapter returns plain text on success
                return {"raw": raw.decode("utf-8", "replace"), "status": "ok"}
        except Exception as e:
            last = e
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise last


def vllm_chat(prompt, model, temperature, top_p, max_tokens):
    out = http_json("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return out["choices"][0]["message"]["content"] or ""


def solver_rollout(question, model, temperature, top_p, max_tokens):
    """One rollout; if it clips without FINAL_ANSWER, force the tag and let the
    policy sample the integer (mirrors the legacy api_final_prefill behavior).
    Returns (text, forced_bool) - forced rate is the DAPO-guard metric."""
    text = vllm_chat(SOLVER_TMPL % question, model, temperature, top_p, max_tokens)
    if FA.findall(text):
        return text, False
    forced_prefix = "\nFINAL_ANSWER:"
    out = http_json("/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "user", "content": SOLVER_TMPL % question},
            {"role": "assistant", "content": text + forced_prefix},
        ],
        "continue_final_message": True, "add_generation_prompt": False,
        "temperature": 0.2, "top_p": 0.95, "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    tail = out["choices"][0]["message"]["content"] or ""
    return text + forced_prefix + tail, True


def judge_artifact(question, solution, triple_p, rng):
    """Independent judge rollouts on ONE artifact (question, solution).
    2 tool-loop verdicts; 3rd on split or for a random subportion.
    Returns (verdict_bool, votes_str, runs_detail). Unresolved -> (False, "0/x", runs)."""
    def one(temp):
        try:
            v, n_calls, trace = PTJ.judge_solution(question, solution, temperature=temp)
            return {"temperature": temp, "verdict": v, "tool_calls": n_calls, "trace": trace}
        except Exception as e:
            return {"temperature": temp, "verdict": None, "tool_calls": 0, "trace": [], "error": str(e)}

    with ThreadPoolExecutor(max_workers=2) as ex:
        runs = list(ex.map(one, (0.6, 0.6)))
    votes = [r["verdict"] for r in runs if r["verdict"] is not None]
    if len(votes) < 2 or votes[0] != votes[1] or rng.random() < triple_p:
        runs.append(one(0.6))
        votes = [r["verdict"] for r in runs if r["verdict"] is not None]
    n_true = sum(1 for v in votes if v)
    n_false = sum(1 for v in votes if v is False)
    verdict = n_true > n_false
    maj = max(n_true, n_false)
    return verdict, "%d/%d" % (maj, len(runs)), runs


def judge_question(question, triple_p, rng):
    """2 tool-judge runs; 3rd on disagreement or for a random subportion.
    Returns (reference, agreement_str, runs_detail)."""
    def one(temp):
        try:
            ref, n_calls, trace = PTJ.judge_reference(question, temperature=temp)
            return {"temperature": temp, "reference": ref, "tool_calls": n_calls, "trace": trace}
        except Exception as e:
            return {"temperature": temp, "reference": None, "tool_calls": 0, "trace": [], "error": str(e)}

    with ThreadPoolExecutor(max_workers=2) as ex:
        runs = list(ex.map(one, (0.6, 0.6)))
    if runs[0]["reference"] is None or runs[0]["reference"] != runs[1]["reference"] or rng.random() < triple_p:
        runs.append(one(0.6))
    vals = [r["reference"] for r in runs if r["reference"] is not None]
    ref = max(set(vals), key=vals.count) if vals else None
    if ref is None or vals.count(ref) < 2:
        return None, "judges_agree 0/%d" % len(runs), runs
    return ref, "judges_agree %d/%d" % (vals.count(ref), len(runs)), runs


class Proposer:
    def __init__(self, history_banks, focus_idx=0, detail_dir=None):
        self.history = []
        self.queue = []
        self.detail_dir = detail_dir
        self.n_refills = 0
        self.focus = AQG.CHAIN_FOCUS[focus_idx % len(AQG.CHAIN_FOCUS)]
        for bank in history_banks:
            if not os.path.exists(bank):
                continue
            prior = [json.loads(l) for l in open(bank, encoding="utf-8") if l.strip()]
            with ThreadPoolExecutor(max_workers=24) as ex:
                labels = list(ex.map(
                    lambda r: r.get("structure_label") or AQG.label_structure(r["question"]), prior))
            with AQG._shared_lock:
                AQG._all_labels.extend(labels)
            self.history.extend(
                (r["question"], r.get("pass_count", 0),
                 "3/3" if r.get("reference_answer") is not None else "JUDGES DISAGREE - DISCARDED",
                 "structure: " + l)
                for r, l in zip(prior, labels))
        print("[proposer] seeded %d history entries, %d labels" % (len(self.history), len(AQG._all_labels)), flush=True)

    def refill(self):
        with AQG._shared_lock:
            labels = list(AQG._all_labels)
        prompt = AQG.build_prompt(self.history[-80:], self.focus, labels)
        out = AQG.chat(prompt, 0.9, 0.95, 16000, True)
        qs = AQG.parse_questions(out)
        print("[proposer] proposed %d questions" % len(qs), flush=True)
        self.n_refills += 1
        if self.detail_dir:
            with open(os.path.join(self.detail_dir, "proposer_refill_%d.txt" % self.n_refills), "w", encoding="utf-8") as f:
                f.write("===== FULL PROMPT =====\n%s\n\n===== RAW OUTPUT (incl. thinking) =====\n%s" % (prompt, out))
        self.queue.extend(qs)

    def next_questions(self, n):
        while len(self.queue) < n:
            self.refill()
        out, self.queue = self.queue[:n], self.queue[n:]
        return out

    def feedback(self, question, passes, k, agreement, discarded=False):
        label, n_same = AQG.register_question(question)
        verif = "JUDGES DISAGREE - DISCARDED" if discarded else agreement
        note = "structure: %s (%s)" % (label, "novel" if n_same == 0 else "%d produced before" % n_same)
        self.history.append((question, passes, verif, note))
        return label


def build_training_batch(tokenizer, question, completions, max_len=10240):
    """Tokenize chat-template prompt + completion exactly as vLLM produced them."""
    msgs = [{"role": "user", "content": SOLVER_TMPL % question}]
    prompt_text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    rows, prompt_ends = [], []
    for c in completions:
        comp_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            comp_ids = comp_ids + [tokenizer.eos_token_id]
        ids = (prompt_ids + comp_ids)[:max_len]
        rows.append(ids)
        prompt_ends.append(min(len(prompt_ids), max_len))
    T = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(rows), T), pad, dtype=torch.long)
    attn = torch.zeros((len(rows), T), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r)
        attn[i, : len(r)] = 1
    return input_ids, attn, torch.tensor(prompt_ends, dtype=torch.long)


def grpo_step(policy, optimizer, tokenizer, device, groups, kl_beta, microbatch=1, pg_norm=None):
    """groups: list of (question, completions[k], rewards[k]). Returns metrics."""
    policy.train()
    total_loss = 0.0
    total_kl = 0.0
    n_groups = 0
    optimizer.zero_grad(set_to_none=True)
    for question, comps, rewards in groups:
        r = torch.tensor(rewards, dtype=torch.float32, device=device)
        if r.std() < 1e-6:
            continue  # degenerate group: no advantage signal
        adv = (r - r.mean()) / (r.std() + 1e-4)
        input_ids, attn, pe = build_training_batch(tokenizer, question, comps)
        input_ids, attn, pe = input_ids.to(device), attn.to(device), pe.to(device)
        for s in range(0, len(comps), microbatch):
            e = min(len(comps), s + microbatch)
            sum_logp, gen_mask = sequence_logprobs(policy, input_ids[s:e], attn[s:e], pe[s:e])
            n_tok = gen_mask.sum(dim=1).clamp(min=1)
            with torch.no_grad(), policy.disable_adapter():
                ref_logp, _ = sequence_logprobs(policy, input_ids[s:e], attn[s:e], pe[s:e])
            # k1 KL estimate per generated token, sequence-averaged
            kl = (sum_logp - ref_logp) / n_tok
            denom = n_tok if pg_norm is None else float(pg_norm)
            pg = -(adv[s:e] * sum_logp / denom)
            loss = (pg + kl_beta * kl).mean() * (e - s) / len(comps)
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach().mean())
        n_groups += 1
    if n_groups:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return {"n_trained_groups": n_groups, "loss": total_loss, "kl": total_kl / max(1, n_groups)}


def push_adapter(policy, step, out_dir, prev_name=None):
    path = os.path.join(out_dir, "adapter_step%d" % step)
    policy.save_pretrained(path)
    name = "solver_step%d" % step
    try:
        http_json("/v1/load_lora_adapter", {"lora_name": name, "lora_path": os.path.abspath(path)}, timeout=120)
    except Exception as e:
        print("[adapter] vLLM load failed (%s); rollouts stay on previous adapter" % e, flush=True)
        return None
    if prev_name and prev_name != name:
        try:
            http_json("/v1/unload_lora_adapter", {"lora_name": prev_name}, timeout=60)
        except Exception as e:
            print("[adapter] unload of %s failed (%s)" % (prev_name, e), flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--questions_per_step", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--triple_judge_p", type=float, default=0.34)
    ap.add_argument("--kl_beta", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--grad_epochs", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="outputs/online_selfplay")
    ap.add_argument("--history_bank", type=str, nargs="*", default=[])
    ap.add_argument("--push_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    detail_dir = os.path.join(args.out_dir, "detail")
    os.makedirs(detail_dir, exist_ok=True)
    rng = random.Random(args.seed)
    log_path = os.path.join(args.out_dir, "steps.jsonl")

    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and os.environ.get("WANDB_MODE") != "offline"
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "grpo-math"), name="online-selfplay-grpo")

    bundle = load_policy_and_ref(MODEL, torch.bfloat16, gradient_checkpointing=True, load_ref=False)
    tokenizer = bundle.tokenizer
    policy = get_peft_model(bundle.policy, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules="all-linear",
        task_type="CAUSAL_LM"))
    device = torch.device("cuda")
    policy.to(device)
    policy.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    proposer = Proposer(args.history_bank, detail_dir=detail_dir)
    solver_model = MODEL  # until first adapter push

    for step in range(1, args.steps + 1):
      try:
        t0 = time.time()
        questions = proposer.next_questions(args.questions_per_step)

        def process_question(qi_q):
            qi, q = qi_q
            # 1) THE rollouts come FIRST: one set, used for training AND measurement
            #    (clipped rollouts get a forced FINAL_ANSWER continuation)
            with ThreadPoolExecutor(max_workers=args.k) as ex:
                outs = list(ex.map(lambda _: solver_rollout(
                    q, solver_model, 1.1, 0.98, args.max_new_tokens), range(args.k)))
            comps = [t for t, _ in outs]
            n_forced = sum(1 for _, fb in outs if fb)
            answers = [(FA.findall(c) or [None])[-1] for c in comps]

            # 2) judges roll out AFTER, independently, per unique artifact
            #    (for math the artifact collapses to the claimed answer; identical
            #    artifacts share verdicts - a no-op once artifacts are repos)
            uniq = {}
            for a, c in zip(answers, comps):
                if a is not None and a not in uniq:
                    uniq[a] = c
            verdicts = {}
            with ThreadPoolExecutor(max_workers=max(1, len(uniq))) as ex:
                futs = {a: ex.submit(judge_artifact, q, c, args.triple_judge_p, rng) for a, c in uniq.items()}
            for a, fut in futs.items():
                verdicts[a] = fut.result()

            rewards = [1.0 if (a is not None and verdicts[a][0]) else 0.0 for a in answers]
            passes = int(sum(rewards))
            votes = [verdicts[a][1] for a in verdicts]
            agree_fracs = [int(v.split("/")[0]) / int(v.split("/")[1]) for v in votes] or [0.0]
            agreement = "judge_agreement %.2f over %d artifacts" % (sum(agree_fracs) / len(agree_fracs), len(verdicts))

            with open(os.path.join(detail_dir, "step%d_q%d_judge.json" % (step, qi + 1)), "w", encoding="utf-8") as f:
                json.dump({"question": q, "agreement": agreement,
                           "artifacts": {a: {"verdict": v[0], "votes": v[1], "runs": v[2]} for a, v in verdicts.items()}}, f, indent=1)
            with open(os.path.join(detail_dir, "step%d_q%d_rollouts.json" % (step, qi + 1)), "w", encoding="utf-8") as f:
                json.dump({"question": q, "solver_model": solver_model,
                           "rollouts": [{"i": i, "answer": a, "reward": r, "completion": c}
                                        for i, (c, a, r) in enumerate(zip(comps, answers, rewards))]}, f, indent=1)
            return q, comps, rewards, passes, agreement, answers, verdicts, n_forced

        # all questions in parallel: rollouts + judging overlap across questions
        with ThreadPoolExecutor(max_workers=args.questions_per_step) as ex:
            results = list(ex.map(process_question, enumerate(questions)))

        groups = []
        records = []
        step_forced = 0
        for q, comps, rewards, passes, agreement, answers, verdicts, n_forced in results:
            step_forced += n_forced
            label = proposer.feedback(q, passes, args.k, agreement)
            groups.append((q, comps, rewards))
            records.append({"step": step, "question": q, "agreement": agreement,
                            "structure_label": label, "pass_count": passes, "k": args.k,
                            "answers": answers, "n_forced_answers": n_forced,
                            "artifact_verdicts": {a: {"verdict": v[0], "votes": v[1]} for a, v in verdicts.items()}})

        metrics = {"n_trained_groups": 0, "loss": 0.0, "kl": 0.0}
        for _ep in range(args.grad_epochs):
            metrics = grpo_step(policy, optimizer, tokenizer, device, groups, args.kl_beta)
            if metrics["n_trained_groups"] == 0:
                break
        if step % args.push_every == 0 and metrics["n_trained_groups"] > 0:
            name = push_adapter(policy, step, args.out_dir,
                                prev_name=None if solver_model == MODEL else solver_model)
            if name:
                solver_model = name

        passes_str = [r["pass_count"] for r in records]
        metrics.update({"step": step, "passes": passes_str, "sec": round(time.time() - t0, 1),
                        "forced_frac": round(step_forced / max(1, args.k * len(records)), 3),
                        "solver_model": solver_model})
        print("[step %d] %s" % (step, json.dumps(metrics)), flush=True)
        with open(log_path, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if use_wandb:
            import wandb
            n_meas = [p for p in passes_str if p is not None]
            all_answers = [a for r in records for a in r.get("answers", [])]
            agrees = [float(m.group(1)) for r in records
                      for m in [re.search(r"judge_agreement ([0-9.]+)", r.get("agreement", ""))] if m]
            wandb.log({"step": step, "loss": metrics["loss"], "kl": metrics["kl"],
                       "n_trained_groups": metrics["n_trained_groups"],
                       "mean_pass": sum(n_meas) / max(1, len(n_meas)),
                       "mean_reward": sum(n_meas) / max(1, len(n_meas)) / args.k,
                       "format_rate": sum(1 for a in all_answers if a is not None) / max(1, len(all_answers)),
                       "judge_agreement": sum(agrees) / max(1, len(agrees)),
                       "in_band_frac": sum(1 for p in n_meas if 1 <= p <= args.k - 1) / max(1, len(n_meas)),
                       "mean_completion_chars": sum(len(c) for _, cs, _ in groups for c in cs) / max(1, sum(len(cs) for _, cs, _ in groups)),
                       "forced_frac": metrics["forced_frac"],
                       "step_seconds": metrics["sec"]})
      except Exception as e:
        import traceback
        print("[step %d] FAILED, skipping: %s" % (step, e), flush=True)
        traceback.print_exc()
        time.sleep(30)

    push_adapter(policy, args.steps + 1, args.out_dir,
                 prev_name=None if solver_model == MODEL else solver_model)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
