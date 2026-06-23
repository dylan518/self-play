"""Minimal robust microscope: can GRPO sharpen ONE mid-difficulty question?
Fixed question (no auto-pick), model loaded first, 3072 tokens, prints every step."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import online_self_play_grpo as OSP
from peft import LoraConfig, get_peft_model
from grpo_math.models.policy import load_policy_and_ref
FA = OSP.FA
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
MT = 3072

def measure(question, ref, model, k):
    with ThreadPoolExecutor(max_workers=k) as ex:
        outs = list(ex.map(lambda _: OSP.solver_rollout(question, model, 1.1, 0.98, MT), range(k)))
    ans = [(FA.findall(t) or [None])[-1] for t, _ in outs]
    comps = [t for t, _ in outs]
    rewards = [1.0 if (a is not None and int(a) == ref) else 0.0 for a in ans]
    return comps, rewards, sum(rewards) / k

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    items = json.load(open("outputs/fixed_train_questions.json"))
    # fixed pick: a pass-5 question with a NON-zero answer (genuine mid, not the 0-trick)
    q = next(x for x in items if x["pass"] == 5 and x["answer"] != 0)
    question, ref = q["question"], q["answer"]
    print("MICRO q (v3 pass %d, ans=%s): %s" % (q["pass"], ref, question[:90]), flush=True)

    print("loading model...", flush=True)
    bundle = load_policy_and_ref(MODEL, torch.bfloat16, gradient_checkpointing=True, load_ref=False)
    tok = bundle.tokenizer
    policy = get_peft_model(bundle.policy, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules="all-linear",
        task_type="CAUSAL_LM")).to("cuda")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    device = torch.device("cuda")
    solver = MODEL
    print("model loaded, measuring base...", flush=True)

    _, _, base = measure(question, ref, solver, k)
    print("step 0 (base): pass %.3f" % base, flush=True)
    for step in range(1, steps + 1):
        comps, rewards, p = measure(question, ref, solver, k)
        for _ in range(2):
            m = OSP.grpo_step(policy, opt, tok, device, [(question, comps, rewards)], 0.01)
            if m["n_trained_groups"] == 0:
                break
        name = OSP.push_adapter(policy, step, "outputs/microscope", prev_name=None if solver == MODEL else solver)
        if name:
            solver = name
        print("step %d: pass %.3f loss %.4f trained=%d" % (step, p, m["loss"], m["n_trained_groups"]), flush=True)
    _, _, final = measure(question, ref, solver, k)
    print("\n==== MICRO: base %.3f -> final %.3f (delta %+.3f) ====" % (base, final, final - base), flush=True)

if __name__ == "__main__":
    main()
