"""Paired re-measurement of a live online run's EARLY questions under the CURRENT adapter.

Hypothesis test (user-specified): if the current policy beats its at-the-time pass rates
on the run's own early questions, learning is real (and flat GSM8K = distribution gap);
if not, no learning is occurring.

Reuses original artifact verdicts where the same answer reappears; tool-judges novel answers.
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_question_gen as AQG  # noqa: E402
import online_self_play_grpo as OSP  # noqa: E402

FA = AQG.FA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--steps_from", type=int, default=1)
    ap.add_argument("--steps_to", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--fast", action="store_true",
                    help="Score by matching known-correct artifact answers; judge only when a question has none.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "steps.jsonl")) if l.strip()]
    early = [r for r in rows if args.steps_from <= r["step"] <= args.steps_to]
    latest_step = max(r["step"] for r in rows)
    # Pin a snapshot under a name the live loop will not unload (it only unloads solver_stepN).
    adapter_dir = None
    for s in range(latest_step, 0, -1):
        p = os.path.join(args.run_dir, "adapter_step%d" % s)
        if os.path.isdir(p):
            adapter_dir = os.path.abspath(p)
            latest_step = s
            break
    solver_model = "retest_snapshot_step%d" % latest_step
    OSP.http_json("/v1/load_lora_adapter", {"lora_name": solver_model, "lora_path": adapter_dir}, timeout=120)
    print("re-testing %d early questions (steps %d-%d) under %s" % (
        len(early), args.steps_from, args.steps_to, solver_model), flush=True)

    rng = random.Random(0)

    def retest(r):
        q = r["question"]
        with ThreadPoolExecutor(max_workers=args.k) as ex:
            outs = list(ex.map(lambda _: OSP.solver_rollout(
                q, solver_model, 1.1, 0.98, args.max_new_tokens), range(args.k)))
        answers = [(FA.findall(t) or [None])[-1] for t, _ in outs]
        old_verdicts = {a: v["verdict"] for a, v in (r.get("artifact_verdicts") or {}).items()}
        verdicts = dict(old_verdicts)
        if args.fast:
            correct_answers = {a for a, v in old_verdicts.items() if v}
            if correct_answers:
                new_pass = sum(1 for a in answers if a in correct_answers)
                res = {"step": r["step"], "question": q, "old_pass": r.get("pass_count"),
                       "new_pass": new_pass, "answers": answers, "n_novel_judged": 0}
                print("[step %d] old %s/8 -> new %d/8 | %s" % (res["step"], res["old_pass"], new_pass, q[:90]), flush=True)
                return res
        novel = sorted({a for a in answers if a is not None and a not in verdicts})
        if novel:
            with ThreadPoolExecutor(max_workers=len(novel)) as ex:
                futs = {a: ex.submit(OSP.judge_artifact, q,
                                     next(t for (t, _), ans in zip(outs, answers) if ans == a),
                                     0.0, rng) for a in novel}
            for a, fut in futs.items():
                verdicts[a] = fut.result()[0]
        new_pass = sum(1 for a in answers if a is not None and verdicts.get(a))
        res = {"step": r["step"], "question": q, "old_pass": r.get("pass_count"),
               "new_pass": new_pass, "answers": answers, "n_novel_judged": len(novel)}
        print("[step %d] old %s/8 -> new %d/8 | %s" % (res["step"], res["old_pass"], new_pass, q[:90]), flush=True)
        return res

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(retest, early))

    paired = [(x["old_pass"], x["new_pass"]) for x in results if x["old_pass"] is not None]
    deltas = [n - o for o, n in paired]
    mean_old = sum(o for o, _ in paired) / len(paired)
    mean_new = sum(n for _, n in paired) / len(paired)
    import math
    sd = math.sqrt(sum((d - sum(deltas) / len(deltas)) ** 2 for d in deltas) / max(1, len(deltas) - 1))
    se = sd / math.sqrt(len(deltas))
    print("==== PAIRED RESULT: n=%d | mean old %.2f -> new %.2f | mean delta %+.2f (SE %.2f) ====" % (
        len(paired), mean_old, mean_new, sum(deltas) / len(deltas), se), flush=True)
    out = args.out or os.path.join(args.run_dir, "early_retest.json")
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
