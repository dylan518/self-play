"""Companion metrics logger for an online_self_play_grpo run.

Tails steps.jsonl + detail/ of a live run and streams derived metrics to a
wandb run, backfilling existing steps. Metrics: mean_reward, format_rate
(parseable FINAL_ANSWER fraction), judge_agreement, n_artifacts, discard-ish
zero-artifact count, mean completion chars, in-band fraction.
"""

import argparse
import json
import os
import re
import time

import wandb

AGREE_RE = re.compile(r"judge_agreement ([0-9.]+) over (\d+) artifacts")


def step_metrics(rows, detail_dir):
    k = rows[0].get("k", 8)
    passes = [r["pass_count"] for r in rows if r.get("pass_count") is not None]
    answers = [a for r in rows for a in r.get("answers", [])]
    agrees, n_art = [], []
    for r in rows:
        m = AGREE_RE.search(r.get("agreement", ""))
        if m:
            agrees.append(float(m.group(1)))
            n_art.append(int(m.group(2)))
    comp_chars = []
    step = rows[0]["step"]
    for qi in range(1, len(rows) + 1):
        p = os.path.join(detail_dir, "step%d_q%d_rollouts.json" % (step, qi))
        if os.path.exists(p):
            try:
                R = json.load(open(p))
                comp_chars.extend(len(x["completion"]) for x in R["rollouts"])
            except Exception:
                pass
    n = max(1, len(passes))
    return {
        "step": step,
        "mean_reward": sum(p / k for p in passes) / n,
        "mean_pass": sum(passes) / n,
        "format_rate": sum(1 for a in answers if a is not None) / max(1, len(answers)),
        "judge_agreement": sum(agrees) / max(1, len(agrees)),
        "mean_artifacts_per_q": sum(n_art) / max(1, len(n_art)),
        "in_band_frac": sum(1 for p in passes if 1 <= p <= k - 1) / n,
        "forced_frac_q": sum(r.get("n_forced_answers", 0) for r in rows) / max(1, k * len(rows)),
        "mean_completion_chars": sum(comp_chars) / max(1, len(comp_chars)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--name", type=str, default="online-selfplay-grpo-metrics")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    wandb.init(project=os.environ.get("WANDB_PROJECT", "grpo-math"), name=args.name)
    steps_path = os.path.join(args.run_dir, "steps.jsonl")
    detail_dir = os.path.join(args.run_dir, "detail")
    logged = set()
    while True:
        rows = [json.loads(l) for l in open(steps_path)] if os.path.exists(steps_path) else []
        by_step = {}
        for r in rows:
            by_step.setdefault(r["step"], []).append(r)
        done_steps = sorted(s for s in by_step if s not in logged)
        # last step may still be mid-write; only log steps strictly older than the max
        new = False
        for s in done_steps[:-1] if len(done_steps) > 1 else []:
            m = step_metrics(by_step[s], detail_dir)
            wandb.log(m, step=s)
            logged.add(s)
            new = True
            print("logged step", s, m, flush=True)
        if new:
            t = wandb.Table(columns=["step", "pass_count", "agreement", "structure_label", "answers", "question"])
            for r in rows:
                t.add_data(r["step"], r.get("pass_count"), r.get("agreement", ""),
                           r.get("structure_label", ""), str(r.get("answers", "")), r["question"])
            wandb.log({"questions": t})
            jt = wandb.Table(columns=["step", "q", "artifact_answer", "verdict", "judge_run", "temperature",
                                      "tool_calls", "final_response", "code", "execution_result", "question"])
            import glob as _glob
            for jp in sorted(_glob.glob(os.path.join(detail_dir, "step*_q*_judge.json"))):
                m = re.search(r"step(\d+)_q(\d+)_judge", jp)
                try:
                    J = json.load(open(jp))
                except Exception:
                    continue
                for a, info in J.get("artifacts", {}).items():
                    for ri, run in enumerate(info.get("runs", []), 1):
                        tr = run.get("trace", [])
                        code = "\n---\n".join(x["code"] for x in tr if "code" in x)[:4000]
                        res = "\n---\n".join(x["result"] for x in tr if "result" in x)[:2000]
                        final = "\n".join(x["final_response"] for x in tr if "final_response" in x)[:1500]
                        jt.add_data(int(m.group(1)), int(m.group(2)), a,
                                    {True: "CORRECT", False: "INCORRECT", None: "NONE"}.get(run.get("verdict")),
                                    ri, run.get("temperature"), run.get("tool_calls"),
                                    final, code, res, J.get("question", "")[:300])
            wandb.log({"judge_rollouts": jt})
            print("logged questions table:", len(rows), "and judge table", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
