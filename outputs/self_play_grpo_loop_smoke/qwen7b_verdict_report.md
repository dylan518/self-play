# Qwen 7B verdict GRPO smoke report

Status: blocked

## Scope

- Branch run remotely: `orch/self-play-rl/qwen-verdict-rl-loop`
- Training config prepared: `grpo_math/configs/train_solver_verdict_grpo_qwen7b_gpu_smoke.yaml`
- Requested trainer command: `python -m grpo_math.trl.train_solver_verdict_grpo --config grpo_math/configs/train_solver_verdict_grpo_qwen7b_gpu_smoke.yaml`
- Configured model: `Qwen/Qwen2.5-7B-Instruct`
- Configured `train.steps`: `4`
- Completed GRPO steps: `0`

## Setup and smoke gate results

1. Remote branch sync completed: the remote checkout was reset to `origin/orch/self-play-rl/qwen-verdict-rl-loop`.
2. First required setup dry run failed in the default login environment:
   - Command: `bash scripts/washu_setup.sh --dry-run`
   - Exit code: `1`
   - Relevant output: `Python executable: /usr/bin/python3`, `Python version: 3.9.25`, `Python >= 3.10 is required.`
3. I bootstrapped a workspace-local Python 3.11 with `uv` and reran the setup sequence with that interpreter first on `PATH`.
4. The setup dry run then passed:
   - Command: `bash scripts/washu_setup.sh --dry-run`
   - Exit code: `0`
   - Relevant output: `Python executable: .../.remote_python/cpython-3.11.15-linux-x86_64-gnu/bin/python3`, `Python version: 3.11.15`
5. The real setup did not complete:
   - Command: `bash scripts/washu_setup.sh`
   - Last observed phase: `.venv/bin/python -m pip install -e .[dev]`
   - Last emitted setup output: `Installing collected packages: ... torch, huggingface_hub, tokenizers, deepspeed, datasets, accelerate, transformers, trl, peft, grpo-math`
   - The process was still in `pip install -e .[dev]` after roughly 50 minutes, so I killed the setup process to avoid leaking remote work.
6. Because setup did not complete, `bash scripts/washu_run_smoke.sh --dry-run`, `bash scripts/washu_run_smoke.sh`, vLLM startup, and `train_solver_verdict_grpo` were not reached.

## Trainer-state metrics

No `trainer_state.json` was produced because zero GRPO steps completed. Requested per-step metrics are therefore unavailable:

| step | loss | grad_norm | reward | reward_std | frac_reward_zero_std | kl | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Verdict reward variance

No verifier cache was produced because training never started. There are no sample prompts with verdicts from this run, and no evidence of mixed `CORRECT` / `INCORRECT` verdicts.

## Python-assisted verifier subprocess evidence

No `outputs/verifier_cache/solver_verdict_grpo*.jsonl` file was produced by this smoke attempt, so there are no `python_enumerated_*` cache artifacts to inspect. The Python-assisted verifier is enabled in the committed config via:

```yaml
verifier:
  python_assisted: true
  python_timeout_s: 5
```

but its subprocess was not invoked because the GRPO trainer did not start.

## Assessment

RL is not working yet in this cloud-GPU smoke because the run was blocked before the no-GPU smoke gates and before any SLURM GPU allocation. The committed config is shaped correctly for the requested verdict-based Qwen loop (`rollout.k: 4`, no `sampling_groups`, OpenAI-compatible `gpt-4.1`, `python_assisted: true`, and `train.steps: 4`), but the remote substrate needs a working Python >=3.10 environment whose `pip install -e .[dev]` completes reliably before vLLM or GRPO training can be exercised.
