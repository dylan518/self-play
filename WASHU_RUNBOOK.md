# Wash U Server Runbook

This runbook is for bringing the self-play repo up on a Wash U GPU server and
running progressively larger smoke tests before launching full iterations.

The target loop is:

```text
proposer -> question filter -> solver samples -> verifier verdicts
  -> solver rewards/advantages -> proposer rewards/advantages
  -> solver GRPO -> proposer GRPO -> evaluation -> next iteration
```

The local no-GPU path now has entry points for fixture rollout collection,
solver verdict GRPO validation, proposer question GRPO validation, and
iteration/loop command planning.

## One-Time Setup

From the repository root on the server:

```bash
bash scripts/washu_setup.sh --dry-run
bash scripts/washu_setup.sh
source .venv/bin/activate
```

If the node has a compatible Linux/CUDA/PyTorch stack and you want Flash
Attention, install the optional dependency:

```bash
bash scripts/washu_setup.sh --install-flash-attn
```

The setup script checks:

- Python version, requiring Python 3.10 or newer.
- `nvidia-smi` availability and GPU count.
- PyTorch import status, CUDA availability, and CUDA device count.
- Disk space for the repo filesystem.
- Presence of `OPENAI_API_KEY`, `GEMINI_API_KEY`, `WANDB_API_KEY`, and `HF_TOKEN`.
- Editable install commands for `.[dev]`, with optional `.[flash]`.

## Required Environment

Set only the keys needed by the configs you are running:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export HF_TOKEN="..."
export WANDB_API_KEY="..."
```

Notes:

- OpenAI-backed rollout configs read `OPENAI_API_KEY`.
- Gemini OpenAI-compatible rollout configs read `GEMINI_API_KEY`.
- Hugging Face model downloads may require `HF_TOKEN`, depending on model access.
- W&B is enabled in the main training config, so set `WANDB_API_KEY` or disable
W&B in the config for non-logged runs.
- Local vLLM/OpenAI-compatible configs may use `api_key: EMPTY` and a local
`api_base_url`; start that server separately before running those configs.

Recommended runtime variables:

```bash
export VENV_DIR=.venv
export PYTHON=.venv/bin/python
export TOKENIZERS_PARALLELISM=false
```

## Smoke Test Order

Run the dry-run first:

```bash
bash scripts/washu_run_smoke.sh --dry-run
```

Then run the smoke sequence:

```bash
bash scripts/washu_run_smoke.sh
```

The current smoke script covers:

- Fixture rollout collection using `grpo_math.self_play.collect_self_play_rollouts`.
- Solver verdict GRPO config validation using
`grpo_math.trl.train_solver_verdict_grpo --validate-only`.
- Proposer question GRPO config validation using
`grpo_math.trl.train_proposer_question_grpo --validate-only`.
- Full iteration dry-run using `grpo_math.self_play.run_self_play_iteration`.

The default rollout smoke is fixture backed and does not require a GPU:

```bash
python -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --overwrite
```

The solver/proposer steps in the smoke script are validation-only locally:

```bash
python -m grpo_math.trl.train_solver_verdict_grpo \
  --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml \
  --validate-only

python -m grpo_math.trl.train_proposer_question_grpo \
  --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml \
  --validate-only
```

For an API-backed rollout smoke after keys are set, choose one of the existing
API configs:

```bash
python -m grpo_math.self_play.generate_pairwise_data \
  --config grpo_math/configs/pairwise_rollouts_gemini25flash_pairwise_smoke.yaml
```

## Full Iteration

Once the Agent 7 runner exists, use:

```bash
bash scripts/washu_run_iteration.sh --dry-run
bash scripts/washu_run_iteration.sh \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --output-dir outputs/self_play_iterations/washu_iter_000
```

Useful flags:

- `--skip-train` collects/evaluates without solver/proposer training when the
runner supports it.
- `--resume` resumes an interrupted output directory when the runner supports it.
- `--overwrite` replaces an output directory; use this only for disposable smoke
runs.

When moving beyond fixture configs, set `CONFIG` and `OUTPUT_DIR` explicitly:

```bash
CONFIG=grpo_math/configs/self_play_verdict_grpo_washu.yaml \
OUTPUT_DIR=outputs/self_play_iterations/washu_iter_001 \
bash scripts/washu_run_iteration.sh
```

## Outputs And Checkpoints

Current and planned paths:

- Pairwise/verdict rollout JSONL: configured by each rollout YAML under
`output.jsonl_path`, commonly `outputs/pairwise_rollouts*/samples*.jsonl`.
- Fixture self-play rollout JSONL:
`outputs/self_play_fixture/rollouts.jsonl`.
- Fixture rejected-question log:
`outputs/self_play_fixture/rejected_questions.jsonl`.
- Solver smoke checkpoints: `outputs/washu_smoke/solver_grpo`.
- Full iteration artifacts: `outputs/self_play_iterations/<run_name>`.
- Iteration report: `<iteration_output>/report.md`.
- Next iteration config: `<iteration_output>/next_iteration.yaml`.
- Solver/proposer checkpoints: planned under the iteration output directory or
the train config `output_dir`.
- W&B runs: project `grpo-math` unless overridden in the YAML.

Before launching a long run, confirm the target filesystem has enough free
space. Model checkpoints, rollout traces, and W&B logs can grow quickly.

## Resume Procedure

For an interrupted full iteration:

```bash
bash scripts/washu_run_iteration.sh \
  --config <same-config-used-before> \
  --output-dir <same-output-dir-used-before> \
  --resume
```

For a solver GRPO run using the current TRL path, inspect the output directory
for checkpoint folders and restart from the trainer-supported checkpoint path
once that is wired into the config/command. Do not overwrite a run directory you
intend to resume.

For rollout collection, prefer configs with `output.write_mode: append` for long
API-backed jobs that may be restarted. Use `overwrite` only for deliberate smoke
tests.

## Local Validation Results

Validated locally without a GPU on May 7, 2026 using Python 3.10.16.

Passed:

- `bash -n scripts/washu_setup.sh scripts/washu_run_smoke.sh scripts/washu_run_iteration.sh`
- `bash scripts/washu_setup.sh --dry-run`
- `bash scripts/washu_run_smoke.sh --dry-run`
- `bash scripts/washu_run_smoke.sh`
- `bash scripts/washu_run_iteration.sh --validate-only --output-dir /tmp/self_play_washu_agent8_iter_validate --overwrite`
- `python3.10 -m grpo_math.self_play.run_self_play_loop --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml --dry-run --iterations 1 --output-dir /tmp/self_play_washu_agent8_loop --overwrite`
- `python3.10 -m pytest tests/test_collect_self_play_rollouts.py tests/test_train_solver_verdict_grpo.py tests/test_train_proposer_question_grpo.py tests/test_run_self_play_iteration.py tests/test_run_self_play_loop.py`

The targeted pytest run collected 23 tests and all 23 passed.

Observed smoke outputs:

- Fixture collector wrote 5 rollout rows: 4 accepted questions and 1 rejected
answer-leak question.
- Rollouts were written to `outputs/self_play_fixture/rollouts.jsonl`.
- Rejected questions were written to
`outputs/self_play_fixture/rejected_questions.jsonl`.
- Proposer validation reported acceptance rate `0.8`, duplicate rate `0.0`,
mean question reward `0.6`, and zero-reward rate `0.2`.
- Iteration and loop dry-runs wrote reports and `next_iteration.yaml` files in
their output directories.

Remaining expected gaps on this local machine:

- No CUDA/GPU is available locally, so actual GRPO training was not executed.
- `torch` is not installed in the local Python used for no-GPU validation; this
is expected before running `bash scripts/washu_setup.sh` on the server.
- API-backed rollout smoke was not executed because it would spend external API
budget and requires the intended provider keys on the server.
- The evaluation stage is command-planned in iteration dry-runs; it should be
run after a real solver checkpoint exists.

## First Real Server Run

1. `bash scripts/washu_setup.sh --dry-run`
2. `bash scripts/washu_setup.sh`
3. `bash scripts/washu_run_smoke.sh --dry-run`
4. `bash scripts/washu_run_smoke.sh`
5. Run an API-backed rollout smoke with the chosen provider key.
6. Run one full iteration with `bash scripts/washu_run_iteration.sh --dry-run`.
7. Run the full iteration for real.
8. Inspect `report.md`, rollout JSONL, checkpoints, acceptance rates, zero-reward
  rates, verifier verdict distributions, and W&B curves.
9. Launch multi-iteration runs only after the smoke artifacts look sane.

