# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Move fast — time is the binding constraint (IMPORTANT)

There is real-world time pressure on this work that the model does not feel intuitively, so make it explicit: **optimize for wall-clock, move as fast as possible.** Concretely:
- Treat slow runs/iterations as bugs to fix, not facts to accept. A pipeline running far below hardware capability (e.g. generation at concurrency ~1 vs the GPU's batched ceiling, or training stages 10–60× slower than their FLOP budget) is a defect — diagnose and fix it, don't wait it out.
- Prefer the fastest correct path: parallelize (batch generation, data-parallel replicas, concurrent tool calls/agents), validate fixes with cheap smokes before committing multi-hour runs, and keep idle GPUs working.
- When estimating, always give wall-clock ETAs and flag when something will take hours/days so the user can decide. Don't silently let a 15-hour job run when a 1-hour path exists.
- Speed up infrastructure (kernels, batching, offload, micro-batching, parallelism), **never** by changing the experiment (token lengths, sample counts `n`, training steps, reward, sampling) — those must stay intact. Near-equivalent kernel/attention swaps are fine; learning-altering shortcuts are not.

### Fast debug loop (smokes)
- **Smokes must be tiny.** Validate that a config *path runs* with the smallest possible footprint — a handful of rows, batch ~8–16, 1 step. Don't smoke at production batch/row counts; that wastes minutes per attempt. Measure the *real* per-step time on the actual run's first step, not a heavyweight standalone smoke.
- **Don't relaunch vLLM / reload the model if you don't have to.** Each verl/vLLM start pays a ~1–2 min model-load tax. Batch all your config fixes and test once; don't kill-and-reload for each single change. If a job is already loaded and running, let it return the number rather than killing it to start a "cleaner" one.
- Anticipate failure modes before launching (missing deps, deleted data files, batch>rows → empty dataloader, enforce_eager/cudagraph on unsupported GPUs, OOM from grad-ckpt-off) so one launch validates many things at once.

### Pull regularly and verify runs are ACTUALLY working (don't trust "launched")
- **"Launched" ≠ "working."** Within the first ~2–3 min of any launch, confirm it is genuinely progressing — GPU utilization > 0, the log advancing, a step/loss/metric actually moving — not just that the tmux/process exists. Jobs routinely crash seconds after start (missing file, empty dataloader, OOM, engine-init failure) while looking "up." (Lost ~3h once by assuming a launched smoke was training when it had crashed instantly on a deleted parquet.)
- **Pull throughout the run, at every stage, not just at launch.** Re-check at each danger zone and stage transition (env load, first step, checkpoint save, stage handoff, eval). A watcher that only fires once is not enough — re-arm it for the next milestone and keep checking until the run produces its result.
- **Rigorously verify, don't pattern-match.** Read the actual numbers (step time, GPU util, loss, format/acc metrics). Confirm a stage produced its expected artifact before assuming success. If a watcher exits without a decisive result, investigate immediately rather than waiting.
- Keep the cadence tight while the user is away: short per-poll SSH (long-held connections drop), local background watchers that re-invoke on milestone/failure, and surface status proactively.

### Monitoring cadence + auto-diagnose (HARD RULE — protects research velocity)
**One standing health loop covers ALL active runs (every cluster), diagnoses failures, and relaunches them autonomously — never monitor a single run in isolation while others drift.**

Two-tier cadence (don't churn a fixed short loop across a 20h run):
- **Danger zones — check every 60–90s:** the first ~5 min after any launch, and around every stage transition (env load, first step, checkpoint save, A→B handoff, eval start). Catch crashes/hangs here in minutes.
- **Steady state — heartbeat AT LEAST hourly:** once a run is confirmed progressing, sweep all runs on a ~hourly heartbeat (and a light issue-check every few min that only escalates on a problem). Don't re-invoke every 8 min for 20h; do guarantee a checkup never goes more than ~1h.
- **Re-invoke on event, not just clock:** immediately on (a) stage advanced, (b) error/traceback/OOM, (c) process/tmux died, (d) **STALL** = all GPUs idle (<~10%) AND log mtime/step frozen ~3 min. Otherwise heartbeat hourly.

Stall detection (the velocity-killer to prevent): track the metric that should move (step #, log mtime, processed-prompts count) across polls. "Still `0/N` for several minutes" with **idle GPUs** = hang → act now, don't wait it out. Caveat: a busy phase (reward generation) shows GPU≈100% with the main log quiet — that's NOT a stall; stall = idle GPUs + no progress together.

**Stall detection for AUTO-RESTART must check ALL log sources + a LONG (~20min) window — false positives are destructive.** Benchmark/eval phases run on 1 GPU at intermittent util AND write to *separate* logs (`oly_cont.log`/`m500_cont.log`/RUN log), so the main training log looks frozen — a naive "idle GPU + main-log-frozen" detector FALSE-restarts a healthy *finishing* run. Since the restart wipes checkpoints to avoid resume-hang, a false positive is catastrophic: we once false-restarted a just-completed iteration mid-eval and lost ~4h of solver training (the eval *numbers* were luckily already on disk). Rule: liveness = **max mtime across every log the pipeline writes**; only restart when GPU idle AND **all** logs frozen for **~20min** (real hangs were 27min+ fully dead). Never restart on a short single-log freeze.

Auto-diagnose + relaunch: when a run trips, pull the real traceback/log, identify root cause (OOM / config / hang / data), apply the mechanical fix, and relaunch — then verify the relaunch through startup per above. Record each failure+cause+fix in PROJECT_NOTES.

## Training launches MUST capture rollouts (verify before/at launch — IMPORTANT)

Every training launch must log curves to **wandb** AND dump **actual rollout text** (prompt + response + ground-truth + score, per step) to disk — this is the standard-experiment-logging discipline and is non-negotiable: without it, post-hoc divergence/reward-hack investigations have nothing to read and degrade into reverse-engineering metrics from reward-service scraps. The R-Zero pipeline (`scripts/iteration_rzero.sh`) silently overrode this with `trainer.logger=[console]` and no dump, so a whole iter2 divergence investigation had no rollout text to mine. Before (or at) any verl launch, confirm these are set on BOTH the questioner and solver `main_ppo.py` invocations:
- `trainer.logger=[console,wandb]` (wandb auth is persisted in `~/.netrc` on Brev — no prompt/hang; verify `.netrc` has `api.wandb.ai` before assuming).
- `trainer.rollout_data_dir=$STORAGE_PATH/rollout_dumps/<experiment_name>` — dumps training rollout text every step (verl `_log_rollout_data`→`_dump_generations`, `ray_trainer.py:1681`). **This is the one that's usually missing.**
- `trainer.validation_data_dir=…` + `trainer.log_val_generations=20` — saves eval generations.
These are pure logging side-channels — zero effect on the experiment (no token/n/step/reward change). Patched into Brev `iteration_rzero.sh` (backup `.bak_logging`); the same edit is staged for any multi-iter Cornell launch. **Caveat:** can't edit `iteration_rzero.sh` mid-iteration (bash re-reads by byte offset → corruption); apply only when `fuser` shows it unopened (between iterations / at a boundary).

## Project notes (keep updated)

`PROJECT_NOTES.md` at the repo root is the running experiment log. **At the end of any significant work session** — experiment runs, debugging investigations, root-cause analyses, config changes, training/eval launches — append a dated entry (newest first) covering:

- What was done and why
- Key findings with actual numbers (metrics, CIs, job IDs)
- Root causes identified and what was ruled out
- Fixes applied, with exact file paths (note whether applied locally, on the server, or both)
- Pending follow-ups as a checklist

Read `PROJECT_NOTES.md` at the start of a session when context about prior experiments is needed. Update existing entries' checklists when pending items complete rather than duplicating them.

## Compute environment (as of 2026-06-17)

Training has **migrated off the shared Brev A100 box to Empire AI** (Cornell) for dedicated multi-GPU: `ssh empire`, `cornell` partition (8× H100-80GB nodes, 7-day walltime). Env: `~/venvs/selfplay` (Python 3.10.15; torch 2.10/cu128, transformers 5.8.0, vllm 0.19.1, trl 1.4.0 — Qwen3.5 needs transformers 5.x). **Always** `export LD_LIBRARY_PATH=/mnt/home/software/software/Python/3.10.15-system/lib:$LD_LIBRARY_PATH` before using the venv (non-interactive shells don't have the module's libpython). Code at `~/self-play`; scratch `/mnt/lustre/cornell/ch2263`. The Brev box (GPUs 4–7) has been released back to R-Zero. The multi-GPU win is the R-Zero lever: vLLM-backed generation + bigger batch (see PROJECT_NOTES); single-A100 colocate maxed at ~1024 tokens / 16-prompt batch.

## Evaluation methodology — ALWAYS decompose format vs accuracy (IMPORTANT)

A raw pass-rate (strict `FINAL_ANSWER: <int>` match) **conflates two things**: whether the model *emitted a parseable answer in budget* (format_rate) and whether that answer was *correct* (accuracy-given-format). On GSM-hard especially, completions are long (big-number arithmetic) and frequently truncate before the `FINAL_ANSWER` line — so a "pass" gain can be pure **formatting/conciseness** (learning to finish in budget), NOT improved reasoning.

**Every eval result MUST report all three:** `format_rate` (fraction emitting a parseable `FINAL_ANSWER`), `accuracy | formatted` (correct among those that emitted), and overall `pass` (= format_rate × acc-given-format). Never report pass alone — it hides whether the gain is reasoning or formatting. Concretely observed on the mid-band GSM-hard heldout: base @1024 was format_rate 0.15 / acc|fmt 1.00, @2048 format_rate 0.49 / acc|fmt 0.97 — i.e. the difficulty was almost entirely truncation, and pass gains were largely format_rate, not accuracy. Use `acc | formatted` as the real reasoning metric; watch `format_rate` separately. Also always report the `max_tokens`/budget the eval ran at (it dominates format_rate) and the k / n.

## Commands

**Lint:**

```bash
ruff check .
```

**Tests:**

```bash
python -m pytest tests/test_reward.py
python -m pytest tests/test_logprobs.py
python -m pytest tests/test_left_padding_masking.py
```

**Pairwise rollout (smoke test):**

```bash
python -m grpo_math.self_play.generate_pairwise_data --config grpo_math/configs/pairwise_rollouts_smoke.yaml
```

**Debug rollout with reset:**

```bash
python tests/debug_single_pairwise_rollout.py --config grpo_math/configs/pairwise_rollouts_debug.yaml --reset-output
```

**Training:**

```bash
# Single GPU
python -m grpo_math.trl.train_grpo_trl --config grpo_math/configs/train_gsm8kv2_trl.yaml

# Multi-GPU
accelerate launch -m grpo_math.trl.train_grpo_trl --config grpo_math/configs/train_gsm8kv2_trl.yaml
```

**Evaluation:**

```bash
python -m grpo_math.eval.eval_gsm8k --config grpo_math/configs/train_gsm8kv2_trl.yaml --checkpoint <path> --max_samples 1000 --k 4
```

## Architecture

This is a self-improving training loop for math reasoning with three roles: **Generator** (creates problems), **Solver** (generates solutions), and **Judge** (evaluates quality). The goal is iterative solver improvement via pairwise preference data, where the generator is rewarded for producing *reliably evaluable* problems (not just hard ones).

### Core loop (`grpo_math/self_play/generate_pairwise_data.py`)

The main orchestration file (~1300 lines). Each rollout:

1. Samples questions from the generator model
2. Generates K solutions per question in configurable sampling groups (different temperatures/models to create quality variation)
3. Judges solution pairs in one of two modes:
  - `pairwise` — judge compares A vs B directly, outputs `PREFERENCE:`
  - `single_verify` — judge grades each solution individually (`VERDICT: CORRECT/INCORRECT`), with optional `CONFIDENCE:` tracing
4. Computes Elo ratings from comparison outcomes
5. Saves results to JSONL (append or overwrite)

The judge can be a local HuggingFace model or any OpenAI-compatible API (GPT-4, Gemini, etc.). API calls are parallelized with exponential backoff.

### Reliability metrics (see `README.md`)

Three metrics gate which questions enter training:

- **R_cons**: score consistency — low variance across repeated judging
- **R_sep**: strong/weak separation — better sampling regimes should win more often
- **R_stab**: preference stability — pairwise judgments agree on re-run

### Reward and format (`grpo_math/data/reward.py`)

All solver outputs must end with `FINAL_ANSWER: <integer>` (strict). Two reward signals:

- **Correctness**: `binary_reward()` via `extract_final_answer_int_strict()`
- **Format**: presence of the `FINAL_ANSWER:` tag

### Training (`grpo_math/trl/train_grpo_trl.py`)

Uses TRL's `GRPOTrainer` on GSM8K. Supports DeepSpeed ZeRO-2/3 (configs in `grpo_math/configs/deepspeed_*.json`). Default model: Qwen2.5-7B-Instruct.

### Configuration

Everything is YAML-driven. Rollout configs live in `grpo_math/configs/pairwise_rollouts_*.yaml`; training configs in `grpo_math/configs/train_*.yaml`. Prompt templates are in `grpo_math/prompts/` and referenced from configs with `{question}`, `{answer_a}`, `{solution}` placeholders.

### Outputs

Rollout results are written to `outputs/<experiment_name>/` as JSONL. Each record contains the question, all solutions, all pairwise comparison outcomes, and Elo ratings.