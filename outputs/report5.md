# Self-Play GRPO Report 5 (Llama 3.1 8B)

**Date:** 2026-02-28  
**System:** Closed-loop self-play rollout + GRPO training  
**Model family:** `meta-llama/Llama-3.1-8B-Instruct` across generator, solver, judge, and policy training

---

## Current Status

The end-to-end experiment is running and stable with decently strong models, including Llama 3.1 8B throughout the full loop.

We have verified the full pipeline:
- live question generation
- multi-solution rollout
- single-solution verification voting
- GRPO training
- W&B logging of rollout artifacts (JSONL + per-question markdown exports)

---

## Reliability and Voting Findings

We observe high reliability when verifier votes are unanimous. In current configured runs, unanimous outcomes show roughly **95% consistency** and are strongly correlated with correctness.

Important caveat: this correlation is strongest only when the system is configured correctly end-to-end (prompting, parse format, verifier settings, and reward mapping). If the setup is off, verdicts can look good in isolation while the GRPO training signal degrades.

---

## Judge Setup (Current Best)

Judge temperature materially affects verification quality. In current testing, **`temperature: 0.4`** gives the best accuracy for the 3-vote setting.

Verifier voting has been updated to 3 votes per solution in:
- `grpo_math/configs/pairwise_rollouts_llama31_8b_vllm_single_verify_fast.yaml`
- `judge.repeats_per_solution: 3`

---

## What We Have Run So Far

1. Validated one full 8B end-to-end cycle with local vLLM and GRPO connected.
2. Confirmed rollout artifacts are uploaded to W&B and include generated question markdowns.
3. Confirmed non-degenerate GRPO behavior (training signal present) after fixing parser/reward path and generation length settings.
4. Validated two-GPU operating mode:
   - GPU 1: vLLM server for rollout + judge API
   - GPU 0: GRPO training loop

---

## How To Run

### 1) Start local vLLM (GPU 1)

```bash
export CUDA_VISIBLE_DEVICES=1
. .venv/bin/activate
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization 0.45
```

### 2) Run self-play + GRPO loop (GPU 0)

```bash
set -a && . .env && set +a
export WANDB_API_KEY="${WANDB_API_KEY:-$WANDBKEY}"
export CUDA_VISIBLE_DEVICES=0
. .venv/bin/activate
python -m grpo_math.self_play.run_self_play_grpo_loop \
  --rollout_config grpo_math/configs/pairwise_rollouts_llama31_8b_vllm_single_verify_fast.yaml \
  --train_config grpo_math/configs/train_pairwise_verdict_llama31_8b_lora_local_teacher_smoke_trl.yaml \
  --cycles 4 \
  --max_steps 1 \
  --max_train_samples 8 \
  --max_eval_samples 2 \
  --log_rollouts_to_wandb
```

---

## TODO (Next)

- See what maximum stable throughput/speed we can achieve.
- Examine signal biases (verdict bias, parse bias, unanimity bias, sampling-group effects).
- Run with only solver-side GRPO and compare signal quality vs the full loop setup.

---

## W&B Artifact Note

Rollout artifacts (JSONL and per-question markdown files) are uploaded under rollout artifacts. Generated questions are in:
- `cycle_XXX/markdown_exports/question_*.md`
