#!/bin/bash
# Agentic-bank solver GRPO (7-GPU, 8192-token rollouts) -> per-checkpoint merge + GSM8K eval
# -> markdown run report. Assumes outputs/agentic_bank/bank.jsonl exists.
set -euo pipefail

cd "$HOME/self-play"
source "$HOME/venvs/selfplay/bin/activate"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export PROJECT_ROOT="$PWD"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${WANDBKEY:-}" ]; then
  export WANDB_API_KEY="$WANDBKEY"
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || export WANDB_MODE=offline
else
  export WANDB_MODE=offline
fi
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-math}"
export WANDB_LOG_MODEL=checkpoint

OUT_DIR="outputs/trl_grpo_agentic_correctness_qwen35_9b_8k7g"
REPORT="outputs/agentic_run_report.md"

mkdir -p outputs/slurm
exec > >(tee -a outputs/slurm/agentic_run.log) 2>&1

{
  echo "# Agentic GRPO run $(date -u +%Y-%m-%dT%H:%MZ)"
  echo
  echo "- Bank: outputs/agentic_bank/bank.jsonl (agent-generated, code-judge references)"
  echo "- Config: grpo_math/configs/train_agentic_correctness_qwen35.yaml"
  echo "- Rollouts: 8192 tokens (matches bank calibration), k=8, 7 GPUs x micro-batch 1 x grad_accum 8 = 7 question groups/step, 60 steps"
  echo "- Reward: correctness vs code-judge reference (+0.2 format, +0.1 boundary)"
  echo
  echo "## GSM8K evals (500q, k=4, reason-first prompt, temp 1.1)"
  echo
  echo "| checkpoint | mean_reward | format_rate | pass@1 | pass@4 |"
  echo "|---|---|---|---|---|"
  echo "| base Qwen3.5-9B | 0.4570 | 0.6070 | 0.4760 | 0.8600 |"
} > "$REPORT"

echo "==== [1/3] solver GRPO training $(date -u) ===="
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes 7 --mixed_precision bf16 \
  -m grpo_math.trl.train_grpo_trl \
  --config grpo_math/configs/train_agentic_correctness_qwen35.yaml

echo "==== [2/3] per-checkpoint merge + GSM8K eval $(date -u) ===="
eval_ckpt() {
  local CKPT="$1" GPU="$2"
  local STEP="${CKPT##*-}"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/merge_lora_adapter.py \
    --base Qwen/Qwen3.5-9B --adapter "$CKPT" --out "$OUT_DIR/merged_step$STEP" \
    > "outputs/slurm/eval_step$STEP.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python -m grpo_math.eval.eval_gsm8k \
    --config grpo_math/configs/eval_gsm8k_agentic.yaml \
    --checkpoint "$OUT_DIR/merged_step$STEP" \
    --max_samples 500 --k 4 >> "outputs/slurm/eval_step$STEP.log" 2>&1
  local LINE
  LINE=$(grep -oE "mean_reward=[0-9.]+ format_rate=[0-9.]+ pass@1=[0-9.]+ pass@4=[0-9.]+" "outputs/slurm/eval_step$STEP.log" | tail -1)
  echo "| step $STEP | $(echo "$LINE" | grep -oE "mean_reward=[0-9.]+" | cut -d= -f2) | $(echo "$LINE" | grep -oE "format_rate=[0-9.]+" | cut -d= -f2) | $(echo "$LINE" | grep -oE "pass@1=[0-9.]+" | cut -d= -f2) | $(echo "$LINE" | grep -oE "pass@4=[0-9.]+" | cut -d= -f2) |" > "outputs/slurm/evalrow_$(printf '%04d' "$STEP").md"
  echo "eval done: $CKPT -> $LINE"
  rm -rf "$OUT_DIR/merged_step$STEP"
}

rm -f outputs/slurm/evalrow_*.md
CKPTS=($(ls -d "$OUT_DIR"/checkpoint-* | sort -t- -k2 -n))
GPUS=(1 3 5)
i=0
for CKPT in "${CKPTS[@]}"; do
  eval_ckpt "$CKPT" "${GPUS[$((i % 3))]}" &
  i=$((i + 1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
cat outputs/slurm/evalrow_*.md >> "$REPORT"

echo "==== [3/3] report $(date -u) ===="
{
  echo
  echo "## Pointers"
  echo "- Training log: outputs/slurm/agentic_run.log"
  echo "- Per-checkpoint eval logs: outputs/slurm/eval_step*.log"
  echo "- Checkpoints: $OUT_DIR/checkpoint-*"
  echo "- wandb: project grpo-math, run trl-grpo-agentic-correctness-qwen35-9b-8k-7gpu (checkpoints uploaded as artifacts via WANDB_LOG_MODEL)"
} >> "$REPORT"
cat "$REPORT"
echo "==== ALL DONE $(date -u) ===="
