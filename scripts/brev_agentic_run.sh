#!/bin/bash
# Agentic question-gen -> solver GRPO (correctness reward vs code-judge refs) -> GSM8K eval.
# Assumes the vLLM server (Qwen/Qwen3.5-9B, port 8001) is already running on GPU0.
# Training/eval run on GPU1. Run under tmux.
set -euo pipefail

cd "$HOME/self-play"
source "$HOME/venvs/selfplay/bin/activate"
export PROJECT_ROOT="$PWD"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

mkdir -p outputs/slurm
exec > >(tee -a outputs/slurm/agentic_run.log) 2>&1

echo "==== [1/4] agentic question generation $(date -u) ===="
python scripts/agentic_question_gen.py --chains 4 --rounds 8 --out outputs/agentic_bank/bank.jsonl

TRAINABLE=$(python - <<'PY'
import json
n = 0
with open("outputs/agentic_bank/bank.jsonl") as f:
    for line in f:
        line = line.strip()
        if line and json.loads(line).get("trainable_for_solver"):
            n += 1
print(n)
PY
)
echo "trainable questions: $TRAINABLE"
if [ "$TRAINABLE" -lt 40 ]; then
  echo "FATAL: too few trainable questions ($TRAINABLE < 40); aborting before training."
  exit 1
fi

echo "==== [2/4] solver GRPO training $(date -u) ===="
CUDA_VISIBLE_DEVICES=1 python -m grpo_math.trl.train_grpo_trl \
  --config grpo_math/configs/train_agentic_correctness_qwen35.yaml

echo "==== [3/4] merge final adapter $(date -u) ===="
CKPT=$(ls -d outputs/trl_grpo_agentic_correctness_qwen35_9b/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "merging $CKPT"
CUDA_VISIBLE_DEVICES=1 python scripts/merge_lora_adapter.py \
  --base Qwen/Qwen3.5-9B \
  --adapter "$CKPT" \
  --out outputs/trl_grpo_agentic_correctness_qwen35_9b/merged_final

echo "==== [4/4] GSM8K eval $(date -u) ===="
CUDA_VISIBLE_DEVICES=1 python -m grpo_math.eval.eval_gsm8k \
  --config grpo_math/configs/eval_gsm8k_agentic.yaml \
  --checkpoint outputs/trl_grpo_agentic_correctness_qwen35_9b/merged_final \
  --max_samples 500 --k 4

echo "==== ALL DONE $(date -u) ===="
