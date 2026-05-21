#!/bin/bash
#SBATCH --job-name=qwen35-vllm-smoke
#SBATCH --account=engr-lab-jiaxinh
#SBATCH --partition=condo-jiaxinh
#SBATCH --gres=gpu:a100-sxm4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/qwen35_vllm_smoke_%j.out
#SBATCH --error=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/qwen35_vllm_smoke_%j.err

set -euo pipefail

cd /engrfs/project/jiaxinh/dylan_work/self-play
source .venv-qwen-slurm/bin/activate

mkdir -p \
  outputs/slurm \
  .cache/xdg \
  .cache/huggingface/hub \
  .cache/huggingface/transformers \
  .cache/torch \
  .cache/hf-assets \
  .jobhome

JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER}/self-play-${SLURM_JOB_ID}}"
mkdir -p "$JOB_TMP/tmp" "$JOB_TMP/triton" "$JOB_TMP/torchinductor" "$JOB_TMP/cuda"

export HOME="$PWD/.jobhome"
export TMPDIR="$JOB_TMP/tmp"
export TMP="$JOB_TMP/tmp"
export TEMP="$JOB_TMP/tmp"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export TRITON_CACHE_DIR="$JOB_TMP/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor"
export TORCH_HOME="$PWD/.cache/torch"
export CUDA_CACHE_PATH="$JOB_TMP/cuda"
export HF_ASSETS_CACHE="$PWD/.cache/hf-assets"
export COMPILER_PATH="${COMPILER_PATH:-/usr/bin}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-math}"

PORT="${VLLM_PORT:-8001}"
VLLM_LOG="outputs/slurm/vllm_${SLURM_JOB_ID}.log"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.82 \
  --enforce-eager \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

python - <<'PY'
import time
import urllib.request

url = "http://127.0.0.1:8001/v1/models"
last = ""
for idx in range(180):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(f"[wait] vLLM ready: HTTP {resp.status}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = f"{type(exc).__name__}: {exc}"
        print(f"[wait] vLLM not ready {idx + 1}/180: {last}", flush=True)
        time.sleep(5)
print(f"[wait] vLLM failed to become ready: {last}", flush=True)
raise SystemExit(1)
PY

export CUDA_VISIBLE_DEVICES=1

python -m grpo_math.self_play.run_self_play_grpo_loop \
  --rollout_config grpo_math/configs/pairwise_rollouts_local4b_single_verify_loop_smoke.yaml \
  --question_train_config grpo_math/configs/train_question_qwen35_loop_smoke.yaml \
  --train_config grpo_math/configs/train_pairwise_verdict_qwen25_05b_external_teacher_loop_smoke.yaml \
  --cycles 2 \
  --run_tag qwen35_vllm_one_group_high_temp_sparse_smoke \
  --max_steps 1 \
  --max_train_samples 4 \
  --max_eval_samples 1 \
  --log_rollouts_to_wandb
