#!/bin/bash
#SBATCH --job-name=q35-dpo-pair-eval
#SBATCH --account=engr-lab-jiaxinh
#SBATCH --partition=condo-jiaxinh
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_dpo_pair_eval_%j.out
#SBATCH --error=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_dpo_pair_eval_%j.err

set -euo pipefail
cd /engrfs/project/jiaxinh/dylan_work/self-play
source .venv-qwen-slurm/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

RUN_TAG="${RUN_TAG:-qwen35_question_dpo_only_8x8_vllmlora}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
CONFIG="${CONFIG:-grpo_math/configs/train_question_qwen35_overnight.yaml}"
RUN_DIR="outputs/self_play_grpo_loop/${RUN_TAG}"
OUT_ROOT="outputs/dpo_pair_eval/${RUN_TAG}"

mkdir -p outputs/slurm "${OUT_ROOT}" .cache/xdg .cache/huggingface/hub .cache/huggingface/transformers .cache/torch .cache/hf-assets .jobhome
JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER}/self-play-dpo-eval-${SLURM_JOB_ID}}"
mkdir -p "$JOB_TMP/tmp" "$JOB_TMP/triton" "$JOB_TMP/torchinductor" "$JOB_TMP/cuda"
export HOME="$PWD/.jobhome"
if [ -f "/home/compute/${USER}/.netrc" ]; then cp "/home/compute/${USER}/.netrc" "$HOME/.netrc"; chmod 600 "$HOME/.netrc"; fi
export TMPDIR="$JOB_TMP/tmp" TMP="$JOB_TMP/tmp" TEMP="$JOB_TMP/tmp"
export XDG_CACHE_HOME="$PWD/.cache/xdg" HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export TRITON_CACHE_DIR="$JOB_TMP/triton" TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor"
export TORCH_HOME="$PWD/.cache/torch" CUDA_CACHE_PATH="$JOB_TMP/cuda"
export HF_ASSETS_CACHE="$PWD/.cache/hf-assets"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PREV_CKPT=""
for samples in "${RUN_DIR}"/cycle_*_samples.jsonl; do
  [ -f "${samples}" ] || continue
  cycle="$(basename "${samples}" _samples.jsonl)"
  qdir="${RUN_DIR}/${cycle}_question_grpo"
  if [ ! -d "${qdir}" ]; then
    echo "skip ${cycle}: no question_grpo dir"
    continue
  fi
  ckpt=""
  if [ -d "${qdir}/checkpoint-25" ]; then
    ckpt="${qdir}/checkpoint-25"
  else
    ckpt="$(ls -d "${qdir}"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  fi
  if [ -z "${ckpt}" ] || [ ! -d "${ckpt}" ]; then
    echo "skip ${cycle}: no checkpoint"
    continue
  fi
  pair_dir="${OUT_ROOT}/${cycle}"
  mkdir -p "${pair_dir}"
  REF_ARGS=()
  if [ -n "${PREV_CKPT}" ]; then
    REF_ARGS=(--ref-checkpoint "${PREV_CKPT}")
  fi
  echo "=== eval ${cycle} ckpt=${ckpt} ref=${PREV_CKPT:-<none>} ==="
  python3 scripts/eval_question_dpo_pairs.py \
    --config "${CONFIG}" \
    --rollout-jsonl "${samples}" \
    --checkpoint "${ckpt}" \
    "${REF_ARGS[@]}" \
    --eval-fraction "${EVAL_FRACTION}" \
    --save-pairs-dir "${pair_dir}" \
    --output-json "${pair_dir}/metrics.json"
  PREV_CKPT="${ckpt}"
done

echo "done -> ${OUT_ROOT}"
