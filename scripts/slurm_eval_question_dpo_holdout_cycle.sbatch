#!/bin/bash
# Evaluate question-DPO checkpoints on a held-out *cycle* rollout (never used in that ckpt's training).
# Example: train/eval ckpts from cycles 001-004 on preference pairs built from cycle_005_samples.jsonl.
#SBATCH --job-name=q35-dpo-holdout
#SBATCH --account=engr-lab-jiaxinh
#SBATCH --partition=condo-jiaxinh
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_dpo_holdout_cycle_%j.out
#SBATCH --error=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_dpo_holdout_cycle_%j.err

set -euo pipefail
cd /engrfs/project/jiaxinh/dylan_work/self-play
source .venv-qwen-slurm/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

RUN_TAG="${RUN_TAG:-qwen35_question_dpo_only_8x8_vllmlora}"
HELDOUT_CYCLE="${HELDOUT_CYCLE:-cycle_005}"
# Evaluate checkpoints with cycle number strictly less than held-out (e.g. 001-004 for holdout 005).
MAX_TRAIN_CYCLE_NUM="${MAX_TRAIN_CYCLE_NUM:-4}"
EVAL_FRACTION="${EVAL_FRACTION:-1.0}"
CONFIG="${CONFIG:-grpo_math/configs/train_question_qwen35_overnight.yaml}"
RUN_DIR="outputs/self_play_grpo_loop/${RUN_TAG}"
HELDOUT_JSONL="${RUN_DIR}/${HELDOUT_CYCLE}_samples.jsonl"
OUT_ROOT="outputs/dpo_pair_eval/${RUN_TAG}/holdout_${HELDOUT_CYCLE}"

if [ ! -f "${HELDOUT_JSONL}" ]; then
  echo "missing held-out rollout: ${HELDOUT_JSONL}" >&2
  exit 1
fi

mkdir -p outputs/slurm "${OUT_ROOT}" .cache/xdg .cache/huggingface/hub .cache/huggingface/transformers .cache/torch .cache/hf-assets .jobhome
JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER}/self-play-dpo-holdout-${SLURM_JOB_ID}}"
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

echo "holdout=${HELDOUT_JSONL} max_train_cycle=${MAX_TRAIN_CYCLE_NUM} eval_fraction=${EVAL_FRACTION}"

PREV_CKPT=""
for samples in "${RUN_DIR}"/cycle_*_samples.jsonl; do
  [ -f "${samples}" ] || continue
  cycle="$(basename "${samples}" _samples.jsonl)"
  cycle_num="${cycle##cycle_}"
  cycle_num=$((10#${cycle_num}))
  if [ "${cycle_num}" -gt "${MAX_TRAIN_CYCLE_NUM}" ]; then
    echo "skip ${cycle}: cycle_num ${cycle_num} > max ${MAX_TRAIN_CYCLE_NUM}"
    continue
  fi
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
  echo "=== holdout eval train_cycle=${cycle} ckpt=${ckpt} rollout=${HELDOUT_JSONL} ref=${PREV_CKPT:-<none>} ==="
  python3 scripts/eval_question_dpo_pairs.py \
    --config "${CONFIG}" \
    --rollout-jsonl "${HELDOUT_JSONL}" \
    --checkpoint "${ckpt}" \
    "${REF_ARGS[@]}" \
    --eval-fraction "${EVAL_FRACTION}" \
    --save-pairs-dir "${pair_dir}" \
    --output-json "${pair_dir}/metrics.json"
  PREV_CKPT="${ckpt}"
done

echo "done -> ${OUT_ROOT}"
