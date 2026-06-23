#!/bin/bash
export STORAGE_PATH=/home/nvidia/rzero_run
export FSDP_OFFLOAD=true S_GPU_MEM=0.55 N_SVC=2 CVBAND=0
export ITER=1 EXP=iter1clean ARM=verified VERIFY_WEIGHT=0.75 DIVERSITY_WEIGHT=0.5
export NUM_SAMPLES=1000 EVAL_N=9 Q_STEPS=6 S_STEPS=20
BASE=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export QUESTIONER_MODEL=$BASE
export SOLVER_MODEL=$BASE
export HF_TOKEN=$HF_TOKEN
echo "=== ITER1CLEAN START $(date) (base->iter1 positive control, fresh verified questioner) ==="
bash /home/nvidia/R-Zero/scripts/iteration_rzero.sh
echo "=== ITER1CLEAN ITER rc=$? $(date) ==="
CKPT=$(ls -d /home/nvidia/rzero_run/models/iter1clean_it1_verified_s/global_step_*/actor/huggingface 2>/dev/null | tail -1)
if [ -d "$CKPT" ]; then
  for fcfg in preprocessor_config.json video_preprocessor_config.json; do
    [ -f "$CKPT/$fcfg" ] || cp $BASE/$fcfg "$CKPT/" 2>/dev/null
  done
  echo "=== ITER1CLEAN EVAL (ckpt=$CKPT) ==="
  bash /home/nvidia/rzero_run/run_oly.sh "$CKPT" iter1clean_8k 8192
else echo "NO ITER1CLEAN SOLVER CHECKPOINT"; fi
echo "=== ITER1CLEAN A/B: base=0.573 orig_iter1=0.630 (target) ==="
/home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py iter1clean_8k 2>/dev/null
echo "=== ITER1CLEAN PIPELINE DONE $(date) ==="
