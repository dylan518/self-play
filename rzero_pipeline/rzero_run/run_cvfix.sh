#!/bin/bash
export STORAGE_PATH=/home/nvidia/rzero_run
export FSDP_OFFLOAD=true S_GPU_MEM=0.55 N_SVC=2 CVBAND=1 CVLO=0.2 CVHI=0.8
export ITER=2 EXP=cvfix ARM=verified VERIFY_WEIGHT=0.75 DIVERSITY_WEIGHT=0.5
export NUM_SAMPLES=1000 EVAL_N=9 Q_STEPS=6 S_STEPS=20
export QUESTIONER_MODEL=/home/nvidia/rzero_run/models/clip80_it1_verified_q/global_step_6/actor/huggingface
export SOLVER_MODEL=/home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface
export HF_TOKEN=$HF_TOKEN
echo "=== CVFIX ITER START $(date) (end-to-end CVBAND, Q=clip80_it1_q S=verified-20) ==="
bash /home/nvidia/R-Zero/scripts/iteration_rzero.sh
echo "=== CVFIX ITER rc=$? $(date) ==="
CKPT=$(ls -d /home/nvidia/rzero_run/models/cvfix_it2_verified_s/global_step_*/actor/huggingface 2>/dev/null | tail -1)
if [ -d "$CKPT" ]; then
  for fcfg in preprocessor_config.json video_preprocessor_config.json; do
    [ -f "$CKPT/$fcfg" ] || cp /home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface/$fcfg "$CKPT/" 2>/dev/null
  done
  echo "=== CVFIX EVAL (ckpt=$CKPT) ==="
  bash /home/nvidia/rzero_run/run_oly.sh "$CKPT" cvfix_8k 8192
else
  echo "NO CVFIX SOLVER CHECKPOINT"
fi
echo "=== CVFIX A/B (OlympiadBench full 675 @8k): iter1=0.630 broken_iter2=0.545 cvband_solveronly=0.550 ==="
/home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py cvfix_8k 2>/dev/null
echo "=== CVFIX PIPELINE DONE $(date) ==="
