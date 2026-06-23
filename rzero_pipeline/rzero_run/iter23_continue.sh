#!/bin/bash
# Continue verified self-play: iters 2 & 3, seeded from iter1 solver (verified-20, OlympiadBench 41.5%).
# Questioner trains each iter (no iter1 questioner existed -> starts from base). Standard verified reward (no clip).
# Evals solver on OlympiadBench + uploads checkpoints/questions to HF after each iter.
set -uo pipefail
export PATH=/home/nvidia/venvs/rzero2/bin:$PATH
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export FSDP_OFFLOAD=true
export S_GPU_MEM=0.55
export N_SVC=2
export STORAGE_PATH=/home/nvidia/rzero_run
RZERO=/home/nvidia/R-Zero
PYB=/home/nvidia/venvs/rzero2/bin/python
S=/home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface   # iter1 solver (41.5%)
Q=/home/nvidia/rzero_run/models/clip80_it1_verified_q/global_step_6/actor/huggingface
echo "=== iter23 continue START $(date) S=$S ==="
for it in 2 3 4; do
  L=$STORAGE_PATH/logs/cont_verified_it${it}.log
  echo "==== ITER $it START $(date) Q=$Q S=$S -> $L"
  ITER=$it EXP=cont ARM=verified VERIFY_WEIGHT=0.75 DIVERSITY_WEIGHT=0.4 VERIFY_SUBSAMPLE=1.0 \
    QUESTIONER_MODEL="$Q" SOLVER_MODEL="$S" Q_STEPS=6 S_STEPS=20 NUM_SAMPLES=1000 EVAL_N=9 \
    bash $RZERO/scripts/iteration_rzero.sh > $L 2>&1
  echo "==== ITER $it rc=$? $(date)"
  NQ=$(grep "^QUESTIONER_HF=" $L | tail -1 | cut -d= -f2-)
  NS=$(grep "^SOLVER_HF=" $L | tail -1 | cut -d= -f2-)
  [ -n "$NQ" ] && Q="$NQ"; [ -n "$NS" ] && S="$NS"
  echo "==== ITER $it newQ=$Q newS=$S"
  # eval solver on OlympiadBench
  bash $STORAGE_PATH/brev_eval.sh "$S" cont_it${it}_oly olympiad 200 8192 1 0 >> $STORAGE_PATH/oly_cont.log 2>&1
  echo "==== ITER $it OLYMPIAD: $(grep RESULT $STORAGE_PATH/oly_cont.log | tail -1)"
  bash $STORAGE_PATH/brev_eval.sh "$S" cont_it${it}_m500 math500 200 8192 1 0 >> $STORAGE_PATH/m500_cont.log 2>&1
  echo "==== ITER $it MATH500: $(grep RESULT $STORAGE_PATH/m500_cont.log | tail -1)"
  WANDB_API_KEY=$WANDB_API_KEY $PYB $STORAGE_PATH/wandb_log_iter.py $it $STORAGE_PATH/oly_cont.log $STORAGE_PATH/m500_cont.log $STORAGE_PATH/artifacts/cont_it${it}_verified/challenger_batches.md 2>&1 | tail -1
  # upload solver + questions to HF
  $PYB -c "
from huggingface_hub import HfApi
a=HfApi()
rid='Dylan1631/selfplay-verified-qwen35-4b-iter${it}'
a.create_repo(rid,repo_type='model',exist_ok=True)
a.upload_folder(folder_path='$S',repo_id=rid,repo_type='model',commit_message='verified iter${it} solver')
a.upload_folder(folder_path='$STORAGE_PATH/generated_question',repo_id='Dylan1631/selfplay-iter1-data',repo_type='dataset',allow_patterns=['*cont_it${it}*'],commit_message='iter${it} questions')
print('HF_UPLOADED iter${it}')
" 2>&1 | tail -2
  # disk hygiene: drop FSDP .pt shards (keep HF)
  find $STORAGE_PATH/models/cont_it${it}_* -name "*.pt" -delete 2>/dev/null
  df -h / | tail -1
done
echo "=== iter23 continue DONE $(date) ==="
