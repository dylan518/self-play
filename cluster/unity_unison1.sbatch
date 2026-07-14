#!/bin/bash
#SBATCH -N1 -n1 -c16
#SBATCH --mem=250G
#SBATCH -t 08:00:00
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:4
#SBATCH -J unison1
#SBATCH -o /work/pi_general_dartmouth_edu/dylan/logs/unison1_%j.log
# CONTINUOUS UNISON LOOP v1: alternate questioner GRPO chunks (pmap_wb reward, 3
# steps/cycle) with solver replay-GRPO updates trained ONLY on harvested Stage-A
# rollouts (zero extra rollout compute). Buffer rule: accumulate across Q-chunks while
# solver frozen (stays on-policy); one solver step at >=MIN_LIVE live groups; flush
# after (older rollouts off-policy once solver moves). Band recomputes vs the UPDATED
# solver each cycle = repeated questions self-deflate out of band (the unison effect).
set -e
D=/work/pi_general_dartmouth_edu/dylan
export HF_HOME=$D/.cache/huggingface
export STORAGE_PATH=$D/unison1
mkdir -p $STORAGE_PATH $D/logs
export DUMP_VERIFY_DIR=$STORAGE_PATH/verify_rollouts
export HARVEST_DIR=$STORAGE_PATH/harvest
export VERIFY_K_PROGRAMS=3 VERIFY_MIN_AGREE=2 VERIFY_SUBSAMPLE=1.0
export PY=$D/envs/rzero/bin/python
export MAIN=$D/envs/rzero/lib/python3.12/site-packages/verl/trainer/main_ppo.py
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$D/envs/rzero/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
MIN_LIVE=${MIN_LIVE:-96}
CYCLES=${CYCLES:-6}
echo "UNISON1_START $(date) node=$(hostname) gpus=$(nvidia-smi -L | wc -l) cycles=$CYCLES min_live=$MIN_LIVE"

# questioner diversity term: within-batch hinged pmap (validated in pmapwb1)
export DIVERSITY_MODE=pmap_wb
export DIVERSITY_EMB_MODEL=unsloth/embeddinggemma-300m EMB_USE_ST=1
export DIVERSITY_WEIGHT=0.5
CAL=$($PY - <<'PYEOF'
import json, sys, os, numpy as np
sys.path.insert(0, "/work/pi_general_dartmouth_edu/dylan/R-Zero/examples/reward_function")
from diversity import embed
D="/work/pi_general_dartmouth_edu/dylan"
gold=[json.loads(l)["question"] for l in open(D+"/golden_set_v5.jsonl")]
G=embed(gold); S=G@G.T; n=len(gold)
nn=[]; mn=[]
for i in range(n):
    r=S[i].copy(); r[i]=-1.0
    nn.append(float(r.max())); mn.append(float((S[i].sum()-S[i,i])/(n-1)))
print(f"{np.mean(nn):.4f} {np.mean(mn):.4f}")
PYEOF
)
export PMAP_TMAX=$(echo $CAL | awk '{print $1}') PMAP_TMEAN=$(echo $CAL | awk '{print $2}')
echo "pmap_wb thresholds: PMAP_TMAX=$PMAP_TMAX PMAP_TMEAN=$PMAP_TMEAN"

QCUR=Qwen/Qwen3.5-4B
SCUR=Qwen/Qwen3.5-4B
cd $D
set +e
for CYCLE in $(seq 1 $CYCLES); do
  echo "=== UNISON CYCLE $CYCLE: Q-chunk (3 GRPO steps) questioner=$QCUR solver=$SCUR $(date) ==="
  COLOCATE=1 Q_STEPS=3 Q_TBATCH=60 Q_MINI=12 \
    ITER=$CYCLE EXP=unison ARM=verified \
    QUESTIONER_MODEL=$QCUR SOLVER_MODEL=$SCUR \
    VERIFY_WEIGHT=0.75 NUM_SAMPLES=50 EVAL_N=9 SKIP_STRAIN=1 \
    bash $D/R-Zero/scripts/iteration_rzero.sh
  RC=$?
  QNEW=$STORAGE_PATH/models/unison_it${CYCLE}_verified_q/global_step_3/actor/huggingface
  if [ -d "$QNEW" ]; then QCUR=$QNEW; else echo "CYCLE $CYCLE: Q ckpt missing (rc=$RC) — keeping $QCUR"; fi

  echo "=== UNISON CYCLE $CYCLE: solver replay check $(date) ==="
  if [ -f $HARVEST_DIR/solver_harvest.jsonl ]; then
    CUDA_VISIBLE_DEVICES=0 $PY $D/solver_replay_grpo.py \
      --harvest $HARVEST_DIR/solver_harvest.jsonl \
      --model $SCUR --out $STORAGE_PATH/solver_c$CYCLE --min_live $MIN_LIVE
    SRC=$?
    if [ $SRC -eq 0 ]; then
      SCUR=$STORAGE_PATH/solver_c$CYCLE
      mv $HARVEST_DIR/solver_harvest.jsonl $HARVEST_DIR/consumed_c${CYCLE}.jsonl
      echo "SOLVER_STEP cycle=$CYCLE -> $SCUR (buffer flushed)"
    else
      echo "SOLVER_WAIT cycle=$CYCLE (rc=$SRC, accumulating)"
    fi
  fi
done
echo "UNISON1_LOOP_DONE questioner=$QCUR solver=$SCUR $(date)"
cp $STORAGE_PATH/artifacts/unison_it*_verified/challenger_batches.md $D/logs/ 2>/dev/null
echo "UNISON1_DONE $(date)"
