#!/bin/bash
#SBATCH -J unison1w4
#SBATCH -A engr-lab-jiaxinh
#SBATCH -p general-gpu
#SBATCH --nodelist=a100s-2306
#SBATCH --gres=gpu:a100-sxm4:4
#SBATCH -N1 -n1 -c16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /home/compute/jiaxinh/unison1w4_%j.log
# WashU race entrant for the CONTINUOUS UNISON LOOP v1 (see unity_unison1.sbatch).
# 2-GPU colocate on a100s-2305 (node-local env). Q-chunks (pmap_wb, 3 steps) x solver
# replay-GRPO on harvested Stage-A rollouts; >=MIN_LIVE live groups -> solver step,
# flush buffer, service swaps to updated solver.
set -e
H=/home/compute/jiaxinh
SC=/scratch/jiaxinh
PYV=$SC/envs/rzero/bin/python
export HF_HOME=$SC/hf HF_DATASETS_CACHE=$SC/hf/datasets TRANSFORMERS_CACHE=$SC/hf
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export WORKER_ENFORCE_EAGER=1
MIN_LIVE=${MIN_LIVE:-96}
CYCLES=${CYCLES:-6}
mkdir -p $SC
ENVDIR=$SC/envs/rzero
export PIP_CACHE_DIR=$SC/pipcache CONDA_PKGS_DIRS=$SC/condapkgs
if ! $ENVDIR/bin/python -c "import torch,vllm,verl,transformers,flashinfer" 2>/dev/null; then
  echo "=== building env on $(hostname) $(date) ==="
  [ -f $SC/mf.sh ] || curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o $SC/mf.sh
  [ -d $SC/mf ] || bash $SC/mf.sh -b -p $SC/mf
  source $SC/mf/etc/profile.d/conda.sh
  [ -d $ENVDIR ] || conda create -y -p $ENVDIR python=3.12
  $ENVDIR/bin/python -m pip -q install --upgrade pip
  $ENVDIR/bin/pip -q install --no-deps --timeout 90 --retries 20 -r $H/freeze_noopencv.txt
  $ENVDIR/bin/python -c "import torch,vllm,verl,transformers,flashinfer; print('ENV_OK')" || { echo ENV_BUILD_FAILED; exit 1; }
fi
echo "UNISON1W_START $(date) node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
$PYV -c "import sentence_transformers" 2>/dev/null || $PYV -m pip -q install sentence_transformers
$PYV -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/embeddinggemma-300m')"
$PYV -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"

export STORAGE_PATH=$SC/unison1
mkdir -p $STORAGE_PATH
export DUMP_VERIFY_DIR=$STORAGE_PATH/verify_rollouts
export HARVEST_DIR=$STORAGE_PATH/harvest
export VERIFY_K_PROGRAMS=3 VERIFY_MIN_AGREE=2 VERIFY_SUBSAMPLE=1.0
export PY=$PYV
export MAIN=$SC/envs/rzero/lib/python3.12/site-packages/verl/trainer/main_ppo.py

export DIVERSITY_MODE=pmap_wb
export DIVERSITY_EMB_MODEL=unsloth/embeddinggemma-300m EMB_USE_ST=1
export DIVERSITY_WEIGHT=0.5
CAL=$($PYV - <<'PYEOF'
import json, sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/R-Zero/examples/reward_function"))
from diversity import embed
gold=[json.loads(l)["question"] for l in open(os.path.expanduser("~/v5stage/golden_set_v5.jsonl"))]
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
export SVC_MEM=0.25 Q_GPU_MEM=0.30
cd $SC
set +e
for CYCLE in $(seq 1 $CYCLES); do
  echo "=== UNISON CYCLE $CYCLE: Q-chunk (3 GRPO steps) questioner=$QCUR solver=$SCUR $(date) ==="
  COLOCATE=1 Q_STEPS=3 Q_TBATCH=60 Q_MINI=12 \
    ITER=$CYCLE EXP=unisonw4 ARM=verified \
    QUESTIONER_MODEL=$QCUR SOLVER_MODEL=$SCUR \
    VERIFY_WEIGHT=0.75 NUM_SAMPLES=50 EVAL_N=9 SKIP_STRAIN=1 \
    bash $H/R-Zero/scripts/iteration_rzero.sh
  RC=$?
  QNEW=$STORAGE_PATH/models/unisonw_it${CYCLE}_verified_q/global_step_3/actor/huggingface
  if [ -d "$QNEW" ]; then QCUR=$QNEW; else echo "CYCLE $CYCLE: Q ckpt missing (rc=$RC) — keeping $QCUR"; fi

  echo "=== UNISON CYCLE $CYCLE: solver replay check $(date) ==="
  if [ -f $HARVEST_DIR/solver_harvest.jsonl ]; then
    CUDA_VISIBLE_DEVICES=0 $PYV $H/solver_replay_grpo.py \
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
  A=$H/unison1w_artifacts; mkdir -p $A
  cp $STORAGE_PATH/artifacts/unisonw_it${CYCLE}_verified/challenger_batches.md $A/challenger_c${CYCLE}.md 2>/dev/null
  cp $HARVEST_DIR/consumed_c${CYCLE}.jsonl $A/ 2>/dev/null
done
echo "UNISON1W_LOOP_DONE questioner=$QCUR solver=$SCUR $(date)"
cp -r $HARVEST_DIR $H/unison1w_artifacts/harvest_final 2>/dev/null
echo "UNISON1W_DONE $(date)"
