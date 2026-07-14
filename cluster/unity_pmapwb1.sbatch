#!/bin/bash
#SBATCH -N1 -n1 -c16
#SBATCH --mem=250G
#SBATCH -t 06:00:00
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:4
#SBATCH -J pmapwb1
#SBATCH -o /work/pi_general_dartmouth_edu/dylan/logs/pmapwb1_%j.log
# ITER1 recipe with ONLY the diversity term swapped: within-batch hinged pmap
# (EmbeddingGemma on question text, thresholds = golden_set_v5 pairwise means computed
# at launch via the SAME diversity.embed() code path as the Desktop rescoring).
# r = uncertainty + 0.75*verified(K=3,graded) + 0.5*D_wb. One questioner GRPO step from
# BASE. Current 4-GPU colocate speedups. NOT Lagrangian, no band gate, no bank state.
set -e
D=/work/pi_general_dartmouth_edu/dylan
export HF_HOME=$D/.cache/huggingface
export STORAGE_PATH=$D/pmapwb1
mkdir -p $STORAGE_PATH $D/logs
export DUMP_VERIFY_DIR=$STORAGE_PATH/verify_rollouts
export VERIFY_K_PROGRAMS=3 VERIFY_MIN_AGREE=2 VERIFY_SUBSAMPLE=1.0
export PY=$D/envs/rzero/bin/python
export MAIN=$D/envs/rzero/lib/python3.12/site-packages/verl/trainer/main_ppo.py
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$D/envs/rzero/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
echo "PMAPWB1_START $(date) node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
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
export PMAP_TMAX=$(echo $CAL | awk "{print \$1}") PMAP_TMEAN=$(echo $CAL | awk "{print \$2}")
echo "pmap_wb thresholds (Gemma, golden_set_v5 pairwise means): PMAP_TMAX=$PMAP_TMAX PMAP_TMEAN=$PMAP_TMEAN"
cd $D
set +e
COLOCATE=1 Q_STEPS=1 Q_TBATCH=60 Q_MINI=12 \
  ITER=1 EXP=pmapwb ARM=verified \
  QUESTIONER_MODEL=Qwen/Qwen3.5-4B SOLVER_MODEL=Qwen/Qwen3.5-4B \
  VERIFY_WEIGHT=0.75 NUM_SAMPLES=250 EVAL_N=9 SKIP_STRAIN=1 \
  bash $D/R-Zero/scripts/iteration_rzero.sh
RC=$?
cp $STORAGE_PATH/artifacts/pmapwb_it1_verified/challenger_batches.md $D/logs/pmapwb1_batches.md 2>/dev/null
echo "PMAPWB1_DONE rc=$RC $(date)"
