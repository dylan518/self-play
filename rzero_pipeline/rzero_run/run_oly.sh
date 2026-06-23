#!/bin/bash
# args: MODEL TAG MAXTOK
MODEL="$1"; TAG="$2"; MAXTOK="${3:-8192}"
cd /home/nvidia/rzero_run
PY=/home/nvidia/venvs/rzero2/bin/python
export CUDA_HOME=/home/nvidia/venvs/rzero2/lib/python3.12/site-packages/nvidia/cu13
export PATH=/home/nvidia/venvs/rzero2/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/software/software/Python/3.10.15-system/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_COMPILE_CACHE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "=== EVAL START $(date) tag=$TAG maxtok=$MAXTOK model=$MODEL ==="
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY eval_oly_shard.py --model "$MODEL" --out /data/selfplay/oly_${TAG}_shard$s.jsonl --shard $s --nshards 4 --max_tokens $MAXTOK > /data/selfplay/oly_${TAG}_shard$s.log 2>&1 &
done
wait
$PY - "$TAG" <<'PYAGG'
import json,glob,sys,statistics
tag=sys.argv[1]
rows=[]
for f in sorted(glob.glob(f"/data/selfplay/oly_{tag}_shard*.jsonl")): rows+=[json.loads(l) for l in open(f)]
n=len(rows); fmt=sum(r["fmt"] for r in rows)
cf=sum(r["corr_first"] for r in rows); ca=sum(r["corr_any"] for r in rows)
print(f"RESULT_{tag} n={n} format_rate={fmt/n:.4f} acc_given_fmt_first={cf/fmt:.4f} pass_first={cf/n:.4f} acc_any={ca/fmt:.4f} pass_any={ca/n:.4f} mean_resp_chars={statistics.mean(r['resp_len'] for r in rows):.0f}")
PYAGG
echo "=== EVAL DONE $(date) tag=$TAG ==="
