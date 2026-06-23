#!/bin/bash
PY=/home/nvidia/venvs/rzero2/bin/python
export CUDA_HOME=/home/nvidia/venvs/rzero2/lib/python3.12/site-packages/nvidia/cu13
export PATH=/home/nvidia/venvs/rzero2/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/software/software/Python/3.10.15-system/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_COMPILE_CACHE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
BASE=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
IT1=/home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface
IT2=/home/nvidia/rzero_run/models/cont_it2_verified_s/global_step_20/actor/huggingface
CUDA_VISIBLE_DEVICES=0 $PY /home/nvidia/R-Zero/pipeline/eval_compare.py --model $BASE --tag m500_base --n 500 --k 1 --max_tokens 4096 --benchmark math500 --out /data/selfplay/m500.json > /data/selfplay/m500_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY /home/nvidia/R-Zero/pipeline/eval_compare.py --model $IT1 --tag m500_it1 --n 500 --k 1 --max_tokens 4096 --benchmark math500 --out /data/selfplay/m500.json > /data/selfplay/m500_it1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY /home/nvidia/R-Zero/pipeline/eval_compare.py --model $IT2 --tag m500_it2 --n 500 --k 1 --max_tokens 4096 --benchmark math500 --out /data/selfplay/m500.json > /data/selfplay/m500_it2.log 2>&1 &
wait
echo "=== MATH500 DONE $(date) ==="
grep -h RESULT /data/selfplay/m500_base.log /data/selfplay/m500_it1.log /data/selfplay/m500_it2.log
