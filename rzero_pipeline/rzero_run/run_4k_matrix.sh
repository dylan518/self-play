#!/bin/bash
IT1=/home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface
IT2=/home/nvidia/rzero_run/models/cont_it2_verified_s/global_step_20/actor/huggingface
BASE=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
V5=/data/jinyuan/rzero_storage/models/qwen3_4b_fullrun_authorsettings_solver_v5/global_step_15/actor/huggingface
echo "=== 4K MATRIX START $(date) ==="
bash /home/nvidia/rzero_run/run_oly.sh "$BASE" base_4k 4096
bash /home/nvidia/rzero_run/run_oly.sh "$IT1"  it1_4k  4096
bash /home/nvidia/rzero_run/run_oly.sh "$IT2"  it2_4k  4096
bash /home/nvidia/rzero_run/run_oly.sh "$V5"   v5_4k   4096
echo "=== 4K MATRIX DONE $(date) ==="
echo "--- 4K results ---"
for t in base_4k it1_4k it2_4k v5_4k; do /home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py $t; done
