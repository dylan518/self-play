#!/bin/bash
# wait for cvband training to finish, then eval + compare
while kill -0 346176 2>/dev/null; do sleep 30; done
sleep 10
DIR=/home/nvidia/rzero_run/models/cont_it2_cvband_s
CKPT=$(ls -d $DIR/global_step_*/actor/huggingface 2>/dev/null | sort -V | tail -1)
echo "=== CVBAND TRAIN FINISHED; ckpt=$CKPT ==="
if [ -d "$CKPT" ]; then
  bash /home/nvidia/rzero_run/run_oly.sh "$CKPT" cvband_8k 8192
else
  echo "NO CHECKPOINT — training may have failed"
fi
echo "=== A/B COMPARISON (OlympiadBench full 675 @ 8k) ==="
/home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py it1 2>/dev/null
/home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py it2_8k 2>/dev/null
/home/nvidia/venvs/rzero2/bin/python /home/nvidia/rzero_run/agg.py cvband_8k 2>/dev/null
echo "=== CVBAND PIPELINE DONE $(date) ==="
