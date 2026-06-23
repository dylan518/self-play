#!/bin/bash
# Autonomous held-out-gain sweep. Runs INSIDE a 4-GPU allocation; iterates configs
# back-to-back (train -> merge -> eval held-out acc|formatted -> paired test) and
# STOPS on the first config with a significant reasoning gain. One queue wait, then
# continuous search. Gate metric = held-out acc|formatted (NOT raw pass).
set -uo pipefail
export LD_LIBRARY_PATH=/mnt/home/ch2263/extralib:/mnt/home/software/software/Python/3.10.15-system/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/mnt/lustre/cornell/ch2263/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd ~/self-play
PY=~/venvs/selfplay/bin/python
ACC=~/venvs/selfplay/bin/accelerate
SP=/mnt/lustre/cornell/ch2263/sp
HELD=$SP/rh_heldout.json
LOG=$SP/sweep.log
echo "===== SWEEP START $(date -u) =====" | tee -a "$LOG"

echo "--- base held-out eval ---" | tee -a "$LOG"
$PY scripts/eval_accfmt.py Qwen/Qwen3.5-9B "$HELD" 16 2048 > "$SP/base_eval.json" 2>>"$LOG"
grep RESULT "$SP/base_eval.json" | tee -a "$LOG"

# sweep over (grad_accum, lr). batch is the primary lever (R-Zero); ordered small->large.
CONFIGS=("8 2.0e-5" "8 4.0e-5" "16 2.0e-5" "16 4.0e-5" "32 2.0e-5")
i=0
for c in "${CONFIGS[@]}"; do
  set -- $c; GA=$1; LR=$2; i=$((i+1))
  RUN=$SP/sweep_run$i
  rm -rf "$RUN"
  echo "--- config $i: grad_accum=$GA lr=$LR ($(date -u)) ---" | tee -a "$LOG"
  CFG=grpo_math/configs/_sweep$i.yaml
  sed -e "s#grad_accum_steps: 8#grad_accum_steps: $GA#" -e "s#lr: 2.0e-5#lr: $LR#" \
      -e "s#/mnt/lustre/cornell/ch2263/sp/run#$RUN#" grpo_math/configs/train_empire.yaml > "$CFG"
  $ACC launch --num_processes 4 --num_machines 1 -m grpo_math.trl.train_grpo_trl --config "$CFG" >> "$LOG" 2>&1
  CKPT=$(ls -dt "$RUN"/checkpoint-* 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "config $i: NO CHECKPOINT (train failed) — see log" | tee -a "$LOG"; continue; fi
  MERGED=$RUN/merged
  $PY scripts/merge_lora_adapter.py --base Qwen/Qwen3.5-9B --adapter "$CKPT" --out "$MERGED" >> "$LOG" 2>&1
  $PY scripts/eval_accfmt.py "$MERGED" "$HELD" 16 2048 > "$RUN/eval.json" 2>>"$LOG"
  grep RESULT "$RUN/eval.json" | tee -a "$LOG"
  if $PY scripts/paired_test.py "$SP/base_eval.json" "$RUN/eval.json" | tee -a "$LOG"; then
    echo "===== WINNER config $i: grad_accum=$GA lr=$LR — SIGNIFICANT held-out acc|fmt gain $(date -u) =====" | tee -a "$LOG"
    break
  fi
  rm -rf "$MERGED"   # free disk; LoRA adapter kept under $CKPT
done
echo "===== SWEEP_DONE $(date -u) =====" | tee -a "$LOG"
