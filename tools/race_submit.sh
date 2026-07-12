#!/bin/bash
# race_submit.sh — submit the same work to multiple clusters, keep whichever STARTS
# first, scancel the rest. Solves the "we want something with no queue" problem by
# racing every queue we have instead of predicting them.
#
# Usage:
#   tools/race_submit.sh unity:/path/on/unity/job.sbatch washu:/path/on/washu/job.sbatch [...]
#
# Each arg is <ssh-alias>:<remote sbatch path>. The sbatch files are cluster-specific
# (paths/envs/accounts differ per cluster — write one per cluster, this script only
# handles the submit/poll/cancel choreography). Requires passwordless ssh aliases.
# Polls every 30s for up to POLL_MAX (default 48h). Prints "WINNER <alias> <jobid>".
set -u
POLL_SEC=${POLL_SEC:-30}
POLL_MAX=${POLL_MAX:-172800}
declare -a ALIASES JOBIDS
for spec in "$@"; do
  alias="${spec%%:*}"; path="${spec#*:}"
  jid=$(ssh -o ConnectTimeout=20 "$alias" "sbatch --parsable '$path'" 2>/dev/null | tail -1)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "[race] $alias: submitted $jid ($path)"
    ALIASES+=("$alias"); JOBIDS+=("$jid")
  else
    echo "[race] $alias: SUBMIT FAILED ($path)" >&2
  fi
done
[ ${#ALIASES[@]} -eq 0 ] && { echo "[race] nothing submitted" >&2; exit 1; }
[ ${#ALIASES[@]} -eq 1 ] && { echo "WINNER ${ALIASES[0]} ${JOBIDS[0]} (only entrant)"; exit 0; }

elapsed=0
while [ $elapsed -lt $POLL_MAX ]; do
  for i in "${!ALIASES[@]}"; do
    st=$(ssh -o ConnectTimeout=20 "${ALIASES[$i]}" "squeue -j ${JOBIDS[$i]} -h -o %T" 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      echo "WINNER ${ALIASES[$i]} ${JOBIDS[$i]}"
      for j in "${!ALIASES[@]}"; do
        [ "$j" != "$i" ] && { ssh -o ConnectTimeout=20 "${ALIASES[$j]}" "scancel ${JOBIDS[$j]}" 2>/dev/null; echo "[race] cancelled ${ALIASES[$j]} ${JOBIDS[$j]}"; }
      done
      exit 0
    elif [ -z "$st" ]; then
      # left the queue without us seeing RUNNING — completed fast or died; treat as winner-by-completion
      echo "WINNER-BY-EXIT ${ALIASES[$i]} ${JOBIDS[$i]} (check its log)"
      for j in "${!ALIASES[@]}"; do
        [ "$j" != "$i" ] && ssh -o ConnectTimeout=20 "${ALIASES[$j]}" "scancel ${JOBIDS[$j]}" 2>/dev/null
      done
      exit 0
    fi
  done
  sleep $POLL_SEC; elapsed=$((elapsed+POLL_SEC))
done
echo "[race] timeout after ${POLL_MAX}s — all still pending" >&2
exit 2
