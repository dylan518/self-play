#!/bin/bash
# race_submit.sh v2 — submit the same work to multiple clusters, keep whichever STARTS
# first, cancel the rest. Then keep watching the winner's log, echoing milestones.
#
# Usage:
#   tools/race_submit.sh <alias>:<sbatch path>[:<log glob>] [...]
#
#   alias      ssh alias (unity/washu/empire)
#   sbatch     cluster-specific batch file (paths/accounts differ per cluster)
#   log glob   optional remote glob for the job's log; enables the post-win watch phase
#
# Env knobs:
#   POLL_SEC=30         poll interval
#   POLL_MAX=172800     max seconds to wait for a start
#   WATCH_RE='...'      grep -E pattern for milestone/failure lines in the watch phase
#   DONE_RE='_DONE'     pattern that ends the watch phase
#
# DESIGN NOTE (DPW 07-12): run this under Claude Code's Monitor tool — every line this
# script prints becomes a chat notification, so the race announcement, the no-nodes
# warning, and each milestone of the winning job all arrive as events. One invocation
# = submit + race + waiter.
#
# Pre-flight: prints each cluster's scheduler start ESTIMATE (sbatch --test-only) and
# the sbatch's own walltime as expected runtime; warns loudly if nothing can start
# within AVAIL_WARN_H (default 6h).
set -u
POLL_SEC=${POLL_SEC:-30}
POLL_MAX=${POLL_MAX:-172800}
AVAIL_WARN_H=${AVAIL_WARN_H:-6}
WATCH_RE=${WATCH_RE:-'_START|_DONE|_OK|SAVED|LAG mean_v|pmap thresholds|A done|B done:|band-filter|judge\]|gen\]|Traceback|Error|FAILED|OutOfMemory|timed out'}
DONE_RE=${DONE_RE:-'_DONE|FINISHED'}

declare -a ALIASES PATHS LOGS JOBIDS
for spec in "$@"; do
  alias="${spec%%:*}"; rest="${spec#*:}"
  path="${rest%%:*}"; log=""
  [ "$rest" != "$path" ] && log="${rest#*:}"
  ALIASES+=("$alias"); PATHS+=("$path"); LOGS+=("$log")
done

# ---- pre-flight: estimates + walltime ----
soonest_epoch=""
for i in "${!ALIASES[@]}"; do
  wall=$(ssh -o ConnectTimeout=20 "${ALIASES[$i]}" "grep -m1 -- '-t ' '${PATHS[$i]}' | sed 's/.*-t //'" 2>/dev/null)
  est=$(ssh -o ConnectTimeout=20 "${ALIASES[$i]}" "sbatch --test-only '${PATHS[$i]}' 2>&1 | grep -oE 'to start at [^ ]+' | cut -d' ' -f4" 2>/dev/null)
  echo "[race] ${ALIASES[$i]}: est-start ${est:-unknown} | walltime ${wall:-?} | ${PATHS[$i]}"
  if [ -n "$est" ]; then
    ep=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$est" +%s 2>/dev/null || date -d "$est" +%s 2>/dev/null)
    [ -n "$ep" ] && { [ -z "$soonest_epoch" ] || [ "$ep" -lt "$soonest_epoch" ]; } && soonest_epoch=$ep
  fi
done
now=$(date +%s)
if [ -n "$soonest_epoch" ] && [ $((soonest_epoch - now)) -gt $((AVAIL_WARN_H*3600)) ]; then
  echo "[race] WARNING: NO NODES AVAILABLE SOON — best scheduler estimate is $(( (soonest_epoch-now)/3600 ))h out (estimates are pessimistic; racing anyway)"
fi

# ---- submit everywhere ----
for i in "${!ALIASES[@]}"; do
  jid=$(ssh -o ConnectTimeout=20 "${ALIASES[$i]}" "sbatch --parsable '${PATHS[$i]}'" 2>/dev/null | tail -1)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "[race] ${ALIASES[$i]}: submitted $jid"
    JOBIDS+=("$jid")
  else
    echo "[race] ${ALIASES[$i]}: SUBMIT FAILED" >&2
    JOBIDS+=("")
  fi
done

# ---- race ----
winner=-1; elapsed=0
while [ $elapsed -lt $POLL_MAX ] && [ $winner -lt 0 ]; do
  for i in "${!ALIASES[@]}"; do
    [ -z "${JOBIDS[$i]}" ] && continue
    st=$(ssh -o ConnectTimeout=20 "${ALIASES[$i]}" "squeue -j ${JOBIDS[$i]} -h -o %T" 2>/dev/null)
    if [ "$st" = "RUNNING" ] || { [ -z "$st" ] && [ $elapsed -gt 0 ]; }; then
      winner=$i
      echo "WINNER ${ALIASES[$i]} ${JOBIDS[$i]} (state: ${st:-exited})"
      for j in "${!ALIASES[@]}"; do
        [ "$j" != "$i" ] && [ -n "${JOBIDS[$j]}" ] && { ssh -o ConnectTimeout=20 "${ALIASES[$j]}" "scancel ${JOBIDS[$j]}" 2>/dev/null; echo "[race] cancelled ${ALIASES[$j]} ${JOBIDS[$j]}"; }
      done
      break
    fi
  done
  [ $winner -ge 0 ] && break
  sleep $POLL_SEC; elapsed=$((elapsed+POLL_SEC))
done
[ $winner -lt 0 ] && { echo "[race] TIMEOUT: nothing started in ${POLL_MAX}s"; exit 2; }

# ---- watch phase: echo winner's milestones until DONE ----
wlog="${LOGS[$winner]}"
[ -z "$wlog" ] && { echo "[race] no log glob for winner — watch phase skipped"; exit 0; }
wlog="${wlog//%j/${JOBIDS[$winner]}}"
POS=0
while true; do
  OUT=$(ssh -o ConnectTimeout=20 "${ALIASES[$winner]}" "cat $wlog 2>/dev/null | grep -n -E '$WATCH_RE'" 2>/dev/null)
  if [ -n "$OUT" ]; then
    NEW=$(echo "$OUT" | awk -F: -v p=$POS '$1>p')
    [ -n "$NEW" ] && { echo "$NEW" | cut -d: -f2- | grep -viE '^\s*File ' | head -6; POS=$(echo "$OUT" | tail -1 | cut -d: -f1); }
    echo "$OUT" | grep -qE "$DONE_RE" && { echo "[race] winner job finished"; exit 0; }
  fi
  st=$(ssh -o ConnectTimeout=20 "${ALIASES[$winner]}" "squeue -j ${JOBIDS[$winner]} -h -o %T" 2>/dev/null)
  [ -z "$st" ] && { echo "[race] winner left the queue — final log tail:"; ssh "${ALIASES[$winner]}" "tail -5 $wlog" 2>/dev/null; exit 0; }
  sleep 300
done
