#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a && source ./.env && set +a

REMOTE_CMD=$(python3 - <<'PY'
import base64
import gzip
import os
import shlex
from pathlib import Path

files = [
    "scripts/eval_question_dpo_pairs.py",
    "scripts/eval_gsmhard_sweep.py",
    "grpo_math/trl/train_question_from_rollout_rewards.py",
    "grpo_math/trl/train_grpo_trl.py",
    "grpo_math/self_play/run_self_play_grpo_loop.py",
    "scripts/slurm_qwen35_question_dpo_only_resume_cycle2.sbatch",
    "grpo_math/self_play/generate_pairwise_data.py",
    "scripts/slurm_eval_question_dpo_pairs.sbatch",
    "scripts/slurm_eval_question_dpo_holdout_cycle.sbatch",
    "scripts/analyze_dpo_pair_signal.py",
    "scripts/measure_pinned_prompt_diversity.py",
    "scripts/slurm_measure_pinned_bank_diversity.sbatch",
]
lines = ["set -e", f"cd {shlex.quote(os.environ['PROJECT_ROOT'])}"]
for f in files:
    raw = Path(f).read_bytes()
    payload = gzip.compress(raw)
    b64 = base64.b64encode(payload).decode("ascii")
    lines.append(f"mkdir -p $(dirname {shlex.quote(f)})")
    lines.append(
        "python3 -c "
        + shlex.quote(
            "import base64, gzip, pathlib; "
            f"p=pathlib.Path({f!r}); "
            f"p.write_bytes(gzip.decompress(base64.b64decode({b64!r})))"
        )
    )
lines.append("echo SYNC_OK")
print("\n".join(lines))
PY
)

export REMOTE_CMD
expect <<'EXPECT'
set timeout 180
spawn ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no $env(SSH_USER)@$env(SSH_HOST) bash -lc $env(REMOTE_CMD)
expect {
  -re "(?i)password:" { send -- "$env(SSH_PASS)\r"; exp_continue }
  -re "(?i)yes/no" { send -- "yes\r"; exp_continue }
  "SYNC_OK" { }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT
