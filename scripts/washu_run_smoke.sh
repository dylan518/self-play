#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EXISTING=1
VENV_DIR="${VENV_DIR:-.venv}"
ROLLOUT_CONFIG="${ROLLOUT_CONFIG:-grpo_math/configs/self_play_verdict_grpo_fixture.yaml}"
SOLVER_CONFIG="${SOLVER_CONFIG:-grpo_math/configs/train_solver_verdict_grpo_smoke.yaml}"
PROPOSER_CONFIG="${PROPOSER_CONFIG:-grpo_math/configs/train_proposer_question_grpo_smoke.yaml}"
ITERATION_CONFIG="${ITERATION_CONFIG:-grpo_math/configs/self_play_verdict_grpo_fixture.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/washu_smoke}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/washu_run_smoke.sh [--dry-run] [--run-missing]

Run or print the Wash U smoke sequence:
  1. fixture rollout collection
  2. solver verdict GRPO config validation
  3. proposer question GRPO config validation
  4. full iteration dry-run

Options:
  --dry-run      Print commands and sanity checks without executing jobs.
  --run-missing  Attempt planned commands even if their modules are not present yet.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-missing)
      SKIP_EXISTING=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
    return
  fi
  for candidate in python3.10 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
      then
        command -v "${candidate}"
        return
      fi
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    echo "python"
  fi
}

PYTHON_CMD="$(resolve_python)"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %q' "$1"
    shift || true
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

has_module_file() {
  local module_path="$1"
  [[ -f "${module_path}" ]]
}

run_if_available() {
  local label="$1"
  local module_file="$2"
  shift 2

  echo
  echo "== ${label} =="
  if [[ "${SKIP_EXISTING}" -eq 1 && ! -f "${module_file}" ]]; then
    echo "Skipping: ${module_file} is not present in this checkout yet."
    echo "Planned command:"
    printf '  %q' "$1"
    shift || true
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
    return 0
  fi
  run "$@"
}

echo "== Wash U smoke =="
echo "Repository: ${repo_root}"
echo "Python: ${PYTHON_CMD}"
echo "Output root: ${OUTPUT_ROOT}"

echo
echo "== Sanity checks =="
run "${PYTHON_CMD}" - <<'PY'
import shutil
import sys

print(f"Python: {sys.version.split()[0]}")
try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
else:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

usage = shutil.disk_usage(".")
print(f"Disk free GiB: {usage.free / (1024 ** 3):.1f}")
PY

for key in OPENAI_API_KEY GEMINI_API_KEY HF_TOKEN WANDB_API_KEY; do
  if [[ -n "${!key:-}" ]]; then
    echo "${key}: set"
  else
    echo "${key}: not set"
  fi
done

echo
echo "== Fixture rollout collection =="
run "${PYTHON_CMD}" -m grpo_math.self_play.collect_self_play_rollouts \
  --config "${ROLLOUT_CONFIG}" \
  --overwrite

echo
echo "== Solver verdict GRPO validation =="
run "${PYTHON_CMD}" -m grpo_math.trl.train_solver_verdict_grpo \
  --config "${SOLVER_CONFIG}" \
  --validate-only

run_if_available \
  "Proposer question GRPO validation" \
  "grpo_math/trl/train_proposer_question_grpo.py" \
  "${PYTHON_CMD}" -m grpo_math.trl.train_proposer_question_grpo \
  --config "${PROPOSER_CONFIG}" \
  --validate-only

run_if_available \
  "Full iteration smoke" \
  "grpo_math/self_play/run_self_play_iteration.py" \
  "${PYTHON_CMD}" -m grpo_math.self_play.run_self_play_iteration \
  --config "${ITERATION_CONFIG}" \
  --dry-run \
  --output-dir "${OUTPUT_ROOT}/iteration" \
  --overwrite

echo
echo "Smoke script complete. Inspect ${OUTPUT_ROOT}, configured rollout outputs, and logs."
