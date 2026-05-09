#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_TRAIN=0
RESUME=0
OVERWRITE=0
VENV_DIR="${VENV_DIR:-.venv}"
CONFIG="${CONFIG:-grpo_math/configs/self_play_verdict_grpo_fixture.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/self_play_iterations/washu_iter_000}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/washu_run_iteration.sh [options]

Run one self-play iteration on the Wash U server. Use --dry-run or
--validate-only for no-GPU command planning.

Options:
  --config PATH       Iteration config. Defaults to $CONFIG or fixture config.
  --output-dir PATH   Output directory. Defaults to $OUTPUT_DIR or outputs/self_play_iterations/washu_iter_000.
  --dry-run           Print the command without executing it.
  --validate-only     Let the iteration runner validate/plan commands only.
  --skip-train        Collect/evaluate without solver/proposer training when supported.
  --resume            Resume an interrupted iteration when supported.
  --overwrite         Overwrite an existing output directory when supported.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?missing value for --config}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
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

VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
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

cmd=(
  "${PYTHON_CMD}" -m grpo_math.self_play.run_self_play_iteration
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cmd+=(--dry-run)
fi
if [[ "${VALIDATE_ONLY}" -eq 1 ]]; then
  cmd+=(--validate-only)
fi
if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  cmd+=(--skip-train)
fi
if [[ "${RESUME}" -eq 1 ]]; then
  cmd+=(--resume)
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  cmd+=(--overwrite)
fi

echo "== Wash U one-iteration run =="
echo "Repository: ${repo_root}"
echo "Config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"

if [[ ! -f "grpo_math/self_play/run_self_play_iteration.py" ]]; then
  echo "Full-loop runner is not present in this checkout yet."
  echo "Server-ready command:"
  printf '  %q' "${cmd[0]}"
  for arg in "${cmd[@]:1}"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[dry-run]'
  for arg in "${cmd[@]}"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
else
  "${cmd[@]}"
fi
