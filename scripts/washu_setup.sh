#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
INSTALL_FLASH_ATTN=0
VENV_DIR="${VENV_DIR:-.venv}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/washu_setup.sh [--dry-run] [--install-flash-attn] [--venv-dir PATH]

Prepare a Wash U server checkout for self-play experiments.

Options:
  --dry-run             Print checks and commands without creating a venv/installing packages.
  --install-flash-attn  Include the optional Linux flash-attn dependency.
  --venv-dir PATH       Virtualenv path to create/use. Defaults to .venv or $VENV_DIR.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --install-flash-attn)
      INSTALL_FLASH_ATTN=1
      shift
      ;;
    --venv-dir)
      VENV_DIR="${2:?missing value for --venv-dir}"
      shift 2
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "== Wash U setup =="
echo "Repository: ${repo_root}"
echo "Virtualenv: ${VENV_DIR}"

echo
echo "== Sanity checks =="
python3 - <<'PY'
import platform
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "GPU count: $(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "WARNING: nvidia-smi not found. GPU jobs will not run until CUDA drivers are available."
fi

python3 - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"PyTorch import: unavailable before install ({exc})")
else:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
PY

echo "Disk space:"
df -h "${repo_root}"

echo
echo "== API key checks =="
for key in OPENAI_API_KEY GEMINI_API_KEY WANDB_API_KEY HF_TOKEN; do
  if [[ -n "${!key:-}" ]]; then
    echo "${key}: set"
  else
    echo "${key}: not set"
  fi
done

echo
echo "== Install commands =="
run python3 -m venv "${VENV_DIR}"
run "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools
run "${VENV_DIR}/bin/python" -m pip install -e '.[dev]'
if [[ "${INSTALL_FLASH_ATTN}" -eq 1 ]]; then
  run "${VENV_DIR}/bin/python" -m pip install -e '.[flash]'
else
  echo "Skipping flash-attn. Re-run with --install-flash-attn on compatible Linux/CUDA nodes."
fi

echo
echo "== Post-install validation commands =="
echo "source ${VENV_DIR}/bin/activate"
echo "ruff check ."
echo "python -m pytest tests/test_reward.py tests/test_logprobs.py tests/test_left_padding_masking.py"
echo
echo "Setup script complete."
