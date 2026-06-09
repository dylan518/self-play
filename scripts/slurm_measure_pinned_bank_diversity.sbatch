#!/bin/bash
# Pin fixed question-bank examples, sample k completions from ONE prompt, report diversity.
# Gate before per-cycle pinned GRPO: need unique_q / reward_std high enough for group signal.
#SBATCH --job-name=q35-pin-div
#SBATCH --account=engr-lab-jiaxinh
#SBATCH --partition=condo-jiaxinh
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_pinned_diversity_%j.out
#SBATCH --error=/engrfs/project/jiaxinh/dylan_work/self-play/outputs/slurm/q35_pinned_diversity_%j.err

set -euo pipefail
cd /engrfs/project/jiaxinh/dylan_work/self-play
source .venv-qwen-slurm/bin/activate

K="${K:-32}"
BANK_SEED="${BANK_SEED:-4321}"
GEN_TEMP="${GEN_TEMP:-1.05}"
GEN_TOP_P="${GEN_TOP_P:-0.9}"
OUT_DIR="outputs/pinned_diversity/k${K}_seed${BANK_SEED}"
mkdir -p outputs/slurm "${OUT_DIR}" .jobhome

PORT="${VLLM_PORT:-$((21100 + SLURM_JOB_ID % 80))}"
VLLM_LOG="outputs/slurm/vllm_pinned_diversity_${SLURM_JOB_ID}.log"
JSONL="${OUT_DIR}/samples.jsonl"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.82 --enforce-eager \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

python - "$PORT" <<'PYWAIT'
import os, sys, time, urllib.request
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(f"vLLM ready HTTP {resp.status}", flush=True)
            raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit("vLLM timeout")
PYWAIT

python3 - "$PORT" "$JSONL" <<PY
import json, os, sys, urllib.request
from pathlib import Path

from grpo_math.self_play.generate_pairwise_data import _parse_question, _parse_question_confirm
from grpo_math.self_play.question_bank import render_question_bank_block_from_config

port, jsonl_path = sys.argv[1], sys.argv[2]
k = int(os.environ.get("K", "32"))
bank_seed = int(os.environ.get("BANK_SEED", "4321"))
gen_temp = float(os.environ.get("GEN_TEMP", "1.05"))
gen_top_p = float(os.environ.get("GEN_TOP_P", "0.9"))

prompt_template = Path("grpo_math/prompts/question_generator_prompt.txt").read_text(encoding="utf-8")
bank_cfg = {
    "question_bank": {
        "enabled": True,
        "path": "examples_gemini31flashlite_preview_nocategory_30.json",
        "num_examples": 8,
        "seed": bank_seed,
        "compatible_only": False,
    }
}
bank_block = render_question_bank_block_from_config(bank_cfg, seed=bank_seed)
prompt = prompt_template.format(question_bank_examples=bank_block)
Path(jsonl_path).write_text("", encoding="utf-8")
Path(jsonl_path).with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")

url = f"http://127.0.0.1:{port}/v1/chat/completions"
for idx in range(k):
    payload = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": gen_temp,
        "top_p": gen_top_p,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body["choices"][0]["message"].get("content") or ""
    question = _parse_question(raw)
    accepted, confirm = _parse_question_confirm(raw)
    row = {
        "sample_index": idx,
        "proposer_prompt": prompt,
        "proposer_group_id": f"pinned_seed{bank_seed}",
        "trainable_for_proposer": True,
        "question": question,
        "question_reward": 0.0,
        "accepted": accepted,
        "confirm": confirm,
        "generator_raw_output": raw,
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[gen] {idx+1}/{k} accepted={accepted} q_len={len(question)}", flush=True)
PY

python3 scripts/measure_pinned_prompt_diversity.py "$JSONL" \
  --output-json "${OUT_DIR}/diversity.json"

echo "done -> ${OUT_DIR}"
