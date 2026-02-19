# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Lint:**
```bash
ruff check .
```

**Tests:**
```bash
python -m pytest tests/test_reward.py
python -m pytest tests/test_logprobs.py
python -m pytest tests/test_left_padding_masking.py
```

**Pairwise rollout (smoke test):**
```bash
python -m grpo_math.self_play.generate_pairwise_data --config grpo_math/configs/pairwise_rollouts_smoke.yaml
```

**Debug rollout with reset:**
```bash
python tests/debug_single_pairwise_rollout.py --config grpo_math/configs/pairwise_rollouts_debug.yaml --reset-output
```

**Training:**
```bash
# Single GPU
python -m grpo_math.trl.train_grpo_trl --config grpo_math/configs/train_gsm8kv2_trl.yaml

# Multi-GPU
accelerate launch -m grpo_math.trl.train_grpo_trl --config grpo_math/configs/train_gsm8kv2_trl.yaml
```

**Evaluation:**
```bash
python -m grpo_math.eval.eval_gsm8k --config grpo_math/configs/train_gsm8kv2_trl.yaml --checkpoint <path> --max_samples 1000 --k 4
```

## Architecture

This is a self-improving training loop for math reasoning with three roles: **Generator** (creates problems), **Solver** (generates solutions), and **Judge** (evaluates quality). The goal is iterative solver improvement via pairwise preference data, where the generator is rewarded for producing *reliably evaluable* problems (not just hard ones).

### Core loop (`grpo_math/self_play/generate_pairwise_data.py`)

The main orchestration file (~1300 lines). Each rollout:
1. Samples questions from the generator model
2. Generates K solutions per question in configurable sampling groups (different temperatures/models to create quality variation)
3. Judges solution pairs in one of two modes:
   - `pairwise` — judge compares A vs B directly, outputs `PREFERENCE:`
   - `single_verify` — judge grades each solution individually (`VERDICT: CORRECT/INCORRECT`), with optional `CONFIDENCE:` tracing
4. Computes Elo ratings from comparison outcomes
5. Saves results to JSONL (append or overwrite)

The judge can be a local HuggingFace model or any OpenAI-compatible API (GPT-4, Gemini, etc.). API calls are parallelized with exponential backoff.

### Reliability metrics (see `README.md`)

Three metrics gate which questions enter training:
- **R_cons**: score consistency — low variance across repeated judging
- **R_sep**: strong/weak separation — better sampling regimes should win more often
- **R_stab**: preference stability — pairwise judgments agree on re-run

### Reward and format (`grpo_math/data/reward.py`)

All solver outputs must end with `FINAL_ANSWER: <integer>` (strict). Two reward signals:
- **Correctness**: `binary_reward()` via `extract_final_answer_int_strict()`
- **Format**: presence of the `FINAL_ANSWER:` tag

### Training (`grpo_math/trl/train_grpo_trl.py`)

Uses TRL's `GRPOTrainer` on GSM8K. Supports DeepSpeed ZeRO-2/3 (configs in `grpo_math/configs/deepspeed_*.json`). Default model: Qwen2.5-7B-Instruct.

### Configuration

Everything is YAML-driven. Rollout configs live in `grpo_math/configs/pairwise_rollouts_*.yaml`; training configs in `grpo_math/configs/train_*.yaml`. Prompt templates are in `grpo_math/prompts/` and referenced from configs with `{question}`, `{answer_a}`, `{solution}` placeholders.

### Outputs

Rollout results are written to `outputs/<experiment_name>/` as JSONL. Each record contains the question, all solutions, all pairwise comparison outcomes, and Elo ratings.
