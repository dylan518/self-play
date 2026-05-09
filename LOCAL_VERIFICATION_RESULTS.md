# Local verification results (no GPU)

Date: 2026-05-07  
Machine: macOS (darwin), workspace Python used for tests: **Python 3.10.16** (`python3`).

This document records what was run locally without CUDA/GPU training, whether it passed, and what remains blocked by missing dependencies (notably **PyTorch** on the default interpreter used for pytest).

## Summary


| Area                                                                              | Status                      | Notes                                                                                                   |
| --------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- |
| Verdict + question reward math (Agent 1)                                          | **Pass**                    | Covered by `test_verdict_rewards.py` + `test_question_rewards.py`; integrates with collector / trainers |
| Question filter, verifier, rollout collector, GRPO entrypoints, loop (Agents 2–7) | **Pass**                    | Same pytest batch as below — **95 tests total** for the CPU-safe checklist                              |
| Torch-dependent legacy tests                                                      | **Not run**                 | `torch` not installed on the pytest interpreter → collection errors                                     |
| Fixture rollout CLI                                                               | **Pass**                    | `collect_self_play_rollouts` writes JSONL with `schema_version: self_play_verdict_grpo_v1`              |
| Train scripts `--validate-only`                                                   | **Pass**                    | Solver + proposer smoke configs validate; proposer smoke uses canonical fixture rollouts                |
| Iteration harness `--dry-run`                                                     | **Pass with config caveat** | Planned stages + `report.md` path emitted; see duplicate-key note below                                 |
| Wash U helper scripts `--dry-run`                                                 | **Pass**                    | `washu_setup.sh`, `washu_run_smoke.sh` exit 0                                                           |
| Real pairwise / HF rollout (`generate_pairwise_data`)                             | **Not run**                 | Requires local HF models + GPU memory in typical configs                                                |
| Ruff                                                                              | **Not run**                 | `ruff` is not installed as a command or Python module in this environment                               |


## Pytest (CPU-safe checklist from `LOCAL_SETUP_AGENT_PLAN.md`)

Command:

```bash
python3 -m pytest \
  tests/test_reward.py \
  tests/test_verdict_rewards.py \
  tests/test_question_rewards.py \
  tests/test_question_filter.py \
  tests/test_verifier.py \
  tests/test_collect_self_play_rollouts.py \
  tests/test_train_solver_verdict_grpo.py \
  tests/test_train_proposer_question_grpo.py \
  tests/test_run_self_play_iteration.py \
  tests/test_run_self_play_loop.py \
  -v --tb=short
```

**Result: 95 passed** (about 1.3s).

Focused integration batch:

```bash
python3 -m pytest \
  tests/test_train_proposer_question_grpo.py \
  tests/test_train_solver_verdict_grpo.py \
  tests/test_collect_self_play_rollouts.py \
  tests/test_run_self_play_iteration.py \
  tests/test_run_self_play_loop.py
```

**Result: 22 passed** (about 0.9s).

### Full `tests/` directory

Command: `python3 -m pytest tests/ -v`

**Result: collection failed** on:

- `tests/test_logprobs.py` — `ModuleNotFoundError: No module named 'torch'`
- `tests/test_left_padding_masking.py` — same

To run the full suite locally, install project deps (including `torch`) in the environment used for pytest, per `README.md` / `scripts/washu_setup.sh`.

## CLI smoke (no training)

### Rollout collection (fixture providers)

```bash
python3 -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml
```

**Result:** exit 0; messages reported 5 questions (3 accepted, 2 rejected); wrote `outputs/self_play_fixture/rollouts.jsonl` and `rejected_questions.jsonl`. Rows include top-level `schema_version` aligned with `**self_play_verdict_grpo_v1`**.

The output override also works:

```bash
python3 -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --output-dir /tmp/self_play_fixture_collect_check \
  --overwrite
```

**Result:** exit 0; wrote `/tmp/self_play_fixture_collect_check/rollouts.jsonl` and `/tmp/self_play_fixture_collect_check/rejected_questions.jsonl`.

### Solver GRPO validate-only

```bash
python3 -m grpo_math.trl.train_solver_verdict_grpo \
  --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml \
  --validate-only
```

**Result:** exit 0; printed a server-ready training command.

### Proposer GRPO validate-only

```bash
python3 -m grpo_math.trl.train_proposer_question_grpo \
  --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml \
  --validate-only
```

**Result:** exit 0; emitted JSON metrics:

```json
{"acceptance_rate": 0.6, "duplicate_rate": 0.2, "mean_question_reward": 0.4, "mixed_verdict_rate": 0.0, "num_examples": 5.0, "schema_version": "self_play_verdict_grpo_v1", "zero_reward_rate": 0.4}
```

### One iteration (dry-run)

```bash
python3 -m grpo_math.self_play.run_self_play_iteration \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --dry-run \
  --output-dir /tmp/self_play_local_dry_run_test \
  --overwrite
```

**Result:** exit 0; JSON plan included rollout, solver GRPO, proposer GRPO, and eval stages with `status: "planned"` and paths under `/tmp/self_play_local_dry_run_test`.

Note: `grpo_math/configs/self_play_verdict_grpo_fixture.yaml` currently contains two top-level `rollout:` keys. YAML keeps the later orchestrator-style key, so the dry-run stage plans the default `grpo_math/configs/pairwise_rollouts_smoke.yaml` rollout config instead of the fixture rollout config. The standalone fixture collector command above verifies the collector itself.

### Wash U scripts

```bash
bash scripts/washu_setup.sh --dry-run
bash scripts/washu_run_smoke.sh --dry-run
```

**Result:** both exit 0; printed intended commands and sanity-check summaries (this machine: no `nvidia-smi`, PyTorch import unavailable until venv install).

## Agent 1 integration notes

- `**grpo_math/self_play/rewards.py`** — canonical `SCHEMA_VERSION` and solver verdict scoring / within-question advantages used by tests and downstream code.
- `**grpo_math/self_play/question_rewards.py`** — proposer question rewards; filtered questions pinned to `question_reward = 0.0` with configurable proposer trainability.
- `**grpo_math/self_play/generate_pairwise_data.py**` — when `judge.mode` is `**single_verify**`, JSONL rows gain `schema_version`, per-solution verdict-derived fields, and question-level reward summary. The default `**pairwise_rollouts_smoke.yaml**` uses **pairwise** judging, so those extra fields do not appear unless you switch that config to `single_verify`.

## Conclusion

On this machine, **all verdict-only self-play components that are testable without PyTorch are working**: unit tests, fixture rollout collection, training-entry validation, iteration dry-run, and Wash U dry-run scripts. **GPU training and HF-backed pairwise rollouts were not executed.** Installing `torch` (and following `washu_setup.sh` or `pip install -e ".[dev]"`) unlocks the remaining legacy tests and enables on-machine rollout smoke with model weights.