# Local Setup Agent Plan Before Wash U Server

Goal: implement the full self-play loop locally as far as possible before
running GPU jobs on the Wash U server.

The local machine should produce code, configs, tests, dry-run artifacts, and
server-ready commands. It does not need to run real GPU training locally.

## Non-Negotiable System Goal

Implement the real full loop:

```text
proposer GRPO model
  -> candidate questions
  -> question filter
  -> solver samples
  -> verifier verdicts
  -> solver rewards/advantages
  -> proposer question rewards/advantages
  -> solver GRPO update
  -> proposer GRPO update
  -> evaluation
  -> next iteration
```

Filtered-out questions get `question_reward = 0.0`, are skipped for solver and
verifier compute, and are retained in proposer GRPO groups as zero-reward
samples.

## Local Definition Of Done

Before moving to the Wash U server, local development should have:

- real code paths for rollout collection, solver GRPO, proposer GRPO, and loop
orchestration
- fixture-backed providers for proposer, solver, and verifier tests
- deterministic question filter tests
- verifier parser/cache tests
- solver reward tests
- proposer question reward tests
- dry-run one-iteration command
- generated Wash U run commands/configs
- no required GPU execution in local tests

## Agent 1: Schemas And Reward Math

### Mission

Define the canonical rollout schema and implement both solver and proposer
reward math.

### Files

- Create `grpo_math/self_play/rewards.py`
- Create `grpo_math/self_play/question_rewards.py`
- Create `tests/test_verdict_rewards.py`
- Create `tests/test_question_rewards.py`
- Edit rollout writer/collector to emit the new fields

### Tasks

- Implement solver verdict scoring.
- Implement within-question solver advantages.
- Implement question/proposer rewards.
- Ensure filtered-out questions get exactly `question_reward = 0.0`.
- Ensure filtered-out questions have `trainable_for_solver = false`.
- Ensure filtered-out questions can remain `trainable_for_proposer = true`.
- Add schema constants for `self_play_verdict_grpo_v1`.
- Add tests for accepted, rejected, duplicate, malformed, mixed, all-correct,
all-incorrect, invalid, and unclear cases.

### Local Checks

```bash
python -m pytest tests/test_verdict_rewards.py tests/test_question_rewards.py
```

## Agent 2: Question Filter

### Mission

Build the production question filter and rejected-question logging.

### Files

- Create `grpo_math/self_play/question_filter.py`
- Create `grpo_math/prompts/question_filter_prompt.txt`
- Create `tests/test_question_filter.py`

### Tasks

- Implement deterministic filter.
- Implement optional LLM filter parser.
- Reject malformed, duplicate, ambiguous, non-integer, answer-leaking,
external-context, off-domain, trivial, or impossible questions.
- Emit structured filter results.
- Log rejected questions with zero reward.
- Add tests proving solver/verifier are not called for filtered questions.

### Local Checks

```bash
python -m pytest tests/test_question_filter.py
```

### Agent 2 Local Results

Status: complete on local CPU path.

Implemented:

- `grpo_math/self_play/question_filter.py`
- `grpo_math/prompts/question_filter_prompt.txt`
- `tests/test_question_filter.py`

Verified behavior:

- deterministic question filtering
- optional LLM filter output parsing
- rejection reasons for malformed, duplicate, ambiguous, non-integer,
answer-leaking, external-context, off-domain, trivial, and impossible questions
- structured filter results with `trainable_for_solver`,
`trainable_for_proposer`, and `question_reward`
- rejected-question JSONL logging with `question_reward = 0.0`
- filtered questions are skipped before fake solver/verifier calls in tests

Local test results:

```bash
python3 -m pytest tests/test_question_filter.py
# 16 passed

python3 -m pytest tests/test_question_filter.py tests/test_reward.py
# 20 passed
```

Local environment limitations:

- `python` is not on PATH in this shell; `python3` was used.
- `torch` is not installed, so torch-importing tests such as
`tests/test_left_padding_masking.py` cannot be collected locally right now.
- `ruff` is not installed in the active Python environment, so command-line
Ruff was not run. Cursor lints reported no errors on Agent 2 edited files.

## Agent 3: Verifier Interface

### Mission

Create the verifier abstraction used by rollout collection and online GRPO
rewards.

### Files

- Create `grpo_math/self_play/verifier.py`
- Create `tests/test_verifier.py`

### Tasks

- Parse `CORRECT`, `INCORRECT`, `INVALID`, `UNCLEAR`.
- Support OpenAI-compatible verifier calls.
- Support fixture verifier for local tests.
- Add verifier cache keyed by question, completion, prompt version, and model.
- Add retry/backoff behavior.
- Expose batch verification API.

### Local Checks

```bash
python -m pytest tests/test_verifier.py
```

### Agent 3 Local Results

Status: complete on local CPU path.

Implemented:

- `grpo_math/self_play/verifier.py`
- `tests/test_verifier.py`

Verified behavior:

- parses `CORRECT`, `INCORRECT`, `INVALID`, and `UNCLEAR`
- supports a fixture verifier for local/no-GPU tests
- supports OpenAI-compatible chat completion verifier calls through an
injectable request function
- caches verifier results by question, completion, prompt version, and model
- retries transient API failures with exponential backoff
- exposes `verify_batch` for batch verification call sites

Local test results:

```bash
python3 -m pytest tests/test_verifier.py
# 7 passed

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
  tests/test_run_self_play_loop.py
# 96 passed
```

Additional local checks:

```bash
python3 -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --overwrite
# Collected 5 questions (3 accepted, 2 rejected)
# Wrote rollouts to outputs/self_play_fixture/rollouts.jsonl
# Wrote rejected questions to outputs/self_play_fixture/rejected_questions.jsonl

python3 -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --output-dir /tmp/self_play_fixture_rollouts \
  --overwrite
# Collected 5 questions (3 accepted, 2 rejected)
# Wrote rollouts to /tmp/self_play_fixture_rollouts/rollouts.jsonl
# Wrote rejected questions to /tmp/self_play_fixture_rollouts/rejected_questions.jsonl

python3 -m grpo_math.trl.train_solver_verdict_grpo \
  --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml \
  --validate-only
# Config valid. Server command generated.

python3 -m grpo_math.trl.train_proposer_question_grpo \
  --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml \
  --validate-only
# Printed proposer dry-run metrics JSON.

python3 -m grpo_math.self_play.run_self_play_iteration \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --dry-run \
  --output-dir /tmp/self_play_local_dry_run \
  --overwrite
# Wrote planned stage records, report.md, and next_iteration.yaml.

PYTHON=python3 bash scripts/washu_run_smoke.sh --dry-run
# Printed rollout, solver, proposer, and full-iteration smoke commands.

bash scripts/washu_setup.sh --dry-run
# Printed setup/install commands and local sanity checks.
```

Integration note:

- While testing local dry-runs, the rollout collector accepted only configured
output paths. It now also accepts `--output-dir` and writes `rollouts.jsonl` and
`rejected_questions.jsonl` there, matching generated loop commands and making
fixture rollout collection easier to run in isolated temp directories.

Local environment limitations:

- No GPU is available locally; GPU training was not executed.
- `python` is not on PATH in this shell; `python3` was used for local checks.
- Command-line Ruff is not installed in the active Python environment. Cursor
lints reported no errors on the edited Agent 3 and collector files.

## Agent 4: Full Rollout Collector

### Mission

Implement rollout collection for both solver and proposer training.

### Files

- Create `grpo_math/self_play/collect_self_play_rollouts.py`
- Create `tests/test_collect_self_play_rollouts.py`
- Optionally refactor `generate_pairwise_data.py` to reuse shared providers

### Tasks

- Add proposer provider interface.
- Add solver provider interface.
- Use question filter before solver/verifier calls.
- Include filtered-out questions in proposer groups with zero reward.
- Sample multiple questions per proposer prompt for proposer GRPO grouping.
- Sample multiple solver completions per accepted question.
- Run verifier on accepted question completions.
- Compute solver and proposer rewards/advantages.
- Write canonical rollout JSONL.
- Write `rejected_questions.jsonl`.
- Support fixture provider dry-runs.

### Local Checks

```bash
python -m pytest tests/test_collect_self_play_rollouts.py
python -m grpo_math.self_play.collect_self_play_rollouts \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml
```

## Agent 5: Solver GRPO

### Mission

Implement the solver training path from verifier rewards.

### Files

- Create `grpo_math/trl/train_solver_verdict_grpo.py`
- Create `grpo_math/configs/train_solver_verdict_grpo.yaml`
- Create `grpo_math/configs/train_solver_verdict_grpo_smoke.yaml`
- Create `tests/test_train_solver_verdict_grpo.py`

### Tasks

- Load accepted generated questions.
- Sample solver completions during GRPO.
- Reward completions with verifier verdicts.
- Cache verifier calls.
- Keep format reward as a logged auxiliary metric.
- Add CPU tests for data loading, reward interface, and config validation.
- Generate server command without executing training locally.

### Local Checks

```bash
python -m pytest tests/test_train_solver_verdict_grpo.py
python -m grpo_math.trl.train_solver_verdict_grpo \
  --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml \
  --validate-only
```

### Agent 5 Local Result

Status on 2026-05-07: complete for no-GPU local validation.

- Implemented `grpo_math/trl/train_solver_verdict_grpo.py`.
- Added `grpo_math/configs/train_solver_verdict_grpo.yaml`.
- Added `grpo_math/configs/train_solver_verdict_grpo_smoke.yaml`.
- Added `tests/test_train_solver_verdict_grpo.py`.
- Added `tests/fixtures/solver_verdict_rollouts.jsonl`.
- Uses generated accepted questions from rollout JSONL.
- Uses verifier verdicts for the train reward and keeps strict `FINAL_ANSWER`
format reward as a logged auxiliary metric with zero reward weight.
- Uses the shared verifier parser/cache/client interfaces from
`grpo_math/self_play/verifier.py`.
- Local command passed with `python3`:
`python3 -m pytest tests/test_train_solver_verdict_grpo.py`.
- Local command passed with `python3`:
`python3 -m grpo_math.trl.train_solver_verdict_grpo --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml --validate-only`.
- Real GRPO training was not executed locally because this machine currently has
no GPU and the active Python environment does not have `torch` installed.

## Agent 6: Proposer GRPO

### Mission

Implement the question/proposer model GRPO update. This is the most important
piece for making the loop self-improving.

### Files

- Create `grpo_math/trl/train_proposer_question_grpo.py`
- Create `grpo_math/configs/train_proposer_question_grpo.yaml`
- Create `grpo_math/configs/train_proposer_question_grpo_smoke.yaml`
- Create `tests/test_train_proposer_question_grpo.py`

### Tasks

- Load proposer prompts/groups.
- Sample candidate questions during GRPO.
- Evaluate each generated question through filter, solver, verifier, and
question reward when online mode is enabled.
- Support offline/dry-run mode from precomputed rollout question rewards.
- Normalize question rewards within proposer groups.
- Ensure filtered-out questions receive zero reward.
- Log acceptance rate, zero-reward rate, duplicate rate, mixed-verdict rate,
and mean question reward.
- Add CPU tests for dataset loading, reward interface, group advantages, and
config validation.

### Local Checks

```bash
python -m pytest tests/test_train_proposer_question_grpo.py
python -m grpo_math.trl.train_proposer_question_grpo \
  --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml \
  --validate-only
```

### Agent 6 Local Result

Status on 2026-05-07: complete for no-GPU local validation.

- Implemented `grpo_math/trl/train_proposer_question_grpo.py`.
- Added `grpo_math/configs/train_proposer_question_grpo.yaml`.
- Added `grpo_math/configs/train_proposer_question_grpo_smoke.yaml`.
- Added `tests/test_train_proposer_question_grpo.py`.
- Supports offline/dry-run proposer training data from canonical rollout JSONL
question rewards.
- Loads proposer prompts/groups, carries filtered-out questions as
proposer-trainable zero-reward samples, and normalizes question advantages
within proposer groups.
- Emits validation metrics for acceptance rate, zero-reward rate, duplicate
rate, mixed-verdict rate, and mean question reward.
- Online generated-question evaluation is intentionally guarded until the
shared filter, solver, verifier, and rollout provider interfaces are wired
into the training-time path.
- Local command passed with `python3`:
`python3 -m pytest tests/test_train_proposer_question_grpo.py` (6 passed).
- Local command passed with `python3` after generating fixture rollouts:
`python3 -m grpo_math.trl.train_proposer_question_grpo --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml --validate-only`.
- Latest smoke validation metrics from `outputs/self_play_fixture/rollouts.jsonl`:
`acceptance_rate=0.6`, `zero_reward_rate=0.4`, `duplicate_rate=0.2`,
`mixed_verdict_rate=0.0`, `mean_question_reward=0.4`, `num_examples=5`.
- Real GRPO training was not executed locally because this machine currently has
no GPU and the active Python environment does not have `torch` installed.

## Agent 7: Iteration And Loop Orchestrator

### Mission

Build the commands that run one iteration and multiple iterations.

### Files

- Create `grpo_math/self_play/run_self_play_iteration.py`
- Create `grpo_math/self_play/run_self_play_loop.py`
- Create `tests/test_run_self_play_iteration.py`
- Create `tests/test_run_self_play_loop.py`

### Tasks

- Run rollout collection.
- Run solver GRPO.
- Run proposer GRPO.
- Run evaluation.
- Hand checkpoints forward.
- Support `--dry-run`, `--validate-only`, `--skip-train`, and `--resume`.
- Write `report.md`.
- Write `next_iteration.yaml`.
- Generate exact Wash U commands.

### Local Checks

```bash
python -m pytest tests/test_run_self_play_iteration.py tests/test_run_self_play_loop.py
python -m grpo_math.self_play.run_self_play_iteration \
  --config grpo_math/configs/self_play_verdict_grpo_orchestrator_fixture.yaml \
  --dry-run \
  --output-dir /tmp/self_play_local_dry_run \
  --overwrite
```

### Agent 7 Local Result

Status on 2026-05-07: complete for no-GPU local validation.

- Implemented `grpo_math/self_play/run_self_play_iteration.py`.
- Implemented `grpo_math/self_play/run_self_play_loop.py`.
- Added `grpo_math/configs/self_play_verdict_grpo_orchestrator_fixture.yaml`
for loop orchestration; kept `grpo_math/configs/self_play_verdict_grpo_fixture.yaml`
as the rollout collector fixture config.
- Added `tests/test_run_self_play_iteration.py`.
- Added `tests/test_run_self_play_loop.py`.
- The iteration runner plans rollout collection, solver GRPO, proposer GRPO, and
evaluation using the real local CLI argument shapes.
- The runner writes `commands.jsonl`, `report.md`, and `next_iteration.yaml`.
- `next_iteration.yaml` increments the iteration id and forwards solver/proposer
checkpoint directories for the next iteration.
- The loop runner chains multiple dry-run iterations through each previous
`next_iteration.yaml` and writes `loop_report.md`.
- Supported local controls are `--dry-run`, `--validate-only`, `--skip-train`,
`--resume`, and `--overwrite`.
- Local CPU-safe checklist passed with `python3`: 95 tests passed across reward
math, question rewards, filter, verifier, rollout collection, solver/proposer
validation, and Agent 7 orchestration tests.
- Fixture rollout collection passed locally:
`python3 -m grpo_math.self_play.collect_self_play_rollouts --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml --overwrite`.
- Solver validate-only passed locally:
`python3 -m grpo_math.trl.train_solver_verdict_grpo --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml --validate-only`.
- Proposer validate-only passed locally:
`python3 -m grpo_math.trl.train_proposer_question_grpo --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml --validate-only`.
- One-iteration dry-run passed locally:
`python3 -m grpo_math.self_play.run_self_play_iteration --config grpo_math/configs/self_play_verdict_grpo_orchestrator_fixture.yaml --dry-run --output-dir /tmp/self_play_local_dry_run --overwrite`.
- Two-iteration loop dry-run passed locally:
`python3 -m grpo_math.self_play.run_self_play_loop --config grpo_math/configs/self_play_verdict_grpo_orchestrator_fixture.yaml --dry-run --output-dir /tmp/self_play_local_loop_dry_run --overwrite`.
- The broader legacy test command that includes `tests/test_logprobs.py` and
`tests/test_left_padding_masking.py` does not collect in this local Python
environment because `torch` is not installed.
- Real training and real evaluation were not executed locally because this
machine currently has no GPU and the active Python environment does not have
`torch` installed.

## Agent 8: Wash U Server Prep

### Mission

Make the project easy to run on the Wash U server.

### Files

- Create `scripts/washu_setup.sh`
- Create `scripts/washu_run_smoke.sh`
- Create `scripts/washu_run_iteration.sh`
- Create `WASHU_RUNBOOK.md`

### Tasks

- Document environment setup.
- Document required env vars/API keys.
- Add dependency install commands.
- Add sanity checks:
  - Python version
  - CUDA availability
  - GPU count
  - disk space
  - API key presence
- Add smoke commands:
  - rollout only
  - solver GRPO one step
  - proposer GRPO one step
  - full iteration smoke
- Add resume instructions.
- Add where outputs/checkpoints are written.

### Local Checks

```bash
bash scripts/washu_setup.sh --dry-run
bash scripts/washu_run_smoke.sh --dry-run
```

## Suggested Parallel Execution

Start immediately:

- Agent 1: schemas and reward math
- Agent 2: question filter
- Agent 3: verifier interface

Start once Agent 1 and Agent 2 have stable interfaces:

- Agent 4: full rollout collector
- Agent 8: Wash U server prep

Start once Agent 4 defines rollout outputs:

- Agent 5: solver GRPO
- Agent 6: proposer GRPO

Start once Agents 4, 5, and 6 expose commands:

- Agent 7: iteration and loop orchestrator

## Local Pre-Server Checklist

```bash
ruff check .
python -m pytest \
  tests/test_reward.py \
  tests/test_verdict_rewards.py \
  tests/test_question_rewards.py \
  tests/test_question_filter.py \
  tests/test_verifier.py \
  tests/test_collect_self_play_rollouts.py \
  tests/test_train_solver_verdict_grpo.py \
  tests/test_train_proposer_question_grpo.py \
  tests/test_run_self_play_iteration.py \
  tests/test_run_self_play_loop.py
python -m grpo_math.self_play.run_self_play_iteration \
  --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml \
  --dry-run \
  --output-dir /tmp/self_play_local_dry_run \
  --overwrite
bash scripts/washu_run_smoke.sh --dry-run
```

### Local Validation Result

Status on 2026-05-07:

- Agent 4 rollout collection is implemented in
`grpo_math/self_play/collect_self_play_rollouts.py` with fixture proposer,
solver, and verifier providers, deterministic pre-solver filtering, accepted
and rejected question rows, solver rewards/advantages, proposer question
rewards/advantages, canonical `self_play_verdict_grpo_v1` output, and
`rejected_questions.jsonl`.
- Agent 4 tests passed with `python3`:
`python3 -m pytest tests/test_collect_self_play_rollouts.py` reported
3 passed.
- The Agent 4 fixture dry-run passed:
`python3 -m grpo_math.self_play.collect_self_play_rollouts --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml`
collected 5 questions, accepted 3, rejected 2, and wrote
`outputs/self_play_fixture/rollouts.jsonl` plus
`outputs/self_play_fixture/rejected_questions.jsonl`.
- Filtered-out Agent 4 fixture rows are working as intended: they keep
`question_reward = 0.0`, `trainable_for_solver = false`,
`trainable_for_proposer = true`, no solver completions, and structured filter
reasons such as `duplicate` and `answer_leak`.
- Pre-server no-GPU pytest subset passed with `python3`: 95 tests passed.
- Solver GRPO validate-only passed and printed the server command.
- Proposer GRPO validate-only passed and printed fixture metrics.
- Fixture rollout collector passed and wrote rollout/rejected-question JSONL
under `outputs/self_play_fixture/`.
- Full iteration dry-run passed and wrote planned commands/report files under
`/tmp/self_play_local_dry_run`.
- `bash scripts/washu_setup.sh --dry-run` passed; it correctly reported no
local CUDA/GPU and missing local `torch`.
- `bash scripts/washu_run_smoke.sh --dry-run` passed.
- Full `python3 -m pytest tests` did not complete because
`tests/test_logprobs.py` and `tests/test_left_padding_masking.py` import
`torch`, which is not installed in the current local Python environment.
- `ruff check .` could not run because `ruff` is not installed on PATH in the
current local Python environment; IDE diagnostics showed no linter errors for
the Agent 4 files.
- No GPU training was attempted locally.

## Wash U First Run Order

1. `bash scripts/washu_setup.sh`
2. `bash scripts/washu_run_smoke.sh`
3. Solver GRPO one-step smoke.
4. Proposer GRPO one-step smoke.
5. API-backed rollout smoke.
6. One full self-play iteration.
7. Inspect reports/checkpoints.
8. Launch multi-iteration run only after smoke outputs look sane.

