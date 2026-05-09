# self-play

This repo is an experimental self-play training loop for math reasoning models.
The current direction is a **verdict-only RL loop**: generated questions are
filtered, the solver samples multiple answers, a verifier judges each answer,
and the solver is trained from verifier-derived rewards.

## Current Goal

We are moving away from the earlier strong-vs-weak solver separation setup. The
next implementation should rely on direct verifier verdicts:

```text
proposer generates questions
  -> question filter keeps useful/checkable questions
  -> solver samples multiple answers per question
  -> verifier returns CORRECT/INCORRECT verdicts
  -> rewards and within-question advantages are computed
  -> solver is trained with verdict-derived RL signal
```

The first milestone is a no-GPU development path that gets the data shape,
question filtering, reward math, export tools, and smoke harness working before
running expensive training.

## Important Files

- `grpo_math/self_play/generate_pairwise_data.py` - current rollout generator.
Despite the name, it already supports `single_verify` verdict judging.
- `grpo_math/trl/train_grpo_trl.py` - current GRPO trainer, currently using
GSM8K ground truth rewards rather than generated-question verifier rewards.
- `grpo_math/data/reward.py` - strict `FINAL_ANSWER: <integer>` parsing and
binary GSM8K reward helpers.
- `grpo_math/prompts/question_generator_prompt.txt` - question proposer prompt.
- `grpo_math/prompts/single_solution_verify_prompt.txt` - verifier prompt.

## Planning Docs

- `VERDICT_RL_DESIGN.md` - high-level design for the simplified verdict-only
loop.
- `RL_LOOP_TODO.md` - broad implementation backlog.
- `EXACT_IMPLEMENTATION_TODO.md` - concrete step-by-step implementation list.
- `CPU_DEV_AGENT_PLAN.md` - parallel no-GPU development plan split by agent.

## Near-Term Implementation

The recommended first PR should stay small:

1. Add verdict reward and advantage utilities.
2. Add deterministic question filtering.
3. Add reward/filter fields to rollout JSONL rows.
4. Add a verdict-only smoke config.
5. Add tests for reward math and filtering.

After that, add rollout export, an iteration harness, and generated-question
training support.

## Useful Commands

```bash
ruff check .
python -m pytest tests/test_reward.py
python -m pytest tests/test_logprobs.py
python -m pytest tests/test_left_padding_masking.py
```

Existing rollout smoke command:

```bash
python -m grpo_math.self_play.generate_pairwise_data --config grpo_math/configs/pairwise_rollouts_smoke.yaml
```

Existing GRPO training command:

```bash
python -m grpo_math.trl.train_grpo_trl --config grpo_math/configs/train_gsm8kv2_trl.yaml
```

