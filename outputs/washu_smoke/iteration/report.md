# Self-Play Iteration 0

- Config: `grpo_math/configs/self_play_verdict_grpo_fixture.yaml`
- Output directory: `outputs/washu_smoke/iteration`
- Mode: `dry-run`
- Generated at: `2026-05-07T15:29:13.383686+00:00`

## Stage Status

- `rollout`: planned
- `solver_grpo`: planned
- `proposer_grpo`: planned
- `evaluation`: planned

## Commands

- `rollout`: `/opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.self_play.collect_self_play_rollouts --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml --overwrite`
- `solver_grpo`: `/opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.trl.train_solver_verdict_grpo --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml --output_dir outputs/washu_smoke/iteration/solver`
- `proposer_grpo`: `/opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.trl.train_proposer_question_grpo --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml --output_dir outputs/washu_smoke/iteration/proposer`
- `evaluation`: `/opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.eval.eval_gsm8k --config grpo_math/configs/train_smoke_trl.yaml --checkpoint outputs/washu_smoke/iteration/solver`

## Exact Wash U Commands

```bash
DRY_RUN=1 /opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.self_play.collect_self_play_rollouts --config grpo_math/configs/self_play_verdict_grpo_fixture.yaml --overwrite
DRY_RUN=1 /opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.trl.train_solver_verdict_grpo --config grpo_math/configs/train_solver_verdict_grpo_smoke.yaml --output_dir outputs/washu_smoke/iteration/solver
DRY_RUN=1 /opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.trl.train_proposer_question_grpo --config grpo_math/configs/train_proposer_question_grpo_smoke.yaml --output_dir outputs/washu_smoke/iteration/proposer
DRY_RUN=1 /opt/homebrew/opt/python@3.10/bin/python3.10 -m grpo_math.eval.eval_gsm8k --config grpo_math/configs/train_smoke_trl.yaml --checkpoint outputs/washu_smoke/iteration/solver
```

