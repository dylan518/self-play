<!-- orchestrate handoff
task: diagnose-signal-biases
branch: orch/self-play-rl/diagnose-signal-biases
agentId: bc-27f1371b-1fb2-4353-967a-b1af02049509
runId: run-85ac1220-1c5f-45d5-8f5c-8303738ef79d
resultStatus: finished
finishedAt: 2026-05-09T19:48:42.481Z
-->

## Status
partial

## Branch
`orch/self-play-rl/diagnose-signal-biases`

## What I did
- Added `tests/diagnose_signal_biases.py` as a standalone offline diagnostics CLI that accepts:
  - `--jsonl path1 [path2 ...]`
  - direct script execution (`python tests/diagnose_signal_biases.py ...`) and module execution (`python -m tests.diagnose_signal_biases ...`).
- Implemented required analyses with schema-tolerant handling (missing/older `oracle`, `solution_verifications`, and pairwise fields are skipped safely):
  - `verdict_bias` (overall, per-file, per-question; compared to oracle, with bias gap)
  - `parse_bias` (per sampling group parse rate + oracle solve rate)
  - `unanimity_bias` (unanimous-CORRECT / unanimous-INCORRECT / split-votes confusion matrices + precision/recall)
  - `group_effects` (per-group oracle rate, verifier-correct rate, parse rate, R_sep contribution, ranked)
  - optional `position_bias` from pairwise A/B prefs when present.
- Ran diagnostics on:
  - `outputs/pairwise_rollouts_debug/samples_functional_smoke_50_answer_verify.jsonl`
  - `outputs/pairwise_rollouts_debug/samples_functional_smoke_10_answer_verify.jsonl`
- Committed generated report at `outputs/diagnostics/signal_biases.md` (force-added due `outputs/` ignore).
- Verified new script lint-clean with `python3 -m ruff check tests/diagnose_signal_biases.py`.

## Measurements
- `python3 tests/diagnose_signal_biases.py --jsonl outputs/pairwise_rollouts_debug/samples_functional_smoke_50_answer_verify.jsonl outputs/pairwise_rollouts_debug/samples_functional_smoke_10_answer_verify.jsonl exit_code: nonzero → 0`
- `python3 -m tests.diagnose_signal_biases --jsonl outputs/pairwise_rollouts_debug/samples_functional_smoke_50_answer_verify.jsonl outputs/pairwise_rollouts_debug/samples_functional_smoke_10_answer_verify.jsonl exit_code: nonzero → 0`
- `outputs/diagnostics/signal_biases.md lines: 0 → 111`
- `verdict bias gap (P(verdict=CORRECT) - oracle vote-weighted): n/a → -5.1%pp`
- `parse rate by sampling group (g0 vs g1): n/a → 87.0% vs 71.3% (Δ=15.7%pp)`
- `accuracy (split votes vs all unanimous): n/a → 84.6% vs 79.9%`
- `group ranking by mean R_sep contribution: n/a → g0 (+23.3%) > g1 (-23.3%)`
- `python3 -m ruff check . total_errors: 0 <= 16`  

## Verification
unit-test-verified

## Notes, concerns, deviations, findings, thoughts, feedback
- Headline findings from the committed report:
  - Verdict bias is **negative** on these artifacts: judge does **not** over-grant CORRECT overall; gap is **-5.1%pp** vs vote-weighted oracle (solution-level gap **-6.5%pp**).
  - Clear parse bias across groups: **group 1 parse rate is 15.7%pp lower** than group 0 (71.3% vs 87.0%), with much lower oracle solve rate (14.3% vs 54.3%).
  - Unanimity regimes:
    - `unanimous_CORRECT`: 74.5% precision (42 FP out of 165 predictions), recall 100%.
    - `unanimous_INCORRECT`: strong TN concentration (82.0% accuracy in that regime).
    - `split_votes`: small sample (n=13) but 84.6% accuracy.
  - Group effects rank by R_sep contribution: **group 0 first (+23.3%)**, **group 1 second (-23.3%)**.
- Optional position bias could not be estimated from these two files because they have no pairwise A/B judgment rows.
- Deviation/blocker: repo-wide `ruff check .` currently fails on pre-existing files outside allowed edit scope (`grpo_math/**`, `tests/analyze_rollout_results.py`), so I did not modify those prohibited paths.

## Suggested follow-ups
- If planner wants full acceptance on lint, publish a separate task to resolve existing repo-wide Ruff violations in `grpo_math/**` and legacy test scripts.
- Run this new diagnostic on additional committed rollout JSONLs that include pairwise comparisons to populate the optional position-bias section with real A/B-side metrics.
- Add a small regression test fixture for `tests/diagnose_signal_biases.py` (older schema + current schema rows) to lock in backward-compatible parsing behavior.