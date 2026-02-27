# Report 3 - 10Q Answer-Verification Run

## Run Definition

This report summarizes the latest verification-focused smoke run:

- Output file: `outputs/pairwise_rollouts_debug/samples_functional_smoke_10_answer_verify.jsonl`
- Questions analyzed: `10`
- Solutions per question: `10` (5 strong-group + 5 weak-group)
- Verifier repeats per solution: `3`

Config highlights:

- Generator: `gemini-2.5-flash`
- Solver strong group: `gemini-2.5-flash`, `temperature=0.2`, `top_p=0.85`
- Solver weak group: `gemini-2.0-flash`, `temperature=2.0`, `top_p=1.0`
- Judge (verifier): `gpt-5.2` with `api_reasoning_effort: medium`
- Judge mode: `single_verify`
- Verifier prompt: answer-checking style (verify candidate answer against constraints, return `VERDICT/CONFIDENCE/REASONING`)

---

## Core Metrics

From `tests/analyze_rollout_results.py`:

- Parse rate: `77/100 = 77.0%`
- Oracle solve rate overall: `29/100 = 29.0%`
- Group 0 solve rate (strong): `20/50 = 40.0%`
- Group 1 solve rate (weak): `9/50 = 18.0%`

Verifier vs oracle:

- TP=24, FP=2, TN=69, FN=5
- Accuracy: `93.0%`
- Precision: `92.3%`
- Recall: `82.8%`

---

## Separation Signal (`R_sep`)

Question-level `R_sep` (mean verifier score of group 0 minus group 1):

- Mean `R_sep`: `+0.220`
- Fraction positive: `40%` (4/10)
- Pearson `r(r_sep, oracle_acc)`: `+0.254` (small positive, low-n)

Interpretation:

- Separation signal is materially stronger than the previous configuration and now aligned with expected strong-vs-weak behavior on multiple questions.

Approximate significance check for mean `R_sep` (normal approximation, n=10):

- Mean: `+0.220`, SE: `0.116`
- One-sided p-value (H1: mean > 0): `~0.029`
- Two-sided p-value: `~0.058`
- 95% CI: `[-0.007, +0.447]`

This is borderline at n=10; direction is promising but sample size is still small.

---

## What Changed vs Prior 20Q Run

Compared to the previous 20-question smoke run (`samples_functional_smoke.jsonl`):

- Verifier accuracy improved from `71.0%` -> `93.0%`
- Verifier recall improved from `33.3%` -> `82.8%`
- Mean `R_sep` improved from `+0.037` -> `+0.220`

Most likely driver:

- Switching verifier behavior to direct candidate-answer checking with reasoning enabled, using a strong judge model (`gpt-5.2`).

---

## Remaining Gaps

- Parse rate is still only `77%`; this still removes usable signal.
- n=10 is small for confidence around `R_sep` significance.
- Some questions still show no separation (`R_sep=0`) or weak inversion on individual cases.

---

## Recommendation

Next step is a larger run (`n=30` or `n=50`) with this exact verifier setup to stabilize significance estimates while preserving the improved recall/precision profile.
