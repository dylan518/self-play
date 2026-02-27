# Self-Play Experiments — Consolidated Report

This document consolidates and standardizes findings from:
- `report.md` (pairwise/Elo rollout experiment)
- `report2.md` (follow-up pairwise/Elo refinements)
- `report3.md` (absolute verification / confidence scoring + R_sep)

It is meant to be the single up-to-date reference for what was tried, what worked, and what to run next.

---

## 1) Experiment family A: Pairwise judging + Elo ranking

# Pairwise Rollout Experiment Report

## Scope

This report summarizes the self-play pairwise rollout experiments run so far with:

- Generator/Solver/Judge model family: `Qwen/Qwen3-4B`
- Oracle verifier: `gpt-4.1` (OpenAI)
- Ranking: Elo over repeated pairwise A/B judgments

## Configuration Trends

- Switched from 0.6B to 4B for better solution/judging quality.
- Increased solver context budget up to `max_new_tokens: 16384`.
- Forced A/B judge output (no TIE option).
- Added random A/B presentation remap to remove direct side bias.
- Added balanced A/B ordering per pair (newest fix) to reduce positional artifacts.
- Added parser guard for pathological giant integers to avoid run crashes.

## Main Experiments

### Experiment A: 20-question oracle-aligned evaluation

- Questions: `20`
- Solutions per question: `5`
- Pairs per question: `10` (all pairs for 5 solutions)
- Repeats per pair: `4`
- Total votes: `20 * 10 * 4 = 800`

Observed metrics:

- Elo top-1 accuracy vs oracle: `1/20 = 5%`
- Majority vote accuracy: `2/20 = 10%`
- Any correct candidate present: `4/20 = 20%`

Takeaway:

- Majority voting outperformed Elo top-1 in this run.
- Candidate generation quality was low (few questions had any correct candidate).

### Experiment B: Extended run (target 100, completed 96)

- Questions completed: `96` (run aborted before 100)
- Solutions per question: `5`
- Pairs per question: `10`
- Repeats per pair: `4`
- Total votes observed: `3840`

Observed metrics:

- Elo top-1 accuracy vs oracle: `7/96 = 7.29%`
- Majority vote accuracy: `11/96 = 11.46%`
- Any correct candidate present: `14/96 = 14.58%`

Failure:

- Run crashed due to a very long digit string in `FINAL_ANSWER` parse
  (`ValueError` from Python int conversion limit).
- Parser has now been patched to treat this as invalid (`None`) instead of crashing.

### Experiment C: Focused 5-question rerun

- Questions: `5`
- Solutions per question: `5`

Observed metrics:

- Any correct candidate present: `0/5`
- Elo top-1 accuracy: `0/5`
- Majority vote accuracy: `0/5`

Takeaway:

- This subset could not evaluate judge discrimination because no correct solutions were produced.

## Judge Bias Diagnostics

On a completed 96-row set (before balanced-order fix):

- Mixed correct-vs-incorrect votes analyzed: `224`
- Correct pick rate on mixed votes: `41.96%`
- Incorrect pick rate on mixed votes: `58.04%`

Key positional signal:

- Correct answers were presented as A only `43.75%` of mixed votes.
- Raw judge selected A `60.71%` of mixed votes.
- This combination can systematically favor incorrect answers.

Action taken:

- Implemented balanced A/B order per pair repeat to neutralize side exposure.

## Current State

- Core Elo/pairwise plumbing checks out (no remap mismatches in sampled diagnostics).
- Primary bottlenecks are:
  1) low candidate correctness rate from solver outputs on hard tasks,
  2) judge reliability under noisy, mostly-incorrect candidate pools,
  3) previous positional bias (now patched).

## Next Suggested Experiment

Run a new balanced-order batch with enough questions/solutions to produce mixed pairs:

- Questions: `10-20`
- Solutions per question: `10`
- Keep oracle enabled
- Recompute:
  - mixed-pair correct pick rate,
  - Elo top-1 vs majority vs oracle,
  - hit@k and MRR for first oracle-correct candidate by Elo rank.


---

## 2) Experiment family B: Pairwise judging + Elo (refinements)

# Report 2 - Exact Experimental Results

## Dataset and Run Definition

All results in this report are from:

- Output file: `outputs/pairwise_rollouts_debug/samples_clean_restart.jsonl`
- Combined runs: 2 appended runs of 10 questions each
- Total rows analyzed: `20` (question indices `0..19`)
- Oracle rows with valid answer/error-free: `20`

Configuration used:

- Generator/Solver/Judge: `Qwen/Qwen3-4B`
- Oracle verifier: `gpt-4.1`
- Solutions per question: `10`
- Pairwise pairs per question: `10` (subsampled, not full 45)
- Repeats per pair: `4`
- Solver max_new_tokens: `16384`
- Judge A/B handling: forced-choice with balanced A/B presentation per pair repeat

---

## Core Accuracy Metrics

Across all 20 oracle-valid questions:

- Elo top-1 accuracy: `5/20 = 25.00%`
- Majority-vote accuracy: `5/20 = 25.00%`
- Rows with at least one oracle-correct candidate: `6/20 = 30.00%`

Conditioned on rows with at least one correct candidate:

- Elo top-1 accuracy (given any correct exists): `5/6 = 83.33%`
- Majority accuracy (given any correct exists): `5/6 = 83.33%`

Interpretation:

- Elo and majority perform the same on this dataset.
- Main bottleneck is candidate generation correctness (correct candidate present in only 30% of questions).

---

## Correct-vs-Incorrect Pairwise Discrimination

Analyzed only mixed pairs (one oracle-correct candidate vs one oracle-incorrect candidate).

Exact counts:

- Mixed pairs: `17`
- Mixed votes: `68` (4 votes per mixed pair)

Vote-level metric (primary):

- Judge picked correct candidate in mixed votes: `48/68 = 70.59%`

Pair-level majority on mixed pairs:

- Majority-correct: `8`
- Majority-wrong: `1`
- Majority-tie (2-2 split): `8`
- Majority-correct over all mixed pairs: `8/17 = 47.06%`
- Majority-correct on non-tie mixed pairs only: `8/9 = 88.89%`

Key point:

- Vote-level discrimination is clearly above random (`70.59%` vs `50%` baseline).
- Pair-level result is dragged down by many ties.

---

## Agreement vs Correctness (Calibration Signal)

Consistency bucket analysis on mixed pairs:

- `consistency = 0.50` (2-2 split):
  - Pairs: `8`
  - Vote-level correct rate: `50.00%`
- `consistency = 1.00` (4-0 unanimous):
  - Pairs: `9`
  - Vote-level correct rate: `88.89%`

Conclusion:

- Higher agreement strongly correlates with higher correctness.
- Split decisions are approximately random, unanimous decisions are usually reliable.

---

## Notes on Data Quality and Setup

- Previous below-random behavior was linked to positional effects before balanced A/B handling.
- Current setup includes balanced A/B presentation and no-TIE output.
- Output persistence was fixed:
  - append-capable JSONL
  - run_id tagging
  - overwrite no longer default in debug flow



---

## 3) Experiment family C: Absolute verification (confidence) + separation signal

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


---

## 4) Cross-experiment conclusions

### What failed / was unstable
- Pairwise judging produced a noisy training signal; Elo tended to be dominated by judge noise and/or weak candidates.
- Self-play risk: systems optimize against judge artifacts; ties/near-ties are especially brittle.

### What worked
- Switching to an *answer-checking verifier* that outputs `VERDICT/CONFIDENCE/REASONING` produced a much cleaner signal.
- In the absolute verification setup, verifier-vs-oracle metrics were strong (accuracy ~93%, precision ~92%, recall ~83% in the n=10 run).
- `R_sep` improved materially versus earlier settings (mean +0.220 vs +0.037), with borderline significance at n=10.

### Current bottlenecks
- Parse rate is still a major limiter (77% in the best run described).
- n is small for strong conclusions about `R_sep` and its correlation with oracle accuracy.

---

## 5) Recommended next run (Experiment 4)

**Goal:** validate `R_sep` and train with a stable scalar reward.

- Increase question count to **n=30–50**.
- Keep the absolute verifier prompt (answer-checking) and measure verifier-vs-oracle FP/FN drift.
- Use multiple solver samples per question and compute a within-question baseline (advantage) to stabilize updates.
- Use `R_cons` and `R_sep` as **filters/weights** for generator training; use verifier confidence as the **primary solver reward**.

Metrics to log:
- parse rate
- oracle solve rate overall + by group
- verifier confusion matrix vs oracle
- distribution of verifier confidence
- `R_sep` distribution + fraction positive + correlation with oracle accuracy
