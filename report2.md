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

