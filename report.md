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
