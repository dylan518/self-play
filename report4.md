# Report 4 - 50Q Verification-Focused Rollout

## Run Definition

This report summarizes the latest 50-question rollout using the answer-verification setup.

- Output file: `outputs/pairwise_rollouts_debug/samples_functional_smoke_50_answer_verify.jsonl`
- Total rows: `50` (assembled as 5 appended chunks of 10)
- Unique run IDs: `5`

Configuration used for this 50Q execution:

- Generator: `gemini-2.5-flash`
- Solver strong group: `gemini-2.5-flash` (`temperature=0.2`, `top_p=0.85`)
- Solver weak group: `gemini-2.0-flash` (`temperature=2.0`, `top_p=1.0`)
- Judge: `gpt-5.2` in `single_verify` mode
- Verifier prompt: direct candidate-answer checking with required structured output
- `repeats_per_solution=1` (chosen for runtime/stability at 50Q scale)

---

## Core Rollout Metrics

From `tests/analyze_rollout_results.py`:

- Questions: `50` total, `41` unique (`18%` duplicate rate)
- Oracle answer `None`: `1`
- Parse rate: `398/500 = 80.0%`

Oracle solve rates:

- Overall: `177/500 = 35.4%`
- Group 0 (strong): `143/250 = 57.2%`
- Group 1 (weak): `34/250 = 13.6%`
- Questions with >=1 correct solution: `37/49 = 75.5%`

Verifier vs oracle:

- TP=`105`, FP=`41`, TN=`272`, FN=`72` (`total=490`, `skipped=10`)
- Accuracy: `76.9%`
- Precision: `71.9%`
- Recall: `59.3%`

---

## Separation Signal (`R_sep`)

Question-level strong-minus-weak verifier separation:

- Mean `R_sep = +0.232`
- Positive fraction: `54%` (`27/50`)
- Script conclusion: strong group reliably outscores weak group

Interpretation:

- The strong/weak solver gap remains clearly visible at 50Q.
- `R_sep` remains positive in aggregate and suitable as a training signal.

---

## Judge Variance and Voting Signal

Because `repeats_per_solution=1` in this run:

- Disagreement/variance across votes is not measurable here by design.
- Aggregate confidence is always `1.0` and not useful for calibration in this file.

To evaluate vote-variance utility, we compared earlier 3-vote runs:

- `samples_functional_smoke_10_answer_verify.jsonl`:
  - Unanimous accuracy: `94.25%` (`87` samples)
  - Non-unanimous accuracy: `84.62%` (`13` samples)
  - Gap: `+9.64` percentage points
- `samples_functional_smoke.jsonl` (older/noisier setup):
  - Unanimous accuracy: `75.74%`
  - Non-unanimous accuracy: `45.16%`
  - Gap: `+30.58` percentage points

Conclusion:

- Additional judge votes provide meaningful reliability signal via unanimity/disagreement.
- Unanimous decisions are consistently more accurate than split decisions.

---

## Model Confidence Utility

Confidence remained weakly informative to negatively informative:

- 10Q better setup: Pearson(`model_confidence_mean`, correctness) = `-0.04` (near zero)
- Older 20Q setup: `-0.31` (negative)
- 50Q run (single vote): `-0.15` (negative trend)

Conclusion:

- Raw model confidence is not currently a reliable positive weighting signal.
- Vote agreement (when repeats are enabled) is more trustworthy than confidence.

---

## Practical Recommendation for RL Signal

For generator RL targeting discriminative signal:

1. Keep `R_sep` as the primary optimization target.
2. Keep single judge vote for throughput in large runs.
3. Add selective multi-vote verification on uncertain/high-impact samples to recover variance signal.
4. Use unanimity/disagreement as a reliability feature; do not rely heavily on raw confidence.
