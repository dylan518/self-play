# R_sep Experiment Report (Feb 19 v3)

**Date:** 2026-02-19
**Config:** `pairwise_rollouts_gemini25flash_single_verify_fast.yaml`
**Model:** Gemini 2.5 Flash (generator + solver + judge), GPT-5.2 (oracle)
**Run:** 20 questions × 10 solutions (5 strong @ temp=0.3, 5 weak @ temp=2.0) × 3 judge repeats = 600 verifications

---

## What Changed vs v2


| Parameter                 | v2                           | v3                                               |
| ------------------------- | ---------------------------- | ------------------------------------------------ |
| num_questions             | 10                           | **20**                                           |
| sampling groups           | 1 group (temp=1.0)           | **2 groups: strong (temp=0.3), weak (temp=2.0)** |
| solution_group field      | missing                      | **added to every solution**                      |
| r_sep field               | missing                      | **added to reliability dict**                    |
| skip non-parsed solutions | no (sent "NONE" to verifier) | **yes (auto-INCORRECT)**                         |
| judge temperature         | 0.3                          | 0.3                                              |


---

## Quality Baseline


| Metric            | Value                           |
| ----------------- | ------------------------------- |
| Unique questions  | 20/20 (0% duplicates)           |
| Oracle failures   | 0/20                            |
| Parse rate        | **97%** (194/200 — best so far) |
| Verifier accuracy | **82.5%**                       |
| Precision         | **85.4%**                       |
| Recall            | **90.8%**                       |


97% parse rate is the result of skipping non-parsed solutions before verification — the "NONE" candidate answers were causing noise.

---

## distinct_correct as Verifiability Signal (Confirmed at Scale)


| Bucket | Questions | Avg FP rate |
| ------ | --------- | ----------- |
| dc = 0 | 1         | 0%          |
| dc = 1 | 15        | 7%          |
| dc ≥ 2 | 4         | **44%**     |


**Pearson r (distinct_correct vs fp_rate): +0.327** — meaningful positive signal across 20 questions.

Questions with `distinct_correct > 1` have 6× higher false positive rate than dc=1 questions. This filter is reliable and oracle-free.

4 questions had dc > 1: Q6 (2021), Q8 (722), Q18 (49), Q19 (6). These are questions where the verifier accepted multiple distinct wrong answers — the verifier is judging reasoning style, not answer correctness.

---

## R_sep Results: Temperature Gap Not Sufficient

### Per-question R_sep (strong group mean - weak group mean)


| Q   | Oracle | g0 mean | g1 mean | R_sep     | Note                        |
| --- | ------ | ------- | ------- | --------- | --------------------------- |
| Q0  | 16     | 0.00    | 0.60    | -0.60     | Verifier FP on weak answers |
| Q1  | 2      | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q2  | 800    | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q3  | 800    | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q4  | 4      | 0.93    | 0.80    | **+0.13** | Weak signal                 |
| Q5  | 14     | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q6  | 2021   | 0.80    | 1.00    | -0.20     | Verifier FP on weak answers |
| Q7  | 34     | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q8  | 722    | 0.40    | 0.40    | 0.00      | Both noisy                  |
| Q9  | 12     | 1.00    | 0.80    | **+0.20** | ✓                           |
| Q10 | 50     | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q11 | 46     | 0.07    | 0.80    | -0.73     | Verifier FP on weak answers |
| Q12 | 722    | 0.20    | 0.33    | -0.13     | Both noisy                  |
| Q13 | 800    | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q14 | 7      | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q15 | 0      | 0.00    | 0.20    | -0.20     | Verifier FP on weak answers |
| Q16 | 11     | 0.00    | 0.00    | 0.00      | Both fail                   |
| Q17 | 75     | 1.00    | 1.00    | 0.00      | Both perfect                |
| Q18 | 49     | 0.80    | 1.00    | -0.20     | Verifier FP on weak answers |
| Q19 | 6      | 1.00    | 1.00    | 0.00      | Both perfect                |


**Mean R_sep = -0.087** | **Fraction positive = 10%**

### Why R_sep is Noisy Here

Two root causes:

**1. Most questions are too easy for temp=2.0 to fail.** Simple arithmetic (answers like 2, 4, 6, 7, 12, 34, 50...) is solved correctly by Gemini at any temperature. Temperature noise doesn't cause arithmetic errors on single-step or two-step problems — it mainly causes formatting drift.

**2. Verifier false positives on high-temperature outputs.** At temp=2.0, weak group solutions are more verbose and confidently wrong. The verifier accepts some of these (Q0: -0.60, Q11: -0.73, Q6: -0.20). This inverts R_sep: weak group gets HIGHER verifier scores than strong group despite being wrong.

---

## Key Finding: R_sep Correlates with Oracle Accuracy Despite Noise

**Pearson r (R_sep vs oracle_accuracy per question): +0.717**

Despite the mean being negative, R_sep measured by the verifier IS correlated with oracle accuracy across questions:

- Questions where strong group outscores weak (R_sep > 0) → oracle accuracy is highest
- Questions where weak group outscores strong (R_sep < 0) → oracle accuracy is lower (FP inflation)

This validates the core hypothesis: **R_sep is a signal about question quality**, not just sampling quality. It identifies questions where the verifier is reliable enough to distinguish focused vs noisy sampling. The filter `R_sep > 0 AND distinct_correct == 1` would select the most trustworthy training questions without oracle labels.

---

## Confidence Calibration (Improved)

`agg confidence` now has 2 distinct values (`{0.67, 1.0}`) — judge temp=0.3 is producing some genuine 2/3 disagreements. Previously always 1.0.


| Confidence bucket | n   | Accuracy                        |
| ----------------- | --- | ------------------------------- |
| 0.0               | 6   | 100.0% (auto-marked non-parsed) |
| 0.5               | 8   | 87.5%                           |
| 0.7               | 3   | 66.7%                           |
| 0.8               | 6   | 83.3%                           |
| 0.9               | 3   | 66.7%                           |
| 1.0               | 174 | 82.2%                           |


Confidence is near-flat across all buckets — the judge doesn't distinguish hard vs easy. Persistent negative Pearson r (-0.071) between stated confidence and actual accuracy.

---

## Metric Progression


| Metric                     | Feb 17 | Feb 19 v2 | Feb 19 v3       |
| -------------------------- | ------ | --------- | --------------- |
| Questions                  | 10     | 10        | **20**          |
| Duplicate rate             | 30%    | 0%        | **0%**          |
| Parse rate                 | 32%    | 67%       | **97%**         |
| Oracle failures            | all    | 0         | **0**           |
| Verifier accuracy          | N/A    | 78.3%     | **82.5%**       |
| Precision                  | N/A    | 69.2%     | **85.4%**       |
| distinct_correct Pearson r | —      | —         | **+0.327**      |
| R_sep mean                 | —      | +0.020    | -0.087 (noisy)  |
| R_sep vs oracle Pearson r  | —      | -0.222    | **+0.717**      |
| agg confidence range       | {1.0}  | {1.0}     | **{0.67, 1.0}** |


---

## Conclusions and Next Steps

### What's Working

- **Parser**: 97% parse rate, stable
- **Oracle**: 0 failures across 20 questions
- **distinct_correct filter**: Reliable oracle-free quality gate. dc > 1 → 44% FP rate. dc = 1 → 7% FP rate.
- **Sampling group tracking**: R_sep computed and stored per question
- **R_sep → oracle correlation**: +0.717 validates R_sep as a question-quality proxy

### What R_sep Needs to Work as a Training Signal

Temperature gap alone is insufficient because:

1. Simple arithmetic questions are too easy — temp=2.0 still solves them correctly
2. Verifier FPs on high-temp verbose outputs invert R_sep on some questions

**Fix: Use a genuinely weaker model for group 1.** Capability gap > temperature gap:

```yaml
sampling_groups:
  - count: 5
    temperature: 0.3          # strong group: gemini-2.5-flash focused
  - count: 5
    temperature: 1.0
    api_model: "gemini-2.0-flash-lite"  # weak group: smaller model
```

The model swap creates a stable capability gap on questions at the right difficulty level, independent of whether temp=2.0 happens to output correctly.

### Action Items


| Priority | Item                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| High     | Switch weak group to `gemini-2.0-flash-lite` (model capability gap vs temperature noise)                           |
| High     | Filter training questions by `distinct_correct == 1 AND R_sep > 0` — oracle-free quality gate                      |
| Medium   | Generate harder questions (3-5 step computation, not pure single-step arithmetic) — creates genuine difficulty gap |
| Medium   | Investigate conf calibration — judge states 1.0 for 87% of verifications regardless of difficulty                  |
