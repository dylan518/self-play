# Single-Verify Experiment Report
**Date:** 2026-02-19
**Config:** `pairwise_rollouts_gemini25flash_single_verify_fast.yaml`
**Model:** Gemini 2.5 Flash (generator + solver + judge), GPT-5.2 (oracle)
**Run:** 10 questions × 10 solutions × 3 judge repeats = 300 verifications

---

## Setup Changes vs Prior Runs (Feb 17)

| Parameter | Before | After |
|---|---|---|
| Generator temperature | 0.7 | 1.1 |
| Duplicate guard | none | dedup retry loop |
| Oracle max tokens | 64 (no CoT) | 1024 (step-by-step) |
| Judge repeats_per_solution | 1 | 3 |
| Solver format instruction | "5 short lines, no markdown" | "plain text, under 10 lines" |
| Solver max_new_tokens | 1600 | 2048 |
| Verifier confidence prompt | none | calibration guide added |
| Oracle key priority | shell env first | .env first |

---

## Question Quality

**10/10 unique questions, 0% duplicate rate** (was 30% duplicate rate in Feb 17 run).

| Q | Oracle Answer | Parsed Rate | Oracle-Correct Solutions | Notes |
|---|---|---|---|---|
| Q0 | 1 | 1/10 | 0/10 | Very low parse — solvers bail on infinite-looking set |
| Q1 | None | 3/10 | — | Digit-sum constraint, oracle couldn't answer |
| Q2 | None | 10/10 | — | Oracle couldn't answer |
| Q3 | None | 7/10 | — | Oracle couldn't answer |
| Q4 | 0 | 9/10 | 0/10 | Oracle says empty set; solvers disagree |
| Q5 | None | 9/10 | — | Oracle couldn't answer |
| Q6 | **33** | **10/10** | **10/10** | ✅ Best question — well-posed, all solvers agree |
| Q7 | 1 | 6/10 | 6/10 | Good — 6/10 solvers correct |
| Q8 | 10428 | 10/10 | 6/10 | Good variation in solver quality |
| Q9 | 1 | 2/10 | 1/10 | Low parse — similar to Q0/Q7 infinite-set pattern |

**4/10 questions have `oracle_answer=None`** — the oracle with CoT still couldn't find a definite integer answer. These are ill-posed or have infinite solution sets. All solutions from these questions are excluded from accuracy metrics.

**Key issue:** Generator still produces overlapping question types (Q0, Q7, Q9 are all variants of "n divides 2^n + c"). Diversity within a batch is limited.

---

## Solution Parse Rate

**67/100 solutions have a parseable `FINAL_ANSWER: <int>`** (up from 32% in Feb 17, 45-49% in prior Feb 19 runs).

The solver format instruction change ("plain text, under 10 lines") is working. Remaining 33% are solutions that run out of reasoning mid-way or output LaTeX-heavy derivations.

---

## Verifier Accuracy (oracle-answered questions only)

**40 solutions excluded** (from the 4 questions where oracle returned None).
**60 solutions evaluated** (questions Q0, Q4, Q6, Q7, Q8, Q9).

| | Verifier: CORRECT | Verifier: INCORRECT |
|---|---|---|
| **Oracle: correct** | TP = 18 | FN = 5 |
| **Oracle: wrong** | FP = 8 | TN = 29 |

**Accuracy: 78.3%** | **Precision: 69.2%** | **Recall: 78.3%**

The 8 false positives are cases where the verifier said CORRECT but the solution disagrees with the oracle. The 5 false negatives are cases where the verifier said INCORRECT for a solution that matched the oracle — likely formatting issues confusing the verifier.

---

## Confidence Calibration

`model_confidence_mean` (the judge's own stated confidence, averaged over 3 repeats):

| Confidence bucket | n | Verifier accuracy |
|---|---|---|
| 0.5 | 18 | **94.4%** |
| 0.7 | 1 | 100.0% |
| 0.9 | 16 | 87.5% |
| 1.0 | 25 | **60.0%** |

**Pearson r = -0.314** (negative — confidence is inversely correlated with accuracy)

The judge is most accurate when it expresses uncertainty (conf 0.5–0.7) and least accurate at conf=1.0. This means `mc=1.0` is not a reliable "obviously correct" signal — it's actually where errors concentrate.

**`agg confidence` (consensus across 3 repeats) is always 1.0** — because judge temperature is set to 0.0, making all 3 repeats deterministic and identical. This field carries no information until judge temperature is raised (0.3–0.5 recommended).

---

## Issues & Recommendations

### Immediate
1. **Oracle as question filter** — Questions where oracle returns `None` should be discarded and regenerated before reaching the solver. 4/10 questions in this run were useless. Add a post-generation oracle pass that gates question validity.

2. **Judge temperature** — Currently `0.0`, making `repeats_per_solution: 3` pointless (all repeats identical → `confidence` always 1.0). Raise to `0.3` to get meaningful consensus scoring.

3. **Generator diversity within batch** — Q0, Q7, Q9 are all "n divides 2^n+c" variants. The generator needs more diversity signal, or a prompt that explicitly lists question types to avoid.

### Follow-on
4. **Confidence inversion** — The negative Pearson r is a persistent signal. The verifier is overconfident when wrong. Possible fixes: multi-step chain-of-thought before verdict, or replacing confidence with an independent second-pass verification.

5. **Solver parse rate** — 33% still failing. Consider adding a post-processing pass that appends `FINAL_ANSWER: X` by re-prompting the model with its own output and asking "what is the integer answer?"

---

## Progress vs Feb 17 Baseline

| Metric | Feb 17 | Feb 19 (this run) |
|---|---|---|
| Duplicate rate | 30% | **0%** ✅ |
| Parse rate | 32% | **67%** ✅ |
| Oracle working | No (64 token limit) | **Yes** ✅ |
| Accuracy (valid oracle only) | N/A (oracle broken) | **78.3%** |
| Precision | N/A | **69.2%** |
| Pearson r | -0.47 | -0.31 (improving) |
| agg confidence range | {1.0} | {1.0} (judge temp=0 issue) |
| model_confidence range | {0.1,0.5,0.9,1.0} | {0.5,0.7,0.9,1.0} |
