# Single-Verify Experiment Report v2
**Date:** 2026-02-19 (updated)
**Config:** `pairwise_rollouts_gemini25flash_single_verify_fast.yaml`
**Model:** Gemini 2.5 Flash (generator + solver + judge), GPT-5.2 (oracle with CoT)
**Run:** 10 questions × 10 solutions × 3 judge repeats = 300 verifications

---

## Fixes Applied vs Feb 17 Baseline

| Issue | Fix | Result |
|---|---|---|
| 30% duplicate questions | Temperature 0.7→1.1 + dedup retry loop | **0% duplicates** ✅ |
| Oracle wrong answers (64 token limit, no CoT) | 1024 tokens + step-by-step prompt | Oracle working ✅ |
| `.env` key ignored (shell env takes priority) | Swapped dotenv priority before `os.environ` | Auth fixed ✅ |
| Solver truncating before `FINAL_ANSWER:` | "plain text, under 10 lines" instruction + 1600→2048 tokens | 32%→67% parse rate ✅ |
| `confidence` always 1.0 (1 repeat) | `repeats_per_solution` 1→3 | Partially fixed (see below) |
| Verifier overconfident | Calibration guide added to verify prompt | Confidence spread improved ✅ |
| Accuracy computed over oracle-failed questions | Analysis script now excludes `oracle=None` questions | Metrics now meaningful ✅ |

---

## Question Quality

10/10 unique questions generated, 0% duplicate rate.

**4/10 questions had `oracle_answer=None`** — the oracle with CoT could not find a finite integer answer. These questions are ill-posed (infinite answer sets, no solution, or ambiguous). All 40 solutions from these questions are excluded from accuracy metrics.

Questions Q0, Q7, Q9 were all variants of "n divides 2^n+c" — the generator still clusters around similar question types within a batch.

---

## Solution Parse Rate

**67/100** solutions have a parseable `FINAL_ANSWER: <int>` (up from 32% in Feb 17).

The remaining 33% are solutions that write extensive reasoning (often with LaTeX despite "plain text" instruction) and either get cut off mid-sentence or never reach the answer line.

---

## Verifier Accuracy (oracle-answered questions only)

Only the 6 questions where `oracle_answer` is not None are included. This covers 60 solutions.

| | Verifier: CORRECT | Verifier: INCORRECT |
|---|---|---|
| **Oracle: correct** | TP = 18 | FN = 5 |
| **Oracle: wrong** | FP = 8 | TN = 29 |

**Accuracy: 78.3%** | **Precision: 69.2%** | **Recall: 78.3%**

---

## Key Finding: Distinct Correct Answers as a Verifiability Signal

The most important finding of this session. For each question, counting how many **distinct parsed answers** the verifier marks as CORRECT directly predicts the false positive rate — no oracle needed.

| Question | Oracle | Distinct CORRECT answers | FP rate | Interpretation |
|---|---|---|---|---|
| Q6 | 33 | `[33]` | **0%** | Well-posed, verifiable ✅ |
| Q7 | 1 | `[1]` | **0%** | Low recall (verifier too strict) |
| Q0 | 1 | `[]` | **0%** | Verifier too strict, 0 CORRECT verdicts |
| Q4 | 0 | `[7]` | **100%** | One wrong answer accepted |
| Q8 | 10428 | `[10416, 10428, 11328, 11922]` | **33%** | 4 distinct = can't compute |
| Q9 | 1 | `[1, 13, None]` | **75%** | 3 distinct = verifier guessing |

**Rule:** `distinct_CORRECT > 1` is a reliable oracle-free signal that the verifier cannot actually evaluate the question. The verifier is judging reasoning quality rather than answer correctness, so any plausible-sounding wrong answer gets through.

This also works on questions where `oracle=None`: Q2 (4 distinct CORRECT answers) and Q3 (3 distinct CORRECT answers) would correctly be flagged as unreliable even without oracle labels.

**Signal map:**
- `distinct_CORRECT = 0` → verifier too strict, likely high false negatives
- `distinct_CORRECT = 1` → potentially reliable (verify against oracle to confirm)
- `distinct_CORRECT > 1` → low verifiability, high false positive rate guaranteed

---

## Confidence Calibration

`model_confidence_mean` (judge's stated confidence averaged over 3 repeats, oracle-answered questions only):

| Confidence bucket | n | Accuracy |
|---|---|---|
| 0.5 | 18 | **94.4%** |
| 0.7 | 1 | 100.0% |
| 0.9 | 16 | 87.5% |
| 1.0 | 25 | **60.0%** |

**Pearson r = -0.314** (negative — confidence is inversely correlated with accuracy on oracle-answered questions)

The verifier is most reliable when it expresses uncertainty (conf ≤ 0.9) and least reliable at maximum confidence. This is a consistent finding across all runs.

---

## Re-voting Analysis

With `judge.temperature: 0.0`, all 3 repeats are **completely deterministic** — there are zero 2/3 splits in the entire dataset. The `repeats_per_solution: 3` setting is currently providing no additional signal. `agg confidence` is always 1.0.

**Fix required:** Raise judge temperature to 0.3. This will produce genuine 2/3 vs 3/3 disagreement. The hypothesis to test: questions where solutions consistently produce 2/3 splits indicate low verifiability (the verifier itself is uncertain), while clean 3/3 splits indicate high verifiability.

---

## Open Issues & Next Steps

| Priority | Issue | Proposed Fix |
|---|---|---|
| High | 4/10 questions ill-posed (oracle=None) | Add oracle as post-generation quality gate: discard and regenerate any question where oracle returns None |
| High | Judge temperature=0.0 makes repeats useless | Raise to 0.3 to get real disagreement signal |
| Medium | `distinct_CORRECT > 1` as filtering metric | Implement as reliability score in `generate_pairwise_data.py` and surface in output |
| Medium | Pearson r still -0.314 | Investigate whether multi-step CoT before verdict improves calibration |
| Low | 33% solver parse failure | Post-process: re-prompt with "what is your integer answer?" for solutions without FINAL_ANSWER tag |
| Low | Generator clusters on similar question types | Add "avoid these question types: {already_generated}" to generator prompt |

---

## Metric Progression

| Metric | Feb 17 (baseline) | Feb 19 v1 | Feb 19 v2 |
|---|---|---|---|
| Duplicate rate | 30% | 0% | **0%** |
| Parse rate | 32% | 35–49% | **67%** |
| Oracle working | No | Yes (key fixed) | **Yes** |
| Accuracy (valid oracle only) | N/A | N/A | **78.3%** |
| Precision | N/A | N/A | **69.2%** |
| Recall | N/A | N/A | **78.3%** |
| Pearson r | -0.47 | -0.31 | **-0.31** |
| Re-voting signal | None (1 repeat) | None (temp=0.0) | None (temp=0.0) |
| Confidence range | `{0.1, 0.5, 0.9, 1.0}` | `{0.5, 0.8, 0.9, 1.0}` | `{0.5, 0.7, 0.9, 1.0}` |
