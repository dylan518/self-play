# Signal Bias Diagnostics

## Inputs
- `outputs/pairwise_rollouts_debug/samples_functional_smoke_50_answer_verify.jsonl`
- `outputs/pairwise_rollouts_debug/samples_functional_smoke_10_answer_verify.jsonl`

## 1) verdict_bias
- Overall **P(verdict=CORRECT)**: **27.9%** (223/800 votes)
- Oracle correct rate (vote-weighted): **33.0%** (264/800 vote-weighted labels)
- Oracle correct rate (solution-level): **34.3%** (206/600 solutions)
- Gap vs oracle (vote-weighted): **-5.1%** (positive means judge over-grants CORRECT)
- Gap vs oracle (solution-level): **-6.5%**

### Per-file verdict bias
| file | questions | P(verdict=CORRECT) | oracle(vote-weighted) | gap |
|---|---:|---:|---:|---:|
| `samples_functional_smoke_50_answer_verify.jsonl` | 50 | 29.2% | 35.4% | -6.2% |
| `samples_functional_smoke_10_answer_verify.jsonl` | 10 | 25.7% | 29.0% | -3.3% |

### Per-question verdict bias
| question | P(verdict=CORRECT) | oracle(vote-weighted) | gap |
|---|---:|---:|---:|
| `samples_functional_smoke_50_answer_verify.jsonl#q0` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q1` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q2` | 20.0% | 20.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q3` | 60.0% | 50.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q4` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q5` | 20.0% | 0.0% | 20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q6` | 10.0% | 30.0% | -20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q7` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q8` | 50.0% | 0.0% | 50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q9` | 10.0% | 70.0% | -60.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q10` | 0.0% | 70.0% | -70.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q11` | 50.0% | 60.0% | -10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q12` | 90.0% | 70.0% | 20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q13` | 60.0% | 10.0% | 50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q14` | 80.0% | 70.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q15` | 20.0% | 10.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q16` | 10.0% | 10.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q17` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q18` | 30.0% | 60.0% | -30.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q19` | 10.0% | 0.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q20` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q21` | 20.0% | 70.0% | -50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q22` | 20.0% | 20.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q23` | 40.0% | 10.0% | 30.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q24` | 60.0% | 0.0% | 60.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q25` | 60.0% | 40.0% | 20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q26` | 30.0% | 80.0% | -50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q27` | 50.0% | 60.0% | -10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q28` | 20.0% | 70.0% | -50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q29` | 50.0% | 50.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q30` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q31` | 10.0% | 50.0% | -40.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q32` | 50.0% | 50.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q33` | 20.0% | 20.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q34` | 10.0% | 0.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q35` | 50.0% | 50.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q36` | 70.0% | 60.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q37` | 50.0% | 60.0% | -10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q38` | 40.0% | 30.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q39` | 30.0% | 60.0% | -30.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q40` | 40.0% | 50.0% | -10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q41` | 0.0% | 50.0% | -50.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q42` | 30.0% | 50.0% | -20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q43` | 40.0% | 60.0% | -20.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q44` | 20.0% | 10.0% | 10.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q45` | 0.0% | 60.0% | -60.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q46` | 50.0% | 50.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q47` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q48` | 20.0% | 60.0% | -40.0% |
| `samples_functional_smoke_50_answer_verify.jsonl#q49` | 60.0% | 70.0% | -10.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q0` | 63.3% | 70.0% | -6.7% |
| `samples_functional_smoke_10_answer_verify.jsonl#q1` | 30.0% | 0.0% | 30.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q2` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q3` | 10.0% | 10.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q4` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q5` | 0.0% | 40.0% | -40.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q6` | 30.0% | 30.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q7` | 0.0% | 0.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q8` | 40.0% | 40.0% | 0.0% |
| `samples_functional_smoke_10_answer_verify.jsonl#q9` | 83.3% | 100.0% | -16.7% |

## 2) parse_bias
- Parse-rate spread across groups (max-min): **15.7%**
- Inspect whether weak-temperature groups have lower parse rates and lower oracle solve rates.

| group | parse rate | oracle solve rate | verifier-correct rate |
|---:|---:|---:|---:|
| 0 | 87.0% | 54.3% | 40.3% |
| 1 | 71.3% | 14.3% | 17.0% |

## 3) unanimity_bias
Regimes compare `verifier_pred_correct` (derived from counts when absent) vs `oracle_correct`.

| regime | n | accuracy | precision | recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unanimous_CORRECT | 165 | 74.5% | 74.5% | 100.0% | 123 | 42 | 0 | 0 |
| unanimous_INCORRECT | 422 | 82.0% | n/a | 0.0% | 0 | 0 | 346 | 76 |
| split_votes | 13 | 84.6% | 85.7% | 85.7% | 6 | 1 | 5 | 1 |

## 4) group_effects
R_sep per group is computed as each group's per-question verifier mean minus the mean of other groups on the same question, then averaged.

| rank (by R_sep) | group | mean oracle_correct | mean verifier-correct | parse rate | mean R_sep |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 54.3% | 40.3% | 87.0% | 23.3% |
| 2 | 1 | 14.3% | 17.0% | 71.3% | -23.3% |

## 5) position_bias (optional)
- No pairwise A/B preference rows were available in the provided files.
