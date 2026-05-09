# Consolidated experiment results

Single reference synthesized from all standalone reports in this repository (as of consolidation). Source files:

| Source | Role |
| --- | --- |
| `report.md` | Consolidated narrative (families A–C + cross conclusions + Exp 4 sketch) |
| `report2.md` | Pairwise/Elo: 20q clean restart, detailed numbers |
| `report3.md` | 10q answer-verify + R_sep (Gemini gen, gpt-5.2 judge) |
| `report4.md` | 50q answer-verify scale-up, R_sep, confidence vs unanimity |
| `outputs/experiment_report_feb19.md` | Single-verify 10q (Gemini stack, GPT-5.2 oracle) |
| `outputs/experiment_report_feb19_v2.md` | v2: same + distinct_CORRECT verifiability finding |
| `outputs/experiment_report_feb19_v3.md` | 20q R_sep, strong/weak temps, distinct_correct at scale |
| `outputs/experiment_report_feb19_v4.md` | 100q rigorous run: answer-only verifier, R_sep vs temp |
| `outputs/reports/samples_gpt41_pairwise_10q_report.md` | GPT-4.1 pairwise 10q Elo metrics |
| `outputs/reports/samples_gpt41_pairwise_10q_oracle_report.md` | Same + oracle alignment metrics |
| `outputs/reports/samples_gpt41_pairwise_rsep_smoke_report.md` | GPT-4.1 pairwise R_sep smoke (n=1) |

---

## 1. Research question (implicit across reports)

Build a **self-play loop** for math where a generator produces questions, solvers produce candidates, and a judge yields a **trainable signal**. Tension: judges and rankings must be **reliable enough to optimize against**, while questions must be **evaluable** (oracle-free where possible). Metrics emphasized: **Elo / pairwise preferences**, **single-verify verdicts**, **R_sep** (strong vs weak sampling separation), **R_cons / stability**, **parse rate**, and **oracle alignment** where ground truth exists.

---

## 2. Experiment families (high level)

### 2.1 Pairwise judging + Elo (Qwen3-4B, oracle gpt-4.1)

**Evolution:** 0.6B → 4B; large `max_new_tokens`; forced A/B (no tie); random then **balanced** A/B order; parser guard for pathological integers.

**Runs (from `report.md`):**

- **A — 20q:** Elo top-1 **5%**, majority **10%**, any correct candidate **20%**. Majority &gt; Elo; few correct candidates.
- **B — 96q (incomplete 100):** Elo **~7.3%**, majority **~11.5%**, any correct **~14.6%**; crash from int parse (later fixed).
- **C — 5q:** 0% on all headline metrics (no correct candidates).

**Bias:** Before balanced order, mixed correct/incorrect votes favored incorrect **~58%**; correct side was A only **~43.75%** while raw A rate **~60.71%** — positional artifact; **balanced A/B** mitigates.

**Refined 20q (`report2.md` / `samples_clean_restart.jsonl`):** 10 solutions, 10 pairs × 4 repeats, balanced A/B.

- Elo top-1 **25%**, majority **25%**, rows with ≥1 correct **30%**.
- **Given ≥1 correct:** Elo and majority both **~83%** → ranking works when candidates include truth; **bottleneck is generation**.
- Mixed-pair vote-level correct rate **70.6%** (above chance); many **2–2 ties** hurt pair-level majority (**47%** overall, **~89%** on non-ties).
- **Agreement signal:** 2–2 splits ~**50%** vote-correct; 4–0 unanimous ~**89%** vote-correct.

**Takeaway:** Pairwise pipeline is **mechanically sound** after bias fixes; **low solver correctness** and **judge noise / ties** limit signal. Elo vs majority often **tie** on small-n setups.

---

### 2.2 Single-verify + oracle (Gemini family, GPT-5.2 oracle) — Feb 19 series

**v1 (`experiment_report_feb19.md`):** 10q × 10 solutions × 3 judge repeats; Gemini 2.5 Flash generator/solver/judge; GPT-5.2 oracle.

- **Duplicates:** 0% (was 30% after dedup + higher gen temperature).
- **Parse:** **67%** (up from 32% Feb 17) via format instructions + tokens.
- **4/10** questions `oracle_answer=None` (ill-posed / infinite families) — excluded from accuracy.
- Verifier vs oracle (valid oracle only, 60 solutions): **78.3%** accuracy, **69.2%** precision, **78.3%** recall.
- **Confidence inverted:** Pearson **r ≈ -0.31**; high self-reported confidence often wrong; **judge temp 0** made 3 repeats identical → `agg confidence` useless.

**v2 (`experiment_report_feb19_v2.md`):** Same numbers; adds **distinct_CORRECT** (count of distinct answers the verifier marks CORRECT) as **oracle-free verifiability**:

- **dc &gt; 1** → verifier effectively grading rhetoric; **high FP** on those questions.
- **dc = 1** → plausible reliability band (still validate).
- **dc = 0** → strict / FN-prone.

**v3 (`experiment_report_feb19_v3.md`):** 20q, **two sampling groups** (strong temp 0.3 vs weak 2.0), skip non-parsed before verify, judge temp **0.3**.

- **97%** parse (194/200); **0** oracle failures.
- Verifier: **82.5%** acc, **85.4%** prec, **90.8%** recall.
- **dc buckets:** dc≥2 → **~44%** avg FP rate vs dc=1 → **~7%**; Pearson(dc, FP rate) **+0.327**.
- **Mean R_sep −0.087** (fraction positive **10%**) — temperature gap **insufficient** on easy arithmetic; verifier **FP on verbose wrong** weak group can **invert** R_sep.
- **Pearson(R_sep, per-question oracle accuracy) +0.717** — R_sep still tracks **question / verifier reliability** despite noisy mean.

**v4 (`experiment_report_feb19_v4.md`):** **100q**, answer-only verifier (no solution text), harder generator prompt, weak temp **2.0**.

- Oracle solve **~71.3%** overall; strong **72.0%** vs weak **70.6%** (**1.4pp** gap) → temperature does not create capability gap.
- **Bimodal difficulty:** ~**59%** questions all solvers correct; ~**25%** none correct; thin **middle band** for separation.
- Verifier: **75.8%** acc, **87.7%** prec, **77.1%** recall — precision up from answer-only judging; recall down (judge must self-solve).
- Mean **R_sep ≈ +0.003** (null at scale); **~8%** questions with R_sep ≥ 0.20 show “real” separation.
- **dc signal:** Pearson **+0.21** at 100q; dc=0 questions align with “too hard” floor.
- Confidence: first **positive** Pearson with accuracy (**+0.175**) under answer-only + CoT; mid-buckets still messy.

**Takeaway:** **Answer-only verification** improves **precision** and calibration vs solution-aware judging. **Temperature is a poor weak group.** Recommend **weaker model** (e.g. gemini-2.0-flash-lite) for R_sep. **distinct_CORRECT** is a strong **oracle-free** training-data filter.

---

### 2.3 Single-verify + strong/weak models (`report3.md`, `report4.md`)

**10q (`report3.md`):** Generator gemini-2.5-flash; strong group 2.5-flash low temp; weak **2.0-flash** high temp; judge **gpt-5.2** `single_verify`, 3 repeats.

- Parse **77%**; oracle solve **29%** overall; strong **40%** vs weak **18%**.
- Verifier vs oracle: **93.0%** acc, **92.3%** prec, **82.8%** recall.
- Mean **R_sep +0.220** (4/10 positive); r(R_sep, oracle acc) **+0.254**; one-sided p ~**0.029** (borderline n=10).

**50q (`report4.md`):** Same family, **5×10q** appended; `repeats_per_solution=1` for throughput.

- **18%** duplicate questions; **80%** parse; oracle overall **35.4%** solve; ≥1 correct **~75.5%** of questions (49 usable).
- Verifier: **76.9%** acc, **71.9%** prec, **59.3%** recall — **dropped vs 10q** (single vote + scale effects).
- Mean **R_sep +0.232**, **54%** questions positive — strong/weak **model** gap works better than temperature-only story in v4.
- Single vote: aggregate confidence not useful; **multi-vote** (from 10q file) shows **unanimous &gt; split** accuracy gap (~**+9.6pp** in better setup).

**Takeaway:** **Model-based weak group** restores **positive mean R_sep** at 50q. **Verifier quality** and **repeats** trade off against **cost**; **unanimity** more reliable than raw confidence.

---

### 2.4 GPT-4.1 pairwise + Elo (March 2026 summaries)

**10q standard (`samples_gpt41_pairwise_10q_report.md`):**

- Preference stability mean **~0.78**; R_sep (Elo) mean **−12.93** (high variance); cross-group win-rate mean **0.467** (slightly below 0.5).

**10q oracle variant (`samples_gpt41_pairwise_10q_oracle_report.md`):**

- Similar stability **~0.77**; R_sep (Elo) mean **+10.64**; win-rate mean **0.533**.
- Oracle: any solution correct mean **0.9**; best-by-Elo correct **0.8**.
- Oracle preference accuracy: macro over questions **~0.46** (n=7 questions with informative prefs); micro **~0.42** over votes — **judge preferences misalign with oracle** on many pairs.

**R_sep smoke (`samples_gpt41_pairwise_rsep_smoke_report.md`):** **n=1** question — diagnostic only (R_sep Elo **−22.86**, win-rate **0.444**, stability **0.8**).

**Takeaway:** Small **pairwise GPT-4.1** runs show **stable preferences** but **noisy / sign-flipping R_sep (Elo)** between configs; **oracle–preference agreement ~40–46%** — pairwise judge is **not** a clean oracle proxy at this scale.

---

## 3. Cross-cutting results (one page)

| Theme | Finding |
| --- | --- |
| **Positional / presentation bias** | Major issue for pairwise; **balanced A/B** and remaps are necessary. |
| **Candidate quality** | Often dominates: if no correct solution exists, no ranking metric helps. |
| **Pairwise judge** | Above-chance on mixed pairs after fixes, but **ties** and **~40%** micro oracle-pref accuracy in GPT-4.1 10q suggest **limited gold alignment**. |
| **Single-verify** | Stronger **precision** path when verifier checks **answers** (ideally without solution text); recall and calibration remain sensitive to prompt and model. |
| **R_sep** | Concept validated (correlates with oracle / quality in several analyses); **temperature-only weak group** washes out at n=100; **weaker model weak group** gives **positive mean R_sep** at 50q. |
| **Oracle-free filters** | **distinct_CORRECT &gt; 1** flags **unverifiable** questions; **dc=0** flags hard/dead ends; combine with **R_sep &gt; 0** suggested as training gate. |
| **Confidence vs votes** | Raw judge confidence often **misleading** or inverted; **vote agreement / unanimity** more robust when repeats enabled. |
| **Parse / format** | **`FINAL_ANSWER:`** parsing remains a **ceiling**; skipping non-parsed as auto-incorrect improves cleanliness. |
| **Question difficulty** | Many batches **too easy** or **too hard**; need **targeted difficulty** (e.g. 30–70% solve) for informative comparisons. |

---

## 4. Ideas already surfaced (for proposal “contributions / aims”)

These are **not new experiments**; they are **directions** repeated across reports:

1. **Train / filter the generator** with **R_cons**, **R_sep**, and **distinct_CORRECT** gates; reward **evaluable** questions, not merely hard ones.
2. **Replace temperature weak group** with **strictly weaker models** (capability gap) for stable R_sep.
3. **Oracle pass** after generation to discard **oracle=None** or inconsistent answer sets before costly rollouts.
4. **Selective multi-vote judging** on uncertain cases; use **unanimity** as reliability, not raw confidence.
5. **Pairwise vs single-verify ablation** on identical question sets with shared oracle labels.
6. **Difficulty calibration** for the generator (hit a solvable-but-spread band).
7. **Post-hoc answer extraction** for parse failures (repair pass).
8. **Downstream:** GRPO / RL with verifier-based scalar rewards + filtered self-play data (`README.md` / `CLAUDE.md` architecture).

---

## 5. Gaps to flag in a proposal

- **Scale:** Many claims rest on **n=10–50**; need pre-registered **n≥100–300** per condition.
- **Confounders:** Duplicate questions, oracle failures, judge temperature 0, single vs multi vote.
- **External validity:** Mix of **Qwen**, **Gemini**, **GPT-4.1**, **gpt-5.2** — proposal should define a **reference stack** and **ablation matrix**.
- **Gold alignment:** Pairwise oracle-pref **~42%** micro shows judge ≠ truth even when models are strong — motivate **verification-first** or **hybrid** training.

---

*This file is a results-only merge. For narrative framing (problem, contributions, method), build the research proposal on top of Sections 1–5 and replace path references with publication-ready citations to configs and JSONL where appropriate.*
