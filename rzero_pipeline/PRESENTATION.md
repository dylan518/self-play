# Self-Play Question Generation: From RL Collapse to Selected-SFT
### Full investigation writeup — presentation source material
*(R-Zero pipeline, Qwen3.5-4B, June–July 2026)*

---

## 1. The Setup

**Goal:** a self-improving loop for math reasoning. A **questioner** generates problems, a **solver** learns from them, both co-evolve.

**The R-Zero pipeline (one iteration):**
```
A: questioner GRPO (6 steps, reward = difficulty-band + diversity + verifiability)
B: trained questioner generates ~1000+ fresh Qs
   → band filter (solver self-consistency 0.3–0.8, n=9 samples)
   → program verification (judge writes K Python programs, consensus → label)
C: solver GRPO on the verified curriculum (20 steps, n=5)
→ eval: OlympiadBench-675 @8192 greedy (report format_rate / acc|fmt / pass)
```

**Key design detail:** the questioner's own claimed answer is *discarded* — the program-consensus label is the ground truth. (Justified: the base questioner's claimed answers are only **14% correct**.)

---

## 2. The RL Results (what happened)

| run | questioner | curriculum | eval (Oly-675) |
|---|---|---|---|
| base Qwen3.5-4B | — | — | 0.573 |
| **nopack** | RL, collapsed to digit monoculture | 943 rows, 99% unanimous labels | **0.6415** |
| vfix | RL + fixed-Vendi gate (failed to hold) | 507 rows, collapsed | 0.6296 |
| divfix | RL + z-scored reward (held *surface* diversity) | 479 rows | 0.6326 |

**Headline: a 3-way statistical tie (~±1.9pp SE).** Solver training works (+6–7pp over base) but *diversity of the curriculum never mattered* — and the reason turned out to be that no run ever actually produced skill diversity.

---

## 3. Finding #1 — The metric was lying: surface vs skill diversity

Embedding-based **Vendi score** (the pipeline's diversity metric) tracked *wording*, not *reasoning skill*:

```
divfix per training step:      1     2     3     4     5     6
Vendi (unique, embedding):   21.4  21.1  21.8  22.1  22.4  21.1   ← "held diversity"
exact-unique questions:      100%  100%  100%  100%  100%  100%   ← zero duplicates
LLM-judged skill diversity:
  digit-manipulation share:   25%   30%   35%   40%   50%   55%   ← DOUBLED
  effective skills (exp-H):   6.2   5.0   6.5   5.2   4.4   3.6   ← nearly HALVED
```

**The questioner kept wording varied while collapsing the actual solving skill** — invisible to Vendi and dedup, caught only by an LLM judge tagging each question's core skill (10-tag taxonomy → eff-skills = exp(entropy)).

**Consequence:** every "diverse" curriculum was really an enumerate-→-integer monoculture in varied clothing. That's why eval never moved: nothing new to transfer.

---

## 4. Finding #2 — Why RL collapses: Goodhart on a noisy conjunction

**The useful training question is the conjunction: difficult (in-band) ∧ verifiable.** Measured per-step joint yield (real saved data):

```
divfix step:      1    2    3    4    5    6
verifiable(≥2/3): 16%  21%  33%  32%  44%  49%     ← reward drives this up
in-band [.3,.8]:  25%  17%  19%  25%  22%  28%     ← ~flat
BOTH (usable):     4%   3%   5%   6%   7%  10%     ← tiny; anti-correlated axes
```
(nopack pushed verifiability to **94%** — by full collapse to trivially-checkable digit questions.)

**Mechanisms identified (all measured):**
1. **Graded verify reward** `v = 2·votes/K − 1` pays 3× more for a degenerate 3/3 than a diverse 2/3 → exact gradient into the monoculture. The 3/3-unanimous set is **2.5× more templated** than the 2/3 set — *the diversity lives in the borderline region and the reward punishes it.*
2. **Selection on noise:** verifiability is a random variable (3 programs @ temp 0.6). Filtering+rewarding on a noisy estimate → survivors are lucky + degenerate; iterating compounds it (optimizer's-curse ratchet).
3. **Weak anchor:** KL coef 0.01, reference re-anchored each iteration → bounds per-step drift, not cumulative collapse.
4. Difficulty band is on **self-consistency, not solve-rate**: a confidently-*wrong* question reads "too easy" and is dropped; ~18% of questions have consistency ≠ correctness (footgun; CVBAND fix exists).

**Verifier reliability (measured by independent re-solving):** labels are 98% correct on the digit curriculum, 92% on the diverse one; the failure mode is *all K programs sharing one misreading* of an ambiguous question — invisible in the saved outputs. (Verifier *programs* were deleted by the pipeline; now persisted via `DUMP_VERIFY_DIR` patch.)

---

## 5. The Pivot: Selected-SFT (no RL on the questioner)

**Principle: use RL only where reward = true objective (solver correctness). Where the reward is a proxy (question quality), control the distribution by *construction* — curation + SFT — instead of by a gameable reward.**

**The loop (~30 min/iteration):**
```
golden set (verified base questions, strong-verifier labels)
→ curate target (stratify on every axis the filters select on)
→ SFT questioner FROM BASE (lr 5e-6, 1–2 epochs, eff-batch 32, completion-masked)
→ probe: generate 1,200 → judge skill / difficulty / verifiability / joint
→ measure, attribute, rebalance target, repeat
```
Anti-collapse by construction: skill-flat strata, difficulty strata, always-retrain-from-base (no compounding drift), eff-skills regression tripwire.

### Three iterations, each fixing a measured failure:

| | base | v1 (naive) | v2 (skill-quota) | v3 (skill×difficulty strata) |
|---|---|---|---|---|
| digit share | 25% | **50%** ⚠ | 20% | 23% |
| eff-skills | 6.2 | 4.9 ⚠ | 6.6 | **7.5** ✅ |
| trivial/sweet/hard | 0/25/75 | — | **48**/28/23 ⚠ | 37/33/30 ✅ |
| well-posed | 86% | 88% | 88% | 76%* |
| verifiable (strong) | 92% | 92% | 92% | 94% |
| joint (usable) | ~18% | — | 20% | **23%** |
| prefix-unique | 80% | 85% | 99% | 99% |

*\*mix effect: P(well-posed | hard) ≈ 70% is flat across versions — a 4B generation-capability ceiling (composes constraints it can't self-check), not a data problem.*

**v1 measured the two bias mechanisms separately:** verify-selection bias (base 25% → verified-subset 39% digit) and SFT mode-amplification (target 39% → output 50%). **v2's fix exposed the second bias axis:** verifiability selects for *easiness* (48% trivial output). **v3 fixed both** — the general rule: *the SFT target must be stratified on every axis the downstream filter selects on.*

---

## 6. Pipeline-Real Measurement (Stage B with the SFT questioner)

First end-to-end run of the SFT questioner through the actual pipeline (v2 checkpoint, 3,000 generations):

```
in-band (real n=9 band):      301 / 3,000 = 10%     (easy-skew tax — the v3/v4 target)
K=10, MIN_AGREE=6 consensus:  249 / 301  = 83%      ← VERIFIER BOTTLENECK FIXED
                                                       (was 16–48% at K=3)
TRAINABLE:                    249 / 3,000 = 8.3% at full diversity
label agreement (majority vs program): 79%
verifier programs persisted (first time): 301 rows
```

**vs the RL questioner (divfix): 21% trainable but 77%-digit monoculture.** The whole remaining gap is the difficulty axis — the RL band-reward is genuinely good at difficulty-targeting; the SFT loop closes it with target composition instead of a gameable reward.

**Cost comparison (learnable-area questions per GPU·hour, at held diversity):**
```
RL diverse phase:   ~10–40 /GPU·h  (and Stage-A rollouts are discarded)
selected-SFT:    ~1,500–2,500 /GPU·h  (generation-bound, embarrassingly parallel)
questioner training: 2h × 4 GPU (RL + reward service)  vs  10 min × 1 GPU (SFT)
```

---

## 7. Design Contributions (transferable)

1. **Measure skill diversity, not embedding diversity.** exp-entropy over LLM-judged skill tags; embedding-Vendi provably hid a full collapse.
2. **Gate, don't grade, noisy proxies.** Any graded reward on a noisy quality estimate (verifiability, difficulty) is a collapse gradient. If rewarding at all: binary gates at thresholds, or the Lagrangian form (marginal constraints `E[v]=v*, E[b]=b*` + product term `μ·v·b` for co-occurrence — covariance identity separates levels from coupling). Kept on the shelf; SFT curation achieves the same by construction.
3. **The conjunction (difficult ∧ verifiable) is the scarce resource** — the axes are anti-correlated (hard ⇒ more ill-posed; verifiable ⇒ easier). Optimize both at once; sequential passes undo each other.
4. **Stratify the SFT target on every filter axis** (skill, difficulty, pass-probability/IPW) — every un-stratified axis leaks its selection bias into the clone, then amplifies.
5. **Always retrain from base** — kills compounding drift without any KL machinery.
6. **De-noise the verifier with K** (K=10, majority): fixed the label bottleneck outright (83% consensus) and enables inverse-propensity weighting via votes/K.
7. **Persist everything** (verifier programs, per-step band/votes): the entire investigation was rate-limited by discarded intermediate data.
8. **Epochs are a fidelity knob, not a quality knob** — they amplify whatever the target is.

---

## 8. Current State & Next Steps

**Scoreboard (questioner quality, the current focus — solver GRPO deliberately deferred):**
- v3: eff-skills 7.5, balanced difficulty, joint 23% (strong standard), 8.3% pipeline-real (v2)
- Targets: **trainable ≥40%, label-acc ≥90% (✓ by K=10 construction), well-formed ≥96% on-curriculum**

**v4 (in progress):**
- Mine v2+v3 probes → ~200 verified sweet-spot (joint) examples → 3× bigger hard stratum
- **Scratchpad self-check** generation format (draft → test constraints → emit), few-shot/SFT'd on ~30 labeled ill-posed failures — the only lever aimed at the P(well-posed|hard)≈70% ceiling
- Path to 40%: in-band 33→50% (target fidelity) × P(usable|band) 70→85% (self-check) ≈ 42%

**Then:** Stage-B on v3/v4 (pipeline-real), eff-skills on the actual curriculum, and only after question quality gates pass — solver training (rejection-SFT first, GRPO A/B) and the eval-vs-0.6415 test.

**Compute:** Empire (cornell) primary; Unity 4×A100 holds queued (full pipeline env pre-existing); verifier/logging patches live on Empire.
