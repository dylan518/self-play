# Project Notes

Running log of experiments, findings, and decisions. Newest entries first.

---

## 2026-07-03/05 — THE LOOP COMPOUNDS: round-2 funnel (v4) beats round-1 on every axis; Unity 14-day hold LANDED; loop = pure pipeline measurement (all proxies retired).

**The loop (DPW's spec, now the standard):** generate ~2-4k from current questioner → REAL band n=9 → REAL verify K=10 → SFT from base on survivors {in-band ∧ votes≥7/10} → repeat. No LLM-proxy judgments anywhere in the target (v1-v3's difficulty tags were proxies — retired after measuring proxy 28% vs real 10% in-band for v2).

**Round funnels (pipeline-real):**
| | round 1 (v2 questioner) | round 2 (v4) |
|---|---|---|
| generated | 3,000 | 2,000 |
| in-band [0.3,0.8] n=9 | 10.0% | **12.7%** (+27% rel, ~4σ) |
| K=10 consensus (≥6) | 83% | **87%** (saturating, per DPW's prediction) |
| TRAINABLE | 8.3% | **11.0%** (+32% rel) |
| majority-agrees | 79% | 76% |

**Key results:** (1) v4 = SFT on 237 round-1 survivors (only filter: band ∧ ≥7/10; NO quotas) — output 94% well-posed / 100% verifiable / eff-skills 6.2 / BUT difficulty didn't move on the LLM proxy — the REAL measure shows it DID (+2.7pp in-band). (2) Verifiability saturation confirmed → round-2 rejects are ~pure difficulty rejects → each round's survivors are a purer difficulty selection. (3) 10/10-consensus bucket is NOT degenerate under the band∧verify conjunction (eff-skills 6.1, sr 0.58) — the band pre-cuts the trivial corner; divfix's "3/3=templated" was an unbanded artifact. (4) v4 target composition (24% digit / 6.4 eff-skills) landed ≈base naturally — diversity currently UNENFORCED (from-base retrain + eff-skills tripwire only). (5) Round-2 ran 2h13m end-to-end on the hold, zero queue.

**Compute:** **Unity 14-day hold LANDED (61384319, gpu022, until Jul 17)** — but steps can only bind 2 of its 4 A100s (IDX 5-6 unbindable, node quirk; not worth risking the hold). Empire queue stopped serving ≥2-GPU jobs (12h+ pending). All rounds now run on the hold via `srun --overlap`. Env gotchas fixed on Unity: `~/R-Zero` symlink, verify.py DUMP patch applied, convert needs main-guard (smoke skipped — pipeline scripts are guarded). Brev is dead (shut down 6/30).

**Signal-efficiency analysis (DPW):** SFT-on-survivors wastes the reject signal (2,763 graded rejects discarded/round); GRPO uses all of it — but old Stage-A was 1-GPU-reward-service/3-idle (~90% reward measurement, ~5% gradient math; ~5.5 vs ~2 GPU-min per measured question). Fix = phase-synchronous layout (all GPUs per phase) → GRPO arm ~2.5× faster; gated binary reward 1{band ∧ ≥7/10} + KL-to-base = the defensible A/B arm vs the SFT loop. Difficulty is the axis where RL's per-example pressure has the edge (divfix RL hit 24% in-band).

**Next:** v5 = SFT on cumulative pool (round-1 237 ∪ round-2 ≥7/10 survivors ≈ ~440) → round 3; watch whether in-band keeps climbing (→15%+) or plateaus. GRPO 4-GPU-efficient patch (N_SVC=4 colocated) staged for the A/B.

---

## 2026-07-02 — STAGE-B PIPELINE-REAL RESULT (v2 questioner): trainable 8.3%, K=10 verifier FIXED (83% consensus); v3 measured (joint 23%, eff-skills 7.5); campaign notes consolidated.

### Stage-B run (Empire job 982595, completed 07:48 — first pipeline-real measurement of an SFT questioner)
```
generated (v2 questioner):     3,000
in-band [0.3,0.8] (n=9 real):    301   = 10%    (LLM proxy had said 28% — real band filters v2's easy-skew hard)
K=10 MIN_AGREE=6 consensus:      249   = 83% of band  ← VERIFIER BOTTLENECK FIXED (was 16–48% at K=3)
TRAINABLE:                       249/3,000 = 8.3%   (diverse)
majority-vs-program-label agreement: 197/249 = 79% (≈ nopack's 82%)
verify_rollouts.jsonl: 301 rows — verifier PROGRAMS persisted for the first time (DUMP_VERIFY_DIR patch works)
```
**Pipeline-real comparison:** divfix RL questioner = 21% trainable but skill-collapsed (77% digit, eff-skills ~3.6); v2 SFT = 8.3% trainable at full diversity. The RL band-reward is genuinely better at difficulty-targeting (24% vs 10% in-band); the SFT approach wins verification (83% vs ~50–90% variable) and diversity. **The gap is entirely the difficulty axis** → v3/v4 direction confirmed. Solver GRPO NOT run (per DPW: high-quality questions first).

### v3 measured (full battery): eff-skills **7.5** (campaign best; base 6.2), digit 23%, difficulty 37/33/30 (trivial/sweet/hard vs v2's 48/28/23), verifiable 94%, well-posed 76% (STRICT judge; verifiable>wellposed means 18% are computable-but-flagged; pipeline K=10 standard reads higher), **joint 23%** (vs v2 20%, RL-diverse 4–10%). Ceiling identified: **P(well-posed|hard) ≈ 70% flat across v2/v3** = generation-capability limit (4B composes constraints it can't self-check), NOT target curation — the 6 sweet-spot failures are all incoherence (impossible congruence, vacuous condition, precedence ambiguity).

### Metric definitions (for the record)
- **eff-skills** = exp(entropy) of LLM-judged skill-tag distribution (10-tag taxonomy, judged on core solving skill) — NOT embedding-Vendi (retired: measures wording, hid the divfix skill collapse). Caveat: currently measured on raw output; the number that matters is eff-skills|trainable (pending: judge the 249-row curriculum).
- **well-posed** = independent agent writes+RUNS a program, then judges "exactly one reasonable reading → one integer"; strict (flags dominant-reading ambiguity + vacuous conditions). n=50/version, ±6pp.
- **trainable/joint** = in-band ∧ verifiable ∧ labeled.
- **learnable-area yield (absolute)**: RL divfix Stage-A rollouts 9→25 per 240/step (84 total over ~2h×4GPU, discarded); divfix Stage-B 479/2,243=21% (collapsed); v2 pipeline-real 249/3,000=8.3% (diverse); v3 strong-standard ~23%.

### Diversity enforcement (SFT loop): by construction, 4 layers — skill-flat strata in target; difficulty strata; ALWAYS retrain from base (v1/v2/v3 each from base → drift cannot compound); eff-skills regression tripwire per round. vs RL: Vendi novelty reward (saturated step 3, protected only wording).

### DPW targets for the questioner: trainable ≥40%, label-acc ≥90% (✓ K=10/≥6 by construction), well-formed ≥96% ON CURRICULUM (post-verification — raw-gen 96% not realistic for 4B at difficulty). Path to 40%: in-band 33→50% (bigger hard-stratum via joint-mined pool ~200 examples + epochs-as-fidelity) × P(usable|band) 70→85% (**scratchpad self-check** — draft→test-constraints→emit format, few-shot/SFT on our ~30 labeled ill-posed failures). Batch size NOT a lever (saturated at n=136); LR only damps amplification.

### v4 plan (approved direction: SFT-only, no GRPO yet): mine all v2+v3 probe sweet-spots (→~200 verified joint examples) + scratchpad self-check format + difficulty strata → SFT → probe. Gates: trainable ≥30% (pipeline-real), well-formed|trainable ≥92%, eff-skills ≥7.
### Open items: v3→vLLM conversion fails at smoke (processor-file glob bug — fix: copy preprocessor configs from sftq_v2/vllm); judge eff-skills on the 249-row curriculum; Unity holds (2d + 14d) still queued; solver GRPO + Oly-675 eval deliberately deferred.

---

## 2026-07-02 — v2 measured fully (joint=20%); SECOND selection bias found (difficulty); v3 launched (stratified joint-boosted target); loop design converged.

**v2 complete measurements (1,200-gen probe + judge battery):** digit 20% / eff-skills 6.6 (>base 6.2) / well-posed 88% / verifiable 92% / prefix-unique 99% / format 99.7%. Self-label correctness 9% (≈base 14% — SFT can't teach solving; labels come from program consensus, fine). **Difficulty (LLM proxy, n=60): trivial 48% / sweet-spot 28% / too-hard 23%** vs base 0/25/75. ⇒ **SECOND selection-bias axis: the verified-only target + SFT amplification skewed output EASY** (too-hard mass → trivial mass). Golden set's natural difficulty was ~uniform (92/89/87), so the 48% trivial is mostly AMPLIFICATION of the easy mode. Lesson: **the SFT target must be balanced on EVERY axis the filter touches** (skill AND difficulty), and epochs amplify whatever the target is (fidelity knob, not quality knob — more epochs on a bad target = worse).

**JOINT (difficult∧verifiable) measured for v2: ~20%** (28% in-band × 71% of sweet-spots verifiable; all 5 sweet-spot failures = ambiguity/contradiction — hard end is the ill-posed end, anti-correlation confirmed). vs divfix RL questioner 4–10%. Joint is the scarce resource; optimize BOTH at once (verifiability already saturated at 92% — sequencing moot; anti-correlated axes oscillate if done sequentially; SFT=distribution-matching makes joint-target free).

**v3 LAUNCHED (job 982601):** golden_set_v3 = 136 ex, **skill-flat × difficulty-stratified** (56% sweet-spot / 26% too-hard / 18% trivial) + **12 v2-mined joint examples** (self-distillation: v2's own verified sweet-spot output — ~200 more minable from the 1,010 probe gens). 2 epochs (balanced target → fidelity now wanted). **Gate: joint ≥30%, eff-skills ≥6, verifiable ~90%.**

**Loop design converged (DPW):** re-measure difficulty EVERY round vs current solver (difficulty is a relation, not a property; tags ephemeral); **gradient buckets** = solve-rate deciles from Stage-B band (n=9 → 5 honest buckets, SE≈0.16) replacing the trivial/sweet/hard trichotomy; diversity via skill-flat strata + prefix dedup now, + kernel-novelty admission (taxonomy-free) rounds 2+; explicit digit-cap RETIRED after v2 (replaced by strata + IPW planned). Round template: **generate → measure (verify + solve-rate gradient) → curate (skill-flat × gradient-spread, novelty, joint-boosted) → SFT → probe.**

**Compute:** Empire queue reopened; 1-GPU jobs start in minutes, 4-GPU sftsel Stage-B-only (982595, SKIP_STRAIN — curriculum measurement w/o solver train per DPW "don't run eval yet") queued. **Unity 14-day 4×A100 hold submitted (61384319)** — full pipeline env already exists there (`/work/pi_general_dartmouth_edu/dylan/envs/rzero`: verl 0.8+vllm 0.23+tilelang+R-Zero); prior hold ran Jun 18–22 and was voluntarily released during Empire migration (lesson: don't release until campaign done). wandb netrc installed on Empire (was missing). verify.py + reward-service patched w/ DUMP_VERIFY_DIR (verifier programs+outputs now persisted; was deleted-after-run).

---

## 2026-07-01 — SFT-selected run LAUNCHED (Empire). Round-0 result: selection-bias + SFT-amplification MEASURED; quota fix applied (v2).

**sftq_v1** (Empire job 982543, 1×H100): Qwen3.5-4B SFT on golden_set_v1 = 268 verified base questions (strong-verifier labels), exact Stage-A prompt, completion-masked, lr 5e-6, 2 epochs. Env gotchas fixed: compute nodes need `~/extralib` in LD_LIBRARY_PATH (libcrypt.so.2) + HF_HOME on lustre; text-only `save_pretrained` config breaks vLLM (Qwen3_5TextConfig vs Qwen3_5Config) → probe uses HF generate; TODO convert ckpt for vLLM before 10k-scale gen.

**Probe (1,200 gens) vs baselines:**
| | base | golden v1 (target) | sftq_v1 output |
|---|---|---|---|
| digit share | 25% | **39%** | **50%** |
| eff-skills | 6.2 | 5.6 | 4.9 |
| format parseable | ~99% | — | 99.5% |
| prefix-unique | 80% | — | **85%** |
| well-posed | ~74–86% | — | 88% |
| verifiable (strong) | 85% | — | 86% |

**TWO mechanisms measured separately:** (1) **verify-selection bias**: base 25% → verified-subset 39% digit (DPW's homogenization argument, live at round 0); (2) **SFT mode-amplification**: target 39% → output 50% (+11pp head-of-distribution amplification, 2 epochs on 268 ex). Format/uniqueness/well-posedness all fine — ONLY the skill mixture regressed, and the pre-registered skill-entropy monitor caught it.

**Fix (v2, job 982567):** golden_set_v2 = 202 ex, digit capped at **17%** (anticipating +10pp amplification → ~27% output ≈ base), all other skills kept; 1 epoch instead of 2. Full audit of golden v1 composition (13-agent workflow): digit 39%, divisor 21%, comb 10%, NT-special 10%, mod 8%, poly 8%, seq 3%, geo 0.4%.

**Empire queue OPEN tonight** (5 pending vs 100s earlier); jobs start in ~minutes. Watchers active. Next: v2 probe → if digit ~25%/eff-skills ~6 → scale to 10k generation → verify (K=10) → curriculum → solver GRPO (nopack recipe, attribution-clean) → Oly-675 vs 0.6415.

**V2 RESULT — QUOTA FIX WORKED, GATE PASSED:** sftq_v2 (202 ex, digit capped 17%, 1 epoch) output: **digit 20%** (v1 50%, base 25%) · **eff-skills 6.6** (> base 6.2!) · well-posed 88% · verifiable+label 88–92% · format 99.7% · prefix-unique 99%. First questioner simultaneously MORE skill-diverse than base AND better-posed AND more verifiable. Calibration landed (17% target + ~+10pp amplification → 20%). Judge caveat: "surface variety exceeds true skill variety" — still ~all "count integers in range with condition X" (verifier-expressiveness ceiling, bounds eval upside). **Proceeding to full round:** convert ckpt for vLLM (job 982582: text→multimodal key remap + smoke) → Stage B w/ sftq_v2 questioner (gen→band→judge, DUMP_VERIFY_DIR on) → Stage C solver GRPO (nopack recipe unchanged) → Oly-675 vs 0.6415. SFT hyperparams: lr 5e-6 cosine, 1 epoch, eff-batch 32 (2×16), ~7 steps, completion-masked, bf16, 1×H100 ~10min.

---

## 2026-07-01 — PROPOSED NEXT-RUN DESIGN: Lagrangian-constrained questioner reward + all-SFT single-shot (the fix for everything found this session).

**Motivation (from the session's findings):** collapse = Goodhart/survivorship on a NOISY verifiability proxy (max verify → degenerate 10/10 digit monoculture); the useful property is the CONJUNCTION verifiable∧in-band (neither alone is a usable training example); embedding-Vendi measures SURFACE not SKILL diversity so it hid the collapse; the iterated filter+train loop is a selection ratchet that only a fixed-reference policy anchor stops.

**Questioner reward (per question) — Lagrangian-constrained, joint + marginal:**
```
reward_i = w_D·D_i  +  μ·(v_i·b_i)  +  λ_v·v_i  +  λ_b·b_i
  v_i = verify quality: triangular PEAK at consensus c*=0.7, =0 at ≤0.4 or =1.0  (∈[0,1]; peaked NOT max → penalizes degenerate 10/10)
  b_i = in-band quality: 1 − 2·|solve_rate_i − 0.5|                              (∈[0,1]; peak at solve-rate 0.5, CVBAND)
  D_i = diversity: kernel-coverage novelty (below), σ=median batch dist, + memory bank, on a SKILL embedding
  μ·(v_i·b_i) = JOINT term: given marginals fixed, maximizes Cov(v,b) = co-occurrence (verifiable AND in-band in same Q)
  λ_v,λ_b = Lagrangians pinning MARGINALS E[v]=v*, E[b]=b* (so one axis can't be driven while the other slides)
```
Covariance identity `E[v·b]=E[v]E[b]+Cov(v,b)`: marginals (λ's) set the LEVELS, product (μ) sets the CO-OCCURRENCE. Both needed.

**Diversity measure (replaces Vendi as the reward):** `D_i = 1 − (1/n)Σ_{j≠i} exp(−‖x_i−x_j‖²/2σ²)`, mean over batch. Bounded [0,1], **saturating** (once a point is ≳2σ from all others D_i≈1 → no reward for further spread; rewards FILLING GAPS not infinite spread — the property Vendi lacks; Vendi is NOT asymptotic, grows with n). Distance-weighted across group+neighbors. **Must use a SKILL embedding** (solution-approach / skill-classifier), NOT question text, or you reward surface-wording diversity (the trap). Note: `diversity_penalties` in diversity.py is ~this (currently unused).

**Example hyperparameters:**
```
VERIFIER:  VERIFY_K_PROGRAMS=10  VERIFY_MIN_AGREE=6  VERIFY_TEMPERATURE=0.6
REWARD:    w_D=1.0  μ=1.0  v*=0.6  b*=0.6  η(dual LR)=0.05  λ init 0 clip≥−2  (PID optional)
           λ_v ← λ_v + η(v* − mean v_i);  λ_b ← λ_b + η(b* − mean b_i)  per GRPO step
QUESTIONER GRPO: Q_STEPS=20  rollout.n=8  train_batch=256  lr=1e-6
           KL: use_kl_loss=true kl_loss_coef=0.05 (↑from 0.01), kl REFERENCE=FIXED base (NOT re-anchored per iter) ← anti-collapse
           + replay 20% base-questioner Qs into each batch
DIFFICULTY: CVBAND=1  CVLO=0.2 CVHI=0.8 (drop-filter on true solve-rate vs verified label, NOT self-consistency)
SOLVER (Stage C) = rejection-sampling SFT (STaR/RFT): K_sol=8 solutions/verified-Q (temp1.0, 4096tok), keep answer==label, SFT 2 epochs lr=1e-5 bs64
PIPELINE: single-shot (frozen/anchored questioner → gen ~10k → verify K=10 → rejection-SFT solver → eval). NO band-eval pre-pass (rejection sampling handles difficulty). Eval Oly-675@8192 decompose fmt/acc|fmt/pass.
FIRST SWEEP KNOBS: μ (0.5–2), KL coef (0.03–0.1).
```
**Why each piece:** K=10/MIN_AGREE=6 de-noises selection; v_i peaked@0.7 (0 at 10/10) fights degenerate-verifiable; μ·v·b + 2 Lagrangians = joint+marginal targeting; fixed-base KL@0.05 + 20% replay = the real anti-collapse (stops the ratchet); CVBAND makes in-band mean solve-rate; rejection-SFT avoids RL instability. **Open problem: the SKILL-diversity signal for D_i (top objective) — surface Vendi won't do; needs solution-approach embedding or skill-classifier.** All-SFT loop (SFT questioner-target + rejection-SFT solver) = no RL Goodhart anywhere.

**Compute:** ~3h on 4×H100 (no band-eval). BLOCKED: cornell saturated (jobs pending 2h+); Unity/WashU need env build (cornell has env ready). Diverse-verified-set build running cluster-free via workflow agents (w8gbghyjr).

---

## 2026-07-01 — CRITICAL CORRECTION (DPW's instinct): SKILL diversity DID collapse in divfix; embedding-Vendi was measuring WORDING, not reasoning.

**Overrides the earlier "divfix generation-side success / diversity held" note.** LLM-judge tagged core solving-SKILL of 20 divfix questions/step (fixed taxonomy: digit-manip, divisor-factor, modular, sequence, combinatorics, polynomial, geometry, set-ops, NT-special):
```
step:              1     2     3     4     5     6
digit-manip %:    25%   30%   35%   40%   50%   55%   ↑ doubles
effective skills: 6.2   5.0   6.5   5.2   4.4   3.6   ↓ nearly halves
holistic (1-10):   7     5     6     6     4     4
judge verdict:  "varied"  →  "digit-manipulation monoculture (11/20)"
```
**SKILL diversity collapses (digit 25→55%, eff-skills 6.2→3.6) WHILE surface metrics stayed flat (exact-unique 100% all steps, Vendi_unique ~21).** ⇒ the questioner kept WORDING varied (diff numbers/phrasings) so embedding-Vendi + exact-unique MISSED the reasoning-skill collapse. z-scoring didn't prevent collapse — it HID it from the embedding metric.

**Retro-explains the whole session:** Vendi 9.93/13.2 measured WORDING variety, not skill. Even divfix's curriculum was skill-collapsed to digit-manipulation → nothing new to transfer to OlympiadBench → **that's why diversity never moved the eval.** Even the BASE is only ~6 eff-skills / 25% digit (moderately skill-diverse); training narrows to ~3.6 / 55%.

**Consequences:** (1) **We measured the wrong diversity all session** — use SKILL/topic diversity (LLM-judged), NOT embedding Vendi; on the skill axis EVERY run collapsed. (2) The questioner fix (anchor / SFT-to-diverse) must define its target + reward on SKILL, not surface embeddings — else you get varied-wording monoculture-skill (exactly this). Artifacts: `~/Desktop/divfix_step_samples.md` (20 Q/step), skill-judge workflow wsjqwd4dd.

---

## 2026-07-01 — DESIGN PIVOT: all-SFT single-shot; the diverse-verifiable set = VERIFY THE BASE ROLLOUTS (never done before).

**Key realization (DPW):** every `judge.jsonl` verified set (2292 total, only **355 unique** — 85% degenerate digit dupes) is from TRAINED (collapsed) questioners' Stage-B output. **The base questioner's diverse questions (Vendi ~20) were NEVER verified** — they only exist as unlabeled training-step rollouts. So all diversity experiments trained ON collapsed curricula; we've never had a diverse VERIFIED curriculum.

**But the diverse data is already on disk:** base questioner step-1 rollouts across all runs = 2640 raw, **853 unique**; +step-2 → **1655 unique** (diverse, Vendi ~20, unverified). Verifying these (one pass, ~25% yield → ~400 diverse-verifiable) gives the diverse-verified set we've been missing — no new generation needed. Job 981629 queued (verify all 1655, K=5, SKIP_BAND) → `base_diverse_verify_out.jsonl`.

**Anti-collapse design converged this session (survivorship/Goodhart is the root cause):**
- **Break the iteration** — single-shot, not iterated loop (loop compounds selection bias to degenerate extreme).
- **SFT the QUESTIONER** on a diversity-curated verified set (NOT raw — raw is 85% dupes → clones monoculture). SFT = distribution-matching, not proxy-max → no Goodhart collapse. Source = verified-BASE rollouts (diverse), not reused collapsed curricula.
- **SFT the SOLVER** via rejection sampling (STaR/RFT): sample K solutions, keep correct-vs-verified-label, SFT. Difficulty self-filters → **band-eval redundant** (drop it — it was the un-schedulable expensive stage).
- ⇒ **all-SFT self-improvement loop: no RL anywhere** → no Goodhart, no reward-hacking, no KL-tuning, no collapse. Recipe: verify-base → SFT-questioner (diverse target) → generate → verify once (K≈10 de-noised) → rejection-SFT solver → eval. ~3h compute on 4×H100 (no band-eval).
- Consistency (analytical, calibrated to 86%-3/3): repeat verifiability of SURVIVORS ~95% (mild) — but that's survivorship; real churn is in filtered-out borderline Qs. ⇒ collapse is driven by iterated reward/selection RATCHET, not per-pass noise → **policy fix (fixed-ref KL / SFT-to-diverse-distribution) > de-noising**.

**COMPUTE BLOCKER:** Empire `cornell` saturated (4-7day jobs); ALL diagnostic jobs (even 15-30min) pending 2h+ on Priority, not backfilling. Need a quieter window or reservation for the ~3h run. Unity (`ssh unity`) is a fallback but needs env setup.

---

## 2026-06-30 — ROOT-CAUSE (DPW): collapse = Goodhart/optimizer's-curse on a NOISY verifiability proxy. Fix = GATE not grade.

**The fundamental issue (DPW's insight):** verifiability & difficulty are NON-DETERMINISTIC (3 programs @temp0.6; band from n stochastic solver samples). The pipeline FILTERS and REWARDS on these noisy estimates → selection-on-noise → iterating marches to the degenerate EXTREME (maximally-verifiable = trivially-programmable = digit monoculture), not the target (good diverse Qs). Maximal measured verifiability selects AGAINST interesting Qs (which admit multiple program approaches that disagree on edges).

**Confirmed in data (divfix verified, 3/3 vs 2/3):** distinct-prefix ratio **3/3=0.14 vs 2/3=0.35** (the max-verifiable set is 2.5× MORE monoculture); digit-mention 77% vs 67%. ⇒ **the diversity lives in the 2/3 (borderline) region; the pipeline prefers the degenerate 3/3.**

**Precise mechanism = the reward is GRADED:** `v = 2·votes/3 − 1` → 2/3 gives +0.33 but 3/3 gives +1.0. The questioner is paid **3× more for a degenerate 3/3 question than a diverse 2/3 one** → exact gradient to the extreme.

**FIX (unifies the session's threads):** (1) **Gate, don't grade** — binary verify reward: verified(≥2/3) → same reward regardless of 2/3 vs 3/3, removing the 2/3→3/3 gradient. (2) **De-noise** the estimate (K=5–7 programs, more band samples) so the filter selects less on luck. (3) **Don't compound** — fresh generation/streaming ("each question once") breaks the iterative march. (4) **Sample representatively** across the verifiable spectrum, not the max tail. **Through-line: use verifiability & difficulty as GATES (pass/fail), NEVER as continuous rewards to maximize** — maximizing a noisy proxy → Goodhart → degenerate extreme. Same principle as the band=self-consistency footgun (CVBAND) but for verifiability.

---

## 2026-06-30 — MAJOR REFRAME: the questioner HELD diversity; the BAND+VERIFY FILTER destroys it. (Vendi-per-step, not reward.)

**Vendi SCORE per questioner step (absolute diversity via all-MiniLM + Nystrom, NOT the marginal-novelty reward):**
```
step:    1     2     3     4     5     6
nopack: 22.0  21.0  16.6  10.4   8.1   6.2   <- steep collapse
divfix: 21.2  19.9  19.9  20.6  19.9  17.0   <- HOLDS ~20 (z-scoring worked)
```
The diversity REWARD (→0.0014 both) was MISLEADING — it's marginal novelty vs a saturating bank, not actual diversity. Real Vendi: **divfix barely collapsed (held ~20 ≈ base 21); only nopack collapsed (22→6).** REWARD_FIXEDPCT z-scoring genuinely held the questioner's diversity.

**CORRECTED (was: "filter destroys half the diversity" — that was an n-mismatch artifact).** Comparing Vendi at MATCHED n (=400, since Vendi grows with n) across filter stages:
```
                       nopack Vendi(m400)   divfix Vendi(m400)
0 generated (Stage B)      2.85                15.87
1 after BAND [0.3-0.8]     2.87                14.76  (-7%)
2 after VERIFY (curric)    2.85                13.20  (-11%)
```
⇒ **NEITHER filter is the diversity sink** — BAND and VERIFY each remove only ~7-11% (verify slightly more). The curriculum (13.2) ≈ what the questioner GENERATED (15.9). **Diversity is decided at GENERATION by the trained questioner**, not by the filters. The old "20→9.93 = half lost" compared per-step n=240 (~20) to a curriculum measured at n=65 (~9.93) — a sample-size artifact, not filtering.

**Net (strengthened):** divfix's curriculum is genuinely **~4.6× more diverse than nopack's** (matched-n 13.2 vs 2.85) — even MORE than the old n=65 "3×" (9.93 vs 3.3) implied — and STILL tied on eval (0.6326 vs 0.6415). So the conclusion is stronger: a 4.6×-diverse curriculum doesn't move the eval (verifier-expressiveness ceiling = still enumerate→integer skill), and the tie remains confounded by nopack's 2× data + cleaner labels. z-scoring DID give a diverse curriculum; the filter was never the problem. Lever is NOT the filter and NOT questioner-reward — it's verifier EXPRESSIVENESS (what skills can be certified) + the confounded matched-size control.

**Per-step in-band + verification-agreement** measured faithfully on Empire (job 981609: band-eval n=9@8192 + Qwen3.5-4B verifier over all 1697 per-step Qs, both runs) — pending GPU. Local tooling: `/tmp/vendi_perstep.py` (Vendi/step), `~/Desktop/perstep_verify_input.json`.

---

## 2026-06-30 — THREE-WAY correctly-labeled measurement (110-agent parallel workflow): RL trades diversity for cleanliness (same gradient). "Clean AND diverse" unreachable.

**Method:** 110 independent verifier-agents (1 per question, parallel workflow `wuh1gaiuu`, 2.4M tok, ~9min), each wrote+ran its own Python brute-force BLIND to the program label; compared to labels after. Artifacts `~/Desktop/vresults/`, `*_blind_sample.json`, `*_sample_key.json`.

| | base questioner PRE-RL | divfix post-RL | nopack post-RL |
|---|---|---|---|
| Vendi | ~18.7 | 9.93 | 3.3 |
| well-posed | **74%** (37/50) | ~89% | ~92% |
| correctly labeled (indep vs program label) | n/a (never verified) | **92%** (55/60) | **98%** (59/60) |
| of 3/3 unanimous correct | — | 92% (48/52) | 98% |
| of 2/3 correct | — | 88% (7/8) | — |
| verifiable (<30s brute-force) | 92% | — | — |
| well-posed AND verifiable (label ceiling) | **74%** | — | — |

**Findings:** (1) **The diverse end is the DIRTY end** — pre-RL questioner is Vendi 18.7 but only 74% well-posed; the varied questions (cubics, bounded-coeff polynomials, factorial-digit) are phrased most ambiguously. (2) **RL cleans the questioner (74%→92% well-posed, →98% correct labels) ONLY by collapsing to the digit monoculture** (Vendi 18.7→3.3). Mode-collapse and quality-improvement are the SAME gradient — the verifiability+band reward selects the verifiable-easy corner (digit-7), which is low-diversity by construction. (3) Even divfix 3/3-unanimous labels are only 92% correct (vs nopack 98%); its errors concentrate in genuinely ambiguous/self-contradictory Qs where 3 programs SHARED a misreading. **Conclusion: "clean AND diverse" is unreachable from this questioner** — diverse(74% clean) XOR clean(92%, monoculture). This is THE reason every curriculum evals ~0.63 regardless of the diversity knob. Reinforces: don't iterate on the diversity reward; the binding constraint is verifier expressiveness (only brute-forceable Qs survive → enumerate→integer skill only).

---

## 2026-06-30 — BOTH curricula are monocultures; "diversity" = surface wording, not skill. Base label-correctness measured 98% (n=60 blind).

**Base/nopack is an even TIGHTER monoculture than divfix** (structural analysis of 943 verified Qs): 100% mention "digit", **90% constrain digit '7'**, **97% identical template** "Let S = set of integers n, 1≤n≤N, [contains digit 7]; apply f(n); sum." Only 3 axes vary: N∈{1000,2023,2024}, the digit condition (almost always 7, sometimes 7+x), f∈{digit-sum 49%, count, product, #digits}. ⇒ essentially ONE problem re-rolled 943×. Vendi 3.3 confirms.

**Crux finding:** the questioner mode-collapses in BOTH runs; the diversity reward only changed WHICH monoculture (and surface wording: Vendi 3.3→9.9) — never the underlying skill (bounded enumeration→one integer), because the **program-verifier only passes brute-forceable problems** (geometry=0 in divfix, no proofs/open-reasoning survive). So Vendi (embedding/wording metric) moved while reasoning-skill stayed fixed → eval pinned ~0.63 regardless (base 0.6415 / divfix 0.6326, a tie). **Diversity-as-measured is real; diversity-as-skill is not.** This is why the diversity lever never moved the eval.

**Base label correctness MEASURED (independent blind brute-force re-solve, subagent wrote+ran Python, n=60 random 3/3, blind to labels):** 59/60 = **98% correct** (95% CI ~91–99.7%); 0/60 flagged ambiguous. The 1 error (idx 49): question asks "number of elements in the SET of pairs (n,d)" with d=f(n) deterministic → correct=|S|=54, but all 3 programs shared a misreading and computed sum-of-d=154. ⇒ base label-error mode is NOT arithmetic — it's 3 programs sharing a misread of an unusually-phrased Q (rare, ~2%). Combined w/ ~92% well-posed (gpt-5.5, 46/50) → base ≈ **~90% well-formed AND correctly labeled**. Artifacts: `~/Desktop/nopack_{blind_sample,sample_key,independent_answers}.json`, `~/Desktop/divfix_examples_by_topic.md`.

---

## 2026-06-30 — WHY diverse(divfix 0.6326) didn't beat digit-monoculture(nopack 0.6415): it's a TIE confounded by curriculum size + label noise, NOT a diversity verdict.

**First: 0.6326 vs 0.6415 is within noise.** ~6 Qs / 675; SE at p≈0.63,n=675 is ~1.9pp ≫ the 0.9pp gap. digit/vfix/divfix (0.6415/0.6296/0.6326) are a 3-way statistical tie. Headline = "diversity didn't HELP," not "it hurt."

**Mechanism (from judge.jsonl consensus-strength analysis, staged locally + WashU/HF):**
| | nopack/digit | divfix/diverse |
|---|---|---|
| judged | 946 | 534 |
| verified (got label) | 943 (100%) | 479 (90%) |
| 3/3 unanimous (gold) | 929 (99%) | 407 (85%) |
| 2/3 consensus (shaky) | 14 (1%) | 72 (15%) |

Three confounds, all favoring the monoculture: (1) **noisier reward** — digit labels 99% gold vs diverse 85% unanimous, **15× more 2/3 labels** (RL is sensitive to label noise; dirty labels push policy wrong); consistent w/ gpt-validation 89% (divfix) vs 92% (nopack) well-posed. (2) **~half the data** — 479 vs 943 verified rows (diverse Qs harder to program-verify → more culled). (3) **verifiability filter partially undoes diversity** — only 90% of diverse Qs labeled (vs 100%), survivors skew to verifiable/digit-like tail. ⇒ monoculture wins on signal-cleanliness what it loses on breadth; tie at ~0.63. **This run does NOT isolate diversity** (confounded w/ size+noise).

**Sharper follow-up experiment (cheaper than another iter1, reuses existing curricula on HF/WashU):** keep only the **407 unanimous(3/3) diverse** Qs, subsample digit to 407, so diversity is the ONLY variable. clean-diverse-407 vs clean-digit-407 → if tie, diversity genuinely doesn't matter here; if diverse pulls ahead, noise was masking a real effect.

**Resume compute:** WashU is NOT a training home (borrowed `jiaxinh` acct, home quota full 29.6/30GB, GPU submit blocked: `spank-auks` cred fail + needs `-A engr-lab-<PI>`). **Empire AI (Cornell) is the home** — own acct, `~/venvs/selfplay`+`~/self-play` present, `cornell` H100 partition has free nodes. Pull curricula/models from HF+WashU.

---

## 2026-06-30 — divfix COMPLETED (eval 0.6326); this run drew Vendi 9.93 (moderate), NOT 22. Pre-shutdown preservation + auto-sync set up.

**divfix result: `RESULT_divfix_it1_8k` n=675 format_rate=1.0000 acc|fmt=0.6326 pass=0.6326** (mean_resp 6984 chars, ~0 truncated). Curriculum: 479 verified rows, **92% well-posed**, **Vendi 9.93** — i.e. THIS run drew a *moderate*-diversity curriculum, not the 21.7 favorable draw. Questioner still collapsed (dom-topic 32→68% over 6 steps) despite REWARD_FIXEDPCT z-scoring + VENDI_GOLDEN=0 (no gate), NO clip.

**Updated eval scoreboard (Oly-675 @8192 greedy):** base 0.573 · nopack/digit (Vendi 3.3) **0.6415** · vfix (Vendi 4.68) 0.6296 · **divfix (Vendi 9.93) 0.6326**. ⇒ Across Vendi 3.3→9.93 the eval sits flat at ~0.63; moderate diversity does NOT move it off the digit baseline. The Vendi-22 end is STILL untested (no completed run ever trained on it — diversity-hold is a non-deterministic lottery: same divfix config drew 21.7 vs 9.93 vs collapsed; well-posedness ~92% is the reliable part, diversity is the draw).

**Preservation (Brev box shuts down ~9am PDT 2026-06-30):**
- divfix solver → HF `Dylan1631/sp-divfix-solver-oly0p6326`; rollouts → HF dataset `selfplay-rollouts-analysis-backup/selfplay_divfix_rollouts.tar.gz`; rollout text → WashU `/home/compute/jiaxinh/selfplay_live/selfplay_divfix/`.
- **Auto-sync loop** (local `~/Desktop/sync_rollouts_to_washu.sh`, pid running, 15-min cycles, Brev→local→WashU; rollout text only, weights excluded → HF-only). Log `~/Desktop/sync_rollouts_to_washu.log`. WashU dest `/home/compute/jiaxinh/selfplay_live/`.
- **8am pre-shutdown verification** = one-shot session cron (fires 7:56 PDT): confirms sync alive, diffs WashU vs Brev counts, confirms 3 WashU tars + HF models/dataset + wandb, rescues anything Brev-only. ⚠️ session-only — dies if Claude session closes; caffeinate keeping Mac awake.
- HF write token = Dylan1631 fineGrained in **local repo `.env`** (`hf_rIy…`). Note: `/home/nvidia/self-play/.env` token is Dylan1631 **read-only**; Brev cached `~/.cache/huggingface/token` = different account `shrango` (write). Use the local-repo token for Dylan1631 writes.

---

## 2026-06-29 (late) — divfix relaunched to completion (THE clean-diverse test); vfix curriculum Vendi=4.68 (collapsed≈digit)

**Vendi scoreboard (matched n=65):** nopack digit 3.3 (eval 0.6415) · vfix curriculum 4.68 (eval 0.6296, COLLAPSED — barely above digit) · divfix verified curriculum 21.7 (UNTESTED). So every COMPLETED run trained on a ~monoculture (Vendi 3-5 → ~0.63); we have ZERO eval at the diverse end (Vendi ~22).

**Iter-time breakdown (from vfix end-to-end, ~10.7h total):** Stage A questioner 2.0h (18%) · Stage B generate+band-eval+judge 3.0h (28%, but judge itself ~3min — the rest is solver band-sampling) · **Stage C solver-train 5.6h (52%)** · eval ~10min. ⇒ ~80% of wall-clock is the SOLVER (train + band-sampling), only ~18% the questioner. The band-eval (~2.9h sampling solver n=9 ×1000 Qs, then discarded) is the biggest throwaway-cost optimization target.

**Verification deep-dive (DPW pushed on the low verifiable rate):** the '10% verifiable' was a THRESHOLD ARTIFACT — I used VERIF_MIN=0.34 which counts only 3/3-unanimous (verified=1.0); the 2/3-consensus (verified=0.333, passes MIN_AGREE=2, gets a label) were excluded. Real rate ~24% (verified>0). **My REWARD_CLIP VERIF_MIN=0.34 was therefore an over-strict BUG (should be ≤0.33) — contributed to vfix/divclip collapse.** Inspected no-consensus rollouts (programs run 99%, not crashes): failures are ~25% simple-but-buggy programs (base-4B miscodes, e.g. 3 diff answers incl 0 on a trivial digit∩digit-sum Q), ~40% genuinely hard (iterated σ-sequences, factorial-trailing-zeros → error/timeout), ~25% ambiguous/garbage Qs (rambling word-letter spec), few non-integer answers. Lever for more verifiable-diverse Qs = stronger judge/coder or more programs (K=3→5-7), NOT parsing. Program SOURCE not saved by pipeline (verify.py writes temp file, deletes); only parsed program_outputs in judge.jsonl. Qwen3.5 self-critique (SELF_FILTER) catches only 38% of its own ill-posed Qs — not worth enabling; program verifier (89% clean) is better.

**LAUNCHED divfix fresh** (no resume point — prior cancel was pre-step-1): run_divfix_chain.sh tmux sp_divfix, REWARD_FIXEDPCT=1 VENDI_GOLDEN=0 (no gate) NO clip, from base, MAX_ITER=1, A→B→C→eval. Watcher bz5bnbf5d. ETA ~10.7h (~07:40 UTC Jun 30). THE test: does clean-diverse (Vendi 22) beat digit 0.6415? Then chain iter2 (never successfully run — cbfix iter2 stalled kept=0, nopack iter2 killed).

---

## 2026-06-29 — Vendi of verified-diverse ≈ starting (NOT collapsed); vfix(fixed-21) FAILED to hold 21 but eval'd 0.6296; divfix relaunched to completion (the clean-diverse test)

**Vendi comparison (pipeline's own `_vendi_nystrom`, matched n=65):** divfix RAW (base/start) **18.7** · divfix VERIFIED>0 subset **21.7** · nopack DIGIT (0.6415) **3.3**. ⇒ **the verifiability filter PRESERVES diversity** (verified subset Vendi 21.7 ≈ starting ~21, even slightly higher than a raw sample b/c it drops the questioner's near-duplicates) and is **~6.6× the digit monoculture**. So the clean (89% well-posed) verified-diverse curriculum is also genuinely diverse — clean AND diverse, sitting unused after Stage B.

**Run-identity clarification (DPW asked):** the Vendi-21.7 curriculum = **divfix** (NO cap, held diversity naturally, never trained a solver). The **fixed-21 run = vfix** (`VENDI_GOLDEN=21` setpoint gate) which **FAILED to hold 21** — dom-topic collapsed 33→47→52→59→68→**76%** over 6 steps (the verifiability clip dominated; the setpoint gate couldn't counter it). **vfix DID finish end-to-end → `RESULT_vfix_it1 = 0.6296`** (n=675, fmt 1.0), but on a COLLAPSED 507-row curriculum ≈ digit (within noise of 0.6415). So vfix's 0.6296 is NOT a diverse-curriculum result.

**Eval scoreboard (Oly-675 @8192):** base 0.573 · nopack digit 0.6415 · vfix(collapsed) 0.6296 · **divfix(clean diverse, Vendi 22) = UNTESTED**.

**LAUNCHED divfix to completion** (`run_divfix_chain.sh`, tmux `sp_divfix`, REWARD_FIXEDPCT=1 DW=0.5 **VENDI_GOLDEN=0** to disable the later-added gate = original divfix full-diversity behavior, NO clip, from base, MAX_ITER=1, A→B→C→eval). Watcher `bscs1sqee`. **This is THE test: does a clean+diverse (Vendi~22) curriculum beat/match the digit-monoculture 0.6415?** Watch: does Stage-A diversity hold (~31%), Stage-B band+verified row count (curriculum size — diverse Qs have lower verified rate, could be small), solver curve, eval.

## 2026-06-29 — MEASUREMENT-ERROR CORRECTION (DPW caught it): judge the PROGRAM label, not the questioner's claim. "98% broken" was wrong.

**The whole "questioner can't author correct problems → capability wall" conclusion (this session) was based on judging the WRONG column.** gpt-5.5 was scoring the questioner's `\boxed{}` CLAIMED answer — but the pipeline THROWS THAT AWAY. Per `judge.py`/`verify.py`: the judge writes K Python programs per question, executes them, and **program-consensus becomes the label** (`answer` column = `verified_answer`); the questioner's claim is only used for the difficulty/band score. So the questioner being bad at solving its own problems **does not matter** — that's the entire point of the verified arm, which I'd ignored.

**Corrected metrics (judging the actual labels / well-posedness):**
- **nopack (0.6415 curriculum), judged with PROGRAM labels: 92% well-formed** (46/50) — NOT 98% broken. That's simply why 0.6415 worked: clean curriculum. (Kills the earlier "robust to label noise" hand-wave — there wasn't much noise.)
- **divfix diverse questions, well-posedness only (gpt-5.5, ignore answer): RAW 68% well-posed** (34/50) — 32% ill-posed (empty sets / contradictions / ambiguity). `~/Desktop/divfix_diverse_subsample.md`.
- **divfix VERIFIED subset (verified>0), well-posedness: 89% well-posed** (40/45) — the verifiability filter DROPS most ill-posed Qs (empty/contradictory fail program consensus), bringing diverse questions to near digit-level cleanliness (digit=92%). `~/Desktop/divfix_VERIFIED_subsample.md`. **Yield ~24% verified>0 → ~1000 gen → ~240 verifiable, ~210 clean diverse questions = a real usable curriculum (0.6415 trained on 943).** This is the strongest evidence yet that a clean diverse curriculum is achievable; the remaining 11% ill-posed (verifier false-positives) could be cleaned by a gpt-5.5 validation pass.
- **vfix diverse non-digit, program labels: 67% well-formed** (8/12).
- nopack judge stats: **943/946 (99.7%) got program consensus**; questioner/majority agreed with program only **775/943 (82%)** → 18% relabeled by programs.

**Corrected understanding:**
1. The pipeline WORKS — the program verifier relabels the questioner's wrong guesses with correct answers (~92% correct for digit, ~67% for diverse).
2. The real constraint is **program-verifiability** (can K programs solve+agree), NOT questioner self-correctness. Digit/NT/counting → program-checkable → clean labels. Geometry/proofs → no consensus → dropped (correct behavior).
3. Diverse questions are ~68% well-posed (4× more broken than digit's ~92%, but a usable 2/3 majority). A clean diverse curriculum is achievable after filtering the bad third (program-verifier drops many; gpt-5.5 could validate the rest).

**WE HAVE ZERO END-TO-END DIVERSE RESULT.** divfix/divclip/divvendi/vfix were ALL killed before Stage C — no solver, no eval, no saved program labels (divfix `models/` has only `_q`). Only nopack (digit→0.6415) ran end-to-end. **The diverse-vs-digit question is empirically OPEN** — I'd wrongly declared it closed on a broken metric. To answer it: run a diverse questioner A→B→C→eval (optionally insert a gpt-5.5 validation filter between B and C for a known-clean curriculum). vfix still running (collapsing, not useful); divfix is the right relaunch candidate.

## 2026-06-29 — vfix: FIXED per-batch Vendi diversity (replaced cumulative-bank gate)

divvendi diagnosis: it was COLLAPSING anyway (dom-topic 35→43% by step2, *faster* than divfix's 32%) despite full diversity gate — because the 98% verifiability-CLIP zeros the diversity contribution for the broken majority, so diversity only "counts" for the ~2% verifiable (≈digit) → clip drags toward the narrow verifiable set. Verifiability only weakly rising (mean_verified −0.61→−0.24, noisy, still negative). Also the cumulative-bank gate never engaged (needs ≥500 bank samples; showed gate 1.0) and setting VENDI_GOLDEN=21 would've *zeroed* diversity (bank already past 21, gate→0) — the sticky cumulative bank can't hold a setpoint.

**DPW: "do vendi as fixed."** `diversity.py` already had the mechanism as its cold-start branch (within-batch leave-self-NN novelty). Patched (backup `.bak_vfixed`, env-gated `VENDI_FIXED=1`, jy-safe) to FORCE that branch always: **diversity = per-batch novelty (each question vs the rest of the CURRENT batch), no cumulative bank, no golden cap.** This directly penalizes within-batch redundancy, is a fixed window (can move up/down, no stickiness), and removes the gate weirdness. Launcher `run_vfix_chain.sh` tmux `sp_vfix`, `/data/selfplay_vfix`, DW=0.5, kept verif+difficulty clip. Watcher `b6fib4u9y` (tracks diversity + clip-rate + verifiability-trend).
- **Caveat still open:** the 98%-clip-dominance is unchanged — fixed-Vendi gives a cleaner diversity *signal* but it still only applies to the ~2% non-clipped. If it collapses again, the clip (not the diversity mechanism) is the binding constraint, → loosen VERIF_MIN or accept the capability wall.
- **Infra (compounding lesson):** killing a `sp_*` tmux orphans BOTH the `start_vllm_server.py` service AND its `VLLM::EngineCore` children (all reparent to PPID 1, keep ~70GB GPU each). Must `kill -9` all PPID=1 vLLM/service procs after a tmux kill (filter by start-time=launch-time; jy untouched). Done before each relaunch.

## 2026-06-29 — divvendi: Vendi-capped diversity (replaced decay) + verifiability clip

DPW: undo the diversity decay; instead **cap diversity reward by Vendi (~starting Vendi)**. **Found the mechanism already exists** in `diversity.py`: `gate = clip((VENDI_GOLDEN − bank_matched)/VENDI_GOLDEN, 0,1)`, `diversity_reward = DIVERSITY_WEIGHT·gate·novelty` — ramps diversity reward → 0 as bank Vendi → `VENDI_GOLDEN`. It was inert because `VENDI_GOLDEN=85.6` (MATH-500 ref) is unreachable (bank tops ~10–33). **Subtle bug fixed:** the REWARD_FIXEDPCT z-scoring CANCELS the gate (gate is a per-batch constant scale, removed by `(x−mean)/std`), so setting VENDI_GOLDEN alone does nothing — patched `caller_penalty.py` (backup `.bak_vendi`) to apply `_gate` to the **z-scored** diversity term: `W_DIV·_gate·z(div)`. Removed the decay block.
- **Measured Vendi levels (from divfix diverse run):** starting/first-batch bank Vendi **≈21**, rising to **~33** as the diverse bank fills. Set **`VENDI_GOLDEN=28`** (cap around starting, with a ramp), `DIVERSITY_WEIGHT=0.5`. Kept verif+difficulty clip (VERIF_MIN0.34, diff[0.3,0.7]).
- **HONEST CAVEAT (sticky bank):** `bank_matched` is the cumulative bank's Vendi (only grows). So once the bank fills to ~28 (a few batches), gate→0 and diversity reward turns OFF for the rest of training — functionally similar to the decay we just removed (diverse early, quality-focused later). If sustained diversity is wanted, raise VENDI_GOLDEN or use a windowed bank. Watching whether this lands "diverse-then-consolidate" vs holds.
- Launcher `run_divvendi_chain.sh` tmux `sp_divvendi`, `/data/selfplay_divvendi`. Watcher `b6rp7ygpx` tracks diversity + gate/bank_matched + clip count; will re-run gpt-5.5 broken-% at step 4.
- **Infra note:** killing a `sp_*` tmux ORPHANS the detached `start_vllm_server.py` reward service (PPID→1) which keeps holding GPUs — must `kill` those PIDs explicitly after a tmux kill (identified by start-time = launch-time; jy left untouched).

## 2026-06-29 — divfix RESULT (98% broken) + LAUNCHED divclip (verifiability-clip + decaying diversity)

**divfix verdict: the fixed-pct diversity weight PREVENTED collapse (dom-topic flat ~31% across steps 1–4 vs OLD 35→77%, 7 topics held) — but exposed a capability ceiling.** gpt-5.5 judged 60 sampled questions: **57/58 = 98% ILL-FORMED** (mostly *wrong claimed answer* — questioner can't solve its own problems — plus genuinely ill-posed/empty/contradictory). Cross-ref to pipeline `verified`: the program-verifier *correctly rejects ~91%* of broken (verified≤0), only 9% leak as verified>0 — BUT because 98% of the pool is broken, **~80% of what survives verify is still broken** (9%×98% false-pass ≫ 2% true-good). So diversity-fix gives a **diverse-but-broken-and-starved** curriculum; the digit-collapse gave a **clean+verifiable** one (→ why 0.6415 worked). **Bottleneck = questioner CAPABILITY to author correct problems, not reward weights/verifier.** divfix killed.

**divclip (DPW's fix): clip + decay.** Extended `caller_penalty.py` (backup `.bak_clip`, AST-checked, env-gated, jy-safe): (1) **`REWARD_CLIP=1` verifiability+difficulty GATE** — `verified < VERIF_MIN(0.34)` OR `solver_score ∉ [CLIP_LO 0.3, CLIP_HI 0.7]` → reward=0. Rationale: currently a broken-diverse question nets POSITIVE reward (uncertainty+diversity > −0.75 verif penalty); the gate zeroes it so the questioner can only profit from verifiable+frontier questions. (2) **dynamic diversity decay** `DIV_DECAY=1`: W_DIV 0.3→floor 0.1 linearly over ~`DIV_DECAY_CALLS=10` reward calls (early diversity to break monoculture, later let verif/difficulty dominate). Launcher `run_divclip_chain.sh` tmux `sp_divclip`, `/data/selfplay_divclip`, else same recipe (flash+rp=false, from base, MAX_ITER=1). Watcher `bqxf0k6nl`.
- **Expectation (honest):** clip likely pushes the questioner PARTLY back toward the verifiable digit-class — but a CLEAN one (correct labels), not the broken-diverse mess. Will measure: diversity curve + re-run gpt-5.5 broken-% on divclip questions (target: broken% ≪ 98%). Key test: does clean+somewhat-broader beat the clean-monoculture 0.6415?

## 2026-06-29 — LAUNCHED divfix run: fixed-percentage-weight (variance-normalized) diversity reward

Implemented the fix from the diversity-reward discussion (DPW's "fixed percentage weight, use current weighting"). **Patch** (`caller_penalty.py`, backup `.bak_fixedpct`, AST-checked): env-gated `REWARD_FIXEDPCT=1` branch that **z-scores each reward component across the batch then weights by current weights {uncertainty 1.0, diversity DW=0.5, verif VW=0.75}** → diversity gets a fixed ~22% SHARE of the reward signal regardless of raw magnitude (fixes scale-dwarfing → mode collapse). std≈0 → 0 (no noise blowup); malformed → −1 floor. **Purely additive/guarded: default (no env) = byte-identical original behavior**, so co-tenant `jy` unaffected (jy currently idle anyway; all 8 GPUs were free).
- **NOT included (per DPW — current weighting only):** the bank-anchored *absolute* penalty (the part normalization can't give — can't reverse a fully-collapsed state, only prevent the slide from a fresh start). This run tests whether a real diversity *weight* prevents collapse forming from base.
- **Launcher** `run_divfix_chain.sh` tmux `sp_divfix`, **NEW path `/data/selfplay_divfix`** (preserves the 0.6415 `selfplay_nopack` artifacts), `REWARD_FIXEDPCT=1`, else identical recipe (flash+rp=false, verified VW0.75 DW0.5, no CVBAND, from base, MAX_ITER=1, Q6/S20/NUM1000/EVAL_N9).
- **Gate (fast):** watcher `b8selvlul` tracks per-step questioner dominant-topic% vs the OLD collapse trajectory (35→46→59→77→89→95%). If diversity holds (stays low) → let it flow to B/C/eval; if it still collapses by ~step4 → kill (~2h saved) and add the bank-anchored penalty. **A/B target: does diversity-fixed beat/≈ the collapsed 0.6415?** (iter1 was 100%-collapsed yet hit 0.6415, so "diversity helps eval" is the hypothesis under test — could be neutral.)

## 2026-06-28 — QUESTIONER MODE-COLLAPSE found (DPW caught the sus-high 0.962 reward) — diversity reward is failing

Inspected iter2 Stage-A questioner rollouts (`rollout_dumps/nopack_it2_verified_q/{1,6}.jsonl`, 240 rows/step) after questioner reward hit **0.962** (vs iter1's 0.561). **Questions are individually well-formed (240/240 have `<question>…</question>`+`\boxed`, median 268 ch) but SEVERELY mode-collapsed onto ONE template** ("set of integers in [1,N] containing digit d; f(n)=product/sum of digits; compute Σf(n)"):

| | digit/number-rep topic share |
|---|---|
| iter1 step1 (from base) | 35% (7 topics — healthy) |
| iter1 step6 | **95%** |
| iter2 step1 | 97% |
| iter2 step6 | **100%** (240/240 identical 60-char prefix) |

**The 0.962 is reward-GAMING via collapse**, not quality: the questioner found "digit-sum-over-range" reliably maxes self-consistency-band + verifiable + format, and collapsed onto it; continued training drove 95%→100%. **The Vendi diversity reward (DW=0.5) is NOT preventing semantic mode collapse.** Claimed `\boxed` answers are unreliable (acc≈0 in dumps — SAME for iter1, so acc≈0 is normal/not the signal; but the same problem n≤1000/digit7/product is labeled BOTH 14562 and 142680 ⇒ questioner can't solve its own problems, just emits a number for format).
- **Key nuance:** iter1 was ALSO 95% collapsed by step6 yet its solver still hit 0.6415 (digit/number-theory problems are apparently OK solver training + OlympiadBench has NT). So collapse ≠ guaranteed failure. iter2 at 100% collapse + reward-gaming questioner ⇒ uncertain whether 2nd iter helps or overfits solver to one type.
- **This is arguably a deeper bottleneck than packing/band:** the questioner objective rewards collapse. Real fixes: stronger/embedding-level diversity penalty, topic-coverage constraint, or cap questioner training steps before collapse. (Tie to §9.10 diversity-bank issues.)
- **Decision (DPW): KILLED iter2** (2026-06-28 ~16:30, in Stage B) — once the 100% collapse was diagnosed it was no longer informative; freed the box for a fixed-diversity-reward run instead. Clean kill: `tmux kill-session -t sp_nopack2`, GPUs 0–3 → 0 MiB, no orphans, co-tenant `jy` untouched. **iter1 0.6415 solver preserved** at `/data/selfplay_nopack/models/nopack_it1_verified_s/`.

**DIVERSITY-REWARD ROOT CAUSE (why Vendi doesn't stop collapse) + fix design.** Reward formula `caller_penalty.py:175`: `final_score = uncertainty + diversity_reward + verif_term` (the dump field "accuracy" IS `diversity_reward`). From `artifacts/.../challenger_batches.md`: per-question `diversity` column = **0.00** for all iter2 questions; reward driven by `verified`(±1)+`uncertainty`. Bank Vendi actively FALLS across iter2 batches (12.8→11.8→10.7→9.7→9.0, mean_novelty 0.00, rel_delta ~−0.08) — **Vendi correctly DETECTS collapse but the reward derived from it is a non-negative novelty bonus floored at ~0, so redundancy is never penalized**, while `verified` is trivially maxed by one easy template ⇒ monotone collapse. (iter1 fresh bank: vendi 1.0→21.3, novelty +0.28 — diverse questions paid out big early, then the bonus decays to 0 as the bank fills.)
- **Fix design (after DPW discussion):** (1) **variance-normalize reward components** so diversity isn't scale-dwarfed by verif±1 (necessary for scale; also subsumes "signed-relative-to-batch" — but NOT sufficient: at full collapse there's no within-batch variance to normalize, and dividing by ~0 std amplifies noise). (2) **bank-anchored ABSOLUTE diversity penalty** (e.g. `−λ·bank_match` / negative when novelty-vs-bank < thresh) — the part normalization can't give: provides escape gradient even at zero batch variance, and flags "batch repeats history." Note GRPO already group-centers total reward yet collapse still happened ⇒ relative centering alone insufficient, confirming the absolute term is the missing piece. (3) **difficulty reward = smooth bump (triangular/Gaussian peaked at 0.5) on TRUE solve-rate-vs-verified**, not the self-consistency `uncertainty` (§9.1) — hard-clip [0.4,0.6] avoided (zero gradient outside the band = dead zone). Optional cheap stopgaps: KL/entropy-to-base on questioner; within-batch pairwise-similarity penalty; cap Q-steps (collapse ~done by step 4–5). **MUST A/B (collapsed vs diversity-fixed → eval): iter1 was 95% collapsed yet hit 0.6415, so "less collapse → better eval" is a hypothesis, not given.** Next: inspect `diversity.py` (novelty/bank_match/`gate`) to wire the absolute penalty. GPUs free.

## 2026-06-28 — LAUNCHED nopack ITER2 (does the loop compound from 0.6415?)

Seeded iter2 from iter1's trained checkpoints (NOT base): Q=`nopack_it1_verified_q/global_step_6`, S=`nopack_it1_verified_s/global_step_20` (the 0.6415 solver, complete HF ckpt w/ safetensors). Same recipe (flash+rp=false, verified VW=0.75 DW=0.5, no CVBAND, Q_STEPS=6 S_STEPS=20 NUM_SAMPLES=1000 EVAL_N=9), `run_nopack_iter2.sh` tmux `sp_nopack2`, `/data/selfplay_nopack`, ITER=2, **diversity bank PERSISTED from iter1** (not reset — continues the bank). Watcher `b4h5xdwn9`. **Test:** does a 2nd iteration beat iter1 0.6415 (compounds), plateau (~0.64), or regress? Note prior cbfix iter2 STALLED with CVBAND (kept=0, questioner couldn't hit the band for the stronger solver) — here CVBAND is OFF so the self-consistency band should still pass questions; watching whether the iter2 solver curve still CLIMBS (the real signal, per the iter1 lesson). ETA ~5–6h.

---

## 2026-06-28 — LAUNCHED nopack full-pipeline run (fast script, packing reverted) — the fix applied to the real loop

After the 2×2 settled packing as the −0.12 culprit (flash innocent), applied the fix to the production pipeline and launched a real iteration. **Edit:** `iteration_rzero.sh:71` `use_remove_padding=true→false` (backup `.bak_nopack`; flash_attention_2 + FSDP offload + ppo_max_token_len speedups all kept; shared block ⇒ both questioner Stage A and solver Stage C get rp=false). **Launcher** `run_nopack_chain.sh` (tmux `sp_nopack`, `/data/selfplay_nopack`): from base, verified VW=0.75 DW=0.5, **no CVBAND** (matches original 0.630 recipe — only packing changed vs original), MAX_ITER=1, Q_STEPS=6 S_STEPS=20 NUM_SAMPLES=1000 EVAL_N=9, `CUDA_VISIBLE_DEVICES=0,1,2,3`, logger=[console,wandb]+rollout/val dumps, per-iter Oly@8192. Refs: base 0.573 / rzc_it1 0.630 / flash_iso solver-iso 0.633.
- **Goal:** does the *fixed fast pipeline* produce a good iteration (beat base, approach/≥0.63)? Solver-isolation already says the fixed kernel gives 0.633; this tests the full loop (questioner+band+solver) end-to-end with the fix.
- **Caveat (flagged):** CVBAND off + single questioner draw ⇒ questioner-curriculum variance still in play (the known 0.474–0.630 spread); a low landing would likely be the draw, not packing. Repeat / add CVBAND after seeing this draw.
- Watcher `b7c99hmcy`. Stage A starting clean (512 qprompts written, reward svc on GPU 0). ETA ~5–6h (Stage A ~2h + band + solver ~2.5h + eval).

**STAGE A RESULT — questioner curve REPRODUCES rzc iter1 (packing fix recovered the divergence).** Per-step questioner reward (critic/score/mean) vs rzc iter1 reference:

| step | nopack | rzc it1 | Δ |
|---|---|---|---|
| 1 | −0.249 | −0.344 | +0.095 |
| 2 | −0.360 | −0.270 | −0.090 |
| 3 | −0.093 | −0.081 | −0.012 |
| 4 | +0.207 | +0.218 | −0.011 |
| 5 | +0.428 | +0.463 | −0.035 |
| 6 | **+0.561** | **+0.587** | **−0.026** |

vs the packing-on **redo step-6 +0.259** (the divergence we were explaining). ⇒ **`use_remove_padding=true` was corrupting the QUESTIONER's gradients too** — reverting it makes the Stage-A curve track iter1 within ~0.03 at step 6 (residual Δ explained by diversity-bank cold-start + stochastic n-sample solver scoring + timing-split reward §9.10, none of which are packing). This is independent corroboration of the packing finding on a *different* model/stage. Stage A saved questioner ckpt `nopack_it1_verified_q/global_step_6`; Stage B (band) running. (The Traceback at the A→B handoff is the benign DataLoader-worker teardown, same as all runs — pipeline advanced to B normally.)

**STAGE B/C — DIVERGES from iter1 despite matching questioner (band/curriculum, NOT packing).** Stage B band-filter [0.3,0.8] passed **943 solver-training rows** (`[judge] 946 band-filtered → 943 verified`, of NUM_SAMPLES=1000 ⇒ **94.6% band-pass**) vs **rzc iter1's 108**. Adaptive batch → **256/128** (vs rzc 64/32). The kept set is HARD/low-correctness: solver Stage-C reward starts **0.150** (rzc started 0.565) and climbs slowly (0.15→0.19→0.24→0.28→0.34 by step10) — won't reach rzc's 0.79. So even with packing fixed and the questioner reproducing iter1, the **band passed a 9× larger, much-harder, mostly-wrong-but-self-consistent set** (the §9.1 band-bug, more extreme than the redo's 196). **This is the questioner-non-determinism (§9.10: cold diversity bank + stochastic scoring) × band-has-no-correctness-guardrail (§9.1) bottleneck — exactly what the 06-25 investigation concluded is the REAL cap, and packing was never it.** rzc's 108-learnable-row draw was favorable; this draw is unfavorable.

**RESULT (decisive, and it OVERTURNS the mid-run pessimism): nopack iter1 = OlympiadBench-675 @8192 `acc|fmt=0.6415 fmt=1.0 trunc=0 mean 7960ch` — NEW BEST, beats base +0.068.** Solver curve CLIMBED monotonically 0.150→0.228→0.339→0.412→**0.515** (steps 1/5/10/15/20). The "943 rows / 0.15 start" alarm was a MISREAD: the big in-band set was hard-but-LEARNABLE, not unlearnable garbage. **The correct discriminator is the solver-reward TRAJECTORY, not row-count or start value:** the failed redo (eval 0.474) had reward that *declined* 0.246→0.17 (garbage curriculum); this run *climbed* 0.15→0.515 (learnable). Same surface features (low start, large band-pass), opposite slope, opposite outcome.

| Oly-675 @8192 | pass |
|---|---|
| base Qwen3.5-4B | 0.573 |
| rzc iter1 (original 0.630) | 0.630 |
| flash_iso (solver-iso) | 0.633 |
| **nopack full pipeline (this)** | **0.6415** |

**Conclusions:** (1) The full self-play loop (questioner+band+solver from base) is **net-beneficial (+0.068 over base)** once packing is fixed — it did NOT cap at base or collapse. The earlier "loop caps at base / iter2 regresses" story was substantially the packing bug + unlucky *declining* draws, not a fundamental ceiling. (2) 0.6415 vs 0.633/0.630 is within the ~0.01 run-to-run noise measured in the 2×2 ⇒ "top of cluster, clearly beats base," not a confident record. (3) Lesson recorded: judge curriculum quality by reward SLOPE, not band-pass count or starting reward — the 943-vs-108 row gap was a red herring. **Next levers (DPW's call): (a) iter2 from this 0.6415 solver — does a 2nd guarded iteration keep climbing or plateau? (b) variance: repeat nopack iter1 ×N for an error bar; (c) nopack + CVBAND to compare a correctness-gated curriculum. GPUs free; sessions cleaned up.**

---

## 2026-06-28 — DECISIVE: `remove_padding=true` (packing) ALONE costs −11.3pt on Qwen3.5 GDN (0.621→0.508), drops below base. 2×2 complete.

**The kernel-iso 2×2 finished. Clean apples-to-apples isolation of `use_remove_padding` (packing), holding attn=sdpa + every other knob byte-identical (108-row `rzcev_it1_verified_verl08`, 20 steps, n=5, 4096, lr1e-6, kl1e-2).**

| arm | `remove_padding` | attn | **OlympiadBench-675 @8192** | format_rate | mean chars | vs base 0.573 |
|---|---|---|---|---|---|---|
| **orig-iso** (control) | **false** | sdpa | **0.621** | 1.00 | 7509 | +0.048 — reproduces SOLVER2 0.630 (within noise) |
| **pad-iso** | **true** | sdpa | **0.508** | 1.00 | 5948 | −0.065 — **below base** |

**Δ(packing) = −0.113.** Both `format_rate=1.0`, `truncated~=0`, same harness (eval_oly_shard.py, 4-GPU greedy @8192, full 675) ⇒ the gap is **pure `acc|formatted`** (real reasoning), NOT formatting/truncation. Packing also made the model **terser** (5948 vs 7509 chars) AND worse — the opposite of a length problem.

**Conclusions:**
1. **`use_remove_padding=true` is a real gradient corruptor on the Qwen3.5 hybrid (GDN/linear-attn) model — ~11pt eval hit, sufficient to push a 0.621 solver below the 0.573 base, independent of flash attention.** Sole variable changed; attn held at sdpa.
2. **The control (sdpa+rp=false) reproduces the 0.630-era number (0.621)** — confirms once more the code/data are clean; the earlier reproduction failures (redo 0.474, kernel-iso ramble) were driven by *self-introduced* confounds (rp=true packing here quantified; questioner-curriculum variance separately).
3. **Distinct from the flash+rp=true kernel-iso collapse:** that one *rambled* (no-`\boxed` rumination, train-acc 0.80→0.58); sdpa+rp=true here is *terser-and-worse*, not rambling. So flash adds a separate failure mode on top of packing — packing alone is already −11pt; flash+packing is worse.
4. **Does NOT change the real-experiment story:** the actual iter1(0.630)→iter2(0.545) regression ran rp=**false** both iters (provenance settled below), so packing was never in the real loop — that regression remains the questioner-curriculum/favorable-draw finding. Packing is the confound that corrupted my *optimization/repro* runs (smoke_opt, kernel-iso, redo), now quantified.

**Harness fix landed + validated in production:** the verl ckpt-format footgun (FSDP-sharded `.pt`, empty `huggingface/` subdir → vLLM "Cannot find any model weights" → eval n=0 ZeroDivisionError) is fixed by an auto-merge block in `run_oly.sh` (`.bak_premerge`) that runs `verl.model_merger merge --backend fsdp` if `$MODEL` lacks safetensors. pad_iso's auto-eval exercised it live (merged → repointed → 675 rows). Merged HF ckpts at `/data/{orig,pad}_iso/solver/global_step_20/actor/hf_merged` (9.08GB each). See ARCHITECTURE.md Footguns.

**Status:** sdpa-row runs DONE + cleaned up. **Then completed the full 2×2 over attn × remove_padding** (GPUs were idle → kept them working): eval'd the existing kernel_iso (flash+rp=true) ckpt via the merge fix (no retraining), and launched flash_iso (flash+rp=false, the missing cell). flash+rp=false is **untested on this env** but **smoke-passed** (reached step 2, score 0.548 — the path runs; revises §9.9's "old kernel path crashes" which referred to offload specifics).

**FULL 2×2 COMPLETE (OlympiadBench-675 @8192, greedy, format_rate≈1.0, same harness; 108-row/20-step/n5/4096, only attn × remove_padding vary):**

| | rp=false (no packing) | rp=true (packing) |
|---|---|---|
| **sdpa** | **0.621** (orig, 7509 ch) | **0.508** (pad, 5948 ch) |
| **flash** | **0.633** (flash_iso, 7612 ch) | **0.495** (kernel, 8536 ch) |

- **`use_remove_padding=true` (packing) is the ENTIRE effect, −0.12, identical across both attn backends.** No-packing reproduces ≈0.63 (beats base 0.573); packing drops to ≈0.50 (below base). This is **no longer n=1**: two independent no-packing runs (0.621, 0.633) and two independent packing runs (0.508, 0.495) — within-level spread ≈0.012, so the −0.12 packing gap is ~10× the noise. Variance-controlled by construction.
- **Flash attention is INNOCENT** (the original suspect, now cleared): 0.621→0.633 at rp=false and 0.508→0.495 at rp=true — flash moves nothing beyond noise (±0.013). The kernel-swap confound was **100% the `remove_padding=true` that rode along with it**, not flash. Interaction negligible.
- **The two packing arms fail in OPPOSITE directions** — sdpa+packing goes **terse**-and-worse (5948 ch), flash+packing goes **verbose/rambling**-and-worse (8536 ch, longest of all four). Packing corrupts accuracy regardless; attn only steers *how* it fails (terse vs ramble). Both no-packing arms sit at normal length (~7500–7600 ch).
- **Bottom line:** never enable `use_remove_padding` on Qwen3.5 GDN for a learning-faithful run; flash_attention_2 is fine. flash_iso (flash+rp=false) is the recommended fast+faithful kernel: 0.633, healthy curve (score→0.77, rlen stable ~2050, no rambling), and it RUNS on the current env (the §9.9 "old path crashes" was an offload-config issue, not flash itself).
- kernel_iso eval caveat: its ckpt had no `generation_config` (merge fell back to model-config default) — negligible under greedy eval; format_rate 0.9985 (1 no-answer). All four trained cleanly rc=0 (the end-of-run `DataLoader worker … Killed` Traceback is benign teardown).

**Open fork for DPW (after flash_iso lands):** (a) the questioner-curriculum thread (the actual iter2-regression lever) — loosen CVBAND / variance study; (b) upload iso ckpts to HF (Dylan1631) if worth keeping.

---

## 2026-06-28 — orig-iso 2×2 control HEALTHY at step16; auto-chain to pad_iso wired server-side

**Status check (00:51 UTC), after the orig-iso watcher hit its poll cap.** `sp_oiso` (control: sdpa + `use_remove_padding=false` + 8192, 108-row `rzcev_it1_verified_verl08`, 20 steps) is **alive and progressing cleanly**: at global_step 16, `critic/score/mean` ~0.71–0.76 (rising), `response_length/mean` 2159 (stable, not rambling), `aborted_ratio=0.0`, step time ~456s (~7.6 min/step; gen 270s dominates, update_actor 106s, throughput 405 tok/s). This is the SOLVER2-like trajectory (monotone rise, terse) — i.e. the **control does NOT degrade**, unlike kernel-iso (flash+rp=true) which rambled to 0.58. ~4 steps + full-675 Oly@8192 eval remain (~30–40 min).
- **Auto-chain wired (was NOT actually wired before):** `run_orig_iso.sh` ends after its own eval (`RESULT orig_iso_8k`) and does **not** launch pad_iso; the prior watcher only polled. Added server-side waiter `chain_oiso_to_piso.sh` (tmux `sp_chain`, log `/data/chain_oiso_to_piso.log`) that blocks on `tmux has-session -t sp_oiso`, then on session-end launches `run_pad_iso.sh` in tmux `sp_piso` → `/data/pad_iso_run.log` (guarded against double-launch). Survives laptop sleep. pad_iso = identical config but `use_remove_padding=TRUE` → isolates packing alone.
- Co-tenant: `jy` session (jinyuan/R-Zero) untouched; staying in GPU lane 0–3. Local milestone watcher `bxdmo0wp0` re-invokes on eval-result / pad-launch / error.

**UPDATE (01:10–01:14 UTC): orig-iso TRAIN finished clean (20 steps, rc=0) but its Oly EVAL crashed `ZeroDivisionError` (n=0) — root-caused to a verl checkpoint-format footgun, NOT the experiment. FIXED.**
- **Root cause:** verl (vllm-0.23 env) saves the actor checkpoint as **FSDP-sharded `.pt`** (`model_world_size_4_rank_{0-3}.pt`, 5.17 GB each = intact weights) + an `actor/huggingface/` subdir containing **only config + tokenizer, NO safetensors**. `run_oly.sh` pointed vLLM at `actor/huggingface` → `RuntimeError: Cannot find any model weights` → all 4 eval shards died in 67s → 0 rows → `agg.py` `fmt/n` with n=0. **`kernel_iso`'s ckpt has the identical empty-`huggingface` structure — this is systematic in this verl version; a merge step is mandatory before any vLLM eval.** Training weights are fully intact → nothing lost.
- **Fix 1 (orig-iso):** ran `python -m verl.model_merger merge --backend fsdp --local_dir .../global_step_20/actor --target_dir .../actor/hf_merged` to consolidate shards → HF safetensors (in progress; CPU-side, didn't disturb pad_iso GPUs).
- **Fix 2 (harness, durable):** patched `run_oly.sh` (backup `.bak_premerge`, bash-syntax-checked, was not open) to **auto-merge if `$MODEL` lacks safetensors** (merges `dirname $MODEL` → `hf_merged`, copies processor configs, repoints `MODEL`). This makes pad_iso's own step-20 auto-eval work, and any future eval. Invoked fresh per eval so the running `run_pad_iso.sh` was NOT edited (safe).
- **Fix 3 (sequencing):** `sp_finish` waiter re-runs the orig-iso Oly eval on the merged ckpt after pad_iso's session ends (GPUs freed) — both 2×2 numbers land on the same harness, no contention.
- **Note:** a pre-existing "cell-B auto-launcher" watcher from the prior session ALSO tried to launch pad_iso — hit `duplicate session: sp_piso` (my `sp_chain` won the race by 7s); the tmux guard prevented a double-launch. Single pad_iso confirmed.

---

## 2026-06-27 — PROVENANCE settled: real verified iter1+iter2 ran sdpa+remove_padding=FALSE → padding is NOT the regression cause; + padding-bug isolation

**Question (DPW):** is the WashU `iteration_rzero.sh` actually the code that ran Qwen3.5 iter1, or could the real run have used `use_remove_padding=true` (Chengson was confused about padding)? **Answered from runtime ground truth** — verl dumps the fully-resolved `.hydra/config.yaml` at every launch, so we don't have to trust the script.

**Findings (from `.hydra/config.yaml` of the ACTUAL runs):**
- WashU `/home/compute/jiaxinh/outputs/2026-06-21/*` = the **majority** arm (`rzcm_*_majority_{q,s}`), all **ACTOR `use_remove_padding=False`, `attn=sdpa`**. (The `use_remove_padding: true` seen by naive grep is `critic.model.use_remove_padding` — a dead default; `adv_estimator=grpo` ⇒ no critic.)
- The **verified** arm (the 0.630→0.545 subject) ran on **Brev**, not WashU: `~/outputs/2026-06-21/18-22-47` = `rzcev_it1_verified_s` (the 0.630), and `~/outputs/2026-06-22/{12-22-16,23-49-38}` = `cont_it2_verified_s` (the 0.545). **Both: ACTOR `use_remove_padding=False`, `attn=sdpa`, max_resp=4096, 20 steps — byte-identical solver config.**
- The ONLY runs that ever used `rp=True + flash_attention_2` are my own `smoke_opt` (Jun 23) and `kernel-iso` (Jun 26) optimization experiments.

**Conclusions:**
1. **Yes, the diffed script == what ran iter1** (proven by runtime config, not mtime), and it ran the **safe unpacked path**.
2. **Padding/packing is NOT the cause of iter2 < base** — the real experiment never enabled it. The `rp=true` degradation is a *separate, self-introduced* bug.
3. iter1 and iter2 ran **identical solver config** ⇒ the regression is in the **iter2 training DATA** (evolved questioner + band/CVBAND), not the solver kernel. Back to the CVBAND thread.
4. **Logging gap confirmed:** real iter1/iter2 used `logger=[console]` w/ no rollout dump → in-training reward curves were NOT saved (only final eval 0.630/0.545 survive). This is exactly why the rollout-dump patch was later added. `main_ppo.log` files are 0-byte; `RUN_verified.log` is a 530B summary; `svc_*.log` are the questioner scoring service, not solver curves.

**Padding-bug isolation (characterizing the footgun, NOT the regression):** kernel-iso (flash+rp=true, 108-row data) DEGRADED acc 0.80(step7)→0.58(step20) via rambling/no-`\boxed` rumination, while SOLVER2 (sdpa+rp=false, same data) rose monotonically to 0.79. Running a clean 2×2 to pin flash-vs-packing: `orig-iso` (sdpa+rp=false+8192, tmux sp_oiso) as control, then `pad_iso` (sdpa+rp=true+8192, tmux sp_piso, auto-launches after) isolates `remove_padding` alone. **Reward code is byte-identical Brev↔WashU; `math.py` format_reward is a DEAD constant 0** (regex requires `<think>` but `enable_thinking=false`) ⇒ solver reward = `0.9·accuracy` exactly, nothing rewards length/rambling — the degradation is corrupted-gradient (packing), not reward-driven.

---

## 2026-06-26 — rzero2 CUDA toolchain FIXED (was broken: cold-compile of Qwen3.5 GDN tilelang kernel); + kernel-vs-questioner isolation

**FOOTGUN found + fixed:** the Brev `rzero2` env (vllm 0.23/torch 2.11+cu130) had an **internally-inconsistent cu13 toolkit** — `nvidia-cuda-nvcc==13.2.78`, `nvidia-cuda-runtime==13.0.96` (CUDART_VERSION 13000), `cccl==13.3.3.3.1`, `crt==13.3.33`. It could NOT cold-compile the Qwen3.5 hybrid (GDN/linear-attn) **tilelang** kernel; all training "worked" only via a **warm `~/.tilelang` compile cache** (built earlier when the toolchain was consistent at 13.2). During the iter1-repro effort I `rm -rf ~/.tilelang`, which **exposed the breakage** — every cold compile then failed with `tl_templates/cuda/instruction/mma.h` → really two layered errors: (1) cccl `cuda_toolkit.h:40` "CUDA compiler and toolkit headers incompatible" = the check `nvcc_version == CUDART_VERSION` failing (nvcc 13.2 vs runtime 13.0); (2) after nvcc→13.0, `ptxas: Unsupported .version 9.2; current 9.0` = tilelang emits **PTX 9.2 (CUDA 13.2 ISA)** but ptxas 13.0 only does 9.0.

**FIX (verified working):** align the whole cu13 compiler set to a consistent **13.2**: `pip install nvidia-cuda-nvcc==13.2.78 nvidia-cuda-runtime==13.2.75 nvidia-cuda-cccl==13.2.75 nvidia-cuda-crt==13.2.78 nvidia-cuda-nvrtc==13.2.78`, then `rm -rf ~/.tilelang`. Now nvcc==CUDART==13.2 (cccl passes) AND ptxas 13.2 handles tilelang's 9.2 PTX; cubin runs fine on the CUDA-13.0 **driver** (580.65.06) via forward-compat. **torch 2.11 verified still working on CUDA** (doesn't hard-pin cuda-runtime). tilelang now **cold-compiles the GDN kernel successfully** and the cache repopulates. The env is now *better* than the original freeze (which relied on a cache to mask the inconsistency). **Lesson: never `rm -rf ~/.tilelang` on a box whose toolchain can't cold-recompile; back it up first.** Also: a tooling bug bit repeatedly — `pkill -f oldkernel_smoke` matched the ssh shell running the command (its cmdline contained the string) → silently killed its own shell → launches "didn't take" + stale logs. Use PID-based kills, never `pkill -f <string-also-in-your-command>`.

**Kernel-vs-questioner isolation (in progress):** to test whether the 0.474-vs-0.630 regression is the kernel flag (sdpa→flash) or the questioner's curriculum, train the solver on the **original's exact 108-row parquet** (`rzcev_it1_verified_verl08`) with the current **flash** kernel, 20 steps + eval, vs SOLVER2 (sdpa, same data, → 0.630). **Step-1 already matched** (flash 0.554 ≈ sdpa 0.565, from the earlier it1-repro). SOLVER2 sdpa curve rose 0.565→0.79 (clip 0.44→0.07). The full flash run (`run_kernel_iso.sh`, tmux sp_kiso) now runs on the fixed env — streaming flash-vs-sdpa per step. **Working hypothesis (DPW's, well-argued):** a 0.15 sign-flipping swing is too large for near-equivalent infra OR random variance → it's **systematic, and it's the questioner's curriculum** (the questioner diverges at Stage-A step 1→2: gap 0.006→0.067→…→0.328 by step 6, compounding). If kernel-iso ≈ 0.630, that's confirmed and we focus on *why the questioner's first update diverges*.

---

## 2026-06-25 — iter1 code is CLEAN (repro 0.554≈0.565); from-base iter1 redo launched; reproducibility footguns found (kernel patch, diversity bank, timing-split reward batching, adaptive batch)

**Goal:** decide whether the iter1→iter2 U-shape is real learning or expected plateau. First validate the code reproduces iter1. Established the iter1→iter2 regression is **not a code/reward change** — it's inputs.

**iter1 solver step-1 repro (CODE IS CLEAN):** re-ran iter1's Stage-C step 1 with current code on iter1's exact inputs (base Qwen3.5-4B + `rzcev_it1_verified_verl08.parquet` 108 rows, batch 64, 4096, n=5). Step-1 reward **0.554 vs original SOLVER2.log 0.565** (rlen 3034 vs 2967, clip 0.45 vs 0.44) — match within vLLM sampling noise. ⇒ current code faithfully reproduces iter1; the iter1→iter2 drop is in the **inputs** (questioner lineage verified→clip80, question difficulty, batch 64→256), not code. (Had to swap sdpa+remove_padding=false → flash+remove_padding=true to avoid an FSDP-offload assertion + `mma.h` kernel crash on the current env — generation/reward unaffected since vLLM does generation.)

**Budget test (it2_8k, cancelled after step 1):** input solver (verified-20) on iter2 questions at 8192 gave step-1 acc **0.521** vs 0.32 at 4096 — confirms the *measurement* gap (difficulty_dist 0.50 vs Stage-C 0.32) was the 4096 cap suppressing long-reasoning rollouts. BUT budget is a **constant** across iter1 & iter2 (both trained Stage-C at 4096 — verified in `iteration_rzero.sh.bak_preopt`), so budget does NOT explain the regression. Eval is greedy @8192 (all arms), fmt_rate=1.0 everywhere → the iter1→iter2 eval drop is pure acc|fmt (reasoning). Solver gets terser each iter (8k mean tokens base 3627→it1 2785→it2 2616); it1's shortening helped, it2's hurt (U-shape).

**Dead clip from-base runs:** clip80 and clip65 BOTH died in **Stage B band-eval** (clip80 stopped ~06:00 after Stage A — its questioner was reused by cont; clip65 vLLM engine died 21:09, hung 2h, killed this session). Neither produced a solver or eval. Only the original **rzcev iter1** (solver `rzcev_it1_verified_s2`, eval 0.630) has real iter1 solver data.

**cont iter2-4 run is BROKEN (flagged):** `iter23_continue.sh` had `WANDB_API_KEY: unbound` (wandb OFF), HF upload failing (token perms), per-iter eval used the n=200 first-200 bug (reports 0.325 not full-675 0.545), and **the solver FROZE at iter2** (iter3 Stage-C OOM → iter3/iter4 reused `cont_it2_verified_s`). Live iter4 was improving only the questioner against a stale iter2 solver.

**iter1 REDO launched (tmux `sp_it1redo`, `/data/selfplay_it1redo`):** from base, verified VW=0.75 DW=0.5, MAX_ITER=1, faithful knobs (Q_STEPS=6 S_STEPS=20 NUM_SAMPLES=1000 EVAL_N=9 4096), wandb on (.netrc), rollout+val dumps, full-675 Oly eval. **First from-base run to clear Stage B band-eval** (the `/data` move killed the disk failure mode). Cleared Stage A (questioner 6/6, ~15-18min/step) and Stage B → **196 band-passed rows**.

**REPRODUCIBILITY FOOTGUNS (the redo is NOT bit-faithful to rzc, on 3 axes — none are experiment knobs):**
1. **Questioner curve diverges**: step-1 reward identical (−0.344 vs −0.350) but rzc ramps to **+0.587** by step 6 while redo only **+0.259**; redo entropy much higher (0.39→0.32 vs 0.61→0.47). Hyperparams/qprompts(md5)/dtype/enable_thinking all **byte-identical**. Causes: (a) **kernel patch** — rzc ran `sdpa`+`use_remove_padding=False`, redo runs `flash_attention_2`+`True` (changes training-forward numerics → entropy/KL/grads; old path now CRASHES on the upgraded env so can't revert easily); (b) **env drift** vllm→0.23.0/transformers 5.12.1; (c) **diversity bank cold-start** — launcher `rm -f diversity_mem.npy`, redo starts vendi=1.0. `remove_padding` alone shouldn't move learning 55% (DPW correct) — the entropy gap is likely partly a metric-normalization artifact; the real outcome signal is the reward curve, most plausibly diversity-bank + stochastic solver scoring. rzc's `challenger_batches.md` is DELETED → can't do direct component comparison.
2. **Timing-split reward batching** (`caller_reward_manager_verl08.py`): challenger reward is "irreducibly BATCH" — coalesces a step's questions via a **wall-clock debounce** (`REWARD_COALESCE_DEBOUNCE=0.8s`) then computes diversity as nearest-neighbor similarity across that flush. Redo splits the 240 (60×4) questions into ~220 + ~16 stragglers each step (`challenger_batches.md`: judged 220|17, 225|15, 224|16) → stragglers get diversity over ~16 neighbors → inflated. Wall-clock debounce makes questioner reward **non-deterministic w.r.t. timing/env speed**. Fix: make flush count-based, seed/persist the diversity bank.
3. **Adaptive solver batch**: redo's 196 rows → adaptive logic picks **batch 128/mini 64** vs rzc's 108 rows → **64/32**. So Stage C trains on ~2× data at 2× batch.

**RESULT (2026-06-25, decisive): iter1 redo = OlympiadBench 0.474 — did NOT reproduce, and is BELOW BASE (0.573).** `RESULT_it1redo_8k n=675 format_rate=1.0 acc|fmt=0.474 pass=0.474 mean 7117ch trunc=3` — clean reasoning number, same harness. vs base 0.573 / rzc iter1 0.630 / iter2 0.545. The self-play iteration **made the solver worse than base.**

**Solver training reward DECLINED during the run** (not plateaued): step1 0.246 → peak 0.287 (step7) → **0.174 (step20)**. GRPO on this curriculum actively degraded the model. The redo's 196 band-passed questions were far harder/less-learnable than rzc's 108 (solver step-1 reward 0.246 vs rzc 0.565; clip_ratio 0.68 vs 0.44). Both passed the identical self-consistency band [0.3,0.8] — but the redo's are low-correctness (the band-bug §8.1: self-consistency ≠ correctness).

**CONCLUSION — the whole investigation lands here:** iter1's 0.630 was **NOT robustly reproducible**. Same code (proven by the 0.554≈0.565 step-1 repro), same config (byte-identical hyperparams/prompts), but the questioner is non-deterministic (env/kernel drift §9.9 + diversity-bank cold-start + wall-clock timing-split reward batching §9.10 + stochastic n=10 solver scoring), so it produced a *different, harder* question distribution. The self-consistency band has **no correctness guardrail**, so the harder set flowed through and training degraded the solver below base. **The original iter1 (0.630) was substantially a FAVORABLE DRAW** — a questioner run that happened to land a learnable curriculum. The U-shape (iter1↑/iter2↓) is the same phenomenon across iterations: the loop is not reliably beneficial; a bad questioner draw regresses below base.

**Two concrete fixes for robustness (not yet applied):** (1) make the questioner deterministic — seed+persist `diversity_mem.npy`, change the reward flush from wall-clock debounce to **count-based**, pin kernel+env; (2) gate the solver curriculum on **correctness vs the program label** (`correctness_band.py`/CVBAND), not self-consistency — so a hard-questioner draw can't hand the solver an unlearnable set. Also still TODO: fix the cont run's wandb/HF/eval-n200/frozen-solver bugs (§9.11) before trusting any of its numbers.

**ADDENDUM (2026-06-25, later) — WashU diff CONFIRMS kernel is the ONLY substantive change; but the old kernel CANNOT run on Brev's upgraded env.** Diffed the WashU iteration_rzero.sh (`/home/compute/jiaxinh/R-Zero/scripts/`, dated **Jun 21 12:56 — hours before rzc iter1**) vs current Brev: the only training-math difference is **`sdpa+use_remove_padding=false` → `flash_attention_2+use_remove_padding=true`**. Everything else benign (logger, REWARD_COALESCE_DEBOUNCE 0.8→2.0, N_SVC plumbing, FSDP_OFFLOAD-as-var default-unchanged, CVBAND additive/off). **Tried to isolate the kernel** by re-running iter1's Stage-C inputs with sdpa+remove_padding=false: FSDP-offload-ON → `offload_fsdp_model_to_cpu` AssertionError; offload-OFF + gpu_mem 0.5 → CUDA OOM (padded path is memory-heavy); offload-OFF + gpu_mem 0.3 → **TileLang `mma.h` kernel compile crash** (`tilelang/.../src/tl_templates/cuda/instruction/mma.h`). The Qwen3.5 hybrid (GDN/mamba) `remove_padding=false` path won't compile on CUDA13/vllm-0.23. **So the 0.630-era kernel can't be tested on Brev without rebuilding the old env** (WashU has it but a known verl-training bug; would need fixing). NOTE: the kernel's causal role is *indirect* anyway — 0.474 was driven by questioner *variance* → harder questions → self-consistency band passing them; kernel is one variance source, the band-bug is the amplifier. **DECISION: relaunched the robust-fix run (cbfix: CVBAND=1 correctness band + debounce 2.0, from base, verified, MAX_ITER=2, /data/selfplay_cbfix, tmux sp_cbfix) — it addresses the actual failure mode (bad draw → unlearnable curriculum) and is runnable on Brev. Watching: does the solver reward RISE (vs the nofix run's decline to 0.17) and the eval beat base 0.573 / approach 0.630.** Pending: cbfix iter1 eval (watcher bm6530ky5).

**CBFIX ITER1 RESULT (2026-06-25): correctness-band fix WORKED on the failure mode — recovered +0.071, but still below base.** `RESULT_cbfix_it1_8k n=675 acc|fmt=0.545 fmt=1.0 trunc=0`. vs nofix 0.474 / base 0.573 / rzc 0.630. CVBAND filtered 196→**73 rows** (the genuinely-learnable solve-rate-in-(0.2,0.8) subset). **Solver reward ROSE 0.40→0.56** (peak step10) instead of the nofix run's collapse to 0.17; clip dropped 0.46→0.20. So the correctness guardrail **eliminated the below-base degradation** (no reward collapse, +0.071 eval recovery) — but one iteration still **did NOT beat base** (0.545 < 0.573). Conclusion strengthened: the band-bug fix makes self-play *safe* (prevents the unlearnable-curriculum collapse) but not *beneficial* on a strong Qwen3.5 base — a single iteration from a variance-affected questioner lands ~0.545 regardless. Now running iter2 (watcher bb78frwqv) to see if a 2nd guarded iteration climbs or stays flat.

**CBFIX ITER2 RESULT (2026-06-25) — guardrail correctly REFUSED an unlearnable curriculum; loop STALLED, not regressed.** iter2 `CVBAND: in=65 matched=65 kept=0` → `B done: 0 solver-training rows` → `C: skipped` → solver stayed = cbfix iter1 (`newS=cbfix_it1_verified_s` unchanged) → iter2 eval = iter1 eval = **0.545**. The iter2 questioner (trained vs the stronger iter1 solver) produced 65 band-passed questions but **NONE had solve-rate-vs-label in (0.2,0.8)** — all too-easy (>0.8) or too-hard (<0.2) for the iter1 solver. So the correctness guardrail did its job (no training on garbage → no degradation), but the loop made **no progress**.

**FINAL TWO-ITER CONCLUSION (the whole investigation lands here):** nofix iter1 0.474 (collapse, below base) → cbfix iter1 0.545 (safe, recovered, still below base) → cbfix iter2 0.545 (0 learnable Qs, stalled). The correctness-band fix is a **successful SAFETY mechanism** (eliminates the unlearnable-curriculum collapse; refuses bad curricula) but makes the loop **safe, not beneficial**. On a strong Qwen3.5 base: (a) one self-play iteration doesn't beat base even with a clean curriculum, and (b) as the solver strengthens, the questioner can't hit the learnable competence-frontier band → the loop stalls (kept=0). **The original iter1 0.630 was a favorable draw.** The real bottleneck is NOT the band mechanics — it's **questioner curriculum quality on a strong base** (can't reliably generate learnable-and-beneficial problems). The kernel (sole code change from 0.630) couldn't be tested on Brev (TileLang mma.h) and is at most an indirect variance source. **Next levers (user's call): (1) loosen CVBAND band (e.g. (0.1,0.9)) so iter2 isn't starved; (2) variance study — re-run iter1 ×N to quantify the draw spread; (3) rebuild old env to isolate the kernel; (4) accept self-play caps at ~base on this strong base and pivot the questioner objective.** cbfix chain DONE 14:31 UTC; GPUs free.

---

## 2026-06-24 — clip65 (clipped+verified) LAUNCHED to /data by the watchdog loop

The mission-control `/loop` (assigned 06-23 ~07:25 to launch the clipped+verified run when the Brev lane frees) completed its mission after ~36 iterations / ~18h. iter234 (the verified chain 2→3→4) held the lane all night, then went through a volatile crash/relaunch window (iter4 reward-svc ConnectionError ~02:04, then idle→busy→idle oscillation while the parallel Claude recovered it). The loop correctly **did NOT** grab the lane during those brief/contested gaps. At ~20:45 the lane went **stably idle** (GPUs 0 MiB, recent run exited; only 52-day-old ray zombies left).

**Launched** (tmux `clip65`): `STORAGE_PATH=/data/clip65 FSDP_OFFLOAD=true S_GPU_MEM=0.55 bash ~/rzero_run/chain_clip65.sh` → `R-Zero/scripts/run_rzero.sh` (ARM=verified, CLIP_EASY=0.65, EXP=clip65, MAX_ITER=1, VW=0.75, DW=0.4; NUM_SAMPLES=1000 EVAL_N=9 Q_STEPS=6 S_STEPS=20).
- **Writes to `/data`** — verified `run_rzero.sh`/`iteration_rzero.sh` honor `STORAGE_PATH=${STORAGE_PATH:-$HOME/rzero_run}`, so the override sends all models/generated_question/artifacts/logs to `/data/clip65` (4.9TB free). Root is only 499G; was 94% full, and chain cleanup freed it to 83%. (See CLAUDE.md Brev `/data` storage rule, added this session.)
- **Optimized config inherited** — `iteration_rzero.sh` bakes in `attn_implementation=flash_attention_2` + `use_remove_padding=true` + `ppo_max_token_len=12288` + `param/optimizer_offload`. No `fla` installed (it deadlocks the reward service).
- **VERIFIED advancing**: questioner generated 996/1000 prompts at ~4800 tok/s in / ~2600 tok/s out (batched — not the old concurrency-1). The startup `/hello?name=None` 500 was a one-time probe, NOT the recurring reward-svc crash that hit iter234's iter4.
- Loop **STOPPED** (mission complete). Earlier in the run, a mid-run disk emergency (root 97%) was resolved by moving stale `~/rzero_storage`→`/data` + symlink.

---

## 2026-06-24 (later) — Band fix A/B: REFUTED at the outcome level. Self-play peaks at iter1; 2nd iter degrades regardless of band.

Isolated A/B: retrained the iter1 solver on the **correctness-variance-filtered** iter2 questions (553 rows, solve-rate-vs-label in (0.2,0.8)) vs broken iter2 (710 rows, self-consistency band). Same input solver, same questions, only the filter differs.

| OlympiadBench full 675 @8k | pass |
|---|---|
| base Qwen3.5-4B | 0.573 |
| iter1 (verified-20) | **0.630** |
| iter2 broken band (710) | 0.545 |
| **iter2 fixed cvband (553)** | **0.550** |

**The band fix barely moved the eval (0.545→0.550, within noise); did NOT recover toward iter1.** Training signal DID improve (reward trajectory ended 0.40 vs broken iter2's ~0.30; ~60% mixed-gradient batch vs ~45% dead) — so the fix improved performance *on the self-generated training set* but **did not transfer to the benchmark**. Conclusion: **the band was not the main problem.** A 2nd self-play iteration on a strong Qwen3.5 base degrades below base/iter1 **regardless of question filtering**. The self-generated curriculum isn't aligned with benchmark-improving signal. Real levers are upstream (questioner question quality/distribution, or cap at 1 iteration, or a different curriculum signal) — not a band patch. cvband solver ckpt needed `preprocessor_config.json` copied in for vLLM eval (verl text-only save omits it; now an ARCHITECTURE.md footgun).

---

## 2026-06-24 — OlympiadBench eval audited: the "0.415→0.325 regression" and the "0.63 vs 0.33" were BOTH measurement artifacts; real finding = self-play barely beats base, iter2 < base

**Cancelled all runs** (Brev iter4, Cornell clip80 977878, monitors) per DPW to re-establish the eval baseline. Re-ran OlympiadBench rigorously: **full 675** (not first-200), **all 4 GPUs** data-parallel (`run_oly.sh` + `eval_oly_shard.py`, 4 shards/GPUs 0-3), greedy, grading both `final_answer[0]` and any-answer.

**Two bugs in the original `eval_compare.py`:** (1) `probs[:n]` with `--n 200` → only the **first 200 of 675** problems (unrepresentative, ~20 pts harder); (2) grades vs `final_answer[0]` only → mis-scores the 94/675 multi-answer problems (turned out not to matter here: any>first gained 0).

**Corrected matrix (full 675, 8k, same harness):**
| model | base gen | format_rate | pass |
|---|---|---|---|
| Qwen3.5-4B **base** | qwen3_5 | 1.00 | **0.573** |
| iter1 verified-20 | qwen3_5 | 1.00 | **0.630** (+0.057 over base) |
| iter2 cont | qwen3_5 | 1.00 | **0.545** (BELOW base) |
| R-Zero v5 (jinyuan) | **qwen3** | — | re-eval pending (expect ~0.33) |

**Resolution of the "0.63 can't be right vs external 0.33" suspicion:** NOT an eval bug. `grade_answer` is sane (4≠5, 3≠3.00001, rejects garbage); dataset is the standard 675-problem math OlympiadBench. The skew is the **base model generation** — verified-20 `config.json model_type=qwen3_5` (Qwen3.5-4B, base already 0.573) vs R-Zero v5 `model_type=qwen3` (Qwen3-4B, the external ~0.33). Cross-base comparison is invalid.

**The actual signal (within Qwen3.5, same harness):** self-play adds almost nothing — iter1 = base +5.7 pts, and **iter2 regresses to 0.545, BELOW the 0.573 base** → the training is net-harmful by iter2. This supersedes the earlier "iter1 0.415 → iter2 0.325" framing (both were first-200 numbers; the relative ~8.5-pt regression holds on full set: 0.630→0.545). Ties to the reward-starvation finding (iter2 solver solve_rate 0.336 on its own band-filtered training Qs; ~46% zero-advantage).

**CLOSED — calibration + budget + 2nd-benchmark all confirm:**
- R-Zero v5 (Qwen3, step_15) on our harness = **0.304** ✓ matches external ~0.33 → harness calibrated; step_20 ckpt was empty (0 safetensors), used step_15.
- 4k (training budget) matrix: base 0.502 / iter1 0.560 / iter2 0.533 — format_rate=1.0 at 4k too (zero truncation), so 8k didn't cause a format problem; budget only buys reasoning room (8k > 4k by ~5-7 pts for the verbose Qwen3.5 models, ~0 for concise Qwen3 v5). iter2 gains least from extra budget (it learned shorter responses).
- **MATH-500 robustness (full 500, 4k):** base 0.858 / iter1 **0.884** / iter2 0.864 → regression iter1>iter2 holds on a 2nd benchmark.

**FINAL VERDICT:** iter1 > iter2 on EVERY benchmark × budget (Oly 8k 0.630>0.545; Oly 4k 0.560>0.533; M500 0.884>0.864). Self-play peaks at iter1 (+2.6 to +5.8 over base) then **declines at iter2**, landing ~base. Not sustained improvement on the strong Qwen3.5 base. Eval was never the issue (calibrated). The fix lever = challenger/band → solver **correctness-variance vs program label** (solve-rate in (0.2,0.8)), inserted after `judge.py` in Stage B (have evaluate answer-dists + judge `verified_answer` to compute it). DESIGNED + launch-ready, NOT launched — strategic call (patch band vs rethink approach/base) escalated to DPW.

Eval artifacts on Brev `/data/selfplay/`: `oly_*_shard*.jsonl` (per-problem text+grade), `agg.py`, `run_oly.sh`/`eval_oly_shard.py` (4-GPU sharded full-675 OlympiadBench), `run_math500.sh`. All runs cancelled, GPUs idle.

---

## 2026-06-23 (later) — ROOT-CAUSE of the blind investigation: rollouts were never logged; fixed for iter3+

**The divergence investigation was crippled because the current run captured no rollout text.** `scripts/iteration_rzero.sh:87` hardcoded `trainer.logger=[console]` with no rollout dump, so for the whole cont run there was *zero* saved prompt/response/score text — the reward service stores only extracted answers, not generations. This silently overrides the standard-experiment-logging discipline (wandb + saved generations), which is why mining iter2's divergence meant reverse-engineering metrics from reward-service jsonl + parquet scraps instead of reading the actual rollouts.

**Fix applied to Brev `iteration_rzero.sh` (backup `.bak_logging`), at the iter2→iter3 boundary while the script was unopened (`fuser` empty; iter2 done rc=0, iter3 not started, launcher in eval phase) — so iter3+ captures everything:**
- Line 87: `trainer.logger=[console,wandb] trainer.log_val_generations=20`.
- Questioner launch (L125) + solver launch (L180): each appended `trainer.rollout_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_{q,s}` and `validation_data_dir=…_val`. Keyed by `${TAG}` (=`${EXP}_it${ITER}_${ARM}`) so questioner/solver/iters never collide.
- Verified non-no-op: verl `ray_trainer.py:1681-1683` → `_log_rollout_data` → `_dump_generations(inputs, outputs, gts, scores, reward_extra_infos, dump_path)` writes prompt+response+gt+score per step. wandb auth persisted in `~/.netrc` (has `api.wandb.ai`) → no prompt/hang. `bash -n` passes.
- These are pure logging side-channels: **no change to tokens/n/steps/reward/experiment.**

**Cornell clip80 (977443):** `MAX_ITER=1` (single iter) — no future iteration to patch and can't edit mid-iter1, so its iter1 rollout text is NOT captured this run. Acceptable: clip80's deliverable is the eval numbers (effect of `CLIP_EASY=0.80` on the band), not rollout mining. Cornell verl confirmed to support the same 3 keys; patch staged for any future multi-iter Cornell launch.

**CLAUDE.md:** added "Training launches MUST capture rollouts" section (verify `logger=[console,wandb]` + `rollout_data_dir` + `validation_data_dir`/`log_val_generations` on both Q and S invocations; note the mid-iteration edit hazard).

**CONFIRMED working:** Brev **iter3 started 22:23:50 UTC** (Q=cont_it2_verified_q/gs6, S=cont_it2_verified_s/gs20 — correctly chained from iter2-redo). `rollout_dumps/cont_it3_verified_q/1.jsonl` (548KB) **contains real prompt+response text** (`{"input":"system\nYou are an expert competition-math problem setter..."}`) → logging patch verified live. GPUs 0,1 at 100% (train), 2,3 hold DP reward-service replicas. iter2-redo eval reproduced the original: OlympiadBench **0.325** (orig 0.32), MATH-500 done.

**Cornell clip80 CRASH + fix (job 977443 → 977878):** old job silently crashed at questioner-train with `ImportError: FlashAttention2 ... doesn't seem to be installed` (the Brev A100 flash speed-patch was in the Cornell script, but `~/envs/rzero` has no flash_attn), then continued to the generate stage on the **untrained base questioner** → experiment invalid. Fix: scancel 977443; revert Cornell `iteration_rzero.sh` to `attn_implementation=sdpa` + `use_remove_padding=false` (exact-attention / pure-efficiency, experiment-equivalent; backup `.bak_flashfix`); also applied the logging patch + added `WANDB_API_KEY` to `rzc_env.sh` (Cornell had no wandb auth). Relaunched **977878** (PD, MAX_ITER=1). Cornell compute nodes have internet (HF upload works) so wandb won't hang.

**Overnight monitor (non-destructive):** `/tmp/overnight_monitor2.sh` (bg) — NEVER wipes a Brev checkpoint (the false-restart lesson); stall = all-logs-frozen >20min + GPU idle → ALERT only; auto-resubmits Cornell only if its job vanishes from queue (cap 3); fires on iteration-complete / new-solver-dump / cornell-done/crash. Replaces the destructive `/tmp/overnight_monitor.sh`.

**CORRECTED divergence analysis (data-driven, from training logs — supersedes the length/curtailment story):**
- iter1 solver (`SOLVER2.log`, produced `rzcev_it1_verified_s2`): reward `critic/score/mean` **0.57→0.79 (rising)**, length 2967→1995 (stable) → OlympiadBench **0.415**.
- iter2 solver (`cont_verified_it2.log`): reward **0.29→0.16→0.30 (never >0.30)**, length 3382→1725→2915 (U, recovers) → OlympiadBench **0.325**.
- **Length was a RED HERRING:** solver length recovered to ~2900/4096 and eval format_rate=1.0 — no truncation. The regression is genuine reasoning degradation, NOT curtailment. (I was wrong to lean on shortening.)
- **Real cause = reward starvation / curriculum runaway:** iter2 trained the solver on questions it got right only ~16–30% of the time (2.5–4× lower reward than iter1). Difference is the **input questioner**: iter1=base Qwen3.5-4B → solvable questions; iter2=clip80-lineage *trained* questioner (`clip80_it1_verified_q`) → GRPO-pushed toward harder/uncertain questions. The **self-consistency band (0.3–0.8) does NOT cap difficulty by correctness** — a confidently-but-consistently-WRONG question passes the band yet yields low reward. So difficulty ran away ("gone too far the other way", as DPW called it) and degraded general reasoning.
- **`enable_thinking=false` is NOT the cause:** confirmed present in iter1 too (`SOLVER2.log:567 apply_chat_template_kwargs:{enable_thinking:False}`) — it's a constant. The questioner generates short (~100-tok) blind problems with often-wrong self-`\boxed{}` answers (e.g. "n²+n prime, 1≤n≤100" → true answer 1, questioner said 7) in BOTH iters; the solver still reasons (2–3k-tok CoT in body, thinking-tags off). Note: `scripts/iteration_rzero.sh` is **not git-tracked** (local working file), so no commit history to diff iter1 vs iter2 — logs are the only record, and configs match.
- Core fix (next experiment): challenger/band should target **solver correctness-variance vs the label** (solve-rate strictly in (0,1)), keeping reward in iter1's learnable 0.5–0.8 zone. Optional/experiment-altering: `enable_thinking=true` for questioner so it verifies its own problems.

**Storage moved to /data (16 TB) + root-disk relief (Brev):** root `/dev/root` (484 G) spiked to 96% (Ray `file_system_monitor` warned "Object creation will fail if spilling required", 18 G free) during questioner ckpt-save + Ray spill. Fixes: (1) `rollout_dumps/` → `/data/selfplay/rollout_dumps` with symlink `~/rzero_run/rollout_dumps → /data/...` so the LIVE writer + all future iters store to /data transparently (per DPW "always store rollouts in /data"); (2) moved 2 idle checkpoints (`rzcev_it1_verified_s2`, `clip80_it1_verified_q`, ~34 G, 0 open handles) → `/data/selfplay/models_archive/` + symlinks (no loss). Root now 80% / 102 G free. `/data` = 16 T, 5 T free, shared (langlin 7.6 T, jinyuan/R-Zero-hedge 3.1 T — do not touch). Monitor `/tmp/overnight_monitor2.sh` now also alerts at root ≥93%. **TODO at iter3→iter4 boundary:** edit `iteration_rzero.sh` so `default_local_dir`/`rollout_data_dir` point directly at `/data` (model ckpts still write to root; ~34 G/iter, fine through iter3, relocate for iter4).

**INCIDENT (self-inflicted) — recon on GPU 0 killed iter3 Stage C solver training.** Ran a standalone iter2-vs-iter1 solver reconstruction on Brev **GPU 0** (in the live pipeline's lane). When iter3 reached **Stage C (solver-train, needs all 4 GPUs incl. GPU 0)**, vLLM failed: `Free memory on cuda:0 (27/79 GiB) < desired (43.59 GiB)` — my recon held ~51 GB. iter3 Stage C **crashed**, the launcher advanced to **iter4 with the STALE iter2 solver** (`S=cont_it2_verified_s` — iter3's solver update was lost; questioner did advance to `cont_it3_verified_q`). Fix: killed recon → GPU 0 freed → **iter4 recovered and is healthy** (Stage A, 0 errors, wandb run `tvb…`). **Net loss: iter3 solver checkpoint (`cont_it3_verified_s`) never produced.** **LESSON: never run a standalone job on a GPU the active pipeline needs** — GPU 0 was in-lane; the Stage A service AND Stage C training both use it. Recon results that DID survive: `recon_it2_solver.jsonl` (iter2 solver on its 710 training questions, **solve_rate 0.336** — i.e. iter2 solver gets only ~34% of its own band-filtered training questions right vs the program label, consistent with the reward-starvation story). iter1 recon was killed mid-run (16%).
- Recovery option (experiment-semantics call for DPW): re-run iter3 Stage C standalone (have all inputs: `cont_it3_verified_verl08.parquet` + `cont_it2` solver) to recover `cont_it3_verified_s`, OR accept the solver chain as iter2→iter4 (skipped one update). Needs 4 GPUs — run after iter4 or when Cornell frees.

**Solver rollouts NOW available (DISK) — answer to "do we have them / on wandb":**
- `/data/selfplay/solver_eval_dumps/cont_it3_verified_{0,1,2}.jsonl` (~240 MB) — the Stage-B evaluate dump (my evaluate.py patch): iter2 solver's FULL solutions (n=9) on ~9000 iter3-generated questions. **DISK only** (eval script, not a wandb run).
- `/data/selfplay/recon_it2_solver.jsonl` — iter2 solver on 710 iter2 training Qs, full text + solve_rate 0.336. **DISK only** (standalone inference).
- wandb has questioner curves (`px6nrabx` it3-q, `tvb…` it4-q) but **NO solver `train/generations`** yet — iter3 Stage C (which would have logged them) failed; iter4's Stage C will log them when it gets there.

Pending:
- [ ] DECISION: recover iter3 solver (re-run Stage C) vs accept iter2→iter4 chain.
- [ ] Watch iter4 → solver-train stage; confirm `rollout_dumps/cont_it4_verified_s/` populates (now on /data) + wandb `train/generations`, then finish divergence analysis on REAL solver text.
- [ ] At iter3→iter4 boundary: point model `default_local_dir` to /data in `iteration_rzero.sh` (root disk would tighten by iter4).
- [ ] Confirm Cornell 977878 clears questioner-train (the prior crash point) once it leaves the queue.
- [ ] Quantify useful-signal fraction over the 710 iter2 training rows (correct-answer-in-1..4-of-5 = real advantage vs 0/all-5 = none) — pure data mining, no GPU.
- [ ] iter3/iter4 evals as they complete (OlympiadBench + MATH-500).

---

## 2026-06-23 — iter2 (cont, Q=clip80-questioner) eval results + auto-restart false-positive incident

**iter2 eval (cont run, Q=clip80_it1_verified_q, S=verified-20, DP service, flash+remove_padding, no fla):**
- OlympiadBench: format_rate 0.995, **acc|fmt 0.322, pass 0.32** (n=200, k=1, 8192 tok) — **REGRESSED vs iter1 verified-20's 41.5%.**
- MATH-500: format_rate 1.0, **acc|fmt 0.845, pass 0.845** — high.
- Stage B: 388 in-band, 363/388 = 93.6% program-verifiable, 363 solver rows.
- Solver step (no fla): **~12.6 min/step** (755s) → ~3.5× over the 44-min baseline; **fla was barely adding anything** (its smoke first-step ~696s) — dropping fla cost ~nothing AND kept the reward service working. Realized cycle ~7h (from ~20h). The OlympiadBench regression is one data point with the clip-lineage questioner seed — revisit whether clip80-trained Q hurts generalization vs base/verified Q.

**INCIDENT — auto-restart false positive lost iter2's checkpoint (~4h recompute).** The overnight monitor's stall detector (idle GPU + main-log frozen, ~8min window) FALSE-tripped during iter2's *benchmark eval*: the eval runs on 1 GPU at intermittent util and writes to `oly_cont.log`/`m500_cont.log`/RUN log, NOT main `cont_verified_it2.log`, so the main log looked frozen. `restart_verified.sh` wiped `cont_it2_*` (resume-hang avoidance) and relaunched iter2 — destroying the just-completed iter2 solver (not yet HF-uploaded; eval ran before the upload step). **Eval numbers survived** (already on oly/m500 logs). Relaunched iter2 recovered and is healthy. **Fix:** stall = max-mtime across ALL pipeline logs (it2/it3/it4 + RUN + oly + m500) AND GPU idle AND ~20min window (5 polls); eval phases no longer false-trip. CLAUDE.md updated; Brev restart count in /tmp/mon_brev_rc.

## 2026-06-23 — Experiment watchdog: cheap-path validated end-to-end on a live run

Building an always-on watchdog so runs stay at max speed and "alive-but-broken" jobs get caught/killed/relaunched without hand-babysitting. Design + all cheap components validated today (full design in memory `experiment-watchdog`).

**Design (user directives):** local laptop loop kept awake with `caffeinate` (started detached, `pkill caffeinate` to stop); FULL autonomy — fix/relaunch AND **kill malfunctioning runs**, bias hard to act ("cost of not acting is larger"), call only when unresolvable. **No fixed action library** — the model gets agency. Models: **DeepSeek** for the frequent health-judge, GPT-5.5/Claude on escalation. Core job is **semantic health diagnosis, not crash detection** — the dominant failure is "a run keeps running while clearly not working" (the concurrency-1 70× defect, flat-reward/no-learning, degenerate output).

**Validated today:**
- **Alert channel** (`~/Documents/GitHub/persona-companion/alert.py`): places a Vapi call via a TRANSIENT announcer assistant (built-in gpt-4o-mini + the persona's Cartesia voice) — robust to the companion server/tunnel being down. Verified live: clean call, `assistant-said-end-call-phrase`, $0.01. Root cause it "never called" before = laptop asleep (nothing alive) + a Cloudflare 1010 block on urllib's default UA (fixed with a normal User-Agent). See memory `persona-companion-alerts`.
- **State collector** (read-only ssh): `squeue`/`tmux ls`/`nvidia-smi`/log-tail across Empire, Unity, Brev — works.
- **DeepSeek judge** (`deepseek-chat`, JSON out): correct on synthetic ("DEAD, throughput far below expected") AND on REAL Brev `iter234` state → HEALTHY 0.9, correctly read GPU1-3 idle as the normal generation phase (not a stall), service at ~5000 tok/s as near-ceiling, zombies as old; emitted watch-condition "step-0 past 40 min → re-check". 601 tokens (~$0.0002).

**Cluster snapshot (06-23 ~06:28):** Brev `iter234` healthy (R-Zero iter2-4 cont, Qwen3.5-4B, step 0/6 in gen+reward phase, reward svc 51% of 2250 prompts @ ~5000 tok/s, GPU0 100%); Empire IDLE (no jobs, stale May tmux — wasted capacity); Unity job 61050175 `rztraino` PENDING; old 36-day defunct VLLM zombies on Brev (harmless).

**Pending:**
- [ ] Package collect→judge→act into a standing `watchdog.py` + run registry (host/session/log/expectations) + act layer (kill/relaunch over ssh, escalate, `alert.py` when stuck) + `caffeinate`-wrapped loop.
- [ ] Per-run "what healthy looks like" expectations (most generic: progress, util-vs-ceiling, reward trend, output sanity).
- [ ] Decide watchdog code home (self-play repo vs standalone).

---

## 2026-06-23 — Speed optimization: solver step 44min→~7-12min; fla broke reward service (root-caused)

Optimized the solver training (the ~14h Stage-C elephant) WITHOUT changing the experiment (user constraint: no token/n/step/reward changes; near-equivalent kernels OK; 15h/iter unacceptable).

**Solver step (batch 256×n5, 4096 max): 44min → ~7-12min.** Breakdown of the win (verl `timing_s`):
- `use_remove_padding=true` + `attn_implementation=flash_attention_2` (in `iteration_rzero.sh` COMMON_TRAIN): packing eliminates padding-token compute → ~1.8× (44→~24min). flash-attn 2.8.3 installed.
- `flash-linear-attention` (fla) + `causal-conv1d` kernels for Qwen3.5's 24 linear-attn layers (else torch fallback): another ~2× → `update_actor` 206s vs clip80's ~2560s = **~12×** on the update; first step 696s (incl one-time Triton JIT), steady ~7min.

**BUG (root-caused): fla deadlocks the standalone reward service** (`vllm_service_init/start_vllm_server.py`). With fla installed, the questioner Stage-A reward service hung at `Processed prompts 0/N, 0.00 toks/s`, GPUs idle, whole questioner frozen at step 0/6. Isolated by reverting offload (still hung → not offload) leaving fla as the only change; the service ran fine pre-fla. **Fix: `pip uninstall flash-linear-attention causal-conv1d`.** Service then healthy at **~3,800 toks/s** (batched — also showed the earlier "concurrency-1 / 104 toks/s" was a transient, not a standing problem → the planned A/B reward-batching refactor is likely unnecessary). Trade-off: solver keeps flash+`remove_padding` (~1.8×, ~24min/step) but loses fla's extra ~2×. fla-for-training-only (env-isolated from the service) is a future lever.

**Other config gotchas hit (all in `iteration_rzero.sh`/launcher):** rollout `enforce_eager=false` → CUDA-graph capture fails on A100(cap 8.0)+Qwen3.5+TP2 (`cancelled`) → keep `true`. grad-ckpt off + `ppo_max_token_len=32768` → OOM offload-off → keep grad-ckpt on, `ppo_max_token_len=12288`. **Final working config:** offload-ON (offload-off hangs the questioner handoff), `S_GPU_MEM=0.55`, flash+`remove_padding`, no fla, grad-ckpt on, `ppo_max_token_len=12288`, `enforce_eager=true`. iter234 relaunched 06:09, service healthy. Projected cycle ~4.5-8h (solver-dominated), down from ~20h.

**CLAUDE.md updated** with: "Move fast" (time is binding), "Fast debug loop" (tiny smokes, don't reload vLLM per-change), and "Monitoring cadence + auto-diagnose" (one loop over ALL runs, 60-90s in danger zones, hourly heartbeat when stable, trip-on-STALL = idle GPUs + frozen log ~3min, auto-relaunch). Lesson: a launched job ≠ working; verify progress within ~2-3min; lost ~3h once assuming a smoke trained when it crashed on a deleted parquet.

## 2026-06-22 — PERF: rollout runs at concurrency≈1 (~70× too slow); theoretical optimal cycle ~30min

**Headline:** the self-play loop is generation-bound and running **~70–300× below hardware capability**. Root-caused, benchmarked, and planned the fix.

**Diagnosis (Brev A100-80GB):**
- verl per-step timing: `gen ~1,820s` vs `update_actor ~80s` → generation is ~95% of every step; FSDP offload is irrelevant (update is tiny). Earlier "~2h update" was the OLD big-batch (256) attempt; current small-batch questioner update is ~80s.
- Realized throughput ~24–104 tok/s = **single-stream** (reward service logged `output: 104 toks/s`). KV cache had **184× concurrency headroom unused** (1.5M tokens, max_num_seqs 1024) → not batch/cache limited.
- GPU util by stage: standalone vLLM (`evaluate.py`) + raw benchmark = 100% util; **verl agent-loop training rollout = ~25% util**. So the verl/AgentLoop rollout submits generation at ~concurrency-1, not the hardware.
- **Localized:** the slowness is the **questioner stage (A) reward path** — solver-scoring of generated questions (n=10 difficulty + verifier). The **solver stage (C) batches fine** (clip80 solver: 1,110 prompts/59s, 100% util) because its reward is a correctness check (no generation).

**Benchmark (raw vLLM, GPU0, Qwen3.5-4B, this exact model):**
- concurrency 1 → 109 tok/s; 64 → 4,944; 256 → 7,769 (512-tok out).
- With 2048-tok outputs: conc 256 → 7,034; **512 → 7,305 (peak)**; 1024 → 6,881. → **per-GPU ceiling ~7–7.8k tok/s, plateaus by conc ~256** (decode is memory-bandwidth-bound). DP scales ~linearly: 3 GPU ≈ 21k, 4 GPU ≈ 28k.

**Reward-service internals (`vllm_service_init/start_vllm_server.py`):** solutions = `model.generate(questions, n=10)` (line 179); verifier = SEPARATE `model.generate(code_prompts, n=1)` (lines 289–305). Two sequential calls. `enforce_eager=True` on the service (CUDA graphs off). With `VERIFY_SUBSAMPLE=1.0` both passes cover all questions → can be merged. Verifier Python tool (`question_evaluate/verify.py:144`): `subprocess.run(["python3", tmpfile], timeout=10)` — fresh cold interpreter per program (thread-parallel, no pool).

**Token budget per iteration ≈ 50M decode tokens** (Stage A ~11M, B ~21M, C ~18M); **~70% is the solver scoring questions** (n=10 + n=9). **Theoretical optimal cycle (4×A100, batched+async): ~30 min bf16; ~15–20 min FP8; ~8–10 min if n=10→5.** vs current ~9–10 h → **~15–20× available.** Eval is negligible (~30s at ceiling); updates hide under gen if async; Python tool negligible if pooled. Binding floor = memory-bandwidth-bound decode of those ~50M tokens.

**Fix plan (ordered):** (1) batch the rollout/reward (conc 1→256) — most of the gap; (2) merge solutions+verifier into one `generate()` w/ per-request SamplingParams; (3) `enforce_eager=False` on the service; (4) `N_SERVICES=3` DP across all GPUs (needs offload-during-gen to free training GPUs); (5) async rollout/train overlap; (6) FP8 inference (~2× decode); (7) persistent Python sandbox pool; (8) design lever n=10→5. Validate offline on a free GPU BEFORE baking into a run.

**Decision (user):** let **iter2 (Brev) + clip80 (Cornell) finish** their current iteration and bank results (iter2 = 1,075 in-band / 98.4% verifiable), then **STOP before iter3** (watcher kills tmux iter234 after iter2 HF upload). Unity double-clip stays queued (patch batching in before it starts). Then rebuild the optimized pipeline. Benchmark scripts: `/tmp/vllm_bench.py`, `/tmp/vllm_bench2.py` (Brev).

---

## 2026-06-22 — Launch recovery: iter2-4 continuation (cont) + clip80; Stage-A resume-hang root cause

Brought up the two pending runs the user wanted live before sleeping: **iter2-4 verified continuation** (seeded from verified-20 = `rzcev_it1_verified_s2/global_step_20`, OlympiadBench 41.5%) and **clip80** (CLIP_EASY=0.80).

**Brev iter2-4 hang — ROOT CAUSE = stale-checkpoint resume artifact (not offload, not AgentLoop deadlock).** Two prior relaunches of `iter23_continue.sh` hung in **Stage A** (questioner GRPO). Log evidence: `Found checkpoint .../cont_it2_verified_q/global_step_6` → `Setting global step to 6` → `Resuming from ... global_step_6` → froze (GPU0 100%, no `Processed prompts`, then process exited). An *earlier* attempt had already completed Stage A (step 6) **and** Stage B (the `cont_it2_verified_verl08.parquet` training file existed), so the pipeline works end-to-end on Brev — but on relaunch verl auto-resumes the completed `global_step_6` (= max steps) and deadlocks instead of advancing. My earlier offload-on/off theory was wrong (it hung under both).
- **Fix:** `rm -rf models/cont_it2_* generated_question/cont_it2_* artifacts/cont_it2_*` (clear the stale checkpoint), relaunch fresh. Verified clean start 17:48 UTC: Stage A launching, `"Resuming from"` count = **0**. Cornell clip80 confirms fresh Stage A works (actively generating at step 0/6). Config: tmux `iter234`, FSDP_OFFLOAD=true S_GPU_MEM=0.55 (reliable-but-slow ~48min/solver-step on A100; offload-off was never the issue).

**Routing decision — Brev over Cornell for iter2-4.** Built `~/iter234_cornell.sbatch` (isolated `STORAGE_PATH=$HOME/rzero_run_cont234` so it can't clobber clip80's shared Vendi bank / temp_results; seeds solver directly from HF `Dylan1631/selfplay-verified-qwen35-4b-iter1-step20`; `--qos=cornell` since `standard` QOS caps at 48h). Submitted (977282) but est. start **2026-06-24** (cornell partition saturated, free GPUs reserved for higher-priority partitions). **Cancelled it** — 2-day wait + it would double-push to the same `selfplay-verified-qwen35-4b-iter{2,3,4}` HF repos as the Brev run. Brev (idle, ours) runs iter2-4 instead; sbatch kept on Cornell as fallback if Brev fails again.

**clip80 on Cornell (job 977116, alphagpu10):** healthy, Stage A questioner step 0/6, reward service generating 860 completions (slow first step, GPU0 100%). FSDP_OFFLOAD=false S_GPU_MEM=0.45. Uploads to `Dylan1631/selfplay-clip80-qwen35-4b-iter1`.

**iter2 progress (Brev):** A→B handoff CLEARED at 21:08 (the historic hang point) — Stage A done (questioner global_step_6), generate produced 998 questions, now band-scoring (3× evaluate.py) → filter → label → Stage C solver. iter2 in-band (in-training reward, 238q): **27.7%** ([0.3,0.8]); 72.3% trivial (≥0.9), 0% too-hard; mean_verified −0.56.

**Pending:**
- [x] Brev iter2 cleared smoke (questioner step1) + all 6 steps + A→B generate handoff.
- [ ] Verify Brev iter2 Stage C solver learns → eval → iter3/iter4. (watcher running)
- [ ] Verify clip80 clears band-filter → label → solver → evals (Cornell 977116, ~4h in).
- [ ] Collect iter2/3/4 + clip80 olympiad+math500 results; confirm HF uploads.

## 2026-06-22 — Double-clip challenger-reward experiment (Unity, queued)

New reward-shaping experiment on the challenger, patched into Unity's `examples/reward_function/caller_penalty.py` (env-gated behind `DOUBLE_CLIP=1`, compiles clean, inert for other runs):
1. **Double clip (difficulty):** replace continuous `min(score,1−score)` with a flat band indicator — in-band `[BAND_LO=0.30, BAND_HI=0.80]` → constant `BAND_FLAT=0.5`; outside (too easy OR too hard) → 0. Band matches data filter (`MINSC/MAXSC=0.3/0.8`).
2. **Stop-update difficulty:** if batch in-band fraction > `BAND_GATE=0.80`, zero the difficulty term for the whole batch.
3. **Stop-update verifiability:** if batch verifiable fraction > `VERIF_GATE=0.90`, zero the verifiability term — but do NOT flatten verified values (only 0..3/3 resolution). "Verifiable" = graded `verified > VERIF_OK` (0 ⇒ ≥2/3 program agreement).
Batch md header now logs `inband=.. [GATED] verif=.. [GATED]`. Seeds from base Qwen3.5-4B (sibling to clip80), iters 1–5, per-iter olympiad+math500 evals + HF push `selfplay-doubleclip-qwen35-4b-iter{N}`. Patcher at `/tmp/patch_double_clip.py` (Unity), sbatch `/work/pi_general_dartmouth_edu/dylan/unity_double_clip.sbatch`.

**Compute:** Unity job **61033801**, `-p gpu --gres=gpu:a100:4 --qos=long -t 7-00:00:00 --mem=400G`. The old "Unity too small" failure was a 64G `--mem` cap on the prior hold (`60905723`), not the hardware — uri-gpu001 has 515G RAM; requested 400G. Released the idle 64G hold to free fairshare. All Unity A100 nodes currently full → Slurm est start ~2026-07-01; **user chose to wait in queue** rather than preempt/shrink/run-on-Brev.

---

## 2026-06-22 — Verified arm co-evolution over 3 iterations (rzrun): metrics + generation funnel

Reconstructed the full per-iteration trajectory of the 3-iteration verified run (`rzrun`, Qwen3.5-4B, 06-19/20). Difficulty + diversity recomputed from `artifacts/rzrun_it{1,2,3}_verified/challenger_batches.md` (iter1 filtered to post-23:27 batches to drop ~22 stale appends from an earlier run that polluted the same file); verifiability from the B-stage judge logs; well-posed via GPT-5.5 on the band-filtered judge sets.

| iter | total gen | in-band | band yield | verified (consensus) | solver rows | difficulty (mean consistency, ↓=harder) | diversity (vendi, ↑) | verifiability | well-posed (GPT-5.5) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 188 | 15 | 8.0% | 9 | 9 | 0.839 | 18.5 | 60% (9/15) | n/a* |
| 2 | 178 | 12 | 6.7% | 11 | 11 | 0.874 | 33.3 | 92% (11/12) | 67% (8/12) |
| 3 | 190 | 26 | 13.7% | 15 | 15 | 0.631 | 42.8 | 58% (15/26) | 69% (18/26) |

\*iter1 judge.jsonl + verified.parquet were purged by disk hygiene; only 160-char-truncated questions survive in challenger_batches.md → well-posed not recomputable.

**Trajectories:** diversity ↑↑ (18.5→33.3→42.8, reaching ~50% of MATH-500 golden 85.6); difficulty → harder (consistency 0.84→0.63, in-band yield 8%→14%, solver rows 9→15); well-posed ~stable 67–69%; **verifiability dips hardest at iter3 (58%)**.

**Key findings:**
- **Co-evolution works**: across iterations the challenger makes questions harder AND more diverse, with ~constant total generation (~185/iter) but ~2× the in-band/usable yield by iter3.
- **Verifiability is the bottleneck**: the 3-program verifier confirms fewer of the harder iter3 questions (58%); GPT-5.5 says 69% well-posed but only ~38% well-posed AND program-verifiable → good hard questions get discarded. Fix: stronger verifier / GPT-5.5 fallback when programs return None.
- **Vendi reward worse than BLEU rep_penalty for diversity**: the vendi-reward run (`rzc`, 1 iter) collapsed to vendi 13.9 (~14 modes / 1661 q) — *below* every rzrun iteration — because marginal-novelty-vs-accumulating-bank decays to 0 as the bank fills. rep_penalty (this rzrun) kept diversity rising. Consider reverting to rep_penalty or rewarding absolute (not marginal) diversity.
- **Difficulty regime is the model, not batch**: Qwen3-4B-Base produces hard questions (band 66%, consistency 0.3, matches R-Zero); Qwen3.5-4B is too strong (band ~19%, consistency ~0.9). Confirmed at step 1 (pre-training), so not a batch-size confound.
- **Disk purge cost data 3×** (iter1 checkpoints, iter1 judge file, intermediate solvers) → motivated standing practice: always upload to HF (Dylan1631) + per-iter small evals + wandb (entity dylanpwilson2005-dartmouth). See memory.

---

## 2026-06-22 — OlympiadBench (harder, calibrated): verified step-20 = 41.5%, +8.5pp over base, +7pp over majority

MATH-500 was too easy (base 0.79). Re-evaluated on OlympiadBench (zwhe99/simplerl-OlympiadBench, n=200, k=1, max_tokens=8192, format_rate=1.0 everywhere → no truncation, real reasoning). Base = **33.0%**, matching the predicted ~33% → eval harness calibrated.

| arm | acc \| fmt | vs base |
|---|---|---|
| base (Qwen3.5-4B) | 0.330 | — |
| majority (vanilla R-Zero) | 0.345 | +1.5pp |
| verified step-8 | 0.350 | +2.0pp |
| **verified step-20** | **0.415** | **+8.5pp** |

**Verified step-20 is the standout: +8.5pp over base, +7pp over majority** (SE≈3.5pp at n=200 → ~2.4σ vs base, likely significant; majority/verified-8 gains within noise). Step-8→20 (0.35→0.415) shows more solver training compounds. Program-consensus labeling beats majority-vote much more clearly on hard problems than on MATH-500 (where it was +3.8pp). Notable: despite easy self-generated questions (band 19%), well-posed+correctly-labeled training still transfers to hard competition math. Eval logs: Brev `/home/nvidia/rzero_run/oly_{v8,v20,maj}.log`, `eval_olympiad.json`. Caveat: k=1, n=200 — a paired/larger-k pass would firm significance.

**Infra:** the CUDA fix (NVCC bypass + lib symlinks + ninja PATH) was ALSO needed on Brev for the 8192-token olympiad shape (fresh flashinfer compile; m500's 4096 ran off warm cache masking the latent skew). Now applied uniformly Brev/Unity/Cornell.

---

## 2026-06-21 — CUDA fix: making the frozen Brev env (vLLM/Qwen3.5) compile on fresh HPC nodes (Unity + Cornell)

Rebuilt envs from `brev_freeze.txt` on Unity & Cornell could `import torch` + `torch.cuda.is_available()` but **vLLM-Qwen3.5 failed at engine init** — the Qwen3.5 hybrid (Gated-DeltaNet) + flashinfer kernels JIT-compile at load and broke. Three distinct causes, all because the fresh nodes have **no system CUDA** (only the pip `cu13` libs, which lack dev symlinks) and a **version-skewed frozen dep set**:

1. **`ninja` not found** (Unity) → `ninja` is in `envs/rzero/bin`; that wasn't on PATH. Fix: prepend `envs/rzero/bin` to PATH.
2. **"CUDA compiler and CUDA toolkit headers are incompatible"** (both) → frozen set has `nvidia-cuda-nvcc 13.2.78` but `nvidia-cuda-runtime 13.0.96` (CUDART 13.0). flashinfer's bundled CCCL (`cuda/std/__cccl/cuda_toolkit.h:41`) hard-errors on nvcc≠CTK. Fix: `export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"` (documented escape hatch; same major 13, safe). Also `CUDA_HOME=.../nvidia/cu13` with `cu13/bin` on PATH so nvcc=13.2 is used.
3. **Linker `cannot find -lcudart` / `-lcuda`** (both) → pip ships `libcudart.so.13` (no unversioned `.so`), no `lib64/`, no `libcuda` stub. Fix (symlinks in `.../nvidia/cu13/`): `ln -sfn lib lib64`; `ln -sf libcudart.so.13 lib/libcudart.so`; `ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 lib/libcuda.so` (driver lib on the GPU node).

**Validated:** Unity `VLLM_GEN_OK` (17×23→391 correct). Brev "worked" only because its flashinfer kernels were pre-cached (mismatch latent). Env files: Unity `/work/pi_general_dartmouth_edu/dylan/unity_env.sh`, Cornell `/mnt/home/ch2263/rzc_env.sh` (both updated). **Both HPC clusters now usable for training + eval.**

---

## 2026-06-21 — GPT-5.5 audit of verified-arm well-posedness: labeling is reliable, degeneracy is the real leak

Adjudicated all **102** program-consensus ("verified", majority_agrees=True) questions from `rzcev_it1_verified_judge.jsonl` with **GPT-5.5** (blind: judged well-posedness + solved independently, no program answer shown), `/tmp/gpt55_judge.py` → `/tmp/gpt55_wellposed.json`.

| metric | result |
|---|---|
| well-posed (unambiguous, single answer) | 96/102 = **94%** |
| labeling correct (consensus == GPT-5.5 answer) | 98/102 = **96%** |
| degenerate (well-posed but vacuous → 0) | 28/102 = **27%** |
| well-posed AND non-degenerate (useful) | 69/102 = **68%** |

**Program-consensus labeling is RELIABLE — earlier mislabeling suspicion was wrong.** Cases like `['64','0','0']`/`['9908','0','0']` (I'd flagged as buggy-programs-outvoting-correct) are confirmed by GPT-5.5 to be genuinely 0 (e.g. "no factorial has exactly 5 trailing zeros; v5 jumps 4→6 at 25" → empty set). The `64`/`9908` were the buggy programs; majority-vote MIN_AGREE=2 correctly filtered them. All 4 GPT-vs-program disagreements are cases GPT judged NOT well-posed (returned NONE) — zero cases of program getting a real answer wrong.

**The real leak is DEGENERACY (27%), a verifiability reward-hack.** Vacuous questions (impossible constraints → answer 0) are trivially program-checkable, so they earn the verifiability reward without requiring reasoning. Examples (GPT reasons): "any four integers contain two of same parity → even difference → empty"; "V={0..78}, total 3081 odd → no equal partition → 0"; "three distinct digits ⇒ product 0 but digit-sum positive → empty". Plus ~6% not-well-posed = leaked meta-text ("Wait, re-reading the definition…" — the model's thinking bleeding into the question) or genuine ambiguity (conflicting bounds, ambiguous quantification).

**Implication:** verifiability machinery genuinely works (94% well-posed, 96% correct labels) — the hard part (4B self-generating verifiable questions) is solved. Combined with the flat-difficulty finding, the fix set for the next run is: (1) **non-degeneracy gate** (require non-empty solution set / reject trivial-0), (2) **strip leaked meta-text**, (3) **difficulty-weight fix** (full-scale uncertainty + max(0) floor). Tooling: `gpt-5.5` via OpenAI API (key in `.env`), needs certifi SSL context in detached procs.

---

## 2026-06-21 — Difficulty curve + challenger reward audit: we diverge from the R-Zero PAPER (uncertainty half-scale, no max(0) floor)

**Difficulty reward policy (majority/replication arm, `caller_penalty.py` @ commit 6935d2e):**
```
uncertainty = min(score, 1-score)            # score = solver self-consistency ∈[0,1]; tent peaks 0.5; -1 if no question
penalty     = cluster_share_per_problem(...)  # BLEU-cluster share |C_k|/B ∈[0,1]
final_score = uncertainty - penalty + verif   # verif=0 for majority; NO max(0,·)
```

**Difficulty curve (per GRPO step, mean solver self-consistency `score`; from `artifacts/rzrun_it{1,2}_majority/challenger_batches.md`, 256 questions each):**
- iter1: mean score **0.886**, mean uncertainty ~0.02, band(0.3–0.8) ~10–25%, easy(≥0.9) 74–100%/batch
- iter2: mean score **0.788**, mean uncertainty ~0.00, band ~10–36%
- Reward curve (critic/score/mean) it1: −1.08 → −0.48 → … → +0.15. The climb is almost entirely from cutting repetition/format-fails, NOT from raising difficulty. **Difficulty essentially flatlined at "too easy" (~0.8–0.9, far from the 0.5 target).**

**Divergence from the R-Zero PAPER** (arxiv 2508.05004; verified against upstream `caller.py` + paper):
- Paper: `r_uncertainty = 1 − 2|p̂−½|` (max **1.0**); `r_rep = λ|C_k|/B`, λ=1; `r = max(0, r_uncertainty − r_rep)`.
- Ours: uncertainty = `min(s,1−s)` (max **0.5** = HALF the paper); penalty = cluster-share λ=1 (matches paper); **no max(0,·) floor** (paper floors at 0).
- Upstream RELEASED `caller.py` is byte-identical to ours for uncertainty (`min(s,1−s)`) and has **no penalty at all** — i.e. R-Zero's own released code already contradicts their paper.
- Net: difficulty carries ~½ the weight vs the paper, and the (unfloored) penalty can drive reward negative and invert the signal → optimizer favors de-duplication over hardening. **Root cause of the flat difficulty curve, compounding the strong-base-model issue (Qwen3.5-4B solves challenger questions ~80–90% → uncertainty pinned near 0).**
- Anomaly to check: `challenger_batches.md` logs `rep_penalty` up to **1.50** though cluster-share ≤1.0 — 06-20 working tree may have had an extra penalty multiplier vs committed λ=1.

**They did NOT release the paper's runs** (checkpoints/datasets/logs go to user `STORAGE_PATH`); only code + reported metrics (+3.7 iter1, +6.49 after 3 iters on Qwen3-**4B-Base**). No reference artifacts to diff against.

**Fix for next run:** switch uncertainty to `1 − 2|p̂−½|`, restore `max(0, unc − penalty)`, confirm penalty λ=1 (cluster-share, not >1), and/or use a colder/weaker band-eval solver so uncertainty has range against the strong base.

---

## 2026-06-21 — ITER-1 EVAL IN HAND: verified (program-consensus) beats base AND majority on MATH-500 (significant)

The iter-1 comparison the run was designed to produce is complete and **positive for the verified arm.** Three MATH-500 evals (06-20 12:35, full test set n=500, k=1, max_tokens=2048, one RESULT each, both checkpoints valid HF models) on Brev `/home/nvidia/rzero_run`:

| Arm | model | format_rate | acc \| formatted | pass |
|---|---|---|---|---|
| BASE | Qwen/Qwen3.5-4B | 1.000 | 0.792 | 0.792 |
| MAJORITY (vanilla R-Zero) | majority_FINAL_solver | 0.998 | 0.808 | 0.806 |
| **VERIFIED (program-consensus)** | VERIFIED_FINAL_solver/global_step_8/actor/hf | 1.000 | **0.846** | **0.846** |

**format_rate ≈ 1.0 for all three including base** → no truncation on MATH-500 at 2048 tok, so the gain is real accuracy, NOT the format/termination artifact. Verified beats base **+5.4pp** acc and beats majority **+3.8pp** pass; majority's own gain over base (+1.5pp) is marginal (~1 SE).

**Paired significance (McNemar, n=200, same questions, `pp_base.jsonl`/`pp_verified.jsonl` 06-21 07:22):** base 0.780 → verified 0.845, delta **+6.5pp**, both_right=151, both_wrong=26, verified-only-right=18, base-only-right=5, discordant=23, **exact two-sided p=0.0106**. Verified fixes 18 / breaks 5 — a reasoning-improvement pattern, significant at p<0.05.

**Reconciliation with the training-side forensics entry below (same day).** That entry concluded the gradient fed mostly on the format/termination term and the questions were degenerate/easy — true of the *training signal* on it2/it3. But the held-out MATH-500 eval is the ground truth for "does the trained solver reason better," and at format-saturated conditions (base already 1.0 format_rate) it does, significantly. Not contradictory: the reward's format term dominated the *advantage spread*, yet the resulting solver still generalizes to +6.5pp correct reasoning on unseen problems. The verified arm's edge over majority (+3.8pp) is the cleanest signal that program-consensus labeling > majority-vote labeling at iter 1.

**Caveats:** k=1 (single sample/question — some decode variance); verified checkpoint is global_step_8, not the canonical 20. **In flight to strengthen:** (1) Brev `rzcev_it1_verified` re-run training the verified solver to step 20 with the rebalanced vendi/graded-verifiability rewards; (2) Cornell H100 env for a clean both-arm re-run (robust home, no Brev disk/lane traps); (3) majority arm re-run for parity (WashU eval 84230 abandoned — pinned to contended node a100s-2305, node-local checkpoint can't be relocated).

**FAILURE + FIX (06-21 ~19:35): step-20 verified rerun OOM'd at step 4/20.** `torch.OutOfMemoryError` in `ray_trainer._compute_old_log_prob → compute_log_prob → entropy = torch.logsumexp(logits)` — the LM-head logits over a packed micro-batch (`ppo_max_token_len_per_gpu=12288`) × 151k vocab. GPU 0 had 57.6GB in use (vLLM KV from `gpu_memory_utilization=0.7`), only 19.65GB free vs 19.90GB needed — failed by a hair on step 4 (first batch to pack to the full token budget). `save_freq=20` → nothing saved, fell back to base. **Root cause:** vLLM reservation (0.7) + log-prob entropy logits tensor (12288 tok) left no activation headroom — same logits-memory killer noted in the 2026-06-16 TRL entry. **Fix (proven):** `gpu_memory_utilization 0.7→0.55`, `ppo_max_token_len_per_gpu 12288→8192`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Relaunched cheaply** as solver-only on the *existing* 108-row parquet (`rzcev_it1_verified_verl08.parquet`) — skips the ~2hr questioner/generate/judge — via `~/rzero_run/rzc_solver2.sh` (tmux `rzc_sv2`, GPUs 0-3, → `models/rzcev_it1_verified_s2`, log `SOLVER2.log`). Same fix baked into the Cornell `rzc_full.sbatch` (would hit the identical OOM on H100-80GB).

---

## 2026-06-21 — Training-side forensics on the verified arm: difficulty / diversity / advantage

Pulled the actual trained artifacts off Brev (`/home/nvidia/rzero_run`) to answer "what difficulty/diversity were the questions, and what % of the learning signal did they carry?" Confirms the null/format-artifact conclusion from the *training* side, not just eval.

**Real trained counts** (the "26"/"35" were judge *candidates*, not the trained set; many judge rows abstained — e.g. `program_outputs=[None,'1024',None]`, only 1/3 programs answered, MIN_AGREE=2 → dropped):
- it2: 12 judge candidates → **11 questions trained**, 12 GRPO steps
- it3: 26 judge candidates → **15 questions trained**, 12 GRPO steps
- it1 verified parquet purged by keep-last-2 cleanup (12 steps in log)

**Difficulty — 2 coarse buckets, lopsided easy.** EVAL_N=4 means band 0.3–0.8 can only admit score 0.5 (2/4) or 0.75 (3/4). it2 `{0.5:2, 0.75:10}` = 83% easy; it3 `{0.5:7, 0.667:1, 0.75:18}` = 69% easy. 0.75 = base solver already gets it 3/4. No genuinely-hard items; difficulty has 2 levels, not a spectrum. Root cause: EVAL_N too small.

**Diversity — ~half degenerate.** Trivial answers (0/1/2/3): it2 6/11 (55%), it3 7/15 (47%); `"0"` alone = 6/15 (40%) in it3. it3 only 10/15 unique answers + one 579-digit degenerate answer. Topics cluster on integer/sum (narrow number-theory phrasing).

**Reward advantage / learning signal** (from `logs/rzrun_verified_it*.log`):
- last-step reward range = **[0.00, 0.90] every iteration** — floor 0.0 = format reward, ceiling never ~1.0. The gradient feeds on format/termination spread, NOT correct-vs-incorrect reasoning.
- score/mean ~-0.4 → ~+0.1 (the format term being learned), tiny updates: pg_loss 0.006–0.016, grad_norm ~2, adv spread (max-min) ~2.3–2.6 normalized.
- **solver `tool_calls/mean = 0.0` at every step** — solver makes zero python-tool calls in training (tool use is verifier-side).

**Conclusion:** ~100% of the run's gradient came from these 11–15 mostly-degenerate questions (nothing else existed), and ~all usable advantage traces to the format/termination term (reward pinned in [0,0.9], base already 3/4-correct), not reasoning. Training-side and eval-side agree: the model learned to format/terminate, not to reason. Reinforces that a trustworthy re-run needs EVAL_N≥10, a well-posedness/non-degeneracy gate, larger NUM_SAMPLES, and reporting acc|formatted.

---

## 2026-06-16 — TRL reference-check disambiguates "harness bug vs fundamental wall"

### Why
After ~15 configs of our custom `scripts/online_self_play_grpo.py` / `training_only_grpo.py` showing no held-set learning (flat or degrading; per-step reward "bouncing" e.g. m=0.422→0.301→0.320), the open question was: **is our custom `grpo_step` buggy, or is small-batch single-box GRPO on these questions fundamentally stuck?** The clean test is to run a reference implementation (TRL `GRPOTrainer`) on the *same* easy questions and compare.

### Infra wall finally cleared
TRL OOM'd repeatedly on one A100-80GB at `max_new_tokens` 3072 (needed ~78 GB — the LM-head logits over full-seq × 248k vocab in fp32 is the killer, ~48 GB alone). **Fix: halve sequence to `max_new_tokens: 1536`** (config `grpo_math/configs/train_trl_sanity.yaml`, k=4, prompts_per_step=4, lr 1e-5, LoRA r8, kl 0.01, 20 steps). Fits at ~50 GB, ~140 s/step. Running CUDA_VISIBLE_DEVICES=5, tmux `sp_trl`, log `outputs/slurm/trl_sanity.log`, wandb offline run `offline-run-20260616_103116-wps2janq`. Reward trajectory not in stdout (tqdm-only) — parse offline wandb datastore with `wandb.sdk.internal.datastore.DataStore`, history items use `nested_key` not `key`.

### Key finding (interim, run still in flight)
**TRL's per-step reward bounces identically to our custom harness:** step1 0.56 / 2 0.03 / 3 0.60 / 4 0.05 / 5 0.025 / 6 0.00 / 7 0.05. This strongly indicates the per-step "bouncing" is **NOT a bug in our `grpo_step`** — it is intrinsic to tiny-batch GRPO: with prompts_per_step=4 the per-step reward tracks *which 4 questions were drawn*, not learning. **The only valid learning signal is a fixed held-set before/after**, never the per-step reward curve. Reframes the whole prior negative.
- Confound for this run: `mean_completion_length` hits the 1536 cap on some steps (steps 2,7 = 1536 exactly → truncated → 0 reward → degenerate group, frac_reward_zero_std=1). Affects TRL and our harness identically (the GSM-hard easy set was calibrated at ~4096 tok). Eval (`scripts/measure_easy.py`) uses 4096 so the before/after measurement is not truncation-capped.

### VERDICT: WALL, not a harness bug
- **TRL run completed** (20 steps, train_runtime 2899 s, train_loss 0.07). Per-step reward over all 20 steps: **first-5 mean 0.253 ≈ last-5 mean 0.252 — flat**, same noisy bounce as our custom harness.
- **Merge clean**: `scripts/merge_lora_adapter.py` remapped 496 text LoRA keys, dropped 220 vision keys, hard-failed on none ⇒ adapter provably applied (not the silent no-op).
- **vLLM can't serve the merged model**: merged config saved as text-only `Qwen3_5TextConfig`, but vLLM's Qwen3.5 path demands the multimodal `Qwen3_5Config` wrapper (same multimodal/text split that caused the merge bug, now at serve time). Worked around with a direct HF-`generate` eval (`scripts/hf_measure_easy.py`, k=8, temp 1.1, top_p 0.98, max_new 4096, strict last-FINAL_ANSWER) — provably uses the trained weights, no vLLM.
- **Fixed before/after, same strict harness, GPUs 5 & 7 parallel**: base **0.175** → TRL-merged **0.163** (per-Q base [.25,.25,.12,0,0,0,0,0,.12,1.0] vs merged [.12,.25,0,0,0,0,0,0,.25,1.0]). **Null / hair lower — TRL the reference impl learns nothing here either.**
- **Two robust conclusions:** (1) the per-step reward "bounce" is intrinsic to tiny-batch GRPO (TRL bounces identically), NOT a bug in our `grpo_step`; (2) no-learning reproduces under the gold-standard reference implementation ⇒ the wall is the **setup**, not our code.
- **Why the wall (root cause):** under strict HF-generate scoring these "easy" questions are near-zero-signal — **6/10 are 0/8 for BOTH base and trained** (all-zero groups ⇒ advantage 0 ⇒ no gradient), only 1 question is reliably solved (8/8), ~3 sit at 1–2/8. The earlier "0.637 base / mid-band" calibration was harness-specific (OSP's forced-answer continuation inflates pass vs strict). Combined with the 1536-token truncation forced by single-A100-80GB memory (LM-head fp32 logits over 248k vocab), the trainable signal is too thin to move LoRA. This is a compute/signal wall, not an implementation defect.

### CORRECTION — the "wall" was signal starvation from bad question selection (fixable)
Re-reading the null: 6/10 of the easy_gsmhard set were 0/8 for BOTH base and trained ⇒ zero-variance groups ⇒ zero GRPO advantage ⇒ no gradient at all. And those questions fail on **large-number arithmetic precision** (8–10 digit answers like 6277334000), not reasoning — a target RL on a few questions can't move. The earlier 0.637 "mid-band" calibration was an artifact of OSP's lenient forced-answer-continuation scorer; under the strict scorer the set is pathological. So the prior negative was partly self-inflicted by question selection, NOT proof learning is impossible here.

### Mid-band selection experiment (in flight)
- **New tool `scripts/select_midband.py`**: scans a dataset, measures base pass@8 under the STRICT HF-generate scorer (the real training/eval scorer), shardable across GPUs. Scanned 80 GSM-hard questions at max_new 2048 (2 shards, GPU5+7). **Pass distribution: {0:31, 1:7, 2:5, 3:7, 4:3, 5:2, 6:4, 7:6, 8:15}** — 39% are 0/8 (arithmetic-precision-hard), 19% saturated 8/8, **34/80 (42%) signal-bearing (1–7/8)**. Confirms ~1/3 of GSM-hard is genuinely trainable once you measure under the right scorer.
- Built `outputs/midband_train.json` (25 Q, pass dist {1:6,2:4,3:4,4:3,5:1,6:3,7:4}, base mean ≈0.445) + `outputs/midband_heldout.json` (9 Q). JSONL for TRL: `outputs/midband_train.jsonl`.
- **TRL run launched** (`grpo_math/configs/train_trl_midband.yaml`, jsonl=midband_train, max_new 2048 to match selection budget, k=4, prompts_per_step 4, lr 1e-5, LoRA r8, 40 steps, GPU5, tmux `sp_trlmb`). Fits at 74.6 GB, ~216 s/step, ETA ~2.5 h. **Early reward (steps 1–7): 0.07/0.57/0.38/0.85/0.13/0.87/0.37 with `frac_reward_zero_std=0` on every step** — every group now carries gradient variance (vs sanity run's degenerate all-zero groups), completion lengths 640–870 (finishing, not truncating). Selection fixed the starvation.
- **RESULT (before/after, strict HF-generate, k=8, 2048, temp 1.1):**
  - train (in-dist, n=25): base **0.550** → trained **0.475** (−7.5 pts; per-Q deltas broadly negative, sum −0.075/Q, ~18/25 down-or-flat — NOT a single-Q fluke, ~2σ decline)
  - heldout (generalization, n=9): base **0.514** → trained **0.569** (+5.5 pts, but n=9 → ~1σ, within noise)
  - **Verdict: NOT learning.** Mild in-distribution degradation, heldout flat. Fixing signal starvation was necessary but NOT sufficient — TRL still degrades in-dist even with healthy reward (0.5–0.87, zero_std=0 every step). Same "training degrades" pattern as the custom harness.

### CORRECTION (high-n) — the −7.5pt decline was VARIANCE; true result is FLAT, and the policy barely moved
- k=8 was too noisy to trust. Re-ran at **k=32** via a NEW fast vLLM path (see below):
  - train (in-dist, n=800): base **0.4925** [0.458–0.527] vs trained **0.4863** [0.452–0.521] → **−0.6 pts, CIs almost fully overlap**
  - heldout (gen, n=288): base **0.5243** [0.467–0.582] vs trained **0.5382** [0.481–0.596] → **+1.4 pts, CIs overlap**
- **Verdict: FLAT / null — no significant change either direction.** The earlier −7.5/+5.5 were k=8 sampling noise (user correctly called it).
- **Root cause of flat = policy barely moved.** Logprob check (temp 0, first 8 tokens) base vs merged: shifts of ~0.005 nats (e.g. 'Thinking' −0.0559→−0.0494). lr 1e-5 + kl_beta 0.01 + LoRA r8 over 40 steps = a tiny KL-anchored update. Not "GRPO degrades" and not "wrong questions" — the update was too gentle to change pass rate. **Next lever: crank lr (→3-5e-5), drop/zero KL, raise rank (r16-32), more steps — actually move the policy, then re-measure at k=32.**

### ✅ FIRST POSITIVE RESULT — aggressive update produces measurable, GENERALIZING gains
Diagnosis (gentle update → policy barely moved) confirmed by fixing it. Aggressive config `grpo_math/configs/train_trl_midband_aggr.yaml`: **lr 4e-5 (4×), kl_beta 0.001 (10× lower), LoRA r16/α32 (2× rank)**, same 25 midband Q, 40 steps, GPU7, 2.1h.
- **Training reward CLIMBED**: first-5 mean 0.550 → last-5 **0.714** (gentle run was flat 0.55→0.55). KL grew step5 0.0005 → step20 0.0051 → step40 ~0.003-0.014 (~10× the gentle run) — policy moved, stably (no divergence).
- **Measured pass@32 (fast vLLM path), base vs aggressive-trained:**
  - train (in-dist, n=800): 0.4925 → **0.5550** (+6.3 pts; per-Q 18 up / 6 down / 1 flat)
  - heldout (generalization, n=288): 0.5243 → **0.5972** (+7.3 pts; per-Q **7 up / 0 down / 2 flat — ALL non-negative**)
- **Heldout gain ≥ train gain ⇒ genuine generalization, NOT memorization.** The all-non-negative heldout pattern is the cleanest signal.
- Significance caveat: vllm_measure CIs assume independent samples (n=800/288) but samples cluster in 25/9 questions → true CIs wider, per-split ~1.5-2σ (borderline). Strength is the CONSISTENCY: both splits up together + heldout monotone + train-reward up + gentle-run flat control. Real positive, not bulletproof.
- **Recipe that worked = (1) mid-band question selection under the strict scorer [kills signal starvation] + (2) update aggressive enough to move the policy [lr 4e-5, KL 0.001, r16].** Earlier flat/null was under-powered update, NOT a wall.
- Next to harden: more heldout questions (n=9 → 40+), a seed-repeat, and push further (lr/steps/rank) to see how far pass climbs before overfit (train≫heldout) appears.
- Artifacts: `outputs/trl_midband_aggr/{checkpoint-40,merged_full}`, served `mb_aggr` on :8002.

### ⚠️ CRITICAL CAVEAT — the gains are FORMAT/conciseness, NOT reasoning (decompose every eval!)
User asked "is the gain even net of formatting?" → decomposed pass into format_rate × (acc|formatted). On mid-band GSM-hard heldout, base vs bb2:
| budget | model | format_rate | pass | acc\|formatted |
|---|---|---|---|---|
| 1024 | base | 0.153 | 0.153 | 1.000 |
| 1024 | bb2 | 0.250 | 0.250 | 1.000 |
| 2048 | base | 0.493 | 0.479 | 0.972 |
| 2048 | bb2 | 0.618 | 0.597 | 0.966 |
- **The ENTIRE pass gain is format_rate** (@2048: +12.5pts format) while **acc|formatted is FLAT (~0.97, even slightly down)**. The model did NOT get better at math — it learned to FINISH/emit the answer within budget instead of rambling past the token cap. When base commits, it's already ~97% right on these Qs.
- **Root flaw exposed:** "mid-band" (pass 2-6/8) selected questions whose variance is TRUNCATION-driven, not reasoning-driven (these are length-hard, not reasoning-hard — GSM-hard = big-number arithmetic → long CoT → truncation). GRPO optimized the only available signal = conciseness. The benchmark gains (GSM8K +2.6, OOD +3.4, measured @1024) are almost certainly the same format effect.
- **So we have NOT demonstrated reasoning learning.** We demonstrated answer-emission/conciseness learning (real & generalizing, but not math).
- **FIX (methodology, also added to CLAUDE.md):** (1) ALWAYS report format_rate / acc|formatted / pass separately + the max_tokens budget; (2) use **acc|formatted as the reasoning metric**; (3) re-select training questions by acc|formatted band (REASONING-hard = model finishes but is WRONG, acc|formatted in ~0.3-0.7), NOT pass band; (4) eval at a budget high enough that format_rate isn't the bottleneck (or train the model to be concise as an explicit, separate goal). Tool: `/tmp/fmt_decomp.py` (format/acc decomposition).

### ✅✅✅ INDEPENDENT-BENCHMARK VALIDATION — gains generalize off-distribution (NOT memorization)
mb_bb (big-batch model, trained on 25 mid-band GSM-hard Qs) vs base on INDEPENDENT benchmarks, k=8, max 1024:
| benchmark | base | mb_bb | delta |
|---|---|---|---|
| GSM8K (different dataset, n=800) | 0.7063 | 0.7325 | **+2.6 pts** |
| GSM-hard disjoint (idx 1000-1110, never seen, unfiltered difficulty, n=880) | 0.1693 | 0.2034 | **+3.4 pts (+20% relative)** |
- Combined with mid-band heldout (+10.8), the model improves on (1) same-dist heldout, (2) a DIFFERENT dataset (GSM8K), (3) harder unfiltered OOD (disjoint GSM-hard). **Learning is real and generalizes.** CIs overlap modestly (real-but-modest), consistent direction with the larger heldout effect. Benchmark sets: `outputs/bench_gsm8k.json`, `outputs/bench_gsmhard.json`.

### ✅✅ BIG-BATCH RESULT — best generalization yet, confirms batch is the lever
k=32 before/after, three configs (base measured once, reused):
| split | base | aggressive (batch~1, lr4e-5, r16) | **big-batch (16-prompt, lr2e-5, r16, vLLM)** |
|---|---|---|---|
| train (in-dist, n=800) | 0.4925 | 0.5550 | 0.5200 |
| heldout (gen, n=288) | 0.5243 | 0.5972 | **0.6319 (+10.8 pts)** |
- Big-batch heldout CI [0.576–0.688] vs base [0.467–0.582] — **barely overlap, ~significant**; per-Q heldout ALL non-decreasing (8/9 up). Training reward climbed smoothly 0.369→0.562 (correct), KL 0→0.006.
- **Shape confirms batch theory:** big-batch has LOWER train (0.520) but HIGHER heldout (0.632) than aggressive tiny-batch (train 0.555 / heldout 0.597) ⇒ bigger batch = less overfit = better generalization. Exactly R-Zero's lever.
- **WORKING RECIPE: (1) mid-band selection under strict scorer + (2) vLLM-backed training (speed) + (3) bigger batch, moderate lr.** Served mb_bb on :8003.
- Next: scan C (gsmhard 80-200, GPU6) expands the question bank → enables grad_accum 32-64 (true R-Zero-scale batch) → expect even better heldout if trend holds.

### ✅ BREAKTHROUGH — vLLM-backed GRPO training works (20× faster gen, real batch feasible)
Was the whole time using HF `model.generate()` in-process for TRAINING rollouts (slow ~190s/4-completions) — NOT vLLM (vLLM was only for eval). That's why big batch was ~12h. Fix: wired TRL's `use_vllm` into `train_grpo_trl.py` (added `use_vllm`/`vllm_mode`/`vllm_gpu_memory_utilization`/`vllm_max_model_length` from config into GRPOConfig; `.bak` not needed, edited locally + scp).
- **Colocate smoke (GPU7, 3 steps): WORKS.** ~37s/step for 16 completions (~20× faster than HF), no OOM, **weight sync verified correct** (`sampling_logp_difference` ~0.013 → vLLM and training policy agree; NOT the silent-staleness LoRA-hotswap bug). Reward climbed 0.26→0.54→0.61 in 3 steps.
- **OOM tuning on single 80GB GPU (colocate = 2 model copies + KV + fp32 logits):** narrow window. `vllm_gpu_memory_utilization` must be ≥~0.25 (model is 0.225 of 80GB) AND leave room for training. 0.2 → "No available memory for cache blocks" (vLLM can't fit model). 0.3 + max_new 2048 → OOM (74GB+). 0.3 + 1536 → OOM (just barely, needed ~80.4). **Working config: vllm_util 0.3 + max_new 1024** (smoke-proven; mid-band finishers terminate in ~130–215 tokens so 1024 captures real answers). Also must set `vllm_max_model_length` (default pulls Qwen3.5's full 262144 → KV cache too big).
- **Dataset-size constraint:** TRL generation_batch = per_device(4) × grad_accum. grad_accum 32 → needs 32 prompts but mid-band train = 25 → "not a single sample, stopping at step 0". grad_accum 16 (16 prompts ≤ 25) works. **For bigger batch need a bigger question bank** (scanning gsmhard 80-200 on GPU6 now to expand).
- **RUNNING (GPU7, tmux sp_vbb, `train_trl_vllm_bb.yaml`):** vLLM colocate, grad_accum 16 (= 16 prompts/step, 16× our old effective ~1), lr 2e-5, kl 0.01, r16, max_new 1024, 30 steps, **~108s/step → ~54 min** (vs ~12h HF). Watcher armed → merge_full → serve → k=32 measure vs base on completion.

### Cross-project: R-Zero (hedge) config/dynamics comparison → batch is our missing lever
Studied R-Zero's verl GRPO (hedge project, ~/R-Zero on box + local) to extract learning-dynamics lessons. Same GRPO algo, very different scale:
- **R-Zero solver** (`scripts/solver_train_verified.sh` + `examples/config.yaml`): rollout_batch **256 prompts/step**, rollout.n **5** → **1280 rollouts/step**; global_batch 128; **lr 1e-6**; **FULL fine-tune (FSDP full-shard + param/optim offload)**; KL low_var_kl coef 1e-2 (as kl_loss); max_grad_norm 1.0; resp 4096; temp 1.0/top_p 0.99; max_steps 20; **4–8 GPUs, generation (vLLM services) SPLIT from training (FSDP)**.
- **Ours** (TRL): prompts_per_step **4**, k **4** → **16 rollouts/step**; lr 1e-5→4e-5; **LoRA r8–r16**; KL 0.01→0.001; single GPU (gen+train same device, HF generate); resp 1536–2048.
- **Dynamics**: R-Zero solver accuracy climbs MONOTONICALLY (0.387→0.407→0.421→0.441, grad_norm steady 0.24–0.84, kl~0). Ours BOUNCES (m 0.422→0.301→0.320). Same algo — the difference is **batch = 80× theirs** (1280 vs 16 rollouts/step). Our per-step reward just tracks which 4 prompts were drawn = the bouncing.
- **Recommendations (priority):** (1) **batch is THE lever** — simulate via grad_accum 32–64 (micro-batch 1) → effective 64–256 prompts/step, kills bouncing; (2) **stop cranking lr** — R-Zero's 1e-6 works because big batch → clean gradient; our 4e-5 on batch-4 = big steps in noisy directions; once batch is big, drop to ~1–2e-5; (3) **structural unlock = split gen from train** (TRL `use_vllm` server-mode across 4 GPUs, or FSDP) → enables 256-prompt × 4096-tok batches AND full-FT (higher ceiling than LoRA, why our policy "barely moved"); (4) KL ~1e-2 is fine — gentle-run failure was batch+lr, not KL.
- Our recent "aggressive" win (lr↑/rank↑/KL↓) was COMPENSATING for tiny batch. Bigger/cheaper/more-stable win = batch.

### Fast eval path FIXED (full multimodal merge → vLLM)
- `scripts/merge_lora_full.py`: loads base via `AutoModelForImageTextToText` (the multimodal `Qwen3_5ForConditionalGeneration` class), applies the RAW adapter (keys already target the `language_model.layers` multimodal tree → 716/716 matched, NO remap), `merge_and_unload`, saves full model. Config stays `model_type: qwen3_5` (multimodal) so **vLLM serves it** (text-only `AutoModelForCausalLM` merge was rejected). Output `outputs/trl_midband/merged_full` (18.8 GB incl. vision).
- Served on GPU5:8002 (`--served-model-name mb_full`). Verified APPLIED via logprobs (differ from base) — greedy tokens were identical only because the update is tiny (no argmax flips).
- `scripts/vllm_measure.py`: parallel HTTP pass@1, k samples/Q, strict scorer, prints 95% CI. ~10x faster than HF generate. base via :8001, trained via :8002.

### RETRACTED — "vLLM silently does NOT apply LoRA to Qwen3.5"
- Earlier I concluded the LoRA hot-swap was a silent no-op based on byte-identical GREEDY output. That test is too weak: this trained model's update is so small it produces identical greedy tokens even though logprobs DO shift (confirmed on the fully-merged model). So the hot-swap may well have been working. **The online-loop "root cause" claim is WITHDRAWN pending a logprob-level re-test of `push_adapter`.** Do not rely on the earlier no-op conclusion.

### (superseded) earlier note — vLLM silently does NOT apply LoRA to Qwen3.5 (likely the online-loop root cause)
- Tried to speed up eval via vLLM LoRA hot-swap (avoid HF generate). Loaded the step-40 adapter onto the running base server (port 8001, launched WITH `--enable-lora --max-loras 8`), both raw AND key-remapped (text keys `base_model.model.model.layers...`, vision dropped). **Greedy (temp 0) output was BYTE-IDENTICAL to base in both cases** ⇒ vLLM accepts the load call ("Success") but applies nothing.
- Cause: the adapter is `all-linear` over Qwen3.5's hybrid arch — keys dominated by `linear_attn.in_proj_a/b` (linear-attention projections) which vLLM's LoRA/punica kernels don't map; it silently skips. `load_lora_adapter` returns Success regardless.
- **Implication for the online loop**: `scripts/online_self_play_grpo.py::push_adapter()` hot-swaps the freshly-trained adapter onto vLLM for the next round's rollouts via this exact API. If it's a silent no-op, **every round generated from the BASE model** — the trained policy never fed back into rollouts ⇒ would fully explain the online loop's persistent no-learning, independent of all the GRPO tuning. NEEDS VERIFICATION on the online path (compare base vs hot-swapped greedy output there too).
- Speed fix options (none done yet): (a) serve a FULLY-merged multimodal model in vLLM — merge into the multimodal class so config stays `Qwen3_5Config` (current merge uses AutoModelForCausalLM → text-only config vLLM rejects); (b) keep HF generate but install flash-linear-attention/causal-conv1d + bigger batch. (a) is the real fix and also unblocks fast eval.
- Open cheap diagnostic: greedy (temp 0) base-vs-trained pass on the midband sets — distinguishes genuine no-learning from GRPO mode-collapse lowering sampled pass@k (train_loss hit 0.023 at step 40 = possible collapse).
- Artifacts: `outputs/trl_midband/merged` (merged weights), `outputs/trl_midband/adapter_remapped`, `outputs/midband_train.json|jsonl`, `outputs/midband_heldout.json`, `scripts/select_midband.py`, `scripts/hf_measure_easy.py` (now takes `<model> <questions.json> [k] [max_new]`).

### Decision / next steps
- ~~This arm is **paused pending dedicated multi-GPU capacity**~~ (superseded — retesting with proper mid-band selection first)
- (original note, still true for full-length hard questions) Dedicated multi-GPU capacity (FSDP/ZeRO or TRL+vLLM-served rollouts) that would allow full-length (≥3072-tok) generation AND a question set that is genuinely mid-band under the *training* scorer. On one contended box the reference impl confirms it can't learn.
- Lean on the **R-Zero hedge** (GPUs 0–3, untouched) as the active line.
- GPU state left clean: my eval/merge/vLLM-merged tmux sessions killed; `sp_vllmA` (GPU4) + `sp_vllmB` (GPU6) left up but idle (base Qwen3.5-9B servers) — free to reclaim. Merged weights + adapter at `outputs/trl_sanity/`.

---

## 2026-06-11 — Agentic in-context proposer + program-consensus code judge replace question-DPO; full pipeline launched

### TL;DR
Question-DPO (the run1 blocker, cross-cycle holdout 0.71 > ln2) is replaced by an **agentic proposer**: the model proposes questions in-context conditioned on measured `[pass n/8 | verifier_programs_agree k/3 | note]` feedback lines, no gradients. Judging is replaced by a **3-program code-judge consensus** (model writes 3 independent Python programs at temp 0.6, subprocess-executed, majority = reference; <2 agreement = discarded). Validated over 8 test rounds on the Brev box; full pipeline (gen → solver GRPO with deterministic correctness reward → merge → GSM8K eval) launched 2026-06-11 06:52 UTC in tmux `agentic`.

### Validation arc (all on box, vLLM Qwen/Qwen3.5-9B port 8001, tmux logs in outputs/slurm/)
- **Self-verify ceiling**: forcing the proposer to solve its own questions ⇒ all 8/8 (proposer==solver). Removed self-solve.
- **Feedback alone insufficient**: R4→R5 chain without hardness coaching stayed 8/8 and drifted easier. The explicit "propose beyond the routine" coaching paragraph is load-bearing.
- **LLM judge (temp 1.4) noisy both directions**: false positives (wrong answers 7,5,5 credited) and false negatives (correct answer 100 rejected 4/8). Code judge is deterministic and caught both.
- **R7→R8 chained test** (`scripts/test_agentic_final2.py`, log `outputs/slurm/agentic_final2.log`): R7 pass=[2,8,7,8,8,8,8,8] mean 7.1, 2/8 trainable; R8 (with R7 feedback) pass=[7,6,2,8,8,7,8,7] mean 6.6, **5/8 trainable (1–7 band)**, 0 discarded both rounds, programs 3/3 nearly everywhere, no zero-trick collapse. Feedback measurably moves difficulty toward the band.
- Goldmine class confirmed: program-trivial/hand-brutal questions (lattice x²+y²≤2025 → 1/8) are perfectly verifiable.

### Production pipeline (launched, tmux `agentic`, log `outputs/slurm/agentic_run.log`)
1. `scripts/agentic_question_gen.py` — 4 parallel chains × 8 rounds × 8 questions, seeded with 24 real measured R6–R8 feedback entries, per-chain topic focus, anti-zero-trick rule, stronger calibration coaching ("≥5 of 8 must be as hard as past ≤6/8 questions"). Writes `outputs/agentic_bank/bank.jsonl` (question, reference_answer, pass_count, trainable_for_solver = ref∧1≤pass≤7; dedup across chains).
2. Gate: abort if <40 trainable questions.
3. `grpo_math/configs/train_agentic_correctness_qwen35.yaml` — solver GRPO on GPU1, LoRA r8, k=8, max_new_tokens 4096, 30 steps, **reward.mode=correctness** vs code-judge references (no LLM teacher at all).
4. `scripts/merge_lora_adapter.py` merges final checkpoint → `merged_final`.
5. `eval_gsm8k` 500q k=4 on merged model. Baseline eval of untrained model running in parallel on GPU2 (tmux `baseeval`, log `outputs/slurm/baseeval_gsm8k.log`).

### Code changes (local + synced to box)
- `grpo_math/trl/train_grpo_trl.py` `_make_pairwise_jsonl_dataset`: now carries `reference_answer`→`answer_text` column so reward.mode=correctness works with agent banks (verified: bare-integer ground truth parses; non-trainable/duplicate rows filtered).
- New: `scripts/agentic_question_gen.py`, `scripts/merge_lora_adapter.py`, `scripts/brev_agentic_run.sh`, `grpo_math/configs/train_agentic_correctness_qwen35.yaml`.

### Confirmed bugs (production path, still unfixed)
- `{question_bank_examples}` placeholder left literally unfilled in cycle-1 proposer prompts (seen live in smoke data).
- `run_python_verification_probe` (grpo_math/self_play/verifier.py:239) is a hardcoded ~3-template pattern matcher, not real code execution.

### Run results (2026-06-11, Brev box)
- **Generation**: 284 questions, 124 trainable (44% yield), 3h04m, 4 chains × 8 rounds. Calibration improved within chains (C1: 3/8 → 5/8 trainable R1→R2, incl. an exact 4/8).
- **Training**: 30 steps × 191s = 1h35m on GPU1 after two OOM iterations (fp32 logits buffer batch×seq×248k-vocab: fixed via max_new_tokens 4096→2048 + prompts_per_step 8→4 with grad_accum 2 = one k=8 group/step). **Caveat: weak signal** — 62-100% of rollout completions truncated at 2048 tokens (bank difficulty was measured at 8192), reward_correct/mean 0 most steps, 0.125-0.25 occasionally. Only 30 prompts visited.
- **MERGE BUG (critical, recurring class)**: TRL saves Qwen3.5 LoRA keys against the full multimodal tree (`base_model.model.model.language_model.layers...` + 220 `model.visual` keys), but AutoModelForCausalLM exposes `model.layers...`. PeftModel.from_pretrained loads NOTHING (all keys "missing", warning only) → first "trained" eval was bit-identical to baseline (pass@1 0.4760 both — that's how it was caught). Fixed in `scripts/merge_lora_adapter.py`: remap text keys, drop vision keys, hard-fail if any key unmatched. Sanity check confirmed all 248 text lora_B matrices nonzero (vision B's all zero) — adapter did train.
- **Baseline GSM8K (Qwen3.5-9B base, reason-first prompt, temp 1.1, k=4, 500q, 1024 tok)**: mean_reward 0.4570, format_rate 0.6070, pass@1 0.4760, pass@4 0.8600. Low format rate = many runs exceed 1024 tokens or miss the tag.
- **Trained GSM8K (merged_final_v2, clean merge, same eval settings)**: mean_reward 0.4400, format_rate 0.6010, pass@1 0.4260, pass@4 0.8600. **Null result vs baseline** (pass@1 −5.0 pts, ≈2σ at prompt level; pass@4 identical): 30 steps of truncation-starved reward (correctness mostly 0) taught nothing on GSM8K, mild pass@1 dip. The PIPELINE is proven end-to-end; the training config is the bottleneck, not the data.

### Follow-up experiments (same day, after user review)
- **Truncation theory proven directly**: 12 trainable bank questions × 2 solver attempts — at max_tokens 2048: 2/24 finished (2 correct); at 8192: 19/24 finished, 15/24 correct (62%, matching the bank's calibrated band). The first training run graded the model on questions it could not finish 92% of the time.
- **Per-step GSM8K evals** (merged via fixed script, 500q k=4): base 0.4760/0.8600 (pass@1/pass@4) → step10 0.4660/0.8400 → step20 0.4780/0.8420 → step30 0.4260/0.8600. Pure noise, no trend — confirms null.
- **8k rerun LAUNCHED** (tmux `agentic8k`, wandb https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/rqmxrwap): max_new_tokens 8192 to MATCH the bank calibration budget (user directive), prompts_per_step 1 × grad_accum 8 (micro-batch 1 is the only shape that fits 8192-token backward on A100-80GB: fp32 logits = batch×seq×248k×4B), steps 90, save_every 15, wandb on (key from box .env). ~14.6 min/step → ETA ~21h (~2026-06-12 14:00 UTC). Eval queued after via driver.
- **Multi-GPU tested and rejected**: 4-rank accelerate OOMs at per-rank bsz 2 (backward); at bsz 1 it works but only ~1.25× faster (719s vs 902s/step) because each rank still generates sequentially at micro-batch 1 and the slowest rank gates the step. Not worth restarting the running job. Real speedup requires TRL vLLM-served rollouts — untested risk: TRL↔vLLM LoRA weight-sync may hit the same `language_model` key-prefix mismatch that broke the merge.
- GPU usage answer for the record: first run used 3/8 GPUs (vLLM=0, train=1, baseline eval=2); per-step evals used 3-4; rerun is single-GPU by the analysis above.

### Restart as 7-GPU run (2026-06-11 ~18:50 UTC, user directive: use all GPUs + eval each round + markdown log)
- Killed the 1-GPU 8k run at step ~12 (wandb rqmxrwap). Key realization: multi-GPU at micro-batch 1 doesn't speed up steps (~15 min either way) but multiplies DATA per step — 7 ranks × grad_accum 8 = 56 completions = **7 question groups/step vs 1**. My earlier "only 1.25× faster" framing compared fixed global batch; scaling batch with ranks is the right move.
- **Live run**: tmux `agentic8k7g`, wandb https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/335aod23, GPUs 1-7, 60 steps × ~15 min ≈ 15h (ETA ~2026-06-12 10:00 UTC), `WANDB_LOG_MODEL=checkpoint` so adapter checkpoints upload as wandb artifacts.
- Driver (`scripts/brev_agentic_train_eval.sh`) now: trains → **merges + GSM8K-evals EVERY checkpoint** (every 10 steps, 3-way parallel on freed GPUs) → writes a markdown report `outputs/agentic_run_report.md` (config summary, per-checkpoint eval table incl. base row, artifact pointers).
- Bank sample reviewed with user: compound-constraint questions (e.g. "x²+y²≤1600, x even, y odd, x+y div 5" → 62, pass 5/8). Trainable pass distribution: 8×1/8, 8×2/8, 7×3/8, 8×4/8, 14×5/8, 25×6/8, 54×7/8.

### Diversity problem found + feedback mechanism validated (2026-06-11 evening)
- **User spotted heavy repetition in the bank.** Confirmed by clustering: 30/124 trainable were lattice x²+y²≤N variants, 19 floor equations, 16 divisor-counts; 15 questions shared the same 8-word opening. Causes: (1) the 4 chains never saw each other, (2) feedback lines carried NO diversity signal, (3) calibration coaching pointed at proven templates (exploit>explore).
- **Rejected lazy fixes per user**: no template bans, no jaccard gating. Instead, pure-feedback redesign of `scripts/agentic_question_gen.py`: model-generated structure labels (one cheap call per question), structure frequency report in the prompt (counts + WHY diversity matters, nothing forbidden), per-question `structure: <label> (novel|N produced before)` in feedback lines, cross-chain shared history, `--history_bank` flag seeds a new generation with the prior cycle's bank (this is the cycle-to-cycle feedback path; previously unwired). Gotcha fixed: 120-line history + 24k proposer max_tokens overflowed the 32768 ctx → history capped at 80 lines, proposer 16k.
- **Divtest result (2 chains × 3 rounds on idle GPU0, seeded with all 284 cycle-1 questions): feedback works.** Novel structures 10/16 → 14/16 → 7/16; trainable 6 → 5 → 9; round-3 passes [0,0,1,1,2,3,5,5,5,6,7,8,8,8,8,8] — best-calibrated batch of the project (56% trainable vs cycle-1 44%). R3's novelty dip = re-use of R1/R2 structures with freshly measured pass rates (explore-then-calibrate), not collapse. New structures cycle 1 never touched: phi/sigma functions, digit products, Legendre/factorial valuations, binomial divisibility, semiprimes.
- **Question banks now on wandb** as browsable tables + artifacts: run `question-banks` https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/msdbgoh3 (cycle1_bank 284 rows, divtest_bank 48 rows). Local copies + human-readable questions.md on user's Desktop (~/Desktop/agentic_run_artifacts/). NOTE: training-run checkpoint artifacts appear on run 335aod23 only when checkpoints save (step 10, 20, ...).

### Tool-loop judge (user directive: judge must have python as a TOOL, not one-shot programs)
- vLLM restarted with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Qwen3.5 parser, vLLM 0.19.1) — native OpenAI-style tool_calls now work on port 8001. Probe confirmed (model emitted proper tool_call).
- New `scripts/python_tool_judge.py`: judge model gets a `python` tool (subprocess sandbox, 15s, stateless per call), iterates write→execute→see output→revise until `FINAL_REFERENCE: <int>` or `UNVERIFIABLE`; max 6 turns, ~2.3 tool calls/question avg.
- **Validation vs 3-program majority on 16 divtest questions: 14 agree, 1 resolved, 1 exposes ambiguity.** The resolved one: forbidden-points lattice path the old judge DISCARDED — tool judge got 12, hand-verified correct by inclusion-exclusion. The disagreement: permutation question ambiguous re 0/1-based indexing (23616 vs consistent 22320) — question-quality fault, not judge noise.
- Wired into `scripts/agentic_question_gen.py` as default (`--judge tool`: 2 independent runs temps 0.2/0.5, tiebreak 3rd run on disagreement; `--judge programs` = legacy). Smoke-tested: d(n)=d(n+1) → 63=known ref, runs agree. Prompt/output dumps also added (outputs/agentic_bank/prompts/C<i>_R<n>.txt).

### Architecture pivot: unified online loop + per-ARTIFACT independent judges (user directives, 2026-06-11 late)
- **Single rollout set** (user: "we should just do 1 set of rollouts"): new `scripts/online_self_play_grpo.py` — per step: proposer (in-context) → solver k=8 rollouts via vLLM with CURRENT LoRA adapter → judges AFTER → GRPO update on training GPU (repo's memory-safe logprob gather, micro-batch 1, KL via adapter-disable) → adapter hot-swapped into vLLM (`/v1/load_lora_adapter`, plain-text response!) → pass+agreement+structure feedback to proposer. Measurement and training are the SAME rollouts; calibration always matches the live solver and live token budget by construction.
- **Per-artifact independent judge rollouts** (user: reference-matching "empirically wrong" — e.g. ambiguous-indexing question 23616 vs 22320 where exact-match zeroes legitimate interpretations — and unscalable; future = "llm coding stuff, each judge checks the different repos"): NO precomputed reference. Each unique artifact gets 2 independent tool-loop judges (`judge_solution` in python_tool_judge.py: verify by independent computation, never trust candidate reasoning, judge within candidate's reasonable interpretation), 3rd on split or random ~34% subportion; majority = reward; mean inter-judge agreement per artifact = verifiability signal in proposer feedback (R_cons realized with execution-grounded judges). Artifact=claimed-answer dedup is a math-only optimization, no-op for repos.
- **Positioning**: this repo is the moonshot (open-ended artifacts, repo-checking judges); a parallel agent runs the hedge (R-Zero + verifiability).
- Smoke-validated incrementally: smoke1 (3 steps) — gradient flowed on a 2/8 group, KL=0 pre-first-update correct; smoke2 (full-detail dumps: proposer prompt 23k chars + raw thinking output, per-judge code+stdout traces, all rollout texts; adapter hot-swap verified IN-RUN, step-2 rollouts served by solver_step1); smoke3 (per-artifact judges) running. Full-detail report: ~/Desktop/agentic_run_artifacts/online_smoke_FULL_detail.md.
- Fixes en route: apply_chat_template returns BatchEncoding (tokenize via text); /v1/load_lora_adapter returns plain text not JSON (silent off-policy risk if push fails — now logged); vLLM relaunched with --enable-lora --max-lora-rank 16 + tool parser at 0.55 mem (shares GPU0 with smoke trainer).

### 7-GPU run CANCELED at step 27/60 (2026-06-12 ~01:50 UTC, user decision) + full online run launched
- Cancel reason: recipe flaw — HF-generate training rollouts clip at 8192 46-86% of the time vs ~21% for the vLLM rollouts the bank was calibrated with (training/measurement distribution mismatch), reward flat-noisy (~0.17, no trend over 27 steps). Checkpoints 10/20 + wandb metrics (335aod23) retained for reference.
- **Full online run LAUNCHED**: tmux `online`, `outputs/online_selfplay_run1`, 60 steps × 4 questions × k=8 @ 8192, trainer GPU1, vLLM GPU0 (rollouts via vLLM → no clipping mismatch by construction), per-artifact judges, wandb run name `online-selfplay-grpo`.
- smoke3 validated per-artifact judging (step 2: passes [5,3] both trained, split-claim 6-vs-5 resolved 2/2 vs 2/2). Adapter-push 400 root-caused: vLLM default `--max-loras 1` → restarted with `--max-loras 8` + unload-on-swap in loop.
- Desktop reports refreshed: `online_smoke_FULL_detail.md` (316KB, complete process: prompts/thinking/rollouts/judge code+stdout per artifact), `online_smoke_report.md` (checklist, final architecture).

### 7-GPU run post-mortem evals (canceled run, checkpoints evaluated anyway): NO improvement
- base 0.4760/0.8600 (pass@1/pass@4) → step10 0.4560/0.8380 → step20 0.4680/0.8440. Second clean negative for the two-phase recipe; cancel decision validated.

### Overnight run v3 LAUNCHED (2026-06-12 06:55 UTC) — hardened "best working version"
- v1 (sequential questions, no forced answers) ran 6 steps (preserved at outputs/online_selfplay_run1_v1_6steps), v2 superseded before step 1.
- v3 hardening: (1) **retries with backoff** on all loop+judge HTTP calls (transient vLLM error no longer kills the run); (2) **per-step try/except** isolation; (3) **forced-answer on clip** (legacy api_final_prefill behavior; integer still policy-sampled) with `forced_frac` logged per step as the DAPO guard metric — step 1 measured 0.031, so training-on-forced is marginal, watch for drift; (4) **cross-question parallelism** (4 questions' rollouts+judges overlap); (5) vLLM bumped 0.55→0.85 mem (trainer owns GPU1 now); (6) lesson: never `pkill -f <name>` over ssh where the command string contains <name> — it kills your own session (tonight's recurring exit-255s).
- Run: tmux `online`, outputs/online_selfplay_run1, 70 steps × 4 q × k=8 @ 8192, wandb `online-selfplay-grpo`. Step 1: passes [7,8,5,8], 2 trained, solver_step1 pushed, 23.5 min (incl. load). ETA ~12-15h.
- Industry-standard note (user asked): forcing answers on clip is standard in EVAL (s1 budget forcing), NOT standard in RL training (DAPO filters overlong instead); our guard = forced_frac + completion-length trend; hybrid fix identified (forced answers for feedback only, exclude clipped from gradient) if it drifts.

### KEY NEGATIVE RESULT (2026-06-13): paired early-question retest → NO learning (hypothesis DENIED)
- Test design: re-rolled the online run's own step 1-10 questions (40 Q, originally measured first-time-out at near-base policy, mean 6.28/8 = 78%) under the step-41 adapter, identical sampling (k=8, temp 1.1, 8192), reusing original judge verdicts (`--fast`: score by matching each Q's known-correct answer; `scripts/early_question_retest.py`).
- Result: mean 6.28 → 6.15, **delta −0.12 (SE 0.22)** — flat. Sub-8 subgroup (21 Q w/ headroom): −0.05 (SE 0.40). Hard ≤4/8 (8 Q): +0.62 (SE 0.62), within 1 SE = not significant. 22/40 flat, 10 up / 8 down = resampling noise around an unchanged policy.
- **Interpretation (pre-agreed grid): this is "no learning," NOT "learning that doesn't transfer." My distribution-gap hypothesis for flat GSM8K is DENIED.** The model is not improving even in-distribution on its own questions.
- New suspect = the GRPO update is too weak to matter: LoRA r8 @ lr 1e-5, losses ~0.01, advantage signal starved (most groups at high pass → low reward std → ~0 advantage; 0/8 & 8/8 groups skip). Healthy-looking loop mechanics (nonzero reward/KL, adapter advancing) masked a near-no-op update.
- 7-GPU run confirmed canceled (step 27); its checkpoint evals also showed no improvement (step10 0.456, step20 0.468 vs base 0.476) — consistent.

### Static diagnosis of the no-learning result (2026-06-13, NO GPU jobs — pure artifact analysis)
v3 completed all 70 steps (not a crash). Diagnosis from steps.jsonl + saved adapters + detail traces:
- **A. Weight drift** (`adapter_step*` LoRA delta ‖B·A‖·scale, CPU): monotonic 0.039→0.328 over 70 steps, all 496 tensors nonzero. → gradient flows, NOT a broken/no-op/tokenization-garbage update; but 0.33 total across ~250 layer-pairs = tiny nudge to 9B.
- **B. Signal starvation (DOMINANT)**: 280 groups → pass dist {0:12,1:7,2:7,3:7,4:10,5:18,6:21,7:43,8:155}. **155/280 (55%) were 8/8**, only 113/280 (40%) in 1-7 contributed gradient; **mean 1.61 trainable groups/step** (of 4), 7 steps had 0. Proposer's escalation feedback failed to bite — most rollout budget wasted on too-easy questions giving zero advantage.
- **C. Recurring-structure pass early→late: 6.56→5.44** (flat-to-down, confounded, n=8) — corroborates retest, no improvement.
- **D. Answer entropy 2.41→1.96** (mild concentration, benign). **E. Forced ~20% stable** (not gaming).
- **F. Judge audit**: 602 artifacts, 3% split (consistent), but 2/9 spot-checked CORRECT verdicts had code stdout ≠ claimed answer (n tiny → inconclusive; flag for proper audit).
- **CONCLUSION: not broken, UNDERFITTING from (1) signal starvation + (2) tiny LoRA/low-LR update.** Full-weight RL fixes (2) but NOT (1) — an 8/8 group has zero advantage at any capacity. Fix needs BOTH + judge-integrity check.

### Overfit diagnostic (scripts/grpo_diagnose.py) — REFINES diagnosis: mechanism sound, recipe not inherently weak
- Tier 1: tokenization round-trip 0/8 mismatches, mean per-token logprob -0.15 → trainer scores the real sampled tokens, gradient is the correct quantity (rules out garbage-gradient class).
- Tier 2 overfit, EXACT production recipe (LoRA-r8, lr1e-5), fixed batch (5 correct/3 wrong), correct−wrong logprob gap: start -0.085 → step10 -0.071 → step20 -0.025 → step30 +0.063 → step40 +0.215. **Repeated signal DOES separate correct from wrong in ~30 steps at lr1e-5.** So lr1e-5/r8 is not inherently too weak.
- **Refined root cause: signal STARVATION + each question seen ~once (no sustained pressure).** Not "step too small." Production never accumulates the steps-on-coherent-signal that the overfit shows works.
- **Fix priority reordered:** (1) signal density (3/8 target + curated 2-4/8 exemplars — DONE: target changed, questions_for_curation.md exported, 49 sweet-spot Qs); (2) **reuse each rollout batch for several gradient mini-epochs** (cheap, = the "40 steps on same signal" that worked) — highest-leverage code change; (3) full-weight/higher-LR secondary (speeds, not the blocker; full-weight needs multi-GPU ZeRO).

### Verifiability / judge-reliability audit (2026-06-13, static + external oracle)
- Inter-judge agreement (per-group): mean **0.973**, 83% of groups at 1.0. Judge temp was 0.2/0.5 → **raised to 0.6** for next run.
- Correct-rollout answer consensus: **260/268 groups (97%)** had a single agreed correct integer; 8 groups accepted contradictory answers (reward noise, on ambiguous Qs).
- Program-oracle from saved judge stdout: too sparse to trust (only 27/602 parseable) — needs re-execution (TODO).
- **External oracle gpt-5.4 (no code, independent solve, n=80): 86% overall, but 95% (60/63) on the easy 6-8/8 band where our answer is robustly established.** Agreement falls with difficulty (mid 67%, hard 20%) because gpt-5.4 does hard enumeration in-head; 7/11 disagreements within ±2 of our answer = gpt-5.4's own counting slips, NOT judge error. (nano was useless: 62%, wild hallucinations — confirms a non-code model can't verify these computational Qs; this is why the code-executing judge exists.) Scripts: `scripts/oracle_compare_nano.py` (ORACLE_MODEL env).
- **Conclusion: judge is reliable where it matters (95% external corroboration on established answers); residual noise concentrated on hardest questions, exactly where code execution beats reasoning.**

### run2 (2026-06-14): over-aggressive update CAUGHT by pre-launch sanity sweep; killed at step 6
- Config: lr 3e-5, mini-epochs 3, r16, COLD start (outputs/ got wiped by disk cleanup @93% full; banks restored from Desktop).
- Sanity sweep findings: reward/judge CLEAN (0 contradictions, 0.98 agreement). But: (1) band only 12% mid (cold-start removed the diverse seed the counterfactual relied on); (2) **WEIGHT DRIFT RUNAWAY: ||delta|| 0.44 at step1, 1.165 by step6 = ~0.2/step = ~40x v3's 0.005/step.** Model was being THRASHED not trained (KL diverging to -0.031, difficulty oscillating 6.5->7.5->1.5->8.0->7.0). The 3x LR x 3 mini-epochs stack compounded worse-than-linear on tiny 2-group batches.
- Lesson: v3 too weak (0.005/step, no learning); run2 too strong (0.2/step, thrashing). Sweet spot ~0.02-0.05/step.
- Diverse Gemini GOLDEN candidates calibrated (scripts/calibrate_golden.py): 6/8 too EASY for Qwen3.5-9B (BANANA->12, CRT->378 etc). Tension: diverse topics = easy; hard = narrow number-theory. Resolution: GOLDEN anchors DIFFICULTY (calibrated mid), seed bank provides DIVERSITY.

### run3 (2026-06-14) LAUNCHED with corrections: lr 1.5e-5, mini-epochs 2, r16, SEEDED (332 diverse history), 5 calibrated-mid GOLDEN. wandb rrjbia70. Watching weight-drift (target 0.02-0.05/step) + band. Early-question retest at ~step 25 = verdict.

### PIVOT to fast training-only harness (2026-06-14, goal: get measurable in-dist + OOD learning)
- run3 killed (full loop too slow to iterate, ~35min/step). Proposer+judge are validated; open question is purely whether TRAINING learns. So stripped to `scripts/training_only_grpo.py`: 62 fixed mid-difficulty (2-6/8) questions w/ known refs (no proposer, no judge, exact-match reward), 46 train / 16 held-out. Measures IN-DIST (held-out pass) + OOD (GSM8K pass@4) before vs after.
- Running 2 configs in parallel (4 GPUs): A=batch6, B=batch12, both lr1.5e-5/epochs2/r16, 25 steps. Tests "does it learn" + batch-size effect. ~3-5h. Question set: outputs/fixed_train_questions.json.
- This is the iteration unit for the goal: short training-only run -> dual (in-dist+OOD) check -> adjust -> repeat.

### Box contention RESOLVED + fast learning sweep (2026-06-14/15)
- R-Zero (`rzv`, GPUs 0-3) was repeatedly SIGKILLing self-play processes (all sessions die, rzv survives). Fix: added SHARED COMPUTE COORDINATION protocol to global ~/.claude/CLAUDE.md (R-Zero=GPU0-3, self-play=GPU4-7 `sp_*` sessions, no box-wide pkill/reset). WORKING: sp_ runs survived 4.5h incl. an rzv restart.
- training-only harness parallelized (rollouts fired batch-wide concurrently, not sequential) — was ~1.5-2h/step, far too slow.
- Confirmed: fixed-good-questions gives DENSE signal (4-6/6 trained groups vs v3's 1.6/4). Baselines: in-dist ~0.47-0.56, GSM8K pass@4 ~0.99.
- Running fast bracket: A=lr1.5e-5(~0.15/step), B=lr1e-5(~0.1/step), both batch8/ep2/r16/12 steps. Verdict = in-dist + GSM8K before/after deltas. First clean test of whether GRPO learns on known-good questions.

### Training-only sweep RESULT (2026-06-15): no measurable gain in 12 steps, but UNDERPOWERED
- A (lr1.5e-5): in-dist held-out 0.509->0.473 (-0.036), GSM8K pass@4 0.967->0.983 (+0.017)
- B (lr1e-5): in-dist 0.536->0.527 (-0.009), GSM8K 0.983->0.983 (0.000)
- Per-step training passes FLAT (A ~5.3, B ~5.7) — but confounded (different batch each step).
- **NOT a clean "training broken" verdict — the test was underpowered**: (1) only 12 steps ~2 exposures/Q (overfit test needed ~30 on a FIXED batch); (2) GSM8K pass@4 at ceiling 0.98 (no OOD headroom — must use pass@1); (3) held-out n=14 too noisy (±0.13); (4) held-out measures GENERALIZATION not memorization. The earlier overfit test ALREADY proved the gradient mechanism works on a fixed batch.
- Memorization check (did trained-Q improve) launched but vLLM too slow at 8192 / one arm 400'd — inconclusive.
- **Next iteration design (decisive): small FIXED train set (~8-12 Q) x many steps (30-40), measure pass on the TRAINED Qs (memorization) + GSM8K pass@1 + larger held-out. If trained-Q pass rises -> learning works, scale exposures for generalization. If flat -> training truly broken.**
- Infra wins this round: coordination protocol stopped box kills (sp_ runs survived 5h+); parallelized harness ~2x faster.

### MEASUREMENT FIX (2026-06-15): GSM8K was a broken yardstick; switched OOD to GSM-hard
- GSM8K pass@1 at FULL 8192 tokens = 0.980 for base Qwen3.5-9B (saturated/likely contaminated). The earlier "0.476" was a 1024-TOKEN TRUNCATION ARTIFACT, not real ability. So GSM8K has no headroom -> useless for detecting OOD learning. A chunk of "no OOD gain" was this.
- **GSM-hard base pass@1 0.790 (pass@4 0.850)** — real headroom. New OOD benchmark. `scripts/gsmhard_eval.py` (loads reasoning-machines/gsm-hard, integer answers, fits FINAL_ANSWER extraction).
- Proper measurement now: IN-DIST = trained-Q pass (memo run, baseline 0.542); OOD = GSM-hard pass@1 (baseline 0.790).
- Decisive memo run (12 fixed Q x 30 steps, lr1e-5) in progress; will eval its adapter on GSM-hard for OOD. This is the first time both axes are measured against benchmarks that can move.

### CORE FINDING (2026-06-15): logprobs separate but generation pass-rate doesn't follow (RLVR wall)
- Overfit floor test (6 solvable golden Qs, aggressive lr3e-5 x3 epochs, batch-all): FLAT (baseline 4.33/8, steps 3.0-4.5/8 bouncing). Cannot overfit even 6 questions.
- BUT grpo_diagnose earlier showed logprobs DO separate (correct completions up, wrong down). So gradient direction is correct; the model is pushed right at the logprob level but doesn't change what it GENERATES. Weights move a lot, pass flat.
- Diagnosis: reinforcing specific correct trajectories (which share ~95% tokens with wrong ones, differ only at final answer) barely shifts the generation distribution. Classic RLVR trajectory-vs-generation gap, compounded by signal dilution (answer-token gradient divided by ~1000 reasoning tokens via per-token length norm) + KL pullback.
- Also suspect: the golden Qs are BRUTAL stacked-constraint number theory; "5/8" may be inconsistent luck, not a sharpenable reliable method.
- **Fix being tested**: added `pg_norm` param to grpo_step (fixed normalizer instead of /n_tok, undilutes answer-token gradient) + KL_BETA/PG_NORM env hooks in harness. Rerunning overfit on golden8 with KL_BETA=0 PG_NORM=256 lr1e-5 (tmux sp_overfit2). If it climbs, dilution was the wall; if flat, deeper reward-design issue or questions-too-hard-for-overfit.
- Infra: vLLM restarted faster (no enforce-eager, max-model-len 8192 -> needs max_new_tokens<=~6000 to avoid 400). Still ~35min/step (training compute: 3 epochs x 6 groups x 8 completions at 6000 tok).

### ROBUST NEGATIVE VERDICT (2026-06-16): GRPO-on-LoRA self-play does not learn; aggressive settings DEGRADE
- Greedy check (temp 0.2, de-diluted overfit2 adapter step6 vs base on golden8): BASE 0.469 -> ADAPTER 0.344 (-0.125). Per-Q: trained adapter DROPPED on Q4(.75->.25), Q7(1->.75), Q8(.75->.25); Q5,Q6 = 0/4 both (model genuinely cannot solve those 2 of 8). NOT masked learning — training actively harmed generation. De-dilution hypothesis REFUTED (undiluting/amplifying made it worse).
- Ruled out across ~12 configs: measurement, scale, dilution, KL, capacity, signal density, temp masking. No setting improves pass rate: gentle=flat, aggressive=degrades. Gradient locally moves logprobs correctly (grpo_diagnose) but generation doesn't follow / degrades = trajectory-vs-generation wall + partly-unsolvable/luck-based questions.
- **STOP tuning this arm.** Genuine next directions if continuing self-play: (a) prove RL can sharpen EASY questions first (50% = reliable-method-with-slips, not brutal stacked number theory); (b) different RL formulation — proper PPO clip ratio for multi-epoch, or full-weight; (c) lean on R-Zero hedge (running on GPUs 0-3). Recommend strategic reassessment over more hyperparameter tuning.

### Diagnosis refined (2026-06-16): NOT bimodal; the flatness is BATCH-SIZE noise (wobble)
- Corrected earlier "bimodal" claim: v3 per-Q pass dist (n=280, k=8) is SKEWED-EASY — 8/8:55%, 7-8:71%, MID 2-6:22% (63 Q), HARD 0-1:7%. There IS a real middle; the bank is just mostly too-easy. GSM-hard: only 11/80 in [0.25,0.75].
- Bouncing mechanism (KEY): per-step trained-Q pass swings (e.g. Q5: 0→3→0→8, Q4: 7→6→8→1) — a 0/8↔8/8 swing is the POLICY lurching, not measurement noise. Cause: effective batch ~48-64 rollouts/step (6-8 Q × k8), per-Q advantage from only 8 samples = high-variance gradient; lr1e-5×2-3 epochs steps hard on that noise; fresh noisy rollouts each step → updates point different ways → cancel → flat-with-swings (random walk, not climb). This is 10-20x smaller batch than standard RLVR (100s-1000s prompts).
- **BATCH-SIZE HYPOTHESIS (user's call, testing now):** too-small batch → noisy gradient → wobble. Fix = bigger batch. Launched `sp_bigbatch`: 16 Q × k16 = 256 rollouts/step (4-5x), 1 epoch (no multi-epoch wobble), lr1e-5, heldout 16, on mid set. vs `sp_easyfit` small-batch baseline. If bigbatch climbs where small-batch was flat → batch size was the wall.
- Other fixes implied: PPO clip ratio (bounds per-step lurch; our unclipped REINFORCE degraded under aggressive settings), more rollouts/Q (lower-variance advantage).

### BATCH-SIZE TEST RESULT (2026-06-16): bigger batch did NOT fix it; degrades
- bigbatch (16 Q x k16 = 256 rollouts/step, 4-5x prior, single-epoch on-policy, lr1e-5): trained-Q 0.359 -> 0.42 -> 0.30 -> 0.32 -> 0.26. Flat-to-DECLINING (256-sample noise ~0.03, so the decline is real policy degradation). easyfit small-batch control: flat 0.39-0.56.
- KEY: bigbatch is single-epoch ON-POLICY (PPO clip wouldn't change it) yet DEGRADES. On-policy PG with correct rewards should not make the model worse. Strong signal of an implementation issue in grpo_step/logprob/advantage, OR fundamental instability of this LoRA-GRPO-on-long-completions setup.
- **ROBUST NEGATIVE, FULLY INVESTIGATED**: swept batch, LR, epochs, dilution, KL, question difficulty/selection, length-norm. None learn; several degrade. Logprobs separate (gradient locally right) but generation doesn't follow / degrades.
- **Recommended next (NOT more tuning):** (1) TRL GRPOTrainer sanity run on the easy questions — battle-tested independent impl; if it learns, our grpo_step has a bug; if not, fundamental. (2) Real compute (full-weight, 1000s-rollout batches, multi-node). (3) R-Zero hedge. AWAITING USER STRATEGIC DECISION.

### Next-run levers to actually move the policy (for the next overnight)
- [ ] Raise LR (1e-5 → 3e-5/1e-4) and/or LoRA rank (8 → 32/64); these are the most likely culprits
- [ ] Harder calibration target: proposer is still emitting too many 8/8 (low-advantage) — push difficulty so more groups land 2-6/8 where GRPO advantage is strongest
- [ ] Consider multiple gradient steps / larger effective batch per set of rollouts
- [ ] In-distribution eval (bank-holdout) as primary metric; GSM8K secondary
- [ ] Verify the custom GRPO step numerically (advantage sign, logprob grad path) — rule out an implementation no-op before blaming hyperparams

### Pending (infra)
- [ ] Online run v3 still training (~step 41); let it finish or stop — it will not show gains as-is
- [ ] vLLM2 on GPU2 (port 8002) still up from the retest — free it
- [ ] Report 7-GPU run result + per-checkpoint eval table when done (watcher live; superseded architecturally but still answers "does correctness-reward training transfer")
- [ ] Full online run launch when GPUs free (~10:00 UTC): online_self_play_grpo.py, 60 steps, q/step 4, 8192 tokens, dedicated GPUs
- [ ] Phase-2 judging: proposer emits verification predicates (multi-answer/constructions); phase-3: repo artifacts
- [ ] Final post-loop adapter push returned HTTP 400 in smoke2 — chase before full run
- [ ] run1 (full GRPO rerun) still staged+held in tmux — likely superseded by this pipeline; decide after eval
- [ ] Unity cluster: user must paste public key into unity.rc.umass.edu/panel; then `ssh unity`
- [ ] Integrate agentic proposer + code judge into the production loop proper (replace generate_pairwise_data path)

---

## 2026-06-10 — Root cause of GSM-hard "gains": answer-first GRPO template (training was wrong, must rerun)

### TL;DR
The 25-cycle Qwen3.5-9B self-play run (`outputs/self_play_grpo_loop/qwen35_ablation_update_gen0_8x8_vllmlora/`) did **not** improve reasoning. Pass@1 "gains" on GSM-hard were format compliance (format_rate 42% → 98%) while conditional accuracy (condAcc = correct | parseable) **declined**. Root cause: the GRPO *training-time* prompt template demanded `FINAL_ANSWER:` on the **first line** (answer before any reasoning), capped at 384 tokens, with a clean-tail bonus — so the reward-optimal completion was literally a bare `FINAL_ANSWER: <n>` with no reasoning. All existing checkpoints are tainted; the loop must be rerun with the fixed template.

### Evidence (collapse is real, not variance)
condAcc on GSM-hard (150 q, k=4, Wilson 95% CI):
- base: 0.352 [0.296, 0.412]
- cycle014: 0.375 [0.331, 0.422]
- cycle022: 0.221 [0.189, 0.256]
- cycle025: 0.206 [0.175, 0.241]  ← non-overlapping with base

Tag-position collapse (median chars before `FINAL_ANSWER:` in eval completions):
base 236 → c014 157 → c016 54 → c022 30 → c025 33. "Blurt rate" (<200 chars) 0.375 → 0.872.

### Ruled out
- **LoRA**: single continued r=8/alpha=16/all-linear adapter across all cycles; Frobenius norm flat (c1 30.90, c7 25.72, c14 25.73, c20 25.74, c25 25.75) while condAcc fell. Not the cause.
- **Self-judge quality**: gpt-5.4-nano proxy-oracle audit of 320 actual GRPO rollout solutions (cycles 1–25): pooled agreement 0.936, proxy false-positive rate 0.026, false-negative 0.111, nano unanimous 98%. The judge verdicts were fine; earlier "brevity bias" claim retracted.

### Root cause detail (two generation streams — important!)
1. **Rollout artifacts** (`cycle_*_samples.jsonl`, via `pairwise_rollouts_qwen35_overnight.yaml`): reasoning-first, `enable_thinking: true`, thinking_budget 512, `api_final_prefill: "FINAL_ANSWER: "`, 4096 tokens. These look healthy — but they're only used for question selection and judging. **No gradients flow through them.**
2. **GRPO-time completions** (TRL GRPOTrainer generates online under `train_pairwise_verdict_qwen35_overnight.yaml`): the old template said "Put the final answer first. Your first line must be exactly: FINAL_ANSWER: ... Do not write more than 6 lines total", max_new_tokens 384. Combined with format_weight 0.2, answer_boundary_weight 0.1, and `reward_clean_final_answer_format` (1.0 only if zero chars after tag), the optimum is a bare tag line. These completions are never saved, which is why inspecting samples.jsonl was misleading.

Compounding issue (open): proposer distribution misaligned with GSM-hard — self-generated question numbers log10 median ~1.0–1.6 vs GSM-hard 6.61, stable from cycle 1.

### Eval was also answer-first
`scripts/eval_gsmhard_sweep.py:18` uses the same answer-first prompt → eval numbers also conflated. Created `scripts/eval_gsmhard_reasonfirst.py` (reason-first, tag-last, 1024 tokens) and submitted **Slurm job 79035** (base, c014, c020, c025; 150q, k=4) → `outputs/evals/gsmhard/reasonfirst_150/summary.json`. Interpretation: c025 ≈ base → collapse was prompt compliance only; c025 < base → genuine degradation; c014 > base → first real gain.

### Fixes applied (local + server, verified identical)
- `grpo_math/configs/train_pairwise_verdict_qwen35_overnight.yaml`:
  - prompt → reason-first: "Solve the problem step by step... End your response with one final line that is exactly: FINAL_ANSWER: <integer>"
  - `rollout.max_new_tokens: 384 → 1024` (judge cap of 128 at line 46 untouched)
- Reward stack needs no code change — format/boundary/clean-tail rewards align correctly with tag-last completions.

### Pre-rerun code audit (2026-06-10, all clear with caveats)
- **Loop driver** (`run_self_play_grpo_loop.py:1510`): re-loads the train template fresh each cycle and overwrites snapshots — resume can NOT reuse old answer-first snapshots. Rerun hazards are launch-args only: don't pass a tainted `--initial_solver_model`, and use a NEW `--run_tag` (old run dir would reuse old cycle JSONLs/checkpoints via `--skip_rollout_if_jsonl_exists`).
- **Reward stack**: all tag-position-agnostic (`final_answer_tail_char_count` = chars after tag → tag-last scores format 1.0 / boundary 1.0). With reason-first completions the verdict judge now sees full reasoning (canonicalize keeps the prefix). Trainer `max_completion_length` = `rollout.max_new_tokens` = 1024 (train_grpo_trl.py:1124). `inherit_solver_max_new_tokens` not set → no 4096 override.
- **Server reward.py is NEWER than local** (uncommitted on server): last-occurrence used consistently in strict-extract/canonicalize/tail (fixes first-vs-last asymmetry), case-insensitive regex, adds `extract_final_answer_text_strict`. Correct for tag-last. Local repo is stale here — sync eventually.
- **Launcher**: the 25-cycle run used `scripts/slurm_qwen35_ablation_update.sbatch` (run_tag qwen35_ablation_update_gen0_8x8_vllmlora), which points at the FIXED train config — safe to reuse with a new run_tag. (`slurm_qwen35_vllm_loop_overnight.sbatch` also points at it.)
- **Server/local drift**: server on `main` + ~12 uncommitted modified files; local on `simplify-single-integer-no-python`. Differing: reward.py (server newer), rollout solver prompt_template (server: think-channel + "final response must be exactly one line FINAL_ANSWER", local: answer-first + optional check). Server is what runs.
- **Rollout solver stream** intentionally answer-only in visible text (reasoning in thinking channel, budget 512, prefill) — style now differs from the reason-first training template. Not a bug, but rollout-judged solutions won't match GRPO-time completion style.
- **Still stale answer-first templates on server** (only matter if used): `train_pairwise_verdict_qwen35_{low_format,no_boundary,confirm_strict_overnight,confirm_strict_overnight_safe}.yaml`, `train_question_qwen35_confirm_strict_overnight{,_safe}.yaml`, `scripts/eval_gsmhard_sweep.py` (superseded by `eval_gsmhard_reasonfirst.py`), `scripts/question_reward_signal_experiment.py`. Note: `gated_format` variant is already tag-last.

### Reason-first eval results (job 79035, COMPLETED 2026-06-10, 55 min)
`outputs/evals/gsmhard/reasonfirst_150/summary.json` (150q, k=4, reason-first prompt, 1024 tokens):

| model | pass@1 | pass@k | format | condAcc | matched both-formatted acc |
|---|---|---|---|---|---|
| base | 0.287 | 0.580 | 0.615 | 0.295 | 0.453–0.509 |
| c014 | 0.273 | 0.547 | 0.738 | 0.280 | 0.421 |
| c020 | 0.273 | 0.587 | 0.865 | 0.282 | 0.364 |
| c025 | 0.233 | 0.507 | 0.952 | 0.218 | 0.279 |

**Conclusion: no checkpoint beats base even under reason-first prompting.** c014/c020 ≈ base (reasoning mostly intact; only format improved); c025 genuinely degraded (matched-pair acc 0.279 vs base 0.453). → **Seed the rerun from fresh base.** Curiosity: contains_correct@1 is *higher* for trained cycles (0.61 → 0.69) — they compute the right number somewhere more often but finalize the wrong one.

### Smoke test before rerun (job 79040, submitted 2026-06-10)
1-cycle end-to-end smoke of the fixed pipeline, same 2-GPU topology as production (GPU0 vLLM, GPU1 train). Run dir: `outputs/self_play_grpo_loop/qwen35_reasonfirst_smoke/`, log `outputs/slurm/qwen35_rf_smoke_79040.out`.
- Configs (server): `pairwise_rollouts_qwen35_smoke_reasonfirst.yaml` (overnight copy, num_questions 8), `train_pairwise_verdict_qwen35_reasonfirst_smoke.yaml` (fixed overnight copy: steps 4, save_every 4, **debug_rollouts enabled** → TRL log_completions), `scripts/slurm_qwen35_reasonfirst_smoke.sbatch` (--cycles 1, --max_steps 4, --max_train_samples 16, run_tag qwen35_reasonfirst_smoke)
- Pass criteria: (1) cycle_001_samples.jsonl written; (2) snapshot configs/cycle_001_train.yaml carries the reason-first prompt; (3) logged GRPO completions show reasoning BEFORE a final `FINAL_ANSWER:` line (not blurted first-line tags); (4) nonzero verdict rewards (teacher API path works); (5) cycle_001_grpo/checkpoint-4 saved.

**RESULT (COMPLETED 2026-06-10, 25 min wall): PASS — pipeline works end-to-end; cleared for full rerun.**
- Loop stages all ran: generator 24s, solver 220s (56 prompts), judge batches 6–48s, question-train checkpoint-25, GRPO 4 steps (train_runtime 284.8s), checkpoint-4 saved, "[loop] complete".
- Criteria: #1 ✅ #2 ✅ (snapshot prompt is reason-first/tag-last) #5 ✅.
- #3 ✅ — parquet completion dump (16 completions across 4 steps): of the 7 with a parseable `FINAL_ANSWER:` tag, chars BEFORE tag = min 13 / median 2397 / max 3699. The blurting pathology is gone; the model reasons at length before the tag.
- #4 ✅ (marginal) — mean reward_verdict 0.0625 (1/16 correct), reward_format 0.219, boundary 0.057. Teacher API path works; low correctness expected from base model on hard self-generated questions at cycle 1.
- Caveats observed (not blockers): 9/16 completions had NO parseable tag — base model is verbose (`<think>`-style blocks, "Final Answer: \boxed{}" preambles) and gets truncated at 1024 tokens mid-reasoning; 0/7 tagged completions ended exactly at the tag (trailing text after it, boundary reward partial). Expectation: format (0.2) + boundary (0.1) rewards now pressure toward "reason, then tag, then stop" *within* budget — the correct optimization target — instead of the old "tag immediately" optimum. Watch format_rate and tail_chars trend in the real run; if tag rate stays low after a few cycles, consider bumping max_new_tokens to 1536.

### 8k-token smoke test (jobs 79048 → 79062 → 79063, 2026-06-10; attempt #3 PASSED on uni + Brev)
Decision: rollout/GRPO completion cap 1024 → **8192** (the 1024-token smoke truncated 9/16 completions mid-`<think>`; Qwen3.5 naturally reasons 2–8k tokens). vLLM server already at `--max-model-len 32768`, no other change needed. Server files: `grpo_math/configs/train_pairwise_verdict_qwen35_reasonfirst_smoke_8k.yaml` (only diffs from 1024 smoke: line 52 `max_new_tokens: 8192`, output_dir `…_8k`; judge cap 128 untouched), `scripts/slurm_qwen35_reasonfirst_smoke_8k.sbatch` (run_tag `qwen35_rf_smoke_8k`, logs `outputs/slurm/qwen35_rf_smoke8k_79048.{out,err}`). Pass criteria same 5 as before, plus: tag rate should rise well above 7/16 now that truncation is removed. If 8k passes, the full rerun should use 8192 (edit `train_pairwise_verdict_qwen35_overnight.yaml` 1024→8192 before launching).

**Job 79048 (first 8k attempt) FAILED with CUDA OOM at GRPO step 1** — rollout/judge/question-train all fine (and notably fast: question-train 104s). OOM in TRL `_get_per_token_logps_and_entropies`: scoring forward materializes logits then accelerate casts to fp32 → `8 completions × ~6k tokens × 151k vocab × 4B ≈ 30.3GB` single allocation on the train GPU (which also holds model+LoRA, ~50GB in use). The scoring chunk size = `per_device_train_batch_size` = our `prompts_per_step` (grpo_trainer.py:1935, train_grpo_trl.py:1040-1045).
**Fix (config-only, both machines):** `prompts_per_step: 8 → 2`, `grad_accum_steps: 1 → 4` in `train_pairwise_verdict_qwen35_reasonfirst_smoke_8k.yaml`. Generation batch per optimizer step stays 8 (=2×4, divisible by k=4) so training semantics are unchanged; scoring forward now 2 sequences ≈ 7.6GB. **The full-rerun overnight config needs the same treatment when bumped to 8k.** Resubmitted as **job 79062**; failed run archived at `outputs/self_play_grpo_loop/qwen35_rf_smoke_8k_oom_79048/`.

**OOM #2 — job 79062 (uni) AND the Brev box run both died identically at GRPO step 1**: `Tried to allocate 7.58 GiB` in the **backward pass** (`accelerator.backward` → `_engine_run_backward`) with ~75.7–75.85 GB already PyTorch-allocated. The *loss* path (unlike scoring) saves bf16 logits + the fp32 cast + log-softmax for autograd, so even micro-batch 2 at ~8k tokens doesn't fit on an 80GB A100 alongside model+LoRA.
**Fix v2 (config-only, both machines, current):** `prompts_per_step: 2 → 1`, `grad_accum_steps: 4 → 8`. Generation batch still 8 (=1×8, divisible by k=4); semantics unchanged. Uni: archived to `qwen35_rf_smoke_8k_oom_79062/`, resubmitted as **job 79063**. Box: tmux `smoke8k` killed, run dir archived to `qwen35_rf_smoke_8k_brev_oom1/` (log → `brev_8k_smoke_oom1.log`), relaunched. If a third OOM hits, escalation options: lower max_new_tokens below 8192, or switch TRL to vLLM-based generation (TRL currently generates in-process via HF on the train GPU — also the 8k speed bottleneck; no `use_vllm` anywhere in train_grpo_trl.py).

**RESULT (attempt #3, both machines, 2026-06-10): PASS — micro-batch 1 fixed the OOM; cleared for full 8k rerun.**
- **Box** (run_tag `qwen35_rf_smoke_8k_brev`): 4/4 steps, checkpoint-4, `[loop] complete`, clean tmux exit. Step 1 completions: **8/8 tagged** (vs 7/16 at 1024), chars-before-tag med 4462, mean reward_verdict per step 0.75/0.25/0.875/0.375, format 0.625/0.5/0.625/0.375, `frac_reward_zero_std: 0` all steps (good gradient signal). step_time 855–900s; clipped_ratio 0.25–0.375.
- **Uni** (job 79063): 4/4 steps, checkpoint-4, `[loop] complete`. 19/32 tagged overall (59%); verdict per step 0.0/0.125/0.25/0.125 — its proposer drew **unsolvable questions** (no integer answer: cylinder r³−50r+200=0; chord segment 5π transcendental), so the solver reasons to the 8192 cap and can't conclude. NOT a pipeline bug — confirms backlog item "proposer distribution alignment". One step had `frac_reward_zero_std: 0.5` (all-same-reward groups = no GRPO gradient = wasted 8k-token generations). step_time 753–755s.
- Length facts at 8k: median completion ~4.3–5.9k tokens; "early tags at char 9/13" were echoed *examples* ("For example, FINAL_ANSWER: 42"), not blurts. Note: format/boundary rewards trim the never-concluding tail but do NOT pressure median length down — budget ~5k-token median generation for the whole run.
- **Full-run prep (done)**: `train_pairwise_verdict_qwen35_overnight.yaml` patched on BOTH machines (max_new_tokens 1024→8192, prompts_per_step 8→1, grad_accum_steps 1→8). Box launcher staged: `scripts/brev_run_overnight_8k.sh` (port of `slurm_qwen35_vllm_loop_overnight.sbatch`: 10 cycles × 10 steps, 32 q/cycle, --max_train_samples 28, --log_rollouts_to_wandb, run_tag `qwen35_rf_8k_run1`, log `outputs/slurm/brev_overnight_8k.log`). Decision: launch on box (user choice), report-first before launch. Considered+deferred: parallelizing question-train with solver GRPO (semantically safe — both depend only on rollout output — but saves only minutes/cycle and needs a loop-driver change); vLLM-based TRL generation is the real speed lever (~90% of step_time is generation).

### Brev 8xA100 box (operational 2026-06-10)
Access granted; `ssh -F ~/.brev/ssh_config awesome-gpu-name0` works (instance `awesome-gpu-name0`, azurerm.a100x8.sxm.brev-dgxc, 8×A100-80GB, CUDA 13.0 driver, 96 CPU, 1.7TB RAM, ~178GB disk free; note: A100s not H100s). Setup: repo rsynced to `~/self-play` (plus server's newer reward.py + reason-first/smoke/overnight configs), venv at `~/venvs/selfplay` pinned to uni versions (torch 2.10.0+cu128, vllm 0.19.1, trl 1.4.0, transformers 5.8.0, peft 0.19.1), `.env` synced (WANDB key is `WANDBKEY` there; launcher maps it to `WANDB_API_KEY`). Launcher: `scripts/brev_run_8k_smoke.sh` (non-Slurm port of the smoke sbatch; GPU0 vLLM @32768 ctx, GPU1 train; logs `outputs/slurm/brev_8k_smoke.log`; gotcha: script does its own `exec > >(tee …)` after mkdir because tmux-level tee raced the missing dir). Running 8k smoke in tmux session `smoke8k`, run_tag `qwen35_rf_smoke_8k_brev`. Box has other users' tmux sessions (one attached `train`) but GPUs were idle. Remaining 6 GPUs open the option of parallel ablations for the rerun.

### Pending / next steps
- [x] Poll job 79035 → done, table above; **seed rerun from fresh base**
- [x] Verify smoke job 79040 against the 5 pass criteria → PASS (see above)
- [ ] **Rerun the GRPO loop** with the fixed template — new `--run_tag`, fresh base seed (all prior checkpoints trained under the broken template)
- [x] Brev 8xA100 box access + setup → operational (see section above); 8k smoke running there in parallel
- [x] 8k smoke attempt #3 → PASS on both machines (see above); micro-batch 1 is the working 8k recipe
- [ ] Launch full rerun on box: `tmux new-session -d -s run1 "bash ~/self-play/scripts/brev_run_overnight_8k.sh"` (awaiting user go)
- [ ] Follow-ups for run2: TRL vLLM generation (speed), parallel question-train, per-cycle metrics to watch: tag rate, frac_reward_zero_std, completion length median
- [ ] Patch stale sibling configs on server if any ablation rerun will use them
- [ ] Decide whether to align rollout solver prompt with the reason-first training style
- [ ] Sync server's newer reward.py (+ other uncommitted server changes) back into the repo
- [ ] Proposer distribution alignment (questions don't resemble GSM-hard number scale/difficulty)

### Infra notes
- Remote: `/tmp/rrun.sh '<cmd>'` (expect+base64 SSH to shell.engr.wustl.edu, project at `/engrfs/project/jiaxinh/dylan_work/self-play`, venv `.venv-qwen-slurm`). Write remote files via *remote* heredoc inside one single-quoted rrun arg (no single quotes in content); local heredoc subshells hang.
- Eval lineage: base–c014 `outputs/evals/gsmhard/clean_update_early/71968/`, c016–021 `new_rounds/72499/`, c022–025 `update_remaining/72547/` (all answer-first — superseded by reasonfirst_150).
- Helper scripts on server: `scripts/{nano_rollout_audit,condacc_ci,adapter_norm,dist_shift_tests,len_per_cycle}.py`, `scripts/eval_gsmhard_reasonfirst.py`, `scripts/slurm_gsmhard_reasonfirst.sbatch`.
