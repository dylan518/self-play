# Research note: In-context agentic proposer + program-grounded judging for self-play math

**Date:** 2026-06-11 · **Hardware:** Brev 8×A100-80GB · **Model (all roles):** Qwen3.5-9B

## TL;DR

Gradient-based question-proposer training (question-DPO) is replaced by an **in-context agentic
proposer**: a frozen model conditioned on measured feedback lines for every question it has
produced. LLM-judge verdicts are replaced by **executed-Python references** (first 3-program
majority, then a native tool-loop judge). Three claims are now empirically supported:

1. **In-context feedback steers difficulty.** Pass-rate feedback alone moved batch difficulty
   toward the 4/8 target band without any weight updates.
2. **In-context feedback steers diversity — if and only if the signal is in the feedback.**
   With no diversity signal, generation mode-collapsed onto ~5 templates. Adding descriptive
   structure-frequency feedback (no bans, no filters) raised novel-structure rate and produced
   the best-calibrated batch of the project by round 3.
3. **Execution-grounded judging strictly dominates LLM verdicts**, and a tool-loop judge
   dominates one-shot program majority.

## Background / problem

The previous pipeline trained the proposer with DPO over reward-bucketed question pairs. It
failed in a characteristic way (see 2026-06-09 notes): in-cycle eval_loss 0.06 but cross-cycle
holdout 0.71 (> ln 2 = random), reward 0.97-correlated with difficulty rather than
verifiability. Separately, the LLM judge (temp 1.4) was shown to err in both directions:
crediting wrong answers (7,5,5 credited against a different reference) and rejecting correct
ones (correct answer 100 rejected 4/8 times). Any proposer signal routed through that judge
inherits its noise.

## Method

**Proposer (frozen, in-context).** Each round, the model receives: role + stakes, three
prioritized objectives (A verifiable > B calibrated to 4/8 > C diverse), golden examples,
a difficulty guide, and a feedback line for every prior question:

```
[pass n/8 | verifier_programs_agree k/3 | structure: <label>] <question text>
```

It proposes 8 new questions; each is measured (8 solver attempts, temp 1.1, 8192 tokens) and
judged; the measured line is appended to history for the next round. Chains share one history
(cross-chain visibility) and a structure-frequency report. Nothing is banned or filtered —
diversity is steered only by information.

**Judge (execution-grounded).** v1: 3 independently sampled programs (temp 0.6), subprocess
execution, majority = reference; <2 agreement → question discarded. v2 (current): native
vLLM tool-calling (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`); the judge writes
code, sees stdout/stderr, iterates (≤6 turns), commits to `FINAL_REFERENCE: <int>` or
`UNVERIFIABLE`; 2 independent runs + tiebreak = agreement signal.

**Solver training.** GRPO (TRL), LoRA r8, correctness reward = exact match vs judge reference
(deterministic; no LLM teacher anywhere in the reward path), +0.2 format +0.1 boundary.

## Results

### 1. Difficulty calibration responds to feedback (no weight updates)

Chained rounds, pass distribution mean / trainable yield (trainable = 1–7 of 8):

| round | condition | mean pass | trainable |
|---|---|---|---|
| R2–R5 (earlier tests) | self-verify or no coaching | 8.0 | ~0/8 |
| R7 | coaching + feedback | 7.1 | 2/8 |
| R8 | + R7 feedback | 6.6 | 5/8 |

Self-verify ceiling (proposer must solve own question ⇒ everything 8/8) and
feedback-alone-insufficiency (no hardness coaching ⇒ drift easier) were both established
before R7; the coaching paragraph is load-bearing.

### 2. Scale-up exposed mode collapse; descriptive feedback fixed it

Production run (4 chains × 8 rounds): 284 questions, 124 trainable (44%), **but** 24% of
trainable questions were lattice-point x²+y²≤N variants; 15 questions shared an identical
8-word opening. Cause: chains were mutually blind, feedback carried no diversity signal, and
calibration coaching pointed at proven templates (pure exploit).

Fix (information only): model-generated structure labels per question (284 questions → 153
distinct labels), a frequency report in the prompt, labels in every feedback line, shared
cross-chain history. Test (2 chains × 3 rounds, seeded with all 284 cycle-1 questions):

| round | novel structures | trainable | pass values |
|---|---|---|---|
| 1 | 10/16 | 6/16 | bimodal (five 0/8, five 8/8) |
| 2 | 14/16 | 5/16 | bimodal |
| 3 | 7/16 | 9/16 | 0,0,1,1,2,3,5,5,5,6,7,8,8,8,8,8 |

Interpretation: **explore-then-calibrate.** Rounds 1–2 explore (new structures, badly
calibrated, as expected); round 3 re-uses round-1/2 structures *whose pass rates it had just
measured* and produces the best-calibrated batch of the project (56% trainable). The R3
novelty dip is exploitation of fresh measurements, not collapse. New territory never seen in
cycle 1: totient/sigma properties, Legendre valuations v_p(n!), derangement parity, trailing
zeros, d(n)=d(n+1), binomial-coefficient divisibility.

### 3. Tool-loop judge ≥ program majority ≥ LLM judge

On 16 questions with existing 3-program references: 14/16 agree; 1 question the program
judge had wrongly discarded was resolved by the tool judge (lattice paths avoiding 2 points
→ 12; hand-verified by inclusion–exclusion); the single disagreement traced to genuine
question ambiguity (1- vs 0-based indexing) — a proposer fault the judge surfaced.
Cost ≈ 2.3 tool calls/question of short non-thinking generations.

### 4. Solver training: rollout budget must match the bank's calibration budget

First GRPO run used 2048-token rollouts on questions calibrated at 8192. Direct measurement:
at 2048 only 2/24 solver attempts finish (vs 19/24 at 8192) — the model was graded on
questions it could not finish 92% of the time. Result: reward_correct ≈ 0 most steps, GSM8K
flat-to-noise across checkpoints (base 0.476 pass@1; steps 10/20/30: 0.466/0.478/0.426).
A second pitfall nearly produced a false null: TRL saves Qwen3.5 LoRA keys under
`model.language_model.layers` (+ vision-tower adapters); PEFT merge against the text-only
model silently loads nothing — caught because "trained" eval was bit-identical to baseline.

Current run (live): 8192-token rollouts, 7 GPUs × micro-batch 1 × grad_accum 8 = 7 question
groups/step (micro-batch >1 OOMs in backward: fp32 logits = batch × seq × 248k vocab × 4B),
60 steps, reward_correct/mean nonzero every step (0.07–0.23), per-checkpoint GSM8K evals
queued. Open concern: completions/clipped_ratio 0.57–0.80 vs ~21% non-finish in offline
measurement — possible HF-generate EOS handling mismatch; first suspect if the curve is flat.

## Open questions

- Does in-context proposer calibration track a *moving* solver (cycle 2 against the trained
  checkpoint)? This is the core self-play question and is now cheap to test.
- The answer-0-via-impossible-setup trick keeps resurfacing in subtler forms (lcm/gcd
  contradictions, impossible-diagonal rectangles). One-per-batch rule holds for now; may need
  the judge to flag vacuous-emptiness explicitly.
- Structure labels are model-generated and slightly generous; novelty rates are upper bounds.
- Why is training-time clipping 3× offline non-finish rate (EOS handling? rambling past the
  answer tag?).
- GSM8K may be the wrong transfer target for this question distribution (competition-style
  counting/number theory vs grade-school word problems); a held-out slice of the bank itself
  is the cleaner in-distribution eval.

## Artifacts

- Banks (tables + files): wandb `grpo-math/question-banks` (msdbgoh3); local
  `~/Desktop/agentic_run_artifacts/`; box `~/self-play/outputs/agentic_bank/`.
- Training: wandb run 335aod23 (metrics + checkpoint artifacts); markdown report written by
  the driver to `outputs/agentic_run_report.md` on completion.
- Code: `scripts/agentic_question_gen.py` (proposer loop, diversity feedback, prompt dumps),
  `scripts/python_tool_judge.py` (tool-loop judge), `scripts/merge_lora_adapter.py`
  (Qwen3.5 key remap), `grpo_math/configs/train_agentic_correctness_qwen35.yaml`.
