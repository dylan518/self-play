# Self-Improving Training Loop with Reliability-Guided Task Generation

## Overview

This repository describes a self-improving training system composed of three components:

- **Question Generator** `G`
- **Solver** `S`
- **Judge** `J`

The goal of the system is to iteratively improve the solver using self-generated tasks while maintaining a stable and reliable training signal.

Unlike prior approaches that optimize for difficulty alone, this framework explicitly optimizes for **evaluation reliability** and **strong-vs-weak separability**.

---

# System Components

## 1. Question Generator

At each iteration, the generator produces a batch of tasks:

\[
q \sim G_\theta
\]

where `q` is drawn from the generator distribution.

The generator is optimized not for difficulty, but for how reliably the resulting task can be evaluated downstream.

---

## 2. Solver

For each generated task `q`, the solver produces multiple candidate solutions.

To induce quality variation, we sample from different parts of the solver’s distribution:

- Higher temperature sampling
- Best-of-N sampling
- Weaker checkpoints or smaller models (optional)

This produces a candidate answer set:

\[
A = \{a_1, \dots, a_k\}
\]

These answers represent a distribution of outputs with varying expected quality.

---

## 3. Judge

The judge evaluates:

- Individual answers
- Pairwise comparisons
- Rankings over the solution set

Formally:

\[
s_i = J(q, a_i)
\]

or

\[
J(q, a_i, a_j) \rightarrow \text{preference}
\]

Because the judge is stochastic, we run multiple judging passes:

\[
s_i^{(1)}, s_i^{(2)}, \dots
\]

From these we compute evaluation stability statistics.

---

# Reliability Metrics


# Training Signals (how to update each component)

This README so far defines the *question-generator* signal (reliability + separation). This section makes the update rules explicit.

## Solver learning signal (primary)

Prefer **absolute verification** over pairwise Elo as the main training signal whenever possible.

- Let the judge/verifier return either a scalar score or a probability-like correctness confidence:
  \(v(q,a)\in[0,1]\).
- Define an outcome reward, e.g. \(r_{solve}=2v-1\) (maps to \([-1,1]\)).
- For stability, use *within-question advantage normalization* (GRPO-style group baseline):

\[
A(q,a_k)=r_{solve}(q,a_k) - \frac{1}{M}\sum_{j=1}^M r_{solve}(q,a_j)
\]

This reduces variance and prevents the solver from only chasing "easy" questions.

## Question generator learning signal (reliability-guided)

The generator should be updated to propose questions that:
1) produce **reliable evaluation** (high consistency / low ambiguity), and
2) **separate** strong vs weak solver regimes (discriminative),
3) are in the solver's **learnable band** (not trivial, not impossible).

A practical recipe:

### (A) Reliability / separability scoring (task quality)
Compute on a candidate question \(q\):
- \(R_{cons}(q)\): score consistency (low variance across repeated judging)
- \(R_{sep}(q)\): strong-vs-weak separation (difference in mean verifier score between regimes)

Use these as either:
- **filters** (only keep tasks with \(R_{cons}>\tau_c\) and \(R_{sep}>\tau_s\)), and/or
- **weights** (upweight tasks during training).

### (B) Automated curriculum / learnability (AZR-style)
Let \(\bar r(q)\) be the solver's average outcome reward across \(M\) attempts on \(q\).
Reward the generator for tasks that fall into a "Goldilocks" region:

\[
r_{prop}(q)=\bar r(q)\,\big(1-\bar r(q)\big)
\]

This peaks when the solver succeeds about half the time and goes to 0 for trivial or impossible tasks.

### (C) Combined generator reward
One simple combined signal:

\[
r_G(q)= r_{prop}(q) + \lambda\,\max(0,R_{sep}(q)) + \mu\,R_{cons}(q)
\]

where \(\lambda,\mu\) are tuned so the system prioritizes *reliable, discriminative, learnable* tasks.

## Notes on Elo / pairwise

Pairwise Elo can be kept as a **diagnostic** (or a secondary signal), but it is typically noisier than absolute verification and is easier to Goodhart.

For each question `q`, we compute reliability signals to measure how stable and discriminative the evaluation is.

## Score Consistency

Low variance across repeated judging:

\[
R_{\text{cons}}(q) = 1 - \mathrm{Var}(J(q, a_i))
\]

---

## Strong vs Weak Separation

Measures whether stronger sampling regimes consistently outperform weaker ones:

\[
R_{\text{sep}}(q) = \mathbb{E}[J(q, a_{\text{strong}})] - \mathbb{E}[J(q, a_{\text{weak}})]
\]

---

## Preference Stability

Agreement rate across repeated pairwise judgments:

\[
R_{\text{stab}}(q) = \Pr\big(J^{(k)}(a_i \succ a_j)\ \text{consistent}\big)
\]

---

These metrics approximate whether a question is **easy to verify**, meaning good and bad answers can be reliably distinguished.

---

# Generator Objective (Current Oracle-Free Form)

The generator is updated via reinforcement learning using an oracle-free approximate signal:

\[
R_G(q) = \alpha \cdot \underbrace{R_{\text{sep}}(q) \cdot A(q)}_{\text{approximate signal}}
          - \beta \cdot \underbrace{v_{\text{strong}}(q)}_{\text{strong solve rate penalty}}
\]

where:

- \(R_{\text{sep}}(q) = v_{\text{strong}}(q) - v_{\text{weak}}(q)\)
- \(v_{\text{strong}}(q)\) is the judge-verified pass rate for the strong solver group
- \(v_{\text{weak}}(q)\) is the judge-verified pass rate for the weak solver group
- \(A(q)\) is **judge agreement** (not advantage), e.g. majority agreement across repeated judging

Interpretation:

- maximize separability between strong and weak groups
- penalize high strong solve rate to avoid trivially easy questions
- weight by judge agreement so noisy judgments contribute less

This differs from difficulty-only objectives by explicitly favoring tasks that produce stable, discriminative training signals.

Over time, the generator shifts toward questions where evaluation is consistent and discriminative, with an implied target difficulty controlled by the \(\alpha/\beta\) weighting ratio.

---

# Solver Update

The solver is updated using judged scores or preferences.

Possible optimization methods:

- Preference optimization (DPO-style)
- Policy gradient
- Offline ranking loss

For example, if \(a_i\) is preferred over \(a_j\):

\[
\theta_S \leftarrow \theta_S + \nabla \log P(a_i \mid q)
\]

Training is restricted to questions passing reliability thresholds.

---

# Iterative Training Loop

Each iteration:

1. Sample questions from generator
2. Sample multiple solver responses
3. Evaluate responses with judge
4. Compute reliability metrics and agreement-weighted separability
5. Update generator using \(R_G(q)\)
6. Update solver using judged preferences

This forms a closed loop:

- Generator adapts the task distribution
- Solver improves on reliably evaluable problems
- Judge provides feedback

---

# Deployment Context

This procedure:

- Does not require training a new base model
- Can be applied as post-training to open-weight models
- May use the same model for generator/solver roles
- Requires no new human annotations

The system enables iterative behavioral refinement using only model-generated data and reliability-based feedback.

---