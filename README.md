# Self-Improving Training Loop with Reliability-Guided Task Generation

## Overview

This repository describes a self-improving training system composed of three components:

- **Question Generator** `G`
- **Solver** `S`
- **Judge** `J`

The goal of the system is to iteratively improve the solver using self-generated tasks while maintaining a stable and reliable training signal.

Unlike prior approaches that optimize for difficulty alone, this framework explicitly optimizes for **evaluation reliability**.

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

# Generator Objective

The generator is updated via reinforcement learning to maximize reliability:

\[
R_G(q) = f(R_{\text{cons}}(q), R_{\text{sep}}(q), R_{\text{stab}}(q))
\]

This differs from difficulty-only objectives by explicitly favoring tasks that produce stable training signals.

Over time, the generator shifts toward questions where evaluation is consistent and discriminative.

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
4. Compute reliability metrics
5. Update generator using reliability reward
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