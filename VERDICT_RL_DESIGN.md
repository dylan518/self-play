# Verdict-Only Self-Play RL Design

This document describes the simplified self-play training loop after removing the
strong-vs-weak solver separation signal. The system trains from verifier verdicts
directly, with a question filter added before expensive solving and RL updates.

## Goal

Train a solver model on generated questions using judge/verifier verdicts as the
reward signal.

The loop should prefer questions that are:

- well-formed
- answerable
- appropriate for the target domain
- neither trivially easy nor impossible
- likely to produce useful solver learning signal

## Main Components

### Proposer

The proposer generates candidate questions.

In the first version, the proposer can be static: a prompted model that samples
questions from the target domain. Later, the proposer can be trained with RL to
produce more useful questions.

### Question Filter

The question filter runs before solver sampling. Its job is to reject bad
questions before spending compute on answers and judging.

The filter should remove questions that are:

- malformed or incomplete
- duplicates or near-duplicates
- outside the target domain
- ambiguous or underspecified
- dependent on unavailable external context
- too easy for the current solver
- too hard or likely unsolvable
- unsafe or otherwise unsuitable for training

The filter can be implemented as a combination of deterministic checks and an
LLM-based rubric.

Example filter output:

```json
{
  "keep": true,
  "reason": "Clear number theory problem with a unique checkable answer.",
  "difficulty": "medium",
  "estimated_usefulness": 0.72
}
```

Rejected questions should be logged with reasons. These logs are useful for
improving proposer prompts and, later, proposer rewards.

### Solver

The solver samples multiple independent answers for each accepted question.

The system no longer needs strong and weak solver groups. Instead, it should
sample a small answer set from the current solver policy:

```text
question -> answer_1, answer_2, ..., answer_n
```

Sampling should preserve diversity. Temperature, top-p, or multiple reasoning
paths can be used to avoid identical answers.

### Judge / Verifier

The judge evaluates each solver answer and returns a verdict.

Minimal verdict schema:

```json
{
  "verdict": "correct",
  "score": 1.0,
  "reason": "The answer correctly proves the required divisibility claim."
}
```

Recommended verdict labels:

- `correct`
- `incorrect`
- `invalid`
- `unclear`

The reward mapping can start simple:

```text
correct   -> 1.0
incorrect -> 0.0
invalid   -> 0.0 or ignored
unclear   -> ignored
```

If compute allows, the judge can run multiple independent passes per answer and
use the mean verdict as the score. This is optional in the simplified system.

## Solver RL Update

For each accepted question, collect multiple solver answers and judge verdicts.
Convert verdicts into scalar scores.

Example:

```text
question_q answers: [a1, a2, a3, a4]
verdict scores:     [1.0, 0.0, 0.0, 1.0]
```

Use within-question advantage normalization:

```text
advantage(a_i) = score(a_i) - mean(score(a_1 ... a_n))
```

For the example above:

```text
mean score = 0.5
advantages = [0.5, -0.5, -0.5, 0.5]
```

The RL update should increase the probability of answers with positive
advantages and decrease the probability of answers with negative advantages.

Within-question normalization matters because it prevents the solver from being
rewarded merely for receiving easy questions. The model is trained to produce
better answers than its other samples for the same question.

## Proposer Training

The first implementation can leave the proposer untrained. Once the solver-only
loop is stable, train the proposer using question-level usefulness rewards.

Without strong-vs-weak separation, proposer reward should come from the verdict
distribution over solver samples.

Useful questions produce mixed verdicts:

```text
[correct, incorrect, incorrect, correct]
```

Less useful questions produce collapsed verdicts:

```text
all correct   -> too easy
all incorrect -> too hard, flawed, or beyond the solver
all invalid   -> malformed or incompatible with solver format
all unclear   -> poor judgeability
```

A simple proposer reward:

```text
proposer_reward(q) =
  filter_score(q)
  * verdict_variance(q)
  * judgeability(q)
```

Where:

- `filter_score(q)` is the question filter's usefulness estimate.
- `verdict_variance(q)` is high when solver samples produce mixed outcomes.
- `judgeability(q)` is high when the judge returns clear verdicts instead of
`unclear` or parse failures.

The proposer should learn to generate questions near the solver's frontier:
questions that are clear, checkable, and challenging enough to produce learning
signal.

## End-to-End Loop

```text
repeat:
  1. proposer generates candidate questions
  2. question filter accepts or rejects each question
  3. solver samples multiple answers for accepted questions
  4. judge/verifier scores each answer
  5. compute within-question advantages
  6. update solver with verdict-derived RL rewards
  7. optionally update proposer with question usefulness rewards
  8. log accepted questions, rejected questions, verdicts, and rewards
```

## What This Removes

This design removes:

- strong-vs-weak solver groups
- separation-based question scoring
- reliance on a strong/weak performance gap as the main reliability signal

The system now depends on:

- quality of the question filter
- quality of verifier verdicts
- multiple solver samples per question
- within-question normalization

## Minimum Viable Version

The first working version should implement:

1. Static proposer prompt.
2. Question filter with deterministic checks plus LLM rubric.
3. Multiple solver samples per accepted question.
4. Single-pass judge verdicts.
5. Verdict-to-reward mapping.
6. Within-question advantage computation.
7. Solver-only RL update.
8. Full JSONL logging for every stage.

Proposer RL and multi-pass judging can be added after the solver-only loop is
stable.