# Verifier-Only Self-Play Status Report

## Executive Summary

We are moving the current self-play loop away from pairwise Elo and toward a
verifier-only training signal. In the current implementation, the generator
creates math questions, the solver produces multiple candidate solutions, and a
strong verifier labels each candidate as `CORRECT`, `INCORRECT`, `INVALID`, or
`UNCLEAR`. The intended training signal is the majority verifier verdict, not an
Elo ranking.

The latest local API-backed smoke run confirms that the verifier-only pipeline is
mechanically wired correctly: Elo is disabled, question-bank prompting is active,
verifier responses use 10k token budgets, and Python-assisted verification is
enabled.

The main finding is that the current Gemini smoke stack is **not ready to use as
training data**. Across older oracle-labeled artifacts, stronger vote agreement
correlates with better oracle agreement. However, the newest tiny run shows that
a verifier can be unanimously wrong on a class of generated questions, so
majority vote should be treated as a reliability feature rather than a
standalone ground truth.

## Current Run Configuration

The current smoke run used:

- Config: `grpo_math/configs/pairwise_rollouts_gemini25flashlite_single_verify_tiny.yaml`
- Output: `outputs/pairwise_rollouts_debug/samples_gemini25flashlite_single_verify_tiny.jsonl`
- Oracle comparison: `outputs/reports/samples_gemini25flashlite_single_verify_tiny_oracle_compare.json`
- Mode: `single_verify`
- Pairwise/Elo: disabled with `pairwise.enabled: false`
- Generator: `gemini-2.5-flash-lite`
- Solver: `gemini-2.5-flash-lite`
- Verifier: `gemini-2.5-flash`
- Verifier repeats: 3 per solution
- Verifier token budget: 30,000
- Python-assisted verifier prompt: enabled
- Question bank examples: enabled

This run had 3 generated questions, 4 solver candidates per question, and 12
total solution-verifier comparisons.

## Current Tiny Run Results

The fresh current-version oracle comparison produced:

- Verifier/oracle agreement: 5 / 12 = 41.7%
- Majority bucket coverage: all 12 examples were unanimous 3/3
- 3/3 oracle agreement in this run: 5 / 12 = 41.7%
- Question rewards: 0.0 for all three questions
- Solver advantages: 0.0 for every completion

The low agreement is not because the verifier was uncertain. All examples had
unanimous verifier votes. The failure mode is systematic disagreement between
the verifier pass used during rollout and the independent oracle pass used after
the rollout.

In plain terms, yes: the problematic behavior was that the rollout verifier
marked two entire questions as wrong. For one modular-exponent question, all
four solvers answered `12`; the independent oracle judged `12` correct, but the
rollout verifier voted `INCORRECT` on all four candidates by 3/3. For the second
question, the verifier again marked all four candidates `INCORRECT`, while the
oracle found two of the no-solution arguments mathematically correct but not
cleanly parseable as final answers. This makes the result a blocker rather than
a passing smoke test.

Per-question behavior:

- Question 0 asked for the smallest `n` satisfying two modular exponent
conditions modulo 13. All solvers answered `12`. The rollout verifier marked
all four candidates `INCORRECT` by 3/3 vote, while the oracle marked `12`
`CORRECT`.
- Question 1 asked for the smallest `n` such that `2^n ≡ 17 mod 23`. Two solver
outputs effectively answered that no such integer exists, but they lacked a
parseable final answer. The rollout verifier marked all candidates
`INCORRECT`; the oracle treated the no-solution reasoning as mathematically
correct for two candidates, one malformed candidate as `INVALID`, and one
`n=17` candidate as `INCORRECT`.
- Question 2 asked for the smallest Armstrong-style integer divisible by 7. All
solvers answered `371`; both rollout verifier and oracle agreed these were
`CORRECT`.

The reward consequence is important. Because each question was all-correct or
all-incorrect after majority voting, the within-question GRPO advantages were
all zero. This is expected mathematically: if every solution in a group receives
the same scalar score, mean-centered advantages provide no solver learning
signal.

## Majority Voting Finding

The current tiny run cannot directly answer whether stronger majority is better,
because every example landed in the same 3/3 bucket. It shows a negative result:
unanimity alone does not prevent systematic verifier error.

Historical artifacts with oracle labels do show the expected trend:

- 3/3 agreement: 3,324 examples, 85.4% oracle agreement
- 1/3 majority margin, meaning 2-1 split: 188 examples, 48.4% oracle agreement
- 1/1 single vote: 530 examples, 76.8% oracle agreement
- 2/2 two-vote unanimity: 145 examples, 82.1% oracle agreement

Interpretation: vote agreement is a useful reliability signal, and unanimous
verdicts are generally more trustworthy than split verdicts. But the tiny run
demonstrates that correlated verifier errors remain possible, especially when
the same verifier model and prompt are repeated on questions that induce a
consistent misconception.

## Implementation Corrections Made During This Check

Several verifier-path issues were found and fixed or instrumented.

First, one verifier prompt bug was found and fixed. The prompt renderer was still
extracting only integer-looking answers from `FINAL_ANSWER`, so an answer like
`35/396` was passed to the verifier as `35`. This conflicted with the new design
where final answers are not restricted to integers.

The fix changes verifier prompt rendering to preserve the full text after
`FINAL_ANSWER:`.

Second, the shared verifier prompt renderer was still using a separate
first-match regex, so integrated verifier/oracle paths could parse the first
`FINAL_ANSWER:` line instead of the final corrected answer. It now uses the same
last-full-line parser as rollout collection. An integrated test captures the
actual outgoing API prompt and verifies that a completion ending in
`FINAL_ANSWER: 35/396` is rendered as `Candidate answer: 35/396`, not `35`.

Third, the verifier remains answer-only by design: the prompt includes the
question, candidate final answer, and Python tool output, but not the candidate
solution reasoning.

Fourth, the Python-assisted verifier probe was too weak. It only extracted
integer final answers and printed the integers found in the question. It now
preserves the full `FINAL_ANSWER:` string and attempts to parse useful numeric
forms such as integers and fractions. This matters for non-integer final answers
like `35/396`.

Fifth, a new signal-health diagnostic was added:

- Script: `grpo_math/self_play/summarize_verifier_signal.py`
- Failed-run summary: `outputs/reports/samples_gemini25flashlite_single_verify_tiny_signal_summary.json`

This diagnostic reports majority buckets, high-agreement oracle accuracy,
question reward coverage, nonzero solver advantages, and distinct final answers
marked correct. On the failed Gemini tiny run, it correctly flags:

- high-agreement oracle accuracy: 41.7%
- nonzero question rewards: 0
- nonzero solver advantages: 0
- red flag: all questions zero reward
- red flag: all solver advantages zero

Regression tests were added, and the focused tests passed:

```bash
python3.10 -m pytest \
  tests/test_verifier.py \
  tests/test_generate_pairwise_data.py \
  tests/test_summarize_verifier_signal.py \
  -q
```

Result: 21 tests passed.

## Proposed Next Runs

I added three verifier-only configs for scaling this experiment using the model
stack intended for the actual loop: Together AI serving `Qwen/Qwen3.5-9B`.
All three explicitly disable pairwise/Elo and use 5 verifier repeats per
solution so we can observe stronger majority buckets such as 3/5, 4/5, and 5/5.

- Mini: `grpo_math/configs/pairwise_rollouts_qwen35_9b_together_single_verify_mini.yaml`
- Small: `grpo_math/configs/pairwise_rollouts_qwen35_9b_together_single_verify_small.yaml`
- Medium: `grpo_math/configs/pairwise_rollouts_qwen35_9b_together_single_verify_medium.yaml`

Planned sizes:

- Mini: 10 questions × 4 solutions × 5 verifier votes = 200 verifier calls
- Small: 30 questions × 6 solutions × 5 verifier votes = 900 verifier calls
- Medium: 100 questions × 8 solutions × 5 verifier votes = 4,000 verifier calls

All three use:

- Generator, solver, and verifier: `Qwen/Qwen3.5-9B`
- API provider: Together AI OpenAI-compatible endpoint
- API key environment variable: `TOGETHER_API_KEY`
- Verifier mode: `single_verify`
- Verifier temperature: 0.2
- Verifier token budget: 10,000
- Python-assisted verification: enabled
- Question bank prompting: enabled
- Pairwise/Elo: disabled
- Parallelism: generator 8-way, solver 24-way, verifier 32-way

Recommended command sequence:

```bash
python3.10 -m grpo_math.self_play.generate_pairwise_data \
  --config grpo_math/configs/pairwise_rollouts_qwen35_9b_together_single_verify_mini.yaml

python3.10 -m grpo_math.self_play.compare_verifier_to_oracle \
  --jsonl outputs/pairwise_rollouts_debug/samples_qwen35_9b_together_single_verify_mini.jsonl \
  --model gemini-3.1-pro-preview \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env GEMINI_API_KEY \
  --max-tokens-param max_tokens \
  --max-tokens 30000 \
  --max-parallel 8 \
  --python-assisted \
  --output-json outputs/reports/samples_qwen35_9b_together_single_verify_mini_gemini31_oracle_compare.json
```

Run the small and medium configs the same way after the mini run confirms that
vote buckets are non-degenerate and the cost/runtime is acceptable.

The mini run should be considered a pass only if it clears these gates:

- high-agreement verifier/oracle agreement is at least 80%
- there is at least one mixed-verdict question
- at least one question has nonzero `question_reward`
- at least one solver completion has nonzero `solver_advantage`
- no question has multiple distinct final answers marked `CORRECT`
- parse failures are low enough that `INVALID`/`NONE` answers are not dominating

The command for the post-run signal check is:

```bash
python3.10 -m grpo_math.self_play.summarize_verifier_signal \
  --jsonl outputs/pairwise_rollouts_debug/samples_qwen35_9b_together_single_verify_mini.jsonl \
  --oracle-json outputs/reports/samples_qwen35_9b_together_single_verify_mini_gemini31_oracle_compare.json \
  --output-json outputs/reports/samples_qwen35_9b_together_single_verify_mini_signal_summary.json
```

Markdown exports are written with `export_rollout_readmes.py`. The “Full
continuation” block is the solver completion stored in `sol["text"]`. For the
Qwen/Together solver configs, thinking capture is enabled and the stored
completion includes both sections when Together returns them:

```text
[Thinking]
...

[Final]
...
FINAL_ANSWER: <final answer>
```

Generator and verifier/judge calls keep thinking disabled so question parsing
and verdict parsing stay clean.

## Recommendations Before Treating This as Training Data

First, use majority voting as a filter, not as truth. The historical data
supports 5/5 or 3/3 unanimity as a better signal than split votes, but the
current run shows that repeated calls can share the same blind spot.

Second, add an oracle-free distinct-answer diagnostic. Questions where the
verifier marks multiple distinct final answers as `CORRECT` should be filtered
or downweighted, because this often indicates the verifier is grading rhetoric
instead of answer validity.

Third, target questions that produce mixed solver outcomes. All-correct and
all-incorrect groups produce zero mean-centered solver advantage, so they are
not useful for GRPO updates even if the verifier is correct.

Fourth, keep the independent oracle comparison for experimental reporting. For
actual self-play training we want oracle-free filtering, but for the professor
report and method validation we need oracle-labeled audits to quantify verifier
quality.

## Bottom Line

The verifier-only loop is the correct path for this version, and Elo is disabled
in the new configs. But the Gemini smoke result is not accurate enough to train
on. Majority voting appears valuable in aggregate, but the latest run shows that
majority strength must be combined with oracle-free filters and periodic oracle
audits. The next empirical step is a Qwen/Together mini run, followed by small
and medium runs if the mini run shows a healthy spread of majority buckets,
reasonable oracle agreement, and nonzero solver advantages.