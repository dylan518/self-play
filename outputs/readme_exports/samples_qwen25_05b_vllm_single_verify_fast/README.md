# Rollout Export

- Source JSONL: `outputs/pairwise_rollouts_debug/samples_qwen25_05b_vllm_single_verify_fast.jsonl`
- Config: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/grpo_math/configs/pairwise_rollouts_qwen25_05b_vllm_single_verify_fast.yaml`
- Questions exported: `5`

## Per-Question Files

- `question_000.md`: Given the function \( f(x) = 3x^2 - 4x + 5 \), determine the domain of this function over the interval \([-1, 3]\).
- `question_001.md`: What is the smallest positive integer \( n \) such that \( 2^n - 1 \) is divisible by \( 3 \)?
- `question_002.md`: What is the smallest positive integer greater than 500 that leaves a remainder of 1 when divided by any prime number les...
- `question_003.md`: Solve for x in 2x^2 - 5x + 1 = 0.
- `question_004.md`: What is the positive difference between the largest and smallest prime numbers less than 100?

Each per-question file includes full question text, prompts used, full solution continuations, and judge traces.
