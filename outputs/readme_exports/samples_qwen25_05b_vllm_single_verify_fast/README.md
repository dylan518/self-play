# Rollout Export

- Source JSONL: `outputs/pairwise_rollouts_debug/samples_qwen25_05b_vllm_single_verify_fast.jsonl`
- Config: `/home/ubuntu/self-play/grpo_math/configs/pairwise_rollouts_qwen25_05b_vllm_single_verify_fast.yaml`
- Questions exported: `5`

## Per-Question Files

- `question_000.md`: Find the sum of all positive integers less than 100 whose digits add up to 9.
- `question_001.md`: Given an equation involving three unknowns \(a\), \(b\), and \(c\) such that \(2a + b - c = 7\) and \(3b - c = 10\), fin...
- `question_002.md`: Find the positive integer \( x \) such that \( 2^x + 3^x = 100 \).
- `question_003.md`: A rectangular garden has a length of 10 meters and a width of 6 meters. What is the area of the garden?
- `question_004.md`: What is the value of x if 2x + 3 = 17?

Each per-question file includes full question text, prompts used, full solution continuations, and judge traces.
