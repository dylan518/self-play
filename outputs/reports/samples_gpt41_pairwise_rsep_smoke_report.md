# Pairwise Rollout Report

- JSONL: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/pairwise_rollouts_debug/samples_gpt41_pairwise_rsep_smoke.jsonl`
- Config: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/grpo_math/configs/pairwise_rollouts_gpt41_pairwise_rsep_smoke.yaml`
- READMEs: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/readme_exports/samples_gpt41_pairwise_rsep_smoke`
- run_id: `20260301T010056Z`
- ranking.method: `elo`

## Preference stability

- n: `1`  mean: `0.800`  std: `0.000`  min: `0.800`  max: `0.800`

## Pairs per question

- n: `1`  mean: `15`  std: `0`  min: `15`  max: `15`

## R_sep (Elo)

- n: `1`  mean: `-22.86`  std: `0.00`  min: `-22.86`  max: `-22.86`

## R_sep (cross-group win-rate)

- n: `1`  mean: `0.444`  std: `0.000`  min: `0.444`  max: `0.444`

## Group 0 mean Elo

- n: `1`  mean: `988.57`  std: `0.00`  min: `988.57`  max: `988.57`

## Group 1 mean Elo

- n: `1`  mean: `1011.43`  std: `0.00`  min: `1011.43`  max: `1011.43`

## Per-question breakdown


| q_idx | r_sep_elo | g0_elo | g1_elo  | winrate_g0_vs_g1 | pref_stability | num_pairs |
| ----- | --------- | ------ | ------- | ---------------- | -------------- | --------- |
| 0     | -22.86    | 988.57 | 1011.43 | 0.444            | 0.800          | 15        |
