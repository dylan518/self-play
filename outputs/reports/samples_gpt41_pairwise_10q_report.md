# Pairwise Rollout Report

- JSONL: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/pairwise_rollouts_debug/samples_gpt41_pairwise_10q.jsonl`
- Config: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/grpo_math/configs/pairwise_rollouts_gpt41_pairwise_10q.yaml`
- READMEs: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/readme_exports/samples_gpt41_pairwise_10q`
- run_id: `20260301T011603Z`
- ranking.method: `elo`

## Preference stability

- n: `10`  mean: `0.782`  std: `0.077`  min: `0.667`  max: `0.889`

## Pairs per question

- n: `10`  mean: `15`  std: `0`  min: `15`  max: `15`

## R_sep (Elo)

- n: `10`  mean: `-12.93`  std: `40.50`  min: `-84.83`  max: `51.63`

## R_sep (cross-group win-rate)

- n: `10`  mean: `0.467`  std: `0.121`  min: `0.259`  max: `0.667`

## Group 0 mean Elo

- n: `10`  mean: `993.54`  std: `20.25`  min: `957.59`  max: `1025.81`

## Group 1 mean Elo

- n: `10`  mean: `1006.46`  std: `20.25`  min: `974.19`  max: `1042.41`

## Per-question breakdown


| q_idx | r_sep_elo | g0_elo  | g1_elo  | winrate_g0_vs_g1 | pref_stability | num_pairs |
| ----- | --------- | ------- | ------- | ---------------- | -------------- | --------- |
| 0     | 51.63     | 1025.81 | 974.19  | 0.667            | 0.733          | 15        |
| 1     | -14.19    | 992.91  | 1007.09 | 0.444            | 0.889          | 15        |
| 2     | -11.03    | 994.49  | 1005.51 | 0.481            | 0.844          | 15        |
| 3     | -9.08     | 995.46  | 1004.54 | 0.481            | 0.711          | 15        |
| 4     | 43.75     | 1021.87 | 978.13  | 0.630            | 0.711          | 15        |
| 5     | -84.83    | 957.59  | 1042.41 | 0.259            | 0.756          | 15        |
| 6     | -7.43     | 996.29  | 1003.71 | 0.481            | 0.844          | 15        |
| 7     | -50.41    | 974.80  | 1025.20 | 0.333            | 0.867          | 15        |
| 8     | -7.73     | 996.14  | 1003.86 | 0.481            | 0.667          | 15        |
| 9     | -39.98    | 980.01  | 1019.99 | 0.407            | 0.800          | 15        |
