# Pairwise Rollout Report

- JSONL: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/pairwise_rollouts_debug/samples_gpt41_pairwise_10q_oracle.jsonl`
- Config: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/grpo_math/configs/pairwise_rollouts_gpt41_pairwise_10q_oracle.yaml`
- READMEs: `/Users/dylanwilson/Documents/GitHub/self-play/Untitled/outputs/readme_exports/samples_gpt41_pairwise_10q_oracle`
- run_id: `20260301T013355Z`
- ranking.method: `elo`

## Preference stability

- n: `10`  mean: `0.773`  std: `0.066`  min: `0.667`  max: `0.889`

## Pairs per question

- n: `10`  mean: `15`  std: `0`  min: `15`  max: `15`

## R_sep (Elo)

- n: `10`  mean: `10.64`  std: `36.88`  min: `-54.92`  max: `51.42`

## R_sep (cross-group win-rate)

- n: `10`  mean: `0.533`  std: `0.117`  min: `0.333`  max: `0.667`

## Group 0 mean Elo

- n: `10`  mean: `1005.32`  std: `18.44`  min: `972.54`  max: `1025.71`

## Group 1 mean Elo

- n: `10`  mean: `994.68`  std: `18.44`  min: `974.29`  max: `1027.46`

## Oracle acc: any solution correct

- n: `10`  mean: `0.900`  std: `0.316`  min: `0.000`  max: `1.000`

## Oracle acc: group 0 (mean)

- n: `10`  mean: `0.667`  std: `0.416`  min: `0.000`  max: `1.000`

## Oracle acc: group 1 (mean)

- n: `10`  mean: `0.667`  std: `0.314`  min: `0.000`  max: `1.000`

## Oracle acc: best-by-Elo correct

- n: `10`  mean: `0.800`  std: `0.422`  min: `0.000`  max: `1.000`

## Oracle pref acc (macro over questions)

- n: `7`  mean: `0.464`  std: `0.252`  min: `0.074`  max: `0.867`

## Oracle-informative pairs per question

- n: `10`  mean: `4.20`  std: `3.22`  min: `0.00`  max: `9.00`

## Oracle preference accuracy (micro)

- acc_micro: `0.421`  correct_votes: `53`  total_votes: `126`

## Per-question breakdown


| q_idx | r_sep_elo | g0_elo  | g1_elo  | winrate_g0_vs_g1 | pref_stability | num_pairs | oracle_any_ok | oracle_best_by_elo_ok | oracle_g0_acc | oracle_g1_acc | oracle_pref_acc | oracle_pref_votes |
| ----- | --------- | ------- | ------- | ---------------- | -------------- | --------- | ------------- | --------------------- | ------------- | ------------- | --------------- | ----------------- |
| 0     | 29.09     | 1014.54 | 985.46  | 0.593            | 0.778          | 15        | 1             | 1                     | 0.667         | 0.667         | 0.375           | 24                |
| 1     | -3.78     | 998.11  | 1001.89 | 0.481            | 0.667          | 15        | 1             | 1                     | 1.000         | 1.000         | NA              | 0                 |
| 2     | -29.64    | 985.18  | 1014.82 | 0.407            | 0.800          | 15        | 1             | 1                     | 0.000         | 0.333         | 0.867           | 15                |
| 3     | 51.42     | 1025.71 | 974.29  | 0.667            | 0.733          | 15        | 1             | 1                     | 1.000         | 0.667         | 0.467           | 15                |
| 4     | 17.32     | 1008.66 | 991.34  | 0.556            | 0.778          | 15        | 1             | 1                     | 1.000         | 0.667         | 0.333           | 15                |
| 5     | -25.30    | 987.35  | 1012.65 | 0.407            | 0.822          | 15        | 1             | 1                     | 0.667         | 1.000         | 0.467           | 15                |
| 6     | 45.28     | 1022.64 | 977.36  | 0.630            | 0.689          | 15        | 1             | 1                     | 1.000         | 0.667         | 0.667           | 15                |
| 7     | -54.92    | 972.54  | 1027.46 | 0.333            | 0.822          | 15        | 1             | 1                     | 1.000         | 1.000         | NA              | 0                 |
| 8     | 42.21     | 1021.10 | 978.90  | 0.630            | 0.756          | 15        | 0             | 0                     | 0.000         | 0.000         | NA              | 0                 |
| 9     | 34.70     | 1017.35 | 982.65  | 0.630            | 0.889          | 15        | 1             | 0                     | 0.333         | 0.667         | 0.074           | 27                |


