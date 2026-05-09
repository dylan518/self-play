# Self-Play GRPO Loop Smoke Report

## Status

Blocked before rollout generation. No `cycle_001_samples.jsonl` or `cycle_002_samples.jsonl`
was produced, so the rollout analyzer could not be run and there are no per-cycle solver
metrics to quote.

## Requested run

- Branch: `orch/self-play-rl/rl-loop-smoke`
- Remote execution: SLURM two-GPU allocation
- vLLM command attempted on visible GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization 0.45
```

- GRPO loop command was staged for visible GPU 0 but did not start because vLLM never
  became healthy.

## What failed

The remote workspace path existed but was empty, so I initialized it from the pushed branch
before running. The configured SLURM GRES value allocated only one visible GPU, so I used a
single two-GPU allocation (`gpu:2`) to match the required GPU 1 vLLM / GPU 0 trainer layout.

The first submitted wrapper exited immediately during conda activation and did not leave a
running job. After replacing the wrapper and resubmitting, the vLLM process exited before
serving `127.0.0.1:8001`. The relevant failure was:

```text
WARNING ... No model architectures are specified
TypeError: 'NoneType' object is not iterable
```

A targeted diagnostic in the same remote conda environment showed:

```text
transformers 4.51.3
vllm 0.8.5
AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").architectures == None
```

Because the task required the exact report5 vLLM serve command and explicitly said to stop
if vLLM fails to start, I did not proceed with command-line overrides or alternate model
serving settings.

## Cycle metrics

| Cycle | samples JSONL | parse rate | oracle solve rate | R_sep mean | R_sep positive fraction | verifier accuracy | verifier precision | verifier recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cycle_001 | not produced | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cycle_002 | not produced | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Training metrics

No GRPO training process started, so there are no observed values for:

- `grpo/train/loss`
- `grpo/train/grad_norm`
- `grpo/train/reward`
- `grpo/train/reward_std`
- `grpo/train/kl`
- `grpo/train/entropy`

## Job cleanup

After the vLLM failure, `squeue` showed no active jobs for this run.

## Assessment

RL is inconclusive at this scale because the run never reached cycle 1 rollout generation:
the exact local-vLLM Llama-3.1-8B serve command failed before health check, so the committed
evidence contains no parse rate, oracle solve rate, R_sep, verifier-vs-oracle, or GRPO
training metrics. The next step is to repair the remote vLLM/model-config environment, then
rerun the same two-cycle loop and analyze the generated JSONLs.
