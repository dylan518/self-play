# Self Play

This repo contains **Math GRPO v0** scaffolding for training on **GSM8K** with a **binary reward** and **KL penalty to a frozen reference**.

## Quickstart

### Install

```bash
cd /home/ubuntu/self-play
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Train with Hugging Face TRL (recommended)

This uses TRL’s built-in `GRPOTrainer` (less custom RL code to debug).

```bash
cd /home/ubuntu/self-play
source .venv/bin/activate

accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file grpo_math/configs/deepspeed_zero2.json \
  -m grpo_math.trl.train_grpo_trl \
  --config grpo_math/configs/train_gsm8kv2_trl.yaml
```

### Config knobs (v0)

- **`rollout.k`**: K samples per prompt (default 16)
- **`train.prompts_per_step`**: B prompts per step (default 64)
- **`train.kl_beta`**: KL penalty β (default 0.02)
- **`train.lr`**: RL-ish LR (default 2e-6)
- **`rollout.max_new_tokens`**: generation length (default 512)

### Eval

```bash
accelerate launch \
  --num_processes 8 \
  -m grpo_math.eval.eval_gsm8k \
  --config grpo_math/configs/train_gsm8kv2_trl.yaml \
  --checkpoint outputs/trl_grpo_v0/checkpoint-200
```

## Repo layout

```
grpo_math/
  configs/
    deepspeed_zero3.json
    train_gsm8kv2_trl.yaml
  data/
    gsm8k.py
    reward.py
  models/
    policy.py
  eval/
    eval_gsm8k.py
  utils/
    metrics.py
```

