#!/bin/bash
cd /home/nvidia
PY=/home/nvidia/venvs/rzero2/bin/python
MAIN=/home/nvidia/venvs/rzero2/lib/python3.12/site-packages/verl/trainer/main_ppo.py
export CUDA_HOME=/home/nvidia/venvs/rzero2/lib/python3.12/site-packages/nvidia/cu13
export PATH=/home/nvidia/venvs/rzero2/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/software/software/Python/3.10.15-system/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_COMPILE_CACHE=1 TOKENIZERS_PARALLELISM=false VERL_FORCE_TEXT_ONLY=1
PQ=/data/selfplay/cont_it2_cvband_verl08.parquet
MODEL=/home/nvidia/rzero_run/models/rzcev_it1_verified_s2/global_step_20/actor/huggingface
DIR=/home/nvidia/rzero_run/models/cont_it2_cvband_s
echo "=== CVBAND TRAIN START $(date) (iter1 solver on 553 cvband rows) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY $MAIN \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  data.prompt_key=prompt data.max_prompt_length=2048 data.filter_overlong_prompts=true data.shuffle=true \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=1e-2 actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.99 \
  data.train_files=$PQ data.val_files=$PQ data.max_response_length=4096 \
  data.train_batch_size=256 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.max_model_len=6144 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.model.path=$MODEL \
  reward.reward_manager.source=register reward.reward_manager.name=naive \
  reward.custom_reward_function.path=/home/nvidia/R-Zero/examples/reward_function/math_verl08_single.py reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 trainer.test_freq=-1 trainer.max_actor_ckpt_to_keep=1 trainer.val_before_train=false \
  trainer.logger=[console,wandb] trainer.log_val_generations=20 trainer.project_name=rzero_run trainer.total_epochs=1000 \
  trainer.total_training_steps=20 trainer.save_freq=20 trainer.n_gpus_per_node=4 \
  trainer.experiment_name=cont_it2_cvband_s trainer.default_local_dir=$DIR \
  trainer.rollout_data_dir=/data/selfplay/rollout_dumps/cont_it2_cvband_s 2>&1
echo "=== CVBAND TRAIN rc=$? $(date) ==="
