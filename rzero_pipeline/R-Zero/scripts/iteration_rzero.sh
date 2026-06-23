#!/bin/bash
# M5/M6: ONE full self-play iteration on verl 0.8 + Qwen3.5-4B (one arm).
# questioner-train(challenger reward via service) -> HF -> generate -> evaluate -> band-filter
#   -> [verified: judge] -> convert -> solver-train(naive math reward) -> HF.
# Inputs(env): ITER EXP ARM(verified|majority) QUESTIONER_MODEL SOLVER_MODEL VERIFY_WEIGHT
#   NUM_SAMPLES EVAL_N Q_STEPS S_STEPS Q_NPROMPTS  SKIP_QTRAIN SKIP_STRAIN
# Outputs(stdout): QUESTIONER_HF=<path>  SOLVER_HF=<path>  (latest merged HF dirs)
set -uo pipefail
RZERO=$HOME/R-Zero
export STORAGE_PATH=${STORAGE_PATH:-$HOME/rzero_run}
# Portable interpreter: defaults to Brev venv; WashU sets PY/MAIN to its conda env.
PY=${PY:-$HOME/venvs/rzero2/bin/python}
MAIN=${MAIN:-$HOME/venvs/rzero2/lib/python3.12/site-packages/verl/trainer/main_ppo.py}
# Brev-only libpython/cuda paths (harmless no-ops elsewhere; override CUDA_HOME="" to skip).
export CUDA_HOME=${CUDA_HOME-$HOME/venvs/rzero2/lib/python3.12/site-packages/nvidia/cu13}
[ -n "$CUDA_HOME" ] && export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${EXTRA_LD:-/mnt/home/software/software/Python/3.10.15-system/lib}:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_COMPILE_CACHE=1
export WORKER_ENFORCE_EAGER=${WORKER_ENFORCE_EAGER:-0}   # 0=CUDA graphs (faster) for service/gen/eval/judge; training rollout stays eager via COMMON
export TOKENIZERS_PARALLELISM=false
export GEN_MAX_MODEL_LEN=${GEN_MAX_MODEL_LEN:-8192}   # prompt + 4096 gen must fit (was 4096 -> truncated questions)
EXP=${EXP:-run}; ARM=${ARM:-verified}; ITER=${ITER:-1}
# Questioner prompt by arm: verified -> verifiable (integer/program-checkable, train==gen);
# majority -> standard R-Zero prompt. Honored by make_q_prompts.py AND question_generate.py.
if [ "$ARM" = "verified" ]; then export QUESTIONER_PROMPT=$HOME/R-Zero/examples/format_prompt/questioner_verifiable.jinja
else export QUESTIONER_PROMPT=$HOME/R-Zero/examples/format_prompt/questioner.jinja; fi
QUESTIONER_MODEL=${QUESTIONER_MODEL:-Qwen/Qwen3.5-4B}
SOLVER_MODEL=${SOLVER_MODEL:-Qwen/Qwen3.5-4B}
VERIFY_WEIGHT=${VERIFY_WEIGHT:-0.75}                       # graded verifiability term, worth a bit more
DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-0.5}                  # Vendi marginal diversity reward weight
VENDI_LANDMARKS=${VENDI_LANDMARKS:-512}
VERIFY_SUBSAMPLE=${VERIFY_SUBSAMPLE:-1.0}                  # judge ALL questions so graded verif has within-group contrast
# R-Zero canonical defaults (examples/config.yaml + solver_train.sh + questioner_train_penalty.sh).
# Only the reward signal (program-verifier label) + verifier rollout (judge.py) differ from vanilla R-Zero.
NUM_SAMPLES=${NUM_SAMPLES:-1000}; EVAL_N=${EVAL_N:-9}      # gen 1000/gpu; band eval n=9
Q_STEPS=${Q_STEPS:-6}; S_STEPS=${S_STEPS:-20}; Q_NPROMPTS=${Q_NPROMPTS:-512}
# Questioner reward runs through ONE service GPU (we have 4 GPUs, not R-Zero's 8), so the
# questioner batch must be small or the reward call stalls. Solver batch stays canonical.
Q_TBATCH=${Q_TBATCH:-60}; Q_MINI=${Q_MINI:-12}            # 60 prompts x n4 = 240 questions/step (÷3 GPUs ok)
GEN_GPUS=${GEN_GPUS:-1,2,3}; SVC_GPU=${SVC_GPU:-0}
N_SVC=${N_SVC:-1}   # DP reward-service replicas (1=GPU0 only; 2=GPUs 0,1 with training on 2,3)
if [ "$N_SVC" -ge 2 ]; then SVC_GPUS="0 1"; Q_TRAIN_GPUS=${Q_TRAIN_GPUS:-2,3}; else SVC_GPUS="${SVC_GPU}"; Q_TRAIN_GPUS=${Q_TRAIN_GPUS:-1,2,3}; fi
S_TRAIN_GPUS=${S_TRAIN_GPUS:-0,1,2,3}                      # solver gets all 4 GPUs (no svc), TP=2
MINSC=${MINSC:-0.3}; MAXSC=${MAXSC:-0.8}
GG=(${GEN_GPUS//,/ })
NTRAIN_Q=$(($(echo $Q_TRAIN_GPUS | tr -cd , | wc -c)+1))
NTRAIN_S=$(($(echo $S_TRAIN_GPUS | tr -cd , | wc -c)+1))
TAG=${EXP}_it${ITER}_${ARM}
mkdir -p $STORAGE_PATH/generated_question $STORAGE_PATH/artifacts $STORAGE_PATH/models
log(){ echo "=== [$TAG] $* $(date +%H:%M:%S)"; }
latest_hf(){ ls -dt "$1"/global_step_*/actor/huggingface 2>/dev/null | head -1; }
BASE_HF_SNAP=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ 2>/dev/null | head -1)
fixup_hf(){  # verl save (text-only) omits VL processor files -> vllm cannot load. copy from base.
  local d="$1"; [ -d "$d" ] || return 0
  for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json processor_config.json; do
    [ -f "$BASE_HF_SNAP/$f" ] && [ ! -f "$d/$f" ] && cp "$BASE_HF_SNAP/$f" "$d/" 2>/dev/null
  done
}

# Mirrors R-Zero examples/config.yaml: lr 1e-6, KL low_var_kl coef 1e-2, max_grad_norm 1.0,
# max_prompt 2048, gpu_mem 0.7. (TP + batch + max_response set PER STAGE below.)
COMMON_TRAIN=(
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
  data.prompt_key=prompt data.max_prompt_length=2048
  data.filter_overlong_prompts=true data.shuffle=true
  +data.apply_chat_template_kwargs.enable_thinking=false
  actor_rollout_ref.model.trust_remote_code=true
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=1e-2
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_OFFLOAD:-true}
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OFFLOAD:-true}
  actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.99
  actor_rollout_ref.ref.fsdp_config.param_offload=${FSDP_OFFLOAD:-true}
  trainer.nnodes=1 trainer.test_freq=-1 trainer.max_actor_ckpt_to_keep=1
  trainer.val_before_train=false trainer.logger=[console,wandb] trainer.log_val_generations=20
  trainer.project_name=rzero_run trainer.total_epochs=1000
)

# ---------------- Stage A: questioner-train ----------------
QUESTIONER_HF=$QUESTIONER_MODEL
if [ "${SKIP_QTRAIN:-0}" != "1" ]; then
  log "A: questioner-train (arm=$ARM VERIFY_WEIGHT=$VERIFY_WEIGHT) — start service on GPU $SVC_GPU"
  QDATA=$STORAGE_PATH/generated_question/${EXP}_qprompts.parquet
  [ -f "$QDATA" ] || $PY $RZERO/pipeline/make_q_prompts.py --n $Q_NPROMPTS --out $QDATA
  RUN_ID=$(date +%s%N); export RUN_ID
  SVC_PIDS=(); _si=0
  for _sg in $SVC_GPUS; do
    CUDA_VISIBLE_DEVICES=$_sg SERVICE_MAX_MODEL_LEN=8192 SERVICE_GPU_MEM=0.85 VERIFY_SUBSAMPLE=$VERIFY_SUBSAMPLE \
      setsid $PY $RZERO/vllm_service_init/start_vllm_server.py --port $((5000+_si)) --model_path $SOLVER_MODEL > $STORAGE_PATH/svc_${TAG}_${_si}.log 2>&1 &
    SVC_PIDS+=($!); _si=$((_si+1))
  done
  for _p in $(seq 0 $((_si-1))); do for t in $(seq 1 80); do curl -s "http://127.0.0.1:$((5000+_p))/hello" -o /dev/null --max-time 3 && break; sleep 5; done; done
  log "A: service up; launching questioner GRPO"
  QCKPT=$STORAGE_PATH/models/${TAG}_q
  # R-Zero questioner: rollout.n=4, gbs 16, max_resp 4096, 6 steps. On 3 train GPUs use 504/24 (~512/16, TP=1).
  ( cd $HOME && PYTHONPATH= CUDA_VISIBLE_DEVICES=$Q_TRAIN_GPUS N_SERVICES=$N_SVC SERVICE_PORT_BASE=5000 VERIFY_WEIGHT=$VERIFY_WEIGHT \
      DIVERSITY_WEIGHT=$DIVERSITY_WEIGHT VENDI_LANDMARKS=$VENDI_LANDMARKS \
      ARTIFACTS_DIR=$STORAGE_PATH/artifacts/$TAG REWARD_COALESCE_DEBOUNCE=0.8 VERL_FORCE_TEXT_ONLY=1 \
      $PY $MAIN "${COMMON_TRAIN[@]}" \
      data.train_files=$QDATA data.val_files=$QDATA data.max_response_length=2048 \
      data.train_batch_size=$Q_TBATCH actor_rollout_ref.actor.ppo_mini_batch_size=$Q_MINI \
      actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.max_model_len=4096 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=${Q_GPU_MEM:-0.5} \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
      actor_rollout_ref.model.path=$QUESTIONER_MODEL \
      reward.reward_manager.source=importlib \
      reward.reward_manager.module.path=$RZERO/examples/reward_function/caller_reward_manager_verl08.py \
      reward.reward_manager.name=CallerBatchManager \
      reward.custom_reward_function.path=$RZERO/examples/reward_function/caller_penalty_verl08.py \
      reward.custom_reward_function.name=compute_score reward.num_workers=1 \
      trainer.total_training_steps=$Q_STEPS trainer.save_freq=$Q_STEPS trainer.n_gpus_per_node=$NTRAIN_Q \
      trainer.experiment_name=${TAG}_q trainer.default_local_dir=$QCKPT trainer.rollout_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_q trainer.validation_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_q_val ) 2>&1
  QRC=$?
  for _sp in "${SVC_PIDS[@]}"; do kill -- -$_sp 2>/dev/null; kill -9 -- -$_sp 2>/dev/null; done; sleep 10
  H=$(latest_hf $QCKPT); if [ -n "$H" ]; then QUESTIONER_HF=$H; fixup_hf "$QUESTIONER_HF"; find $QCKPT -name "*.pt" -delete 2>/dev/null; log "A done -> $QUESTIONER_HF"; else log "A FAILED rc=$QRC (no HF); using base questioner"; fi
fi

# ---------------- Stage B: data pipeline ----------------
log "B: generate ($NUM_SAMPLES x ${#GG[@]} gpus) with questioner=$QUESTIONER_HF"
export PYTHONPATH=$RZERO
pids=(); for idx in "${!GG[@]}"; do CUDA_VISIBLE_DEVICES=${GG[$idx]} $PY $RZERO/question_generate/question_generate.py --model $QUESTIONER_HF --suffix $idx --num_samples $NUM_SAMPLES --save_name $TAG & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
log "B: evaluate (solver=$SOLVER_MODEL n=$EVAL_N)"
pids=(); for idx in "${!GG[@]}"; do CUDA_VISIBLE_DEVICES=${GG[$idx]} $PY $RZERO/question_evaluate/evaluate.py --model $SOLVER_MODEL --suffix $idx --num_samples $EVAL_N --save_name $TAG & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
log "B: band-filter $MINSC-$MAXSC"
$PY $RZERO/question_evaluate/upload.py --repo_name $TAG --experiment_name $TAG --min_score $MINSC --max_score $MAXSC || true
[ -f $STORAGE_PATH/generated_question/${TAG}.parquet ] || { log "B: NO parquet (all filtered) — abort iter"; echo "QUESTIONER_HF=$QUESTIONER_HF"; echo "SOLVER_HF=$SOLVER_MODEL"; exit 3; }
TRAIN_PARQUET=$STORAGE_PATH/generated_question/${TAG}.parquet
if [ "$ARM" = "verified" ]; then
  log "B: judge program-consensus"
  CUDA_VISIBLE_DEVICES=${GG[0]} $PY $RZERO/question_evaluate/judge.py --model $SOLVER_MODEL \
    --in_parquet $STORAGE_PATH/generated_question/${TAG}.parquet \
    --out_parquet $STORAGE_PATH/generated_question/${TAG}_verified.parquet \
    --report_jsonl $STORAGE_PATH/generated_question/${TAG}_judge.jsonl || true
  TRAIN_PARQUET=$STORAGE_PATH/generated_question/${TAG}_verified.parquet
fi
# --- correctness-variance band fix (CVBAND=1): keep solve-rate-vs-program-label in (CVLO,CVHI) ---
if [ "${CVBAND:-0}" = "1" ] && [ -f $STORAGE_PATH/artifacts/$TAG/all_questions.jsonl ]; then
  log "B: correctness-variance band CVLO=${CVLO:-0.2} CVHI=${CVHI:-0.8}"
  $PY $RZERO/question_evaluate/correctness_band.py --verified_parquet $TRAIN_PARQUET --all_questions $STORAGE_PATH/artifacts/$TAG/all_questions.jsonl --out_parquet $STORAGE_PATH/generated_question/${TAG}_cvband.parquet --lo ${CVLO:-0.2} --hi ${CVHI:-0.8} && TRAIN_PARQUET=$STORAGE_PATH/generated_question/${TAG}_cvband.parquet
fi
VERL_PARQUET=$STORAGE_PATH/generated_question/${TAG}_verl08.parquet
$PY $RZERO/pipeline/to_verl08_parquet.py --in_parquet $TRAIN_PARQUET --out_parquet $VERL_PARQUET --data_source ${TAG}
NROWS=$($PY -c "import pandas as pd; print(len(pd.read_parquet('$VERL_PARQUET')))" 2>/dev/null || echo 0)
log "B done: $NROWS solver-training rows -> $VERL_PARQUET"

# ---------------- Stage C: solver-train ----------------
SOLVER_HF=$SOLVER_MODEL
if [ "${SKIP_STRAIN:-0}" != "1" ] && [ "$NROWS" -ge 8 ]; then
  log "C: solver-train ($S_STEPS steps) on $NROWS rows"
  SCKPT=$STORAGE_PATH/models/${TAG}_s
  # R-Zero solver: rollout_batch 256, global_batch 128, n=5, max_resp 4096, 20 steps. Cap to NROWS if scarce.
  if [ "$NROWS" -ge 256 ]; then SBATCH=256; SMINI=128
  elif [ "$NROWS" -ge 128 ]; then SBATCH=128; SMINI=64
  elif [ "$NROWS" -ge 64 ]; then SBATCH=64; SMINI=32
  else SBATCH=$(( NROWS/8*8 )); [ "$SBATCH" -lt 8 ] && SBATCH=8; SMINI=$SBATCH; fi
  log "C: train_batch=$SBATCH mini=$SMINI n=5 max_resp=4096 (NROWS=$NROWS)"
  ( cd $HOME && PYTHONPATH= CUDA_VISIBLE_DEVICES=$S_TRAIN_GPUS VERL_FORCE_TEXT_ONLY=1 \
      $PY $MAIN "${COMMON_TRAIN[@]}" \
      data.train_files=$VERL_PARQUET data.val_files=$VERL_PARQUET data.max_response_length=4096 \
      data.train_batch_size=$SBATCH actor_rollout_ref.actor.ppo_mini_batch_size=$SMINI \
      actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.max_model_len=6144 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
      actor_rollout_ref.rollout.gpu_memory_utilization=${S_GPU_MEM:-0.7} \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${S_TOKLEN:-12288} \
      actor_rollout_ref.model.path=$SOLVER_MODEL \
      reward.reward_manager.source=register reward.reward_manager.name=naive \
      reward.custom_reward_function.path=$RZERO/examples/reward_function/math_verl08_single.py \
      reward.custom_reward_function.name=compute_score \
      trainer.total_training_steps=$S_STEPS trainer.save_freq=$S_STEPS trainer.n_gpus_per_node=$NTRAIN_S \
      trainer.experiment_name=${TAG}_s trainer.default_local_dir=$SCKPT trainer.rollout_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_s trainer.validation_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_s_val ) 2>&1
  H=$(latest_hf $SCKPT); [ -n "$H" ] && SOLVER_HF=$H && fixup_hf "$SOLVER_HF" && { find $SCKPT -name "*.pt" -delete 2>/dev/null; log "C done -> $SOLVER_HF"; }
else
  log "C: skipped (rows=$NROWS)"
fi
echo "QUESTIONER_HF=$QUESTIONER_HF"
echo "SOLVER_HF=$SOLVER_HF"
log "ITERATION COMPLETE"
