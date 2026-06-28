# ARCHITECTURE.md — Hyper-detailed system reference

> **Purpose.** Exhaustive, code-verified description of how the project *actually runs today* — every reward formula, parsing regex, config value, control-flow branch, and the **footguns** that have silently corrupted experiments. Read [Footguns](#9-footguns--silent-failures) first.
>
> Last full **code-verified** recount: **2026-06-25** (every section below was checked line-by-line against the running source on Brev; line numbers are `file:NN`). **What runs now: the R-Zero pipeline (System B).** The local `grpo_math/` pairwise system is legacy — [appendix](#a-appendix-legacy-local-pairwise-self-play-grpo_math).

## 0. Current setup snapshot

Active experiment = R-Zero self-play. Driver `scripts/iteration_rzero.sh` (one iteration) wrapped by `scripts/run_rzero.sh` (N-iter loop) or a per-run launcher (`run_it1redo.sh`). Code on Brev `/home/nvidia/R-Zero/`, mirrored in this repo's `rzero_pipeline/`.

- **Models:** Qwen3.5-4B base (`model_type: qwen3_5`) for both roles. NOT Qwen3-4B (§9.2).
- **Trainer/env:** verl 0.8 `main_ppo.py` (GRPO); **vllm 0.23.0, transformers 5.12.1**, venv `~/venvs/rzero2` (py3.12, CUDA13). Qwen3.5 is **hybrid mamba-attention** → `enforce_eager=True` everywhere (no cudagraph).
- **Kernel (current):** `attn_implementation=flash_attention_2` + `use_remove_padding=true` + `enable_gradient_checkpointing=true`, bf16. Pre-2026-06-22 runs used `sdpa`+`remove_padding=false`; that path **now crashes** on the upgraded env (FSDP-offload assertion + `mma.h`) — see §9.9.
- **Storage:** `STORAGE_PATH=/data/...` (16 TB; scripts honor it fully). Root `/` (499 GB) fills to ~94% → never put checkpoints there. GPUs 0–3 only (4–7 = `langlin`).
- **Logging:** `trainer.logger=[console,wandb]` (auth via `~/.netrc`), `rollout_data_dir`+`validation_data_dir` on both stages, `log_val_generations=20`. wandb `project_name=rzero_run`.
- **Answer format:** `\boxed{...}`, graded by `mathruler`. `enable_thinking=false` for both roles AND eval.

**Calibrated results (full-675 OlympiadBench, greedy pass@1, same harness):**
| model | base type | Oly 8k / 4k | MATH-500 4k |
|---|---|---|---|
| Qwen3.5-4B base | qwen3_5 | 0.573 / 0.502 | 0.858 |
| iter1 (verified-20) | qwen3_5 | **0.630** / 0.560 | 0.884 |
| iter2 (cont) | qwen3_5 | 0.545 / 0.533 | 0.864 |
| R-Zero v5 (ref) | **qwen3** | 0.304 / 0.299 | — |

Self-play **peaks at iter1, degrades by iter2** (U-shape), regardless of band filter (band-fix A/B refuted). The iter1→iter2 drop is in the **inputs**, not the code (an exact step-1 repro on iter1's inputs reproduced 0.554 vs original 0.565).

---

## 1. The self-play loop

Each **iteration** = three sequential phases; the two models do **not** train simultaneously. Co-evolution is **across iterations** (alternating optimization with a frozen opponent for a stable reward). The questioner's RL-phase questions are **thrown away** — only its weights carry forward; Phase B regenerates fresh questions.

```
seed prompts ─▶[A: train questioner; FIXED solver service scores it]─▶ trained questioner weights
trained questioner ─▶[B: generate 1000 fresh Qs → self-consistency band → program-judge label]─▶ parquet
parquet ─▶[C: train solver vs the program label; questioner frozen]─▶ trained solver ─▶ next iter's fixed solver
```

**`run_rzero.sh` (the N-iter driver, 33 lines):** defaults `ARM=verified VW=0.75 DW=0.5 MAX_ITER=5 EXP=rzc Q_STEPS=6 S_STEPS=20 NUM_SAMPLES=1000 EVAL_N=9 VERIFY_SUBSAMPLE=1.0`, `Q=S=Qwen/Qwen3.5-4B`. **`rm -f diversity_mem.npy` at start (L13)** → fresh Vendi bank per run. Loop: run `iteration_rzero.sh` → log → **chain by grepping stdout**: `NQ=grep "^QUESTIONER_HF=" | tail -1`, `NS=grep "^SOLVER_HF="`; `Q=$NQ S=$NS` for next iter (L23-26). Disk hygiene: delete this iter's `*.pt` shards, and `rm -rf` iter `it-2` model dirs (keep last 2). Final solver → `${ARM}_FINAL_solver`. **No `set -e`** — failed stages fall through to base-model fallbacks; the only hard abort is "no parquet" (`exit 3`).

---

## 2. Stage A — questioner (challenger) training (`iteration_rzero.sh:91-129`)

`Q_STEPS=6` GRPO steps. A standalone **vLLM reward service** (the fixed current solver) scores the questioner's questions each step.

**GPU layout (`:42-49`):** `N_SVC≥2` → service on GPUs 0,1 + train on 2,3; else (default `N_SVC=1`) service on GPU 0 + **train on 1,2,3** (`NTRAIN_Q=3`, TP=1). Solver (Stage C) uses 0,1,2,3 (TP=2).

**Seed prompts (`:95-96`):** `make_q_prompts.py --n Q_NPROMPTS=512 --out ${EXP}_qprompts.parquet` (seeded → identical md5 across runs). Prompt template from `QUESTIONER_PROMPT` (verified arm → `questioner_verifiable.jinja`).

**Reward service startup (`:97-104`):** for each `SVC_GPUS`, `setsid start_vllm_server.py --port $((5000+i)) --model_path $SOLVER_MODEL` with `SERVICE_MAX_MODEL_LEN=8192 SERVICE_GPU_MEM=0.85`. Readiness poll: up to **80×5s = 400s** curling `/hello`.

**Questioner GRPO (`:106-125`), exact verl args** (`COMMON_TRAIN` `:62-89` + overrides):
- `CUDA_VISIBLE_DEVICES=$Q_TRAIN_GPUS` (1,2,3), `PYTHONPATH=` cleared, `cd $HOME`.
- Reward env: `N_SERVICES=N_SVC SERVICE_PORT_BASE=5000 VERIFY_WEIGHT=0.75 DIVERSITY_WEIGHT=0.5 VENDI_LANDMARKS=512 ARTIFACTS_DIR=.../artifacts/$TAG REWARD_COALESCE_DEBOUNCE=0.8 VERL_FORCE_TEXT_ONLY=1`.
- `adv_estimator=grpo use_kl_in_reward=false`; `max_prompt_length=2048 filter_overlong_prompts=true shuffle=true +enable_thinking=false`.
- `attn_implementation=flash_attention_2 use_remove_padding=true enable_gradient_checkpointing=true` (bf16).
- `lr=1e-6 weight_decay=1e-2 grad_clip=1.0 use_dynamic_bsz=true use_kl_loss=true kl_loss_coef=1e-2 kl_loss_type=low_var_kl`; FSDP param+optimizer offload (`FSDP_OFFLOAD=true`).
- **Stage-A specifics:** `max_response_length=2048`, `train_batch_size=Q_TBATCH=60`, `ppo_mini_batch_size=Q_MINI=12`, `rollout.n=4`, `max_model_len=4096`, `tensor_model_parallel_size=1`, `gpu_memory_utilization=Q_GPU_MEM=0.5`, `ppo_max_token_len_per_gpu=6144`, `model.path=$QUESTIONER_MODEL`.
- **Reward = the verifier-service shim:** `reward_manager.source=importlib module.path=caller_reward_manager_verl08.py name=CallerBatchManager`; `custom_reward_function=caller_penalty_verl08.py:compute_score`; **`reward.num_workers=1`** (required, §3).
- `total_training_steps=6 save_freq=6 n_gpus_per_node=3 experiment_name=${TAG}_q rollout_data_dir=.../${TAG}_q validation_data_dir=.../${TAG}_q_val`.

**Teardown (`:127-128`):** kill service process-groups, `sleep 10`; `fixup_hf` copies `preprocessor_config.json, video_preprocessor_config.json, merges.txt, vocab.json, processor_config.json` from base into the saved ckpt (verl text-only save omits them → vLLM can't load otherwise — §9.8); **delete all `*.pt` shards** (keep `huggingface/`). On failure: log + reuse base questioner (graceful, no abort).

→ 60 prompts × n4 = **240 questions/step**.

---

## 3. Challenger reward — exact (`caller_penalty.py`, `_verl08.py`, `caller_reward_manager_verl08.py`, `diversity.py`, served by `start_vllm_server.py`)

### 3.1 The formula (`caller_penalty.py:167-178`)
Per question *i*:
```
uncertainty_i = min(score_i, 1 − score_i)      if parsed else −1
v_i           = verified_i  if not None  else  mean_verified(batch)
final_i       = uncertainty_i + DIVERSITY_WEIGHT·gate·novelty_i + VERIFY_WEIGHT·v_i
if CLIP_EASY > 0 and parsed and score_i > CLIP_EASY:  final_i = 0.0     # HARD zero, not clip-to-value
return {"overall": final_i, "format": 1 if parsed else 0, "accuracy": diversity_reward_i}
```
Constants: `VERIFY_WEIGHT=0.75`, `DIVERSITY_WEIGHT=0.5` (script; module default 0.4 — §9), `CLIP_EASY=0` (disabled by default; external-env override only — the script never sets it). **The `"accuracy"` field is overloaded to carry the diversity reward** (logging hack — anything reading "accuracy" here is misreading).

### 3.2 `score` = solver self-consistency, computed by the SERVICE (`start_vllm_server.py:185-247`)
The service samples the **fixed solver n=10 times** per question (`max_tokens=4096, temp=1.0, top_p=1.0, top_k=40`), extracts `\boxed{}` from each, clusters by `grade_answer` equivalence (10s timeout), `majority`=largest cluster: **`score = max_count / len(parseable_results)`**. This is **self-consistency (agreement), NOT correctness** — the generator's own answer and any ground truth are never used here. `uncertainty = min(score,1−score)` peaks at score=0.5 (solver maximally split) → the questioner is rewarded for questions at the solver's competence edge. Unparsed question → `score=−1` → `uncertainty=−1`.

### 3.3 `verified` ∈ {−1, −⅓, +⅓, +1} (`start_vllm_server.py:273-329`, gated on `VERIFY_SUBSAMPLE`)
Server default `VERIFY_SUBSAMPLE=0` (disabled → reward = uncertainty+diversity only), **but the script exports `VERIFY_SUBSAMPLE=1.0`** so ALL questions are judged. For each: generate `VERIFY_K_PROGRAMS=3` Python programs (`temp=0.6`), run each via `run_program` (plain `subprocess`, **no sandbox**, 25s timeout, must print an integer), `program_consensus` → `votes` (needs `MIN_AGREE=2`). **`verified = 2·votes/3 − 1`** → votes {0,1,2,3} → {−1,−⅓,+⅓,+1}. Unjudged items use `mean_verified` (neutral under GRPO's group baseline). (NOTE: this is the *challenger-reward* verified term, computed live by the service — distinct from Stage B's `judge.py` which produces the solver-training *label*; both call `program_consensus`.)

### 3.4 Diversity — Vendi/Nyström (`diversity.py`)
- Embeddings: `all-MiniLM-L6-v2` on **CPU**, mean-pooled, L2-norm, max_len 128.
- **Bank `diversity_mem.npy`** loaded each call, grows **unbounded across batches AND iterations** (the run's whole history). `run_rzero.sh` wipes it at run start (cold start → `vendi 1.0`).
- **Per-question novelty (`:140-155`):** landmarks `L=hist[even_idx(VENDI_LANDMARKS=512)]`, `W=LLᵀ+1e-3·I`, `quad_i=k_iᵀW⁻¹k_i` (residual against the bank's span), **`novelty_i=clip(1−quad_i,0,1)`** (dup→0, novel→1). Cold start (<2 in bank) → leave-self-out batch nearest-neighbor.
- **Golden gate (`:161-169`):** `VENDI_GOLDEN=85.6091` (MATH-500's Vendi over 500), `VENDI_GOLDEN_N=500`; when bank ≥500, `gate=clip((GOLDEN−vendi(bank))/GOLDEN,0,1)` → as the bank's diversity approaches the golden reference, gate→0 (anti-reward-hacking). Below 500 → `gate=1`.
- **diversity_reward = DIVERSITY_WEIGHT·gate·novelty ∈ [0, 0.5]**. (The legacy `diversity_penalties` NN function exists but is unused.)
- "vendi 1.0→20" in `challenger_batches.md` = the effective-distinct-question count growing as novel questions accumulate.

### 3.5 `CallerBatchManager` — the timing-coalesced batching (`caller_reward_manager_verl08.py`)
verl 0.8 calls reward **per-sample** (`run_single`), but the challenger reward is **irreducibly batch** (one solver POST + cross-batch diversity). So:
- Each `run_single`: decode the question, append `(q, future)` to a shared `_pending`, **(re)arm a `loop.call_later(_debounce, _flush)` timer — last registration wins** (`_debounce=REWARD_COALESCE_DEBOUNCE`, **default 0.5s, script sets 0.8s**), `await future`.
- `_flush` (under an `asyncio.Lock` — serializes flushes because the diversity model + bank are **non-thread-safe**; hence `num_workers=1`): snapshot pending, run `compute_score` over ALL coalesced questions in an executor thread, resolve each future. Errors → fill `score=0.0` (never hang the trainer).
- **Service call is file-based IPC:** `caller_penalty.py:104-123` splits the batch into `N_SERVICES` shards → temp JSON → HTTP `GET /hello?name=<shardfile>` (ports 5000+i, 4-worker pool, **no GET timeout**) → polls `<shard>_results.json`.

> **Footgun (§9.10):** the debounce is **wall-clock**, so a step's 240 questions can split across flushes (observed 220+16). Diversity is computed *per flush* (batch-wide nearest-neighbor) → stragglers get novelty over a tiny set → inflated. Reward is non-deterministic w.r.t. timing/env-speed.

---

## 4. Stage B — generate → band → judge (`iteration_rzero.sh:131-159`)

**1. Generate (`question_generate.py`):** `--model $QUESTIONER_HF --num_samples 1000` × 3 GPUs (`--suffix`=GPU idx **and vLLM seed**). System prompt = "expert competition-math problem setter"; the **integer-only HARD-REQUIREMENT block is included only if `QUESTIONER_PROMPT` ends `questioner_verifiable.jinja`** (difficulty target: "<30% of advanced HS students"). `max_tokens=4096, temp=1.0, top_p=0.95, n=1`. Output `<question>…</question>\n\boxed{answer}`; parse takes the **last** `<question>` and last `\boxed{}`; parse-fail → `score:-1`. **No dedup.**

**2. Evaluate / band-eval (`evaluate.py`):** `--model $SOLVER_MODEL --num_samples EVAL_N=9` (`temp=1.0, top_p=1.0, top_k=40, max_tokens=8192`). For each question: extract `\boxed{}` from the 9 samples, drop None/empty; **`if len(results)<2: continue` (drops the question)**; cluster, `majority`, **`score = max_count/len(results)`** (self-consistency). Skips questions containing `证明`/`box` or whose answer contains `text`. Writes `{question, answer=majority_self_vote, score, results=[per-sample answers]}`. **This is the recurring crash point** (clip80/clip65 died here; `/data` removed the disk failure mode).

**3. Band filter (`upload.py:58-63`):** keeps `min_score ≤ score ≤ max_score` (**inclusive**) and `answer ∉ {"","None"}`. **Script passes `MINSC=0.3 MAXSC=0.8`** (upload.py's *own* defaults are 0.3/0.7 — §9). Writes **all** rows pre-filter to `artifacts/$TAG/all_questions.jsonl` (the `results` source for cvband), and filtered `${TAG}.parquet` (cols `problem, answer=self-vote, score`). **Abort if no parquet → `exit 3`.**

**4. Judge — verified arm (`judge.py`+`verify.py`):** on the filtered parquet, generate `K_PROGRAMS=3` programs/question (`temp=0.6, max_tokens=2048`), run (`subprocess`, no sandbox, 25s, integer-only output), `program_consensus` needs `MIN_AGREE=2` → `verified_answer`. **Replaces the `answer` column with the program label; `score` carried through unchanged**; no-consensus rows **dropped**. → `${TAG}_verified.parquet`. `report_jsonl` records `majority_agrees` (self-vote vs program label).

**5. CVBAND (opt-in, `CVBAND=1`, `correctness_band.py`):** recompute true solve-rate `sr = mean(grade_answer(x, verified_label) for x in results)` per question (match = **exact stripped question text** vs `all_questions.jsonl`); keep **`lo < sr < hi` (exclusive**, defaults 0.2/0.8). The band-bug fix (§9.1). Default OFF.

**6. Convert (`to_verl08_parquet.py`):** → `${TAG}_verl08.parquet`: `prompt=[{system: "\boxed{}" prompt},{user: q}]`, `ability=math`, `reward_model={style:rule, ground_truth: answer}`, `extra_info={index, score}`. Drops empty q / `answer∈{"","None"}`. **`NROWS`** = row count → drives Stage C batch.

---

## 5. Stage C — solver training (`iteration_rzero.sh:161-189`)

Runs if `NROWS ≥ 8`. **Adaptive batch (`:166-171`):**
```
NROWS≥256 → SBATCH=256 SMINI=128
NROWS≥128 → SBATCH=128 SMINI=64
NROWS≥64  → SBATCH=64  SMINI=32
else      → SBATCH=NROWS//8*8 (min 8), SMINI=SBATCH
```
(iter1 108→64; 2026-06-25 redo 196→128 — a genuine run-to-run difference, §9.10.)

**verl args (`:172-185`):** `CUDA_VISIBLE_DEVICES=0,1,2,3`, `max_response_length=4096`, `train_batch_size=SBATCH ppo_mini_batch_size=SMINI`, `rollout.n=5`, `max_model_len=6144`, `tensor_model_parallel_size=2`, `gpu_memory_utilization=S_GPU_MEM=0.7`, `ppo_max_token_len_per_gpu=S_TOKLEN=12288`, `model.path=$SOLVER_MODEL` (cross-iter chaining via `run_rzero.sh`). **Reward = built-in `naive` manager + `math_verl08_single.py:compute_score`** (no service). `total_training_steps=20 save_freq=20 n_gpus_per_node=4 experiment_name=${TAG}_s rollout_data_dir/validation_data_dir` set. Step ≈12-13min @4096/batch≤128 (≈22min @8192). Iteration wall-clock ~3.5-7.5h.

---

## 6. Solver reward + answer parsing — exact (`math.py`, `math_verl08_single.py`, `mathruler`)

**Reward (`math.py`, `format_weight=0.1`):**
```python
predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)               # whitespace-strip around <>/
format   = 1.0 if re.fullmatch(re.compile(r"<think>.*</think>.*\boxed\{.*\}.*", re.DOTALL), predict) else 0.0
accuracy = 1.0 if grade_answer(extract_boxed_content(predict), ground_truth) else 0.0  # try/except→0
overall  = 0.9·accuracy + 0.1·format
```
Returns list of `{overall, format, accuracy}`. **`math_verl08_single.py`** is a per-sample shim: calls the same core, returns one dict with `overall`→**`score`** (verl reads `score`).

> **§9.6 CONFIRMED FOOTGUN:** the format regex **requires literal `<think>…</think>` before `\boxed{}`**. With `enable_thinking=false` the model emits no `<think>`, so `re.fullmatch` is **False for essentially every rollout → format = 0 always → overall = 0.9·accuracy**, and the reported `format` metric is a flat 0 (expected artifact, not a generation bug). Verified empirically: `"reasoning \boxed{42}"` → fullmatch False.

**`extract_boxed_content` (mathruler):** last `\boxed{` via `rfind`, brace-depth scan (nested OK: `\boxed{\frac{1}{2}}`→`\frac{1}{2}`; two boxes→last). **No box, or truncated mid-box → returns the STRING `"None"`** (not Python `None`) → grades as wrong (accuracy 0). Truncation before `\boxed` ⇒ reward 0 (dominant 0-reward cause on long completions).

**`grade_answer` (mathruler):** two normalization layers then sympy. Accepts: `1,000==1000`, `\frac{1}{2}==0.5`, `million→*10^6`, latex/unit normalization, tuples element-wise. **Rejects:** unreduced fractions (`2/4≠1/2`, by design — exact string match for fractions), integer-asymmetry (if gt is int, given must be strict int). **Disables sympy** (BAD_REGEXES/BAD_SUBSTRINGS) when the answer has exponents (`2^10`, `x^{2}`) or >2 unknown letters → those pass only via exact normalization.

---

## 7. Evaluation — current (`run_oly.sh` / `eval_oly_shard.py` / `agg.py`)

**`eval_oly_shard.py`:** `zwhe99/simplerl-OlympiadBench` test (~675), shard `i%nshards==shard` (4 GPUs). **Greedy** (`temp=0, n=1, max_tokens` default 8192), `enable_thinking=False`, system "reason step by step, `\boxed{}`", `max_model_len=max(4096, max_tokens+2048)`, `gpu_mem=0.80, enforce_eager`. golds = `final_answer` as **list** (multi-answer). `fmt = boxed nonempty`; `corr_first` vs golds[0]; **`corr_any` vs any gold** (the multi-answer-correct metric). `resp_len` = **characters**. `solution` truncated 3000 chars.

**`run_oly.sh <ckpt> <tag> <max_tokens=8192>`:** one shard per GPU 0-3, writes `/data/selfplay/oly_<tag>_shard*.jsonl`, inline-aggregates `format_rate, acc|fmt (first AND any), pass`. **Div-by-zero if fmt=0** (unguarded). **`agg.py <tag>`:** standalone aggregator but reports **`corr_first` only** (understates multi-answer) — guarded div, `trunc` heuristic ≥31000 chars.

**Always report the decomposition:** `format_rate`, `acc | formatted`, `pass`, with `max_tokens`. At greedy fmt_rate=1.0 → deltas are pure acc|fmt. `run_math500.sh` evals MATH-500 but is **hardcoded to base/it1/it2** and uses the deprecated `eval_compare.py`.

**DEPRECATED `pipeline/eval_compare.py`:** `probs[:n]` with `--n 200` = first-200-of-675 (`:47`); grades `final_answer[0]` only (`:27`, mis-scores multi-answer); default `max_tokens=1024` (crushes format_rate). The cont per-iter eval still calls it (reports 0.325 not 0.545) — don't trust those (§9.11).

---

## 8. The reward service (`vllm_service_init/start_vllm_server.py`)

Flask app wrapping one vLLM engine; the questioner-reward backend. **Spawn-safe:** heavy `_init_model()` runs only under `if __name__=='__main__' and parent_process() is None` (the spawn child re-imports the module). A **GPU idle-worker thread** does matmuls to keep clocks up between requests. **Only one route: `/hello`** (no `/generate`). Flow: `name=request.args['name']` = a JSON task-file path → load → **`os.remove(name)`** → render chats (`enable_thinking=False`) → `model.generate(n=10)` → `process_single` computes `score=max_count/len` → optional verifiability judge → write `<task>_results.json` (the caller polls this file; the HTTP body is only a status string). **`/hello?name=None` → HTTP 500** (`open('None')` FileNotFoundError) — so a bare `/hello` is NOT a liveness ping; a 500 there means "bad/missing file arg," not "down."

---

## 9. FOOTGUNS & silent failures

**9.1 Band = self-consistency, not correctness (THE big one).** `evaluate.py`/`upload.py` `score` and `caller_penalty` `uncertainty` key on the solver's **self-agreement** (mode fraction), orthogonal to correctness vs the label. A confidently-WRONG question passes the band → all samples ~equal reward → **zero GRPO advantage**. iter2 had ~45% zero-gradient rows. Fix = `correctness_band.py` (`CVBAND=1`) on solve-rate-vs-label. **Band-fix A/B refuted at eval (0.545→0.550)** — not the main cause of the U-shape.

**9.2 Cross-base comparisons invalid (Qwen3.5 vs Qwen3).** verified arm `qwen3_5` (0.573); v5 `qwen3` (0.30≈external 0.33). Check `config.json model_type` before comparing absolutes.

**9.3 Eval subset/multi-answer bugs (`eval_compare.py`).** first-200-of-675 + single-gold grading. Use the full-675 sharded eval; even there, `agg.py` reports `corr_first` only — use `run_oly.sh`'s `*_any`.

**9.4 Eval/train budget.** Solver trains @4096, eval @8192. Budget is **constant across iters** → not the regression cause; just report `max_tokens`. (In training, temp-1.0 sampling tails DO hit the 4096 cap — ~10% of long rollouts truncate before `\boxed` → pressures terseness.)

**9.5 Rollout logging was off.** `iteration_rzero.sh:87` once hardcoded `[console]`. Now `[console,wandb]`+`rollout_data_dir`+`validation_data_dir` on both stages; `ray_trainer.py::_log_rollout_data` patched to push `train/generations`. Verify each launch.

**9.6 Solver format reward is ~always 0** (the `<think>` regex + `enable_thinking=false`) → `overall≈0.9·accuracy`, `format` metric flat 0 (expected). Treat per-step reward as ≈0.9·accuracy. §6.

**9.7 `extract_boxed_content` returns the STRING `"None"`** on missing/truncated box (not Python `None`) — grades as wrong (fine), but `is None` checks downstream are fooled.

**9.8 verl `save_contents=[hf_model]` omits processor files** → vLLM auto-detects Qwen3.5 multimodal and fails. `fixup_hf` copies `preprocessor_config.json + video_preprocessor_config.json + merges.txt + vocab.json + processor_config.json` from base — load-bearing, baked into the script.

**9.9 Kernel/env drift breaks bit-faithful reproduction; `use_remove_padding=true` (packing) is a REAL corruptor on Qwen3.5 GDN (−11pt, quantified 2026-06-28).** Pipeline patched `sdpa→flash_attention_2` + `remove_padding False→True` (speed) AND env upgraded (vllm 0.23). These change training-forward numerics. **Full 2×2 (attn × `remove_padding`, else byte-identical 108-row/20-step/n5/4096; OlympiadBench-675 @8192, greedy, fmt≈1.0):**

| | rp=false | rp=true |
|---|---|---|
| sdpa | **0.621** (7509 ch) | **0.508** (5948 ch) |
| flash | **0.633** (7612 ch) | **0.495** (8536 ch) |

**`remove_padding=true` (packing) is the ENTIRE effect: −0.12, identical across both attn backends; flash attention is INNOCENT (±0.013).** No-packing reproduces ≈0.63 (beats base 0.573) on BOTH attn; packing drops to ≈0.50 (below base) on BOTH. Two runs per packing level (no-pack 0.621/0.633, pack 0.508/0.495), within-level spread ≈0.012 ⇒ the −0.12 gap is ~10× noise, **variance-controlled by construction (not n=1)**. So packing genuinely corrupts gradients on the hybrid GDN/linear-attn model — NOT a metric-normalization artifact (revises the earlier guess), and NOT flash (the kernel-swap confound was 100% the packing that rode along). Packing's failure *mode* depends on attn: sdpa+pack → **terse**-and-worse (5948 ch), flash+pack → **verbose/rambling**-and-worse (8536 ch); both no-pack arms sit at normal ~7500 ch. **NOTE this never touched the real experiment** (real iter1/iter2 ran rp=false — §provenance); packing only corrupted the *optimization/repro* runs (smoke_opt, kernel-iso, redo). **flash+rp=false is the recommended fast+faithful kernel (0.633, healthy curve, runs on the current env — the earlier "old sdpa+rp=false path crashes" was an FSDP-offload-config issue, fixable, not flash). NEVER enable `use_remove_padding` on Qwen3.5 for a learning-faithful run.**

**9.10 Questioner reward is non-deterministic w.r.t. timing & run-state.** (a) diversity bank cold-start (`rm -f`, vendi 1.0); (b) **wall-clock debounce** splits a step's 240 Qs (e.g. 220+16), diversity computed per-flush → straggler inflation; (c) stochastic solver scoring (n=10 sampled). Then **adaptive Stage-C batch** amplifies (108→64 vs 196→128). Fix: seed/persist the bank, make the flush **count-based** not time-based, pin band count/batch. **rzc's `challenger_batches.md` was deleted** → keep these artifacts.

**9.11 The cont (iter2-4) run is instrumented-broken.** `iter23_continue.sh`: `WANDB_API_KEY` unbound (wandb OFF), HF upload failing, per-iter eval = n=200 bug (0.325 not 0.545), and the **solver FROZE at iter2** (iter3 Stage-C OOM → iter3/iter4 reused `cont_it2_verified_s`). Don't trust cont's numbers.

**9.12 Operational.** GPU lanes: side job on a pipeline GPU → Stage C OOM (killed iter3's solver); 4–7 = langlin. vLLM standalone needs `cd /home/nvidia` (NOT `/R-Zero` — local `verl/` shadows the package → `No module named verl.trainer.ppo`), `CUDA_HOME`+venv-bin on PATH, `enforce_eager=True`. **No sandbox** in `judge.py`/service `run_program` — model-written Python runs as plain subprocess (25s timeout) on the shared box. `--suffix` doubles as the vLLM seed (must be numeric). `.bash_history` BANNED (CrowdStrike).

**9.14 verl checkpoint = FSDP-sharded `.pt`, NOT HF safetensors → eval needs an explicit merge (silent: crashes as eval `n=0`).** With this verl/vllm-0.23 env, `save_freq` writes `global_step_N/actor/model_world_size_4_rank_{0-3}.pt` (sharded weights, intact) + `actor/huggingface/` containing **only config+tokenizer, NO `*.safetensors`** (`save_contents` lacks `hf_model`). Pointing vLLM at `actor/huggingface` → `RuntimeError: Cannot find any model weights` → all eval shards die in ~60s → 0 rows → `agg.py`/`run_oly.sh` `fmt/n` raises **`ZeroDivisionError` (n=0)** — the failure surfaces only at aggregation, looking like an eval bug, while training was fine. **Fix (in `run_oly.sh`, `.bak_premerge`):** if `$MODEL` has no `*.safetensors`, run `python -m verl.model_merger merge --backend fsdp --local_dir <…/actor> --target_dir <…/actor/hf_merged>`, copy processor files (§9.8) into `hf_merged`, repoint `MODEL` there. Merge loads the 4 shards on CPU (~2s) and writes one `model.safetensors` (~9GB for 4B); idempotent. Affects every solver ckpt from this env (orig_iso, pad_iso, kernel_iso all had the empty `huggingface/`).

**9.13 Default skews to know.** `DIVERSITY_WEIGHT`: script 0.5 vs `diversity.py` 0.4. `MAXSC`: script 0.8 vs `upload.py` 0.7. Band conventions: `upload.py` **inclusive** [0.3,0.8] vs `correctness_band.py` **exclusive** (0.2,0.8). `VERIFY_SUBSAMPLE`: server default 0 (off) vs script 1.0 (on). `N_SERVICES`: `caller_penalty.py` default 4 vs box `N_SVC=1`. `CLIP_EASY`: read but never script-set (external override only). `caller_penalty.py` `STORAGE_PATH` fallback is a hardcoded Tencent path (harmless while exported).

---

## 10. Key file index

**Pipeline (Brev `/home/nvidia/R-Zero/`, mirror `rzero_pipeline/`):** `scripts/iteration_rzero.sh` (one iter, 192 lines), `scripts/run_rzero.sh` (N-iter loop), `run_it1redo.sh`/`run_iter1clean.sh` (per-run launchers). `examples/reward_function/`: `math.py`+`math_verl08_single.py` (solver reward), `caller_penalty.py`+`_verl08.py` (challenger reward), `caller_reward_manager_verl08.py` (`CallerBatchManager`), `diversity.py` (Vendi). `question_generate/question_generate.py`; `question_evaluate/`: `evaluate.py` (band-eval), `upload.py` (band filter), `judge.py`+`verify.py` (program consensus), `correctness_band.py` (cvband). `pipeline/`: `to_verl08_parquet.py`, `eval_compare.py` (DEPRECATED). `vllm_service_init/start_vllm_server.py` (reward service). Eval/analysis `/data/selfplay/`: `eval_oly_shard.py`, `run_oly.sh`, `run_math500.sh`, `agg.py`, `acc_by_len.py`, `difficulty_dist.py`.

---

## A. Appendix: LEGACY local pairwise self-play (`grpo_math/`)

> Not the active path. In-repo pairwise-preference variant: TRL `GRPOTrainer`, `FINAL_ANSWER: <int>` strict.

**Loop** (`self_play/generate_pairwise_data.py::main` @883): generate → validity retry+repair → confirm gate (`single_unambiguous_answer`, ≤400 chars, not self-correcting) → dedup → solve K via sampling groups → judge (`pairwise` PREFERENCE or `single_verify` VERDICT) → score (Elo / verify-score, R_stab/R_sep) → JSONL.

**Strict parser** (`data/reward.py:11`): `re.compile(r"FINAL_ANSWER\s*:\s*(-?\d+)(?:\.0+)?(?![\d./])")` — case-sensitive, no commas/`+`/unicode/`$`, integer-floats OK, LAST match. `binary_reward`: 1.0 iff `pred==gt`.

**TRL training** (`trl/train_grpo_trl.py::main` @755): `max_steps`-driven, `num_generations`=`rollout.k`(4), `beta`=`train.kl_beta`, LoRA optional, DeepSpeed zero2/3.

**Legacy footguns:** malformed `PREFERENCE:`→`random.choice(["A","B"])` (poisons Elo); `CONFIDENCE:` defaults 0.5; `FINAL_ANSWER: 1,000` captures `1`; first-vs-last tag inconsistency; ignored YAML keys (`weight_decay`/`adam_beta*`/`max_*_samples` never reach GRPOConfig); `train_solver_verdict_grpo.yaml`/`train_proposer_question_grpo.yaml` stale (fall through to GSM8K).
