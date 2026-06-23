# ARCHITECTURE.md — Hyper-detailed system reference

> **Purpose.** A living, exhaustive description of how this project actually works — every reward formula, every answer-parsing regex, the training setup, the data flow, and (most importantly) the **footguns** that have silently corrupted experiments. Read the [Footguns](#10-footguns--silent-failures-read-this-first) section first. Keep this updated; we should regenerate it whenever the pipeline changes.
>
> Last full recount: 2026-06-24.

## 0. There are TWO systems — don't confuse them

| | **(A) `grpo_math/` — local pairwise self-play** | **(B) R-Zero pipeline — the experimental system** |
|---|---|---|
| Where | This repo (`/grpo_math/...`) | Brev box `/home/nvidia/R-Zero/` + `scripts/iteration_rzero.sh` |
| Roles | Generator → Solver → Judge (pairwise PREFERENCE or single VERDICT) | Challenger(questioner) → Solver → program-Judge; band-filtered GRPO |
| Trainer | TRL `GRPOTrainer` (`grpo_math/trl/train_grpo_trl.py`) | verl 0.8 `main_ppo.py` (GRPO) |
| Answer format | `FINAL_ANSWER: <int>` (strict) | `\boxed{...}` (mathruler) |
| Status | The self-play research code in *this* repo | **What the iter1–4 experiments, the regression, and the band bug all live in** |

The recent experiments (iter1 0.630 / iter2 0.545, the "regression", the eval audit) all ran on **system (B)**. The local repo (A) is the pairwise-preference variant. They share *ideas* (self-play, difficulty-balanced curriculum, format-vs-accuracy discipline) but are **separate codebases with different reward/parsing**.

---

## 1. System (A): local pairwise self-play — loop overview

`grpo_math/self_play/generate_pairwise_data.py` (~1983 lines, entry `main()` @883) orchestrates one rollout:

1. **Generate** questions (one batched generator call, `_parse_question` @226).
2. **Validity retry** (parallel rounds, `_looks_like_question_only` @328) + a temp-0.2 **repair pass**; hard `RuntimeError` if still invalid.
3. **Confirm gate** (`_parse_question_confirm` @238) — the real accept/reject. Requires `different_from_samples=yes`, `single_unambiguous_answer=yes`, concise (≤400 chars, no newline), and NOT self-correcting.
4. **Dedup** (naive lowercased exact-match @1250).
5. **Solve** — K solutions per question via **sampling groups** (§2), generator unloaded first to free VRAM.
6. **Judge** — `pairwise` (PREFERENCE) or `single_verify` (VERDICT/CONFIDENCE) (§3).
7. **Score** — Elo (pairwise) or verify-score; reliability metrics R_stab/R_sep; optional oracle; proposer reward.
8. **Write** one JSONL row per question (`f.flush()` each).

### 2. Sampling groups (quality variation)
`solver.sampling_groups` = list of `{count, temperature, top_p, [api_model], [api_base_url], [api_key_env]}`. Hard constraint: `sum(count) == num_solutions_per_question` (else `ValueError`). Variation created by (1) per-group temp/top_p, (2) per-group **model override** (route a group to a weaker model), (3) group identity recorded in `solution_group_map` (group 0 = "strong", group 1 = "weak" — **hardcoded** for R_sep).

### 3. Judge modes
- **pairwise**: for each pair, `repeats` judgments with **A/B order balanced** (`use_swapped = (rep_idx+swap_offset)%2`); parse `PREFERENCE: A|B`; `consistency = max(n_a,n_b)/repeats`. Ties structurally impossible.
- **single_verify**: per-solution or batched `VERDICT: CORRECT|INCORRECT` (+ optional `CONFIDENCE:`). `score_points[s] += n_correct/parsed_repeats`. Malformed verdicts **excluded** (not counted incorrect).

### 4. Elo (pairwise only) — `_elo_*` @366
`expected = 1/(1+10^((r_b−r_a)/400))`; `new_a = r_a + K·(s_a − e_a)`, K=`elo_k`(24), init 1000. **Order-dependent** (sequential updates over pairs/prefs — not a simultaneous solve). single_verify leaves all Elo at init and ranks by verify_score.

### 5. Reliability metrics
- **R_stab** (`reliability.preference_stability`) = mean self-agreement across repeats (per pair or per solution).
- **R_sep** (single_verify, ≥2 groups) = `group_verify_mean[0] − group_verify_mean[1]` (strong − weak). Only groups 0,1 compared.
- **R_cons**: no field by that name; operationalized as per-solution `confidence` and `question_rewards.verdict_variance = 4·p·(1−p)`.
- **Proposer reward** (`question_rewards.compute_question_reward`): `D = clamp(1 − 2|p−0.5|)` (difficulty, peaks at p=0.5), `r_q = clamp((1−λ)·D + λ·V)` with λ=`lambda_verifiability`(0.35), then `× clamp(filter_score)`. Rejected/duplicate → 0.0.

---

## 2'. System (A): rewards & the strict answer parser

**The single load-bearing regex** — `grpo_math/data/reward.py:11`:
```python
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(-?\d+)(?:\.0+)?(?![\d./])")
```
- `FINAL_ANSWER` is **case-sensitive** (no IGNORECASE). `final_answer:` → no match.
- Colon mandatory. `(-?\d+)` = optional single minus + ASCII digits. **No commas, no `+`, no unicode digits.**
- `(?:\.0+)?` accepts integer-valued floats (`30.00`→30).
- `(?![\d./])` rejects fractions/decimals (`3/4`, `3.5`) from false-passing.

`extract_final_answer_int_strict` (reward.py:42) takes the **LAST** match. `binary_reward` (reward.py:87): `1.0 if pred==gt else 0.0`, else 0.0 if either is None. `extract_ground_truth_int` (reward.py:14) takes the **last int anywhere** in the GSM8K answer.

**TRL reward funcs** (`train_grpo_trl.py:817–1022`), selected by `reward.mode`:
- `correctness` (default): `reward_correct`(w=1.0) + `reward_format`(w=0.0) + `reward_answer_boundary`(w=0.0).
- `verdict`: `reward_verdict` (external teacher API, default gpt-4.1) + format + boundary.
- `verdict_gated_format`: `reward_verdict × reward_clean_final_answer_format` (multiplicative).
- `question`: full proposer pipeline (`compute_question_reward`).
- `reward_format` graded: tail==0 →1.0, tail>0 →0.5, lenient →0.25, else 0.0. `reward_answer_boundary`: linear decay over ≤400 trailing chars.

---

## 3'. System (A): TRL training setup

Entry `train_grpo_trl.py::main()` @755 (`python -m ...` or `accelerate launch -m ...`). Delegates the loop to TRL `GRPOTrainer.train()`; `num_train_epochs=1`, run is **`max_steps`-driven**.

**Data** (`_make_dataset` @56), dispatch on `data.source`: `gsm8k` (default), `pairwise_jsonl` (filters `trainable_for_solver/accepted`, dedups, deterministic split by `eval_fraction`), `question_generation` (synthesized generator prompts).

**Key hyperparameters** (GRPOConfig @1108–1147):
| field | YAML key | default | notes |
|---|---|---|---|
| `num_generations` | `rollout.k` | required | **GRPO group size G**; configs use 4 |
| `beta` | `train.kl_beta` | required | KL-to-ref; 0.005 full / 0.01 LoRA |
| `learning_rate` | `train.lr` | required | 4e-6 full / 1e-5 LoRA |
| `max_completion_length` | `rollout.max_new_tokens` | required | 256 (gsm8k) / 1024 (verdict) |
| `temperature`/`top_p` | `rollout.*` | required | |
| `save_steps` | `train.save_every` | 200 | |
| `lr_scheduler_type` | — | `cosine` (hardcoded) | |
| `num_train_epochs` | — | 1 (hardcoded) | |

**LoRA** (`train.lora.enabled`, default False): `r`(16)/`alpha`(32)/`dropout`(0.05)/`target_modules`(`all-linear`). Adapter-resume path (PEFT dir without `config.json`) is hard-verified by norm ratio and **raises** if weights don't load.

**DeepSpeed** (consumed by `accelerate`, NOT by `main()`): zero2 (grad+opt sharded, default) / zero3 (+param sharded, micro-batch hardcoded 1) / zero3_offload (opt→CPU, `overlap_comm:false`).

---

## 4. System (B): the R-Zero pipeline — the experimental system

`scripts/iteration_rzero.sh` runs, per iteration (env: `ITER EXP ARM QUESTIONER_MODEL SOLVER_MODEL`):

**Stage A — questioner(challenger)-train** (`Q_STEPS=6`, GRPO): the challenger generates questions; reward via a standalone vLLM **reward service** (`vllm_service_init/start_vllm_server.py`, the current solver, `n=10` samples) → challenger reward (§5). GPU layout `N_SVC=2` → service on GPUs 0,1, train on 2,3.

**Stage B — data pipeline:**
1. `question_generate.py` — mass-generate `NUM_SAMPLES=1000`/GPU questions.
2. `evaluate.py --num_samples EVAL_N=9` — the **solver** answers each question 9×; computes **self-consistency** `score = max_count/len(results)` (mode fraction). Writes answer distributions to `all_questions.jsonl` (`results` = per-sample answers).
3. **Band filter** `upload.py --min_score MINSC=0.3 --max_score MAXSC=0.8` → keeps `MINSC ≤ score ≤ MAXSC` (`upload.py:56`). **`score` is SELF-CONSISTENCY, not correctness.** → `${TAG}.parquet`.
4. (verified arm) `judge.py` — **program-consensus**: generates a Python program per question, runs it, majority vote → `verified_answer` (the program label) + `${TAG}_judge.jsonl`. → `${TAG}_verified.parquet` (cols `problem, answer=label, score=self-consistency`).
5. `to_verl08_parquet.py` → solver-training parquet (`prompt` with system "reason step by step, put answer in `\boxed{}`", `reward_model.ground_truth`).

**Stage C — solver-train** (`S_STEPS=20`, GRPO): train the solver on the band-filtered rows; reward = solver reward (§5). `n=5`, `max_response_length=4096`, batch ≤256, `n_gpus_per_node=4`.

Output: `cont_itN_verified_q` (questioner) + `cont_itN_verified_s` (solver), fed to iter N+1.

### 5. R-Zero reward formulas (exact)

**Solver reward** (`examples/reward_function/math.py::compute_score`, `format_weight=0.1`):
```
format_score   = 1.0 if re.fullmatch(<answer pattern>, predict) else 0.0
accuracy_score = 1.0 if grade_answer(extract_boxed_content(predict), ground_truth) else 0.0
overall = (1 − 0.1)·accuracy + 0.1·format  =  0.9·accuracy + 0.1·format
```
→ A finished-but-wrong answer scores **0.1** (format only); a truncated answer scores **0.0**. This 0.1 floor + zero-variance is the engine of the reward-starvation failure (§10).

**Challenger reward** (`examples/reward_function/caller_penalty.py::compute_score`):
```
uncertainty = min(score, 1 − score)          # score = solver SELF-CONSISTENCY; peaks 0.5
verif_term  = VERIFY_WEIGHT(0.75) · verified  # verified ∈ [−1,1] = 2·votes/K − 1 (program consensus)
diversity   = vendi marginal diversity reward (Vendi-score Nyström)
final_score = uncertainty + diversity + verif_term
# CLIP_EASY (clip80 experiment): if score > CLIP_EASY(0.80) → clip reward (penalize too-easy)
```

> **Note both `uncertainty` (challenger) and the band filter (upload.py) key on solver SELF-CONSISTENCY — never on correctness vs the program label.** This is the design center of the band bug (§10.1).

### 6. R-Zero answer parsing
- Solver answers: `mathruler.extract_boxed_content` (last `\boxed{}`), graded by `mathruler.grade_answer` (symbolic equivalence — `\frac{14}{15}` == `14/15`, `n^2-n+1` == `$n^{2}-n+1$`). The grader is **calibrated/sane** (verified: 4≠5, 3≠3.00001, rejects garbage).
- `enable_thinking=false` in the chat template (`+data.apply_chat_template_kwargs.enable_thinking=false`) — applies to **both** questioner and solver, all iterations. The questioner emits short (~100-tok) problems with no scratch-pad; the solver still reasons in the body (2–3k-tok CoT) without `<think>` tags.

---

## 7. Evaluation (`pipeline/eval_compare.py`)

Benchmarks: gsm8k, math500, amc23, minerva, **olympiad** (`zwhe99/simplerl-OlympiadBench`, 675 math problems), aime24/25. Prompt = "reason step by step, `\boxed{}`", `enable_thinking=false`, greedy at k=1. Extract `\boxed{}` → `grade_answer` vs gold. Reports `format_rate`, `acc_given_formatted`, `pass` (the mandatory decomposition).

**Eval bugs found 2026-06-24 (see §10):** `--n 200` takes only the **first 200 of 675** (`probs[:n]`); grades only against `final_answer[0]` (mis-scores multi-answer). The 4-GPU sharded, full-set, multi-answer-aware replacement is `eval_oly_shard.py` + `run_oly.sh` + `agg.py` (on Brev `/data/selfplay/`).

**Calibrated results (full 675, same harness):**
| model | base | OlympiadBench 8k / 4k | MATH-500 4k |
|---|---|---|---|
| Qwen3.5-4B base | qwen3_5 | 0.573 / 0.502 | 0.858 |
| iter1 (verified-20) | qwen3_5 | **0.630** / 0.560 | 0.884 |
| iter2 (cont) | qwen3_5 | 0.545 / 0.533 | 0.864 |
| R-Zero v5 | **qwen3** | 0.304 / 0.299 | — |

---

## 10. FOOTGUNS & silent failures (READ THIS FIRST)

### 10.1 The band measures self-consistency, not correctness (THE big one)
Both `upload.py` (band filter, keeps `score ∈ [0.3,0.8]`) and `caller_penalty.py` (`uncertainty = min(score,1−score)`) use the solver's **self-consistency** (mode fraction across samples), which is **orthogonal to correctness vs the program label**. A question the solver *confidently and consistently answers WRONG* passes the band but gives **all samples the same ~0.1 reward → zero variance → zero GRPO advantage**. On iter2's training set, **~45% of band-passed questions were zero-gradient** (37.7% always-wrong + 7.6% always-right). **Fix** (`question_evaluate/correctness_band.py`): post-judge filter on solve-rate *vs `verified_answer`* in (0.2, 0.8).

### 10.2 Cross-base comparisons are invalid (Qwen3.5 vs Qwen3)
verified-20 is `model_type: qwen3_5` (base 0.573); R-Zero v5 is `model_type: qwen3` (0.30, matches external "0.33"). The "massive jump" was the **base generation**, not training/eval. **Always check `config.json model_type` before comparing absolute numbers.** Only within-base, within-harness deltas are comparable.

### 10.3 Eval subset & multi-answer bugs (`eval_compare.py`)
`--n 200` silently evaluates only the **first 200 of 675** (not random, ~20 pts harder). Grades against `final_answer[0]` only → mis-scores the 94/675 multi-answer problems. Inflated the apparent "regression". Use the full-set sharded eval.

### 10.4 Eval/train budget mismatch
Solver trains at `max_response_length=4096` (4k) but `eval_compare.py` defaults/ran at `max_tokens=8192` (8k) → 2× training budget. format_rate is 1.0 at both (no truncation here), but the larger budget adds ~5–7 acc points for verbose Qwen3.5 models. **Always report the eval `max_tokens` and match it to training when comparing.**

### 10.5 Rollout logging was off (cost the most time)
`iteration_rzero.sh:87` hardcoded `trainer.logger=[console]` with no rollout dump → the whole iter2 divergence investigation had no rollout text. **Non-negotiable now** (see CLAUDE.md): every verl launch must set `trainer.logger=[console,wandb]` + `trainer.rollout_data_dir=...` (+ `validation_data_dir`/`log_val_generations`). Also patched `ray_trainer.py::_log_rollout_data` to push a sample to wandb `train/generations`.

### 10.6 Parsing footguns (system A)
- **Malformed `PREFERENCE:` → `random.choice(["A","B"])`** (`generate_pairwise_data.py:271`) — a bad judge output becomes a coin-flip, silently poisoning Elo.
- **`_BOOL_VERDICT_RE = \b(TRUE|FALSE)\b`** over-matches any "true/false" in judge prose.
- **`CONFIDENCE:` silently defaults to 0.5** on unparseable (`.9`, `85%`, `1.2` all fail).
- **`FINAL_ANSWER: 1,000` captures `1`** (commas truncate) → reward 0. Same risk in ground-truth extraction.
- **Case-sensitivity split**: strict extractor is `FINAL_ANSWER` only; verifier regex is case-insensitive. `final_answer: 5` is "formatted" to the verifier but scores 0 on correctness.
- **`$`/unit prefix** (`FINAL_ANSWER: $42`) → **no match at all**. Trailing period (`FINAL_ANSWER: 42.`) → fails the lookahead.
- **first-vs-last tag inconsistency**: `extract_final_answer_int_strict` uses the LAST tag; `canonicalize`/`tail_char_count` use the FIRST. Two `FINAL_ANSWER:` lines with different ints disagree.

### 10.7 Ignored YAML keys (system A training)
`train.weight_decay`(0.1), `train.adam_beta1/2`(0.9/0.95), `train.max_grad_norm`, `data.max_train/eval_samples` are **never passed to GRPOConfig** → HF defaults silently used (wd=0.0, beta2=0.999, eval=256 via `--max_eval_samples`). `train_solver_verdict_grpo.yaml` / `train_proposer_question_grpo.yaml` reference offline JSONL keys the loader doesn't read → fall through to GSM8K → KeyError. Treat those two configs as stale.

### 10.8 Other operational footguns
- **GPU lanes / contention**: a standalone job on a GPU the live pipeline needs → Stage C fails on OOM (this exact mistake killed iter3's solver training). Never run a side job on a GPU the active pipeline uses; Brev GPUs 4–7 belong to `langlin` (do not touch).
- **enable_thinking=false** is a constant across all iters — not a regression cause, but means the questioner generates blindly (short, often self-`\boxed{}`-wrong; the label comes from program consensus, not the questioner).
- **vLLM env on standalone runs**: needs `CUDA_HOME` + venv-bin on PATH (nvcc + ninja) or engine init fails (`nvcc`/`ninja` not found); `enforce_eager=True` to skip torch.compile; `gpu_memory_utilization ≤ 0.80` if anything else shares the GPU; `__main__` guard for `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **Disk**: Brev root fills to ~96% (Ray then fails object creation). Use `/data` (16 TB) via symlink for rollouts/checkpoints; never delete another project's dirs.
- **verl `save_contents=[hf_model]` omits `preprocessor_config.json`** → vLLM auto-detects the Qwen3.5 (multimodal) architecture and fails to load the checkpoint with `OSError: Can't load image processor ... preprocessor_config.json`. Fix: copy `preprocessor_config.json` + `video_preprocessor_config.json` from the base model / a working checkpoint into the saved `.../actor/huggingface/` dir before eval/serve.
- **`.bash_history` is BANNED** (CrowdStrike alert) — see CLAUDE.md.

---

## 11. Key file index

**System A (`grpo_math/`):** `self_play/generate_pairwise_data.py` (orchestrator), `data/reward.py` (strict parser + binary_reward), `self_play/rewards.py` (verdict scoring), `self_play/question_rewards.py` (proposer reward), `trl/train_grpo_trl.py` (TRL GRPO), `configs/*.yaml`, `configs/deepspeed_*.json`. (`models/policy.py` is a standalone reference impl — NOT on the live path.)

**System B (Brev `/home/nvidia/R-Zero/`):** `scripts/iteration_rzero.sh` (pipeline), `examples/reward_function/{math.py,caller_penalty.py}` (rewards) + `*_verl08.py` shims, `question_evaluate/{evaluate.py,upload.py,judge.py,correctness_band.py}` (band+label), `question_generate/question_generate.py`, `pipeline/{eval_compare.py,to_verl08_parquet.py}`, `vllm_service_init/start_vllm_server.py` (reward service). Eval tooling on `/data/selfplay/`: `eval_oly_shard.py`, `run_oly.sh`, `agg.py`.
