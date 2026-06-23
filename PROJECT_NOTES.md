# Project Notes

Running log of experiments, findings, and decisions. Newest entries first.

---

## 2026-06-23 (later) — ROOT-CAUSE of the blind investigation: rollouts were never logged; fixed for iter3+

**The divergence investigation was crippled because the current run captured no rollout text.** `scripts/iteration_rzero.sh:87` hardcoded `trainer.logger=[console]` with no rollout dump, so for the whole cont run there was *zero* saved prompt/response/score text — the reward service stores only extracted answers, not generations. This silently overrides the standard-experiment-logging discipline (wandb + saved generations), which is why mining iter2's divergence meant reverse-engineering metrics from reward-service jsonl + parquet scraps instead of reading the actual rollouts.

**Fix applied to Brev `iteration_rzero.sh` (backup `.bak_logging`), at the iter2→iter3 boundary while the script was unopened (`fuser` empty; iter2 done rc=0, iter3 not started, launcher in eval phase) — so iter3+ captures everything:**
- Line 87: `trainer.logger=[console,wandb] trainer.log_val_generations=20`.
- Questioner launch (L125) + solver launch (L180): each appended `trainer.rollout_data_dir=$STORAGE_PATH/rollout_dumps/${TAG}_{q,s}` and `validation_data_dir=…_val`. Keyed by `${TAG}` (=`${EXP}_it${ITER}_${ARM}`) so questioner/solver/iters never collide.
- Verified non-no-op: verl `ray_trainer.py:1681-1683` → `_log_rollout_data` → `_dump_generations(inputs, outputs, gts, scores, reward_extra_infos, dump_path)` writes prompt+response+gt+score per step. wandb auth persisted in `~/.netrc` (has `api.wandb.ai`) → no prompt/hang. `bash -n` passes.
- These are pure logging side-channels: **no change to tokens/n/steps/reward/experiment.**

**Cornell clip80 (977443):** `MAX_ITER=1` (single iter) — no future iteration to patch and can't edit mid-iter1, so its iter1 rollout text is NOT captured this run. Acceptable: clip80's deliverable is the eval numbers (effect of `CLIP_EASY=0.80` on the band), not rollout mining. Cornell verl confirmed to support the same 3 keys; patch staged for any future multi-iter Cornell launch.

**CLAUDE.md:** added "Training launches MUST capture rollouts" section (verify `logger=[console,wandb]` + `rollout_data_dir` + `validation_data_dir`/`log_val_generations` on both Q and S invocations; note the mid-iteration edit hazard).

**CONFIRMED working:** Brev **iter3 started 22:23:50 UTC** (Q=cont_it2_verified_q/gs6, S=cont_it2_verified_s/gs20 — correctly chained from iter2-redo). `rollout_dumps/cont_it3_verified_q/1.jsonl` (548KB) **contains real prompt+response text** (`{"input":"system\nYou are an expert competition-math problem setter..."}`) → logging patch verified live. GPUs 0,1 at 100% (train), 2,3 hold DP reward-service replicas. iter2-redo eval reproduced the original: OlympiadBench **0.325** (orig 0.32), MATH-500 done.

**Cornell clip80 CRASH + fix (job 977443 → 977878):** old job silently crashed at questioner-train with `ImportError: FlashAttention2 ... doesn't seem to be installed` (the Brev A100 flash speed-patch was in the Cornell script, but `~/envs/rzero` has no flash_attn), then continued to the generate stage on the **untrained base questioner** → experiment invalid. Fix: scancel 977443; revert Cornell `iteration_rzero.sh` to `attn_implementation=sdpa` + `use_remove_padding=false` (exact-attention / pure-efficiency, experiment-equivalent; backup `.bak_flashfix`); also applied the logging patch + added `WANDB_API_KEY` to `rzc_env.sh` (Cornell had no wandb auth). Relaunched **977878** (PD, MAX_ITER=1). Cornell compute nodes have internet (HF upload works) so wandb won't hang.

**Overnight monitor (non-destructive):** `/tmp/overnight_monitor2.sh` (bg) — NEVER wipes a Brev checkpoint (the false-restart lesson); stall = all-logs-frozen >20min + GPU idle → ALERT only; auto-resubmits Cornell only if its job vanishes from queue (cap 3); fires on iteration-complete / new-solver-dump / cornell-done/crash. Replaces the destructive `/tmp/overnight_monitor.sh`.

Pending:
- [ ] Watch iter3 → solver-train stage; confirm `rollout_dumps/cont_it3_verified_s/` populates, then finish divergence analysis on REAL text (band=self-consistency admits uniformly-wrong-but-consistent → all-0.1 reward → zero advantage hypothesis).
- [ ] Confirm Cornell 977878 clears questioner-train (the prior crash point) once it leaves the queue.
- [ ] Quantify useful-signal fraction over the 710 iter2 training rows (correct-answer-in-1..4-of-5 = real advantage vs 0/all-5 = none) — pure data mining, no GPU.
- [ ] iter3/iter4 evals as they complete (OlympiadBench + MATH-500).

---

## 2026-06-23 — iter2 (cont, Q=clip80-questioner) eval results + auto-restart false-positive incident

**iter2 eval (cont run, Q=clip80_it1_verified_q, S=verified-20, DP service, flash+remove_padding, no fla):**
- OlympiadBench: format_rate 0.995, **acc|fmt 0.322, pass 0.32** (n=200, k=1, 8192 tok) — **REGRESSED vs iter1 verified-20's 41.5%.**
- MATH-500: format_rate 1.0, **acc|fmt 0.845, pass 0.845** — high.
- Stage B: 388 in-band, 363/388 = 93.6% program-verifiable, 363 solver rows.
- Solver step (no fla): **~12.6 min/step** (755s) → ~3.5× over the 44-min baseline; **fla was barely adding anything** (its smoke first-step ~696s) — dropping fla cost ~nothing AND kept the reward service working. Realized cycle ~7h (from ~20h). The OlympiadBench regression is one data point with the clip-lineage questioner seed — revisit whether clip80-trained Q hurts generalization vs base/verified Q.

**INCIDENT — auto-restart false positive lost iter2's checkpoint (~4h recompute).** The overnight monitor's stall detector (idle GPU + main-log frozen, ~8min window) FALSE-tripped during iter2's *benchmark eval*: the eval runs on 1 GPU at intermittent util and writes to `oly_cont.log`/`m500_cont.log`/RUN log, NOT main `cont_verified_it2.log`, so the main log looked frozen. `restart_verified.sh` wiped `cont_it2_*` (resume-hang avoidance) and relaunched iter2 — destroying the just-completed iter2 solver (not yet HF-uploaded; eval ran before the upload step). **Eval numbers survived** (already on oly/m500 logs). Relaunched iter2 recovered and is healthy. **Fix:** stall = max-mtime across ALL pipeline logs (it2/it3/it4 + RUN + oly + m500) AND GPU idle AND ~20min window (5 polls); eval phases no longer false-trip. CLAUDE.md updated; Brev restart count in /tmp/mon_brev_rc.

## 2026-06-23 — Experiment watchdog: cheap-path validated end-to-end on a live run

Building an always-on watchdog so runs stay at max speed and "alive-but-broken" jobs get caught/killed/relaunched without hand-babysitting. Design + all cheap components validated today (full design in memory `experiment-watchdog`).

**Design (user directives):** local laptop loop kept awake with `caffeinate` (started detached, `pkill caffeinate` to stop); FULL autonomy — fix/relaunch AND **kill malfunctioning runs**, bias hard to act ("cost of not acting is larger"), call only when unresolvable. **No fixed action library** — the model gets agency. Models: **DeepSeek** for the frequent health-judge, GPT-5.5/Claude on escalation. Core job is **semantic health diagnosis, not crash detection** — the dominant failure is "a run keeps running while clearly not working" (the concurrency-1 70× defect, flat-reward/no-learning, degenerate output).

**Validated today:**
- **Alert channel** (`~/Documents/GitHub/persona-companion/alert.py`): places a Vapi call via a TRANSIENT announcer assistant (built-in gpt-4o-mini + the persona's Cartesia voice) — robust to the companion server/tunnel being down. Verified live: clean call, `assistant-said-end-call-phrase`, $0.01. Root cause it "never called" before = laptop asleep (nothing alive) + a Cloudflare 1010 block on urllib's default UA (fixed with a normal User-Agent). See memory `persona-companion-alerts`.
- **State collector** (read-only ssh): `squeue`/`tmux ls`/`nvidia-smi`/log-tail across Empire, Unity, Brev — works.
- **DeepSeek judge** (`deepseek-chat`, JSON out): correct on synthetic ("DEAD, throughput far below expected") AND on REAL Brev `iter234` state → HEALTHY 0.9, correctly read GPU1-3 idle as the normal generation phase (not a stall), service at ~5000 tok/s as near-ceiling, zombies as old; emitted watch-condition "step-0 past 40 min → re-check". 601 tokens (~$0.0002).

**Cluster snapshot (06-23 ~06:28):** Brev `iter234` healthy (R-Zero iter2-4 cont, Qwen3.5-4B, step 0/6 in gen+reward phase, reward svc 51% of 2250 prompts @ ~5000 tok/s, GPU0 100%); Empire IDLE (no jobs, stale May tmux — wasted capacity); Unity job 61050175 `rztraino` PENDING; old 36-day defunct VLLM zombies on Brev (harmless).

**Pending:**
- [ ] Package collect→judge→act into a standing `watchdog.py` + run registry (host/session/log/expectations) + act layer (kill/relaunch over ssh, escalate, `alert.py` when stuck) + `caffeinate`-wrapped loop.
- [ ] Per-run "what healthy looks like" expectations (most generic: progress, util-vs-ceiling, reward trend, output sanity).
- [ ] Decide watchdog code home (self-play repo vs standalone).

---

## 2026-06-23 — Speed optimization: solver step 44min→~7-12min; fla broke reward service (root-caused)

Optimized the solver training (the ~14h Stage-C elephant) WITHOUT changing the experiment (user constraint: no token/n/step/reward changes; near-equivalent kernels OK; 15h/iter unacceptable).

**Solver step (batch 256×n5, 4096 max): 44min → ~7-12min.** Breakdown of the win (verl `timing_s`):
- `use_remove_padding=true` + `attn_implementation=flash_attention_2` (in `iteration_rzero.sh` COMMON_TRAIN): packing eliminates padding-token compute → ~1.8× (44→~24min). flash-attn 2.8.3 installed.
- `flash-linear-attention` (fla) + `causal-conv1d` kernels for Qwen3.5's 24 linear-attn layers (else torch fallback): another ~2× → `update_actor` 206s vs clip80's ~2560s = **~12×** on the update; first step 696s (incl one-time Triton JIT), steady ~7min.

**BUG (root-caused): fla deadlocks the standalone reward service** (`vllm_service_init/start_vllm_server.py`). With fla installed, the questioner Stage-A reward service hung at `Processed prompts 0/N, 0.00 toks/s`, GPUs idle, whole questioner frozen at step 0/6. Isolated by reverting offload (still hung → not offload) leaving fla as the only change; the service ran fine pre-fla. **Fix: `pip uninstall flash-linear-attention causal-conv1d`.** Service then healthy at **~3,800 toks/s** (batched — also showed the earlier "concurrency-1 / 104 toks/s" was a transient, not a standing problem → the planned A/B reward-batching refactor is likely unnecessary). Trade-off: solver keeps flash+`remove_padding` (~1.8×, ~24min/step) but loses fla's extra ~2×. fla-for-training-only (env-isolated from the service) is a future lever.

**Other config gotchas hit (all in `iteration_rzero.sh`/launcher):** rollout `enforce_eager=false` → CUDA-graph capture fails on A100(cap 8.0)+Qwen3.5+TP2 (`cancelled`) → keep `true`. grad-ckpt off + `ppo_max_token_len=32768` → OOM offload-off → keep grad-ckpt on, `ppo_max_token_len=12288`. **Final working config:** offload-ON (offload-off hangs the questioner handoff), `S_GPU_MEM=0.55`, flash+`remove_padding`, no fla, grad-ckpt on, `ppo_max_token_len=12288`, `enforce_eager=true`. iter234 relaunched 06:09, service healthy. Projected cycle ~4.5-8h (solver-dominated), down from ~20h.

**CLAUDE.md updated** with: "Move fast" (time is binding), "Fast debug loop" (tiny smokes, don't reload vLLM per-change), and "Monitoring cadence + auto-diagnose" (one loop over ALL runs, 60-90s in danger zones, hourly heartbeat when stable, trip-on-STALL = idle GPUs + frozen log ~3min, auto-relaunch). Lesson: a launched job ≠ working; verify progress within ~2-3min; lost ~3h once assuming a smoke trained when it crashed on a deleted parquet.

## 2026-06-22 — PERF: rollout runs at concurrency≈1 (~70× too slow); theoretical optimal cycle ~30min

**Headline:** the self-play loop is generation-bound and running **~70–300× below hardware capability**. Root-caused, benchmarked, and planned the fix.

**Diagnosis (Brev A100-80GB):**
- verl per-step timing: `gen ~1,820s` vs `update_actor ~80s` → generation is ~95% of every step; FSDP offload is irrelevant (update is tiny). Earlier "~2h update" was the OLD big-batch (256) attempt; current small-batch questioner update is ~80s.
- Realized throughput ~24–104 tok/s = **single-stream** (reward service logged `output: 104 toks/s`). KV cache had **184× concurrency headroom unused** (1.5M tokens, max_num_seqs 1024) → not batch/cache limited.
- GPU util by stage: standalone vLLM (`evaluate.py`) + raw benchmark = 100% util; **verl agent-loop training rollout = ~25% util**. So the verl/AgentLoop rollout submits generation at ~concurrency-1, not the hardware.
- **Localized:** the slowness is the **questioner stage (A) reward path** — solver-scoring of generated questions (n=10 difficulty + verifier). The **solver stage (C) batches fine** (clip80 solver: 1,110 prompts/59s, 100% util) because its reward is a correctness check (no generation).

**Benchmark (raw vLLM, GPU0, Qwen3.5-4B, this exact model):**
- concurrency 1 → 109 tok/s; 64 → 4,944; 256 → 7,769 (512-tok out).
- With 2048-tok outputs: conc 256 → 7,034; **512 → 7,305 (peak)**; 1024 → 6,881. → **per-GPU ceiling ~7–7.8k tok/s, plateaus by conc ~256** (decode is memory-bandwidth-bound). DP scales ~linearly: 3 GPU ≈ 21k, 4 GPU ≈ 28k.

**Reward-service internals (`vllm_service_init/start_vllm_server.py`):** solutions = `model.generate(questions, n=10)` (line 179); verifier = SEPARATE `model.generate(code_prompts, n=1)` (lines 289–305). Two sequential calls. `enforce_eager=True` on the service (CUDA graphs off). With `VERIFY_SUBSAMPLE=1.0` both passes cover all questions → can be merged. Verifier Python tool (`question_evaluate/verify.py:144`): `subprocess.run(["python3", tmpfile], timeout=10)` — fresh cold interpreter per program (thread-parallel, no pool).

**Token budget per iteration ≈ 50M decode tokens** (Stage A ~11M, B ~21M, C ~18M); **~70% is the solver scoring questions** (n=10 + n=9). **Theoretical optimal cycle (4×A100, batched+async): ~30 min bf16; ~15–20 min FP8; ~8–10 min if n=10→5.** vs current ~9–10 h → **~15–20× available.** Eval is negligible (~30s at ceiling); updates hide under gen if async; Python tool negligible if pooled. Binding floor = memory-bandwidth-bound decode of those ~50M tokens.

**Fix plan (ordered):** (1) batch the rollout/reward (conc 1→256) — most of the gap; (2) merge solutions+verifier into one `generate()` w/ per-request SamplingParams; (3) `enforce_eager=False` on the service; (4) `N_SERVICES=3` DP across all GPUs (needs offload-during-gen to free training GPUs); (5) async rollout/train overlap; (6) FP8 inference (~2× decode); (7) persistent Python sandbox pool; (8) design lever n=10→5. Validate offline on a free GPU BEFORE baking into a run.

**Decision (user):** let **iter2 (Brev) + clip80 (Cornell) finish** their current iteration and bank results (iter2 = 1,075 in-band / 98.4% verifiable), then **STOP before iter3** (watcher kills tmux iter234 after iter2 HF upload). Unity double-clip stays queued (patch batching in before it starts). Then rebuild the optimized pipeline. Benchmark scripts: `/tmp/vllm_bench.py`, `/tmp/vllm_bench2.py` (Brev).

---

## 2026-06-22 — Launch recovery: iter2-4 continuation (cont) + clip80; Stage-A resume-hang root cause

Brought up the two pending runs the user wanted live before sleeping: **iter2-4 verified continuation** (seeded from verified-20 = `rzcev_it1_verified_s2/global_step_20`, OlympiadBench 41.5%) and **clip80** (CLIP_EASY=0.80).

**Brev iter2-4 hang — ROOT CAUSE = stale-checkpoint resume artifact (not offload, not AgentLoop deadlock).** Two prior relaunches of `iter23_continue.sh` hung in **Stage A** (questioner GRPO). Log evidence: `Found checkpoint .../cont_it2_verified_q/global_step_6` → `Setting global step to 6` → `Resuming from ... global_step_6` → froze (GPU0 100%, no `Processed prompts`, then process exited). An *earlier* attempt had already completed Stage A (step 6) **and** Stage B (the `cont_it2_verified_verl08.parquet` training file existed), so the pipeline works end-to-end on Brev — but on relaunch verl auto-resumes the completed `global_step_6` (= max steps) and deadlocks instead of advancing. My earlier offload-on/off theory was wrong (it hung under both).
- **Fix:** `rm -rf models/cont_it2_* generated_question/cont_it2_* artifacts/cont_it2_*` (clear the stale checkpoint), relaunch fresh. Verified clean start 17:48 UTC: Stage A launching, `"Resuming from"` count = **0**. Cornell clip80 confirms fresh Stage A works (actively generating at step 0/6). Config: tmux `iter234`, FSDP_OFFLOAD=true S_GPU_MEM=0.55 (reliable-but-slow ~48min/solver-step on A100; offload-off was never the issue).

**Routing decision — Brev over Cornell for iter2-4.** Built `~/iter234_cornell.sbatch` (isolated `STORAGE_PATH=$HOME/rzero_run_cont234` so it can't clobber clip80's shared Vendi bank / temp_results; seeds solver directly from HF `Dylan1631/selfplay-verified-qwen35-4b-iter1-step20`; `--qos=cornell` since `standard` QOS caps at 48h). Submitted (977282) but est. start **2026-06-24** (cornell partition saturated, free GPUs reserved for higher-priority partitions). **Cancelled it** — 2-day wait + it would double-push to the same `selfplay-verified-qwen35-4b-iter{2,3,4}` HF repos as the Brev run. Brev (idle, ours) runs iter2-4 instead; sbatch kept on Cornell as fallback if Brev fails again.

**clip80 on Cornell (job 977116, alphagpu10):** healthy, Stage A questioner step 0/6, reward service generating 860 completions (slow first step, GPU0 100%). FSDP_OFFLOAD=false S_GPU_MEM=0.45. Uploads to `Dylan1631/selfplay-clip80-qwen35-4b-iter1`.

**iter2 progress (Brev):** A→B handoff CLEARED at 21:08 (the historic hang point) — Stage A done (questioner global_step_6), generate produced 998 questions, now band-scoring (3× evaluate.py) → filter → label → Stage C solver. iter2 in-band (in-training reward, 238q): **27.7%** ([0.3,0.8]); 72.3% trivial (≥0.9), 0% too-hard; mean_verified −0.56.

**Pending:**
- [x] Brev iter2 cleared smoke (questioner step1) + all 6 steps + A→B generate handoff.
- [ ] Verify Brev iter2 Stage C solver learns → eval → iter3/iter4. (watcher running)
- [ ] Verify clip80 clears band-filter → label → solver → evals (Cornell 977116, ~4h in).
- [ ] Collect iter2/3/4 + clip80 olympiad+math500 results; confirm HF uploads.

## 2026-06-22 — Double-clip challenger-reward experiment (Unity, queued)

New reward-shaping experiment on the challenger, patched into Unity's `examples/reward_function/caller_penalty.py` (env-gated behind `DOUBLE_CLIP=1`, compiles clean, inert for other runs):
1. **Double clip (difficulty):** replace continuous `min(score,1−score)` with a flat band indicator — in-band `[BAND_LO=0.30, BAND_HI=0.80]` → constant `BAND_FLAT=0.5`; outside (too easy OR too hard) → 0. Band matches data filter (`MINSC/MAXSC=0.3/0.8`).
2. **Stop-update difficulty:** if batch in-band fraction > `BAND_GATE=0.80`, zero the difficulty term for the whole batch.
3. **Stop-update verifiability:** if batch verifiable fraction > `VERIF_GATE=0.90`, zero the verifiability term — but do NOT flatten verified values (only 0..3/3 resolution). "Verifiable" = graded `verified > VERIF_OK` (0 ⇒ ≥2/3 program agreement).
Batch md header now logs `inband=.. [GATED] verif=.. [GATED]`. Seeds from base Qwen3.5-4B (sibling to clip80), iters 1–5, per-iter olympiad+math500 evals + HF push `selfplay-doubleclip-qwen35-4b-iter{N}`. Patcher at `/tmp/patch_double_clip.py` (Unity), sbatch `/work/pi_general_dartmouth_edu/dylan/unity_double_clip.sbatch`.

**Compute:** Unity job **61033801**, `-p gpu --gres=gpu:a100:4 --qos=long -t 7-00:00:00 --mem=400G`. The old "Unity too small" failure was a 64G `--mem` cap on the prior hold (`60905723`), not the hardware — uri-gpu001 has 515G RAM; requested 400G. Released the idle 64G hold to free fairshare. All Unity A100 nodes currently full → Slurm est start ~2026-07-01; **user chose to wait in queue** rather than preempt/shrink/run-on-Brev.

---

## 2026-06-22 — Verified arm co-evolution over 3 iterations (rzrun): metrics + generation funnel

Reconstructed the full per-iteration trajectory of the 3-iteration verified run (`rzrun`, Qwen3.5-4B, 06-19/20). Difficulty + diversity recomputed from `artifacts/rzrun_it{1,2,3}_verified/challenger_batches.md` (iter1 filtered to post-23:27 batches to drop ~22 stale appends from an earlier run that polluted the same file); verifiability from the B-stage judge logs; well-posed via GPT-5.5 on the band-filtered judge sets.

| iter | total gen | in-band | band yield | verified (consensus) | solver rows | difficulty (mean consistency, ↓=harder) | diversity (vendi, ↑) | verifiability | well-posed (GPT-5.5) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 188 | 15 | 8.0% | 9 | 9 | 0.839 | 18.5 | 60% (9/15) | n/a* |
| 2 | 178 | 12 | 6.7% | 11 | 11 | 0.874 | 33.3 | 92% (11/12) | 67% (8/12) |
| 3 | 190 | 26 | 13.7% | 15 | 15 | 0.631 | 42.8 | 58% (15/26) | 69% (18/26) |

\*iter1 judge.jsonl + verified.parquet were purged by disk hygiene; only 160-char-truncated questions survive in challenger_batches.md → well-posed not recomputable.

**Trajectories:** diversity ↑↑ (18.5→33.3→42.8, reaching ~50% of MATH-500 golden 85.6); difficulty → harder (consistency 0.84→0.63, in-band yield 8%→14%, solver rows 9→15); well-posed ~stable 67–69%; **verifiability dips hardest at iter3 (58%)**.

**Key findings:**
- **Co-evolution works**: across iterations the challenger makes questions harder AND more diverse, with ~constant total generation (~185/iter) but ~2× the in-band/usable yield by iter3.
- **Verifiability is the bottleneck**: the 3-program verifier confirms fewer of the harder iter3 questions (58%); GPT-5.5 says 69% well-posed but only ~38% well-posed AND program-verifiable → good hard questions get discarded. Fix: stronger verifier / GPT-5.5 fallback when programs return None.
- **Vendi reward worse than BLEU rep_penalty for diversity**: the vendi-reward run (`rzc`, 1 iter) collapsed to vendi 13.9 (~14 modes / 1661 q) — *below* every rzrun iteration — because marginal-novelty-vs-accumulating-bank decays to 0 as the bank fills. rep_penalty (this rzrun) kept diversity rising. Consider reverting to rep_penalty or rewarding absolute (not marginal) diversity.
- **Difficulty regime is the model, not batch**: Qwen3-4B-Base produces hard questions (band 66%, consistency 0.3, matches R-Zero); Qwen3.5-4B is too strong (band ~19%, consistency ~0.9). Confirmed at step 1 (pre-training), so not a batch-size confound.
- **Disk purge cost data 3×** (iter1 checkpoints, iter1 judge file, intermediate solvers) → motivated standing practice: always upload to HF (Dylan1631) + per-iter small evals + wandb (entity dylanpwilson2005-dartmouth). See memory.

---

## 2026-06-22 — OlympiadBench (harder, calibrated): verified step-20 = 41.5%, +8.5pp over base, +7pp over majority

MATH-500 was too easy (base 0.79). Re-evaluated on OlympiadBench (zwhe99/simplerl-OlympiadBench, n=200, k=1, max_tokens=8192, format_rate=1.0 everywhere → no truncation, real reasoning). Base = **33.0%**, matching the predicted ~33% → eval harness calibrated.

| arm | acc \| fmt | vs base |
|---|---|---|
| base (Qwen3.5-4B) | 0.330 | — |
| majority (vanilla R-Zero) | 0.345 | +1.5pp |
| verified step-8 | 0.350 | +2.0pp |
| **verified step-20** | **0.415** | **+8.5pp** |

**Verified step-20 is the standout: +8.5pp over base, +7pp over majority** (SE≈3.5pp at n=200 → ~2.4σ vs base, likely significant; majority/verified-8 gains within noise). Step-8→20 (0.35→0.415) shows more solver training compounds. Program-consensus labeling beats majority-vote much more clearly on hard problems than on MATH-500 (where it was +3.8pp). Notable: despite easy self-generated questions (band 19%), well-posed+correctly-labeled training still transfers to hard competition math. Eval logs: Brev `/home/nvidia/rzero_run/oly_{v8,v20,maj}.log`, `eval_olympiad.json`. Caveat: k=1, n=200 — a paired/larger-k pass would firm significance.

**Infra:** the CUDA fix (NVCC bypass + lib symlinks + ninja PATH) was ALSO needed on Brev for the 8192-token olympiad shape (fresh flashinfer compile; m500's 4096 ran off warm cache masking the latent skew). Now applied uniformly Brev/Unity/Cornell.

---

## 2026-06-21 — CUDA fix: making the frozen Brev env (vLLM/Qwen3.5) compile on fresh HPC nodes (Unity + Cornell)

Rebuilt envs from `brev_freeze.txt` on Unity & Cornell could `import torch` + `torch.cuda.is_available()` but **vLLM-Qwen3.5 failed at engine init** — the Qwen3.5 hybrid (Gated-DeltaNet) + flashinfer kernels JIT-compile at load and broke. Three distinct causes, all because the fresh nodes have **no system CUDA** (only the pip `cu13` libs, which lack dev symlinks) and a **version-skewed frozen dep set**:

1. **`ninja` not found** (Unity) → `ninja` is in `envs/rzero/bin`; that wasn't on PATH. Fix: prepend `envs/rzero/bin` to PATH.
2. **"CUDA compiler and CUDA toolkit headers are incompatible"** (both) → frozen set has `nvidia-cuda-nvcc 13.2.78` but `nvidia-cuda-runtime 13.0.96` (CUDART 13.0). flashinfer's bundled CCCL (`cuda/std/__cccl/cuda_toolkit.h:41`) hard-errors on nvcc≠CTK. Fix: `export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"` (documented escape hatch; same major 13, safe). Also `CUDA_HOME=.../nvidia/cu13` with `cu13/bin` on PATH so nvcc=13.2 is used.
3. **Linker `cannot find -lcudart` / `-lcuda`** (both) → pip ships `libcudart.so.13` (no unversioned `.so`), no `lib64/`, no `libcuda` stub. Fix (symlinks in `.../nvidia/cu13/`): `ln -sfn lib lib64`; `ln -sf libcudart.so.13 lib/libcudart.so`; `ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 lib/libcuda.so` (driver lib on the GPU node).

**Validated:** Unity `VLLM_GEN_OK` (17×23→391 correct). Brev "worked" only because its flashinfer kernels were pre-cached (mismatch latent). Env files: Unity `/work/pi_general_dartmouth_edu/dylan/unity_env.sh`, Cornell `/mnt/home/ch2263/rzc_env.sh` (both updated). **Both HPC clusters now usable for training + eval.**

---

## 2026-06-21 — GPT-5.5 audit of verified-arm well-posedness: labeling is reliable, degeneracy is the real leak

Adjudicated all **102** program-consensus ("verified", majority_agrees=True) questions from `rzcev_it1_verified_judge.jsonl` with **GPT-5.5** (blind: judged well-posedness + solved independently, no program answer shown), `/tmp/gpt55_judge.py` → `/tmp/gpt55_wellposed.json`.

| metric | result |
|---|---|
| well-posed (unambiguous, single answer) | 96/102 = **94%** |
| labeling correct (consensus == GPT-5.5 answer) | 98/102 = **96%** |
| degenerate (well-posed but vacuous → 0) | 28/102 = **27%** |
| well-posed AND non-degenerate (useful) | 69/102 = **68%** |

**Program-consensus labeling is RELIABLE — earlier mislabeling suspicion was wrong.** Cases like `['64','0','0']`/`['9908','0','0']` (I'd flagged as buggy-programs-outvoting-correct) are confirmed by GPT-5.5 to be genuinely 0 (e.g. "no factorial has exactly 5 trailing zeros; v5 jumps 4→6 at 25" → empty set). The `64`/`9908` were the buggy programs; majority-vote MIN_AGREE=2 correctly filtered them. All 4 GPT-vs-program disagreements are cases GPT judged NOT well-posed (returned NONE) — zero cases of program getting a real answer wrong.

**The real leak is DEGENERACY (27%), a verifiability reward-hack.** Vacuous questions (impossible constraints → answer 0) are trivially program-checkable, so they earn the verifiability reward without requiring reasoning. Examples (GPT reasons): "any four integers contain two of same parity → even difference → empty"; "V={0..78}, total 3081 odd → no equal partition → 0"; "three distinct digits ⇒ product 0 but digit-sum positive → empty". Plus ~6% not-well-posed = leaked meta-text ("Wait, re-reading the definition…" — the model's thinking bleeding into the question) or genuine ambiguity (conflicting bounds, ambiguous quantification).

**Implication:** verifiability machinery genuinely works (94% well-posed, 96% correct labels) — the hard part (4B self-generating verifiable questions) is solved. Combined with the flat-difficulty finding, the fix set for the next run is: (1) **non-degeneracy gate** (require non-empty solution set / reject trivial-0), (2) **strip leaked meta-text**, (3) **difficulty-weight fix** (full-scale uncertainty + max(0) floor). Tooling: `gpt-5.5` via OpenAI API (key in `.env`), needs certifi SSL context in detached procs.

---

## 2026-06-21 — Difficulty curve + challenger reward audit: we diverge from the R-Zero PAPER (uncertainty half-scale, no max(0) floor)

**Difficulty reward policy (majority/replication arm, `caller_penalty.py` @ commit 6935d2e):**
```
uncertainty = min(score, 1-score)            # score = solver self-consistency ∈[0,1]; tent peaks 0.5; -1 if no question
penalty     = cluster_share_per_problem(...)  # BLEU-cluster share |C_k|/B ∈[0,1]
final_score = uncertainty - penalty + verif   # verif=0 for majority; NO max(0,·)
```

**Difficulty curve (per GRPO step, mean solver self-consistency `score`; from `artifacts/rzrun_it{1,2}_majority/challenger_batches.md`, 256 questions each):**
- iter1: mean score **0.886**, mean uncertainty ~0.02, band(0.3–0.8) ~10–25%, easy(≥0.9) 74–100%/batch
- iter2: mean score **0.788**, mean uncertainty ~0.00, band ~10–36%
- Reward curve (critic/score/mean) it1: −1.08 → −0.48 → … → +0.15. The climb is almost entirely from cutting repetition/format-fails, NOT from raising difficulty. **Difficulty essentially flatlined at "too easy" (~0.8–0.9, far from the 0.5 target).**

**Divergence from the R-Zero PAPER** (arxiv 2508.05004; verified against upstream `caller.py` + paper):
- Paper: `r_uncertainty = 1 − 2|p̂−½|` (max **1.0**); `r_rep = λ|C_k|/B`, λ=1; `r = max(0, r_uncertainty − r_rep)`.
- Ours: uncertainty = `min(s,1−s)` (max **0.5** = HALF the paper); penalty = cluster-share λ=1 (matches paper); **no max(0,·) floor** (paper floors at 0).
- Upstream RELEASED `caller.py` is byte-identical to ours for uncertainty (`min(s,1−s)`) and has **no penalty at all** — i.e. R-Zero's own released code already contradicts their paper.
- Net: difficulty carries ~½ the weight vs the paper, and the (unfloored) penalty can drive reward negative and invert the signal → optimizer favors de-duplication over hardening. **Root cause of the flat difficulty curve, compounding the strong-base-model issue (Qwen3.5-4B solves challenger questions ~80–90% → uncertainty pinned near 0).**
- Anomaly to check: `challenger_batches.md` logs `rep_penalty` up to **1.50** though cluster-share ≤1.0 — 06-20 working tree may have had an extra penalty multiplier vs committed λ=1.

**They did NOT release the paper's runs** (checkpoints/datasets/logs go to user `STORAGE_PATH`); only code + reported metrics (+3.7 iter1, +6.49 after 3 iters on Qwen3-**4B-Base**). No reference artifacts to diff against.

**Fix for next run:** switch uncertainty to `1 − 2|p̂−½|`, restore `max(0, unc − penalty)`, confirm penalty λ=1 (cluster-share, not >1), and/or use a colder/weaker band-eval solver so uncertainty has range against the strong base.

---

## 2026-06-21 — ITER-1 EVAL IN HAND: verified (program-consensus) beats base AND majority on MATH-500 (significant)

The iter-1 comparison the run was designed to produce is complete and **positive for the verified arm.** Three MATH-500 evals (06-20 12:35, full test set n=500, k=1, max_tokens=2048, one RESULT each, both checkpoints valid HF models) on Brev `/home/nvidia/rzero_run`:

| Arm | model | format_rate | acc \| formatted | pass |
|---|---|---|---|---|
| BASE | Qwen/Qwen3.5-4B | 1.000 | 0.792 | 0.792 |
| MAJORITY (vanilla R-Zero) | majority_FINAL_solver | 0.998 | 0.808 | 0.806 |
| **VERIFIED (program-consensus)** | VERIFIED_FINAL_solver/global_step_8/actor/hf | 1.000 | **0.846** | **0.846** |

**format_rate ≈ 1.0 for all three including base** → no truncation on MATH-500 at 2048 tok, so the gain is real accuracy, NOT the format/termination artifact. Verified beats base **+5.4pp** acc and beats majority **+3.8pp** pass; majority's own gain over base (+1.5pp) is marginal (~1 SE).

**Paired significance (McNemar, n=200, same questions, `pp_base.jsonl`/`pp_verified.jsonl` 06-21 07:22):** base 0.780 → verified 0.845, delta **+6.5pp**, both_right=151, both_wrong=26, verified-only-right=18, base-only-right=5, discordant=23, **exact two-sided p=0.0106**. Verified fixes 18 / breaks 5 — a reasoning-improvement pattern, significant at p<0.05.

**Reconciliation with the training-side forensics entry below (same day).** That entry concluded the gradient fed mostly on the format/termination term and the questions were degenerate/easy — true of the *training signal* on it2/it3. But the held-out MATH-500 eval is the ground truth for "does the trained solver reason better," and at format-saturated conditions (base already 1.0 format_rate) it does, significantly. Not contradictory: the reward's format term dominated the *advantage spread*, yet the resulting solver still generalizes to +6.5pp correct reasoning on unseen problems. The verified arm's edge over majority (+3.8pp) is the cleanest signal that program-consensus labeling > majority-vote labeling at iter 1.

**Caveats:** k=1 (single sample/question — some decode variance); verified checkpoint is global_step_8, not the canonical 20. **In flight to strengthen:** (1) Brev `rzcev_it1_verified` re-run training the verified solver to step 20 with the rebalanced vendi/graded-verifiability rewards; (2) Cornell H100 env for a clean both-arm re-run (robust home, no Brev disk/lane traps); (3) majority arm re-run for parity (WashU eval 84230 abandoned — pinned to contended node a100s-2305, node-local checkpoint can't be relocated).

**FAILURE + FIX (06-21 ~19:35): step-20 verified rerun OOM'd at step 4/20.** `torch.OutOfMemoryError` in `ray_trainer._compute_old_log_prob → compute_log_prob → entropy = torch.logsumexp(logits)` — the LM-head logits over a packed micro-batch (`ppo_max_token_len_per_gpu=12288`) × 151k vocab. GPU 0 had 57.6GB in use (vLLM KV from `gpu_memory_utilization=0.7`), only 19.65GB free vs 19.90GB needed — failed by a hair on step 4 (first batch to pack to the full token budget). `save_freq=20` → nothing saved, fell back to base. **Root cause:** vLLM reservation (0.7) + log-prob entropy logits tensor (12288 tok) left no activation headroom — same logits-memory killer noted in the 2026-06-16 TRL entry. **Fix (proven):** `gpu_memory_utilization 0.7→0.55`, `ppo_max_token_len_per_gpu 12288→8192`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Relaunched cheaply** as solver-only on the *existing* 108-row parquet (`rzcev_it1_verified_verl08.parquet`) — skips the ~2hr questioner/generate/judge — via `~/rzero_run/rzc_solver2.sh` (tmux `rzc_sv2`, GPUs 0-3, → `models/rzcev_it1_verified_s2`, log `SOLVER2.log`). Same fix baked into the Cornell `rzc_full.sbatch` (would hit the identical OOM on H100-80GB).

---

## 2026-06-21 — Training-side forensics on the verified arm: difficulty / diversity / advantage

Pulled the actual trained artifacts off Brev (`/home/nvidia/rzero_run`) to answer "what difficulty/diversity were the questions, and what % of the learning signal did they carry?" Confirms the null/format-artifact conclusion from the *training* side, not just eval.

**Real trained counts** (the "26"/"35" were judge *candidates*, not the trained set; many judge rows abstained — e.g. `program_outputs=[None,'1024',None]`, only 1/3 programs answered, MIN_AGREE=2 → dropped):
- it2: 12 judge candidates → **11 questions trained**, 12 GRPO steps
- it3: 26 judge candidates → **15 questions trained**, 12 GRPO steps
- it1 verified parquet purged by keep-last-2 cleanup (12 steps in log)

**Difficulty — 2 coarse buckets, lopsided easy.** EVAL_N=4 means band 0.3–0.8 can only admit score 0.5 (2/4) or 0.75 (3/4). it2 `{0.5:2, 0.75:10}` = 83% easy; it3 `{0.5:7, 0.667:1, 0.75:18}` = 69% easy. 0.75 = base solver already gets it 3/4. No genuinely-hard items; difficulty has 2 levels, not a spectrum. Root cause: EVAL_N too small.

**Diversity — ~half degenerate.** Trivial answers (0/1/2/3): it2 6/11 (55%), it3 7/15 (47%); `"0"` alone = 6/15 (40%) in it3. it3 only 10/15 unique answers + one 579-digit degenerate answer. Topics cluster on integer/sum (narrow number-theory phrasing).

**Reward advantage / learning signal** (from `logs/rzrun_verified_it*.log`):
- last-step reward range = **[0.00, 0.90] every iteration** — floor 0.0 = format reward, ceiling never ~1.0. The gradient feeds on format/termination spread, NOT correct-vs-incorrect reasoning.
- score/mean ~-0.4 → ~+0.1 (the format term being learned), tiny updates: pg_loss 0.006–0.016, grad_norm ~2, adv spread (max-min) ~2.3–2.6 normalized.
- **solver `tool_calls/mean = 0.0` at every step** — solver makes zero python-tool calls in training (tool use is verifier-side).

**Conclusion:** ~100% of the run's gradient came from these 11–15 mostly-degenerate questions (nothing else existed), and ~all usable advantage traces to the format/termination term (reward pinned in [0,0.9], base already 3/4-correct), not reasoning. Training-side and eval-side agree: the model learned to format/terminate, not to reason. Reinforces that a trustworthy re-run needs EVAL_N≥10, a well-posedness/non-degeneracy gate, larger NUM_SAMPLES, and reporting acc|formatted.

---

## 2026-06-16 — TRL reference-check disambiguates "harness bug vs fundamental wall"

### Why
After ~15 configs of our custom `scripts/online_self_play_grpo.py` / `training_only_grpo.py` showing no held-set learning (flat or degrading; per-step reward "bouncing" e.g. m=0.422→0.301→0.320), the open question was: **is our custom `grpo_step` buggy, or is small-batch single-box GRPO on these questions fundamentally stuck?** The clean test is to run a reference implementation (TRL `GRPOTrainer`) on the *same* easy questions and compare.

### Infra wall finally cleared
TRL OOM'd repeatedly on one A100-80GB at `max_new_tokens` 3072 (needed ~78 GB — the LM-head logits over full-seq × 248k vocab in fp32 is the killer, ~48 GB alone). **Fix: halve sequence to `max_new_tokens: 1536`** (config `grpo_math/configs/train_trl_sanity.yaml`, k=4, prompts_per_step=4, lr 1e-5, LoRA r8, kl 0.01, 20 steps). Fits at ~50 GB, ~140 s/step. Running CUDA_VISIBLE_DEVICES=5, tmux `sp_trl`, log `outputs/slurm/trl_sanity.log`, wandb offline run `offline-run-20260616_103116-wps2janq`. Reward trajectory not in stdout (tqdm-only) — parse offline wandb datastore with `wandb.sdk.internal.datastore.DataStore`, history items use `nested_key` not `key`.

### Key finding (interim, run still in flight)
**TRL's per-step reward bounces identically to our custom harness:** step1 0.56 / 2 0.03 / 3 0.60 / 4 0.05 / 5 0.025 / 6 0.00 / 7 0.05. This strongly indicates the per-step "bouncing" is **NOT a bug in our `grpo_step`** — it is intrinsic to tiny-batch GRPO: with prompts_per_step=4 the per-step reward tracks *which 4 questions were drawn*, not learning. **The only valid learning signal is a fixed held-set before/after**, never the per-step reward curve. Reframes the whole prior negative.
- Confound for this run: `mean_completion_length` hits the 1536 cap on some steps (steps 2,7 = 1536 exactly → truncated → 0 reward → degenerate group, frac_reward_zero_std=1). Affects TRL and our harness identically (the GSM-hard easy set was calibrated at ~4096 tok). Eval (`scripts/measure_easy.py`) uses 4096 so the before/after measurement is not truncation-capped.

### VERDICT: WALL, not a harness bug
- **TRL run completed** (20 steps, train_runtime 2899 s, train_loss 0.07). Per-step reward over all 20 steps: **first-5 mean 0.253 ≈ last-5 mean 0.252 — flat**, same noisy bounce as our custom harness.
- **Merge clean**: `scripts/merge_lora_adapter.py` remapped 496 text LoRA keys, dropped 220 vision keys, hard-failed on none ⇒ adapter provably applied (not the silent no-op).
- **vLLM can't serve the merged model**: merged config saved as text-only `Qwen3_5TextConfig`, but vLLM's Qwen3.5 path demands the multimodal `Qwen3_5Config` wrapper (same multimodal/text split that caused the merge bug, now at serve time). Worked around with a direct HF-`generate` eval (`scripts/hf_measure_easy.py`, k=8, temp 1.1, top_p 0.98, max_new 4096, strict last-FINAL_ANSWER) — provably uses the trained weights, no vLLM.
- **Fixed before/after, same strict harness, GPUs 5 & 7 parallel**: base **0.175** → TRL-merged **0.163** (per-Q base [.25,.25,.12,0,0,0,0,0,.12,1.0] vs merged [.12,.25,0,0,0,0,0,0,.25,1.0]). **Null / hair lower — TRL the reference impl learns nothing here either.**
- **Two robust conclusions:** (1) the per-step reward "bounce" is intrinsic to tiny-batch GRPO (TRL bounces identically), NOT a bug in our `grpo_step`; (2) no-learning reproduces under the gold-standard reference implementation ⇒ the wall is the **setup**, not our code.
- **Why the wall (root cause):** under strict HF-generate scoring these "easy" questions are near-zero-signal — **6/10 are 0/8 for BOTH base and trained** (all-zero groups ⇒ advantage 0 ⇒ no gradient), only 1 question is reliably solved (8/8), ~3 sit at 1–2/8. The earlier "0.637 base / mid-band" calibration was harness-specific (OSP's forced-answer continuation inflates pass vs strict). Combined with the 1536-token truncation forced by single-A100-80GB memory (LM-head fp32 logits over 248k vocab), the trainable signal is too thin to move LoRA. This is a compute/signal wall, not an implementation defect.

### CORRECTION — the "wall" was signal starvation from bad question selection (fixable)
Re-reading the null: 6/10 of the easy_gsmhard set were 0/8 for BOTH base and trained ⇒ zero-variance groups ⇒ zero GRPO advantage ⇒ no gradient at all. And those questions fail on **large-number arithmetic precision** (8–10 digit answers like 6277334000), not reasoning — a target RL on a few questions can't move. The earlier 0.637 "mid-band" calibration was an artifact of OSP's lenient forced-answer-continuation scorer; under the strict scorer the set is pathological. So the prior negative was partly self-inflicted by question selection, NOT proof learning is impossible here.

### Mid-band selection experiment (in flight)
- **New tool `scripts/select_midband.py`**: scans a dataset, measures base pass@8 under the STRICT HF-generate scorer (the real training/eval scorer), shardable across GPUs. Scanned 80 GSM-hard questions at max_new 2048 (2 shards, GPU5+7). **Pass distribution: {0:31, 1:7, 2:5, 3:7, 4:3, 5:2, 6:4, 7:6, 8:15}** — 39% are 0/8 (arithmetic-precision-hard), 19% saturated 8/8, **34/80 (42%) signal-bearing (1–7/8)**. Confirms ~1/3 of GSM-hard is genuinely trainable once you measure under the right scorer.
- Built `outputs/midband_train.json` (25 Q, pass dist {1:6,2:4,3:4,4:3,5:1,6:3,7:4}, base mean ≈0.445) + `outputs/midband_heldout.json` (9 Q). JSONL for TRL: `outputs/midband_train.jsonl`.
- **TRL run launched** (`grpo_math/configs/train_trl_midband.yaml`, jsonl=midband_train, max_new 2048 to match selection budget, k=4, prompts_per_step 4, lr 1e-5, LoRA r8, 40 steps, GPU5, tmux `sp_trlmb`). Fits at 74.6 GB, ~216 s/step, ETA ~2.5 h. **Early reward (steps 1–7): 0.07/0.57/0.38/0.85/0.13/0.87/0.37 with `frac_reward_zero_std=0` on every step** — every group now carries gradient variance (vs sanity run's degenerate all-zero groups), completion lengths 640–870 (finishing, not truncating). Selection fixed the starvation.
- **RESULT (before/after, strict HF-generate, k=8, 2048, temp 1.1):**
  - train (in-dist, n=25): base **0.550** → trained **0.475** (−7.5 pts; per-Q deltas broadly negative, sum −0.075/Q, ~18/25 down-or-flat — NOT a single-Q fluke, ~2σ decline)
  - heldout (generalization, n=9): base **0.514** → trained **0.569** (+5.5 pts, but n=9 → ~1σ, within noise)
  - **Verdict: NOT learning.** Mild in-distribution degradation, heldout flat. Fixing signal starvation was necessary but NOT sufficient — TRL still degrades in-dist even with healthy reward (0.5–0.87, zero_std=0 every step). Same "training degrades" pattern as the custom harness.

### CORRECTION (high-n) — the −7.5pt decline was VARIANCE; true result is FLAT, and the policy barely moved
- k=8 was too noisy to trust. Re-ran at **k=32** via a NEW fast vLLM path (see below):
  - train (in-dist, n=800): base **0.4925** [0.458–0.527] vs trained **0.4863** [0.452–0.521] → **−0.6 pts, CIs almost fully overlap**
  - heldout (gen, n=288): base **0.5243** [0.467–0.582] vs trained **0.5382** [0.481–0.596] → **+1.4 pts, CIs overlap**
- **Verdict: FLAT / null — no significant change either direction.** The earlier −7.5/+5.5 were k=8 sampling noise (user correctly called it).
- **Root cause of flat = policy barely moved.** Logprob check (temp 0, first 8 tokens) base vs merged: shifts of ~0.005 nats (e.g. 'Thinking' −0.0559→−0.0494). lr 1e-5 + kl_beta 0.01 + LoRA r8 over 40 steps = a tiny KL-anchored update. Not "GRPO degrades" and not "wrong questions" — the update was too gentle to change pass rate. **Next lever: crank lr (→3-5e-5), drop/zero KL, raise rank (r16-32), more steps — actually move the policy, then re-measure at k=32.**

### ✅ FIRST POSITIVE RESULT — aggressive update produces measurable, GENERALIZING gains
Diagnosis (gentle update → policy barely moved) confirmed by fixing it. Aggressive config `grpo_math/configs/train_trl_midband_aggr.yaml`: **lr 4e-5 (4×), kl_beta 0.001 (10× lower), LoRA r16/α32 (2× rank)**, same 25 midband Q, 40 steps, GPU7, 2.1h.
- **Training reward CLIMBED**: first-5 mean 0.550 → last-5 **0.714** (gentle run was flat 0.55→0.55). KL grew step5 0.0005 → step20 0.0051 → step40 ~0.003-0.014 (~10× the gentle run) — policy moved, stably (no divergence).
- **Measured pass@32 (fast vLLM path), base vs aggressive-trained:**
  - train (in-dist, n=800): 0.4925 → **0.5550** (+6.3 pts; per-Q 18 up / 6 down / 1 flat)
  - heldout (generalization, n=288): 0.5243 → **0.5972** (+7.3 pts; per-Q **7 up / 0 down / 2 flat — ALL non-negative**)
- **Heldout gain ≥ train gain ⇒ genuine generalization, NOT memorization.** The all-non-negative heldout pattern is the cleanest signal.
- Significance caveat: vllm_measure CIs assume independent samples (n=800/288) but samples cluster in 25/9 questions → true CIs wider, per-split ~1.5-2σ (borderline). Strength is the CONSISTENCY: both splits up together + heldout monotone + train-reward up + gentle-run flat control. Real positive, not bulletproof.
- **Recipe that worked = (1) mid-band question selection under the strict scorer [kills signal starvation] + (2) update aggressive enough to move the policy [lr 4e-5, KL 0.001, r16].** Earlier flat/null was under-powered update, NOT a wall.
- Next to harden: more heldout questions (n=9 → 40+), a seed-repeat, and push further (lr/steps/rank) to see how far pass climbs before overfit (train≫heldout) appears.
- Artifacts: `outputs/trl_midband_aggr/{checkpoint-40,merged_full}`, served `mb_aggr` on :8002.

### ⚠️ CRITICAL CAVEAT — the gains are FORMAT/conciseness, NOT reasoning (decompose every eval!)
User asked "is the gain even net of formatting?" → decomposed pass into format_rate × (acc|formatted). On mid-band GSM-hard heldout, base vs bb2:
| budget | model | format_rate | pass | acc\|formatted |
|---|---|---|---|---|
| 1024 | base | 0.153 | 0.153 | 1.000 |
| 1024 | bb2 | 0.250 | 0.250 | 1.000 |
| 2048 | base | 0.493 | 0.479 | 0.972 |
| 2048 | bb2 | 0.618 | 0.597 | 0.966 |
- **The ENTIRE pass gain is format_rate** (@2048: +12.5pts format) while **acc|formatted is FLAT (~0.97, even slightly down)**. The model did NOT get better at math — it learned to FINISH/emit the answer within budget instead of rambling past the token cap. When base commits, it's already ~97% right on these Qs.
- **Root flaw exposed:** "mid-band" (pass 2-6/8) selected questions whose variance is TRUNCATION-driven, not reasoning-driven (these are length-hard, not reasoning-hard — GSM-hard = big-number arithmetic → long CoT → truncation). GRPO optimized the only available signal = conciseness. The benchmark gains (GSM8K +2.6, OOD +3.4, measured @1024) are almost certainly the same format effect.
- **So we have NOT demonstrated reasoning learning.** We demonstrated answer-emission/conciseness learning (real & generalizing, but not math).
- **FIX (methodology, also added to CLAUDE.md):** (1) ALWAYS report format_rate / acc|formatted / pass separately + the max_tokens budget; (2) use **acc|formatted as the reasoning metric**; (3) re-select training questions by acc|formatted band (REASONING-hard = model finishes but is WRONG, acc|formatted in ~0.3-0.7), NOT pass band; (4) eval at a budget high enough that format_rate isn't the bottleneck (or train the model to be concise as an explicit, separate goal). Tool: `/tmp/fmt_decomp.py` (format/acc decomposition).

### ✅✅✅ INDEPENDENT-BENCHMARK VALIDATION — gains generalize off-distribution (NOT memorization)
mb_bb (big-batch model, trained on 25 mid-band GSM-hard Qs) vs base on INDEPENDENT benchmarks, k=8, max 1024:
| benchmark | base | mb_bb | delta |
|---|---|---|---|
| GSM8K (different dataset, n=800) | 0.7063 | 0.7325 | **+2.6 pts** |
| GSM-hard disjoint (idx 1000-1110, never seen, unfiltered difficulty, n=880) | 0.1693 | 0.2034 | **+3.4 pts (+20% relative)** |
- Combined with mid-band heldout (+10.8), the model improves on (1) same-dist heldout, (2) a DIFFERENT dataset (GSM8K), (3) harder unfiltered OOD (disjoint GSM-hard). **Learning is real and generalizes.** CIs overlap modestly (real-but-modest), consistent direction with the larger heldout effect. Benchmark sets: `outputs/bench_gsm8k.json`, `outputs/bench_gsmhard.json`.

### ✅✅ BIG-BATCH RESULT — best generalization yet, confirms batch is the lever
k=32 before/after, three configs (base measured once, reused):
| split | base | aggressive (batch~1, lr4e-5, r16) | **big-batch (16-prompt, lr2e-5, r16, vLLM)** |
|---|---|---|---|
| train (in-dist, n=800) | 0.4925 | 0.5550 | 0.5200 |
| heldout (gen, n=288) | 0.5243 | 0.5972 | **0.6319 (+10.8 pts)** |
- Big-batch heldout CI [0.576–0.688] vs base [0.467–0.582] — **barely overlap, ~significant**; per-Q heldout ALL non-decreasing (8/9 up). Training reward climbed smoothly 0.369→0.562 (correct), KL 0→0.006.
- **Shape confirms batch theory:** big-batch has LOWER train (0.520) but HIGHER heldout (0.632) than aggressive tiny-batch (train 0.555 / heldout 0.597) ⇒ bigger batch = less overfit = better generalization. Exactly R-Zero's lever.
- **WORKING RECIPE: (1) mid-band selection under strict scorer + (2) vLLM-backed training (speed) + (3) bigger batch, moderate lr.** Served mb_bb on :8003.
- Next: scan C (gsmhard 80-200, GPU6) expands the question bank → enables grad_accum 32-64 (true R-Zero-scale batch) → expect even better heldout if trend holds.

### ✅ BREAKTHROUGH — vLLM-backed GRPO training works (20× faster gen, real batch feasible)
Was the whole time using HF `model.generate()` in-process for TRAINING rollouts (slow ~190s/4-completions) — NOT vLLM (vLLM was only for eval). That's why big batch was ~12h. Fix: wired TRL's `use_vllm` into `train_grpo_trl.py` (added `use_vllm`/`vllm_mode`/`vllm_gpu_memory_utilization`/`vllm_max_model_length` from config into GRPOConfig; `.bak` not needed, edited locally + scp).
- **Colocate smoke (GPU7, 3 steps): WORKS.** ~37s/step for 16 completions (~20× faster than HF), no OOM, **weight sync verified correct** (`sampling_logp_difference` ~0.013 → vLLM and training policy agree; NOT the silent-staleness LoRA-hotswap bug). Reward climbed 0.26→0.54→0.61 in 3 steps.
- **OOM tuning on single 80GB GPU (colocate = 2 model copies + KV + fp32 logits):** narrow window. `vllm_gpu_memory_utilization` must be ≥~0.25 (model is 0.225 of 80GB) AND leave room for training. 0.2 → "No available memory for cache blocks" (vLLM can't fit model). 0.3 + max_new 2048 → OOM (74GB+). 0.3 + 1536 → OOM (just barely, needed ~80.4). **Working config: vllm_util 0.3 + max_new 1024** (smoke-proven; mid-band finishers terminate in ~130–215 tokens so 1024 captures real answers). Also must set `vllm_max_model_length` (default pulls Qwen3.5's full 262144 → KV cache too big).
- **Dataset-size constraint:** TRL generation_batch = per_device(4) × grad_accum. grad_accum 32 → needs 32 prompts but mid-band train = 25 → "not a single sample, stopping at step 0". grad_accum 16 (16 prompts ≤ 25) works. **For bigger batch need a bigger question bank** (scanning gsmhard 80-200 on GPU6 now to expand).
- **RUNNING (GPU7, tmux sp_vbb, `train_trl_vllm_bb.yaml`):** vLLM colocate, grad_accum 16 (= 16 prompts/step, 16× our old effective ~1), lr 2e-5, kl 0.01, r16, max_new 1024, 30 steps, **~108s/step → ~54 min** (vs ~12h HF). Watcher armed → merge_full → serve → k=32 measure vs base on completion.

### Cross-project: R-Zero (hedge) config/dynamics comparison → batch is our missing lever
Studied R-Zero's verl GRPO (hedge project, ~/R-Zero on box + local) to extract learning-dynamics lessons. Same GRPO algo, very different scale:
- **R-Zero solver** (`scripts/solver_train_verified.sh` + `examples/config.yaml`): rollout_batch **256 prompts/step**, rollout.n **5** → **1280 rollouts/step**; global_batch 128; **lr 1e-6**; **FULL fine-tune (FSDP full-shard + param/optim offload)**; KL low_var_kl coef 1e-2 (as kl_loss); max_grad_norm 1.0; resp 4096; temp 1.0/top_p 0.99; max_steps 20; **4–8 GPUs, generation (vLLM services) SPLIT from training (FSDP)**.
- **Ours** (TRL): prompts_per_step **4**, k **4** → **16 rollouts/step**; lr 1e-5→4e-5; **LoRA r8–r16**; KL 0.01→0.001; single GPU (gen+train same device, HF generate); resp 1536–2048.
- **Dynamics**: R-Zero solver accuracy climbs MONOTONICALLY (0.387→0.407→0.421→0.441, grad_norm steady 0.24–0.84, kl~0). Ours BOUNCES (m 0.422→0.301→0.320). Same algo — the difference is **batch = 80× theirs** (1280 vs 16 rollouts/step). Our per-step reward just tracks which 4 prompts were drawn = the bouncing.
- **Recommendations (priority):** (1) **batch is THE lever** — simulate via grad_accum 32–64 (micro-batch 1) → effective 64–256 prompts/step, kills bouncing; (2) **stop cranking lr** — R-Zero's 1e-6 works because big batch → clean gradient; our 4e-5 on batch-4 = big steps in noisy directions; once batch is big, drop to ~1–2e-5; (3) **structural unlock = split gen from train** (TRL `use_vllm` server-mode across 4 GPUs, or FSDP) → enables 256-prompt × 4096-tok batches AND full-FT (higher ceiling than LoRA, why our policy "barely moved"); (4) KL ~1e-2 is fine — gentle-run failure was batch+lr, not KL.
- Our recent "aggressive" win (lr↑/rank↑/KL↓) was COMPENSATING for tiny batch. Bigger/cheaper/more-stable win = batch.

### Fast eval path FIXED (full multimodal merge → vLLM)
- `scripts/merge_lora_full.py`: loads base via `AutoModelForImageTextToText` (the multimodal `Qwen3_5ForConditionalGeneration` class), applies the RAW adapter (keys already target the `language_model.layers` multimodal tree → 716/716 matched, NO remap), `merge_and_unload`, saves full model. Config stays `model_type: qwen3_5` (multimodal) so **vLLM serves it** (text-only `AutoModelForCausalLM` merge was rejected). Output `outputs/trl_midband/merged_full` (18.8 GB incl. vision).
- Served on GPU5:8002 (`--served-model-name mb_full`). Verified APPLIED via logprobs (differ from base) — greedy tokens were identical only because the update is tiny (no argmax flips).
- `scripts/vllm_measure.py`: parallel HTTP pass@1, k samples/Q, strict scorer, prints 95% CI. ~10x faster than HF generate. base via :8001, trained via :8002.

### RETRACTED — "vLLM silently does NOT apply LoRA to Qwen3.5"
- Earlier I concluded the LoRA hot-swap was a silent no-op based on byte-identical GREEDY output. That test is too weak: this trained model's update is so small it produces identical greedy tokens even though logprobs DO shift (confirmed on the fully-merged model). So the hot-swap may well have been working. **The online-loop "root cause" claim is WITHDRAWN pending a logprob-level re-test of `push_adapter`.** Do not rely on the earlier no-op conclusion.

### (superseded) earlier note — vLLM silently does NOT apply LoRA to Qwen3.5 (likely the online-loop root cause)
- Tried to speed up eval via vLLM LoRA hot-swap (avoid HF generate). Loaded the step-40 adapter onto the running base server (port 8001, launched WITH `--enable-lora --max-loras 8`), both raw AND key-remapped (text keys `base_model.model.model.layers...`, vision dropped). **Greedy (temp 0) output was BYTE-IDENTICAL to base in both cases** ⇒ vLLM accepts the load call ("Success") but applies nothing.
- Cause: the adapter is `all-linear` over Qwen3.5's hybrid arch — keys dominated by `linear_attn.in_proj_a/b` (linear-attention projections) which vLLM's LoRA/punica kernels don't map; it silently skips. `load_lora_adapter` returns Success regardless.
- **Implication for the online loop**: `scripts/online_self_play_grpo.py::push_adapter()` hot-swaps the freshly-trained adapter onto vLLM for the next round's rollouts via this exact API. If it's a silent no-op, **every round generated from the BASE model** — the trained policy never fed back into rollouts ⇒ would fully explain the online loop's persistent no-learning, independent of all the GRPO tuning. NEEDS VERIFICATION on the online path (compare base vs hot-swapped greedy output there too).
- Speed fix options (none done yet): (a) serve a FULLY-merged multimodal model in vLLM — merge into the multimodal class so config stays `Qwen3_5Config` (current merge uses AutoModelForCausalLM → text-only config vLLM rejects); (b) keep HF generate but install flash-linear-attention/causal-conv1d + bigger batch. (a) is the real fix and also unblocks fast eval.
- Open cheap diagnostic: greedy (temp 0) base-vs-trained pass on the midband sets — distinguishes genuine no-learning from GRPO mode-collapse lowering sampled pass@k (train_loss hit 0.023 at step 40 = possible collapse).
- Artifacts: `outputs/trl_midband/merged` (merged weights), `outputs/trl_midband/adapter_remapped`, `outputs/midband_train.json|jsonl`, `outputs/midband_heldout.json`, `scripts/select_midband.py`, `scripts/hf_measure_easy.py` (now takes `<model> <questions.json> [k] [max_new]`).

### Decision / next steps
- ~~This arm is **paused pending dedicated multi-GPU capacity**~~ (superseded — retesting with proper mid-band selection first)
- (original note, still true for full-length hard questions) Dedicated multi-GPU capacity (FSDP/ZeRO or TRL+vLLM-served rollouts) that would allow full-length (≥3072-tok) generation AND a question set that is genuinely mid-band under the *training* scorer. On one contended box the reference impl confirms it can't learn.
- Lean on the **R-Zero hedge** (GPUs 0–3, untouched) as the active line.
- GPU state left clean: my eval/merge/vLLM-merged tmux sessions killed; `sp_vllmA` (GPU4) + `sp_vllmB` (GPU6) left up but idle (base Qwen3.5-9B servers) — free to reclaim. Merged weights + adapter at `outputs/trl_sanity/`.

---

## 2026-06-11 — Agentic in-context proposer + program-consensus code judge replace question-DPO; full pipeline launched

### TL;DR
Question-DPO (the run1 blocker, cross-cycle holdout 0.71 > ln2) is replaced by an **agentic proposer**: the model proposes questions in-context conditioned on measured `[pass n/8 | verifier_programs_agree k/3 | note]` feedback lines, no gradients. Judging is replaced by a **3-program code-judge consensus** (model writes 3 independent Python programs at temp 0.6, subprocess-executed, majority = reference; <2 agreement = discarded). Validated over 8 test rounds on the Brev box; full pipeline (gen → solver GRPO with deterministic correctness reward → merge → GSM8K eval) launched 2026-06-11 06:52 UTC in tmux `agentic`.

### Validation arc (all on box, vLLM Qwen/Qwen3.5-9B port 8001, tmux logs in outputs/slurm/)
- **Self-verify ceiling**: forcing the proposer to solve its own questions ⇒ all 8/8 (proposer==solver). Removed self-solve.
- **Feedback alone insufficient**: R4→R5 chain without hardness coaching stayed 8/8 and drifted easier. The explicit "propose beyond the routine" coaching paragraph is load-bearing.
- **LLM judge (temp 1.4) noisy both directions**: false positives (wrong answers 7,5,5 credited) and false negatives (correct answer 100 rejected 4/8). Code judge is deterministic and caught both.
- **R7→R8 chained test** (`scripts/test_agentic_final2.py`, log `outputs/slurm/agentic_final2.log`): R7 pass=[2,8,7,8,8,8,8,8] mean 7.1, 2/8 trainable; R8 (with R7 feedback) pass=[7,6,2,8,8,7,8,7] mean 6.6, **5/8 trainable (1–7 band)**, 0 discarded both rounds, programs 3/3 nearly everywhere, no zero-trick collapse. Feedback measurably moves difficulty toward the band.
- Goldmine class confirmed: program-trivial/hand-brutal questions (lattice x²+y²≤2025 → 1/8) are perfectly verifiable.

### Production pipeline (launched, tmux `agentic`, log `outputs/slurm/agentic_run.log`)
1. `scripts/agentic_question_gen.py` — 4 parallel chains × 8 rounds × 8 questions, seeded with 24 real measured R6–R8 feedback entries, per-chain topic focus, anti-zero-trick rule, stronger calibration coaching ("≥5 of 8 must be as hard as past ≤6/8 questions"). Writes `outputs/agentic_bank/bank.jsonl` (question, reference_answer, pass_count, trainable_for_solver = ref∧1≤pass≤7; dedup across chains).
2. Gate: abort if <40 trainable questions.
3. `grpo_math/configs/train_agentic_correctness_qwen35.yaml` — solver GRPO on GPU1, LoRA r8, k=8, max_new_tokens 4096, 30 steps, **reward.mode=correctness** vs code-judge references (no LLM teacher at all).
4. `scripts/merge_lora_adapter.py` merges final checkpoint → `merged_final`.
5. `eval_gsm8k` 500q k=4 on merged model. Baseline eval of untrained model running in parallel on GPU2 (tmux `baseeval`, log `outputs/slurm/baseeval_gsm8k.log`).

### Code changes (local + synced to box)
- `grpo_math/trl/train_grpo_trl.py` `_make_pairwise_jsonl_dataset`: now carries `reference_answer`→`answer_text` column so reward.mode=correctness works with agent banks (verified: bare-integer ground truth parses; non-trainable/duplicate rows filtered).
- New: `scripts/agentic_question_gen.py`, `scripts/merge_lora_adapter.py`, `scripts/brev_agentic_run.sh`, `grpo_math/configs/train_agentic_correctness_qwen35.yaml`.

### Confirmed bugs (production path, still unfixed)
- `{question_bank_examples}` placeholder left literally unfilled in cycle-1 proposer prompts (seen live in smoke data).
- `run_python_verification_probe` (grpo_math/self_play/verifier.py:239) is a hardcoded ~3-template pattern matcher, not real code execution.

### Run results (2026-06-11, Brev box)
- **Generation**: 284 questions, 124 trainable (44% yield), 3h04m, 4 chains × 8 rounds. Calibration improved within chains (C1: 3/8 → 5/8 trainable R1→R2, incl. an exact 4/8).
- **Training**: 30 steps × 191s = 1h35m on GPU1 after two OOM iterations (fp32 logits buffer batch×seq×248k-vocab: fixed via max_new_tokens 4096→2048 + prompts_per_step 8→4 with grad_accum 2 = one k=8 group/step). **Caveat: weak signal** — 62-100% of rollout completions truncated at 2048 tokens (bank difficulty was measured at 8192), reward_correct/mean 0 most steps, 0.125-0.25 occasionally. Only 30 prompts visited.
- **MERGE BUG (critical, recurring class)**: TRL saves Qwen3.5 LoRA keys against the full multimodal tree (`base_model.model.model.language_model.layers...` + 220 `model.visual` keys), but AutoModelForCausalLM exposes `model.layers...`. PeftModel.from_pretrained loads NOTHING (all keys "missing", warning only) → first "trained" eval was bit-identical to baseline (pass@1 0.4760 both — that's how it was caught). Fixed in `scripts/merge_lora_adapter.py`: remap text keys, drop vision keys, hard-fail if any key unmatched. Sanity check confirmed all 248 text lora_B matrices nonzero (vision B's all zero) — adapter did train.
- **Baseline GSM8K (Qwen3.5-9B base, reason-first prompt, temp 1.1, k=4, 500q, 1024 tok)**: mean_reward 0.4570, format_rate 0.6070, pass@1 0.4760, pass@4 0.8600. Low format rate = many runs exceed 1024 tokens or miss the tag.
- **Trained GSM8K (merged_final_v2, clean merge, same eval settings)**: mean_reward 0.4400, format_rate 0.6010, pass@1 0.4260, pass@4 0.8600. **Null result vs baseline** (pass@1 −5.0 pts, ≈2σ at prompt level; pass@4 identical): 30 steps of truncation-starved reward (correctness mostly 0) taught nothing on GSM8K, mild pass@1 dip. The PIPELINE is proven end-to-end; the training config is the bottleneck, not the data.

### Follow-up experiments (same day, after user review)
- **Truncation theory proven directly**: 12 trainable bank questions × 2 solver attempts — at max_tokens 2048: 2/24 finished (2 correct); at 8192: 19/24 finished, 15/24 correct (62%, matching the bank's calibrated band). The first training run graded the model on questions it could not finish 92% of the time.
- **Per-step GSM8K evals** (merged via fixed script, 500q k=4): base 0.4760/0.8600 (pass@1/pass@4) → step10 0.4660/0.8400 → step20 0.4780/0.8420 → step30 0.4260/0.8600. Pure noise, no trend — confirms null.
- **8k rerun LAUNCHED** (tmux `agentic8k`, wandb https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/rqmxrwap): max_new_tokens 8192 to MATCH the bank calibration budget (user directive), prompts_per_step 1 × grad_accum 8 (micro-batch 1 is the only shape that fits 8192-token backward on A100-80GB: fp32 logits = batch×seq×248k×4B), steps 90, save_every 15, wandb on (key from box .env). ~14.6 min/step → ETA ~21h (~2026-06-12 14:00 UTC). Eval queued after via driver.
- **Multi-GPU tested and rejected**: 4-rank accelerate OOMs at per-rank bsz 2 (backward); at bsz 1 it works but only ~1.25× faster (719s vs 902s/step) because each rank still generates sequentially at micro-batch 1 and the slowest rank gates the step. Not worth restarting the running job. Real speedup requires TRL vLLM-served rollouts — untested risk: TRL↔vLLM LoRA weight-sync may hit the same `language_model` key-prefix mismatch that broke the merge.
- GPU usage answer for the record: first run used 3/8 GPUs (vLLM=0, train=1, baseline eval=2); per-step evals used 3-4; rerun is single-GPU by the analysis above.

### Restart as 7-GPU run (2026-06-11 ~18:50 UTC, user directive: use all GPUs + eval each round + markdown log)
- Killed the 1-GPU 8k run at step ~12 (wandb rqmxrwap). Key realization: multi-GPU at micro-batch 1 doesn't speed up steps (~15 min either way) but multiplies DATA per step — 7 ranks × grad_accum 8 = 56 completions = **7 question groups/step vs 1**. My earlier "only 1.25× faster" framing compared fixed global batch; scaling batch with ranks is the right move.
- **Live run**: tmux `agentic8k7g`, wandb https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/335aod23, GPUs 1-7, 60 steps × ~15 min ≈ 15h (ETA ~2026-06-12 10:00 UTC), `WANDB_LOG_MODEL=checkpoint` so adapter checkpoints upload as wandb artifacts.
- Driver (`scripts/brev_agentic_train_eval.sh`) now: trains → **merges + GSM8K-evals EVERY checkpoint** (every 10 steps, 3-way parallel on freed GPUs) → writes a markdown report `outputs/agentic_run_report.md` (config summary, per-checkpoint eval table incl. base row, artifact pointers).
- Bank sample reviewed with user: compound-constraint questions (e.g. "x²+y²≤1600, x even, y odd, x+y div 5" → 62, pass 5/8). Trainable pass distribution: 8×1/8, 8×2/8, 7×3/8, 8×4/8, 14×5/8, 25×6/8, 54×7/8.

### Diversity problem found + feedback mechanism validated (2026-06-11 evening)
- **User spotted heavy repetition in the bank.** Confirmed by clustering: 30/124 trainable were lattice x²+y²≤N variants, 19 floor equations, 16 divisor-counts; 15 questions shared the same 8-word opening. Causes: (1) the 4 chains never saw each other, (2) feedback lines carried NO diversity signal, (3) calibration coaching pointed at proven templates (exploit>explore).
- **Rejected lazy fixes per user**: no template bans, no jaccard gating. Instead, pure-feedback redesign of `scripts/agentic_question_gen.py`: model-generated structure labels (one cheap call per question), structure frequency report in the prompt (counts + WHY diversity matters, nothing forbidden), per-question `structure: <label> (novel|N produced before)` in feedback lines, cross-chain shared history, `--history_bank` flag seeds a new generation with the prior cycle's bank (this is the cycle-to-cycle feedback path; previously unwired). Gotcha fixed: 120-line history + 24k proposer max_tokens overflowed the 32768 ctx → history capped at 80 lines, proposer 16k.
- **Divtest result (2 chains × 3 rounds on idle GPU0, seeded with all 284 cycle-1 questions): feedback works.** Novel structures 10/16 → 14/16 → 7/16; trainable 6 → 5 → 9; round-3 passes [0,0,1,1,2,3,5,5,5,6,7,8,8,8,8,8] — best-calibrated batch of the project (56% trainable vs cycle-1 44%). R3's novelty dip = re-use of R1/R2 structures with freshly measured pass rates (explore-then-calibrate), not collapse. New structures cycle 1 never touched: phi/sigma functions, digit products, Legendre/factorial valuations, binomial divisibility, semiprimes.
- **Question banks now on wandb** as browsable tables + artifacts: run `question-banks` https://wandb.ai/dylanpwilson2005-dartmouth/grpo-math/runs/msdbgoh3 (cycle1_bank 284 rows, divtest_bank 48 rows). Local copies + human-readable questions.md on user's Desktop (~/Desktop/agentic_run_artifacts/). NOTE: training-run checkpoint artifacts appear on run 335aod23 only when checkpoints save (step 10, 20, ...).

### Tool-loop judge (user directive: judge must have python as a TOOL, not one-shot programs)
- vLLM restarted with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Qwen3.5 parser, vLLM 0.19.1) — native OpenAI-style tool_calls now work on port 8001. Probe confirmed (model emitted proper tool_call).
- New `scripts/python_tool_judge.py`: judge model gets a `python` tool (subprocess sandbox, 15s, stateless per call), iterates write→execute→see output→revise until `FINAL_REFERENCE: <int>` or `UNVERIFIABLE`; max 6 turns, ~2.3 tool calls/question avg.
- **Validation vs 3-program majority on 16 divtest questions: 14 agree, 1 resolved, 1 exposes ambiguity.** The resolved one: forbidden-points lattice path the old judge DISCARDED — tool judge got 12, hand-verified correct by inclusion-exclusion. The disagreement: permutation question ambiguous re 0/1-based indexing (23616 vs consistent 22320) — question-quality fault, not judge noise.
- Wired into `scripts/agentic_question_gen.py` as default (`--judge tool`: 2 independent runs temps 0.2/0.5, tiebreak 3rd run on disagreement; `--judge programs` = legacy). Smoke-tested: d(n)=d(n+1) → 63=known ref, runs agree. Prompt/output dumps also added (outputs/agentic_bank/prompts/C<i>_R<n>.txt).

### Architecture pivot: unified online loop + per-ARTIFACT independent judges (user directives, 2026-06-11 late)
- **Single rollout set** (user: "we should just do 1 set of rollouts"): new `scripts/online_self_play_grpo.py` — per step: proposer (in-context) → solver k=8 rollouts via vLLM with CURRENT LoRA adapter → judges AFTER → GRPO update on training GPU (repo's memory-safe logprob gather, micro-batch 1, KL via adapter-disable) → adapter hot-swapped into vLLM (`/v1/load_lora_adapter`, plain-text response!) → pass+agreement+structure feedback to proposer. Measurement and training are the SAME rollouts; calibration always matches the live solver and live token budget by construction.
- **Per-artifact independent judge rollouts** (user: reference-matching "empirically wrong" — e.g. ambiguous-indexing question 23616 vs 22320 where exact-match zeroes legitimate interpretations — and unscalable; future = "llm coding stuff, each judge checks the different repos"): NO precomputed reference. Each unique artifact gets 2 independent tool-loop judges (`judge_solution` in python_tool_judge.py: verify by independent computation, never trust candidate reasoning, judge within candidate's reasonable interpretation), 3rd on split or random ~34% subportion; majority = reward; mean inter-judge agreement per artifact = verifiability signal in proposer feedback (R_cons realized with execution-grounded judges). Artifact=claimed-answer dedup is a math-only optimization, no-op for repos.
- **Positioning**: this repo is the moonshot (open-ended artifacts, repo-checking judges); a parallel agent runs the hedge (R-Zero + verifiability).
- Smoke-validated incrementally: smoke1 (3 steps) — gradient flowed on a 2/8 group, KL=0 pre-first-update correct; smoke2 (full-detail dumps: proposer prompt 23k chars + raw thinking output, per-judge code+stdout traces, all rollout texts; adapter hot-swap verified IN-RUN, step-2 rollouts served by solver_step1); smoke3 (per-artifact judges) running. Full-detail report: ~/Desktop/agentic_run_artifacts/online_smoke_FULL_detail.md.
- Fixes en route: apply_chat_template returns BatchEncoding (tokenize via text); /v1/load_lora_adapter returns plain text not JSON (silent off-policy risk if push fails — now logged); vLLM relaunched with --enable-lora --max-lora-rank 16 + tool parser at 0.55 mem (shares GPU0 with smoke trainer).

### 7-GPU run CANCELED at step 27/60 (2026-06-12 ~01:50 UTC, user decision) + full online run launched
- Cancel reason: recipe flaw — HF-generate training rollouts clip at 8192 46-86% of the time vs ~21% for the vLLM rollouts the bank was calibrated with (training/measurement distribution mismatch), reward flat-noisy (~0.17, no trend over 27 steps). Checkpoints 10/20 + wandb metrics (335aod23) retained for reference.
- **Full online run LAUNCHED**: tmux `online`, `outputs/online_selfplay_run1`, 60 steps × 4 questions × k=8 @ 8192, trainer GPU1, vLLM GPU0 (rollouts via vLLM → no clipping mismatch by construction), per-artifact judges, wandb run name `online-selfplay-grpo`.
- smoke3 validated per-artifact judging (step 2: passes [5,3] both trained, split-claim 6-vs-5 resolved 2/2 vs 2/2). Adapter-push 400 root-caused: vLLM default `--max-loras 1` → restarted with `--max-loras 8` + unload-on-swap in loop.
- Desktop reports refreshed: `online_smoke_FULL_detail.md` (316KB, complete process: prompts/thinking/rollouts/judge code+stdout per artifact), `online_smoke_report.md` (checklist, final architecture).

### 7-GPU run post-mortem evals (canceled run, checkpoints evaluated anyway): NO improvement
- base 0.4760/0.8600 (pass@1/pass@4) → step10 0.4560/0.8380 → step20 0.4680/0.8440. Second clean negative for the two-phase recipe; cancel decision validated.

### Overnight run v3 LAUNCHED (2026-06-12 06:55 UTC) — hardened "best working version"
- v1 (sequential questions, no forced answers) ran 6 steps (preserved at outputs/online_selfplay_run1_v1_6steps), v2 superseded before step 1.
- v3 hardening: (1) **retries with backoff** on all loop+judge HTTP calls (transient vLLM error no longer kills the run); (2) **per-step try/except** isolation; (3) **forced-answer on clip** (legacy api_final_prefill behavior; integer still policy-sampled) with `forced_frac` logged per step as the DAPO guard metric — step 1 measured 0.031, so training-on-forced is marginal, watch for drift; (4) **cross-question parallelism** (4 questions' rollouts+judges overlap); (5) vLLM bumped 0.55→0.85 mem (trainer owns GPU1 now); (6) lesson: never `pkill -f <name>` over ssh where the command string contains <name> — it kills your own session (tonight's recurring exit-255s).
- Run: tmux `online`, outputs/online_selfplay_run1, 70 steps × 4 q × k=8 @ 8192, wandb `online-selfplay-grpo`. Step 1: passes [7,8,5,8], 2 trained, solver_step1 pushed, 23.5 min (incl. load). ETA ~12-15h.
- Industry-standard note (user asked): forcing answers on clip is standard in EVAL (s1 budget forcing), NOT standard in RL training (DAPO filters overlong instead); our guard = forced_frac + completion-length trend; hybrid fix identified (forced answers for feedback only, exclude clipped from gradient) if it drifts.

### KEY NEGATIVE RESULT (2026-06-13): paired early-question retest → NO learning (hypothesis DENIED)
- Test design: re-rolled the online run's own step 1-10 questions (40 Q, originally measured first-time-out at near-base policy, mean 6.28/8 = 78%) under the step-41 adapter, identical sampling (k=8, temp 1.1, 8192), reusing original judge verdicts (`--fast`: score by matching each Q's known-correct answer; `scripts/early_question_retest.py`).
- Result: mean 6.28 → 6.15, **delta −0.12 (SE 0.22)** — flat. Sub-8 subgroup (21 Q w/ headroom): −0.05 (SE 0.40). Hard ≤4/8 (8 Q): +0.62 (SE 0.62), within 1 SE = not significant. 22/40 flat, 10 up / 8 down = resampling noise around an unchanged policy.
- **Interpretation (pre-agreed grid): this is "no learning," NOT "learning that doesn't transfer." My distribution-gap hypothesis for flat GSM8K is DENIED.** The model is not improving even in-distribution on its own questions.
- New suspect = the GRPO update is too weak to matter: LoRA r8 @ lr 1e-5, losses ~0.01, advantage signal starved (most groups at high pass → low reward std → ~0 advantage; 0/8 & 8/8 groups skip). Healthy-looking loop mechanics (nonzero reward/KL, adapter advancing) masked a near-no-op update.
- 7-GPU run confirmed canceled (step 27); its checkpoint evals also showed no improvement (step10 0.456, step20 0.468 vs base 0.476) — consistent.

### Static diagnosis of the no-learning result (2026-06-13, NO GPU jobs — pure artifact analysis)
v3 completed all 70 steps (not a crash). Diagnosis from steps.jsonl + saved adapters + detail traces:
- **A. Weight drift** (`adapter_step*` LoRA delta ‖B·A‖·scale, CPU): monotonic 0.039→0.328 over 70 steps, all 496 tensors nonzero. → gradient flows, NOT a broken/no-op/tokenization-garbage update; but 0.33 total across ~250 layer-pairs = tiny nudge to 9B.
- **B. Signal starvation (DOMINANT)**: 280 groups → pass dist {0:12,1:7,2:7,3:7,4:10,5:18,6:21,7:43,8:155}. **155/280 (55%) were 8/8**, only 113/280 (40%) in 1-7 contributed gradient; **mean 1.61 trainable groups/step** (of 4), 7 steps had 0. Proposer's escalation feedback failed to bite — most rollout budget wasted on too-easy questions giving zero advantage.
- **C. Recurring-structure pass early→late: 6.56→5.44** (flat-to-down, confounded, n=8) — corroborates retest, no improvement.
- **D. Answer entropy 2.41→1.96** (mild concentration, benign). **E. Forced ~20% stable** (not gaming).
- **F. Judge audit**: 602 artifacts, 3% split (consistent), but 2/9 spot-checked CORRECT verdicts had code stdout ≠ claimed answer (n tiny → inconclusive; flag for proper audit).
- **CONCLUSION: not broken, UNDERFITTING from (1) signal starvation + (2) tiny LoRA/low-LR update.** Full-weight RL fixes (2) but NOT (1) — an 8/8 group has zero advantage at any capacity. Fix needs BOTH + judge-integrity check.

### Overfit diagnostic (scripts/grpo_diagnose.py) — REFINES diagnosis: mechanism sound, recipe not inherently weak
- Tier 1: tokenization round-trip 0/8 mismatches, mean per-token logprob -0.15 → trainer scores the real sampled tokens, gradient is the correct quantity (rules out garbage-gradient class).
- Tier 2 overfit, EXACT production recipe (LoRA-r8, lr1e-5), fixed batch (5 correct/3 wrong), correct−wrong logprob gap: start -0.085 → step10 -0.071 → step20 -0.025 → step30 +0.063 → step40 +0.215. **Repeated signal DOES separate correct from wrong in ~30 steps at lr1e-5.** So lr1e-5/r8 is not inherently too weak.
- **Refined root cause: signal STARVATION + each question seen ~once (no sustained pressure).** Not "step too small." Production never accumulates the steps-on-coherent-signal that the overfit shows works.
- **Fix priority reordered:** (1) signal density (3/8 target + curated 2-4/8 exemplars — DONE: target changed, questions_for_curation.md exported, 49 sweet-spot Qs); (2) **reuse each rollout batch for several gradient mini-epochs** (cheap, = the "40 steps on same signal" that worked) — highest-leverage code change; (3) full-weight/higher-LR secondary (speeds, not the blocker; full-weight needs multi-GPU ZeRO).

### Verifiability / judge-reliability audit (2026-06-13, static + external oracle)
- Inter-judge agreement (per-group): mean **0.973**, 83% of groups at 1.0. Judge temp was 0.2/0.5 → **raised to 0.6** for next run.
- Correct-rollout answer consensus: **260/268 groups (97%)** had a single agreed correct integer; 8 groups accepted contradictory answers (reward noise, on ambiguous Qs).
- Program-oracle from saved judge stdout: too sparse to trust (only 27/602 parseable) — needs re-execution (TODO).
- **External oracle gpt-5.4 (no code, independent solve, n=80): 86% overall, but 95% (60/63) on the easy 6-8/8 band where our answer is robustly established.** Agreement falls with difficulty (mid 67%, hard 20%) because gpt-5.4 does hard enumeration in-head; 7/11 disagreements within ±2 of our answer = gpt-5.4's own counting slips, NOT judge error. (nano was useless: 62%, wild hallucinations — confirms a non-code model can't verify these computational Qs; this is why the code-executing judge exists.) Scripts: `scripts/oracle_compare_nano.py` (ORACLE_MODEL env).
- **Conclusion: judge is reliable where it matters (95% external corroboration on established answers); residual noise concentrated on hardest questions, exactly where code execution beats reasoning.**

### run2 (2026-06-14): over-aggressive update CAUGHT by pre-launch sanity sweep; killed at step 6
- Config: lr 3e-5, mini-epochs 3, r16, COLD start (outputs/ got wiped by disk cleanup @93% full; banks restored from Desktop).
- Sanity sweep findings: reward/judge CLEAN (0 contradictions, 0.98 agreement). But: (1) band only 12% mid (cold-start removed the diverse seed the counterfactual relied on); (2) **WEIGHT DRIFT RUNAWAY: ||delta|| 0.44 at step1, 1.165 by step6 = ~0.2/step = ~40x v3's 0.005/step.** Model was being THRASHED not trained (KL diverging to -0.031, difficulty oscillating 6.5->7.5->1.5->8.0->7.0). The 3x LR x 3 mini-epochs stack compounded worse-than-linear on tiny 2-group batches.
- Lesson: v3 too weak (0.005/step, no learning); run2 too strong (0.2/step, thrashing). Sweet spot ~0.02-0.05/step.
- Diverse Gemini GOLDEN candidates calibrated (scripts/calibrate_golden.py): 6/8 too EASY for Qwen3.5-9B (BANANA->12, CRT->378 etc). Tension: diverse topics = easy; hard = narrow number-theory. Resolution: GOLDEN anchors DIFFICULTY (calibrated mid), seed bank provides DIVERSITY.

### run3 (2026-06-14) LAUNCHED with corrections: lr 1.5e-5, mini-epochs 2, r16, SEEDED (332 diverse history), 5 calibrated-mid GOLDEN. wandb rrjbia70. Watching weight-drift (target 0.02-0.05/step) + band. Early-question retest at ~step 25 = verdict.

### PIVOT to fast training-only harness (2026-06-14, goal: get measurable in-dist + OOD learning)
- run3 killed (full loop too slow to iterate, ~35min/step). Proposer+judge are validated; open question is purely whether TRAINING learns. So stripped to `scripts/training_only_grpo.py`: 62 fixed mid-difficulty (2-6/8) questions w/ known refs (no proposer, no judge, exact-match reward), 46 train / 16 held-out. Measures IN-DIST (held-out pass) + OOD (GSM8K pass@4) before vs after.
- Running 2 configs in parallel (4 GPUs): A=batch6, B=batch12, both lr1.5e-5/epochs2/r16, 25 steps. Tests "does it learn" + batch-size effect. ~3-5h. Question set: outputs/fixed_train_questions.json.
- This is the iteration unit for the goal: short training-only run -> dual (in-dist+OOD) check -> adjust -> repeat.

### Box contention RESOLVED + fast learning sweep (2026-06-14/15)
- R-Zero (`rzv`, GPUs 0-3) was repeatedly SIGKILLing self-play processes (all sessions die, rzv survives). Fix: added SHARED COMPUTE COORDINATION protocol to global ~/.claude/CLAUDE.md (R-Zero=GPU0-3, self-play=GPU4-7 `sp_*` sessions, no box-wide pkill/reset). WORKING: sp_ runs survived 4.5h incl. an rzv restart.
- training-only harness parallelized (rollouts fired batch-wide concurrently, not sequential) — was ~1.5-2h/step, far too slow.
- Confirmed: fixed-good-questions gives DENSE signal (4-6/6 trained groups vs v3's 1.6/4). Baselines: in-dist ~0.47-0.56, GSM8K pass@4 ~0.99.
- Running fast bracket: A=lr1.5e-5(~0.15/step), B=lr1e-5(~0.1/step), both batch8/ep2/r16/12 steps. Verdict = in-dist + GSM8K before/after deltas. First clean test of whether GRPO learns on known-good questions.

### Training-only sweep RESULT (2026-06-15): no measurable gain in 12 steps, but UNDERPOWERED
- A (lr1.5e-5): in-dist held-out 0.509->0.473 (-0.036), GSM8K pass@4 0.967->0.983 (+0.017)
- B (lr1e-5): in-dist 0.536->0.527 (-0.009), GSM8K 0.983->0.983 (0.000)
- Per-step training passes FLAT (A ~5.3, B ~5.7) — but confounded (different batch each step).
- **NOT a clean "training broken" verdict — the test was underpowered**: (1) only 12 steps ~2 exposures/Q (overfit test needed ~30 on a FIXED batch); (2) GSM8K pass@4 at ceiling 0.98 (no OOD headroom — must use pass@1); (3) held-out n=14 too noisy (±0.13); (4) held-out measures GENERALIZATION not memorization. The earlier overfit test ALREADY proved the gradient mechanism works on a fixed batch.
- Memorization check (did trained-Q improve) launched but vLLM too slow at 8192 / one arm 400'd — inconclusive.
- **Next iteration design (decisive): small FIXED train set (~8-12 Q) x many steps (30-40), measure pass on the TRAINED Qs (memorization) + GSM8K pass@1 + larger held-out. If trained-Q pass rises -> learning works, scale exposures for generalization. If flat -> training truly broken.**
- Infra wins this round: coordination protocol stopped box kills (sp_ runs survived 5h+); parallelized harness ~2x faster.

### MEASUREMENT FIX (2026-06-15): GSM8K was a broken yardstick; switched OOD to GSM-hard
- GSM8K pass@1 at FULL 8192 tokens = 0.980 for base Qwen3.5-9B (saturated/likely contaminated). The earlier "0.476" was a 1024-TOKEN TRUNCATION ARTIFACT, not real ability. So GSM8K has no headroom -> useless for detecting OOD learning. A chunk of "no OOD gain" was this.
- **GSM-hard base pass@1 0.790 (pass@4 0.850)** — real headroom. New OOD benchmark. `scripts/gsmhard_eval.py` (loads reasoning-machines/gsm-hard, integer answers, fits FINAL_ANSWER extraction).
- Proper measurement now: IN-DIST = trained-Q pass (memo run, baseline 0.542); OOD = GSM-hard pass@1 (baseline 0.790).
- Decisive memo run (12 fixed Q x 30 steps, lr1e-5) in progress; will eval its adapter on GSM-hard for OOD. This is the first time both axes are measured against benchmarks that can move.

### CORE FINDING (2026-06-15): logprobs separate but generation pass-rate doesn't follow (RLVR wall)
- Overfit floor test (6 solvable golden Qs, aggressive lr3e-5 x3 epochs, batch-all): FLAT (baseline 4.33/8, steps 3.0-4.5/8 bouncing). Cannot overfit even 6 questions.
- BUT grpo_diagnose earlier showed logprobs DO separate (correct completions up, wrong down). So gradient direction is correct; the model is pushed right at the logprob level but doesn't change what it GENERATES. Weights move a lot, pass flat.
- Diagnosis: reinforcing specific correct trajectories (which share ~95% tokens with wrong ones, differ only at final answer) barely shifts the generation distribution. Classic RLVR trajectory-vs-generation gap, compounded by signal dilution (answer-token gradient divided by ~1000 reasoning tokens via per-token length norm) + KL pullback.
- Also suspect: the golden Qs are BRUTAL stacked-constraint number theory; "5/8" may be inconsistent luck, not a sharpenable reliable method.
- **Fix being tested**: added `pg_norm` param to grpo_step (fixed normalizer instead of /n_tok, undilutes answer-token gradient) + KL_BETA/PG_NORM env hooks in harness. Rerunning overfit on golden8 with KL_BETA=0 PG_NORM=256 lr1e-5 (tmux sp_overfit2). If it climbs, dilution was the wall; if flat, deeper reward-design issue or questions-too-hard-for-overfit.
- Infra: vLLM restarted faster (no enforce-eager, max-model-len 8192 -> needs max_new_tokens<=~6000 to avoid 400). Still ~35min/step (training compute: 3 epochs x 6 groups x 8 completions at 6000 tok).

### ROBUST NEGATIVE VERDICT (2026-06-16): GRPO-on-LoRA self-play does not learn; aggressive settings DEGRADE
- Greedy check (temp 0.2, de-diluted overfit2 adapter step6 vs base on golden8): BASE 0.469 -> ADAPTER 0.344 (-0.125). Per-Q: trained adapter DROPPED on Q4(.75->.25), Q7(1->.75), Q8(.75->.25); Q5,Q6 = 0/4 both (model genuinely cannot solve those 2 of 8). NOT masked learning — training actively harmed generation. De-dilution hypothesis REFUTED (undiluting/amplifying made it worse).
- Ruled out across ~12 configs: measurement, scale, dilution, KL, capacity, signal density, temp masking. No setting improves pass rate: gentle=flat, aggressive=degrades. Gradient locally moves logprobs correctly (grpo_diagnose) but generation doesn't follow / degrades = trajectory-vs-generation wall + partly-unsolvable/luck-based questions.
- **STOP tuning this arm.** Genuine next directions if continuing self-play: (a) prove RL can sharpen EASY questions first (50% = reliable-method-with-slips, not brutal stacked number theory); (b) different RL formulation — proper PPO clip ratio for multi-epoch, or full-weight; (c) lean on R-Zero hedge (running on GPUs 0-3). Recommend strategic reassessment over more hyperparameter tuning.

### Diagnosis refined (2026-06-16): NOT bimodal; the flatness is BATCH-SIZE noise (wobble)
- Corrected earlier "bimodal" claim: v3 per-Q pass dist (n=280, k=8) is SKEWED-EASY — 8/8:55%, 7-8:71%, MID 2-6:22% (63 Q), HARD 0-1:7%. There IS a real middle; the bank is just mostly too-easy. GSM-hard: only 11/80 in [0.25,0.75].
- Bouncing mechanism (KEY): per-step trained-Q pass swings (e.g. Q5: 0→3→0→8, Q4: 7→6→8→1) — a 0/8↔8/8 swing is the POLICY lurching, not measurement noise. Cause: effective batch ~48-64 rollouts/step (6-8 Q × k8), per-Q advantage from only 8 samples = high-variance gradient; lr1e-5×2-3 epochs steps hard on that noise; fresh noisy rollouts each step → updates point different ways → cancel → flat-with-swings (random walk, not climb). This is 10-20x smaller batch than standard RLVR (100s-1000s prompts).
- **BATCH-SIZE HYPOTHESIS (user's call, testing now):** too-small batch → noisy gradient → wobble. Fix = bigger batch. Launched `sp_bigbatch`: 16 Q × k16 = 256 rollouts/step (4-5x), 1 epoch (no multi-epoch wobble), lr1e-5, heldout 16, on mid set. vs `sp_easyfit` small-batch baseline. If bigbatch climbs where small-batch was flat → batch size was the wall.
- Other fixes implied: PPO clip ratio (bounds per-step lurch; our unclipped REINFORCE degraded under aggressive settings), more rollouts/Q (lower-variance advantage).

### BATCH-SIZE TEST RESULT (2026-06-16): bigger batch did NOT fix it; degrades
- bigbatch (16 Q x k16 = 256 rollouts/step, 4-5x prior, single-epoch on-policy, lr1e-5): trained-Q 0.359 -> 0.42 -> 0.30 -> 0.32 -> 0.26. Flat-to-DECLINING (256-sample noise ~0.03, so the decline is real policy degradation). easyfit small-batch control: flat 0.39-0.56.
- KEY: bigbatch is single-epoch ON-POLICY (PPO clip wouldn't change it) yet DEGRADES. On-policy PG with correct rewards should not make the model worse. Strong signal of an implementation issue in grpo_step/logprob/advantage, OR fundamental instability of this LoRA-GRPO-on-long-completions setup.
- **ROBUST NEGATIVE, FULLY INVESTIGATED**: swept batch, LR, epochs, dilution, KL, question difficulty/selection, length-norm. None learn; several degrade. Logprobs separate (gradient locally right) but generation doesn't follow / degrades.
- **Recommended next (NOT more tuning):** (1) TRL GRPOTrainer sanity run on the easy questions — battle-tested independent impl; if it learns, our grpo_step has a bug; if not, fundamental. (2) Real compute (full-weight, 1000s-rollout batches, multi-node). (3) R-Zero hedge. AWAITING USER STRATEGIC DECISION.

### Next-run levers to actually move the policy (for the next overnight)
- [ ] Raise LR (1e-5 → 3e-5/1e-4) and/or LoRA rank (8 → 32/64); these are the most likely culprits
- [ ] Harder calibration target: proposer is still emitting too many 8/8 (low-advantage) — push difficulty so more groups land 2-6/8 where GRPO advantage is strongest
- [ ] Consider multiple gradient steps / larger effective batch per set of rollouts
- [ ] In-distribution eval (bank-holdout) as primary metric; GSM8K secondary
- [ ] Verify the custom GRPO step numerically (advantage sign, logprob grad path) — rule out an implementation no-op before blaming hyperparams

### Pending (infra)
- [ ] Online run v3 still training (~step 41); let it finish or stop — it will not show gains as-is
- [ ] vLLM2 on GPU2 (port 8002) still up from the retest — free it
- [ ] Report 7-GPU run result + per-checkpoint eval table when done (watcher live; superseded architecturally but still answers "does correctness-reward training transfer")
- [ ] Full online run launch when GPUs free (~10:00 UTC): online_self_play_grpo.py, 60 steps, q/step 4, 8192 tokens, dedicated GPUs
- [ ] Phase-2 judging: proposer emits verification predicates (multi-answer/constructions); phase-3: repo artifacts
- [ ] Final post-loop adapter push returned HTTP 400 in smoke2 — chase before full run
- [ ] run1 (full GRPO rerun) still staged+held in tmux — likely superseded by this pipeline; decide after eval
- [ ] Unity cluster: user must paste public key into unity.rc.umass.edu/panel; then `ssh unity`
- [ ] Integrate agentic proposer + code judge into the production loop proper (replace generate_pairwise_data path)

---

## 2026-06-10 — Root cause of GSM-hard "gains": answer-first GRPO template (training was wrong, must rerun)

### TL;DR
The 25-cycle Qwen3.5-9B self-play run (`outputs/self_play_grpo_loop/qwen35_ablation_update_gen0_8x8_vllmlora/`) did **not** improve reasoning. Pass@1 "gains" on GSM-hard were format compliance (format_rate 42% → 98%) while conditional accuracy (condAcc = correct | parseable) **declined**. Root cause: the GRPO *training-time* prompt template demanded `FINAL_ANSWER:` on the **first line** (answer before any reasoning), capped at 384 tokens, with a clean-tail bonus — so the reward-optimal completion was literally a bare `FINAL_ANSWER: <n>` with no reasoning. All existing checkpoints are tainted; the loop must be rerun with the fixed template.

### Evidence (collapse is real, not variance)
condAcc on GSM-hard (150 q, k=4, Wilson 95% CI):
- base: 0.352 [0.296, 0.412]
- cycle014: 0.375 [0.331, 0.422]
- cycle022: 0.221 [0.189, 0.256]
- cycle025: 0.206 [0.175, 0.241]  ← non-overlapping with base

Tag-position collapse (median chars before `FINAL_ANSWER:` in eval completions):
base 236 → c014 157 → c016 54 → c022 30 → c025 33. "Blurt rate" (<200 chars) 0.375 → 0.872.

### Ruled out
- **LoRA**: single continued r=8/alpha=16/all-linear adapter across all cycles; Frobenius norm flat (c1 30.90, c7 25.72, c14 25.73, c20 25.74, c25 25.75) while condAcc fell. Not the cause.
- **Self-judge quality**: gpt-5.4-nano proxy-oracle audit of 320 actual GRPO rollout solutions (cycles 1–25): pooled agreement 0.936, proxy false-positive rate 0.026, false-negative 0.111, nano unanimous 98%. The judge verdicts were fine; earlier "brevity bias" claim retracted.

### Root cause detail (two generation streams — important!)
1. **Rollout artifacts** (`cycle_*_samples.jsonl`, via `pairwise_rollouts_qwen35_overnight.yaml`): reasoning-first, `enable_thinking: true`, thinking_budget 512, `api_final_prefill: "FINAL_ANSWER: "`, 4096 tokens. These look healthy — but they're only used for question selection and judging. **No gradients flow through them.**
2. **GRPO-time completions** (TRL GRPOTrainer generates online under `train_pairwise_verdict_qwen35_overnight.yaml`): the old template said "Put the final answer first. Your first line must be exactly: FINAL_ANSWER: ... Do not write more than 6 lines total", max_new_tokens 384. Combined with format_weight 0.2, answer_boundary_weight 0.1, and `reward_clean_final_answer_format` (1.0 only if zero chars after tag), the optimum is a bare tag line. These completions are never saved, which is why inspecting samples.jsonl was misleading.

Compounding issue (open): proposer distribution misaligned with GSM-hard — self-generated question numbers log10 median ~1.0–1.6 vs GSM-hard 6.61, stable from cycle 1.

### Eval was also answer-first
`scripts/eval_gsmhard_sweep.py:18` uses the same answer-first prompt → eval numbers also conflated. Created `scripts/eval_gsmhard_reasonfirst.py` (reason-first, tag-last, 1024 tokens) and submitted **Slurm job 79035** (base, c014, c020, c025; 150q, k=4) → `outputs/evals/gsmhard/reasonfirst_150/summary.json`. Interpretation: c025 ≈ base → collapse was prompt compliance only; c025 < base → genuine degradation; c014 > base → first real gain.

### Fixes applied (local + server, verified identical)
- `grpo_math/configs/train_pairwise_verdict_qwen35_overnight.yaml`:
  - prompt → reason-first: "Solve the problem step by step... End your response with one final line that is exactly: FINAL_ANSWER: <integer>"
  - `rollout.max_new_tokens: 384 → 1024` (judge cap of 128 at line 46 untouched)
- Reward stack needs no code change — format/boundary/clean-tail rewards align correctly with tag-last completions.

### Pre-rerun code audit (2026-06-10, all clear with caveats)
- **Loop driver** (`run_self_play_grpo_loop.py:1510`): re-loads the train template fresh each cycle and overwrites snapshots — resume can NOT reuse old answer-first snapshots. Rerun hazards are launch-args only: don't pass a tainted `--initial_solver_model`, and use a NEW `--run_tag` (old run dir would reuse old cycle JSONLs/checkpoints via `--skip_rollout_if_jsonl_exists`).
- **Reward stack**: all tag-position-agnostic (`final_answer_tail_char_count` = chars after tag → tag-last scores format 1.0 / boundary 1.0). With reason-first completions the verdict judge now sees full reasoning (canonicalize keeps the prefix). Trainer `max_completion_length` = `rollout.max_new_tokens` = 1024 (train_grpo_trl.py:1124). `inherit_solver_max_new_tokens` not set → no 4096 override.
- **Server reward.py is NEWER than local** (uncommitted on server): last-occurrence used consistently in strict-extract/canonicalize/tail (fixes first-vs-last asymmetry), case-insensitive regex, adds `extract_final_answer_text_strict`. Correct for tag-last. Local repo is stale here — sync eventually.
- **Launcher**: the 25-cycle run used `scripts/slurm_qwen35_ablation_update.sbatch` (run_tag qwen35_ablation_update_gen0_8x8_vllmlora), which points at the FIXED train config — safe to reuse with a new run_tag. (`slurm_qwen35_vllm_loop_overnight.sbatch` also points at it.)
- **Server/local drift**: server on `main` + ~12 uncommitted modified files; local on `simplify-single-integer-no-python`. Differing: reward.py (server newer), rollout solver prompt_template (server: think-channel + "final response must be exactly one line FINAL_ANSWER", local: answer-first + optional check). Server is what runs.
- **Rollout solver stream** intentionally answer-only in visible text (reasoning in thinking channel, budget 512, prefill) — style now differs from the reason-first training template. Not a bug, but rollout-judged solutions won't match GRPO-time completion style.
- **Still stale answer-first templates on server** (only matter if used): `train_pairwise_verdict_qwen35_{low_format,no_boundary,confirm_strict_overnight,confirm_strict_overnight_safe}.yaml`, `train_question_qwen35_confirm_strict_overnight{,_safe}.yaml`, `scripts/eval_gsmhard_sweep.py` (superseded by `eval_gsmhard_reasonfirst.py`), `scripts/question_reward_signal_experiment.py`. Note: `gated_format` variant is already tag-last.

### Reason-first eval results (job 79035, COMPLETED 2026-06-10, 55 min)
`outputs/evals/gsmhard/reasonfirst_150/summary.json` (150q, k=4, reason-first prompt, 1024 tokens):

| model | pass@1 | pass@k | format | condAcc | matched both-formatted acc |
|---|---|---|---|---|---|
| base | 0.287 | 0.580 | 0.615 | 0.295 | 0.453–0.509 |
| c014 | 0.273 | 0.547 | 0.738 | 0.280 | 0.421 |
| c020 | 0.273 | 0.587 | 0.865 | 0.282 | 0.364 |
| c025 | 0.233 | 0.507 | 0.952 | 0.218 | 0.279 |

**Conclusion: no checkpoint beats base even under reason-first prompting.** c014/c020 ≈ base (reasoning mostly intact; only format improved); c025 genuinely degraded (matched-pair acc 0.279 vs base 0.453). → **Seed the rerun from fresh base.** Curiosity: contains_correct@1 is *higher* for trained cycles (0.61 → 0.69) — they compute the right number somewhere more often but finalize the wrong one.

### Smoke test before rerun (job 79040, submitted 2026-06-10)
1-cycle end-to-end smoke of the fixed pipeline, same 2-GPU topology as production (GPU0 vLLM, GPU1 train). Run dir: `outputs/self_play_grpo_loop/qwen35_reasonfirst_smoke/`, log `outputs/slurm/qwen35_rf_smoke_79040.out`.
- Configs (server): `pairwise_rollouts_qwen35_smoke_reasonfirst.yaml` (overnight copy, num_questions 8), `train_pairwise_verdict_qwen35_reasonfirst_smoke.yaml` (fixed overnight copy: steps 4, save_every 4, **debug_rollouts enabled** → TRL log_completions), `scripts/slurm_qwen35_reasonfirst_smoke.sbatch` (--cycles 1, --max_steps 4, --max_train_samples 16, run_tag qwen35_reasonfirst_smoke)
- Pass criteria: (1) cycle_001_samples.jsonl written; (2) snapshot configs/cycle_001_train.yaml carries the reason-first prompt; (3) logged GRPO completions show reasoning BEFORE a final `FINAL_ANSWER:` line (not blurted first-line tags); (4) nonzero verdict rewards (teacher API path works); (5) cycle_001_grpo/checkpoint-4 saved.

**RESULT (COMPLETED 2026-06-10, 25 min wall): PASS — pipeline works end-to-end; cleared for full rerun.**
- Loop stages all ran: generator 24s, solver 220s (56 prompts), judge batches 6–48s, question-train checkpoint-25, GRPO 4 steps (train_runtime 284.8s), checkpoint-4 saved, "[loop] complete".
- Criteria: #1 ✅ #2 ✅ (snapshot prompt is reason-first/tag-last) #5 ✅.
- #3 ✅ — parquet completion dump (16 completions across 4 steps): of the 7 with a parseable `FINAL_ANSWER:` tag, chars BEFORE tag = min 13 / median 2397 / max 3699. The blurting pathology is gone; the model reasons at length before the tag.
- #4 ✅ (marginal) — mean reward_verdict 0.0625 (1/16 correct), reward_format 0.219, boundary 0.057. Teacher API path works; low correctness expected from base model on hard self-generated questions at cycle 1.
- Caveats observed (not blockers): 9/16 completions had NO parseable tag — base model is verbose (`<think>`-style blocks, "Final Answer: \boxed{}" preambles) and gets truncated at 1024 tokens mid-reasoning; 0/7 tagged completions ended exactly at the tag (trailing text after it, boundary reward partial). Expectation: format (0.2) + boundary (0.1) rewards now pressure toward "reason, then tag, then stop" *within* budget — the correct optimization target — instead of the old "tag immediately" optimum. Watch format_rate and tail_chars trend in the real run; if tag rate stays low after a few cycles, consider bumping max_new_tokens to 1536.

### 8k-token smoke test (jobs 79048 → 79062 → 79063, 2026-06-10; attempt #3 PASSED on uni + Brev)
Decision: rollout/GRPO completion cap 1024 → **8192** (the 1024-token smoke truncated 9/16 completions mid-`<think>`; Qwen3.5 naturally reasons 2–8k tokens). vLLM server already at `--max-model-len 32768`, no other change needed. Server files: `grpo_math/configs/train_pairwise_verdict_qwen35_reasonfirst_smoke_8k.yaml` (only diffs from 1024 smoke: line 52 `max_new_tokens: 8192`, output_dir `…_8k`; judge cap 128 untouched), `scripts/slurm_qwen35_reasonfirst_smoke_8k.sbatch` (run_tag `qwen35_rf_smoke_8k`, logs `outputs/slurm/qwen35_rf_smoke8k_79048.{out,err}`). Pass criteria same 5 as before, plus: tag rate should rise well above 7/16 now that truncation is removed. If 8k passes, the full rerun should use 8192 (edit `train_pairwise_verdict_qwen35_overnight.yaml` 1024→8192 before launching).

**Job 79048 (first 8k attempt) FAILED with CUDA OOM at GRPO step 1** — rollout/judge/question-train all fine (and notably fast: question-train 104s). OOM in TRL `_get_per_token_logps_and_entropies`: scoring forward materializes logits then accelerate casts to fp32 → `8 completions × ~6k tokens × 151k vocab × 4B ≈ 30.3GB` single allocation on the train GPU (which also holds model+LoRA, ~50GB in use). The scoring chunk size = `per_device_train_batch_size` = our `prompts_per_step` (grpo_trainer.py:1935, train_grpo_trl.py:1040-1045).
**Fix (config-only, both machines):** `prompts_per_step: 8 → 2`, `grad_accum_steps: 1 → 4` in `train_pairwise_verdict_qwen35_reasonfirst_smoke_8k.yaml`. Generation batch per optimizer step stays 8 (=2×4, divisible by k=4) so training semantics are unchanged; scoring forward now 2 sequences ≈ 7.6GB. **The full-rerun overnight config needs the same treatment when bumped to 8k.** Resubmitted as **job 79062**; failed run archived at `outputs/self_play_grpo_loop/qwen35_rf_smoke_8k_oom_79048/`.

**OOM #2 — job 79062 (uni) AND the Brev box run both died identically at GRPO step 1**: `Tried to allocate 7.58 GiB` in the **backward pass** (`accelerator.backward` → `_engine_run_backward`) with ~75.7–75.85 GB already PyTorch-allocated. The *loss* path (unlike scoring) saves bf16 logits + the fp32 cast + log-softmax for autograd, so even micro-batch 2 at ~8k tokens doesn't fit on an 80GB A100 alongside model+LoRA.
**Fix v2 (config-only, both machines, current):** `prompts_per_step: 2 → 1`, `grad_accum_steps: 4 → 8`. Generation batch still 8 (=1×8, divisible by k=4); semantics unchanged. Uni: archived to `qwen35_rf_smoke_8k_oom_79062/`, resubmitted as **job 79063**. Box: tmux `smoke8k` killed, run dir archived to `qwen35_rf_smoke_8k_brev_oom1/` (log → `brev_8k_smoke_oom1.log`), relaunched. If a third OOM hits, escalation options: lower max_new_tokens below 8192, or switch TRL to vLLM-based generation (TRL currently generates in-process via HF on the train GPU — also the 8k speed bottleneck; no `use_vllm` anywhere in train_grpo_trl.py).

**RESULT (attempt #3, both machines, 2026-06-10): PASS — micro-batch 1 fixed the OOM; cleared for full 8k rerun.**
- **Box** (run_tag `qwen35_rf_smoke_8k_brev`): 4/4 steps, checkpoint-4, `[loop] complete`, clean tmux exit. Step 1 completions: **8/8 tagged** (vs 7/16 at 1024), chars-before-tag med 4462, mean reward_verdict per step 0.75/0.25/0.875/0.375, format 0.625/0.5/0.625/0.375, `frac_reward_zero_std: 0` all steps (good gradient signal). step_time 855–900s; clipped_ratio 0.25–0.375.
- **Uni** (job 79063): 4/4 steps, checkpoint-4, `[loop] complete`. 19/32 tagged overall (59%); verdict per step 0.0/0.125/0.25/0.125 — its proposer drew **unsolvable questions** (no integer answer: cylinder r³−50r+200=0; chord segment 5π transcendental), so the solver reasons to the 8192 cap and can't conclude. NOT a pipeline bug — confirms backlog item "proposer distribution alignment". One step had `frac_reward_zero_std: 0.5` (all-same-reward groups = no GRPO gradient = wasted 8k-token generations). step_time 753–755s.
- Length facts at 8k: median completion ~4.3–5.9k tokens; "early tags at char 9/13" were echoed *examples* ("For example, FINAL_ANSWER: 42"), not blurts. Note: format/boundary rewards trim the never-concluding tail but do NOT pressure median length down — budget ~5k-token median generation for the whole run.
- **Full-run prep (done)**: `train_pairwise_verdict_qwen35_overnight.yaml` patched on BOTH machines (max_new_tokens 1024→8192, prompts_per_step 8→1, grad_accum_steps 1→8). Box launcher staged: `scripts/brev_run_overnight_8k.sh` (port of `slurm_qwen35_vllm_loop_overnight.sbatch`: 10 cycles × 10 steps, 32 q/cycle, --max_train_samples 28, --log_rollouts_to_wandb, run_tag `qwen35_rf_8k_run1`, log `outputs/slurm/brev_overnight_8k.log`). Decision: launch on box (user choice), report-first before launch. Considered+deferred: parallelizing question-train with solver GRPO (semantically safe — both depend only on rollout output — but saves only minutes/cycle and needs a loop-driver change); vLLM-based TRL generation is the real speed lever (~90% of step_time is generation).

### Brev 8xA100 box (operational 2026-06-10)
Access granted; `ssh -F ~/.brev/ssh_config awesome-gpu-name0` works (instance `awesome-gpu-name0`, azurerm.a100x8.sxm.brev-dgxc, 8×A100-80GB, CUDA 13.0 driver, 96 CPU, 1.7TB RAM, ~178GB disk free; note: A100s not H100s). Setup: repo rsynced to `~/self-play` (plus server's newer reward.py + reason-first/smoke/overnight configs), venv at `~/venvs/selfplay` pinned to uni versions (torch 2.10.0+cu128, vllm 0.19.1, trl 1.4.0, transformers 5.8.0, peft 0.19.1), `.env` synced (WANDB key is `WANDBKEY` there; launcher maps it to `WANDB_API_KEY`). Launcher: `scripts/brev_run_8k_smoke.sh` (non-Slurm port of the smoke sbatch; GPU0 vLLM @32768 ctx, GPU1 train; logs `outputs/slurm/brev_8k_smoke.log`; gotcha: script does its own `exec > >(tee …)` after mkdir because tmux-level tee raced the missing dir). Running 8k smoke in tmux session `smoke8k`, run_tag `qwen35_rf_smoke_8k_brev`. Box has other users' tmux sessions (one attached `train`) but GPUs were idle. Remaining 6 GPUs open the option of parallel ablations for the rerun.

### Pending / next steps
- [x] Poll job 79035 → done, table above; **seed rerun from fresh base**
- [x] Verify smoke job 79040 against the 5 pass criteria → PASS (see above)
- [ ] **Rerun the GRPO loop** with the fixed template — new `--run_tag`, fresh base seed (all prior checkpoints trained under the broken template)
- [x] Brev 8xA100 box access + setup → operational (see section above); 8k smoke running there in parallel
- [x] 8k smoke attempt #3 → PASS on both machines (see above); micro-batch 1 is the working 8k recipe
- [ ] Launch full rerun on box: `tmux new-session -d -s run1 "bash ~/self-play/scripts/brev_run_overnight_8k.sh"` (awaiting user go)
- [ ] Follow-ups for run2: TRL vLLM generation (speed), parallel question-train, per-cycle metrics to watch: tag rate, frac_reward_zero_std, completion length median
- [ ] Patch stale sibling configs on server if any ablation rerun will use them
- [ ] Decide whether to align rollout solver prompt with the reason-first training style
- [ ] Sync server's newer reward.py (+ other uncommitted server changes) back into the repo
- [ ] Proposer distribution alignment (questions don't resemble GSM-hard number scale/difficulty)

### Infra notes
- Remote: `/tmp/rrun.sh '<cmd>'` (expect+base64 SSH to shell.engr.wustl.edu, project at `/engrfs/project/jiaxinh/dylan_work/self-play`, venv `.venv-qwen-slurm`). Write remote files via *remote* heredoc inside one single-quoted rrun arg (no single quotes in content); local heredoc subshells hang.
- Eval lineage: base–c014 `outputs/evals/gsmhard/clean_update_early/71968/`, c016–021 `new_rounds/72499/`, c022–025 `update_remaining/72547/` (all answer-first — superseded by reasonfirst_150).
- Helper scripts on server: `scripts/{nano_rollout_audit,condacc_ci,adapter_norm,dist_shift_tests,len_per_cycle}.py`, `scripts/eval_gsmhard_reasonfirst.py`, `scripts/slurm_gsmhard_reasonfirst.sbatch`.
