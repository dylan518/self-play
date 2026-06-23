# rzero_pipeline/ — version-controlled mirror of the R-Zero experimental pipeline

This is a **tracked snapshot** of the R-Zero self-play pipeline that runs on the Brev box
(`/home/nvidia/R-Zero/` and `/home/nvidia/rzero_run/`). It is **not** run from here — it's
under version control so config/script drift is diffable (the thing that bit us: the
solver batch size silently scaled 64→256 via adaptive row-count logic, and the verified-arm
questioner was deleted, both undetected because the pipeline wasn't tracked).

**Keep this in sync**: after any edit to the Brev pipeline, re-pull and `jj commit` so we have
a diffable history. Secrets (HF_TOKEN, WANDB_API_KEY) are redacted to `$VAR` references.

- `R-Zero/` — the pipeline: `scripts/iteration_rzero.sh` (questioner-train → generate → evaluate →
  band-filter → judge → [CVBAND] → solver-train), reward functions (`examples/reward_function/`),
  band/eval/judge (`question_evaluate/`), eval (`pipeline/eval_compare.py`), reward service.
- `rzero_run/` — experiment launchers + eval tooling (`run_*.sh`, `eval_oly_shard.py`, `agg.py`).

See `../ARCHITECTURE.md` for the full system reference + footguns.
