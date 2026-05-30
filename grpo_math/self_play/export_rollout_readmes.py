import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from grpo_math.self_play.question_bank import render_question_bank_block_from_config


def _read_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fenced(text: str) -> str:
    return f"```\n{text}\n```"


def _load_text_maybe(path_value: str, config_dir: Path) -> str:
    p = Path(path_value)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(config_dir / p)
        candidates.append(Path.cwd() / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def _build_prompt_context(config_path: Path | None) -> dict[str, str]:
    if config_path is None:
        return {}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent

    gen_cfg = cfg.get("generator", {})
    sol_cfg = cfg.get("solver", {})
    judge_cfg = cfg.get("judge", {})

    generator_template = ""
    if gen_cfg.get("prompt_template_path"):
        generator_template = _load_text_maybe(str(gen_cfg["prompt_template_path"]), config_dir)
        if generator_template:
            seed = int(cfg.get("seed", 1234))
            question_bank_examples = render_question_bank_block_from_config(gen_cfg, seed=seed)
            generator_template = generator_template.format(
                question_bank_examples=question_bank_examples
            )
            if gen_cfg.get("question_bank", {}).get("enabled", False):
                generator_template = (
                    "NOTE: question bank examples rotate per generated question using seed + question index. "
                    "This is the representative prompt for the first question.\n\n"
                    + generator_template
                )
            prompt_suffix = str(gen_cfg.get("prompt_suffix", "")).strip()
            if prompt_suffix:
                generator_template = generator_template.rstrip() + "\n\n" + prompt_suffix + "\n"

    solver_template = str(sol_cfg.get("prompt_template", ""))

    judge_template = ""
    if judge_cfg.get("prompt_template_path"):
        judge_template = _load_text_maybe(str(judge_cfg["prompt_template_path"]), config_dir)

    verify_template = ""
    verify_path = str(judge_cfg.get("verify_prompt_template_path", "")).strip()
    if verify_path:
        verify_template = _load_text_maybe(verify_path, config_dir)
    elif judge_cfg.get("verify_prompt_template"):
        verify_template = str(judge_cfg.get("verify_prompt_template"))

    return {
        "config_path": str(config_path),
        "generator_template": generator_template,
        "solver_template": solver_template,
        "judge_template": judge_template,
        "verify_template": verify_template,
        "generator_model": str(gen_cfg.get("api_model", gen_cfg.get("model_name_or_path", ""))),
        "generator_temperature": str(gen_cfg.get("temperature", "")),
        "generator_top_p": str(gen_cfg.get("top_p", "")),
        "generator_max_new_tokens": str(gen_cfg.get("max_new_tokens", "")),
        "solver_model": str(sol_cfg.get("api_model", sol_cfg.get("model_name_or_path", ""))),
        "solver_temperature": str(sol_cfg.get("temperature", "")),
        "solver_top_p": str(sol_cfg.get("top_p", "")),
        "solver_max_new_tokens": str(sol_cfg.get("max_new_tokens", "")),
        "judge_model": str(judge_cfg.get("api_model", judge_cfg.get("model_name_or_path", ""))),
        "judge_temperature": str(judge_cfg.get("temperature", "")),
        "judge_top_p": str(judge_cfg.get("top_p", "")),
        "judge_max_new_tokens": str(judge_cfg.get("max_new_tokens", "")),
        "judge_repeats_per_solution": str(judge_cfg.get("repeats_per_solution", "")),
        "judge_python_assisted": str(judge_cfg.get("python_assisted", False)),
    }


def _pairwise_section(row: dict[str, Any]) -> str:
    pairs = row.get("pairwise_comparisons") or []
    if not isinstance(pairs, list) or not pairs:
        return ""

    out: list[str] = []
    out.append("## Pairwise Comparisons (Elo inputs)")
    out.append("")
    out.append(f"- Num comparisons: `{len(pairs)}`")
    out.append("")

    for p_idx, pair in enumerate(pairs, start=1):
        i = pair.get("i")
        j = pair.get("j")
        out.append(f"### Pair {p_idx}: ({i} vs {j})")
        out.append("")
        out.append(f"- Prefs: `{json.dumps(pair.get('prefs', []), ensure_ascii=True)}`")
        out.append(f"- Counts: `{json.dumps(pair.get('counts', {}), ensure_ascii=True)}`")
        out.append(f"- Consistency: `{pair.get('consistency')}`")
        out.append("")

        trace = pair.get("judge_trace", {}) or {}
        raw_prompt = trace.get("raw_prompt")
        raw_outputs = trace.get("raw_outputs", []) or []
        presentations = trace.get("presentations", []) or []
        raw_prefs = trace.get("raw_prefs", []) or []
        mapped_prefs = trace.get("mapped_prefs", []) or []

        if raw_prompt:
            out.append("Judge prompt:")
            out.append("")
            out.append(_fenced(str(raw_prompt)))
            out.append("")
        if presentations:
            out.append(f"- Presentations: `{json.dumps(presentations, ensure_ascii=True)}`")
        if raw_prefs:
            out.append(f"- Raw A/B prefs: `{json.dumps(raw_prefs, ensure_ascii=True)}`")
        if mapped_prefs:
            out.append(f"- Mapped prefs (i/j encoding): `{json.dumps(mapped_prefs, ensure_ascii=True)}`")
        if presentations or raw_prefs or mapped_prefs:
            out.append("")
        if raw_outputs:
            for i_out, ro in enumerate(raw_outputs):
                out.append(f"Judge raw output {i_out}:")
                out.append("")
                out.append(_fenced(str(ro)))
                out.append("")

    return "\n".join(out).rstrip() + "\n"


def _row_readme(row: dict[str, Any], prompt_ctx: dict[str, str]) -> str:
    qi = row.get("question_index")
    question = str(row.get("question", ""))
    generator_raw = str(row.get("generator_raw_output", ""))
    reliability = row.get("reliability", {})
    ranking = row.get("ranking", {})
    solutions = row.get("solutions", [])
    verifications = row.get("solution_verifications", [])

    out: list[str] = []
    out.append(f"# Question {qi}")
    out.append("")
    out.append("## Prompted Question")
    out.append("")
    out.append(_fenced(question))
    out.append("")
    out.append("## Generator Raw Continuation")
    out.append("")
    out.append(_fenced(generator_raw))
    out.append("")
    if prompt_ctx:
        out.append("## Prompts Used")
        out.append("")
        out.append(f"- Config: `{prompt_ctx.get('config_path', '')}`")
        out.append(f"- Generator model: `{prompt_ctx.get('generator_model', '')}`")
        out.append(f"- Generator temperature: `{prompt_ctx.get('generator_temperature', '')}`")
        out.append(f"- Generator top_p: `{prompt_ctx.get('generator_top_p', '')}`")
        out.append(f"- Generator max_new_tokens: `{prompt_ctx.get('generator_max_new_tokens', '')}`")
        out.append(f"- Solver model: `{prompt_ctx.get('solver_model', '')}`")
        out.append(f"- Solver temperature: `{prompt_ctx.get('solver_temperature', '')}`")
        out.append(f"- Solver top_p: `{prompt_ctx.get('solver_top_p', '')}`")
        out.append(f"- Solver max_new_tokens: `{prompt_ctx.get('solver_max_new_tokens', '')}`")
        out.append(f"- Judge model: `{prompt_ctx.get('judge_model', '')}`")
        out.append(f"- Judge temperature: `{prompt_ctx.get('judge_temperature', '')}`")
        out.append(f"- Judge top_p: `{prompt_ctx.get('judge_top_p', '')}`")
        out.append(f"- Judge max_new_tokens: `{prompt_ctx.get('judge_max_new_tokens', '')}`")
        out.append(f"- Judge repeats_per_solution: `{prompt_ctx.get('judge_repeats_per_solution', '')}`")
        out.append(f"- Judge python_assisted: `{prompt_ctx.get('judge_python_assisted', '')}`")
        out.append("")
        gen_template = str(row.get("generator_prompt") or prompt_ctx.get("generator_template", ""))
        if gen_template:
            out.append("Rendered generator prompt for this question:")
            out.append("")
            out.append(_fenced(gen_template))
            out.append("")
        solver_template = prompt_ctx.get("solver_template", "")
        if solver_template:
            out.append("Solver prompt template:")
            out.append("")
            out.append(_fenced(solver_template))
            out.append("")
            out.append("Rendered solver prompt for this question:")
            out.append("")
            out.append(_fenced(solver_template.format(question=question)))
            out.append("")
        judge_template = prompt_ctx.get("judge_template", "")
        if judge_template:
            out.append("Judge prompt template (pairwise mode):")
            out.append("")
            out.append(_fenced(judge_template))
            out.append("")
        verify_template = prompt_ctx.get("verify_template", "")
        if verify_template:
            out.append("Verifier prompt template:")
            out.append("")
            out.append(_fenced(verify_template))
            out.append("")
    out.append("## Solutions")
    out.append("")

    ver_by_idx: dict[int, dict[str, Any]] = {}
    for v in verifications:
        idx = int(v.get("solution_index", -1))
        ver_by_idx[idx] = v

    for sol in solutions:
        sidx = int(sol.get("solution_index", -1))
        out.append(f"### Solution {sidx}")
        out.append("")
        out.append(f"- Sampling group: `{sol.get('sampling_group')}`")
        out.append(f"- Parsed final answer: `{sol.get('parsed_final_answer')}`")
        out.append(f"- Pairwise score: `{sol.get('pairwise_score')}`")
        out.append(f"- Elo rating: `{sol.get('elo_rating')}`")
        out.append("")
        out.append("Full continuation:")
        out.append("")
        out.append(_fenced(str(sol.get("text", ""))))
        out.append("")

        v = ver_by_idx.get(sidx)
        if v is not None:
            out.append("Verifier result:")
            out.append("")
            out.append(
                f"- Verdict counts: `{json.dumps(v.get('counts', {}), ensure_ascii=True)}`"
            )
            if v.get("format_check") is not None:
                out.append(f"- Format check: `{v.get('format_check')}`")
            out.append(f"- Verdict confidence: `{v.get('confidence')}`")
            out.append(f"- Model confidence mean: `{v.get('model_confidence_mean')}`")
            out.append("")
            trace = v.get("judge_trace", {}) or {}
            raw_prompt = trace.get("raw_prompt")
            raw_outputs = trace.get("raw_outputs", []) or []
            if raw_prompt:
                out.append("Verifier prompt:")
                out.append("")
                out.append(_fenced(str(raw_prompt)))
                out.append("")
            if raw_outputs:
                for i, ro in enumerate(raw_outputs):
                    out.append(f"Verifier raw output {i}:")
                    out.append("")
                    out.append(_fenced(str(ro)))
                    out.append("")

    out.append("## Reliability and Ranking")
    out.append("")
    out.append(f"- Reliability: `{json.dumps(reliability, ensure_ascii=True)}`")
    out.append(f"- Ranking: `{json.dumps(ranking, ensure_ascii=True)}`")
    out.append("")

    pairwise_md = _pairwise_section(row)
    if pairwise_md:
        out.append(pairwise_md.rstrip())
        out.append("")
    return "\n".join(out)


def _index_readme(rows: list[dict[str, Any]], source_jsonl: Path, config_path: Path | None) -> str:
    out: list[str] = []
    out.append("# Rollout Export")
    out.append("")
    out.append(f"- Source JSONL: `{source_jsonl}`")
    if config_path is not None:
        out.append(f"- Config: `{config_path}`")
    out.append(f"- Questions exported: `{len(rows)}`")
    out.append("")
    out.append("## Per-Question Files")
    out.append("")
    for row in rows:
        qi = row.get("question_index")
        question = str(row.get("question", "")).replace("\n", " ").strip()
        short_q = (question[:120] + "...") if len(question) > 120 else question
        out.append(f"- `question_{qi:03d}.md`: {short_q}")
    out.append("")
    out.append(
        "Each per-question file includes full question text, prompts used, full solution continuations, and judge traces."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to rollout JSONL output.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/readme_exports",
        help="Directory where markdown files will be written.",
    )
    ap.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Optional cap on number of questions to export (0 = all).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional rollout config YAML to include prompt templates in markdown.",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rows = _read_rows(jsonl_path)
    if args.max_questions and args.max_questions > 0:
        rows = rows[: args.max_questions]
    config_path = Path(args.config).resolve() if args.config else None
    prompt_ctx = _build_prompt_context(config_path)

    run_tag = jsonl_path.stem
    run_dir = Path(args.out_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        qi = int(row.get("question_index", 0))
        target = run_dir / f"question_{qi:03d}.md"
        target.write_text(_row_readme(row, prompt_ctx), encoding="utf-8")

    index_path = run_dir / "README.md"
    index_path.write_text(_index_readme(rows, jsonl_path, config_path), encoding="utf-8")
    print(f"Wrote markdown export to: {run_dir}")


if __name__ == "__main__":
    main()
