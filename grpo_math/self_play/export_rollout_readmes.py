import argparse
import json
from pathlib import Path
from typing import Any

import yaml


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

    solver_template = str(sol_cfg.get("prompt_template", ""))

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
        "verify_template": verify_template,
    }


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
        out.append("")
        gen_template = prompt_ctx.get("generator_template", "")
        if gen_template:
            out.append("Generator prompt template:")
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
