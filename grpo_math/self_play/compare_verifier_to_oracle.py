from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grpo_math.self_play.strong_verifier_smoke import _read_env_var_from_dotenv
from grpo_math.self_play.verifier import (
    OpenAICompatibleVerifier,
    PythonAssistedVerifier,
    Verifier,
    VerifierVerdict,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _build_oracle(args: argparse.Namespace) -> Verifier:
    api_key = os.environ.get(args.api_key_env) or _read_env_var_from_dotenv(
        args.api_key_env, args.dotenv_path
    )
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set in environment or {args.dotenv_path}")
    verifier: Verifier = OpenAICompatibleVerifier(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        prompt_template=_load_prompt(args.prompt_template),
        prompt_version=args.prompt_version,
        temperature=0.0,
        max_completion_tokens=args.max_tokens,
        max_tokens_param=args.max_tokens_param,
        extra_body=(
            {"chat_template_kwargs": {"enable_thinking": False}}
            if args.disable_thinking
            else None
        ),
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    if args.python_assisted:
        verifier = PythonAssistedVerifier(verifier, timeout_s=args.python_timeout_s)
    return verifier


def _majority(verdicts: list[str]) -> tuple[str, int, int]:
    counts = Counter(verdicts)
    if counts["CORRECT"] > counts["INCORRECT"]:
        return "CORRECT", counts["CORRECT"] - counts["INCORRECT"], len(verdicts)
    if counts["INCORRECT"] > counts["CORRECT"]:
        return "INCORRECT", counts["INCORRECT"] - counts["CORRECT"], len(verdicts)
    return "UNCLEAR", 0, len(verdicts)


def _is_positive_label(label: str) -> bool | None:
    if label == "CORRECT":
        return True
    if label in {"INCORRECT", "INVALID"}:
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rollout verifier labels to an oracle verifier.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--prompt-template", default="grpo_math/prompts/single_solution_verify_prompt.txt")
    parser.add_argument("--prompt-version", default="oracle_compare_v1")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-tokens-param", default="max_tokens")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--python-assisted", action="store_true")
    parser.add_argument("--python-timeout-s", type=float, default=5.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.jsonl))
    oracle = _build_oracle(args)
    jobs: list[dict[str, Any]] = []
    for row in rows:
        ver_by_idx = {
            int(v.get("solution_index", -1)): v
            for v in row.get("solution_verifications", []) or []
            if isinstance(v, dict)
        }
        for sol in row.get("solutions", []) or []:
            if not isinstance(sol, dict):
                continue
            sidx = int(sol.get("solution_index", -1))
            local_verdicts = [str(v) for v in (ver_by_idx.get(sidx, {}).get("verdicts") or [])]
            if not local_verdicts:
                continue
            local_label, margin, total_votes = _majority(local_verdicts)
            if local_label == "UNCLEAR":
                continue
            jobs.append(
                {
                    "row": row,
                    "solution": sol,
                    "solution_index": sidx,
                    "local_verdicts": local_verdicts,
                    "local_label": local_label,
                    "margin": margin,
                    "total_votes": total_votes,
                }
            )

    def _compare_job(job: dict[str, Any]) -> dict[str, Any]:
        row = job["row"]
        sol = job["solution"]
        oracle_result = oracle.verify(str(row.get("question", "")), str(sol.get("text", "")))
        oracle_label = oracle_result.verdict.value
        local_label = str(job["local_label"])
        agrees = local_label == oracle_label
        local_positive = _is_positive_label(local_label)
        oracle_positive = _is_positive_label(oracle_label)
        binary_agrees = (
            local_positive == oracle_positive
            if local_positive is not None and oracle_positive is not None
            else None
        )
        return {
            "question_index": row.get("question_index"),
            "solution_index": job["solution_index"],
            "candidate_answer": sol.get("parsed_final_answer"),
            "local_label": local_label,
            "local_votes": job["local_verdicts"],
            "oracle_label": oracle_label,
            "oracle_confidence": oracle_result.confidence,
            "oracle_finish_reason": oracle_result.finish_reason,
            "oracle_usage": oracle_result.usage,
            "agrees": agrees,
            "binary_agrees": binary_agrees,
            "majority_bucket": f"{job['margin']}/{job['total_votes']}",
            "oracle_raw_text": oracle_result.raw_text,
        }

    parallel = max(1, int(args.max_parallel))
    if parallel > 1 and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            comparisons = list(ex.map(_compare_job, jobs))
    else:
        comparisons = [_compare_job(job) for job in jobs]

    by_margin: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_margin_binary: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})

    for item in comparisons:
        bucket = str(item["majority_bucket"])
        by_margin[bucket]["n"] += 1
        by_margin[bucket]["correct"] += int(bool(item["agrees"]))
        if item["binary_agrees"] is not None:
            by_margin_binary[bucket]["n"] += 1
            by_margin_binary[bucket]["correct"] += int(bool(item["binary_agrees"]))

    summary = {
        "jsonl": args.jsonl,
        "max_parallel": parallel,
        "n": len(comparisons),
        "accuracy": (
            sum(1 for item in comparisons if item["agrees"]) / len(comparisons)
            if comparisons
            else None
        ),
        "binary_accuracy": (
            sum(1 for item in comparisons if item["binary_agrees"] is True)
            / sum(1 for item in comparisons if item["binary_agrees"] is not None)
            if any(item["binary_agrees"] is not None for item in comparisons)
            else None
        ),
        "by_majority_bucket": {
            key: {
                "n": value["n"],
                "accuracy": value["correct"] / value["n"] if value["n"] else None,
            }
            for key, value in sorted(by_margin.items())
        },
        "by_majority_bucket_binary": {
            key: {
                "n": value["n"],
                "accuracy": value["correct"] / value["n"] if value["n"] else None,
            }
            for key, value in sorted(by_margin_binary.items())
        },
        "comparisons": comparisons,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
