from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from grpo_math.self_play.verifier import (
    CachedVerifier,
    OpenAICompatibleVerifier,
    PythonAssistedVerifier,
    Verifier,
    VerifierCache,
    VerifierVerdict,
)


CASES = [
    {
        "name": "simple_correct",
        "question": "What is 12 plus 7?",
        "completion": "12 + 7 = 19\nFINAL_ANSWER: 19",
        "expected": VerifierVerdict.CORRECT,
    },
    {
        "name": "simple_incorrect",
        "question": "What is 12 plus 7?",
        "completion": "12 + 7 = 18\nFINAL_ANSWER: 18",
        "expected": VerifierVerdict.INCORRECT,
    },
    {
        "name": "modular_correct",
        "question": "Find the remainder when 5^3 is divided by 13.",
        "completion": "5^3 = 125 and 125 mod 13 is 8.\nFINAL_ANSWER: 8",
        "expected": VerifierVerdict.CORRECT,
    },
]


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_env_var_from_dotenv(name: str, path: str | Path = ".env") -> str | None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return None
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None


def build_verifier(args: argparse.Namespace) -> Verifier:
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
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    if args.python_assisted:
        verifier = PythonAssistedVerifier(verifier, timeout_s=args.python_timeout_s)
    if args.cache_path:
        verifier = CachedVerifier(verifier, VerifierCache(args.cache_path))
    return verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny strong-verifier API smoke test.")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--prompt-template", default="grpo_math/prompts/single_solution_verify_prompt.txt")
    parser.add_argument("--prompt-version", default="single_solution_verify_python_v1")
    parser.add_argument("--cache-path", default="outputs/verifier_cache/strong_verifier_smoke.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens-param", default="max_completion_tokens")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--python-assisted", action="store_true")
    parser.add_argument("--python-timeout-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "base_url": args.base_url,
                    "api_key_env": args.api_key_env,
                    "dotenv_path": args.dotenv_path,
                    "num_cases": len(CASES),
                    "max_tokens_param": args.max_tokens_param,
                    "python_assisted": args.python_assisted,
                },
                sort_keys=True,
            )
        )
        return

    verifier = build_verifier(args)
    rows = []
    failures = 0
    for case in CASES:
        result = verifier.verify(str(case["question"]), str(case["completion"]))
        passed = result.verdict == case["expected"]
        failures += 0 if passed else 1
        rows.append(
            {
                "name": case["name"],
                "expected": case["expected"].value,
                "actual": result.verdict.value,
                "confidence": result.confidence,
                "cached": result.cached,
                "passed": passed,
                "raw_text": result.raw_text,
            }
        )
    print(json.dumps({"failures": failures, "cases": rows}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
