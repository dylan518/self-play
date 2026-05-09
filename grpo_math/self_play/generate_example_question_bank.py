from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from grpo_math.self_play.question_bank import QuestionBankExample, load_question_bank


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


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


def _build_prompt(examples: list[QuestionBankExample]) -> str:
    example_block = "\n\n".join(
        f"Example {idx}:\n"
        f"Category: {example.category}\n"
        f"Question: {example.question}\n"
        f"Verification: {example.verification}"
        for idx, example in enumerate(examples, start=1)
    )
    return (
        "You are expanding a self-play question bank.\n"
        "Use the sampled examples only as inspiration; do not copy or lightly paraphrase them.\n\n"
        "Generate exactly one new question with these constraints:\n"
        "- It should be solvable by Qwen-class reasoning models.\n"
        "- It should have a clear, compact answer or output.\n"
        "- It should be easy to verify with Python by brute force, symbolic computation, "
        "string checks, regex checks, unit tests, or direct execution.\n"
        "- Avoid requiring private data, current events, subjective judgment, images, "
        "or long proofs.\n"
        "- Do not include the answer, hints, solution steps, or Python code.\n\n"
        "Return only valid JSON in this exact shape:\n"
        '{"category":"<short_snake_case_category>","question":"<question text>",'
        '"verification":"<brief Python verification plan>"}\n\n'
        f"Sampled inspiration examples:\n{example_block}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Model output was not a JSON object")
    return parsed


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    category = str(row.get("category", "")).strip().lower().replace("-", "_").replace(" ", "_")
    question = str(row.get("question", "")).strip()
    verification = str(row.get("verification", "")).strip()
    if not category or not question or not verification:
        raise ValueError("Generated row must include category, question, and verification")
    return {
        "category": category,
        "question": question,
        "verification": verification,
    }


def _chat_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "self-play-question-bank/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    raise ValueError("Could not parse chat completion content")


def _generate_one(
    *,
    examples: list[QuestionBankExample],
    rng: random.Random,
    examples_per_prompt: int,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
) -> dict[str, str]:
    sampled = rng.sample(examples, k=min(examples_per_prompt, len(examples)))
    prompt = _build_prompt(sampled)
    backoff_s = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            raw = _chat_completion(
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return _normalize_row(_extract_json_object(raw))
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                raise RuntimeError(f"Gemini request failed: HTTP {exc.code}: {detail}") from exc
        except Exception:
            if attempt == max_retries:
                raise
        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 2.0, 30.0)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate more example questions from a bank.")
    parser.add_argument("--input", default="examples.json", help="Existing question bank JSON.")
    parser.add_argument(
        "--output",
        default="examples_gemini31flashlite_preview_generated.json",
        help="Output JSON path for generated questions.",
    )
    parser.add_argument("--count", type=int, default=200, help="Number of questions to generate.")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
    parser.add_argument("--examples-per-prompt", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip() or (
        _read_env_var_from_dotenv(args.api_key_env, args.dotenv_path) or ""
    )
    if not api_key:
        raise SystemExit(
            f"{args.api_key_env} is required in the environment or {args.dotenv_path} "
            f"to call {args.model}"
        )

    examples = load_question_bank(args.input)
    if not examples:
        raise SystemExit(f"No examples found in {args.input}")

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    output_path = Path(args.output)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_rows = existing.get("tasks_extended", [])
        if isinstance(existing_rows, list):
            for row in existing_rows:
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_row(row)
                key = re.sub(r"\s+", " ", normalized["question"].casefold()).strip()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                rows.append(normalized)

    def _write_output() -> None:
        output = {
            "metadata": {
                "source": args.input,
                "model": args.model,
                "dotenv_path": args.dotenv_path,
                "seed": args.seed,
                "count": len(rows),
                "target_count": args.count,
                "examples_per_prompt": args.examples_per_prompt,
                "constraints": [
                    "sample from pre-existing bank as inspiration",
                    "Qwen-class models should be able to solve",
                    "easy to verify with Python",
                ],
            },
            "tasks_extended": rows,
        }
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    attempts = 0
    max_attempts = max(args.count * 4, args.count + 20)
    while len(rows) < args.count and attempts < max_attempts:
        attempts += 1
        row = _generate_one(
            examples=examples,
            rng=rng,
            examples_per_prompt=args.examples_per_prompt,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_retries=args.max_retries,
        )
        key = re.sub(r"\s+", " ", row["question"].casefold()).strip()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        rows.append(row)
        print(f"[{len(rows)}/{args.count}] {row['category']}: {row['question'][:90]}", flush=True)
        if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
            _write_output()

    if len(rows) < args.count:
        raise SystemExit(f"Generated only {len(rows)} unique questions after {attempts} attempts")

    _write_output()
    print(f"Wrote {len(rows)} questions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
