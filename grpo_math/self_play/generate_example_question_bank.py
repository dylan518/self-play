from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
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


def _render_examples(label: str, examples: list[QuestionBankExample]) -> str:
    if not examples:
        return ""
    example_block = "\n\n".join(
        f"{label} {idx}:\n"
        f"Question: {example.question}\n"
        f"Verification: {example.verification}"
        for idx, example in enumerate(examples, start=1)
    )
    return f"{label}s:\n{example_block}"


def _build_prompt(
    seed_examples: list[QuestionBankExample],
    recent_examples: list[QuestionBankExample],
    diversity_nonce: str = "",
) -> str:
    example_sections = "\n\n".join(
        section
        for section in (
            _render_examples("Seed example", seed_examples),
            _render_examples("Recent generated example", recent_examples),
        )
        if section
    )
    nonce_line = f"Diversity nonce: {diversity_nonce}\n" if diversity_nonce else ""
    return (
        nonce_line
        + "You are expanding a self-play question bank.\n"
        "Use the seed examples to infer the broad dataset goal, not their surface template.\n"
        "Use the recent generated examples as things to move away from immediately.\n"
        "Consider any wording, operation, object type, numeric pattern, "
        "or verification method that is not essential to that goal.\n"
        "Now make a question that is completely different from the examples below "
        "in as many of those ways as possible.\n\n"
        "Generate exactly one new question with these constraints:\n"
        "- It should be solvable by Qwen-class reasoning models.\n"
        "- It should have a clear, compact answer or output.\n"
        "- It should be easy to verify with Python by brute force, symbolic computation, "
        "string checks, regex checks, unit tests, or direct execution.\n"
        "- Avoid requiring private data, current events, subjective judgment, images, "
        "or long proofs.\n"
        "- Avoid copying the surface form, numeric pattern, and verification "
        "method of the sampled examples whenever possible.\n"
        "- Do not include the answer, hints, solution steps, or Python code.\n\n"
        "Return only valid JSON in this exact shape:\n"
        '{"question":"<question text>","verification":"<brief Python verification plan>",'
        '"confirm":"different_from_samples=<yes/no>; why=<short reason>"}\n\n'
        f"{example_sections}"
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
    if not question or not verification:
        raise ValueError("Generated row must include question and verification")
    normalized = {
        "question": question,
        "verification": verification,
    }
    if category:
        normalized["category"] = category
    confirm = str(row.get("confirm", "")).strip()
    if confirm:
        normalized["confirm"] = confirm
    return normalized


def _row_to_question_bank_example(row: dict[str, str]) -> QuestionBankExample:
    return QuestionBankExample(
        category=row.get("category", "generated") or "generated",
        task="generated_feedback",
        question=row["question"],
        verification=row["verification"],
    )


def _question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip()


def _load_feedback_examples(paths: list[str]) -> list[QuestionBankExample]:
    examples: list[QuestionBankExample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Feedback input does not exist: {raw_path}")
        if path.suffix == ".jsonl":
            rows: list[Any] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("tasks_extended", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                examples.append(_row_to_question_bank_example(_normalize_row(row)))
            except ValueError:
                continue
    return examples


def _make_diversity_nonce(rng: random.Random, length: int) -> str:
    if length <= 0:
        return ""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _chat_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    max_tokens: int,
    token_limit_param: str,
    temperature: float | None,
    top_p: float | None,
) -> str:
    if token_limit_param not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("--token-limit-param must be max_tokens or max_completion_tokens")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        token_limit_param: max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
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
    seed_examples: list[QuestionBankExample],
    recent_examples: list[QuestionBankExample],
    rng: random.Random,
    seed_examples_per_prompt: int,
    recent_examples_per_prompt: int,
    nonce_chars: int,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    max_tokens: int,
    token_limit_param: str,
    temperature: float | None,
    top_p: float | None,
    max_retries: int,
) -> dict[str, str]:
    sampled_seeds = rng.sample(seed_examples, k=min(seed_examples_per_prompt, len(seed_examples)))
    sampled_recent = recent_examples[-recent_examples_per_prompt:] if recent_examples_per_prompt > 0 else []
    prompt = _build_prompt(
        sampled_seeds,
        sampled_recent,
        diversity_nonce=_make_diversity_nonce(rng, nonce_chars),
    )
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
                token_limit_param=token_limit_param,
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
    parser.add_argument(
        "--feedback-input",
        action="append",
        default=[],
        help="Existing generated JSON/JSONL file to include as recent feedback without copying to output.",
    )
    parser.add_argument("--count", type=int, default=200, help="Number of questions to generate.")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
    parser.add_argument(
        "--seed-examples-per-prompt",
        type=int,
        default=3,
        help="Number of original seed-bank examples to sample per prompt.",
    )
    parser.add_argument(
        "--recent-examples-per-prompt",
        type=int,
        default=3,
        help="Number of most recent accepted generations to include per prompt.",
    )
    parser.add_argument(
        "--nonce-chars",
        type=int,
        default=8,
        help="Length of a small per-request random string prepended to each prompt.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument(
        "--prefer-dotenv",
        action="store_true",
        help="Read the API key from --dotenv-path before checking the process environment.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--token-limit-param",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
        help="Chat completions token limit parameter name.",
    )
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature in the chat completion request.",
    )
    parser.add_argument(
        "--omit-top-p",
        action="store_true",
        help="Do not send top_p in the chat completion request.",
    )
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    dotenv_api_key = _read_env_var_from_dotenv(args.api_key_env, args.dotenv_path) or ""
    env_api_key = os.environ.get(args.api_key_env, "").strip()
    api_key = (dotenv_api_key or env_api_key) if args.prefer_dotenv else (env_api_key or dotenv_api_key)
    if not api_key:
        raise SystemExit(
            f"{args.api_key_env} is required in the environment or {args.dotenv_path} "
            f"to call {args.model}"
        )

    seed_examples = load_question_bank(args.input)
    if not seed_examples:
        raise SystemExit(f"No examples found in {args.input}")

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    recent_examples: list[QuestionBankExample] = _load_feedback_examples(args.feedback_input)
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
                key = _question_key(normalized["question"])
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                rows.append(normalized)
                recent_examples.append(_row_to_question_bank_example(normalized))

    def _write_output() -> None:
        output = {
            "metadata": {
                "source": args.input,
                "feedback_inputs": args.feedback_input,
                "model": args.model,
                "dotenv_path": args.dotenv_path,
                "seed": args.seed,
                "count": len(rows),
                "target_count": args.count,
                "seed_examples_per_prompt": args.seed_examples_per_prompt,
                "recent_examples_per_prompt": args.recent_examples_per_prompt,
                "nonce_chars": args.nonce_chars,
                "feedback_into_sampling_bank": "recent_examples_explicit",
                "constraints": [
                    "prepend a small per-request diversity nonce",
                    "sample a small rotating subset from the original seed bank",
                    "include most recent accepted generated questions in later prompts",
                    "make each question structurally different from sampled examples",
                    "do not request or provide category labels",
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
            seed_examples=seed_examples,
            recent_examples=recent_examples,
            rng=rng,
            seed_examples_per_prompt=args.seed_examples_per_prompt,
            recent_examples_per_prompt=args.recent_examples_per_prompt,
            nonce_chars=args.nonce_chars,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            token_limit_param=args.token_limit_param,
            temperature=None if args.omit_temperature else args.temperature,
            top_p=None if args.omit_top_p else args.top_p,
            max_retries=args.max_retries,
        )
        key = _question_key(row["question"])
        if key in seen_questions:
            continue
        seen_questions.add(key)
        rows.append(row)
        recent_examples.append(_row_to_question_bank_example(row))
        print(f"[{len(rows)}/{args.count}] {row['question'][:100]}", flush=True)
        if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
            _write_output()

    if len(rows) < args.count:
        raise SystemExit(f"Generated only {len(rows)} unique questions after {attempts} attempts")

    _write_output()
    print(f"Wrote {len(rows)} questions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
