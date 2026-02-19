#!/usr/bin/env python3
"""Sweep solver temperature on a single question to find where it starts failing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo_math.self_play.generate_pairwise_data import (
    _openai_generate_texts,
    _parse_solver_final_answer,
    _read_env_var_from_dotenv,
)

QUESTIONS = [
    {
        "label": "modular-inverse",
        "question": (
            "Find the smallest positive integer x such that 137x ≡ 1 (mod 256)."
        ),
        "oracle": 185,
    },
    {
        "label": "digit-sum-powers",
        "question": (
            "Let S(n) denote the sum of digits of n. Find S(S(S(4444^4444)))."
        ),
        "oracle": 7,
    },
    {
        "label": "perfect-square-param",
        "question": (
            "Find the largest integer n < 1000 such that n^2 + 85n + 2024 is a perfect square."
        ),
        "oracle": 175,
    },
    {
        "label": "trailing-zeros",
        "question": (
            "How many trailing zeros does 1000! have?"
        ),
        "oracle": 249,
    },
]

PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Solve step by step. Use plain text only, no LaTeX, no markdown.\n"
    "Keep your entire response under 30 lines.\n"
    "Your last line must be exactly:\n"
    "FINAL_ANSWER: <integer>"
)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.4, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
N_SAMPLES = 20

def run_sweep(question_text: str, oracle: int, label: str, api_key: str) -> None:
    from collections import Counter
    prompt = PROMPT_TEMPLATE.format(question=question_text)
    print(f"\n{'='*70}")
    print(f"  {label}  |  oracle={oracle}")
    print(f"  {question_text[:100]}...")
    print(f"{'='*70}")
    print(f"{'temp':>5}  {'correct':>7}  {'parsed':>6}  {'wrong':>5}  {'unparsed':>8}  top wrong answers")
    print("-" * 70)

    for temp in TEMPERATURES:
        try:
            outputs = _openai_generate_texts(
                prompts=[prompt] * N_SAMPLES,
                model="gemini-2.5-flash",
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                temperature=temp,
                top_p=0.95,
                max_completion_tokens=512,
                timeout_s=60.0,
                max_tokens_param="max_tokens",
                reasoning_effort="none",
                min_interval_s=0.0,
                max_retries=1,
                initial_backoff_s=0.5,
                max_parallel=20,
            )
        except Exception as e:
            print(f"{temp:>5.1f}  ERROR: {e}")
            continue

        answers = [_parse_solver_final_answer(o) for o in outputs]
        n_correct = sum(1 for a in answers if a == oracle)
        n_parsed = sum(1 for a in answers if a is not None)
        n_wrong = n_parsed - n_correct
        n_unparsed = N_SAMPLES - n_parsed

        wrong_counts = Counter(a for a in answers if a is not None and a != oracle)
        wrong_str = " ".join(f"{a}×{c}" for a, c in wrong_counts.most_common(4))

        marker = "✓" if n_correct >= N_SAMPLES * 0.8 else ("~" if n_correct >= N_SAMPLES * 0.4 else "✗")
        print(
            f"{temp:>5.1f}  {marker} {n_correct:>2}/{N_SAMPLES:<2}  "
            f"{n_parsed:>3}/{N_SAMPLES}  {n_wrong:>5}  {n_unparsed:>8}  {wrong_str}"
        )


def main():
    api_key = (
        _read_env_var_from_dotenv("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Temperature sweep — {N_SAMPLES} samples per temp per question")
    for q in QUESTIONS:
        run_sweep(q["question"], q["oracle"], q["label"], api_key)

if __name__ == "__main__":
    main()
