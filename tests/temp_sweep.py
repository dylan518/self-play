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
        "label": "Q38 pool-rate",
        "question": (
            "A rectangular swimming pool has a length of 50 meters and a width of 25 meters. "
            "It is filled with water to a depth of 2 meters. If water evaporates at a rate of "
            "0.5 cubic meters per hour, and a filtration system pumps water out at a rate of "
            "1.2 cubic meters per hour for 8 hours each day, while a hose adds water at a rate "
            "of 0.8 cubic meters per hour for 6 hours each day, what will be the depth of the "
            "water in centimeters after 10 full days, assuming no one uses the pool?"
        ),
        "oracle": 187,
    },
    {
        "label": "Q17 flour-profit",
        "question": (
            "A baker starts with a 50 kg bag of flour. He uses 1/5 of it to bake cookies, "
            "then 1/4 of the remaining flour for cakes. The rest of the flour is divided "
            "equally into 12 smaller bags for sale. If each smaller bag is sold for $3, and "
            "the baker spent $45 on the original 50 kg bag, what is his total profit from "
            "the sale of these smaller bags?"
        ),
        "oracle": -9,
    },
    {
        "label": "Q49 bakery-revenue",
        "question": (
            "A small artisanal bakery bakes loaves of bread for local restaurants. On Monday, "
            "they produced 120 loaves. On Tuesday, they increased their production by 25% "
            "compared to Monday. On Wednesday, they produced 10% fewer loaves than they did "
            "on Tuesday. For every 10 loaves produced, 3 are rye bread, and the rest are "
            "sourdough. Each rye bread loaf sells for $6, and each sourdough loaf sells for $8. "
            "If all the loaves produced from Monday to Wednesday were sold, what was the total "
            "revenue in dollars for these three days?"
        ),
        "oracle": 2997,
    },
]

PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Solve step by step. Use plain text only, no LaTeX, no markdown.\n"
    "Keep your entire response under 10 lines.\n"
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
