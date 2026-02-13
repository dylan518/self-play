from __future__ import annotations

import re
from typing import Optional, Tuple


_INT_RE = re.compile(r"-?\d+")
# Accept integers, and also tolerate floats that are mathematically integers (e.g. "30.00").
# We intentionally allow trailing text (e.g. "FINAL_ANSWER: 30Human: ...") because some models
# sometimes continue the transcript without inserting a newline. We still require the explicit tag.
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(-?\d+)(?:\.0+)?(?![\d.])")


def extract_ground_truth_int(gsm8k_answer_text: str) -> Optional[int]:
    """
    GSM8K answers often contain a final line like: '#### 42'.
    We take the last integer occurring anywhere in the string.
    """
    matches = _INT_RE.findall(gsm8k_answer_text)
    if not matches:
        return None
    return int(matches[-1])


def extract_final_answer_int(model_text: str) -> Optional[int]:
    """
    Expected format: a line containing 'FINAL_ANSWER: <number>'.
    We extract the first integer after the FINAL_ANSWER tag if possible; otherwise fall back to last int.
    """
    idx = model_text.rfind("FINAL_ANSWER")
    if idx != -1:
        tail = model_text[idx:]
        matches = _INT_RE.findall(tail)
        if matches:
            return int(matches[0])
    matches = _INT_RE.findall(model_text)
    if not matches:
        return None
    return int(matches[-1])


def extract_final_answer_int_strict(model_text: str) -> Optional[int]:
    """
    Strict format compliance:
      - must contain the substring `FINAL_ANSWER: <int>`
    No fallback to other integers elsewhere in the output (prevents reward hacking from scratchpad numbers).
    """
    # Take the last occurrence if multiple exist.
    m = None
    for m in _FINAL_ANSWER_RE.finditer(model_text):
        pass
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        # Guard against pathological giant digit strings from model generations.
        return None


def binary_reward(model_text: str, gsm8k_answer_text: str) -> Tuple[float, Optional[int], Optional[int]]:
    # Reward should be *strict*: only grant reward if the model followed the required
    # `FINAL_ANSWER: <int>` format, otherwise the policy can get credit from any
    # unrelated integer (e.g., in the scratchpad).
    pred = extract_final_answer_int_strict(model_text)
    gt = extract_ground_truth_int(gsm8k_answer_text)
    if pred is None or gt is None:
        return 0.0, pred, gt
    return (1.0 if pred == gt else 0.0), pred, gt

