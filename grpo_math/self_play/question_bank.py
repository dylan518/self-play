from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_COMPATIBLE_CATEGORIES = frozenset({"algebra", "symbolic_manipulation", "logic", "optimization"})
_INCOMPATIBLE_MARKERS = (
    "write a function",
    "derive ",
    "closed-form",
    "poem",
    "estimate",
    "formulate a question",
    "find all functions",
    "find all real solutions",
    "all real solutions",
    "all integer solutions",
    "valid arrangement",
    "which statements",
    "determine whether",
    "shortest path",
)
_INTEGER_ANSWER_MARKERS = (
    "how many",
    "find the smallest",
    "find the remainder",
    "solve for x",
    "value of k",
    "integer",
    "integers",
    "two-digit",
    "area",
    "longer side",
    "maximum",
    "minimum",
)


@dataclass(frozen=True)
class QuestionBankExample:
    category: str
    task: str
    question: str
    verification: str


def load_question_bank(path: str | Path) -> list[QuestionBankExample]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    examples: list[QuestionBankExample] = []
    categories = data.get("categories", {})
    if isinstance(categories, dict):
        for category, rows in categories.items():
            if isinstance(rows, list):
                examples.extend(_parse_rows(rows, category=str(category)))
    extended = data.get("tasks_extended", [])
    if isinstance(extended, list):
        examples.extend(_parse_rows(extended, category=""))
    return examples


def load_recent_generated_questions(
    path: str | Path,
    *,
    max_examples: int = 200,
) -> list[QuestionBankExample]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    examples: list[QuestionBankExample] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = str(row.get("question", "")).strip()
        question = re.sub(
            r"\n?\s*VERIFICATION\s+IDEA\s*:.*$",
            "",
            question,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if not question:
            continue
        examples.append(
            QuestionBankExample(
                category="recent_generated",
                task="avoid_repeat",
                question=question,
                verification="Avoid repeating this topic, surface form, or answer pattern.",
            )
        )
    return examples[-max_examples:]


def _parse_rows(rows: Iterable[Any], *, category: str) -> list[QuestionBankExample]:
    parsed: list[QuestionBankExample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        parsed.append(
            QuestionBankExample(
                category=str(row.get("category") or category or "uncategorized"),
                task=str(row.get("task", "")).strip(),
                question=question,
                verification=str(row.get("verification", "")).strip(),
            )
        )
    return parsed


def is_current_loop_compatible(example: QuestionBankExample) -> bool:
    """Return true when an example fits the current integer-answer solver contract."""
    text = example.question.lower()
    if example.category not in DEFAULT_COMPATIBLE_CATEGORIES:
        return False
    if any(marker in text for marker in _INCOMPATIBLE_MARKERS):
        return False
    if any(marker in text for marker in _INTEGER_ANSWER_MARKERS):
        return True
    return bool(re.search(r"\bsolve for [a-z]\b", text))


def select_question_bank_examples(
    examples: Sequence[QuestionBankExample],
    *,
    count: int,
    seed: int,
    compatible_only: bool = True,
) -> list[QuestionBankExample]:
    candidates = [ex for ex in examples if not compatible_only or is_current_loop_compatible(ex)]
    if count <= 0 or not candidates:
        return []
    rng = random.Random(seed)
    return rng.sample(list(candidates), k=min(count, len(candidates)))


def render_question_bank_block(
    examples: Sequence[QuestionBankExample],
    *,
    recent_examples: Sequence[QuestionBankExample] = (),
) -> str:
    if not examples and not recent_examples:
        return ""
    lines = [
        "Question bank examples:",
        "Generate something novel and meaningfully different from these examples.",
    ]
    for idx, ex in enumerate([*recent_examples, *examples], start=1):
        lines.append(
            f"{idx}.\n"
            f"QUESTION: {ex.question}\n"
            f"VERIFICATION IDEA: {ex.verification}"
        )
    return "\n".join(lines)


def render_question_bank_block_from_config(config: dict[str, Any], *, seed: int) -> str:
    bank_cfg = config.get("question_bank", {}) or {}
    if not isinstance(bank_cfg, dict) or not bool(bank_cfg.get("enabled", False)):
        return ""
    path = str(bank_cfg.get("path", "")).strip()
    if not path:
        raise ValueError("generator.question_bank.path is required when question bank is enabled")
    examples = load_question_bank(path)
    selected = select_question_bank_examples(
        examples,
        count=int(bank_cfg.get("num_examples", 8)),
        seed=int(bank_cfg.get("seed", seed)),
        compatible_only=bool(bank_cfg.get("compatible_only", True)),
    )
    recent_selected: list[QuestionBankExample] = []
    recent_path = str(bank_cfg.get("recent_jsonl_path", "")).strip()
    recent_count = int(bank_cfg.get("recent_num_examples", 0))
    if recent_path and recent_count > 0:
        recent_examples = load_recent_generated_questions(
            recent_path,
            max_examples=int(bank_cfg.get("recent_max_examples", 200)),
        )
        recent_selected = select_question_bank_examples(
            recent_examples,
            count=recent_count,
            seed=seed + 17_029,
            compatible_only=False,
        )
    return render_question_bank_block(selected, recent_examples=recent_selected)
