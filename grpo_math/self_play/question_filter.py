from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


REASON_MALFORMED = "malformed"
REASON_DUPLICATE = "duplicate"
REASON_AMBIGUOUS = "ambiguous"
REASON_NON_INTEGER = "non_integer"
REASON_ANSWER_LEAKING = "answer_leaking"
REASON_EXTERNAL_CONTEXT = "external_context"
REASON_OFF_DOMAIN = "off_domain"
REASON_TRIVIAL = "trivial"
REASON_IMPOSSIBLE = "impossible"
REASON_LLM_REJECTED = "llm_rejected"
REASON_LLM_UNPARSEABLE = "llm_unparseable"

ACCEPTED = "accepted"
REJECTED = "rejected"

DEFAULT_MIN_WORDS = 7
DEFAULT_MIN_NUMERIC_TOKENS = 2

_WORD_RE = re.compile(r"[A-Za-z]+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")
_QUESTION_TAG_RE = re.compile(r"^\s*QUESTION\s*:\s*", flags=re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"\bFINAL_ANSWER\s*:", flags=re.IGNORECASE)
_ANSWER_FIELD_RE = re.compile(r"\b(?:ANSWER|SOLUTION|HINT|EXPLANATION)\s*:", flags=re.IGNORECASE)
_ANSWER_LEAK_RE = re.compile(
    r"\b(?:the\s+answer\s+is|answer\s*[:=]|solution\s*[:=]|final\s+answer\s+is|"
    r"therefore\s+the\s+answer)\b",
    flags=re.IGNORECASE,
)
_AMBIGUOUS_RE = re.compile(
    r"\b(?:opinion|favorite|best|approximately|estimate|about|roughly|could be|"
    r"any\s+number|many\s+answers|open[- ]ended|explain\s+why|prove\s+that|show\s+that)\b",
    flags=re.IGNORECASE,
)
_NON_INTEGER_RE = re.compile(
    r"\b(?:decimal|nearest\s+tenth|nearest\s+hundredth|fraction|ratio|probability|percent|"
    r"percentage|real\s+number|non[- ]integer|irrational|rational\s+number)\b",
    flags=re.IGNORECASE,
)
_EXTERNAL_CONTEXT_RE = re.compile(
    r"\b(?:above|below|attached|image|diagram|graph|table|article|passage|previous|"
    r"following\s+figure|internet|website|file|spreadsheet)\b",
    flags=re.IGNORECASE,
)
_OFF_DOMAIN_RE = re.compile(
    r"\b(?:write\s+(?:a|an)\s+(?:essay|poem|story)|translate|summarize|capital\s+of|"
    r"who\s+was|when\s+was|history|biology|chemistry|recipe)\b",
    flags=re.IGNORECASE,
)
_IMPOSSIBLE_RE = re.compile(
    r"\b(?:impossible|no\s+solution|infinitely\s+many|all\s+integers|any\s+integer|"
    r"smallest\s+positive\s+real|less\s+than\s+every\s+positive\s+real\s+number)\b",
    flags=re.IGNORECASE,
)
_DIRECT_ARITHMETIC_RE = re.compile(
    r"^\s*(?:what\s+is|compute|calculate|find)\s+[-\d\s+*/().^]+\??\s*$",
    flags=re.IGNORECASE,
)
_LLM_DECISION_RE = re.compile(
    r"\b(?:DECISION|VERDICT|FILTER)\s*:\s*(ACCEPT|REJECT)\b",
    flags=re.IGNORECASE,
)
_LLM_REASON_RE = re.compile(r"\bREASONS?\s*:\s*(.+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class QuestionFilterConfig:
    min_words: int = DEFAULT_MIN_WORDS
    min_numeric_tokens: int = DEFAULT_MIN_NUMERIC_TOKENS
    reject_direct_arithmetic: bool = True


@dataclass
class QuestionFilterResult:
    question: str
    status: str
    reasons: list[str] = field(default_factory=list)
    normalized_question: str = ""
    trainable_for_solver: bool = False
    trainable_for_proposer: bool = True
    question_reward: float = 0.0
    source: str = "deterministic"
    llm_decision: str | None = None
    llm_raw_output: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_question(question: str) -> str:
    normalized = _QUESTION_TAG_RE.sub("", question).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def question_fingerprint(question: str) -> str:
    normalized = normalize_question(question).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def parse_llm_filter_output(text: str) -> tuple[bool | None, list[str]]:
    decision_match = _LLM_DECISION_RE.search(text)
    if not decision_match:
        return None, [REASON_LLM_UNPARSEABLE]

    accepted = decision_match.group(1).upper() == "ACCEPT"
    reason_match = _LLM_REASON_RE.search(text)
    reasons: list[str] = []
    if reason_match:
        raw_reasons = re.split(r"[,;]", reason_match.group(1).strip())
        reasons = [r.strip().lower().replace(" ", "_") for r in raw_reasons if r.strip()]
    if not accepted and not reasons:
        reasons = [REASON_LLM_REJECTED]
    return accepted, reasons


def filter_question(
    question: str,
    *,
    seen_fingerprints: set[str] | None = None,
    llm_output: str | None = None,
    config: QuestionFilterConfig | None = None,
) -> QuestionFilterResult:
    cfg = config or QuestionFilterConfig()
    normalized = normalize_question(question)
    reasons = _deterministic_rejection_reasons(normalized, question, cfg)
    fingerprint = question_fingerprint(normalized)

    if seen_fingerprints is not None:
        if fingerprint and fingerprint in seen_fingerprints:
            reasons.append(REASON_DUPLICATE)
        elif fingerprint:
            seen_fingerprints.add(fingerprint)

    llm_decision: str | None = None
    if llm_output is not None:
        llm_accepted, llm_reasons = parse_llm_filter_output(llm_output)
        if llm_accepted is None:
            reasons.extend(llm_reasons)
            llm_decision = "UNPARSEABLE"
        elif not llm_accepted:
            reasons.extend(llm_reasons or [REASON_LLM_REJECTED])
            llm_decision = "REJECT"
        else:
            llm_decision = "ACCEPT"

    deduped_reasons = list(dict.fromkeys(reasons))
    status = ACCEPTED if not deduped_reasons else REJECTED
    return QuestionFilterResult(
        question=question,
        status=status,
        reasons=deduped_reasons,
        normalized_question=normalized,
        trainable_for_solver=status == ACCEPTED,
        trainable_for_proposer=True,
        question_reward=0.0,
        source="deterministic" if llm_output is None else "deterministic+llm",
        llm_decision=llm_decision,
        llm_raw_output=llm_output,
    )


def filter_questions(
    questions: Sequence[str],
    *,
    llm_outputs: Sequence[str | None] | None = None,
    config: QuestionFilterConfig | None = None,
) -> list[QuestionFilterResult]:
    if llm_outputs is not None and len(llm_outputs) != len(questions):
        raise ValueError("llm_outputs must have the same length as questions.")
    seen: set[str] = set()
    return [
        filter_question(
            question,
            seen_fingerprints=seen,
            llm_output=None if llm_outputs is None else llm_outputs[idx],
            config=config,
        )
        for idx, question in enumerate(questions)
    ]


def rejected_question_record(
    result: QuestionFilterResult,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    record = result.to_dict()
    record["question_reward"] = 0.0
    record["trainable_for_solver"] = False
    record["trainable_for_proposer"] = True
    if metadata:
        record["metadata"] = metadata
    return record


def write_rejected_questions(
    results: Iterable[QuestionFilterResult],
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
    append: bool = True,
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for result in results:
            if result.accepted:
                continue
            f.write(json.dumps(rejected_question_record(result, metadata=metadata), sort_keys=True) + "\n")
            count += 1
    return count


def _deterministic_rejection_reasons(
    normalized: str,
    raw_question: str,
    config: QuestionFilterConfig,
) -> list[str]:
    reasons: list[str] = []
    if not normalized or "\n" in raw_question.strip() or not normalized.endswith("?"):
        reasons.append(REASON_MALFORMED)

    if (
        _FINAL_ANSWER_RE.search(raw_question)
        or _ANSWER_FIELD_RE.search(raw_question)
        or _ANSWER_LEAK_RE.search(raw_question)
    ):
        reasons.append(REASON_ANSWER_LEAKING)

    if _AMBIGUOUS_RE.search(normalized):
        reasons.append(REASON_AMBIGUOUS)
    if _NON_INTEGER_RE.search(normalized):
        reasons.append(REASON_NON_INTEGER)
    if _EXTERNAL_CONTEXT_RE.search(normalized):
        reasons.append(REASON_EXTERNAL_CONTEXT)
    if _OFF_DOMAIN_RE.search(normalized):
        reasons.append(REASON_OFF_DOMAIN)
    if _IMPOSSIBLE_RE.search(normalized):
        reasons.append(REASON_IMPOSSIBLE)

    words = _WORD_RE.findall(normalized)
    numbers = _NUMBER_RE.findall(normalized)
    if len(words) < config.min_words:
        reasons.append(REASON_TRIVIAL)
    if config.reject_direct_arithmetic and _DIRECT_ARITHMETIC_RE.match(normalized):
        reasons.append(REASON_TRIVIAL)
    if len(numbers) < config.min_numeric_tokens and not _looks_symbolic_enough(normalized):
        reasons.append(REASON_TRIVIAL)

    return reasons


def _looks_symbolic_enough(question: str) -> bool:
    lower = question.lower()
    symbolic_terms = (
        "integer",
        "positive",
        "consecutive",
        "divisible",
        "remainder",
        "modulo",
        "factor",
        "prime",
    )
    return any(term in lower for term in symbolic_terms) and bool(re.search(r"[a-z]\s*[=<>]|\b\d+[a-z]\b", lower))
