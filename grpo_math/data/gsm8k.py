from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class GSM8KExample:
    question: str
    answer_text: str


def load_gsm8k(
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[GSM8KExample]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), int(max_samples))))
    out: List[GSM8KExample] = []
    for row in ds:
        out.append(GSM8KExample(question=row["question"], answer_text=row["answer"]))
    return out


def format_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def batch_prompts(examples: List[GSM8KExample], template: str) -> List[str]:
    return [format_prompt(template=template, question=ex.question) for ex in examples]


def as_dict(ex: GSM8KExample) -> Dict[str, Any]:
    return {"question": ex.question, "answer_text": ex.answer_text}

