from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml


SCHEMA_VERSION = "self_play_verdict_grpo_v1"
VALID_VERDICTS = {"CORRECT", "INCORRECT", "INVALID", "UNCLEAR"}
_ANSWER_LEAK_RE = re.compile(r"\b(?:answer|final_answer|solution)\s*[:=]", flags=re.IGNORECASE)
_QUESTION_WORD_RE = re.compile(r"\b(?:what|how many|find|compute|calculate|determine)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ProposedQuestion:
    prompt_id: str
    prompt: str
    question_id: str
    question: str
    raw_output: str


@dataclass(frozen=True)
class SolverCompletion:
    completion_id: str
    text: str


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    confidence: float = 1.0
    rationale: str = ""


class ProposerProvider(Protocol):
    def propose(self) -> list[ProposedQuestion]:
        ...


class SolverProvider(Protocol):
    def solve(self, question: ProposedQuestion, *, num_completions: int) -> list[SolverCompletion]:
        ...


class VerifierProvider(Protocol):
    def verify(self, question: str, completions: list[SolverCompletion]) -> list[VerificationResult]:
        ...


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deterministic_question_filter(question: str, seen_questions: set[str]) -> FilterResult:
    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return FilterResult(False, "empty")
    if normalized in seen_questions:
        return FilterResult(False, "duplicate")
    if len(question.strip()) < 12:
        return FilterResult(False, "too_short")
    if "?" not in question:
        return FilterResult(False, "missing_question_mark")
    if _ANSWER_LEAK_RE.search(question):
        return FilterResult(False, "answer_leak")
    if not re.search(r"\d", question):
        return FilterResult(False, "missing_numbers")
    if not _QUESTION_WORD_RE.search(question):
        return FilterResult(False, "not_math_question")
    return FilterResult(True, "accepted")


def verdict_to_reward(verdict: str) -> float:
    return 1.0 if verdict.upper() == "CORRECT" else 0.0


def centered_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    return [reward - mean_reward for reward in rewards]


def question_reward_from_solver_rewards(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    mean_reward = sum(rewards) / len(rewards)
    # Prefer questions with mixed outcomes: all-correct and all-incorrect batches
    # carry little learning signal for the solver.
    return 1.0 - abs((2.0 * mean_reward) - 1.0)


class FixtureProposerProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        prompts = config.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("fixture.proposer.prompts must be a non-empty list")
        self.prompts = prompts

    def propose(self) -> list[ProposedQuestion]:
        proposed: list[ProposedQuestion] = []
        for prompt_idx, prompt_cfg in enumerate(self.prompts):
            prompt_id = str(prompt_cfg.get("id", f"prompt_{prompt_idx:03d}"))
            prompt = str(prompt_cfg.get("prompt", ""))
            questions = prompt_cfg.get("questions", [])
            if not isinstance(questions, list):
                raise ValueError(f"questions for {prompt_id} must be a list")
            for question_idx, item in enumerate(questions):
                if isinstance(item, dict):
                    raw_output = str(item.get("raw_output", item.get("question", "")))
                    question = str(item.get("question", raw_output)).strip()
                else:
                    raw_output = str(item)
                    question = raw_output.strip()
                proposed.append(
                    ProposedQuestion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        question_id=f"{prompt_id}_q{question_idx:03d}",
                        question=question,
                        raw_output=raw_output,
                    )
                )
        return proposed


class FixtureSolverProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.default_completions = [str(item) for item in config.get("default_completions", [])]
        self.by_question_id = {
            str(key): [str(item) for item in value]
            for key, value in (config.get("completions_by_question_id", {}) or {}).items()
        }

    def solve(self, question: ProposedQuestion, *, num_completions: int) -> list[SolverCompletion]:
        texts = self.by_question_id.get(question.question_id, self.default_completions)
        if len(texts) < num_completions:
            raise ValueError(
                f"Fixture solver has {len(texts)} completions for {question.question_id}; "
                f"need {num_completions}"
            )
        return [
            SolverCompletion(completion_id=f"{question.question_id}_c{idx:03d}", text=text)
            for idx, text in enumerate(texts[:num_completions])
        ]


class FixtureVerifierProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.default_verdicts = [str(item).upper() for item in config.get("default_verdicts", [])]
        self.by_question_id = {
            str(key): [str(item).upper() for item in value]
            for key, value in (config.get("verdicts_by_question_id", {}) or {}).items()
        }
        for verdict in [*self.default_verdicts, *[v for values in self.by_question_id.values() for v in values]]:
            if verdict not in VALID_VERDICTS:
                raise ValueError(f"Unsupported fixture verifier verdict: {verdict}")

    def verify(self, question: str, completions: list[SolverCompletion]) -> list[VerificationResult]:
        del question
        question_id = completions[0].completion_id.rsplit("_c", maxsplit=1)[0] if completions else ""
        verdicts = self.by_question_id.get(question_id, self.default_verdicts)
        if len(verdicts) < len(completions):
            raise ValueError(
                f"Fixture verifier has {len(verdicts)} verdicts for {question_id}; "
                f"need {len(completions)}"
            )
        return [VerificationResult(verdict=verdict) for verdict in verdicts[: len(completions)]]


def build_fixture_providers(config: dict[str, Any]) -> tuple[ProposerProvider, SolverProvider, VerifierProvider]:
    fixture = config.get("fixture", {})
    if not isinstance(fixture, dict):
        raise ValueError("fixture config must be a mapping")
    return (
        FixtureProposerProvider(fixture.get("proposer", {}) or {}),
        FixtureSolverProvider(fixture.get("solver", {}) or {}),
        FixtureVerifierProvider(fixture.get("verifier", {}) or {}),
    )


def _json_ready(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(_json_ready(row) + "\n")


def collect_rollouts(
    config: dict[str, Any],
    proposer: ProposerProvider,
    solver: SolverProvider,
    verifier: VerifierProvider,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rollout_cfg = config.get("rollout", {}) or {}
    output_cfg = config.get("output", {}) or {}
    num_solver_completions = int(rollout_cfg.get("num_solver_completions", 4))
    iteration_id = str(rollout_cfg.get("iteration_id", "local_fixture"))
    created_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for proposed in proposer.propose():
        filter_result = deterministic_question_filter(proposed.question, seen_questions)
        normalized_question = " ".join(proposed.question.strip().lower().split())
        if filter_result.accepted:
            seen_questions.add(normalized_question)

        solver_entries: list[dict[str, Any]] = []
        solver_rewards: list[float] = []
        verdict_counts = {verdict: 0 for verdict in sorted(VALID_VERDICTS)}

        if filter_result.accepted:
            completions = solver.solve(proposed, num_completions=num_solver_completions)
            verifications = verifier.verify(proposed.question, completions)
            if len(verifications) != len(completions):
                raise ValueError("Verifier returned a different number of results than completions")
            solver_rewards = [verdict_to_reward(result.verdict) for result in verifications]
            solver_advantages = centered_advantages(solver_rewards)
            for completion, verification, reward, advantage in zip(
                completions, verifications, solver_rewards, solver_advantages, strict=True
            ):
                verdict_counts[verification.verdict] += 1
                solver_entries.append(
                    {
                        "completion_id": completion.completion_id,
                        "text": completion.text,
                        "verdict": verification.verdict,
                        "verifier_confidence": verification.confidence,
                        "verifier_rationale": verification.rationale,
                        "reward": reward,
                        "advantage": advantage,
                    }
                )

        question_reward = (
            question_reward_from_solver_rewards(solver_rewards) if filter_result.accepted else 0.0
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "iteration_id": iteration_id,
            "created_at": created_at,
            "proposer_prompt_id": proposed.prompt_id,
            "proposer_prompt": proposed.prompt,
            "question_id": proposed.question_id,
            "question": proposed.question,
            "raw_question_output": proposed.raw_output,
            "filter": {"accepted": filter_result.accepted, "reason": filter_result.reason},
            "trainable_for_solver": filter_result.accepted,
            "trainable_for_proposer": True,
            "question_reward": question_reward,
            "question_advantage": 0.0,
            "solver_completions": solver_entries,
            "solver_verdict_counts": verdict_counts,
        }
        rows.append(row)

        if not filter_result.accepted:
            rejected_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "iteration_id": iteration_id,
                    "created_at": created_at,
                    "proposer_prompt_id": proposed.prompt_id,
                    "question_id": proposed.question_id,
                    "question": proposed.question,
                    "filter": row["filter"],
                    "question_reward": 0.0,
                    "trainable_for_solver": False,
                    "trainable_for_proposer": True,
                }
            )

    rows_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_prompt.setdefault(str(row["proposer_prompt_id"]), []).append(row)
    for prompt_rows in rows_by_prompt.values():
        advantages = centered_advantages([float(row["question_reward"]) for row in prompt_rows])
        for row, advantage in zip(prompt_rows, advantages, strict=True):
            row["question_advantage"] = advantage

    if output_cfg.get("include_summary", True):
        accepted_count = sum(1 for row in rows if row["filter"]["accepted"])
        print(
            "Collected "
            f"{len(rows)} questions ({accepted_count} accepted, {len(rejected_rows)} rejected)"
        )
    return rows, rejected_rows


def run_from_config(
    config_path: str | Path,
    *,
    overwrite: bool | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    provider_type = str(config.get("provider_type", "fixture"))
    if provider_type != "fixture":
        raise ValueError(f"Unsupported provider_type for local collector: {provider_type}")
    proposer, solver, verifier = build_fixture_providers(config)
    rows, rejected_rows = collect_rollouts(config, proposer, solver, verifier)

    output_cfg = config.get("output", {}) or {}
    if output_dir is not None:
        out_dir = Path(output_dir)
        rollout_path = out_dir / "rollouts.jsonl"
        rejected_path = out_dir / "rejected_questions.jsonl"
    else:
        rollout_path = Path(output_cfg.get("rollout_jsonl_path", "outputs/self_play_rollouts/rollouts.jsonl"))
        rejected_path = Path(
            output_cfg.get("rejected_jsonl_path", rollout_path.parent / "rejected_questions.jsonl")
        )
    should_overwrite = bool(output_cfg.get("overwrite", True)) if overwrite is None else overwrite
    _write_jsonl(rollout_path, rows, overwrite=should_overwrite)
    _write_jsonl(rejected_path, rejected_rows, overwrite=should_overwrite)
    return rollout_path, rejected_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verdict-only self-play rollouts.")
    parser.add_argument("--config", required=True, help="Path to rollout YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override configured output paths with this directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite configured output files.")
    parser.add_argument("--append", action="store_true", help="Append to configured output files.")
    args = parser.parse_args()
    if args.overwrite and args.append:
        raise SystemExit("--overwrite and --append are mutually exclusive")
    overwrite = True if args.overwrite else False if args.append else None
    rollout_path, rejected_path = run_from_config(args.config, overwrite=overwrite, output_dir=args.output_dir)
    print(f"Wrote rollouts to {rollout_path}")
    print(f"Wrote rejected questions to {rejected_path}")


if __name__ == "__main__":
    main()
