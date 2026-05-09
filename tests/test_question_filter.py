import json
import tempfile
import unittest
from pathlib import Path

from grpo_math.self_play.question_filter import (
    ACCEPTED,
    REASON_AMBIGUOUS,
    REASON_ANSWER_LEAKING,
    REASON_DUPLICATE,
    REASON_EXTERNAL_CONTEXT,
    REASON_IMPOSSIBLE,
    REASON_LLM_REJECTED,
    REASON_LLM_UNPARSEABLE,
    REASON_MALFORMED,
    REASON_NON_INTEGER,
    REASON_OFF_DOMAIN,
    REASON_TRIVIAL,
    REJECTED,
    filter_question,
    filter_questions,
    parse_llm_filter_output,
    write_rejected_questions,
)


ACCEPTABLE_QUESTION = (
    "Let n be the smallest positive integer such that n leaves a remainder of 2 when divided by 5 "
    "and a remainder of 3 when divided by 7. What is n?"
)


class TestQuestionFilter(unittest.TestCase):
    def test_accepts_self_contained_integer_math_question(self) -> None:
        result = filter_question(ACCEPTABLE_QUESTION)

        self.assertEqual(result.status, ACCEPTED)
        self.assertEqual(result.reasons, [])
        self.assertTrue(result.trainable_for_solver)
        self.assertTrue(result.trainable_for_proposer)
        self.assertEqual(result.question_reward, 0.0)

    def test_rejects_malformed_question(self) -> None:
        result = filter_question("Find the integer n such that n leaves remainder 2 when divided by 5")

        self.assertEqual(result.status, REJECTED)
        self.assertIn(REASON_MALFORMED, result.reasons)
        self.assertFalse(result.trainable_for_solver)

    def test_rejects_duplicates_in_batch(self) -> None:
        results = filter_questions([ACCEPTABLE_QUESTION, "QUESTION: " + ACCEPTABLE_QUESTION])

        self.assertEqual(results[0].status, ACCEPTED)
        self.assertEqual(results[1].status, REJECTED)
        self.assertIn(REASON_DUPLICATE, results[1].reasons)

    def test_rejects_ambiguous_question(self) -> None:
        result = filter_question(
            "About how many integers would be best to choose from 10, 20, and 30 in your opinion?"
        )

        self.assertIn(REASON_AMBIGUOUS, result.reasons)

    def test_rejects_non_integer_question(self) -> None:
        result = filter_question(
            "A spinner has 2 red sections and 3 blue sections. What probability is red on one spin?"
        )

        self.assertIn(REASON_NON_INTEGER, result.reasons)

    def test_rejects_answer_leaking_question(self) -> None:
        result = filter_question("If x + 7 = 19, what is x? ANSWER: 12")

        self.assertIn(REASON_ANSWER_LEAKING, result.reasons)

    def test_rejects_external_context_question(self) -> None:
        result = filter_question("Using the diagram above with 3 labeled lengths, what is the missing angle?")

        self.assertIn(REASON_EXTERNAL_CONTEXT, result.reasons)

    def test_rejects_off_domain_question(self) -> None:
        result = filter_question("Write a poem with 12 lines about prime numbers and include 3 metaphors?")

        self.assertIn(REASON_OFF_DOMAIN, result.reasons)

    def test_rejects_trivial_question(self) -> None:
        result = filter_question("What is 2 + 2?")

        self.assertIn(REASON_TRIVIAL, result.reasons)

    def test_rejects_impossible_question(self) -> None:
        result = filter_question(
            "What is the largest integer n such that n is less than every positive real number?"
        )

        self.assertIn(REASON_IMPOSSIBLE, result.reasons)

    def test_parse_llm_filter_output_accept(self) -> None:
        accepted, reasons = parse_llm_filter_output("DECISION: ACCEPT\nREASONS: none")

        self.assertTrue(accepted)
        self.assertEqual(reasons, ["none"])

    def test_parse_llm_filter_output_reject(self) -> None:
        accepted, reasons = parse_llm_filter_output("DECISION: REJECT\nREASONS: ambiguous, non integer")

        self.assertFalse(accepted)
        self.assertEqual(reasons, ["ambiguous", "non_integer"])

    def test_parse_llm_filter_output_unparseable(self) -> None:
        accepted, reasons = parse_llm_filter_output("Looks okay to me.")

        self.assertIsNone(accepted)
        self.assertEqual(reasons, [REASON_LLM_UNPARSEABLE])

    def test_llm_rejection_is_applied(self) -> None:
        result = filter_question(ACCEPTABLE_QUESTION, llm_output="DECISION: REJECT\nREASONS: llm rejected")

        self.assertEqual(result.status, REJECTED)
        self.assertIn(REASON_LLM_REJECTED, result.reasons)
        self.assertEqual(result.llm_decision, "REJECT")

    def test_write_rejected_questions_sets_zero_reward_fields(self) -> None:
        results = filter_questions([ACCEPTABLE_QUESTION, "What is 2 + 2?"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rejected_questions.jsonl"
            count = write_rejected_questions(results, output_path, metadata={"run_id": "test"}, append=False)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], REJECTED)
        self.assertEqual(rows[0]["question_reward"], 0.0)
        self.assertFalse(rows[0]["trainable_for_solver"])
        self.assertTrue(rows[0]["trainable_for_proposer"])
        self.assertEqual(rows[0]["metadata"], {"run_id": "test"})

    def test_filtered_questions_do_not_call_solver_or_verifier(self) -> None:
        results = filter_questions([ACCEPTABLE_QUESTION, "What is 2 + 2?"])
        solver_calls: list[str] = []
        verifier_calls: list[tuple[str, str]] = []

        def fake_solver(question: str) -> list[str]:
            solver_calls.append(question)
            return ["FINAL_ANSWER: 23"]

        def fake_verifier(question: str, completion: str) -> str:
            verifier_calls.append((question, completion))
            return "CORRECT"

        for result in results:
            if not result.trainable_for_solver:
                continue
            for completion in fake_solver(result.normalized_question):
                fake_verifier(result.normalized_question, completion)

        self.assertEqual(solver_calls, [ACCEPTABLE_QUESTION])
        self.assertEqual(verifier_calls, [(ACCEPTABLE_QUESTION, "FINAL_ANSWER: 23")])


if __name__ == "__main__":
    unittest.main()
