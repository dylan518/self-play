from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo_math.self_play import strong_verifier_smoke


class TestStrongVerifierSmoke(unittest.TestCase):
    def test_cases_have_expected_verdicts(self) -> None:
        self.assertGreaterEqual(len(strong_verifier_smoke.CASES), 3)
        self.assertTrue(all("expected" in case for case in strong_verifier_smoke.CASES))

    def test_reads_key_from_dotenv(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY='test-key'\n", encoding="utf-8")

            self.assertEqual(
                strong_verifier_smoke._read_env_var_from_dotenv("GEMINI_API_KEY", path),
                "test-key",
            )


if __name__ == "__main__":
    unittest.main()
