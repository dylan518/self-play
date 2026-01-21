import unittest

import torch

from grpo_math.models.policy import _gather_logprobs_for_labels, sequence_logprobs


class TestLogprobs(unittest.TestCase):
    def test_gather_logprobs_matches_log_softmax(self) -> None:
        torch.manual_seed(0)
        b, t, v = 2, 3, 11
        logits = torch.randn(b, t, v, dtype=torch.float32)
        labels = torch.randint(0, v, (b, t), dtype=torch.long)

        # Reference implementation
        ref = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        got = _gather_logprobs_for_labels(logits, labels)
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

    def test_gather_logprobs_preserves_dtype(self) -> None:
        torch.manual_seed(0)
        b, t, v = 2, 3, 11
        logits = torch.randn(b, t, v, dtype=torch.float16)
        labels = torch.randint(0, v, (b, t), dtype=torch.long)
        got = _gather_logprobs_for_labels(logits, labels)
        self.assertEqual(got.dtype, logits.dtype)

    def test_sequence_logprobs_microbatch_equivalence(self) -> None:
        # Use a tiny random model from HF if available; otherwise skip.
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:  # pragma: no cover
            self.skipTest(f"transformers not available: {e}")

        model_name = "hf-internal-testing/tiny-random-gpt2"
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:  # pragma: no cover
            self.skipTest(f"cannot load {model_name}: {e}")

        model.eval()
        torch.manual_seed(0)

        # Fake batch: [B, T]
        bsz, t = 6, 12
        vocab = model.config.vocab_size
        input_ids = torch.randint(0, vocab, (bsz, t), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = torch.full((bsz,), 5, dtype=torch.long)  # generated starts at pos 5

        with torch.no_grad():
            full_sum, full_mask = sequence_logprobs(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_ends=prompt_lens,
                microbatch_size=bsz,
            )
            mb_sum, mb_mask = sequence_logprobs(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_ends=prompt_lens,
                microbatch_size=2,
            )

        self.assertTrue(torch.allclose(full_sum, mb_sum, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(full_mask, mb_mask))


if __name__ == "__main__":
    unittest.main()

