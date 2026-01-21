import unittest

import torch

from grpo_math.models.policy import sequence_logprobs


class _FakeModel:
    """Returns fixed logits so we can test masking deterministically without HF models."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        b, t = input_ids.shape
        # logits = 0 everywhere => logprob uniform
        logits = torch.zeros((b, t, self.vocab_size), dtype=torch.float32, device=input_ids.device)
        return type("Out", (), {"logits": logits})


class TestLeftPaddingMasking(unittest.TestCase):
    def test_prompt_end_mask_for_left_padding(self) -> None:
        # Construct a left-padded sequence:
        # pad pad A B  (prompt ends at index 4)
        # then generated tokens C D appended => pad pad A B C D
        # We want generated tokens to be those at positions >= prompt_end (4): C, D
        vocab = 10
        model = _FakeModel(vocab)

        input_ids = torch.tensor([[0, 0, 3, 4, 5, 6]], dtype=torch.long)  # [1,6]
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        prompt_end = torch.tensor([4], dtype=torch.long)  # generation begins at token position 4

        with torch.no_grad():
            sum_logp, gen_mask = sequence_logprobs(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_ends=prompt_end,
                microbatch_size=1,
            )

        # gen_mask aligns to labels positions (T-1 = 5). Token positions for labels are 1..5.
        # Generated token positions are 4 and 5 -> label indices 3 and 4 should be 1.
        expected = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float32)
        self.assertTrue(torch.equal(gen_mask.cpu(), expected))
        self.assertEqual(sum_logp.shape, (1,))


if __name__ == "__main__":
    unittest.main()

