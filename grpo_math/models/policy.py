from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PolicyBundle:
    tokenizer: any
    policy: any
    ref: any | None


def load_policy_and_ref(
    name_or_path: str,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool = True,
    load_ref: bool = True,
    attn_implementation: str | None = None,
) -> PolicyBundle:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation should use left padding.
    tokenizer.padding_side = "left"

    # Attention impl can materially impact throughput (esp. on H100). We try requested impl,
    # and fall back to a safe default if the environment doesn't support it.
    policy_kwargs = dict(torch_dtype=torch_dtype, device_map=None)
    if attn_implementation is not None:
        policy_kwargs["attn_implementation"] = attn_implementation
    try:
        policy = AutoModelForCausalLM.from_pretrained(name_or_path, **policy_kwargs)
    except Exception:
        policy_kwargs.pop("attn_implementation", None)
        policy = AutoModelForCausalLM.from_pretrained(name_or_path, **policy_kwargs)
    if gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    ref = None
    if load_ref:
        ref_kwargs = dict(torch_dtype=torch_dtype, device_map=None)
        if attn_implementation is not None:
            ref_kwargs["attn_implementation"] = attn_implementation
        try:
            ref = AutoModelForCausalLM.from_pretrained(name_or_path, **ref_kwargs)
        except Exception:
            ref_kwargs.pop("attn_implementation", None)
            ref = AutoModelForCausalLM.from_pretrained(name_or_path, **ref_kwargs)
        ref.requires_grad_(False)
        ref.eval()

    return PolicyBundle(tokenizer=tokenizer, policy=policy, ref=ref)


def _gather_logprobs_for_labels(
    logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,  # [B, T]
) -> torch.Tensor:
    # Memory-efficient log-prob gather:
    # log p(y) = logit_y - logsumexp(logits)
    # Avoids materializing a full [B,T,V] log-softmax tensor (critical for large vocabs like Qwen).
    # NOTE: avoid upcasting the full logits tensor to fp32 (can double memory and OOM).
    lse = torch.logsumexp(logits, dim=-1)  # [B, T]
    chosen = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return chosen - lse


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      - texts: length B*K (completion-only text; does NOT include the prompt)
      - input_ids: [B*K, T]
      - attention_mask: [B*K, T]
      - prompt_ends: [B*K] end index (in token positions) of the prompt within the padded sequence,
        i.e. generated tokens start at this index. This is robust to left-padding.
    """
    # DeepSpeed wraps the model; generation lives on the underlying module.
    gen_model = model.module if hasattr(model, "module") else model
    # Try to infer device robustly (some wrappers don't expose parameters cleanly).
    try:
        device = next(gen_model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    prompt_lens = attention_mask.sum(dim=1)  # [B] number of non-pad tokens in prompt
    pad_lens = (attention_mask == 0).sum(dim=1)  # [B] left-pad tokens (0 for right-padding)
    prompt_ends = pad_lens + prompt_lens  # [B] index where generation begins

    # Repeat each prompt k times
    input_ids = input_ids.repeat_interleave(k, dim=0)
    attention_mask = attention_mask.repeat_interleave(k, dim=0)
    prompt_ends = prompt_ends.repeat_interleave(k, dim=0)

    # Generation should run with dropout disabled for stability.
    was_training = bool(getattr(gen_model, "training", False))
    gen_model.eval()
    gen = gen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )
    if was_training:
        gen_model.train()

    # Recompute attention mask (generation adds tokens)
    attn = (gen != tokenizer.pad_token_id).long()

    # Decode completion-only text: slice out the prompt (including left-padding).
    texts: List[str] = []
    for i in range(gen.shape[0]):
        pe = int(prompt_ends[i].item())
        end = int(attn[i].sum().item())
        end = max(pe, end)
        gen_ids = gen[i, pe:end]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return texts, gen, attn, prompt_ends


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,  # [B, T]
    attention_mask: torch.Tensor,  # [B, T]
    prompt_ends: torch.Tensor,  # [B]
    microbatch_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes:
      - sum_logprob_generated: [B] sum log p(token_t | <t) over generated tokens only
      - mean_kl_generated: [B] placeholder (filled elsewhere) for convenience
    """
    bsz = input_ids.shape[0]
    if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= bsz:
        microbatch_size = bsz

    sum_logp_chunks: list[torch.Tensor] = []
    gen_mask_chunks: list[torch.Tensor] = []

    for start in range(0, bsz, microbatch_size):
        end = min(bsz, start + microbatch_size)
        ids = input_ids[start:end]
        am0 = attention_mask[start:end]
        pe = prompt_ends[start:end]

        # Teacher forcing: logits for positions 0..T-2 predict labels 1..T-1
        outputs = model(input_ids=ids, attention_mask=am0, use_cache=False)
        logits = outputs.logits[:, :-1, :]  # [b, T-1, V]
        labels = ids[:, 1:]  # [b, T-1]
        am = am0[:, 1:]  # align with labels

        token_logp = _gather_logprobs_for_labels(logits, labels)  # [b, T-1]

        # Mask: generated tokens are those with position >= prompt_len (in the original full sequence),
        # but we're aligned to labels (shifted by 1), so compare (pos+1) >= prompt_len -> pos >= prompt_len-1.
        b, t1 = labels.shape
        pos = torch.arange(t1, device=labels.device).unsqueeze(0).expand(b, -1)  # 0..T-2
        # labels positions correspond to token positions 1..T-1 in the original sequence,
        # so a label index `pos` corresponds to token position `pos+1`.
        # Generated tokens start at token position prompt_end, so in label indices: pos >= prompt_end - 1
        gen_mask = (pos >= (pe.unsqueeze(1) - 1)).to(token_logp.dtype) * am.to(token_logp.dtype)

        sum_logp_gen = (token_logp * gen_mask).sum(dim=1)  # [b]
        sum_logp_chunks.append(sum_logp_gen)
        gen_mask_chunks.append(gen_mask)

        # Help allocator between microbatches
        del outputs, logits, labels, token_logp

    return torch.cat(sum_logp_chunks, dim=0), torch.cat(gen_mask_chunks, dim=0)

