"""Merge a TRL Qwen3.5 LoRA adapter into the FULL multimodal model so vLLM can
serve it (vLLM requires the multimodal Qwen3_5Config; AutoModelForCausalLM saves a
text-only config it rejects). The adapter's keys already target the multimodal tree
(base_model.model.model.language_model.layers...), so they match the full model
directly — NO remap needed (unlike scripts/merge_lora_adapter.py which targets the
text-only AutoModelForCausalLM tree).

Usage: python scripts/merge_lora_full.py --base Qwen/Qwen3.5-9B --adapter <ckpt> --out <dir>
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    before = sum(int((base.state_dict()[k] != 0).any()) for k in list(base.state_dict())[:1])  # touch
    model = PeftModel.from_pretrained(base, args.adapter)
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    assert n_lora > 0, "no LoRA params matched the base model tree — keys mismatch"
    print(f"matched {n_lora} LoRA params against full multimodal model")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    # processor/tokenizer for serving
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    except Exception as e:
        print("processor save skipped:", e)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged {args.adapter} into FULL {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
