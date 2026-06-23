"""Merge a PEFT LoRA adapter checkpoint into its base model for evaluation.

TRL checkpoints for Qwen3.5 are saved against the full multimodal module tree
(text stack under `model.language_model.layers`, plus `model.visual` adapters),
while AutoModelForCausalLM exposes the text stack as `model.layers`. Without
remapping, PeftModel.from_pretrained silently loads NOTHING (every key
"missing") and the "merged" model is just the base model. We remap text keys
and drop vision keys before merging.
"""

import argparse
import os
import shutil
import tempfile

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_adapter(adapter_dir: str, out_dir: str) -> None:
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    fixed = {
        k.replace(".model.language_model.layers.", ".model.layers."): v
        for k, v in sd.items()
        if ".model.visual." not in k
    }
    n_remapped = sum(1 for k in sd if ".model.language_model.layers." in k)
    n_dropped = len(sd) - sum(1 for k in sd if ".model.visual." not in k)
    print(f"adapter keys: {len(sd)} -> {len(fixed)} (remapped {n_remapped}, dropped {n_dropped} vision)")
    os.makedirs(out_dir, exist_ok=True)
    save_file(fixed, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(adapter_dir, "adapter_config.json"), os.path.join(out_dir, "adapter_config.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    base_keys = {n for n, _ in base.named_parameters()}

    with tempfile.TemporaryDirectory() as tmp:
        normalize_adapter(args.adapter, tmp)
        sd = load_file(os.path.join(tmp, "adapter_model.safetensors"))
        # Every adapter key must correspond to a real base module, else the load
        # silently no-ops and we ship an unmerged model again.
        unmatched = [
            k for k in sd
            if k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
            not in base_keys
        ]
        if unmatched:
            raise RuntimeError(f"{len(unmatched)} adapter keys missing in base model, e.g. {unmatched[:3]}")
        model = PeftModel.from_pretrained(base, tmp)

    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
