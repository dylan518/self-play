import torch

from grpo_math.trl.train_grpo_trl import _remap_adapter_state_dict_keys


def test_remap_adapter_state_dict_keys_qwen35_style() -> None:
    key = (
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a."
        "lora_A.weight"
    )
    state = {key: torch.zeros(2, 2)}
    remapped = _remap_adapter_state_dict_keys(state)
    assert (
        "base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.default.weight"
        in remapped
    )
    assert "language_model" not in next(iter(remapped.keys()))
