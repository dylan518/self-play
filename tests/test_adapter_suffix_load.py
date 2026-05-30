import torch
from torch import nn

from grpo_math.trl.train_grpo_trl import (
    _adapter_tensors_by_suffix,
    _load_adapter_weights_into_peft_model,
    _remap_adapter_state_dict_keys,
    _verify_adapter_weights_loaded,
)


class _FakeParam(nn.Parameter):
    pass


class _FakePeftModel(nn.Module):
    def __init__(self, suffix_to_shape: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        for suffix, shape in suffix_to_shape.items():
            full_name = f"base_model.model.model.{suffix}"
            self.register_parameter(full_name, nn.Parameter(torch.zeros(shape)))


def test_suffix_copy_loads_linear_attn_lora_b(tmp_path) -> None:
    suffix = "layers.0.linear_attn.in_proj_a.lora_B.default.weight"
    file_key = (
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a."
        "lora_B.weight"
    )
    shape = (4, 8)
    file_tensor = torch.randn(shape)
    ckpt = tmp_path / "adapter_ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3.5-9B", "peft_type": "LORA"}',
        encoding="utf-8",
    )
    from safetensors.torch import save_file

    save_file({file_key: file_tensor}, str(ckpt / "adapter_model.safetensors"))

    model = _FakePeftModel({suffix: shape})
    stats = _load_adapter_weights_into_peft_model(model, ckpt)
    assert stats["adapter_tensors_copied"] == 1
    param = dict(model.named_parameters())[f"base_model.model.model.{suffix}"]
    assert torch.allclose(param.detach().cpu(), file_tensor, atol=1e-5)


def test_verify_requires_all_nonzero_file_tensors(tmp_path) -> None:
    suffix_a = "layers.0.linear_attn.in_proj_a.lora_A.default.weight"
    suffix_b = "layers.0.linear_attn.in_proj_a.lora_B.default.weight"
    shape = (2, 2)
    ckpt = tmp_path / "adapter_ckpt2"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3.5-9B"}',
        encoding="utf-8",
    )
    from safetensors.torch import save_file

    save_file(
        {
            f"base_model.model.model.{suffix_a}": torch.ones(shape),
            f"base_model.model.model.{suffix_b}": torch.ones(shape) * 2.0,
        },
        str(ckpt / "adapter_model.safetensors"),
    )

    model = _FakePeftModel({suffix_a: shape, suffix_b: shape})
    # Only load A; leave B zeros -> verify must fail.
    model_param_a = dict(model.named_parameters())[f"base_model.model.model.{suffix_a}"]
    model_param_a.data.copy_(torch.ones(shape))
    try:
        _verify_adapter_weights_loaded(model, ckpt)
        raised = False
    except RuntimeError:
        raised = True
    assert raised

    _load_adapter_weights_into_peft_model(model, ckpt)
    summary = _verify_adapter_weights_loaded(model, ckpt)
    assert summary["adapter_verify_ok"] is True
    assert summary["adapter_verify_matched"] == summary["adapter_verify_checked"]


def test_remap_and_suffix_index_agree() -> None:
    key = (
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a."
        "lora_A.weight"
    )
    tensor = torch.randn(2, 4)
    by_suffix = _adapter_tensors_by_suffix(_remap_adapter_state_dict_keys({key: tensor}))
    assert "layers.0.linear_attn.in_proj_a.lora_A.default.weight" in by_suffix
