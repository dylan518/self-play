from grpo_math.trl.train_grpo_trl import _ensure_dpo_policy_adapter_trainable


class _Param:
    def __init__(self) -> None:
        self.requires_grad = False


class _FakeModel:
    def __init__(self) -> None:
        self.default_a = _Param()
        self.default_b = _Param()
        self.ref_a = _Param()
        self.active = ""

    def train(self) -> None:
        return None

    def named_parameters(self):
        yield "base_model.model.layers.0.lora_A.default.weight", self.default_a
        yield "base_model.model.layers.0.lora_B.default.weight", self.default_b
        yield "base_model.model.layers.0.lora_A.ref.weight", self.ref_a

    def set_adapter(self, name: str) -> None:
        self.active = name


def test_ensure_dpo_policy_adapter_trainable_unfreezes_default_only() -> None:
    model = _FakeModel()
    stats = _ensure_dpo_policy_adapter_trainable(model)

    assert stats["default"]["lora_tensors"] == 2
    assert stats["default"]["lora_tensors_requires_grad"] == 2
    assert stats["ref"]["lora_tensors"] == 1
    assert stats["ref"]["lora_tensors_requires_grad"] == 0
    assert model.default_a.requires_grad is True
    assert model.ref_a.requires_grad is False
    assert model.active == "default"
