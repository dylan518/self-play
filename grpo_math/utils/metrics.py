from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RollingMean:
    total: float = 0.0
    count: int = 0

    def update(self, x: float, n: int = 1) -> None:
        self.total += float(x) * int(n)
        self.count += int(n)

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def pass_at_k(per_prompt_rewards: List[List[float]], k: int) -> float:
    """per_prompt_rewards: list of length B; each inner list is length K."""
    if not per_prompt_rewards:
        return 0.0
    good = 0
    for rs in per_prompt_rewards:
        kk = min(k, len(rs))
        if any(r > 0 for r in rs[:kk]):
            good += 1
    return good / len(per_prompt_rewards)


def to_scalar_dict(d: Dict[str, float], prefix: Optional[str] = None) -> Dict[str, float]:
    if prefix:
        return {f"{prefix}/{k}": float(v) for k, v in d.items()}
    return {k: float(v) for k, v in d.items()}

