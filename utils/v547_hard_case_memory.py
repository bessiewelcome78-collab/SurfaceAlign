"""Dataset-level hard-case memory for V547 Base tail robustness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import torch



def _normalise_names(names: object, batch_size: int) -> List[str]:
    if names is None:
        return [f"__unknown_{index}" for index in range(batch_size)]
    if isinstance(names, str):
        values = [names]
    elif isinstance(names, torch.Tensor):
        values = [str(value.item()) for value in names.reshape(-1)]
    else:
        try:
            values = [str(value) for value in list(names)]
        except TypeError:
            values = [str(names)]
    if len(values) != batch_size:
        values = (values + [f"__unknown_{index}" for index in range(batch_size)])[:batch_size]
    return values


@dataclass
class V547HardCaseMemory:
    momentum: float = 0.90
    max_boost: float = 1.0
    warmup_observations: int = 1
    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.momentum = min(max(float(self.momentum), 0.0), 0.9999)
        self.max_boost = max(float(self.max_boost), 0.0)
        self.warmup_observations = max(int(self.warmup_observations), 1)

    def weights(
        self,
        names: object,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        keys = _normalise_names(names, batch_size)
        eligible = [
            (key, value)
            for key, value in self.values.items()
            if self.counts.get(key, 0) >= self.warmup_observations
        ]
        if len(eligible) < 2 or self.max_boost <= 0.0:
            return torch.ones(batch_size, device=device, dtype=dtype)
        ordered = sorted(value for _, value in eligible)
        denominator = max(len(ordered) - 1, 1)
        raw = []
        for key in keys:
            if self.counts.get(key, 0) < self.warmup_observations:
                rank = 0.0
            else:
                value = self.values[key]
                # Empirical percentile rank is deterministic and invariant to the
                # absolute scale drift of the training loss.
                lower = sum(1 for item in ordered if item < value)
                equal = sum(1 for item in ordered if item == value)
                rank = (lower + 0.5 * max(equal - 1, 0)) / denominator
            raw.append(1.0 + self.max_boost * rank)
        weight = torch.tensor(raw, device=device, dtype=dtype)
        return weight / weight.mean().clamp_min(1.0e-6)

    def update(self, names: object, difficulty: torch.Tensor) -> None:
        values = difficulty.detach().float().cpu().reshape(-1)
        keys = _normalise_names(names, int(values.numel()))
        for key, value_tensor in zip(keys, values):
            value = float(value_tensor.item())
            if not bool(torch.isfinite(value_tensor).item()):
                continue
            old = self.values.get(key, value)
            count = self.counts.get(key, 0)
            momentum = self.momentum if count > 0 else 0.0
            self.values[key] = momentum * old + (1.0 - momentum) * value
            self.counts[key] = count + 1

    def state_dict(self) -> Dict[str, object]:
        return {
            "momentum": self.momentum,
            "max_boost": self.max_boost,
            "warmup_observations": self.warmup_observations,
            "values": dict(self.values),
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.momentum = float(state.get("momentum", self.momentum))
        self.max_boost = float(state.get("max_boost", self.max_boost))
        self.warmup_observations = int(
            state.get("warmup_observations", self.warmup_observations)
        )
        self.values = {str(k): float(v) for k, v in dict(state.get("values", {})).items()}
        self.counts = {str(k): int(v) for k, v in dict(state.get("counts", {})).items()}

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {"seen": 0.0, "mean": 0.0, "max": 0.0}
        tensor = torch.tensor(list(self.values.values()), dtype=torch.float32)
        return {
            "seen": float(len(self.values)),
            "mean": float(tensor.mean().item()),
            "max": float(tensor.max().item()),
        }
