from __future__ import annotations

from typing import MutableMapping
import torch


def stable_m2_effective_weight(
    *,
    ratio: float,
    base_magnitude: torch.Tensor,
    objective_magnitude: torch.Tensor,
    maximum_weight: float,
    state: MutableMapping[str, float],
    decay: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable Base-relative M2 weight and routing diagnostics.

    The old route divided by the current mini-batch objective. At batch=2,
    near-zero batches produced raw weights above 1e5 even though a later clamp
    hid the instability. This function normalizes by a detached EMA and uses
    the existing maximum weight to derive a non-arbitrary denominator floor.
    """
    ratio = max(float(ratio), 0.0)
    maximum_weight = max(float(maximum_weight), 0.0)
    decay = min(max(float(decay), 0.0), 0.999999)
    eps = 1.0e-8

    current = max(float(objective_magnitude.detach().cpu()), eps)
    previous = state.get("ema_objective_magnitude")
    ema = current if previous is None else decay * float(previous) + (1.0 - decay) * current
    state["ema_objective_magnitude"] = float(ema)

    ema_tensor = objective_magnitude.new_tensor(ema).clamp_min(eps)
    raw_uncapped = ratio * base_magnitude / objective_magnitude.clamp_min(eps)

    # The floor is exactly the denominator implied by the pre-existing max
    # effective weight, so no new arbitrary loss scale is introduced.
    if maximum_weight > 0.0 and ratio > 0.0:
        denominator_floor = ratio * base_magnitude / maximum_weight
        stable_denominator = torch.maximum(ema_tensor, denominator_floor)
    else:
        stable_denominator = ema_tensor

    stable_uncapped = ratio * base_magnitude / stable_denominator.clamp_min(eps)
    weight = (
        stable_uncapped.clamp(max=maximum_weight)
        if maximum_weight > 0.0
        else stable_uncapped
    )
    state["last_effective_weight"] = float(weight.detach().cpu())
    return weight, raw_uncapped.detach(), stable_uncapped.detach(), ema_tensor.detach()
