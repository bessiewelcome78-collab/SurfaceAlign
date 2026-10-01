"""Structural gradient-isolation helpers for two-branch training."""
from __future__ import annotations

from typing import Tuple

import torch


def backward_named_prefix_only(
    model: torch.nn.Module,
    objective: torch.Tensor,
    prefix: str,
) -> Tuple[int, torch.Tensor]:
    """Accumulate ``objective`` gradients only for trainable parameters in ``prefix``.

    Unlike ``objective.backward()``, the explicit input list prevents autograd
    from writing gradients to any other leaf, even if a future refactor
    accidentally reconnects the computational graphs.
    """
    named_parameters = []
    for name, parameter in model.named_parameters():
        clean_name = name[len("module."):] if name.startswith("module.") else name
        if parameter.requires_grad and clean_name.startswith(prefix):
            named_parameters.append((name, parameter))
    if not named_parameters:
        raise RuntimeError(f"gradient isolation found no active parameters under {prefix!r}")
    if not isinstance(objective, torch.Tensor) or not objective.requires_grad:
        raise RuntimeError("isolated objective must be a differentiable tensor")

    gradients = torch.autograd.grad(
        objective,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    used = 0
    norm_sq = objective.detach().new_zeros(())
    for (_, parameter), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        if parameter.grad is None:
            parameter.grad = detached.clone()
        else:
            parameter.grad.add_(detached)
        norm_sq = norm_sq + detached.float().square().sum()
        used += 1
    if used <= 0:
        raise RuntimeError(
            f"isolated objective produced no gradients under {prefix!r}"
        )
    return used, norm_sq.sqrt()
