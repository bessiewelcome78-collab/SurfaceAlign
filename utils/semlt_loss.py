"""Objective for the M1-only Semantic Logit Transport model."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


EPS = 1.0e-5


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _target4(masks: torch.Tensor, hw) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
    masks = (masks > 0.5).to(dtype=torch.float32)
    if tuple(masks.shape[-2:]) != tuple(hw):
        masks = F.interpolate(masks, size=hw, mode="nearest")
    return masks


def _soft_dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3)) + EPS
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS
    return (1.0 - numerator / denominator).mean()


def _soft_boundary(probability: torch.Tensor) -> torch.Tensor:
    maximum = F.max_pool2d(probability, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, 3, stride=1, padding=1)
    return (maximum - minimum).clamp(0.0, 1.0)


def _flow_tv(flow: torch.Tensor) -> torch.Tensor:
    horizontal = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    vertical = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return 0.5 * (horizontal.abs().mean() + vertical.abs().mean())


def compute_semlt_loss(
    cfg,
    candidate_logits: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train the deployed SemLT output with topology-preserving regularizers.

    L = L_seg + lambda_b L_boundary
        + ramp * (lambda_mag L_mag + lambda_tv L_tv + lambda_fold L_fold).
    """
    del candidate_logits
    m1 = _cfg_get(cfg, "M1", None)
    train = _cfg_get(cfg, "TRAIN", None)
    logits = aux["semlt_final_logits"]
    probability = aux["semlt_final_probs"]
    flow = aux["semlt_flow_px"]
    determinant = aux["semlt_jacobian_determinant"]
    target = _target4(masks, probability.shape[-2:]).to(probability)

    ce_weight = float(_cfg_get(train, "CE_WEIGHT", 0.5))
    dice_weight = float(_cfg_get(train, "DICE_WEIGHT", 0.5))
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float()).to(probability)
    dice = _soft_dice_loss(probability, target)
    segmentation = ce_weight * bce + dice_weight * dice

    pred_boundary = _soft_boundary(probability)
    target_boundary = _soft_boundary(target)
    boundary = _soft_dice_loss(pred_boundary, target_boundary)
    magnitude = torch.linalg.vector_norm(flow, dim=1).mean()
    smoothness = _flow_tv(flow)
    fold_margin = float(_cfg_get(m1, "SEMLT_FOLD_MARGIN", 0.0))
    folding = F.relu(fold_margin - determinant).mean()

    boundary_weight = float(_cfg_get(m1, "SEMLT_BOUNDARY_WEIGHT", 0.05))
    magnitude_weight = float(_cfg_get(m1, "SEMLT_MAGNITUDE_WEIGHT", 1.0e-3))
    smoothness_weight = float(_cfg_get(m1, "SEMLT_TV_WEIGHT", 1.0e-2))
    folding_weight = float(_cfg_get(m1, "SEMLT_FOLD_WEIGHT", 1.0e-2))
    ramp_epochs = max(1, int(_cfg_get(m1, "SEMLT_REGULARIZER_RAMP_EPOCHS", 10)))
    ramp = min(max((int(epoch) + 1) / float(ramp_epochs), 0.0), 1.0)
    regularization = (
        magnitude_weight * magnitude
        + smoothness_weight * smoothness
        + folding_weight * folding
    )
    objective = segmentation + boundary_weight * boundary + float(ramp) * regularization

    zero = objective.detach().new_zeros(())
    diagnostics = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation.detach(),
        "mhcs_ce_loss": bce.detach(),
        "mhcs_dice_loss": dice.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": zero,
        "geotopo_total_loss": objective.detach(),
        "geotopo_geometry_loss": segmentation.detach(),
        "geotopo_reconstruction_loss": zero,
        "semlt_total_loss": objective.detach(),
        "semlt_segmentation_loss": segmentation.detach(),
        "semlt_bce_loss": bce.detach(),
        "semlt_dice_loss": dice.detach(),
        "semlt_boundary_loss": boundary.detach(),
        "semlt_flow_magnitude_loss": magnitude.detach(),
        "semlt_flow_tv_loss": smoothness.detach(),
        "semlt_flow_fold_loss": folding.detach(),
        "semlt_regularizer_ramp": objective.detach().new_tensor(float(ramp)),
        "semlt_flow_scale_px": aux["semlt_flow_scale_px"].detach().mean(),
        "semlt_folding_fraction": aux["semlt_folding_fraction"].detach().mean(),
        "semlt_has_m2": zero,
    }
    return objective, diagnostics
