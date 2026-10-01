"""Lightweight boundary objectives for JBT-Lite v3.

JBT-Lite v2 showed a clear pattern on BUSI validation:
- the soft surface-alignment term remained beneficial;
- the absolute inner/outer ring BCE term was harmful.

v3 therefore keeps Surface Alignment and replaces the ring BCE with a
Signed Normal Margin objective inspired by the only clearly useful geometry
component in the older JBT ablation (signed displacement supervision).

The total training objective is

    L = L_base + lambda_edge * L_edge + lambda_normal * L_normal.

Both auxiliary terms operate directly on the existing segmentation logits.
They add no encoder pass, Monte-Carlo posterior, candidate generator, selector,
or inference-time module.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn.functional as F


def _to_bhw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be single-channel, got {tuple(x.shape)}")
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B,H,W] or [B,1,H,W], got {tuple(x.shape)}"
        )
    return x


def _soft_dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return x
    k = 2 * radius + 1
    return F.max_pool2d(x[:, None], kernel_size=k, stride=1, padding=radius)[:, 0]


def _soft_erode(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return x
    k = 2 * radius + 1
    return -F.max_pool2d(-x[:, None], kernel_size=k, stride=1, padding=radius)[:, 0]


def soft_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Differentiable morphological gradient in [0,1]."""
    radius = max(int(radius), 1)
    return (_soft_dilate(mask, radius) - _soft_erode(mask, radius)).clamp(0.0, 1.0)


def _masked_case_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num = (values * weights).flatten(1).sum(dim=1)
    den = weights.flatten(1).sum(dim=1)
    valid = den > eps
    mean = torch.where(valid, num / den.clamp_min(eps), torch.zeros_like(num))
    return mean, valid


def _gt_normals(y: torch.Tensor, smooth_radius: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit normals pointing from background toward foreground.

    A small average-pool before Sobel suppresses staircase normals caused by
    resizing binary masks to 224x224.
    """
    r = max(int(smooth_radius), 0)
    ys = y[:, None]
    if r > 0:
        k = 2 * r + 1
        ys = F.avg_pool2d(ys, kernel_size=k, stride=1, padding=r)

    kx = y.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    ky = y.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(ys, kx, padding=1)[:, 0]
    gy = F.conv2d(ys, ky, padding=1)[:, 0]
    mag = torch.sqrt(gx.square() + gy.square() + eps)
    nx = gx / mag.clamp_min(eps)
    ny = gy / mag.clamp_min(eps)
    return nx, ny, mag


def _sample_along_normals(
    value: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    delta_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a scalar map at +/- delta along local GT normal.

    The positive normal direction points toward foreground, so ``plus`` is the
    expected inside sample and ``minus`` is the expected outside sample.
    """
    b, h, w = value.shape
    yy = torch.linspace(-1.0, 1.0, h, device=value.device, dtype=value.dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=value.device, dtype=value.dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    base = torch.stack((gx, gy), dim=-1)[None].expand(b, -1, -1, -1)

    sx = 0.0 if w <= 1 else 2.0 * float(delta_px) / float(w - 1)
    sy = 0.0 if h <= 1 else 2.0 * float(delta_px) / float(h - 1)
    offset = torch.stack((nx * sx, ny * sy), dim=-1)
    grid_plus = base + offset
    grid_minus = base - offset

    x = value[:, None]
    plus = F.grid_sample(
        x, grid_plus, mode="bilinear", padding_mode="border", align_corners=True
    )[:, 0]
    minus = F.grid_sample(
        x, grid_minus, mode="bilinear", padding_mode="border", align_corners=True
    )[:, 0]

    in_bounds = (
        (grid_plus[..., 0].abs() <= 1.0)
        & (grid_plus[..., 1].abs() <= 1.0)
        & (grid_minus[..., 0].abs() <= 1.0)
        & (grid_minus[..., 1].abs() <= 1.0)
    )
    return plus, minus, in_bounds



def compute_edge_alignment_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    boundary_radius_px: int = 1,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pure soft Surface Alignment loss without normal-ray overhead.

    This is algebraically the same EDGE term used by the original 20-epoch
    JBT-Lite screen when ``RBAL_EDGE_MIX=1``.  Keeping it separate avoids
    Sobel/grid-sample work when the Signed Normal Margin branch is disabled.
    """
    z = _to_bhw(logits, name="logits")
    y = _to_bhw(target, name="target").to(device=z.device, dtype=z.dtype)
    y = (y > 0.5).to(dtype=z.dtype)
    p = torch.sigmoid(z)
    gt_edge = soft_boundary(y, radius=boundary_radius_px)
    pred_edge = soft_boundary(p, radius=boundary_radius_px)
    edge_inter = (pred_edge * gt_edge).flatten(1).sum(dim=1)
    edge_den = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
    edge_case = 1.0 - (2.0 * edge_inter + eps) / (edge_den + eps)
    edge_loss = edge_case.mean()
    return edge_loss, {"edge_loss": edge_loss.detach()}


def _distance_to_boundary(binary: torch.Tensor) -> torch.Tensor:
    """Detached Euclidean distance-to-boundary maps for a BHW binary tensor."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError(
            "Boundary and Hausdorff-DT losses require scipy. "
            "Install it in the training environment with `pip install scipy`."
        ) from exc

    arrays = []
    binary_np = binary.detach().to(device="cpu").numpy().astype(np.bool_)
    for mask in binary_np:
        # Sum of the inner and outer transforms is zero only on the discrete
        # interface and increases with Euclidean distance from that interface.
        dist = distance_transform_edt(mask) + distance_transform_edt(~mask)
        dist = np.maximum(dist - 1.0, 0.0).astype(np.float32, copy=False)
        arrays.append(dist)
    return torch.as_tensor(
        np.stack(arrays, axis=0), device=binary.device, dtype=torch.float32
    )


def compute_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Kervadec-style signed-distance Boundary Loss.

    The target distance field is detached; gradients flow only through the
    foreground probability.  Per-case normalization prevents lesion size from
    changing the effective auxiliary weight.
    """
    z = _to_bhw(logits, name="logits")
    y = (_to_bhw(target, name="target") > 0.5).to(z.dtype)
    p = torch.sigmoid(z)
    unsigned = _distance_to_boundary(y > 0.5).to(dtype=z.dtype)
    signed = torch.where(y > 0.5, -unsigned, unsigned)
    scale = signed.abs().flatten(1).amax(dim=1).clamp_min(eps)
    signed = signed / scale[:, None, None]
    case_loss = (p * signed).flatten(1).mean(dim=1)
    loss = case_loss.mean()
    return loss, {"boundary_loss": loss.detach()}


def compute_hausdorff_dt_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Distance-transform approximation to the symmetric Hausdorff error."""
    z = _to_bhw(logits, name="logits")
    y = (_to_bhw(target, name="target") > 0.5).to(z.dtype)
    p = torch.sigmoid(z)
    with torch.no_grad():
        d_gt = _distance_to_boundary(y > 0.5).to(dtype=z.dtype)
        d_pred = _distance_to_boundary(p >= 0.5).to(dtype=z.dtype)
        h = float(z.shape[-2])
        w = float(z.shape[-1])
        normalizer = max((h * h + w * w) ** 0.5, 1.0)
        d_gt = (d_gt / normalizer).pow(float(alpha))
        d_pred = (d_pred / normalizer).pow(float(alpha))
    case_loss = ((p - y).square() * (d_gt + d_pred)).flatten(1).mean(dim=1)
    loss = case_loss.mean()
    return loss, {"hausdorff_dt_loss": loss.detach()}


def compute_active_contour_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    length_weight: float = 1.0,
    region_weight: float = 1.0,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Normalized Chan-Vese active-contour loss for binary lesions."""
    z = _to_bhw(logits, name="logits")
    y = (_to_bhw(target, name="target") > 0.5).to(z.dtype)
    p = torch.sigmoid(z)
    dx = p[:, :, 1:] - p[:, :, :-1]
    dy = p[:, 1:, :] - p[:, :-1, :]
    dx = dx[:, :-1, :]
    dy = dy[:, :, :-1]
    length = torch.sqrt(dx.square() + dy.square() + eps).flatten(1).mean(dim=1)
    region_in = (p * (1.0 - y)).flatten(1).mean(dim=1)
    region_out = ((1.0 - p) * y).flatten(1).mean(dim=1)
    region = region_in + region_out
    case_loss = float(length_weight) * length + float(region_weight) * region
    loss = case_loss.mean()
    return loss, {
        "active_contour_loss": loss.detach(),
        "active_contour_length": length.mean().detach(),
        "active_contour_region": region.mean().detach(),
    }


def compute_normal_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    boundary_radius_px: int = 1,
    normal_delta_px: float = 1.5,
    normal_margin: float = 1.0,
    normal_smooth_radius_px: int = 1,
    normal_min_grad: float = 0.03,
    normal_orientation_fg: float = 0.65,
    normal_orientation_bg: float = 0.35,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Signed Normal Margin only; kept for legacy diagnostics, not v5 FULL."""
    z = _to_bhw(logits, name="logits")
    y = _to_bhw(target, name="target").to(device=z.device, dtype=z.dtype)
    y = (y > 0.5).to(dtype=z.dtype)
    gt_edge = soft_boundary(y, radius=boundary_radius_px)
    nx, ny, grad_mag = _gt_normals(y, normal_smooth_radius_px, eps)
    z_in, z_out, in_bounds = _sample_along_normals(z, nx, ny, normal_delta_px)
    y_in, y_out, _ = _sample_along_normals(y, nx, ny, normal_delta_px)
    reliable = (
        (gt_edge > 0.05)
        & (grad_mag >= float(max(normal_min_grad, 0.0)))
        & in_bounds
        & (y_in >= float(normal_orientation_fg))
        & (y_out <= float(normal_orientation_bg))
    )
    signed_gap = z_in - z_out
    violation = F.relu(float(normal_margin) - signed_gap)
    pair_penalty = violation.square()
    normal_case, normal_valid = _masked_case_mean(
        pair_penalty, reliable.to(z.dtype), eps=eps
    )
    normal_loss = torch.where(
        normal_valid, normal_case, torch.zeros_like(normal_case)
    ).mean()
    diagnostics = {
        "normal_loss": normal_loss.detach(),
        "normal_valid_fraction": reliable.to(z.dtype).mean().detach(),
        "normal_margin_violation": violation[reliable].mean().detach()
        if bool(reliable.any()) else z.detach().new_zeros(()),
        "normal_signed_gap": signed_gap[reliable].mean().detach()
        if bool(reliable.any()) else z.detach().new_zeros(()),
    }
    return normal_loss, diagnostics


def compute_rbal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    boundary_radius_px: int = 1,
    normal_delta_px: float = 1.5,
    normal_margin: float = 1.0,
    normal_smooth_radius_px: int = 1,
    normal_min_grad: float = 0.03,
    normal_orientation_fg: float = 0.65,
    normal_orientation_bg: float = 0.35,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return Surface Alignment and Signed Normal Margin losses.

    ``L_edge`` is a soft boundary Dice loss.  It rewards overlap of the current
    soft contour with the GT contour and was the only lightweight term that
    improved both DSC and true2d NSD in the previous screen.

    ``L_normal`` is a sparse, zero-after-satisfaction pairwise constraint.  At
    reliable GT-boundary pixels it samples logits just inside and just outside
    the contour along the GT normal and requires

        z_inside - z_outside >= margin.

    Unlike the failed ring BCE, it never pushes a fixed absolute probability at
    every boundary-ring pixel.  Once the local ordering margin is satisfied its
    gradient is exactly zero, which protects already-correct geometry.
    """
    edge_loss, edge_diag = compute_edge_alignment_loss(
        logits, target, boundary_radius_px=boundary_radius_px, eps=eps
    )
    normal_loss, normal_diag = compute_normal_margin_loss(
        logits, target,
        boundary_radius_px=boundary_radius_px,
        normal_delta_px=normal_delta_px,
        normal_margin=normal_margin,
        normal_smooth_radius_px=normal_smooth_radius_px,
        normal_min_grad=normal_min_grad,
        normal_orientation_fg=normal_orientation_fg,
        normal_orientation_bg=normal_orientation_bg,
        eps=eps,
    )
    diagnostics = {**edge_diag, **normal_diag}
    return edge_loss, normal_loss, diagnostics
