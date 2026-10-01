"""Shared continuous geometry utilities for SemLT SDF operator-matched warp.

All geometry is target/conditioning only.  The Base logits and hard masks passed
here are detached by the caller.  We intentionally use an exact Euclidean
distance transform (SciPy) rather than a learned or tunable approximation:
there is no extra trainable parameter and the zero level is located halfway
between opposite-label pixel centres.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
except Exception as exc:  # pragma: no cover - fail closed only on missing dependency
    distance_transform_edt = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def _require_scipy() -> None:
    if distance_transform_edt is None:
        raise RuntimeError(
            "SemLT SDF operator-matched warp requires scipy.ndimage.distance_transform_edt. "
            f"Original import error: {_SCIP_IMPORT_ERROR!r}"
        )


def _as_bhw_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    return mask.bool()


def binary_signed_distance(mask: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Half-pixel centred Euclidean signed-distance field.

    Foreground is positive, background negative.  For a straight binary step,
    the foreground/background boundary pixel centres are +0.5/-0.5, so the
    zero level lies exactly halfway between them instead of forming a two-pixel
    morphological shell.
    """
    _require_scipy()
    mask_bhw = _as_bhw_bool(mask)
    device = mask_bhw.device
    out_dtype = dtype or torch.float32
    arr = mask_bhw.detach().to("cpu").numpy().astype(np.bool_, copy=False)
    b, h, w = arr.shape
    result = np.empty((b, h, w), dtype=np.float32)
    far = float(max(h, w) + 1)
    for bi, m in enumerate(arr):
        if not m.any():
            result[bi].fill(-far)
            continue
        if m.all():
            result[bi].fill(far)
            continue
        inside = distance_transform_edt(m).astype(np.float32, copy=False)
        outside = distance_transform_edt(~m).astype(np.float32, copy=False)
        result[bi] = np.where(m, inside - 0.5, -(outside - 0.5))
    return torch.from_numpy(result).to(device=device, dtype=out_dtype)


def foreground_contour_owner(mask: torch.Tensor) -> torch.Tensor:
    """Single-sided one-pixel contour owner: FG pixels adjacent to background."""
    m = _as_bhw_bool(mask)
    bg = (~m)[:, None].float()
    adjacent_bg = F.max_pool2d(bg, 3, stride=1, padding=1)[:, 0] > 0.5
    return m & adjacent_bg


def sdf_normal(sdf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit normal of a positive-inside SDF, pointing into foreground."""
    if sdf.ndim == 3:
        phi = sdf[:, None]
    elif sdf.ndim == 4 and sdf.shape[1] == 1:
        phi = sdf
    else:
        raise ValueError(f"Expected SDF [B,H,W] or [B,1,H,W], got {tuple(sdf.shape)}")
    kx = phi.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    )[:, None] / 8.0
    ky = phi.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    )[:, None] / 8.0
    gx = F.conv2d(phi, kx, padding=1)
    gy = F.conv2d(phi, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
    valid = mag > 1.0e-5
    nx = torch.where(valid, gx / mag, torch.zeros_like(gx))
    ny = torch.where(valid, gy / mag, torch.zeros_like(gy))
    return torch.cat([nx, ny], dim=1), valid[:, 0]


def nearest_owner_map(owner: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Euclidean distance and nearest owner flat index for every pixel."""
    _require_scipy()
    owner_bhw = _as_bhw_bool(owner)
    device = owner_bhw.device
    arr = owner_bhw.detach().to("cpu").numpy().astype(np.bool_, copy=False)
    b, h, w = arr.shape
    distances = np.full((b, h, w), np.inf, dtype=np.float32)
    flat_index = np.zeros((b, h, w), dtype=np.int64)
    valid_case = np.zeros((b,), dtype=np.bool_)
    for bi, own in enumerate(arr):
        if not own.any():
            continue
        valid_case[bi] = True
        dist, indices = distance_transform_edt(~own, return_indices=True)
        distances[bi] = dist.astype(np.float32, copy=False)
        yy, xx = indices[0].astype(np.int64, copy=False), indices[1].astype(np.int64, copy=False)
        flat_index[bi] = yy * w + xx
    return (
        torch.from_numpy(distances).to(device=device, dtype=dtype),
        torch.from_numpy(flat_index).to(device=device, dtype=torch.long),
        torch.from_numpy(valid_case).to(device=device),
    )


def gather_owner_field(field: torch.Tensor, owner_flat_index: torch.Tensor, valid_case: torch.Tensor) -> torch.Tensor:
    """Extend a field by nearest-contour ownership; gradients return to owner values."""
    if field.ndim != 4:
        raise ValueError(f"field must be [B,C,H,W], got {tuple(field.shape)}")
    b, c, h, w = field.shape
    if tuple(owner_flat_index.shape) != (b, h, w):
        raise ValueError("owner_flat_index shape mismatch")
    flat = field.reshape(b, c, h * w)
    idx = owner_flat_index.reshape(b, 1, h * w).expand(-1, c, -1)
    extended = torch.gather(flat, 2, idx).reshape(b, c, h, w)
    return extended * valid_case[:, None, None, None].to(extended)


def sparse_grid_sample(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinear sample one image field at sparse N x K pixel-centre coordinates.

    field: [1,C,H,W], x/y: [N,K] in pixel coordinates. Returns [C,N,K].
    """
    if field.ndim != 4 or field.shape[0] != 1:
        raise ValueError("sparse_grid_sample expects field [1,C,H,W]")
    _, _, h, w = field.shape
    gx = 2.0 * x / float(max(w - 1, 1)) - 1.0 if w > 1 else torch.zeros_like(x)
    gy = 2.0 * y / float(max(h - 1, 1)) - 1.0 if h > 1 else torch.zeros_like(y)
    grid = torch.stack([gx, gy], dim=-1)[None]  # [1,N,K,2]
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0]


def operator_matched_normal_ray_target(
    base_owner: torch.Tensor,
    base_sdf: torch.Tensor,
    base_normal: torch.Tensor,
    gt_mask: torch.Tensor,
    radius_px: int,
    censored_endpoint_action: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed inverse-warp target that exactly matches the physical normal-ray operator.

    We first place each one-sided Base owner on the Base zero level using its SDF,
    then search the GT SDF along the *same Base normal* over integer samples in
    [-r,r].  A GT zero crossing is linearly interpolated to sub-pixel precision.
    If the true crossing is outside the one-step range, the legacy route uses
    the sampled point with the smallest absolute GT SDF.  With
    ``censored_endpoint_action=True`` the target is instead the signed endpoint
    action ``+/-r``: the exact correspondence is censored, but the maximum safe
    in-range action and its direction remain observable.  This prevents the
    difficult radius-clipped owners from being silently removed from action
    supervision.

    Returns
    -------
    target_sample_offset : [B,H,W]
    valid_owner          : [B,H,W] bool
    crossing_reachable   : [B,H,W] bool
    clipped              : [B,H,W] bool
    contour_motion_abs   : [B,H,W]
    gt_normal_alignment  : [B,H,W], valid on crossing_reachable
    """
    owner = _as_bhw_bool(base_owner)
    gt = _as_bhw_bool(gt_mask)
    if base_sdf.ndim == 4:
        base_sdf = base_sdf[:, 0]
    if base_normal.ndim != 4 or base_normal.shape[1] != 2:
        raise ValueError("base_normal must be [B,2,H,W]")
    r = int(radius_px)
    if r <= 0:
        raise ValueError("radius_px must be positive")
    b, h, w = owner.shape
    dtype, device = base_normal.dtype, base_normal.device
    gt_sdf = binary_signed_distance(gt, dtype=dtype)
    gt_normal, gt_normal_valid = sdf_normal(gt_sdf)

    target = torch.zeros((b, h, w), device=device, dtype=dtype)
    valid_owner = torch.zeros_like(owner)
    reachable = torch.zeros_like(owner)
    clipped = torch.zeros_like(owner)
    motion_abs = torch.zeros((b, h, w), device=device, dtype=dtype)
    alignment = torch.zeros((b, h, w), device=device, dtype=dtype)
    t_values = torch.arange(-r, r + 1, device=device, dtype=dtype)

    with torch.no_grad():
        for bi in range(b):
            coords = torch.nonzero(owner[bi], as_tuple=False)
            if coords.numel() == 0:
                continue
            yy, xx = coords[:, 0], coords[:, 1]
            nx = base_normal[bi, 0, yy, xx]
            ny = base_normal[bi, 1, yy, xx]
            nmag = torch.sqrt(nx.square() + ny.square())
            valid = nmag > 0.5
            if not bool(valid.any()):
                continue
            yy, xx, nx, ny = yy[valid], xx[valid], nx[valid], ny[valid]
            coords = torch.stack([yy, xx], dim=1)
            phi_b = base_sdf[bi, yy, xx]
            # Move the one-sided owner pixel centre onto the Base zero level.
            x0 = xx.to(dtype) - phi_b * nx
            y0 = yy.to(dtype) - phi_b * ny
            n = yy.numel()

            # Empty/full GT has no zero level.  The operator can still take the
            # maximally safe one-step direction: shrink toward FG interior for
            # empty GT, expand outward for full GT.
            gt_any = bool(gt[bi].any())
            gt_all = bool(gt[bi].all())
            if not gt_any or gt_all:
                sample_target = torch.full((n,), -float(r) if not gt_any else float(r), device=device, dtype=dtype)
                valid_owner[bi, yy, xx] = True
                clipped[bi, yy, xx] = True
                target[bi, yy, xx] = sample_target
                motion_abs[bi, yy, xx] = float(r)
                continue

            xs = x0[:, None] + nx[:, None] * t_values[None]
            ys = y0[:, None] + ny[:, None] * t_values[None]
            phi = sparse_grid_sample(gt_sdf[bi:bi+1, None], xs, ys)[0]  # [N,T]

            left, right = phi[:, :-1], phi[:, 1:]
            cross = (left == 0) | (right == 0) | ((left < 0) != (right < 0))
            denom = right - left
            alpha = torch.where(denom.abs() > 1.0e-8, -left / denom, torch.zeros_like(left)).clamp(0.0, 1.0)
            roots = t_values[:-1][None] + alpha
            root_cost = torch.where(cross, roots.abs(), torch.full_like(roots, float("inf")))
            best_cross_cost, best_cross_idx = root_cost.min(dim=1)
            has_cross = torch.isfinite(best_cross_cost)
            root = roots.gather(1, best_cross_idx[:, None]).squeeze(1)

            # If the crossing lies beyond the one-step range, choose the sampled
            # point within [-r,r] nearest to the GT zero level.  This yields the
            # correct clipped direction instead of censoring the hard example.
            best_sample_idx = phi.abs().argmin(dim=1)
            best_sample_t = t_values[best_sample_idx]
            if censored_endpoint_action:
                # The latent crossing lies outside [-r,r].  Its exact distance
                # is unknown, so do not pretend that an interior closest sample
                # is an exact correspondence.  Retain only the identifiable
                # direction and request the largest realizable one-step action.
                # A rare zero-direction tie is resolved by the endpoint with
                # smaller |SDF| (negative on an exact tie, deterministically).
                fallback_sign = torch.where(
                    phi[:, -1].abs() < phi[:, 0].abs(),
                    torch.ones_like(best_sample_t),
                    -torch.ones_like(best_sample_t),
                )
                clipped_sign = torch.where(
                    best_sample_t.abs() > 1.0e-6,
                    torch.sign(best_sample_t),
                    fallback_sign,
                )
                clipped_motion_t = clipped_sign * float(r)
                motion_t = torch.where(has_cross, root, clipped_motion_t)
            else:
                motion_t = torch.where(has_cross, root, best_sample_t)
            sample_target = -motion_t  # inverse-warp convention

            target[bi, yy, xx] = sample_target
            valid_owner[bi, yy, xx] = True
            reachable[bi, yy, xx] = has_cross
            clipped[bi, yy, xx] = ~has_cross
            motion_abs[bi, yy, xx] = motion_t.abs()

            # Surface-normal agreement is now measured between continuous SDF
            # normals at the actual normal-ray crossing, not distance=0 shells.
            if bool(has_cross.any()):
                hx = x0[has_cross] + nx[has_cross] * motion_t[has_cross]
                hy = y0[has_cross] + ny[has_cross] * motion_t[has_cross]
                grid_x, grid_y = hx[:, None], hy[:, None]
                gtn = sparse_grid_sample(gt_normal[bi:bi+1], grid_x, grid_y)[:, :, 0].T  # [Nc,2]
                gvalid = sparse_grid_sample(gt_normal_valid[bi:bi+1, None].to(dtype), grid_x, grid_y)[0, :, 0] > 0.5
                bn = torch.stack([nx[has_cross], ny[has_cross]], dim=1)
                agree = (bn * gtn).sum(1).abs().clamp(0.0, 1.0)
                agree = torch.where(gvalid, agree, torch.zeros_like(agree))
                cy, cx = yy[has_cross], xx[has_cross]
                alignment[bi, cy, cx] = agree

    return target, valid_owner, reachable, clipped, motion_abs, alignment
