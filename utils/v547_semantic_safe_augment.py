"""Geometry-consistent tensor augmentation for the V547 formal protocol."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def apply_v547_semantic_safe_augmentation(
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    probability: float = 0.75,
    rotation_degrees: float = 10.0,
    scale_min: float = 0.97,
    scale_max: float = 1.03,
    translate_fraction: float = 0.02,
    contrast_min: float = 0.95,
    contrast_max: float = 1.05,
    noise_std: float = 0.01,
    hr_images: torch.Tensor | None = None,
    hr_masks: torch.Tensor | None = None,
) -> tuple:
    """Apply one paired affine grid and image-only photometric perturbation.

    The function is independent of the dataset loader, so the formal config can
    be contract-tested.  Masks are always returned binary and in their original
    [B,H,W] or [B,1,H,W] layout.
    """
    if images.ndim != 4:
        raise ValueError(
            f"V547 augmentation expects image [B,C,H,W], got {tuple(images.shape)}"
        )
    original_mask_ndim = masks.ndim
    if masks.ndim == 3:
        mask_view = masks[:, None]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        mask_view = masks
    else:
        raise ValueError(
            "V547 augmentation expects mask [B,H,W]/[B,1,H,W], got "
            f"{tuple(masks.shape)}"
        )
    if int(images.shape[0]) != int(mask_view.shape[0]):
        raise ValueError("V547 image/mask batch size mismatch")
    if hr_images is not None:
        if hr_images.ndim != 4 or int(hr_images.shape[0]) != int(images.shape[0]):
            raise ValueError("V547 HR image must be [B,C,Hh,Wh] with matching batch")
        if int(hr_images.shape[1]) != int(images.shape[1]):
            raise ValueError("V547 LR/HR channel mismatch")
    original_hr_mask_ndim = None
    hr_mask_view = None
    if hr_masks is not None:
        original_hr_mask_ndim = hr_masks.ndim
        if hr_masks.ndim == 3:
            hr_mask_view = hr_masks[:, None]
        elif hr_masks.ndim == 4 and hr_masks.shape[1] == 1:
            hr_mask_view = hr_masks
        else:
            raise ValueError("V547 HR mask must be [B,Hh,Wh] or [B,1,Hh,Wh]")
        if int(hr_mask_view.shape[0]) != int(images.shape[0]):
            raise ValueError("V547 HR mask batch mismatch")
        if hr_images is not None and tuple(hr_mask_view.shape[-2:]) != tuple(hr_images.shape[-2:]):
            raise ValueError("V547 HR image/mask spatial size mismatch")
    if tuple(images.shape[-2:]) != tuple(mask_view.shape[-2:]):
        raise ValueError("V547 image/mask spatial size mismatch")

    batch = int(images.shape[0])
    device, dtype = images.device, images.dtype
    probability = min(max(float(probability), 0.0), 1.0)
    active = (torch.rand(batch, device=device) < probability).to(dtype)
    degrees = max(float(rotation_degrees), 0.0)
    scale_min = max(float(scale_min), 1.0e-3)
    scale_max = max(float(scale_max), scale_min)
    translate = min(max(float(translate_fraction), 0.0), 0.25)

    angle = (
        (2.0 * torch.rand(batch, device=device) - 1.0)
        * math.radians(degrees)
        * active
    )
    sampled_scale = scale_min + (
        scale_max - scale_min
    ) * torch.rand(batch, device=device)
    scale = 1.0 + (sampled_scale - 1.0) * active
    tx = (
        (2.0 * torch.rand(batch, device=device) - 1.0)
        * (2.0 * translate)
        * active
    )
    ty = (
        (2.0 * torch.rand(batch, device=device) - 1.0)
        * (2.0 * translate)
        * active
    )
    cosine = torch.cos(angle) / scale.clamp_min(1.0e-3)
    sine = torch.sin(angle) / scale.clamp_min(1.0e-3)
    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, images.size(), align_corners=False)
    augmented_images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    augmented_masks = F.grid_sample(
        mask_view.to(dtype),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    augmented_hr = None
    augmented_hr_mask = None
    if hr_images is not None:
        theta_hr = theta.to(device=hr_images.device, dtype=hr_images.dtype)
        grid_hr = F.affine_grid(theta_hr, hr_images.size(), align_corners=False)
        augmented_hr = F.grid_sample(
            hr_images, grid_hr, mode="bilinear", padding_mode="border", align_corners=False
        )
    if hr_mask_view is not None:
        theta_hr_mask = theta.to(device=hr_mask_view.device, dtype=images.dtype)
        grid_hr_mask = F.affine_grid(theta_hr_mask, hr_mask_view.size(), align_corners=False)
        augmented_hr_mask = F.grid_sample(
            hr_mask_view.to(images.dtype), grid_hr_mask, mode="nearest", padding_mode="zeros", align_corners=False
        )

    contrast_min = float(contrast_min)
    contrast_max = max(float(contrast_max), contrast_min)
    sampled_contrast = contrast_min + (
        contrast_max - contrast_min
    ) * torch.rand(batch, device=device)
    contrast = 1.0 + (sampled_contrast - 1.0) * active
    spatial_mean = augmented_images.mean(dim=(-2, -1), keepdim=True)
    augmented_images = spatial_mean + contrast[:, None, None, None] * (
        augmented_images - spatial_mean
    )
    if augmented_hr is not None:
        hr_mean = augmented_hr.mean(dim=(-2, -1), keepdim=True)
        augmented_hr = hr_mean + contrast.to(augmented_hr)[:, None, None, None] * (
            augmented_hr - hr_mean
        )

    noise_std = max(float(noise_std), 0.0)
    if noise_std > 0.0:
        spatial_std = augmented_images.std(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1.0e-4)
        noise = torch.randn_like(augmented_images) * spatial_std * noise_std
        augmented_images = augmented_images + noise * active[:, None, None, None]
        if augmented_hr is not None:
            hr_std = augmented_hr.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-4)
            hr_noise = torch.randn_like(augmented_hr) * hr_std * noise_std
            augmented_hr = augmented_hr + hr_noise * active.to(augmented_hr)[:, None, None, None]

    lower = images.amin(dim=(-3, -2, -1), keepdim=True)
    upper = images.amax(dim=(-3, -2, -1), keepdim=True)
    augmented_images = torch.maximum(
        torch.minimum(augmented_images, upper), lower
    )
    augmented_masks = (augmented_masks >= 0.5).to(mask_view.dtype)
    if original_mask_ndim == 3:
        augmented_masks = augmented_masks[:, 0]
    if augmented_hr is not None:
        hr_lower = hr_images.amin(dim=(-3, -2, -1), keepdim=True)
        hr_upper = hr_images.amax(dim=(-3, -2, -1), keepdim=True)
        augmented_hr = torch.maximum(torch.minimum(augmented_hr, hr_upper), hr_lower)
    if augmented_hr_mask is not None:
        augmented_hr_mask = (augmented_hr_mask >= 0.5).to(hr_mask_view.dtype)
        if original_hr_mask_ndim == 3:
            augmented_hr_mask = augmented_hr_mask[:, 0]
    if augmented_hr is not None and augmented_hr_mask is not None:
        return augmented_images, augmented_masks, augmented_hr, augmented_hr_mask
    if augmented_hr_mask is not None:
        return augmented_images, augmented_masks, augmented_hr_mask
    if augmented_hr is not None:
        return augmented_images, augmented_masks, augmented_hr
    return augmented_images, augmented_masks
