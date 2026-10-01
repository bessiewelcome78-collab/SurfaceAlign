"""Fail-closed binary-mask normalization and PNG export.

OpenCV accepts grayscale ``H x W`` images or channel-last images with 1, 3,
or 4 channels.  Segmentation models commonly keep singleton batch/channel
axes (for example ``1 x H x W`` or ``1 x 1 x H x W``).  This module removes
only singleton axes and refuses genuine multi-channel tensors so an invalid
shape can never be silently interpreted as an image layout.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def binary_mask_to_cv2(mask: Any, *, mask_name: str = "<unnamed>") -> np.ndarray:
    """Convert a binary/probability mask to contiguous uint8 ``H x W``.

    Accepted examples are ``H x W``, ``1 x H x W``, ``H x W x 1`` and
    ``1 x 1 x H x W``.  Non-singleton extra axes are rejected with the source
    shape in the exception instead of being passed ambiguously to OpenCV.
    """

    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()

    array = np.asarray(mask)
    source_shape = tuple(int(value) for value in array.shape)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(
            "Binary mask export requires exactly two spatial dimensions after "
            f"removing singleton axes: name={mask_name!r}, "
            f"source_shape={source_shape}, squeezed_shape={tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.bool_) and not np.isfinite(array).all():
        raise ValueError(
            f"Binary mask contains NaN/Inf: name={mask_name!r}, shape={source_shape}"
        )

    # All current callers pass hard masks.  The >0.5 rule also makes this
    # helper safe if a probability map is accidentally supplied later.
    image = (array > 0.5).astype(np.uint8, copy=False) * np.uint8(255)
    return np.ascontiguousarray(image)


def save_binary_mask(directory: str, mask_name: str, mask: Any) -> str:
    """Save one binary prediction and return its path."""

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, mask_name)
    image = binary_mask_to_cv2(mask, mask_name=mask_name)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise RuntimeError(
            f"Could not write prediction: path={path!r}, shape={tuple(image.shape)}, "
            f"dtype={image.dtype}"
        )
    return path
