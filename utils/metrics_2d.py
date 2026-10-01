"""Reliable native-resolution metrics for binary 2-D segmentation masks.

The project historically passed ``H x W x 1`` arrays to a 3-D surface-Dice
implementation.  The singleton third axis makes almost every foreground pixel
look like a surface point and can substantially inflate NSD.  These helpers
operate on the actual 2-D contours and make the empty-mask convention explicit.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage


def _as_binary_2d(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D mask, got shape={array.shape}")
    return array.astype(bool, copy=False)


def binary_dice_2d(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return the Sørensen-Dice coefficient for two 2-D binary masks."""
    reference = _as_binary_2d(reference, "reference")
    prediction = _as_binary_2d(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError(
            "reference and prediction must have the same shape, got "
            f"{reference.shape} and {prediction.shape}"
        )
    denominator = int(reference.sum()) + int(prediction.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(reference, prediction).sum())
    return float(2.0 * intersection / denominator)


def _surface_2d(mask: np.ndarray) -> np.ndarray:
    """Extract the one-pixel inner contour using 4-connectivity."""
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )
    return np.logical_and(mask, np.logical_not(eroded))


def normalized_surface_dice_2d(
    reference: np.ndarray,
    prediction: np.ndarray,
    tolerance: float = 2.0,
    spacing: Sequence[float] = (1.0, 1.0),
) -> float:
    """Return symmetric 2-D surface Dice at ``tolerance``.

    Distances are evaluated in the same units as ``spacing``.  With the
    default spacing, ``tolerance=2`` means two native-image pixels.  Empty vs.
    empty is 1; exactly one empty mask is 0.
    """
    reference = _as_binary_2d(reference, "reference")
    prediction = _as_binary_2d(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError(
            "reference and prediction must have the same shape, got "
            f"{reference.shape} and {prediction.shape}"
        )
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    spacing_tuple: Tuple[float, float] = tuple(float(value) for value in spacing)  # type: ignore[assignment]
    if len(spacing_tuple) != 2 or any(value <= 0.0 for value in spacing_tuple):
        raise ValueError(f"spacing must contain two positive values, got {spacing}")

    reference_nonempty = bool(reference.any())
    prediction_nonempty = bool(prediction.any())
    if not reference_nonempty and not prediction_nonempty:
        return 1.0
    if reference_nonempty != prediction_nonempty:
        return 0.0

    reference_surface = _surface_2d(reference)
    prediction_surface = _surface_2d(prediction)
    distance_to_prediction = ndimage.distance_transform_edt(
        np.logical_not(prediction_surface),
        sampling=spacing_tuple,
    )
    distance_to_reference = ndimage.distance_transform_edt(
        np.logical_not(reference_surface),
        sampling=spacing_tuple,
    )
    reference_hits = int((distance_to_prediction[reference_surface] <= tolerance).sum())
    prediction_hits = int((distance_to_reference[prediction_surface] <= tolerance).sum())
    denominator = int(reference_surface.sum()) + int(prediction_surface.sum())
    return float((reference_hits + prediction_hits) / max(denominator, 1))


def case_metrics_2d(
    reference: np.ndarray,
    prediction: np.ndarray,
    tolerance: float = 2.0,
    spacing: Sequence[float] = (1.0, 1.0),
) -> tuple[float, float]:
    """Return ``(DSC, NSD)`` under one shared 2-D empty-mask policy."""
    return (
        binary_dice_2d(reference, prediction),
        normalized_surface_dice_2d(reference, prediction, tolerance, spacing),
    )
