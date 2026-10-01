"""Boundary-focused evaluation: Boundary IoU, HD95, ASSD.

This module is intentionally a *sibling* of ``utils/eval.py``, not a
replacement. It does not modify eval.py, metrics_2d.py, or SurfaceDice.py.
It imports the file-discovery and case-matching logic directly from
``utils/eval.py`` so that:

  1. It reads predictions from the exact same ``seg_results`` directories
     that DSC/NSD numbers were computed from (no re-inference needed, as
     long as the PNGs from a prior ``test.py`` run still exist on disk).
  2. Case matching (normalized IDs, duplicate-GT resolution, missing/
     unexpected prediction checks) is byte-for-byte identical to the
     evaluator that produced the reported DSC/NSD numbers, so a Boundary
     IoU / HD95 / ASSD row and a DSC / NSD row for the same Case_ID are
     guaranteed to refer to the same image pair.

Metric conventions (documented explicitly so they are defensible in review):

  * Boundary IoU follows Cheng et al., "Boundary IoU: Improving Object-
    Centric Image Segmentation Evaluation" (CVPR 2021). For each mask, an
    interior boundary band is extracted by eroding the mask with a 3x3
    structuring element for ``dilation`` iterations, where
    ``dilation = max(1, round(dilation_ratio * image_diagonal))``, and
    subtracting the eroded mask from the original. Boundary IoU is the
    plain IoU of the two boundary bands. Empty vs. empty -> 1.0 (matches
    the DSC/NSD empty-mask convention already used in metrics_2d.py);
    exactly one empty -> 0.0.

  * HD95 / ASSD use the same 1-pixel, 4-connectivity inner contour
    ("surface") extraction convention as ``utils/metrics_2d.py`` (the
    corrected 2-D surface-Dice implementation), not the legacy singleton-
    depth 3-D SurfaceDice.py path. Both are symmetric:

        ASSD  = mean over all directed point-to-surface distances,
                pooling {d(g, P_surface) : g in G_surface} and
                {d(p, G_surface) : p in P_surface}.
        HD95  = 95th percentile of that same pooled distance set.

    Both empty -> 0.0 (perfect match, no boundary to measure). Exactly one
    empty -> the metric is undefined (infinite Hausdorff distance in the
    usual convention); we report the image diagonal as a finite,
    interpretable worst-case sentinel (the common nnU-Net-style fallback)
    and flag the row via ``Boundary_Undefined_Case`` so it can be
    inspected or excluded rather than silently corrupting the mean.

Usage mirrors utils/eval.py:

    python utils/eval_boundary.py \
        --config-file configs/BUSI_GEOTR_M1_ROBUST52.yaml \
        --seed 42 --split test --output-dir <same --output-dir as test.py> \
        --result-name MedCLIPSeg_unimedclip_ViT-B-16_<RUN_TAG> \
        --csv-name test_boundary_Prompt-original.csv

``--result-name`` must match the prediction-directory prefix actually on
disk under ``<output-dir>/<dataset>/seg_results/seed<seed>/``. Use the
same value you already pass (or would pass) to ``utils/eval.py`` for the
matching DSC/NSD run, so the two CSVs describe the same predictions.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience only; never a hard dependency.
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .eval import collect_image_files
except ImportError:  # ``python utils/eval_boundary.py`` execution path.
    from eval import collect_image_files

from main_utils import load_cfg_from_cfg_file


# ---------------------------------------------------------------------------
# Argument parsing (mirrors utils/eval.py's subset relevant to locating
# GT / prediction directories; does not duplicate --nsd-mode since this
# script only ever computes boundary-family metrics).
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--prompt_design", type=str, default="original", help="Text prompt design")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluation split")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument(
        "--result-name",
        type=str,
        default="",
        help="Prediction-directory prefix. Must match the value used to produce "
        "the existing seg_results PNGs (same as utils/eval.py --result-name). "
        "Empty uses the original MedCLIPSeg default name.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="",
        help="Optional unique output CSV filename (basename only).",
    )
    parser.add_argument(
        "--boundary-dilation-ratio",
        type=float,
        default=0.02,
        help="Boundary IoU band width as a fraction of the image diagonal "
        "(Cheng et al. 2021 default: 0.02).",
    )
    parser.add_argument(
        "--boundary-width-px",
        type=int,
        default=0,
        help="If > 0, overrides --boundary-dilation-ratio with a fixed pixel "
        "band width (e.g. 2, to match the NSD tolerance=2 convention used "
        "elsewhere in this codebase).",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Config overrides",
    )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update(vars(args))

    return cfg


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _as_binary_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape={array.shape}")
    return array.astype(bool, copy=False)


def _inner_surface(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner contour, 4-connectivity. Matches utils/metrics_2d.py."""
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _boundary_band(mask_u8: np.ndarray, dilation: int) -> np.ndarray:
    """Cheng et al. (CVPR 2021) interior boundary band via 3x3 erosion.

    ``mask_u8`` is a {0, 1} uint8 array. Padding by 1px before eroding
    prevents the image border from being treated as an implicit mask edge.
    """
    h, w = mask_u8.shape
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=max(1, int(dilation)))
    eroded = eroded[1 : h + 1, 1 : w + 1]
    return (mask_u8 - eroded).astype(bool)


def boundary_iou(reference: np.ndarray, prediction: np.ndarray, dilation: int) -> float:
    reference = _as_binary_2d(reference)
    prediction = _as_binary_2d(prediction)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference and prediction must match, got {reference.shape} and {prediction.shape}"
        )
    ref_band = _boundary_band(reference.astype(np.uint8), dilation)
    pred_band = _boundary_band(prediction.astype(np.uint8), dilation)

    ref_empty = not bool(ref_band.any())
    pred_empty = not bool(pred_band.any())
    if ref_empty and pred_empty:
        return 1.0
    if ref_empty != pred_empty:
        return 0.0

    intersection = int(np.logical_and(ref_band, pred_band).sum())
    union = int(np.logical_or(ref_band, pred_band).sum())
    return float(intersection / max(union, 1))


def hd95_assd(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float, bool]:
    """Return (HD95, ASSD, undefined_flag).

    ``undefined_flag`` is True exactly when one mask is empty and the other
    is not, in which case both distances are set to the image diagonal
    (finite worst-case sentinel) rather than inf/NaN.
    """
    reference = _as_binary_2d(reference)
    prediction = _as_binary_2d(prediction)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference and prediction must match, got {reference.shape} and {prediction.shape}"
        )

    ref_nonempty = bool(reference.any())
    pred_nonempty = bool(prediction.any())
    if not ref_nonempty and not pred_nonempty:
        return 0.0, 0.0, False
    if ref_nonempty != pred_nonempty:
        diag = float(np.hypot(reference.shape[0] * spacing[0], reference.shape[1] * spacing[1]))
        return diag, diag, True

    ref_surface = _inner_surface(reference)
    pred_surface = _inner_surface(prediction)

    dist_to_pred = ndimage.distance_transform_edt(np.logical_not(pred_surface), sampling=spacing)
    dist_to_ref = ndimage.distance_transform_edt(np.logical_not(ref_surface), sampling=spacing)

    ref_to_pred = dist_to_pred[ref_surface]
    pred_to_ref = dist_to_ref[pred_surface]
    pooled = np.concatenate([ref_to_pred, pred_to_ref])

    if pooled.size == 0:
        # Both surfaces degenerate (e.g. single-pixel masks with no
        # interior/exterior contrast under 4-connectivity erosion).
        return 0.0, 0.0, False

    assd = float(pooled.mean())
    hd95 = float(np.percentile(pooled, 95))
    return hd95, assd, False


# ---------------------------------------------------------------------------
# Main: mirrors utils/eval.py's directory resolution exactly.
# ---------------------------------------------------------------------------

def main():
    cfg = get_arguments()

    split = str(cfg.split).lower()

    if split == "val":
        gt_path = os.path.join(cfg.DATASET.VAL_PATH, "label")
        prediction_suffix = "_Val"
        csv_name = "val_boundary.csv"
    else:
        gt_path = os.path.join(cfg.DATASET.TEST_PATH, "label")
        prediction_suffix = f"_Prompt-{cfg.prompt_design}"
        csv_name = f"test_boundary_Prompt-{cfg.prompt_design}.csv"

    configured_csv_name = os.path.basename(str(getattr(cfg, "csv_name", "") or "").strip())
    if configured_csv_name:
        csv_name = configured_csv_name

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    default_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
    configured_result_name = str(getattr(cfg, "result_name", "") or "").strip()
    results_name = configured_result_name if configured_result_name else default_name

    if int(cfg.data_percentage) != 100:
        dataset_name = f"{cfg.DATASET.NAME}_{cfg.data_percentage}"
    else:
        dataset_name = cfg.DATASET.NAME

    seg_path = os.path.join(
        cfg.output_dir, dataset_name, "seg_results", f"seed{cfg.seed}", results_name + prediction_suffix
    )
    save_path = os.path.join(cfg.output_dir, dataset_name, "seg_results", f"seed{cfg.seed}", csv_name)

    dilation_px = int(cfg.boundary_width_px)

    print("=" * 100)
    print(f"Dataset             : {dataset_name}")
    print(f"Split               : {split}")
    print(f"GT directory        : {gt_path}")
    print(f"Prediction directory: {seg_path}")
    print(f"CSV output          : {save_path}")
    if dilation_px > 0:
        print(f"Boundary band       : fixed {dilation_px}px")
    else:
        print(f"Boundary band       : {cfg.boundary_dilation_ratio} x image diagonal (Cheng et al. 2021)")
    print("=" * 100)

    if not os.path.isdir(seg_path):
        raise FileNotFoundError(
            f"Prediction directory not found: {seg_path}\n"
            "This script does not run inference. If this directory does not "
            "exist, first re-run test.py once for this checkpoint/mode to "
            "produce seg_results PNGs, or point --result-name at an existing "
            "seg_results subfolder."
        )

    gt_files = collect_image_files(gt_path, role="gt")
    seg_files = collect_image_files(seg_path, role="prediction")

    gt_names = set(gt_files)
    seg_names = set(seg_files)

    missing_predictions = sorted(gt_names - seg_names)
    unexpected_predictions = sorted(seg_names - gt_names)

    print(f"GT image count        : {len(gt_files)}")
    print(f"Prediction image count: {len(seg_files)}")

    if missing_predictions or unexpected_predictions:
        raise RuntimeError(
            "Prediction/GT mismatch after normalized matching: "
            f"missing_predictions={missing_predictions[:10]} (total={len(missing_predictions)}), "
            f"unexpected_predictions={unexpected_predictions[:10]} (total={len(unexpected_predictions)}). "
            "This must be resolved (or match utils/eval.py's behaviour on the "
            "same directories) before boundary numbers can be trusted."
        )

    common_names = sorted(gt_names)
    print(f"Found {len(common_names)} normalized matching files for split={split}")

    pairs = [(case_id, gt_files[case_id], seg_files[case_id]) for case_id in common_names]

    rows = OrderedDict(
        Name=[],
        Case_ID=[],
        Prediction_Name=[],
        Boundary_IoU=[],
        HD95=[],
        ASSD=[],
        Boundary_Undefined_Case=[],
        Height=[],
        Width=[],
        GT_Area_Fraction=[],
        Pred_Area_Fraction=[],
    )

    for case_id, gt_file, seg_file in tqdm(pairs, desc=f"Boundary evaluation {split}"):
        gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise RuntimeError(f"Failed to read GT image: {gt_file}")
        if seg_mask is None:
            raise RuntimeError(f"Failed to read prediction image: {seg_file}")

        height, width = gt_mask.shape[:2]
        if seg_mask.shape[:2] != (height, width):
            seg_mask = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        gt_binary = gt_mask > 127
        seg_binary = seg_mask > 127

        dilation = dilation_px if dilation_px > 0 else max(
            1, int(round(float(cfg.boundary_dilation_ratio) * float(np.hypot(height, width))))
        )
        b_iou = boundary_iou(gt_binary, seg_binary, dilation)
        hd95, assd, undefined = hd95_assd(gt_binary, seg_binary)

        rows["Name"].append(os.path.basename(gt_file))
        rows["Case_ID"].append(case_id)
        rows["Prediction_Name"].append(os.path.basename(seg_file))
        rows["Boundary_IoU"].append(round(float(b_iou), 4))
        rows["HD95"].append(round(float(hd95), 4))
        rows["ASSD"].append(round(float(assd), 4))
        rows["Boundary_Undefined_Case"].append(bool(undefined))
        rows["Height"].append(int(height))
        rows["Width"].append(int(width))
        rows["GT_Area_Fraction"].append(round(float(gt_binary.mean()), 6))
        rows["Pred_Area_Fraction"].append(round(float(seg_binary.mean()), 6))

    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frame.to_csv(save_path, index=False)

    n_undefined = int(frame["Boundary_Undefined_Case"].sum())
    print("-" * 100)
    print(f"Cases                 : {len(frame)}")
    print(f"Boundary_IoU (mean)   : {frame['Boundary_IoU'].mean():.4f}")
    print(f"HD95 (mean)           : {frame['HD95'].mean():.4f}")
    print(f"ASSD (mean)           : {frame['ASSD'].mean():.4f}")
    print(f"Undefined cases       : {n_undefined} / {len(frame)} (one mask empty, other non-empty)")
    print(f"Saved CSV             : {save_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
