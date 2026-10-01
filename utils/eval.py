"""Split-aware native-resolution DSC/NSD evaluation.

The file matching and macro-averaging logic follows the MedCLIPSeg evaluator,
with its historical singleton-depth NSD bug corrected:

1. Match prediction and GT by normalized case ID.
2. Resize predictions to the native GT resolution with nearest-neighbour.
3. Threshold masks at 127.
4. Compute per-case DSC and true 2-D surface Dice at tolerance 2.
5. Round each case to four decimal places before macro averaging.

For ISIC, prediction names such as::

    ISIC_0000003_Segmentation.png

are normalized to::

    ISIC_0000003

so that they match GT names such as::

    ISIC_0000003.png
"""

import argparse
import math
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
try:
    from .metrics_2d import case_metrics_2d
except ImportError:  # ``python utils/eval.py`` execution path.
    from metrics_2d import case_metrics_2d
try:
    from .SurfaceDice import (
        compute_dice_coefficient,
        compute_surface_dice_at_tolerance,
        compute_surface_distances,
    )
except ImportError:  # ``python utils/eval.py`` execution path.
    from SurfaceDice import (
        compute_dice_coefficient,
        compute_surface_dice_at_tolerance,
        compute_surface_distances,
    )
from tqdm import tqdm

from main_utils import load_cfg_from_cfg_file


VALID_EXTS = {
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

REMOVABLE_SUFFIXES = (
    "_segmentation",
    "-segmentation",
    "_prediction",
    "-prediction",
    "_pred",
    "-pred",
    "_mask",
    "-mask",
    "_label",
    "-label",
)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--prompt_design",
        type=str,
        default="original",
        help="Text prompt design",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split",
    )
    parser.add_argument(
        "--data_percentage",
        type=int,
        default=100,
        help="Percentage of data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory",
    )
    parser.add_argument(
        "--result-name",
        type=str,
        default="",
        help=(
            "Optional prediction-directory prefix. "
            "Empty uses the original MedCLIPSeg name."
        ),
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="",
        help="Optional unique output CSV filename (basename only).",
    )
    parser.add_argument(
        "--nsd-mode",
        choices=["true2d", "paper_legacy"],
        default="paper_legacy",
        help=(
            "true2d computes geometrically correct 2-D surface Dice; "
            "paper_legacy reproduces the historical HxWx1 evaluator used by "
            "the reference MedCLIPSeg code for protocol-compatible comparison."
        ),
    )
    parser.add_argument(
        "--nsd-tolerance",
        type=float,
        default=2.0,
        help="Surface-distance tolerance in the coordinate space selected below.",
    )
    parser.add_argument(
        "--nsd-tolerance-space",
        choices=["model", "native"],
        default="model",
        help=(
            "model scales the tolerance to native resolution after prediction resize; "
            "native interprets the value directly in native-image pixels."
        ),
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


def canonical_case_id(filename):
    """Return a normalized lower-case case identifier."""
    name = os.path.basename(str(filename))

    # Handles names such as ISIC_0000003.jpg.png.
    while Path(name).suffix.lower() in VALID_EXTS:
        name = Path(name).stem

    while True:
        lowered = name.lower()
        removed = False

        for suffix in REMOVABLE_SUFFIXES:
            if lowered.endswith(suffix):
                name = name[: -len(suffix)]
                removed = True
                break

        if not removed:
            break

    return name.lower()


def explicit_mask_name_priority(filename):
    """Return preference priority for explicit mask filenames.

    A lower value means a stronger indication that the file is a
    segmentation annotation.
    """
    name = os.path.basename(str(filename))

    # Remove one or more image extensions.
    while Path(name).suffix.lower() in VALID_EXTS:
        name = Path(name).stem

    lowered = name.lower()

    priorities = (
        ("_segmentation", 0),
        ("-segmentation", 0),
        ("_mask", 1),
        ("-mask", 1),
        ("_label", 2),
        ("-label", 2),
    )

    for suffix, priority in priorities:
        if lowered.endswith(suffix):
            return priority

    return 100


def collect_image_files(root, role="generic"):
    """Collect images as normalized case_id -> absolute path.

    For GT only, when both a plain filename and an explicit
    segmentation filename exist, the explicit segmentation file is
    selected. Prediction duplicates remain a hard error.
    """
    root = os.path.abspath(str(root))

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Image directory not found: {root}"
        )

    candidates = {}

    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            extension = Path(filename).suffix.lower()

            if extension not in VALID_EXTS:
                continue

            full_path = os.path.abspath(
                os.path.join(current_root, filename)
            )
            case_id = canonical_case_id(filename)

            candidates.setdefault(
                case_id,
                [],
            ).append(full_path)

    mapping = {}
    unresolved_duplicates = {}
    resolved_gt_duplicates = []

    for case_id, paths in candidates.items():
        if len(paths) == 1:
            mapping[case_id] = paths[0]
            continue

        if role == "gt":
            ranked = [
                (
                    explicit_mask_name_priority(
                        os.path.basename(file_path)
                    ),
                    file_path,
                )
                for file_path in paths
            ]

            best_priority = min(
                priority
                for priority, _ in ranked
            )

            best_paths = [
                file_path
                for priority, file_path in ranked
                if priority == best_priority
            ]

            # Resolve only when exactly one explicit mask is present.
            if (
                best_priority < 100
                and len(best_paths) == 1
            ):
                selected_path = best_paths[0]
                mapping[case_id] = selected_path

                resolved_gt_duplicates.append(
                    {
                        "case_id": case_id,
                        "selected": selected_path,
                        "ignored": [
                            file_path
                            for file_path in paths
                            if file_path != selected_path
                        ],
                    }
                )
                continue

        unresolved_duplicates[case_id] = paths

    if resolved_gt_duplicates:
        print(
            "Resolved duplicate GT case IDs by preferring explicit "
            f"segmentation filenames: {len(resolved_gt_duplicates)}"
        )

        for item in resolved_gt_duplicates[:5]:
            print(
                "  GT duplicate resolved | "
                f"case={item['case_id']} | "
                f"selected={os.path.basename(item['selected'])} | "
                f"ignored={[os.path.basename(p) for p in item['ignored']]}"
            )

    if unresolved_duplicates:
        preview = {
            key: paths[:5]
            for key, paths in list(
                unresolved_duplicates.items()
            )[:10]
        }

        raise RuntimeError(
            "Unresolved duplicate normalized case IDs found "
            f"under {root}, role={role}: {preview}"
        )

    return mapping

def calculate_case_metrics(gt_data, seg_data, nsd_mode="true2d", tolerance=2.0):
    """Calculate DSC and NSD for one binary mask pair.

    ``paper_legacy`` is retained only to compare against numbers produced by
    the reference code.  It treats a 2-D mask as a singleton-depth 3-D volume,
    which is not geometrically equivalent to a true 2-D surface metric.

    ``tolerance`` is expressed in the same pixel units as ``gt_data``/
    ``seg_data`` (i.e. native GT resolution once the caller has resized the
    prediction up). A fixed tolerance of "2" only makes sense at the
    resolution it was tuned for; evaluating a much larger native image with
    the same flat 2px tolerance makes the surface-distance test far stricter
    than intended and can silently collapse a good model's NSD. Callers
    should pass a tolerance already scaled to this image's resolution (see
    ``case_nsd_tolerance`` in ``evaluate_directory`` below) rather than rely
    on this default.
    """
    gt_binary = gt_data > 0
    seg_binary = seg_data > 0
    if str(nsd_mode) == "paper_legacy":
        if not gt_binary.any() and not seg_binary.any():
            return 1.0, 1.0
        if not gt_binary.any() or not seg_binary.any():
            return 0.0, 0.0
        dsc = compute_dice_coefficient(gt_binary, seg_binary)
        distances = compute_surface_distances(
            gt_binary[..., None], seg_binary[..., None], [1, 1, 1]
        )
        nsd = compute_surface_dice_at_tolerance(distances, float(tolerance))
        return float(dsc), float(nsd)
    return case_metrics_2d(gt_binary, seg_binary, tolerance=float(tolerance))


def main():
    cfg = get_arguments()

    split = str(cfg.split).lower()

    if split == "val":
        gt_path = os.path.join(
            cfg.DATASET.VAL_PATH,
            "label",
        )
        prediction_suffix = "_Val"
        csv_name = "val_native.csv"
    else:
        gt_path = os.path.join(
            cfg.DATASET.TEST_PATH,
            "label",
        )
        prediction_suffix = (
            f"_Prompt-{cfg.prompt_design}"
        )
        csv_name = (
            f"test_Prompt-{cfg.prompt_design}.csv"
        )

    configured_csv_name = os.path.basename(
        str(getattr(cfg, "csv_name", "") or "").strip()
    )
    if configured_csv_name:
        csv_name = configured_csv_name

    backbone_name = cfg.MODEL.BACKBONE.replace(
        "/",
        "-",
    )

    default_name = (
        f"MedCLIPSeg_"
        f"{cfg.MODEL.CLIP_MODEL}_"
        f"{backbone_name}"
    )

    configured_result_name = str(
        getattr(cfg, "result_name", "") or ""
    ).strip()

    results_name = (
        configured_result_name
        if configured_result_name
        else default_name
    )

    if int(cfg.data_percentage) != 100:
        dataset_name = (
            f"{cfg.DATASET.NAME}_"
            f"{cfg.data_percentage}"
        )
    else:
        dataset_name = cfg.DATASET.NAME

    seg_path = os.path.join(
        cfg.output_dir,
        dataset_name,
        "seg_results",
        f"seed{cfg.seed}",
        results_name + prediction_suffix,
    )

    save_path = os.path.join(
        cfg.output_dir,
        dataset_name,
        "seg_results",
        f"seed{cfg.seed}",
        csv_name,
    )

    print("=" * 100)
    print(f"Dataset             : {dataset_name}")
    print(f"Split               : {split}")
    print(f"GT directory        : {gt_path}")
    print(f"Prediction directory: {seg_path}")
    print(f"CSV output          : {save_path}")
    print(f"NSD protocol        : {cfg.nsd_mode}")
    print(
        f"NSD tolerance       : {float(cfg.nsd_tolerance):g} "
        f"({cfg.nsd_tolerance_space} coordinates)"
    )
    if str(cfg.nsd_mode) == "paper_legacy":
        print(
            "WARNING             : paper_legacy reproduces the reference "
            "HxWx1 protocol for comparison; report true2d as the corrected metric."
        )
    print("=" * 100)

    gt_files = collect_image_files(gt_path, role="gt")
    seg_files = collect_image_files(seg_path, role="prediction")

    gt_names = set(gt_files)
    seg_names = set(seg_files)

    missing_predictions = sorted(
        gt_names - seg_names
    )
    unexpected_predictions = sorted(
        seg_names - gt_names
    )

    print(f"GT image count        : {len(gt_files)}")
    print(f"Prediction image count: {len(seg_files)}")

    if missing_predictions or unexpected_predictions:
        missing_preview = [
            {
                "case_id": case_id,
                "gt_file": gt_files[case_id],
            }
            for case_id in missing_predictions[:10]
        ]

        unexpected_preview = [
            {
                "case_id": case_id,
                "prediction_file": seg_files[case_id],
            }
            for case_id in unexpected_predictions[:10]
        ]

        raise RuntimeError(
            "Prediction/GT mismatch after normalized matching: "
            f"missing_predictions={missing_preview} "
            f"(total={len(missing_predictions)}), "
            f"unexpected_predictions={unexpected_preview} "
            f"(total={len(unexpected_predictions)})."
        )

    common_names = sorted(gt_names)

    print(
        f"Found {len(common_names)} normalized "
        f"matching files for split={split}"
    )

    pairs = [
        (
            case_id,
            gt_files[case_id],
            seg_files[case_id],
        )
        for case_id in common_names
    ]

    seg_metrics = OrderedDict(
        Name=[],
        Case_ID=[],
        Prediction_Name=[],
        DSC=[],
        NSD=[],
        Height=[],
        Width=[],
        Aspect_Ratio=[],
        GT_Area_Fraction=[],
        Pred_Area_Fraction=[],
        NSD_Tolerance_Pixels=[],
        NSD_Tolerance_Space=[],
    )

    with tqdm(
        pairs,
        desc=f"Native evaluation {split}",
    ) as pbar:
        for case_id, gt_file, seg_file in pbar:
            gt_mask = cv2.imread(
                gt_file,
                cv2.IMREAD_GRAYSCALE,
            )
            seg_mask = cv2.imread(
                seg_file,
                cv2.IMREAD_GRAYSCALE,
            )

            if gt_mask is None:
                raise RuntimeError(
                    f"Could not read GT image: {gt_file}"
                )

            if seg_mask is None:
                raise RuntimeError(
                    "Could not read prediction image: "
                    f"{seg_file}"
                )

            prediction_height, prediction_width = seg_mask.shape[:2]
            if str(cfg.nsd_mode) == "true2d" and str(cfg.nsd_tolerance_space) == "model":
                # true2d evaluates at native GT resolution, but the model
                # produces predictions at its own (usually much smaller)
                # resolution. A tolerance tuned in model-resolution pixels
                # (e.g. "2px" at 224x224) is far too strict once the mask is
                # upsampled to native resolution (e.g. 500x500+): it demands
                # near pixel-perfect boundary alignment at a scale the model
                # was never asked to be precise at, silently collapsing NSD
                # even for a visually excellent segmentation. Scale the
                # tolerance by the resize ratio so it stays physically
                # equivalent to the tolerance the model resolution implies.
                scale_y = gt_mask.shape[0] / max(float(prediction_height), 1.0)
                scale_x = gt_mask.shape[1] / max(float(prediction_width), 1.0)
                case_nsd_tolerance = float(cfg.nsd_tolerance) * math.sqrt(
                    scale_y * scale_x
                )
            else:
                case_nsd_tolerance = float(cfg.nsd_tolerance)

            seg_mask = cv2.resize(
                seg_mask,
                (
                    gt_mask.shape[1],
                    gt_mask.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            gt_mask = cv2.threshold(
                gt_mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )[1]

            seg_mask = cv2.threshold(
                seg_mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )[1]

            gt_data = np.uint8(gt_mask)
            seg_data = np.uint8(seg_mask)

            height, width = gt_data.shape[:2]

            dsc, nsd = calculate_case_metrics(
                gt_data,
                seg_data,
                nsd_mode=cfg.nsd_mode,
                tolerance=case_nsd_tolerance,
            )

            seg_metrics["Name"].append(
                os.path.basename(gt_file)
            )
            seg_metrics["Case_ID"].append(
                case_id
            )
            seg_metrics["Prediction_Name"].append(
                os.path.basename(seg_file)
            )
            seg_metrics["DSC"].append(
                round(dsc, 4)
            )
            seg_metrics["NSD"].append(
                round(nsd, 4)
            )
            seg_metrics["Height"].append(
                int(height)
            )
            seg_metrics["Width"].append(
                int(width)
            )
            seg_metrics["Aspect_Ratio"].append(
                float(width) / max(float(height), 1.0)
            )
            seg_metrics["GT_Area_Fraction"].append(
                float((gt_data > 0).mean())
            )
            seg_metrics["Pred_Area_Fraction"].append(
                float((seg_data > 0).mean())
            )
            seg_metrics["NSD_Tolerance_Pixels"].append(
                float(case_nsd_tolerance)
            )
            seg_metrics["NSD_Tolerance_Space"].append(
                str(cfg.nsd_tolerance_space)
            )

            pbar.set_postfix(
                {
                    "Mean DSC": (
                        f"{np.mean(seg_metrics['DSC']):.4f}"
                    ),
                    "Mean NSD": (
                        f"{np.mean(seg_metrics['NSD']):.4f}"
                    ),
                }
            )

    dataframe = pd.DataFrame(seg_metrics)

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    dataframe.to_csv(
        save_path,
        index=False,
    )

    mean_dsc = float(
        dataframe["DSC"].mean()
    )
    mean_nsd = float(
        dataframe["NSD"].mean()
    )

    print(">" * 20)
    print(
        f"Average DSC for {os.path.basename(seg_path)} "
        f"{dataset_name} split={split}: "
        f"{mean_dsc * 100:.2f}%"
    )
    print(
        f"Average NSD for {os.path.basename(seg_path)} "
        f"{dataset_name} split={split}: "
        f"{mean_nsd * 100:.2f}%"
    )
    print(f"Cases evaluated: {len(dataframe)}")
    print(f"Per-case CSV: {save_path}")
    print("<" * 20)


if __name__ == "__main__":
    main()
