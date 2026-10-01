#/home/tsz-25/MedCLIPSeg-pristine/test.py
"""Unified MedCLIPSeg inference for Val/Test.

The effective Monte-Carlo sample count is explicit and logged. Predictions are
saved at model resolution and are evaluated at native ground-truth resolution
by ``utils/eval.py``. No ground-truth label, oracle index, or validation target
is used to choose a Test prediction.
"""
import argparse
import csv
import logging
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, InferenceDataset, ValGenerator
from utils.slr_paired_hr_dataset import SLRPairedResolutionDataset
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, normalize, read_text
from utils.mask_export import save_binary_mask


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _canonical_m1_inference_mode(mode):
    normalized = str(mode).strip().lower()
    return {
        "dense_counterfactual_residual_routing": "unified_action_cf_selection",
    }.get(normalized, normalized)


def _legacy_version_token_in_filename(filename: str, version: str) -> bool:
    stem = os.path.splitext(os.path.basename(str(filename)).lower())[0]
    normalized = stem.replace('-', '_').replace('.', '_')
    tokens = tuple(token for token in normalized.split('_') if token)
    return str(version).strip().lower() in tokens


def m1_enabled(cfg):
    return bool(_cfg_get(_cfg_get(cfg, "M1", None), "ENABLED", False))


def _surface_dice_proxy(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2) -> torch.Tensor:
    """Train-resolution hard surface Dice for Val diagnostics only.

    This does not replace the supplied original native-resolution Test evaluator.
    """
    if target.ndim == 4:
        target = target[:, 0]
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    if pred.ndim == 3:
        pred = pred[:, None]
    b, k, h, w = pred.shape
    gt = target[:, None].expand(b, k, h, w)
    x = pred.reshape(b * k, 1, h, w)
    y = gt.reshape(b * k, 1, h, w)
    ex = -torch.nn.functional.max_pool2d(-x, 3, 1, 1)
    ey = -torch.nn.functional.max_pool2d(-y, 3, 1, 1)
    bx = (x - ex).clamp(0.0, 1.0)
    by = (y - ey).clamp(0.0, 1.0)
    r = max(0, int(tolerance))
    near_x = torch.nn.functional.max_pool2d(bx, 2 * r + 1, 1, r)
    near_y = torch.nn.functional.max_pool2d(by, 2 * r + 1, 1, r)
    close_x = (bx * near_y).sum(dim=(1, 2, 3))
    close_y = (by * near_x).sum(dim=(1, 2, 3))
    denom = bx.sum(dim=(1, 2, 3)) + by.sum(dim=(1, 2, 3))
    score = (close_x + close_y) / denom.clamp_min(1e-6)
    area_x = x.sum(dim=(1, 2, 3))
    area_y = y.sum(dim=(1, 2, 3))
    both_empty = (area_x == 0) & (area_y == 0)
    one_empty = (area_x == 0) ^ (area_y == 0)
    score = torch.where(both_empty, torch.ones_like(score), score)
    score = torch.where(one_empty, torch.zeros_like(score), score)
    return score.reshape(b, k)


def _v20_candidate_names(action_types: torch.Tensor, slots: int, prediction=None):
    """Map action slots to stable audit names, including unordered CEM modes."""
    if isinstance(prediction, dict):
        typed_flag = prediction.get("cem_typed_modes")
        if isinstance(typed_flag, torch.Tensor):
            typed_flag = bool(float(typed_flag.detach().reshape(-1)[0].cpu()) > 0.5)
        if typed_flag:
            typed_labels = [
                "preserve",
                "fp_delete",
                "fn_fill",
                "boundary_trim",
                "boundary_expand",
                "feature_discovery_0",
                "feature_discovery_1",
            ]
            if slots <= len(typed_labels):
                return typed_labels[:slots]
        combo = prediction.get("cem_combo_matrix")
        if isinstance(combo, torch.Tensor) and combo.ndim == 2 and combo.shape[0] == slots:
            names = []
            for row in combo.detach().cpu():
                active = torch.nonzero(row > 0.5, as_tuple=False).flatten().tolist()
                if not active:
                    names.append("preserve")
                elif len(active) == 1:
                    names.append(f"cem_single_mode_{active[0]}")
                elif len(active) == int(row.numel()):
                    names.append("cem_full")
                else:
                    names.append("cem_pair_" + "_".join(map(str, active)))
            return names
    labels = {0: "island_delete", 1: "boundary_trim", 2: "boundary_fill", 3: "hole_fill"}
    if action_types is None:
        return ["preserve"] + [f"action_{idx}" for idx in range(1, slots)]
    types = [int(value) for value in action_types.detach().cpu().tolist()]
    rank_by_type = {}
    names = ["preserve"]
    for action_type in types:
        rank = rank_by_type.get(action_type, 0)
        rank_by_type[action_type] = rank + 1
        names.append(f"{labels.get(action_type, f'type{action_type}')}_r{rank}")
    if len(names) != slots:
        return ["preserve"] + [f"action_{idx}" for idx in range(1, slots)]
    return names


def _value_at(prediction, key, batch_index, action_index=None, default=""):
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        return default
    try:
        if action_index is None:
            item = value[batch_index]
        else:
            item = value[batch_index, action_index]
        item = item.detach().float().cpu()
        if item.numel() == 1:
            return float(item.reshape(-1)[0])
        # Some legacy V25/V27 keys may be returned as local maps [H,W]
        # instead of action scalars.  For audit columns we need one scalar per
        # case/action, so report the mean instead of crashing and hiding the
        # useful diagnostics.  Boolean masks therefore become valid fractions.
        return float(item.mean())
    except (IndexError, TypeError, RuntimeError, ValueError):
        return default


def _value_at_slot(prediction, key, batch_index, candidate_index, default=""):
    """Read a [B, 1+K] tensor with candidate_index including Preserve."""
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        return default
    try:
        item = value[batch_index, candidate_index]
        return float(item.detach().float().cpu())
    except (IndexError, TypeError, RuntimeError):
        return default


def _value_at_component(prediction, key, batch_index, action_index, component_index, default=""):
    """Read a [B, K, C] tensor component for an action slot."""
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        return default
    try:
        item = value[batch_index, action_index, component_index]
        return float(item.detach().float().cpu())
    except (IndexError, TypeError, RuntimeError):
        return default


def _string_at(prediction, key, batch_index, default=""):
    """Read a per-case non-tensor diagnostic, e.g. V393 fallback text."""
    value = prediction.get(key)
    if isinstance(value, (list, tuple)) and 0 <= batch_index < len(value):
        return str(value[batch_index])
    if isinstance(value, str):
        return value
    return default


def _activate_v25_runtime_compat(cfg, config_file):
    """Activate V25/V26/V27 utility-bank defaults for inference."""
    filename = os.path.basename(str(config_file)).lower()
    m1 = _cfg_get(cfg, "M1", None)
    if m1 is None:
        return cfg
    dataset = _cfg_get(cfg, "DATASET", None)
    keys = (
        "V25_TYPE_CONDITIONAL_UTILITY_BANK", "V25_HIDDEN_DIM",
        "V25_TYPE_EMBED_DIM", "V25_MAX_ACTIONS", "M1_TRAIN_NUM_SAMPLES",
        "V26_POSITIVE_CASE_WEIGHT", "V26_NULL_SAFETY_WEIGHT",
        "V27_DSC_BENEFIT_MARGIN", "V27_DSC_HARM_MARGIN",
        "V27_NSD_BENEFIT_FLOOR", "V27_NSD_HARM_MARGIN",
        "V27_NSD_TOLERANCE_PIXELS", "V27_DSC_UTILITY_WEIGHT",
        "V27_NSD_UTILITY_WEIGHT", "V27_CASE_RANK_WEIGHT",
        "V27_CASE_RANK_MARGIN", "V27_NULL_MARGIN",
        "V27_CONTROL_CONTEXT_GUARD",
    )
    for key in keys:
        if _cfg_get(m1, key, None) is None:
            value = _cfg_get(dataset, key, None)
            if value is not None:
                setattr(m1, key, value)

    tag = str(_cfg_get(m1, "RUN_TAG", "")).upper()
    if _legacy_version_token_in_filename(filename, "v27") or tag.startswith("V27_"):
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not tag.startswith("V27_"):
            setattr(m1, "RUN_TAG", "V27_B0Frozen_ParetoUtilityPolicy_100ep")
        for key, value in {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V25_HIDDEN_DIM": 128,
            "V25_TYPE_EMBED_DIM": 16,
            "V25_MAX_ACTIONS": 1,
            "V26_POSITIVE_CASE_WEIGHT": 8.0,
            "V26_NULL_SAFETY_WEIGHT": 1.0,
            "V27_DSC_BENEFIT_MARGIN": 0.003,
            "V27_DSC_HARM_MARGIN": 0.003,
            "V27_NSD_BENEFIT_FLOOR": 0.0,
            "V27_NSD_HARM_MARGIN": 0.010,
            "V27_NSD_TOLERANCE_PIXELS": 2,
            "V27_DSC_UTILITY_WEIGHT": 0.55,
            "V27_NSD_UTILITY_WEIGHT": 0.45,
            "V27_CASE_RANK_WEIGHT": 1.0,
            "V27_CASE_RANK_MARGIN": 0.10,
            "V27_NULL_MARGIN": 0.10,
            "V27_CONTROL_CONTEXT_GUARD": 1,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    if _legacy_version_token_in_filename(filename, "v26") or tag.startswith("V26_") or _cfg_get(m1, "V26_NULL_MARGIN", None) is not None:
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not tag.startswith("V26_"):
            setattr(m1, "RUN_TAG", "V26_B0Frozen_NullRelativeUtilityPolicy_100ep")
        for key, value in {
            "V26_UTILITY_DECISION_WEIGHT": 0.75,
            "V26_POSITIVE_CASE_WEIGHT": 8.0,
            "V26_NULL_MARGIN": 0.10,
            "V26_NULL_SAFETY_WEIGHT": 1.0,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    enabled = (
        "v25_typeconditionalutilitybank" in filename
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower() == "v25_type_conditional_utility"
        or tag.startswith("V25_")
    )
    if enabled:
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not tag.startswith("V25_"):
            setattr(m1, "RUN_TAG", "V25_B0Frozen_TypeConditionalUtilityBank_100ep")
    return cfg

def results_name(cfg):
    name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{cfg.MODEL.BACKBONE.replace('/', '-')}"
    if m1_enabled(cfg):
        name += "_" + str(_cfg_get(cfg.M1, "RUN_TAG", "M1PSEDirectFusionV4"))
    return name


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--prompt_design", type=str, default="original", help="Text prompt design")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Data split")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use")
    parser.add_argument("--source_dataset", type=str, help="Dataset directory holding the checkpoint")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional explicit matching M1 checkpoint. Formal M1 inference requires a checkpoint trained with this code.")
    parser.add_argument("--num-samples", type=int, default=None, help="Explicit MC sample count for this inference run. Overrides TEST.NUM_SAMPLES when provided.")
    parser.add_argument(
        "--export-mode", choices=["joint", "base"], default="joint",
        help="base exports the official Base posterior only and skips OACD.",
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=0,
        help="Explicit DataLoader batch size; 0 uses protocol defaults.",
    )
    parser.add_argument("--verifier-text-override", type=str, default="", help="Val audit only: keep B0/M1 candidates from the original prompt but recompute V15.2 verifier evidence with this fixed alternate prompt.")
    parser.add_argument("--allow-v15-identity-compat", action="store_true", help="Val audit only: permit a Phase-A checkpoint missing exactly the V15 patch adapter, initialized as identity.")
    parser.add_argument(
        "--allow-v15-calibrator-compat",
        action="store_true",
        help="Val-only: allow legacy V15 Sequential calibrator removal for Preserve-only raw-evidence audit.",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Config overrides")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({key: value for key, value in vars(args).items()})
    cfg = _activate_v25_runtime_compat(cfg, args.config_file)
    return cfg


def logger_config(log_path):
    logger = logging.getLogger(f"MedCLIPSegTest:{os.path.abspath(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")
        file_handler = logging.FileHandler(log_path, encoding="UTF-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def build_model(cfg):
    clip_model = str(cfg.MODEL.CLIP_MODEL).lower()
    if clip_model != "unimedclip":
        raise ValueError(
            "This V410-only project supports MODEL.CLIP_MODEL=unimedclip only; "
            f"got {cfg.MODEL.CLIP_MODEL!r}."
        )
    return build_medclipseg_unimedclip(cfg)

def _checkpoint_path(cfg, run_name):
    if cfg.checkpoint:
        return cfg.checkpoint
    checkpoint_dataset = cfg.source_dataset if cfg.source_dataset else cfg.DATASET.NAME
    final_fusion = m1_enabled(cfg) and _canonical_m1_inference_mode(
        _cfg_get(cfg.M1, "INFERENCE_MODE", "preserve")
    ) in {"direct_fusion", "router_fusion", "text_verifier_fusion", "utility_risk_selection", "falsification_m3_selection", "unified_action_cf_selection", "unified_m1_safe_fusion"}
    checkpoint_dir = os.path.join(
        cfg.output_dir,
        checkpoint_dataset,
        "trained_models",
        f"seed{cfg.seed}",
    )
    if cfg.TEST.USE_LATEST:
        candidates = [f"{run_name}_latest.pth", f"{run_name}_last_epoch.pth"]
    elif bool(_cfg_get(cfg.TRAIN, "USE_VALIDATION_SELECTION", False)):
        # train.py writes the pre-declared validation-selected checkpoint with
        # this exact suffix.  Older test.py versions searched only for the
        # legacy best_fusion_val name and therefore could not execute the
        # formal one-shot Test automatically.
        candidates = [
            f"{run_name}_best_val.pth",
            f"{run_name}_best_fusion_val.pth",
            f"{run_name}_best_dice.pth",
        ]
    elif final_fusion:
        candidates = [f"{run_name}_best_fusion_val.pth"]
    else:
        candidates = [f"{run_name}_best_dice.pth"]
    for filename in candidates:
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            return path
    return os.path.join(checkpoint_dir, candidates[0])


def main():
    cfg = get_arguments()
    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)
    if cfg.data_percentage != 100:
        cfg.DATASET.NAME = f"{cfg.DATASET.NAME}_{cfg.data_percentage}"

    if str(getattr(cfg, "verifier_text_override", "")).strip() and cfg.split != "val":
        raise ValueError("--verifier-text-override is permitted only with --split val; it is forbidden for Test.")
    if bool(getattr(cfg, "allow_v15_identity_compat", False)) and cfg.split != "val":
        raise ValueError("--allow-v15-identity-compat is permitted only with --split val; it is forbidden for Test.")

    if bool(getattr(cfg, "allow_v15_calibrator_compat", False)) and cfg.split != "val":
        raise ValueError(
            "--allow-v15-calibrator-compat is permitted only with --split val; it is forbidden for Test."
        )

    run_name = results_name(cfg)
    checkpoint_path = _checkpoint_path(cfg, run_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    result_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}")
    os.makedirs(result_root, exist_ok=True)
    logger = logger_config(os.path.join(result_root, f"log_{cfg.split}.txt"))
    logger.info("************\n** Config **\n************\n%s", cfg)
    logger.info("Run name: %s | split: %s", run_name, cfg.split)
    logger.info("Checkpoint: %s", checkpoint_path)
    if m1_enabled(cfg):
        requested_mode = str(
            _cfg_get(cfg.M1, "INFERENCE_MODE", "preserve")
        ).strip().lower()
        mode = _canonical_m1_inference_mode(requested_mode)
        if requested_mode == "dense_counterfactual_residual_routing":
            logger.info(
                "SPARC-HR3 DCRR deployment: every HR routing cell selects M1/preserve, "
                "a transported posterior, an absolute HR proposal, ADD, or REMOVE. "
                "The public DCRR mode is canonically dispatched through the validated "
                "MHCS unified-action outer path; no GT or test-time label is used."
            )
        elif mode == "falsification_m3_selection":
            logger.info(
                "V15.3 fixed dense mask-text observer with same-area controls enabled: M1 exports local candidate hypotheses; M2 measures "
                "dense patch-token mask-conditioned evidence and same-area control specificity; deterministic M3 filters "
                "by text evidence then applies structural consensus with Preserve fallback. No GT, Oracle or test-time label is used."
            )
        elif mode == "utility_risk_selection":
            logger.info("Legacy V10 hard-Dice utility-risk selection enabled.")
        elif mode == "text_verifier_fusion":
            logger.info(
                "M1 Causal Candidate Bank + M2 Text Verifier: MC Base probabilities and patch features "
                "are averaged first; M1 generates Preserve/Shrink/Expand hypotheses and M2 ranks them "
                "from candidate visual evidence plus the text embedding. No GT, Oracle index, or "
                "test-time selector label is used."
            )
        elif mode == "router_fusion":
            logger.info(
                "M1 Conservative Router-Fusion: MC Base probabilities and patch features are averaged first; "
                "a conservative edit gate decides whether Base may change, then a conditional direction head "
                "selects Shrink/Expand. No GT/Oracle/test-time selector/M2/M3 is used."
            )
        elif mode == "unified_action_cf_selection":
            if str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "geotr_m1":
                logger.info(
                    "GEOTR-M1 exact deployment: MC-mean Base logits are transformed once by the "
                    "validated semantic Geometry/Transport field. M2/M3/GT/Oracle/Test labels are absent."
                )
            elif str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "semlt":
                logger.info(
                    "SemLT-M1 deployment: the final output is one semantic-conditioned, "
                    "bounded dense transport of MC-mean Base logits. No M2/M3/GT/Oracle/Test label is used."
                )
            elif str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "mhcs":
                logger.info(
                    "MHCS-R4.8 deployment: H0 is the fixed NoEdit anchor; H1..HK are "
                    "exchangeable complete-mask proposals. Candidate-set-aware 14x14 global "
                    "context conditions a 28x28 local route. Only proposals with predicted "
                    "positive benefit over H0 may receive entmax mass; otherwise the output is "
                    "exact Preserve. No GT, Oracle, validation label, or Test label is used."
                )
            elif bool(_cfg_get(cfg.M1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", False)):
                if str(_cfg_get(cfg.M1, "RUN_TAG", "")).upper().startswith("V27_"):
                    logger.info(
                        "V27 Pareto utility deployment: image-only factual/control ROI evidence and a learned "
                        "one-action-vs-Null policy choose at most one context-disjoint action; otherwise Preserve "
                        "is emitted exactly. No GT, Oracle, validation label, or Test label is used at inference."
                    )
                else:
                    logger.info(
                        "V25 type-conditional utility deployment: matched factual/control ROI utility "
                        "heads choose at most one low-risk action; otherwise Preserve is emitted exactly. "
                        "No GT, Oracle, validation label, or Test label is used at inference."
                    )
            else:
                logger.info(
                    "V20-family atomic action deployment enabled. No GT, Oracle, or test-time label is used."
                )
        elif mode == "unified_m1_safe_fusion":
            logger.info(
                "V474 compositional error-mode deployment: MC probabilities are averaged first; "
                "the learned safe selector may choose a complete candidate or preserve C0. "
                "No GT, oracle index, or validation label is used at inference."
            )
        elif mode == "direct_fusion":
            logger.info(
                "M1 Direct-Fusion: Base probabilities are MC-averaged first; C1/C2 are generated from that mean "
                "and composed into the trained final output. No GT/Oracle/selector/M2/M3."
            )
        else:
            logger.info("M1 Preserve: C0 Base is the final output; candidates are diagnostics only.")

    if m1_enabled(cfg):
        test_mc_seed = int(_cfg_get(cfg.M1, "TEST_MC_SEED", cfg.seed))
        if test_mc_seed >= 0:
            logger.info("Fixed Test MC seed: %d", test_mc_seed)
            set_random_seed(test_mc_seed)

    requested_num_samples = getattr(cfg, "num_samples", None)
    if requested_num_samples is None:
        effective_num_samples = int(cfg.TEST.NUM_SAMPLES)
    else:
        effective_num_samples = int(requested_num_samples)
    if effective_num_samples <= 0:
        raise ValueError(f"MC sample count must be positive; got {effective_num_samples}.")
    requested_num_samples_for_mc_match = effective_num_samples
    if (
        m1_enabled(cfg)
        and bool(_cfg_get(cfg.M1, "GEOTR_M1_DETERMINISTIC_EVAL", False))
        and effective_num_samples != 1
    ):
        logger.warning(
            "Exact deterministic GEOTR-M1 does not define a stochastic posterior; "
            "overriding requested MC=%d with one effective pass.",
            effective_num_samples,
        )
        effective_num_samples = 1
    if m1_enabled(cfg) and bool(_cfg_get(cfg.M1, "GEOTOPO_REFINEMENT_ENABLED", False)):
        posterior_order = str(
            _cfg_get(cfg.M1, "GEOTR_POSTERIOR_INFERENCE_ORDER", "mean_then_refine")
        ).strip().lower()
        if bool(_cfg_get(cfg.M1, "GEOTR_REQUIRE_VAL_TEST_MC_MATCH", False)):
            val_mc = int(_cfg_get(cfg.M1, "VAL_NUM_SAMPLES", requested_num_samples_for_mc_match))
            if requested_num_samples_for_mc_match != val_mc:
                raise ValueError(
                    "GEOTR formal inference requires the same MC sample count used for "
                    f"validation selection: Val={val_mc}, current={requested_num_samples_for_mc_match}."
                )
        logger.info(
            "[GEOTR_POSTERIOR_PROTOCOL] order=%s | effective_MC=%d",
            posterior_order, effective_num_samples,
        )
    logger.info("Effective MC samples: %d", effective_num_samples)

    model = build_model(cfg)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    raw_state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    state_dict = dict(raw_state_dict)

    # 旧 V15：Sequential semantic_calibrator.0/.2
    # V15.3：Monotone semantic_calibrator.raw_weight/bias
    # 二者无参数等价映射。仅在 Preserve-only 的 Val 原始证据审计中，
    # 丢弃旧 calibrator，保留 V15.3 构造函数初始化的单调 calibrator。
    legacy_keys = {
        "m2_text_verifier.semantic_calibrator.0.weight",
        "m2_text_verifier.semantic_calibrator.0.bias",
        "m2_text_verifier.semantic_calibrator.2.weight",
        "m2_text_verifier.semantic_calibrator.2.bias",
    }
    new_keys = {
        "m2_text_verifier.semantic_calibrator.raw_weight",
        "m2_text_verifier.semantic_calibrator.bias",
    }

    legacy_present = {key for key in legacy_keys if key in state_dict}
    using_legacy_compat = bool(legacy_present)

    if using_legacy_compat:
        if cfg.split != "val" or not bool(
            getattr(cfg, "allow_v15_calibrator_compat", False)
        ):
            raise RuntimeError(
                "旧 V15 calibrator 仅允许用于显式 Val Preserve-only 审计；"
                "禁止用于 Test。"
            )

        if legacy_present != legacy_keys:
            raise RuntimeError(
                "旧 V15 calibrator checkpoint 不完整，拒绝兼容加载："
                f"{sorted(legacy_present)}"
            )

        gates = [
            float(_cfg_get(cfg.M1, "M3_TEXT_CORE_QUALITY_MIN", -1.0)),
            float(_cfg_get(cfg.M1, "M3_TEXT_MASK_DELTA_MIN", -1.0)),
            float(_cfg_get(cfg.M1, "M3_TEXT_CONTROL_SPECIFICITY_MIN", -1.0)),
            float(_cfg_get(cfg.M1, "M3_SEMANTIC_VALID_MIN", -1.0)),
            float(_cfg_get(cfg.M1, "M3_MIN_TEXT_SCORE", -1.0)),
        ]
        if min(gates) < 1e5:
            raise RuntimeError(
                "旧 V15 checkpoint 兼容模式要求所有 M3 门限 >= 1e5，"
                "确保本次只能 Preserve-only Val 审计。"
            )

        for key in legacy_keys:
            state_dict.pop(key)

        logger.info(
            "Val-only legacy compatibility: removed V15 Sequential calibrator; "
            "V15.3 monotone calibrator keeps constructor initialization. "
            "Candidate bank and raw dense evidence are unchanged."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)

    allowed_missing = set()
    if using_legacy_compat:
        allowed_missing |= new_keys

    patch_adapter_key = "m2_text_verifier.patch_adapter.weight"
    if patch_adapter_key in missing:
        if cfg.split != "val" or not bool(
            getattr(cfg, "allow_v15_identity_compat", False)
        ):
            raise RuntimeError(
                "缺少 patch adapter 仅允许用于显式 Val identity 审计；禁止用于 Test。"
            )

        verifier = getattr(model, "m2_text_verifier", None)
        adapter = getattr(verifier, "patch_adapter", None)
        weight = getattr(adapter, "weight", None)

        if weight is None or weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise RuntimeError("patch adapter 不可用或不是方阵，无法初始化 identity。")

        with torch.no_grad():
            weight.copy_(
                torch.eye(
                    weight.shape[0],
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )

        allowed_missing.add(patch_adapter_key)
        logger.info("Val-only identity compatibility: initialized missing patch adapter.")

    disallowed_missing = missing - allowed_missing

    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch is not V15.3-compatible. "
            f"Disallowed missing keys: {sorted(disallowed_missing)}; "
            f"unexpected keys: {sorted(unexpected)}"
        )

    # V393: deploy exactly the weight source used for Val selection.
    cp_weight_source = (
        str(checkpoint.get("weight_source", "unknown")).lower()
        if isinstance(checkpoint, dict) else "raw"
    )
    is_v393 = (
        str(_cfg_get(cfg.M1, "M1_LOSS_VERSION", "")).lower()
        == "v393_preserve_aware_edit_control"
    )
    use_ema = (
        isinstance(checkpoint, dict)
        and "ema_shadow" in checkpoint
        and (
            cp_weight_source == "ema"
            or (cp_weight_source == "unknown" and not is_v393)
        )
    )

    deployed_weight_source = "raw"
    if use_ema:
        matched, skipped = 0, 0
        model_state = model.state_dict()
        for k, v in checkpoint["ema_shadow"].items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                matched += 1
            else:
                skipped += 1
        deployed_weight_source = "ema"
        logger.info(
            "EMA shadow deployed: %d matched, %d skipped.",
            matched, skipped,
        )

    checkpoint_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    deployment_epoch = (
        int(checkpoint.get("deployment_epoch_override", checkpoint_epoch))
        if isinstance(checkpoint, dict) else checkpoint_epoch
    )
    if hasattr(model, "set_epoch"):
        model.set_epoch(deployment_epoch)
    checkpoint_role = (
        str(checkpoint.get("checkpoint_role", ""))
        if isinstance(checkpoint, dict) else ""
    )
    if (
        deployment_epoch < 0
        and checkpoint_role == "best_m1_preserve_fallback_no_safe_m2"
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTR_SPARC_HR_ENABLED", False))
    ):
        refiner = getattr(getattr(model, "m1_pse", None), "v4g_refiner", None)
        if refiner is None or not hasattr(refiner, "force_preserve"):
            raise RuntimeError(
                "SPARC Preserve/M1 fallback requested but refiner has no force_preserve contract"
            )
        refiner.force_preserve = True
        logger.warning(
            "SPARC safe fallback active: M2 execution is disabled; checkpoint's M1 is deployed."
        )
    if (
        checkpoint_role == "best_base_preserve_fallback_no_safe_m2"
        and str(_cfg_get(_cfg_get(cfg, "M1", None), "PROTOCOL", "")).strip().lower()
        == "semlt_autozero"
    ):
        refiner = getattr(model, "m1_pse", None)
        if refiner is None or not hasattr(refiner, "force_preserve"):
            raise RuntimeError(
                "SemLT safe Base fallback requested but m1_pse has no force_preserve contract"
            )
        refiner.force_preserve = True
        logger.warning(
            "SemLT-LST v3 safe fallback active: deployed output is exactly Base because "
            "no validation checkpoint satisfied DSC+NSD non-degradation."
        )
    logger.info(
        "Checkpoint epoch=%d | deployment_epoch=%d | selected_weight_source=%s | deployed_weight_source=%s",
        checkpoint_epoch,
        deployment_epoch,
        cp_weight_source,
        deployed_weight_source,
    )

    model.eval().to(cfg.MODEL.DEVICE)

    test_transform = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    if cfg.split == "test":
        data_path = cfg.DATASET.TEST_PATH
        text_file = f"Test_text_{cfg.prompt_design}.xlsx"
        split_suffix = f"_Prompt-{cfg.prompt_design}"
    else:
        data_path = cfg.DATASET.VAL_PATH
        text_file = "Val_text.xlsx"
        split_suffix = "_Val"
    text_rows = read_text(os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, text_file))
    m1_cfg = _cfg_get(cfg, "M1", None)
    slr_true_hr = bool(
        _cfg_get(m1_cfg, "ENABLED", False)
        and (
            bool(_cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False))
            or (
                str(_cfg_get(m1_cfg, "GEOTR_AEFR_STAGE", "")).strip().lower() == "sparse_local_rerendering"
                and _cfg_get(m1_cfg, "GEOTR_SLR_TRUE_HR_ENABLED", False)
            )
        )
    )
    dataset = (
        SLRPairedResolutionDataset(
            data_path, cfg.DATASET.NAME, text_rows,
            image_size=cfg.DATASET.SIZE,
            hr_size=int(_cfg_get(m1_cfg, "GEOTR_SLR_HR_SIZE", 448)),
            training=False,
            cfg=cfg,
        ) if slr_true_hr else (
            InferenceDataset(
                data_path, cfg.DATASET.NAME, text_rows, test_transform,
                image_size=cfg.DATASET.SIZE,
            )
            if str(cfg.split).lower() == "test"
            else DatasetSegmentation(
                data_path, cfg.DATASET.NAME, text_rows, test_transform,
                image_size=cfg.DATASET.SIZE,
            )
        )
    )
    # Memory-safe inference batching.
    #
    # This is an implementation-only DataLoader batch size.  It does NOT change:
    #   * the trained checkpoint,
    #   * the Test split,
    #   * MC sample count,
    #   * posterior mean/std/disagreement,
    #   * UC-FNRT geometry,
    #   * thresholding or evaluation metrics.
    #
    # UC-FNRT uses MC posterior inference plus dense normal-ray transport, so
    # the historical generic batch=32 can exceed A40 memory even though the
    # mathematical per-case inference is unchanged.  All UC-FNRT variants use
    # the same batch=2 contract for apples-to-apples evaluation.
    is_sparc_hr = bool(
        _cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False)
    )
    is_uc_fnrt = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_FNRT", False)
    )

    if int(getattr(cfg, "inference_batch_size", 0)) > 0:
        test_batch_size = int(cfg.inference_batch_size)
    elif str(getattr(cfg, "export_mode", "joint")) == "base":
        # This is the batch size used by the public test.py. It matters for a
        # stochastic MC posterior because it changes which draw belongs to
        # which case.
        test_batch_size = 32
    else:
        test_batch_size = 2 if (is_sparc_hr or is_uc_fnrt) else 32

    logger.info(
        "[INFERENCE_BATCH] batch_size=%d | effective_MC=%d | "
        "UC_FNRT=%s | SPARC_HR=%s",
        test_batch_size,
        effective_num_samples,
        is_uc_fnrt,
        is_sparc_hr,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
    )

    seg_dir = os.path.join(result_root, run_name + split_suffix)
    sparc_hr_seg_dir = os.path.join(result_root, run_name + "_SPARC_HR" + split_suffix)
    sparc_m1_hr_seg_dir = os.path.join(result_root, run_name + "_SPARC_M1_HR" + split_suffix)
    sparc_base_hr_seg_dir = os.path.join(result_root, run_name + "_SPARC_Base_HR" + split_suffix)
    base_native_seg_dir = os.path.join(result_root, run_name + "_BaseNative" + split_suffix)
    m1_native_seg_dir = os.path.join(result_root, run_name + "_M1Native" + split_suffix)
    m1_native_hard_seg_dir = os.path.join(result_root, run_name + "_M1NativeHard" + split_suffix)
    unc_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "unc_results", f"seed{cfg.seed}", run_name + split_suffix)
    os.makedirs(seg_dir, exist_ok=True)
    if bool(_cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False)):
        os.makedirs(sparc_hr_seg_dir, exist_ok=True)
        os.makedirs(sparc_m1_hr_seg_dir, exist_ok=True)
        os.makedirs(sparc_base_hr_seg_dir, exist_ok=True)
    if m1_enabled(cfg) and str(getattr(cfg, "export_mode", "joint")) != "base":
        os.makedirs(base_native_seg_dir, exist_ok=True)
        os.makedirs(m1_native_seg_dir, exist_ok=True)
        os.makedirs(m1_native_hard_seg_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)

    save_diagnostics = bool(
        m1_enabled(cfg)
        and str(getattr(cfg, "export_mode", "joint")) != "base"
        and _cfg_get(cfg.M1, "SAVE_DIAGNOSTICS", False)
    )
    diagnostics_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "m1_diagnostics", f"seed{cfg.seed}", f"{run_name}_{cfg.split}")
    diagnostic_rows = []
    v27_action_audit_rows = []
    deployment_case_count = 0
    deployment_changed_case_count = 0
    deployment_changed_pixel_fraction_sum = 0.0
    # JBT-v5 optionally exports the shared-field counterfactual strengths.
    # These folders enable a direct Case-Oracle ceiling audit without changing
    # which candidate is actually deployed.
    jbt_v5_strength_dirs = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference {cfg.split}"):
            images = batch["image"].to(cfg.MODEL.DEVICE)
            images_hr = batch.get("image_hr", None)
            if isinstance(images_hr, torch.Tensor):
                images_hr = images_hr.to(cfg.MODEL.DEVICE)
            # All M1 runs use the model's unified MC-aware prediction path.
            # This guarantees Direct-Fusion post-processing is applied even
            # when diagnostics are disabled.
            prediction = None
            if str(getattr(cfg, "export_mode", "joint")) == "base":
                base_probs = model.predict_base_probs(
                    images, batch["text_prompt"], num_samples=effective_num_samples
                )
                final_probs = base_probs
                sparc_hr_probs = None
                sparc_m1_hr_probs = None
                sparc_base_hr_probs = None
            elif m1_enabled(cfg):
                verifier_text = None
                if str(getattr(cfg, "verifier_text_override", "")).strip():
                    verifier_text = [str(cfg.verifier_text_override)] * len(batch["text_prompt"])
                prediction = model.predict_m1_diagnostics(
                    images, batch["text_prompt"], num_samples=effective_num_samples,
                    verifier_text=verifier_text,
                    slr_hr_image=images_hr,
                )
                final_probs = prediction["final_probs"]
                sparc_hr_probs = prediction.get("sparc_final_prob_hr")
                sparc_m1_hr_probs = prediction.get("sparc_anchor_prob_hr")
                sparc_base_hr_probs = prediction.get("sparc_base_prob_hr")
                base_probs = prediction["base_probs"]
                candidate_probs = prediction["candidate_probs"]
                fused_probs = prediction["fused_probs"]
                _v5_strength_probs = prediction.get("jbt_v5_strength_candidate_probs")
                if isinstance(_v5_strength_probs, torch.Tensor) and not jbt_v5_strength_dirs:
                    _strengths = _cfg_get(m1_cfg, "JBT_V5_COUNTERFACTUAL_STRENGTHS", [0.5, 1.0, 1.5])
                    for _idx, _alpha in enumerate(_strengths, start=1):
                        _tag = str(float(_alpha)).replace(".", "p")
                        _dir = os.path.join(result_root, run_name + f"_JBTStrength_{_tag}" + split_suffix)
                        os.makedirs(_dir, exist_ok=True)
                        jbt_v5_strength_dirs[_idx] = _dir
                if str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() in {"semlt", "geotr_m1", "semlt_autozero"}:
                    m1_native_probs = prediction.get(
                        "geotopo_geometry_probs", prediction.get("mhcs_final_probs", base_probs)
                    )
                    m1_native_hard_probs = m1_native_probs
                elif bool(_cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False)):
                    _sparc_mode = str(
                        _cfg_get(m1_cfg, "GEOTOPO_MODE", "full")
                    ).strip().lower()
                    m1_native_probs = (
                        prediction.get("geotopo_geometry_probs", base_probs)
                        if _sparc_mode == "full"
                        else prediction.get("geotopo_base_probs", base_probs)
                    )
                    m1_native_hard_probs = m1_native_probs
                else:
                    m1_native_probs = prediction.get(
                        "v552r4212_m1_native_probability",
                        prediction.get(
                            "v552r4211_m1_native_probability",
                            prediction.get("v552r4210_m1_native_probability", prediction.get("v552r4209_m1_native_probability", base_probs)),
                        ),
                    )
                    m1_native_hard_probs = prediction.get(
                        "v552r4212_m1_native_hard_probability",
                        prediction.get(
                            "v552r4211_m1_native_hard_probability",
                            prediction.get("v552r4210_m1_native_hard_probability", prediction.get("v552r4209_m1_native_hard_probability", m1_native_probs)),
                        ),
                    )
            else:
                seg_samples = model(images, text=batch["text_prompt"], num_samples=effective_num_samples)
                final_probs = torch.sigmoid(seg_samples).mean(dim=0)
                sparc_hr_probs = None
                sparc_m1_hr_probs = None
                sparc_base_hr_probs = None

            final_masks = final_probs > 0.5
            if m1_enabled(cfg) and prediction is not None:
                base_masks = base_probs > 0.5
                changed = final_masks.ne(base_masks)
                per_case_changed = changed.flatten(1).any(dim=1)
                per_case_fraction = changed.float().flatten(1).mean(dim=1)
                deployment_case_count += int(final_masks.shape[0])
                deployment_changed_case_count += int(per_case_changed.sum().item())
                deployment_changed_pixel_fraction_sum += float(per_case_fraction.sum().item())
            # Val-only audit labels: never read or export Test labels.  These
            # per-candidate Dice values make the local text-evidence audit
            # falsifiable before any Test evaluation is allowed.
            # Val-only audit labels: DatasetSegmentation uses
            # ``ground_truth_mask`` (train.py), while older code paths used
            # ``mask``. Resolve both names explicitly so the audit CSV
            # contains per-candidate Val Dice / gain rather than blanks.
            val_candidate_dice = None
            val_candidate_nsd = None
            val_dice_final = None
            val_nsd_final = None
            gt_key = next(
                (key for key in ("ground_truth_mask", "mask", "label", "labels") if key in batch),
                None,
            )
            if str(cfg.split).lower() == "val" and gt_key is not None and prediction is not None:
                gt = batch[gt_key].to(cfg.MODEL.DEVICE).float()
                if gt.ndim == 4:
                    gt = gt[:, 0]
                gt = (gt > 0.5).float()
                cand_hard = (candidate_probs > 0.5).float()
                inter = (cand_hard * gt[:, None]).sum(dim=(-2, -1))
                den = cand_hard.sum(dim=(-2, -1)) + gt[:, None].sum(dim=(-2, -1))
                val_candidate_dice = (2.0 * inter + 1e-4) / (den + 1e-4)
                val_candidate_nsd = _surface_dice_proxy(
                    candidate_probs,
                    gt,
                    tolerance=int(_cfg_get(cfg.M1, "V27_NSD_TOLERANCE_PIXELS", 2)),
                )

                final_hard = (final_probs > 0.5).float()
                final_inter = (final_hard * gt).sum(dim=(-2, -1))
                final_den = final_hard.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
                val_dice_final = (2.0 * final_inter + 1e-4) / (final_den + 1e-4)
                val_nsd_final = _surface_dice_proxy(
                    final_probs,
                    gt,
                    tolerance=int(_cfg_get(cfg.M1, "V27_NSD_TOLERANCE_PIXELS", 2)),
                )[:, 0]
            for index, mask_name in enumerate(batch["mask_name"]):
                save_binary_mask(seg_dir, mask_name, final_masks[index].cpu().numpy())
                if isinstance(sparc_hr_probs, torch.Tensor):
                    _hr_mask = sparc_hr_probs[index]
                    if _hr_mask.ndim == 3 and _hr_mask.shape[0] == 1:
                        _hr_mask = _hr_mask[0]
                    save_binary_mask(
                        sparc_hr_seg_dir, mask_name,
                        (_hr_mask > 0.5).cpu().numpy(),
                    )
                if isinstance(sparc_m1_hr_probs, torch.Tensor):
                    _m1_hr_mask = sparc_m1_hr_probs[index]
                    if _m1_hr_mask.ndim == 3 and _m1_hr_mask.shape[0] == 1:
                        _m1_hr_mask = _m1_hr_mask[0]
                    save_binary_mask(
                        sparc_m1_hr_seg_dir, mask_name,
                        (_m1_hr_mask > 0.5).cpu().numpy(),
                    )
                if isinstance(sparc_base_hr_probs, torch.Tensor):
                    _base_hr_mask = sparc_base_hr_probs[index]
                    if _base_hr_mask.ndim == 3 and _base_hr_mask.shape[0] == 1:
                        _base_hr_mask = _base_hr_mask[0]
                    save_binary_mask(
                        sparc_base_hr_seg_dir, mask_name,
                        (_base_hr_mask > 0.5).cpu().numpy(),
                    )
                if m1_enabled(cfg) and prediction is not None:
                    save_binary_mask(
                        base_native_seg_dir, mask_name,
                        (base_probs[index] > 0.5).cpu().numpy(),
                    )
                    save_binary_mask(
                        m1_native_seg_dir, mask_name,
                        (m1_native_probs[index] > 0.5).cpu().numpy(),
                    )
                    save_binary_mask(
                        m1_native_hard_seg_dir, mask_name,
                        (m1_native_hard_probs[index] > 0.5).cpu().numpy(),
                    )
                    _v5_strength_probs_case = prediction.get("jbt_v5_strength_candidate_probs")
                    if isinstance(_v5_strength_probs_case, torch.Tensor):
                        for _slot, _dir in jbt_v5_strength_dirs.items():
                            if _slot < _v5_strength_probs_case.shape[1]:
                                save_binary_mask(
                                    _dir, mask_name,
                                    (_v5_strength_probs_case[index, _slot] > 0.5).cpu().numpy(),
                                )
                uncertainty = -(
                    final_probs[index] * torch.log(final_probs[index] + 1e-8)
                    + (1.0 - final_probs[index]) * torch.log(1.0 - final_probs[index] + 1e-8)
                )
                uncertainty_map = normalize(uncertainty.cpu().numpy())
                color = (plt.get_cmap("nipy_spectral")(uncertainty_map)[:, :, :3] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(unc_dir, mask_name), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

                if (
                    save_diagnostics
                    and prediction is not None
                    and "v20_action_types" in prediction
                ):
                    names = _v20_candidate_names(
                        prediction["v20_action_types"],
                        candidate_probs.shape[1],
                        prediction=prediction,
                    )
                    save_binary_mask(
                        os.path.join(diagnostics_root, "base"),
                        mask_name,
                        (base_probs[index] > 0.5).cpu().numpy(),
                    )
                    selected = prediction.get("v25_selected_action", prediction.get("v20_selector_hard"))
                    m1_selector_hard_full = prediction.get("m1_selector_hard")
                    for candidate_index, candidate_name in enumerate(names):
                        save_binary_mask(
                            os.path.join(diagnostics_root, f"candidate_{candidate_index}_{candidate_name}"),
                            mask_name,
                            (candidate_probs[index, candidate_index] > 0.5).cpu().numpy(),
                        )
                        # selected is [B, K] — does NOT include Preserve column.
                        # selected_index=0 → Preserve chosen; >0 → action chosen.
                        if candidate_index == 0:
                            if isinstance(m1_selector_hard_full, torch.Tensor):
                                selected_flag = _value_at_slot(prediction, "m1_selector_hard", index, 0, default=0.0)
                            else:
                                sel_idx = prediction.get("v393_selected_index", None)
                                if sel_idx is not None and index < len(sel_idx):
                                    selected_flag = 1.0 if int(sel_idx[index]) == 0 else 0.0
                                else:
                                    selected_flag = 0.0
                            action_index = None
                        else:
                            action_index = candidate_index - 1
                            if isinstance(m1_selector_hard_full, torch.Tensor):
                                selected_flag = _value_at_slot(prediction, "m1_selector_hard", index, candidate_index, default=0.0)
                            else:
                                selected_flag = (
                                    float(selected[index, action_index].detach().float().cpu())
                                    if isinstance(selected, torch.Tensor) else ""
                                )
                        row = {
                            "mask_name": mask_name,
                            "split": cfg.split,
                            "candidate_index": candidate_index,
                            "candidate_name": candidate_name,
                            "action_type": (
                                "preserve" if action_index is None else int(
                                    prediction["v20_action_types"][action_index].detach().cpu()
                                )
                            ),
                            "selected": selected_flag,
                            "val_dsc": (
                                float(val_candidate_dice[index, candidate_index].cpu())
                                if val_candidate_dice is not None else ""
                            ),
                            "val_dsc_gain_vs_base": (
                                float((val_candidate_dice[index, candidate_index] - val_candidate_dice[index, 0]).cpu())
                                if val_candidate_dice is not None else ""
                            ),
                            "val_trainres_nsd": (
                                float(val_candidate_nsd[index, candidate_index].cpu())
                                if val_candidate_nsd is not None else ""
                            ),
                            "val_trainres_nsd_gain_vs_base": (
                                float((val_candidate_nsd[index, candidate_index] - val_candidate_nsd[index, 0]).cpu())
                                if val_candidate_nsd is not None else ""
                            ),
                            "final_val_dsc": float(val_dice_final[index].cpu()) if val_dice_final is not None else "",
                            "final_val_trainres_nsd": float(val_nsd_final[index].cpu()) if val_nsd_final is not None else "",
                            "policy_logit": _value_at(prediction, "v27_policy_logits", index, action_index),
                            "decision_logit": _value_at(prediction, "v26_decision_logits", index, action_index),
                            "utility_benefit_prob": _value_at(prediction, "v25_benefit_prob", index, action_index),
                            "utility_harm_prob": _value_at(prediction, "v25_harm_prob", index, action_index),
                            "utility_margin": _value_at(prediction, "v26_utility_logit_margin", index, action_index),
                            # V455 unified M1 safe-selector audit fields.
                            # These are prediction-only scores; Val DSC/NSD above are diagnostics.
                            "m1_selector_soft": _value_at_slot(prediction, "m1_selector_soft", index, candidate_index),
                            "m1_selector_hard": _value_at_slot(prediction, "m1_selector_hard", index, candidate_index),
                            "m1_choice_logit": _value_at_slot(prediction, "m1_choice_logits", index, candidate_index),
                            "cem_predicted_utility": _value_at_slot(prediction, "cem_quality_utility", index, candidate_index),
                            "cem_benefit_prob": _value_at_slot(prediction, "cem_quality_benefit_prob", index, candidate_index),
                            "cem_harm_prob": _value_at_slot(prediction, "cem_quality_harm_prob", index, candidate_index),
                            "cem_accept_prob": _value_at_slot(prediction, "cem_candidate_accept_prob", index, candidate_index),
                            "cem_selection_score": _value_at_slot(prediction, "cem_selection_scores", index, candidate_index),
                            "cem_predicted_sigma": _value_at_slot(prediction, "cem_quality_sigma", index, candidate_index),
                            "cem_lcb": _value_at_slot(prediction, "cem_selection_lcb", index, candidate_index),
                            "mc_pairwise_disagreement": _value_at(prediction, "mc_pairwise_disagreement", index, default=""),
                            "mc_std_mean": _value_at(prediction, "mc_std_mean", index, default=""),
                            "mc_disagreement_fraction": _value_at(prediction, "mc_disagreement_fraction", index, default=""),
                            "cem_failure_prob": _value_at(prediction, "cem_failure_prob", index, default=""),
                            "cem_accept": _value_at(prediction, "cem_accept", index, default=""),
                            "cem_m3_top_gap": _value_at(prediction, "cem_m3_top_gap", index, default=""),
                            "cem_m3_hard_gate": _value_at(prediction, "cem_m3_hard_gate", index, default=""),
                            "m1_hard_selected_slot": _value_at(prediction, "m1_hard_selected_slot", index, default=""),
                            "m1_preserve_hard": _value_at(prediction, "m1_preserve_hard", index, default=""),
                            "m1_score": _value_at(prediction, "m1_score", index, action_index) if action_index is not None else "",
                            "m1_raw_score": _value_at(prediction, "m1_raw_score", index, action_index) if action_index is not None else "",
                            "m1_action_type_bias": _value_at(prediction, "m1_action_type_bias", index, action_index) if action_index is not None else "",
                            "m1_benefit_logit": _value_at(prediction, "m1_benefit_logits", index, action_index) if action_index is not None else "",
                            "m1_harm_logit": _value_at(prediction, "m1_harm_logits", index, action_index) if action_index is not None else "",
                            "m1_value_delta_dsc": _value_at_component(prediction, "m1_value_delta", index, action_index, 0) if action_index is not None else "",
                            "m1_value_delta_nsd": _value_at_component(prediction, "m1_value_delta", index, action_index, 1) if action_index is not None else "",
                            "m1_valid_action": _value_at(prediction, "m1_valid_action", index, action_index) if action_index is not None else "",
                            "m1_train_valid_action": _value_at(prediction, "m1_train_valid_action", index, action_index) if action_index is not None else "",
                            "m1_pair_valid_action": _value_at(prediction, "m1_pair_valid_action", index, action_index) if action_index is not None else "",
                            "m1_pair_score_used_action": _value_at(prediction, "m1_pair_score_used_action", index, action_index) if action_index is not None else "",
                            "m1_deploy_valid_action": _value_at(prediction, "m1_deploy_valid_action", index, action_index) if action_index is not None else "",
                            "m1_static_deploy_valid_action": _value_at(prediction, "m1_static_deploy_valid_action", index, action_index) if action_index is not None else "",
                            "m1_adaptive_deploy_valid_action": _value_at(prediction, "m1_adaptive_deploy_valid_action", index, action_index) if action_index is not None else "",
                            "m1_adaptive_risk_barrier": _value_at(prediction, "m1_adaptive_risk_barrier", index, action_index) if action_index is not None else "",
                            "m1_adaptive_value_barrier": _value_at(prediction, "m1_adaptive_value_barrier", index, action_index) if action_index is not None else "",
                            "m1_risk_margin": _value_at(prediction, "m1_risk_margin", index, action_index) if action_index is not None else "",
                            "m1_value_utility": _value_at(prediction, "m1_value_utility", index, action_index) if action_index is not None else "",
                            "m1_benefit_probability": _value_at(prediction, "m1_benefit_probability", index, action_index) if action_index is not None else "",
                            "m1_harm_probability": _value_at(prediction, "m1_harm_probability", index, action_index) if action_index is not None else "",
                            "m1_local_area": _value_at(prediction, "m1_local_area", index, action_index) if action_index is not None else "",
                            "valid_control": _value_at(prediction, "v25_valid_control", index, action_index),
                            "raw_control_valid": _value_at(prediction, "v27_raw_control_valid", index, action_index),
                            "context_clean": _value_at(prediction, "v27_context_clean_valid", index, action_index),
                            "context_overlap": _value_at(prediction, "v27_context_overlap", index, action_index),
                            "control_area_ratio": _value_at(prediction, "v25_control_area_ratio", index, action_index),
                            "control_overlap": _value_at(prediction, "v25_control_overlap", index, action_index),
                            "local_action_area": _value_at(prediction, "v27_local_action_area", index, action_index),
                            "pos_delta": _value_at(prediction, "v25_pos_delta", index, action_index),
                            "neg_delta": _value_at(prediction, "v25_neg_delta", index, action_index),
                            "swap_delta": _value_at(prediction, "v25_swap_delta", index, action_index),

                            # V407 FGSR audit fields.  These are prediction-only
                            # values; Val DSC/NSD columns above remain the
                            # diagnostic labels and are never read at Test.
                            "m2_selected_index": _value_at(prediction, "m2_selected_index", index, default=""),
                            "m2_accept": _value_at(prediction, "m2_accept", index, default=""),
                            "m2_gate_probability": _value_at(prediction, "m2_gate_probability", index, default=""),
                            "m2_selected_score": _value_at(prediction, "m2_selected_score", index, default=""),
                            "m2_best_lcb": _value_at(prediction, "m2_best_lcb", index, default=""),
                            "m2_top_lcb_gap": _value_at(prediction, "m2_top_lcb_gap", index, default=""),
                            "m2_gain_mean": _value_at(prediction, "m2_gain_mean", index, action_index) if action_index is not None else "",
                            "m2_gain_std": _value_at(prediction, "m2_gain_std", index, action_index) if action_index is not None else "",
                            "m2_gain_lcb": _value_at(prediction, "m2_gain_lcb", index, action_index) if action_index is not None else "",
                            "m2_action_valid": _value_at(prediction, "m2_action_valid", index, action_index) if action_index is not None else "",
                            "m2_edit_fraction": _value_at(prediction, "m2_edit_fraction", index, action_index) if action_index is not None else "",

                            # Native V393/V395 audit fields.  The old CSV
                            # only wrote V394 aliases, which made V393 q and
                            # eligibility appear as NaN/zero in diagnosis.
                            "v393_action_value": _value_at(prediction, "v393_action_value", index, action_index) if action_index is not None else "",
                            "v393_eligible": _value_at(prediction, "v393_eligible", index, action_index) if action_index is not None else "",
                            "v393_fallback_reason": _string_at(prediction, "v393_fallback_reason", index, default="unknown"),
                            "v395_direct_gain_value": _value_at(prediction, "v395_direct_gain_value", index, action_index) if action_index is not None else "",
                            "v395_selected_value": _value_at(prediction, "v395_selected_value", index, default=""),
                            "v395_eligible": _value_at(prediction, "v395_eligible", index, action_index) if action_index is not None else "",
                            "v396_action_value": _value_at(prediction, "v396_action_value", index, action_index) if action_index is not None else "",
                            "v396_visual_gain": _value_at(prediction, "v396_visual_gain", index, action_index) if action_index is not None else "",
                            "v396_text_evidence": _value_at(prediction, "v396_text_evidence", index, action_index) if action_index is not None else "",
                            "v396_paraphrase_evidence": _value_at(prediction, "v396_paraphrase_evidence", index, action_index) if action_index is not None else "",
                            "v396_swap_evidence": _value_at(prediction, "v396_swap_evidence", index, action_index) if action_index is not None else "",
                            "v396_semantic_gate": _value_at(prediction, "v396_semantic_gate", index, action_index) if action_index is not None else "",
                            "v396_visual_gap": _value_at(prediction, "v396_visual_gap", index, action_index) if action_index is not None else "",
                            "v396_visual_cosine": _value_at(prediction, "v396_visual_cosine", index, action_index) if action_index is not None else "",
                            "v396_action_area": _value_at(prediction, "v396_action_area", index, action_index) if action_index is not None else "",
                            "v396_parent_matched_control": _value_at(prediction, "v396_control_source_parent", index, action_index) if action_index is not None else "",
                            "v396_eligible": _value_at(prediction, "v396_eligible", index, action_index) if action_index is not None else "",
                            "v396_selected_value": _value_at(prediction, "v396_selected_value", index, default=""),
                            "v396_observer_frozen": _value_at(prediction, "v396_observer_frozen", index, default=""),
                            "v396_paraphrase_available": _value_at(prediction, "v396_paraphrase_available", index, default=""),
                            "v396_swap_from_batch": _value_at(prediction, "v396_swap_from_batch", index, default=""),

                            "v394_value": _value_at(prediction, "v394_value", index, action_index) if action_index is not None else "",
                            "v394_value_logit": _value_at(prediction, "v394_value_logit", index, action_index) if action_index is not None else "",
                            "v394_benefit_logit": _value_at(prediction, "v393_benefit_logit", index, action_index) if action_index is not None else "",
                            "v394_harm_logit": _value_at(prediction, "v393_harm_logit", index, action_index) if action_index is not None else "",
                            "v394_text_evidence_logit": _value_at(prediction, "v393_text_evidence_logit", index, action_index) if action_index is not None else "",
                            "v394_eligible": _value_at(prediction, "v394_eligible", index, action_index) if action_index is not None else "",
                            "v394_action_area": _value_at(prediction, "v394_action_area", index, action_index) if action_index is not None else "",
                            "v394_control_valid": _value_at(prediction, "v393_control_valid", index, action_index) if action_index is not None else "",
                            "v394_control_area_ratio": _value_at(prediction, "v393_control_area_ratio", index, action_index) if action_index is not None else "",
                            "v394_control_overlap": _value_at(prediction, "v393_control_overlap", index, action_index) if action_index is not None else "",
                            "v394_context_overlap": _value_at(prediction, "v393_context_overlap", index, action_index) if action_index is not None else "",
                            "v394_selected_value": _value_at(prediction, "v394_selected_value", index, default="") if action_index is not None else "",
                        }
                        # V485 error-state causal audit fields.
                        row.update({
                            "v485_p_no_edit": _value_at(prediction, "p_no_edit", index, default=""),
                            "v485_p_fp": _value_at(prediction, "p_fp", index, default=""),
                            "v485_p_fn": _value_at(prediction, "p_fn", index, default=""),
                            "v485_p_boundary": _value_at(prediction, "p_boundary", index, default=""),
                            "v485_p_failure": _value_at(prediction, "p_failure", index, default=""),
                            "v485_selected_index": _value_at(prediction, "selected_index", index, default=""),
                            "v485_accepted": _value_at(prediction, "accepted", index, default=""),
                            "v485_local_soft_gate": _value_at(prediction, "local_soft_gates", index, action_index) if action_index is not None and action_index < 4 else "",
                            "v485_local_pred_delta_dsc": _value_at(prediction, "local_m2_local_pred_delta_dsc", index, action_index) if action_index is not None and action_index < 4 else "",
                            "v485_local_pred_harm_prob": _value_at(prediction, "local_m2_local_pred_harm_prob", index, action_index) if action_index is not None and action_index < 4 else "",
                        })
                        v27_action_audit_rows.append(row)
                    save_binary_mask(
                        os.path.join(diagnostics_root, str(_cfg_get(cfg.M1, "INFERENCE_MODE", "preserve"))),
                        mask_name,
                        (fused_probs[index] > 0.5).cpu().numpy(),
                    )

                if (
                    save_diagnostics
                    and prediction is not None
                    and "v20_action_types" not in prediction
                ):
                    save_binary_mask(os.path.join(diagnostics_root, "base"), mask_name, (base_probs[index] > 0.5).cpu().numpy())
                    for candidate_index, candidate_name in enumerate(("preserve", "shrink", "expand")):
                        save_binary_mask(
                            os.path.join(diagnostics_root, f"candidate_{candidate_index}_{candidate_name}"),
                            mask_name,
                            (candidate_probs[index, candidate_index] > 0.5).cpu().numpy(),
                        )
                    save_binary_mask(
                        os.path.join(diagnostics_root, str(_cfg_get(cfg.M1, "INFERENCE_MODE", "preserve"))),
                        mask_name,
                        (fused_probs[index] > 0.5).cpu().numpy(),
                    )
                    diagnostic_rows.append({
                        "mask_name": mask_name,
                        "final_role": str(_cfg_get(cfg.M1, "INFERENCE_MODE", "preserve")),
                        "val_dice_preserve": float(val_candidate_dice[index, 0].cpu()) if val_candidate_dice is not None else "",
                        "val_dice_shrink": float(val_candidate_dice[index, 1].cpu()) if val_candidate_dice is not None else "",
                        "val_dice_expand": float(val_candidate_dice[index, 2].cpu()) if val_candidate_dice is not None else "",
                        "val_dice_final": float(val_dice_final[index].cpu()) if val_dice_final is not None else "",
                        "val_gain_shrink": float((val_candidate_dice[index, 1] - val_candidate_dice[index, 0]).cpu()) if val_candidate_dice is not None else "",
                        "val_gain_expand": float((val_candidate_dice[index, 2] - val_candidate_dice[index, 0]).cpu()) if val_candidate_dice is not None else "",
                        "candidate_std": float(prediction["candidate_std"][index].cpu()),
                        "candidate_mean_abs_change": float(prediction["candidate_mean_abs_change"][index].cpu()),
                        "shrink_mean_abs_change": float(prediction["shrink_mean_abs_change"][index].cpu()),
                        "expand_mean_abs_change": float(prediction["expand_mean_abs_change"][index].cpu()),
                        "edit_gate_mean": float(prediction["edit_gate_mean"][index].cpu()),
                        "edit_band_mean": float(prediction["edit_band_mean"][index].cpu()),
                        "inside_band_mean": float(prediction["inside_band_mean"][index].cpu()),
                        "outside_band_mean": float(prediction["outside_band_mean"][index].cpu()),
                        "shrink_gate_mean": float(prediction["shrink_gate_mean"][index].cpu()),
                        "expand_gate_mean": float(prediction["expand_gate_mean"][index].cpu()),
                        "shrink_changed_fraction": float(prediction["shrink_changed_fraction"][index].cpu()),
                        "expand_changed_fraction": float(prediction["expand_changed_fraction"][index].cpu()),
                        "fusion_changed_fraction": float(prediction["fusion_changed_fraction"][index].cpu()),
                        "router_preserve_mean": float(prediction.get("router_preserve_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_shrink_mean": float(prediction.get("router_shrink_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_expand_mean": float(prediction.get("router_expand_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_edit_probability_mean": float(prediction.get("router_edit_probability_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_edit_active_fraction": float(prediction.get("router_edit_active_fraction", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_direction_shrink_mean": float(prediction.get("router_direction_shrink_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_direction_expand_mean": float(prediction.get("router_direction_expand_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_shrink_weight_mean": float(prediction.get("router_shrink_weight_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "router_expand_weight_mean": float(prediction.get("router_expand_weight_mean", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "text_verifier_preserve_weight": float(prediction.get("text_verifier_preserve_weight", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "text_verifier_shrink_weight": float(prediction.get("text_verifier_shrink_weight", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "text_verifier_expand_weight": float(prediction.get("text_verifier_expand_weight", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "text_verifier_score_margin": float(prediction.get("text_verifier_score_margin", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_selected_index": int(prediction.get("m2_selected_index", base_probs.new_zeros(base_probs.shape[0], dtype=torch.long))[index].cpu()),
                        "m2_selected_gain": float(prediction.get("m2_selected_gain", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_selected_risk": float(prediction.get("m2_selected_risk", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_selected_confidence": float(prediction.get("m2_selected_confidence", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_selected_score": float(prediction.get("m2_selected_score", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_accept": float(prediction.get("m2_accept", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_gain_shrink": float(prediction.get("m2_pred_gain_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_gain_expand": float(prediction.get("m2_pred_gain_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_risk_shrink": float(prediction.get("m2_pred_risk_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_risk_expand": float(prediction.get("m2_pred_risk_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_confidence_shrink": float(prediction.get("m2_pred_confidence_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_pred_confidence_expand": float(prediction.get("m2_pred_confidence_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_edit_area_shrink": float(prediction.get("m2_edit_area_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_edit_area_expand": float(prediction.get("m2_edit_area_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_add_area_shrink": float(prediction.get("m2_add_area_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_add_area_expand": float(prediction.get("m2_add_area_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_remove_area_shrink": float(prediction.get("m2_remove_area_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_remove_area_expand": float(prediction.get("m2_remove_area_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m3_score_shrink": float(prediction.get("m3_score_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m3_score_expand": float(prediction.get("m3_score_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_support_shrink": float(prediction.get("m2_support_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_support_expand": float(prediction.get("m2_support_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_contradiction_shrink": float(prediction.get("m2_contradiction_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_contradiction_expand": float(prediction.get("m2_contradiction_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_overseg_shrink": float(prediction.get("m2_overseg_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_overseg_expand": float(prediction.get("m2_overseg_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_underseg_shrink": float(prediction.get("m2_underseg_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_underseg_expand": float(prediction.get("m2_underseg_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_boundary_shrink": float(prediction.get("m2_boundary_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_boundary_expand": float(prediction.get("m2_boundary_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_connectivity_expand": float(prediction.get("m2_connectivity_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_validity_shrink": float(prediction.get("m2_validity_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_validity_expand": float(prediction.get("m2_validity_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_benefit_shrink": float(prediction.get("m2_benefit_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_benefit_expand": float(prediction.get("m2_benefit_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_harm_shrink": float(prediction.get("m2_harm_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_harm_expand": float(prediction.get("m2_harm_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_uncertainty_shrink": float(prediction.get("m2_uncertainty_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_uncertainty_expand": float(prediction.get("m2_uncertainty_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_alignment_shrink": float(prediction.get("m2_text_alignment_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_alignment_expand": float(prediction.get("m2_text_alignment_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_necessity_shrink": float(prediction.get("m2_text_necessity_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_necessity_expand": float(prediction.get("m2_text_necessity_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_edit_direction_shrink": float(prediction.get("m2_text_edit_direction_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_edit_direction_expand": float(prediction.get("m2_text_edit_direction_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_specificity_shrink": float(prediction.get("m2_text_specificity_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_specificity_expand": float(prediction.get("m2_text_specificity_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_keep_positive_shrink": float(prediction.get("m2_keep_positive_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_drop_positive_shrink": float(prediction.get("m2_drop_positive_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_text_pass_shrink": float(prediction.get("m2_text_pass_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_consensus_shrink": float(prediction.get("m2_consensus_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_cf_available_shrink": float(prediction.get("m2_cf_available_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_cf_available_expand": float(prediction.get("m2_cf_available_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_control_area_ratio_shrink": float(prediction.get("m2_dense_control_area_ratio_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_control_area_ratio_expand": float(prediction.get("m2_dense_control_area_ratio_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_control_overlap_ratio_shrink": float(prediction.get("m2_dense_control_overlap_ratio_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_control_overlap_ratio_expand": float(prediction.get("m2_dense_control_overlap_ratio_expand", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_quality_preserve_shrink": float(prediction.get("m2_dense_quality_preserve_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_quality_candidate_shrink": float(prediction.get("m2_dense_quality_candidate_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_quality_control_shrink": float(prediction.get("m2_dense_quality_control_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_control_delta_shrink": float(prediction.get("m2_dense_control_delta_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_positive_candidate_shrink": float(prediction.get("m2_dense_positive_candidate_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_negative_candidate_shrink": float(prediction.get("m2_dense_negative_candidate_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "m2_dense_gt_gap_shrink": float(prediction.get("m2_dense_gt_gap_shrink", base_probs.new_zeros(base_probs.shape[0]))[index].cpu()),
                        "verifier_text_override": str(getattr(cfg, "verifier_text_override", "")),
                    })

    if m1_enabled(cfg) and str(getattr(cfg, "export_mode", "joint")) != "base":
        logger.info(
            "GEOTR-M1 paired masks saved: base=%s m1=%s hard=%s",
            base_native_seg_dir, m1_native_seg_dir, m1_native_hard_seg_dir,
        )
        if bool(_cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False)):
            logger.info(
                "SPARC 448 masks saved: Base=%s M1=%s M2=%s",
                sparc_base_hr_seg_dir, sparc_m1_hr_seg_dir, sparc_hr_seg_dir,
            )

    if m1_enabled(cfg) and deployment_case_count > 0:
        logger.info(
            "Deployment change summary: changed_cases=%d/%d (%.6f), mean_changed_pixel_fraction=%.8f",
            deployment_changed_case_count,
            deployment_case_count,
            deployment_changed_case_count / float(deployment_case_count),
            deployment_changed_pixel_fraction_sum / float(deployment_case_count),
        )

    if save_diagnostics:
        os.makedirs(diagnostics_root, exist_ok=True)
        csv_path = os.path.join(diagnostics_root, "m1_pse.csv")
        fields = [
            "mask_name", "final_role", "val_dice_preserve", "val_dice_shrink", "val_dice_expand", "val_dice_final", "val_gain_shrink", "val_gain_expand", "candidate_std", "candidate_mean_abs_change",
            "shrink_mean_abs_change", "expand_mean_abs_change", "edit_gate_mean",
            "edit_band_mean", "inside_band_mean", "outside_band_mean",
            "shrink_gate_mean", "expand_gate_mean", "shrink_changed_fraction",
            "expand_changed_fraction", "fusion_changed_fraction",
            "router_preserve_mean", "router_shrink_mean", "router_expand_mean",
            "router_edit_probability_mean", "router_edit_active_fraction",
            "router_direction_shrink_mean", "router_direction_expand_mean",
            "router_shrink_weight_mean", "router_expand_weight_mean",
            "text_verifier_preserve_weight", "text_verifier_shrink_weight",
            "text_verifier_expand_weight", "text_verifier_score_margin",
            "m2_selected_index", "m2_selected_gain", "m2_selected_risk", "m2_selected_confidence",
            "m2_selected_score", "m2_accept", "m2_pred_gain_shrink", "m2_pred_gain_expand",
            "m2_pred_risk_shrink", "m2_pred_risk_expand", "m2_pred_confidence_shrink", "m2_pred_confidence_expand",
            "m2_edit_area_shrink", "m2_edit_area_expand", "m2_add_area_shrink", "m2_add_area_expand",
            "m2_remove_area_shrink", "m2_remove_area_expand",
            "m3_score_shrink", "m3_score_expand",
            "m2_support_shrink", "m2_support_expand",
            "m2_contradiction_shrink", "m2_contradiction_expand",
            "m2_overseg_shrink", "m2_overseg_expand",
            "m2_underseg_shrink", "m2_underseg_expand",
            "m2_boundary_shrink", "m2_boundary_expand",
            "m2_connectivity_expand", "m2_validity_shrink", "m2_validity_expand",
            "m2_benefit_shrink", "m2_benefit_expand", "m2_harm_shrink", "m2_harm_expand",
            "m2_uncertainty_shrink", "m2_uncertainty_expand", "m2_text_alignment_shrink", "m2_text_alignment_expand",
            "m2_text_necessity_shrink", "m2_text_necessity_expand", "m2_text_edit_direction_shrink", "m2_text_edit_direction_expand", "m2_text_specificity_shrink", "m2_text_specificity_expand",
            "m2_keep_positive_shrink", "m2_drop_positive_shrink", "m2_text_pass_shrink", "m2_consensus_shrink",
            "m2_cf_available_shrink", "m2_cf_available_expand", "m2_dense_control_area_ratio_shrink", "m2_dense_control_area_ratio_expand",
            "m2_dense_control_overlap_ratio_shrink", "m2_dense_control_overlap_ratio_expand",
            "m2_dense_quality_preserve_shrink", "m2_dense_quality_candidate_shrink", "m2_dense_quality_control_shrink", "m2_dense_control_delta_shrink",
            "m2_dense_positive_candidate_shrink", "m2_dense_negative_candidate_shrink", "m2_dense_gt_gap_shrink", "verifier_text_override",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        logger.info("M1 P/S/E diagnostics written to: %s", diagnostics_root)
        if v27_action_audit_rows:
            audit_path = os.path.join(diagnostics_root, "v27_action_audit.csv")
            audit_fields = sorted({key for row in v27_action_audit_rows for key in row})
            with open(audit_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=audit_fields)
                writer.writeheader()
                writer.writerows(v27_action_audit_rows)
            logger.info("V27 dynamic action audit written to: %s", audit_path)


if __name__ == "__main__":
    main()
