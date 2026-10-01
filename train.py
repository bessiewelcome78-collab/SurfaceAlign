"""/home/tsz-25/MedCLIPSeg-pristine/train.py
V470 fair joint end-to-end training for trainable MedCLIPSeg components.

The pretrained UniMedCLIP/BiomedBERT encoders follow the original baseline
protocol and remain frozen, while the task-specific segmentation path, online
Preserve prediction, multi-candidate generator, causal gate, and CCV selector
are initialized in one run and optimized jointly.  No task-trained B0 checkpoint
is loaded.  Validation is used only for pre-declared checkpoint selection; Test
remains unopened until the final one-shot evaluation.

V390 additions:
  - SAM (Sharpness-Aware Minimization, Foret et al. ICLR 2021): finds flat minima.
  - EMA (Exponential Moving Average): stable inference, proven in BYOL/DINO.
  - R-Drop (Liang et al. NeurIPS 2021): KL consistency between two MC-dropout passes.
"""
import argparse
import copy
import gc
import hashlib
import logging
import math
import os
import random
import re
from contextlib import contextmanager
from statistics import mean

import monai
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from tqdm import tqdm

# AUTO_PARAM_ADAPTER_IMPORT_BEGIN
from utils.auto_param_adapter import AutoParamAdapter
# AUTO_PARAM_ADAPTER_IMPORT_END
from utils.text_prompted_hypothesis_loss import compute_text_prompted_hypothesis_loss
from utils.multi_hypothesis_composition_loss import (
    compute_multi_hypothesis_composition_loss,
    compute_geotr_v4c_validation_diagnostics,
    compute_geotr_v4d_validation_diagnostics,
    compute_geotr_v4e_validation_diagnostics,
    compute_geotr_v4f_validation_diagnostics,
    compute_geotr_v4g_validation_diagnostics,
    _joint_oracle_envelope,
    _target_3d,
)
from utils.semlt_loss import compute_semlt_loss
from utils.geotr_m1_loss import compute_geotr_m1_loss
from utils.jbtl_rbal_loss import (
    compute_rbal_loss,
    compute_edge_alignment_loss,
    compute_normal_margin_loss,
    compute_boundary_loss,
    compute_hausdorff_dt_loss,
    compute_active_contour_loss,
)
from utils.semlt_autozero_loss import compute_semlt_autozero_loss
from utils.geotr_m1_protocol import validate_geotr_m1_checkpoint_protocol
from utils.metrics_2d import case_metrics_2d
from utils.gradient_isolation import backward_named_prefix_only


# ---------------------------------------------------------------------------
# SAM (Sharpness-Aware Minimization) – ICLR 2021 / MICCAI 2023-2024
# ---------------------------------------------------------------------------
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently
    Improving Generalization", ICLR 2021.

    Medical-imaging validation: widely adopted at MICCAI 2023/2024 for
    reducing the train->validation->test generalization gap.
    """

    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        defaults = {"rho": rho, "adaptive": adaptive}
        super().__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step: w + eps * sign(g) to find worst-case neighborhood."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "old_p" not in state:
                    state["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descent step: restore w, then apply gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state["old_p"])
                del state["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires explicit first_step() / second_step() calls."
        )

    def _grad_norm(self):
        norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norm_sq += p.grad.data.pow(2).sum()
        return norm_sq.pow(0.5)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


class _JBTDualOptimizer:
    """Two physical Adam optimizers with one training-loop facade.

    The Base optimizer is constructed *exactly* like the released MedCLIPSeg
    optimizer (one parameter list, default Adam execution policy).  JBT owns a
    completely separate Adam instance.  This prevents auxiliary parameter-group
    layout or foreach batching from perturbing the protected Base numerical
    trajectory while keeping both branches updated in the same mini-batch/run.
    """
    def __init__(self, base_optimizer, aux_optimizer):
        self.base_optimizer = base_optimizer
        self.aux_optimizer = aux_optimizer
        self.param_groups = list(base_optimizer.param_groups) + list(aux_optimizer.param_groups)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        self.aux_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.base_optimizer.step()
        self.aux_optimizer.step()

    def state_dict(self):
        return {
            "jbt_dual_optimizer": True,
            "base": self.base_optimizer.state_dict(),
            "aux": self.aux_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and state_dict.get("jbt_dual_optimizer", False):
            self.base_optimizer.load_state_dict(state_dict["base"])
            self.aux_optimizer.load_state_dict(state_dict["aux"])
            return
        raise ValueError("JBT dual optimizer checkpoint is missing dual state")


class _JBTDualScheduler:
    def __init__(self, base_scheduler, aux_scheduler):
        self.base_scheduler = base_scheduler
        self.aux_scheduler = aux_scheduler

    def step(self):
        self.base_scheduler.step()
        self.aux_scheduler.step()

    def state_dict(self):
        return {
            "jbt_dual_scheduler": True,
            "base": self.base_scheduler.state_dict(),
            "aux": self.aux_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and state_dict.get("jbt_dual_scheduler", False):
            self.base_scheduler.load_state_dict(state_dict["base"])
            self.aux_scheduler.load_state_dict(state_dict["aux"])
            return
        raise ValueError("JBT dual scheduler checkpoint is missing dual state")


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) – validated in BYOL, DINO, medical SSL
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponentially moving average of model weights.

    Proven to stabilise inference and reduce variance in self-supervised
    and medical segmentation literature (MICCAI 2023-2024).
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    @contextmanager
    def apply_ema(self):
        """Context manager: temporarily apply EMA weights to model, restore after."""
        self._backup_current()
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            self._restore_backup()

    def _backup_current(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()

    def _restore_backup(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup.pop(name))

    def apply_to(self, model):
        """Copy EMA weights into a different model instance (for checkpoint saving)."""
        named_params = dict(self.model.named_parameters())
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# R-Drop consistency loss (Liang et al., NeurIPS 2021)
# ---------------------------------------------------------------------------
def rdrop_kl_loss(logits1, logits2, temperature=2.0):
    """Symmetric KL divergence between two MC-dropout forward passes."""
    p = F.log_softmax(logits1 / temperature, dim=-1)
    q = F.softmax(logits2 / temperature, dim=-1)
    kl_pq = F.kl_div(p, q.detach(), reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(
        F.log_softmax(logits2 / temperature, dim=-1),
        F.softmax(logits1 / temperature, dim=-1).detach(),
        reduction="batchmean",
        log_target=False,
    )
    return (kl_pq + kl_qp) / 2.0

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from utils.slr_paired_hr_dataset import SLRPairedResolutionDataset
from utils.sparc_hr_loss import compute_sparc_hr_validation_diagnostics
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text
# Historical loss modules are loaded lazily.
# This repository is intentionally pruned: deleted legacy V18/V19/... modules
# must not prevent a V383 run from starting.
import importlib

from utils.candidate_edit_control_loss import (
    compute_candidate_edit_control_loss,
    compute_evidence_guided_candidate_loss,
    compute_reference_candidate_loss,
    compute_safe_residual_candidate_loss,
)
from utils.candidate_calibration_loss import (
    compute_candidate_calibration_loss,
)
from utils.candidate_consensus_loss import (
    compute_candidate_consensus_loss,
)
from utils.unified_m1_safe_loss import (
    compute_unified_m1_safe_fusion_loss,
)
from utils.v463_residual_ccv_loss import compute_v463_residual_ccv_joint_loss
from utils.v547_hard_case_memory import V547HardCaseMemory
from utils.v548_stable_routing import stable_m2_effective_weight
from utils.v547_semantic_safe_augment import apply_v547_semantic_safe_augmentation

# Historical function aliases retained only for old dispatcher compatibility.
# Actual source files use the semantic candidate_* names above.
compute_m1_v393_preserve_aware_edit_control_loss = (
    compute_candidate_edit_control_loss
)
compute_m1_v396_fecg_loss = (
    compute_evidence_guided_candidate_loss,
    compute_reference_candidate_loss
)

class _MissingLegacyLoss:
    def __init__(self, module_name: str, symbol_name: str):
        self.module_name = module_name
        self.symbol_name = symbol_name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Legacy loss '{self.symbol_name}' was requested, but "
            f"'{self.module_name}' is absent from this pruned project. "
            "Use the currently supported V383 configuration, or restore that "
            "specific historical module before running its historical config."
        )



# V410-only cleanup: historical non-V396 branches intentionally unavailable.
compute_m1_v383_conservative_action_value_loss = _MissingLegacyLoss(
    "utils.m1_v383_conservative_action_value_loss",
    "compute_m1_v383_conservative_action_value_loss",
)
compute_m1_v394_counterfactual_value_rank_loss = _MissingLegacyLoss(
    "utils.m1_v394_counterfactual_value_rank_loss",
    "compute_m1_v394_counterfactual_value_rank_loss",
)
compute_m2_tide_action_loss = _MissingLegacyLoss(
    "utils.m2_tide_action_loss",
    "compute_m2_tide_action_loss",
)

def _lazy_legacy_loss(module_name: str, symbol_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only suppress the explicitly absent historical loss file itself.
        # Missing dependencies inside an existing module must still surface.
        if exc.name == module_name:
            return _MissingLegacyLoss(module_name, symbol_name)
        raise

    return getattr(module, symbol_name)


compute_trainable_pse_loss = _MissingLegacyLoss(
    "utils.m1_pse_train_loss",
    "compute_trainable_pse_loss",
)
compute_m1_v18_atomic_loss = _MissingLegacyLoss(
    "utils.m1_v18_atomic_loss",
    "compute_m1_v18_atomic_loss",
)
compute_m1_v19_action_loss = _MissingLegacyLoss(
    "utils.m1_v19_action_loss",
    "compute_m1_v19_action_loss",
)
compute_m1_v20_unified_loss = _MissingLegacyLoss(
    "utils.m1_v20_unified_loss",
    "compute_m1_v20_unified_loss",
)
compute_m1_v25_type_conditional_utility_loss = _MissingLegacyLoss(
    "utils.m1_v25_type_conditional_utility_loss",
    "compute_m1_v25_type_conditional_utility_loss",
)
compute_m1_v31_competitive_utility_loss = _MissingLegacyLoss(
    "utils.m1_v31_competitive_utility_loss",
    "compute_m1_v31_competitive_utility_loss",
)
compute_m1_v32_island_phaseb_loss = _MissingLegacyLoss(
    "utils.m1_v32_island_phaseb_loss",
    "compute_m1_v32_island_phaseb_loss",
)
compute_m1_v33_signed_gain_regression_loss = _MissingLegacyLoss(
    "utils.m1_v33_signed_gain_regression_loss",
    "compute_m1_v33_signed_gain_regression_loss",
)
compute_m1_v34_spatial_quantile_world_loss = _MissingLegacyLoss(
    "utils.m1_v34_spatial_quantile_world_loss",
    "compute_m1_v34_spatial_quantile_world_loss",
)
compute_m1_v35_residual_purified_world_loss = _MissingLegacyLoss(
    "utils.m1_v35_residual_purified_world_loss",
    "compute_m1_v35_residual_purified_world_loss",
)
compute_m1_v36_casewise_plackett_luce_loss = _MissingLegacyLoss(
    "utils.m1_v36_casewise_plackett_luce_loss",
    "compute_m1_v36_casewise_plackett_luce_loss",
)
compute_m1_v37_text_falsified_structural_consensus_loss = _MissingLegacyLoss(
    "utils.m1_v37_text_falsified_structural_consensus_loss",
    "compute_m1_v37_text_falsified_structural_consensus_loss",
)
# Renamed source file; old function variable retained for legacy configs.
compute_m1_v38_casewise_falsified_delta_consensus_loss = (
    compute_candidate_consensus_loss
)
# Renamed source file; old function variable retained for legacy configs.
compute_m1_v381_lesion_background_calibrated_atomic_loss = (
    compute_candidate_calibration_loss
)


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _legacy_version_token_in_filename(filename: str, version: str) -> bool:
    """Match legacy experiment version tokens without colliding with PC2R names.

    Historical configs use standalone tokens such as ``V31_...`` or
    ``BUSI_V32_...``.  New GEOTR-PC2R names such as ``PC2RV31`` and
    ``PC2RV32`` embed the same character sequence inside a larger token and
    must never activate the old V31/V32 compatibility router.
    """
    stem = os.path.splitext(os.path.basename(str(filename)).lower())[0]
    normalized = stem.replace('-', '_').replace('.', '_')
    tokens = tuple(token for token in normalized.split('_') if token)
    return str(version).strip().lower() in tokens


def m1_enabled(cfg):
    return bool(_cfg_get(_cfg_get(cfg, "M1", None), "ENABLED", False))


def _tc_drcs(cfg):
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() == "tc_drcs"
    )


def _mhcs(cfg):
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and (
            str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() == "mhcs"
            or str(_cfg_get(m1, "CANDIDATE_MODE", "")).strip().lower()
            in {"mhcs", "multi_hypothesis_composition"}
        )
    )


def _semlt(cfg):
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and (
            str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() in {"semlt", "geotr_m1", "semlt_autozero"}
            or str(_cfg_get(m1, "CANDIDATE_MODE", "")).strip().lower()
            in {"semlt_logit_transport", "exact_geometry_transport", "semlt_autozero_transport"}
        )
    )


def _semlt_main_e2e100(cfg):
    """True for the end-to-end SemLT main-table routing contract.

    The Base/PVL path and SemLT are optimized during the same formal run.
    SemLT observes detached Base logits/conditioners inside
    ``ExactGeometryTransportSegmenter``, so the SemLT objective cannot alter
    Base/PVL gradients. This isolates the refinement inductive bias without
    granting the proposed method an extra 100-epoch post-training budget.
    """
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        _semlt(cfg)
        and (
            (str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() == "geotr_m1"
             and bool(_cfg_get(m1, "GEOTR_M1_MAIN_E2E100", False)))
            or
            (str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() == "semlt_autozero"
             and bool(_cfg_get(m1, "SEMLT_LST_V3_MAIN", False)))
        )
    )


def _geotr_m1_causal_ablation(cfg):
    """True only for the declared common-Base causal ablation protocol.

    This protocol is intentionally separate from the from-scratch E2E result:
    it loads one Base/PVL checkpoint, freezes it exactly, and optimizes only the
    M1 transport.  Consequently every variant is evaluated against identical
    Base predictions and a difference between variants is attributable to the
    changed M1 mechanism rather than to a different Base training trajectory.
    """
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        _semlt(cfg)
        and str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() == "geotr_m1"
        and bool(_cfg_get(m1, "GEOTR_M1_CAUSAL_ABLATION", False))
    )


def _clean_dynamic_component_set(cfg):
    """Shared clean-family outer routing (legacy CLEAN + TC-DRCS)."""
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and str(_cfg_get(m1, "PROTOCOL", "")).strip().lower()
        in {"clean_dynamic_component_set", "tc_drcs"}
    )


def m1_uses_unified_action_cf(cfg):
    """True for every action-bank/mechanism/TPMHG route, not only legacy V20.

    SemLT-LST v3.1 Root-Fix is intentionally *not* an action-bank route.
    A historical experiment family also used the token ``V31``.  If that
    legacy flag leaks into the Root-Fix config, the generic MC trainer can
    retain four full Base/PVL graphs and exhaust a 48-GB GPU.  Force the
    Root-Fix onto its intended single-forward SemLT path.
    """
    if not m1_enabled(cfg):
        return False
    m1 = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
        return False
    mode = str(_cfg_get(m1, "CANDIDATE_MODE", "")).strip().lower()
    return bool(
        _cfg_get(m1, "V20_UNIFIED_ACTION_CF", False)
        or _cfg_get(m1, "ACTION_BANK_UNIFIED_ACTION_CF", False)
        or mode in {
            "mechanism",
            "action_bank",
            "text_prompted_hypothesis",
            "tpmhg",
            "v452_tpmhg",
            "compositional_error_modes",
            "cem_candidates",
            "v474_cem",
            "v484_error_state_causal",
            "v485_error_state_causal",
            "v485_v484only_m1local",
            "v486_fixedbase_m1local_repair",
            "v487_basesafe_gated_repair",
            "v531_typed_sparse_refiner",
            "v532_unified_sparse_refiner",
            "clean_dynamic_component_set",
            "tc_drcs",
            "semlt_logit_transport",
            "exact_geometry_transport",
            "mhcs",
            "multi_hypothesis_composition",
        }
    )


def _activate_v25_runtime_compat(cfg, config_file):
    """Activate V25/V26/V27 utility-bank defaults without moving data splits.

    V27 deliberately reuses the stable V25 loss dispatcher and candidate-bank
    flag.  Its distinct behavior is selected solely by ``RUN_TAG=V27_*``.
    """
    filename = os.path.basename(str(config_file)).lower()
    m1 = _cfg_get(cfg, "M1", None)
    if m1 is None:
        return cfg

    # SemLT-LST v3.1 Root-Fix is unrelated to the historical candidate-policy
    # experiment named "V31".  The human-readable config token V31 (= v3.1)
    # must never activate the old compatibility router, which otherwise injects
    # M1_TRAIN_NUM_SAMPLES=4 and retains four full trainable Base/PVL graphs.
    if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
        return cfg

    dataset = _cfg_get(cfg, "DATASET", None)
    policy_keys = (
        "V25_TYPE_CONDITIONAL_UTILITY_BANK",
        "V25_HIDDEN_DIM",
        "V25_TYPE_EMBED_DIM",
        "V25_BENEFIT_MARGIN",
        "V25_HARM_MARGIN",
        "V25_BENEFIT_MIN",
        "V25_HARM_MAX",
        "V25_UTILITY_MARGIN",
        "V25_SELECTOR_MARGIN",
        "V25_MAX_ACTIONS",
        "V26_UTILITY_DECISION_WEIGHT",
        "V26_POSITIVE_CASE_WEIGHT",
        "V26_NULL_MARGIN",
        "V26_NULL_SAFETY_WEIGHT",
        "M1_TRAIN_NUM_SAMPLES",
        "V27_DSC_BENEFIT_MARGIN",
        "V27_DSC_HARM_MARGIN",
        "V27_NSD_BENEFIT_FLOOR",
        "V27_NSD_HARM_MARGIN",
        "V27_NSD_TOLERANCE_PIXELS",
        "V27_DSC_UTILITY_WEIGHT",
        "V27_NSD_UTILITY_WEIGHT",
        "V27_CASE_RANK_WEIGHT",
        "V27_CASE_RANK_MARGIN",
        "V27_NULL_MARGIN",
        "V27_CONTROL_CONTEXT_GUARD",
    )
    for key in policy_keys:
        if _cfg_get(m1, key, None) is None:
            value = _cfg_get(dataset, key, None)
            if value is not None:
                setattr(m1, key, value)

    requested_tag = str(_cfg_get(m1, "RUN_TAG", "")).upper()


    # V383 is the conservative repair of V382: the same action-conditional
    # transition value model now requires positive q10 plus calibrated harmful-
    # outcome and benefit evidence before it can beat Preserve=0.
    v383_requested = (
        _legacy_version_token_in_filename(filename, "v383")
        or requested_tag.startswith("V383_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v383_conservative_action_value"
    )
    if v383_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V383_CONSERVATIVE_ACTION_VALUE", True)
        setattr(m1, "V382_ACTION_CONDITIONAL_QUANTILE_ATOMIC", False)
        setattr(m1, "V381_LESION_BACKGROUND_CALIBRATED_ATOMIC", False)
        setattr(m1, "V38_CASEWISE_FALSIFIED_DELTA_CONSENSUS", False)
        setattr(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
        setattr(m1, "V37_CONTEXT_CLEAN_CONTROLS", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", False)
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        setattr(m1, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v383_conservative_action_value")
        if not requested_tag.startswith("V383_"):
            setattr(m1, "RUN_TAG", "V383_B0Frozen_ConservativeActionValue_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V20_ACTIONS_PER_TYPE": 2,
            "V20_CONTEXT_RADIUS": 8,
            "V20_CONTROL_MIN_SHIFT": 8,
            "V37_CONTROL_CONTEXT_GUARD": 1,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V382_GAIN_SCALE": 0.06,
            "V382_GAIN_CLIP": 0.06,
            "V382_HUBER_BETA": 0.20,
            "V382_SOFTMAX_TEMPERATURE": 0.02,
            "V382_DSC_BENEFIT_MARGIN": 0.002,
            "V382_DSC_HARM_MARGIN": 0.002,
            "V382_SEVERE_HARM_WEIGHT": 3.0,
            "V382_HARM_CLASS_WEIGHT": 2.0,
            "V382_BENEFIT_CLASS_WEIGHT": 4.0,
            "V382_PAIRWISE_MARGIN": 0.002,
            "V382_SOURCE_WEIGHT": 1.0,
            "V382_PURIFICATION_WEIGHT": 1.0,
            "V382_TP_PRESERVE_WEIGHT": 2.0,
            "V382_PROPOSAL_WEIGHT": 1.0,
            "V382_REPAIR_WEIGHT": 0.75,
            "V382_QUANTILE_WEIGHT": 1.0,
            "V382_MEDIAN_WEIGHT": 0.75,
            "V382_OUTCOME_WEIGHT": 0.50,
            "V382_CHOICE_WEIGHT": 1.0,
            "V382_EXPECTED_GAIN_WEIGHT": 0.50,
            "V382_DOWNSIDE_WEIGHT": 4.0,
            "V382_PAIRWISE_WEIGHT": 1.0,
            "V382_SEMANTIC_DIM": 16,
            "V382_SPATIAL_DIM": 96,
            "V382_HIDDEN_DIM": 128,
            "V382_TYPE_EMBED_DIM": 12,
            "V382_DROPOUT": 0.10,
            "V382_DELTA_EPS": 1.0e-08,
            "V382_STRUCT_SIZE": 80,
            "V382_MAX_EDIT_FRACTION": 0.035,
            "V382_MAX_EDIT_PERIMETER_GROWTH": 5.0,
            "V383_MIN_Q10_GAIN": 0.0010,
            "V383_MAX_HARM_PROBABILITY": 0.20,
            "V383_MIN_BENEFIT_PROBABILITY": 0.35,
            "V383_MIN_BENEFIT_HARM_GAP": 0.05,
            "V383_HARM_RISK_PENALTY": 0.006,
            "V383_POLICY_CE_WEIGHT": 1.50,
            "V383_SAFETY_SCORE_WEIGHT": 1.00,
            "V383_MACRO_HARM_WEIGHT": 0.75,
            "V383_POSITIVE_SCORE_MARGIN": 0.0005,
            "V383_HARMFUL_SCORE_MARGIN": 0.0010,
            "V383_NEUTRAL_SCORE_MARGIN": 0.0005,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V382 retains V381's all-type candidate generator but changes the deployed
    # decision to an action-conditional lower-quantile transition-value model.
    # Preserve is an explicit zero-valued action; M3 is geometric veto only.
    v382_requested = (
        (not v383_requested)
        and (
        _legacy_version_token_in_filename(filename, "v382")
        or requested_tag.startswith("V382_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v382_action_conditional_quantile_atomic"
        )
    )
    if v382_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V382_ACTION_CONDITIONAL_QUANTILE_ATOMIC", True)
        setattr(m1, "V381_LESION_BACKGROUND_CALIBRATED_ATOMIC", False)
        setattr(m1, "V38_CASEWISE_FALSIFIED_DELTA_CONSENSUS", False)
        setattr(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
        setattr(m1, "V37_CONTEXT_CLEAN_CONTROLS", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", False)
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        setattr(m1, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        # Retained only as train-split source-residual proposal supervision.
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v382_action_conditional_quantile_atomic")
        if not requested_tag.startswith("V382_"):
            setattr(m1, "RUN_TAG", "V382_B0Frozen_ActionConditionalQuantileAtomic_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V20_ACTIONS_PER_TYPE": 2,
            "V20_CONTEXT_RADIUS": 8,
            "V20_CONTROL_MIN_SHIFT": 8,
            "V37_CONTROL_CONTEXT_GUARD": 1,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V382_GAIN_SCALE": 0.06,
            "V382_GAIN_CLIP": 0.06,
            "V382_HUBER_BETA": 0.20,
            "V382_SOFTMAX_TEMPERATURE": 0.02,
            "V382_DSC_BENEFIT_MARGIN": 0.002,
            "V382_DSC_HARM_MARGIN": 0.002,
            "V382_SEVERE_HARM_WEIGHT": 3.0,
            "V382_HARM_CLASS_WEIGHT": 2.0,
            "V382_BENEFIT_CLASS_WEIGHT": 4.0,
            "V382_PAIRWISE_MARGIN": 0.002,
            "V382_SOURCE_WEIGHT": 1.0,
            "V382_PURIFICATION_WEIGHT": 1.0,
            "V382_TP_PRESERVE_WEIGHT": 2.0,
            "V382_PROPOSAL_WEIGHT": 1.0,
            "V382_REPAIR_WEIGHT": 0.75,
            "V382_QUANTILE_WEIGHT": 1.0,
            "V382_MEDIAN_WEIGHT": 0.75,
            "V382_OUTCOME_WEIGHT": 0.50,
            "V382_CHOICE_WEIGHT": 2.0,
            "V382_EXPECTED_GAIN_WEIGHT": 1.0,
            "V382_DOWNSIDE_WEIGHT": 4.0,
            "V382_PAIRWISE_WEIGHT": 0.75,
            "V382_SEMANTIC_DIM": 16,
            "V382_SPATIAL_DIM": 96,
            "V382_HIDDEN_DIM": 128,
            "V382_TYPE_EMBED_DIM": 12,
            "V382_DROPOUT": 0.10,
            "V382_DELTA_EPS": 1.0e-08,
            "V382_STRUCT_SIZE": 80,
            "V382_MAX_EDIT_FRACTION": 0.035,
            "V382_MAX_EDIT_PERIMETER_GROWTH": 5.0,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V381 is a minimal repair of V38 M2: no batch-rotated swap, no duplicate
    # swap loss, no raw-delta AND gate, no concatenated composition verifier.
    # It uses one lesion-vs-background factual/control contrast calibrated
    # against Preserve=0.  V381 is intentionally atomic-only during preflight.
    v381_requested = (
        (not v382_requested)
        and (
            _legacy_version_token_in_filename(filename, "v381")
        or requested_tag.startswith("V381_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v381_lesion_background_calibrated_atomic"
        )
    )
    if v381_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V381_LESION_BACKGROUND_CALIBRATED_ATOMIC", True)
        setattr(m1, "V38_CASEWISE_FALSIFIED_DELTA_CONSENSUS", False)
        setattr(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
        setattr(m1, "V37_CONTEXT_CLEAN_CONTROLS", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", False)
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        setattr(m1, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v381_lesion_background_calibrated_atomic")
        if not requested_tag.startswith("V381_"):
            setattr(m1, "RUN_TAG", "V381_B0Frozen_LesionBackgroundCalibratedAtomic_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V20_ACTIONS_PER_TYPE": 2,
            "V20_CONTEXT_RADIUS": 8,
            "V20_CONTROL_MIN_SHIFT": 8,
            "V37_CONTROL_CONTEXT_GUARD": 1,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V381_GAIN_MARGIN": 0.002,
            "V381_SOURCE_WEIGHT": 1.0,
            "V381_PURIFICATION_WEIGHT": 1.0,
            "V381_TP_PRESERVE_WEIGHT": 2.0,
            "V381_PROPOSAL_WEIGHT": 1.0,
            "V381_REPAIR_WEIGHT": 0.75,
            "V381_ATOMIC_BCE_WEIGHT": 1.0,
            "V381_ATOMIC_PAIR_WEIGHT": 1.5,
            "V381_ATOMIC_PRESERVE_WEIGHT": 1.0,
            "V381_CF_POS_WEIGHT_MIN": 1.0,
            "V381_CF_POS_WEIGHT_MAX": 12.0,
            "V381_PAIR_MARGIN": 0.10,
            "V381_PRESERVE_MARGIN": 0.05,
            "V381_SCALE_INIT": 10.0,
            "V381_BIAS_INIT": 0.0,
            "V381_DEPLOY_LOGIT_THRESHOLD": 0.0,
            "V381_STRUCT_SIZE": 80,
            "V381_DELTA_DICE_WEIGHT": 0.60,
            "V381_EDIT_BOUNDARY_WEIGHT": 0.40,
            "V381_EDIT_EDGE_WEIGHT": 0.25,
            "V381_EDIT_AREA_PENALTY": 0.10,
            "V381_EDIT_PERIMETER_PENALTY": 0.10,
            "V381_CLUSTER_SIMILARITY_MIN": 0.70,
            "V381_MIN_CLUSTER_SIZE": 2,
            "V381_MAX_EDIT_FRACTION": 0.035,
            "V381_MAX_EDIT_PERIMETER_GROWTH": 5.0,
            "V381_SINGLETON_LOGIT_MIN": 0.0,
            "V381_SINGLETON_MIN_EDIT_EDGE": 0.01,
            "V381_SINGLETON_MIN_STABILITY": 0.0,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V38: keeps M1 trainable, replaces independent M2 BCE-only admission with
    # within-case factual-control ranking and Preserve calibration, and replaces
    # full-mask medoid agreement with signed edit-delta M3 consensus.
    v38_requested = (
        _legacy_version_token_in_filename(filename, "v38")
        or requested_tag.startswith("V38_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v38_casewise_falsified_delta_consensus"
    )
    if v38_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V38_CASEWISE_FALSIFIED_DELTA_CONSENSUS", True)
        setattr(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
        setattr(m1, "V37_CONTEXT_CLEAN_CONTROLS", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", False)
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        setattr(m1, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v38_casewise_falsified_delta_consensus")
        if not requested_tag.startswith("V38_"):
            setattr(m1, "RUN_TAG", "V38_B0Frozen_CasewiseFalsifiedDeltaConsensus_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V20_ACTIONS_PER_TYPE": 2,
            "V20_CONTEXT_RADIUS": 8,
            "V20_CONTROL_MIN_SHIFT": 8,
            "V37_CONTROL_CONTEXT_GUARD": 1,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V38_GAIN_MARGIN": 0.002,
            "V38_SOURCE_WEIGHT": 1.0,
            "V38_PURIFICATION_WEIGHT": 1.0,
            "V38_TP_PRESERVE_WEIGHT": 2.0,
            "V38_PROPOSAL_WEIGHT": 1.0,
            "V38_REPAIR_WEIGHT": 0.75,
            "V38_ATOMIC_BCE_WEIGHT": 1.0,
            "V38_ATOMIC_PAIR_WEIGHT": 1.5,
            "V38_ATOMIC_PRESERVE_WEIGHT": 1.0,
            "V38_COMBO_BCE_WEIGHT": 0.75,
            "V38_COMBO_PAIR_WEIGHT": 1.0,
            "V38_COMBO_PRESERVE_WEIGHT": 0.75,
            "V38_CONTRAST_WEIGHT": 0.25,
            "V38_SWAP_WEIGHT": 0.25,
            "V38_ANTISYMMETRY_WEIGHT": 0.10,
            "V38_GEOMETRY_ADV_WEIGHT": 0.10,
            "V38_CONTROL_OVERLAP_WEIGHT": 0.50,
            "V38_PAIR_MARGIN": 0.20,
            "V38_PRESERVE_MARGIN": 0.10,
            "V38_ATOMIC_TEXT_THRESHOLD": 0.50,
            "V38_COMBO_TEXT_THRESHOLD": 0.55,
            "V38_POS_DELTA_MIN": 0.0,
            "V38_NEG_DELTA_MIN": 0.0,
            "V38_SWAP_GAP_MIN": 0.0,
            "V38_STRUCT_SIZE": 80,
            "V38_DELTA_DICE_WEIGHT": 0.60,
            "V38_EDIT_BOUNDARY_WEIGHT": 0.40,
            "V38_EDIT_EDGE_WEIGHT": 0.25,
            "V38_EDIT_AREA_PENALTY": 0.10,
            "V38_EDIT_PERIMETER_PENALTY": 0.10,
            "V38_CLUSTER_SIMILARITY_MIN": 0.70,
            "V38_MIN_CLUSTER_SIZE": 2,
            "V38_MAX_EDIT_FRACTION": 0.035,
            "V38_MAX_EDIT_PERIMETER_GROWTH": 5.0,
            "V38_SINGLETON_TEXT_THRESHOLD": 0.80,
            "V38_SINGLETON_MIN_EDIT_EDGE": 0.01,
            "V38_SINGLETON_MIN_STABILITY": 0.0,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V37: actual M1 -> M2 -> M3 chain. M1 remains trainable across all action
    # families, M2 owns the factual/control text qualification gate, and M3 is
    # a deterministic structural-consensus medoid. V35 residual is only a
    # source proposal field; its gain/risk decision branch is never instantiated.
    v37_requested = (
        _legacy_version_token_in_filename(filename, "v37")
        or requested_tag.startswith("V37_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v37_text_falsified_structural_consensus"
    )
    if v37_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", True)
        setattr(m1, "V37_CONTEXT_CLEAN_CONTROLS", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", False)
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        setattr(m1, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        setattr(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        # Kept only so the shared M1 generator creates the source residual map.
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v37_text_falsified_structural_consensus")
        if not requested_tag.startswith("V37_"):
            setattr(m1, "RUN_TAG", "V37_B0Frozen_TextFalsifiedStructuralConsensus_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V20_ACTIONS_PER_TYPE": 2,
            "V20_CONTEXT_RADIUS": 8,
            "V20_CONTROL_MIN_SHIFT": 8,
            "V37_CONTROL_CONTEXT_GUARD": 1,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V37_GAIN_MARGIN": 0.002,
            "V37_SOURCE_WEIGHT": 1.0,
            "V37_PURIFICATION_WEIGHT": 1.0,
            "V37_TP_PRESERVE_WEIGHT": 2.0,
            "V37_PROPOSAL_WEIGHT": 1.0,
            "V37_REPAIR_WEIGHT": 0.75,
            "V37_CF_WEIGHT": 1.5,
            "V37_CONTRAST_WEIGHT": 0.5,
            "V37_SWAP_WEIGHT": 0.5,
            "V37_PAIR_SWAP_WEIGHT": 0.2,
            "V37_GEOMETRY_ADV_WEIGHT": 0.1,
            "V37_CONTROL_OVERLAP_WEIGHT": 0.5,
            "V37_TEXT_THRESHOLD": 0.50,
            "V37_POS_DELTA_MIN": 0.0,
            "V37_NEG_DELTA_MIN": 0.0,
            "V37_SWAP_GAP_MIN": 0.0,
            "V37_STRUCT_SIZE": 80,
            "V37_MASK_DICE_WEIGHT": 0.40,
            "V37_BOUNDARY_DICE_WEIGHT": 0.45,
            "V37_MEMBERSHIP_WEIGHT": 0.15,
            "V37_EDGE_WEIGHT": 0.25,
            "V37_AREA_PENALTY": 0.10,
            "V37_PERIMETER_PENALTY": 0.15,
            "V37_MIN_EDITED_HYPOTHESES": 2,
            "V37_MIN_CLUSTER_SIZE": 2,
            "V37_CLUSTER_SIMILARITY_MIN": 0.82,
            "V37_MAX_AREA_LOG_SHIFT": 0.18,
            "V37_MAX_PERIMETER_GROWTH": 0.20,
            "V37_BASE_EDGE_TOLERANCE": 0.02,
            "V37_SINGLETON_TEXT_THRESHOLD": 0.80,
            "V37_SINGLETON_MIN_EDGE_GAIN": 0.005,
            "V37_SINGLETON_MIN_STABILITY": 0.0,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V36 casewise Plackett-Luce ranker.  V36 retains the V35 Train-only
    # residual proposal source but trains the exact Preserve-vs-actions list
    # that deployment ranks, with within-image safety constraints.
    v36_requested = (
        _legacy_version_token_in_filename(filename, "v36")
        or requested_tag.startswith("V36_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v36_casewise_plackett_luce"
    )
    if v36_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", True)
        # This activates only the source residual proposal field. The V36
        # decision branch explicitly disables V35's gain-minus-risk policy.
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "V36_CASEWISE_PLACKETT_LUCE", True)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "M1_LOSS_VERSION", "v36_casewise_plackett_luce")
        if not requested_tag.startswith("V36_"):
            setattr(m1, "RUN_TAG", "V36_B0Frozen_CasewisePLRank_100ep")
        for key, value in {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V32_ISLAND_TYPE_ID": 0,
            "V32_FREEZE_CANDIDATE_BANK": False,
            "V32_DELTA_EPS": 1e-8,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V35_SOURCE_WEIGHT": 1.0,
            "V35_PURIFICATION_WEIGHT": 1.0,
            "V35_TP_PRESERVE_WEIGHT": 2.0,
            "V36_GAIN_SCALE": 0.06,
            "V36_GAIN_CLIP": 0.06,
            "V36_SOFTMAX_TEMPERATURE": 0.003,
            "V36_TARGET_TEMPERATURE": 0.003,
            "V36_MODEL_TEMPERATURE": 0.003,
            "V36_PAIR_TEMPERATURE": 0.003,
            "V36_DSC_BENEFIT_MARGIN": 0.002,
            "V36_DSC_HARM_MARGIN": 0.002,
            "V36_PAIR_UTILITY_GAP": 0.002,
            "V36_PAIR_SCORE_MARGIN": 0.001,
            "V36_POSITIVE_SCORE_MARGIN": 0.001,
            "V36_HARM_SCORE_MARGIN": 0.002,
            "V36_LISTWISE_WEIGHT": 2.0,
            "V36_CHOICE_WEIGHT": 1.0,
            "V36_PAIRWISE_WEIGHT": 1.5,
            "V36_POSITIVE_ABOVE_PRESERVE_WEIGHT": 1.0,
            "V36_HARM_BELOW_PRESERVE_WEIGHT": 2.5,
            "V36_SCORE_REGRESSION_WEIGHT": 1.0,
            "V36_OUTCOME_WEIGHT": 0.5,
            "V36_EXPECTED_UTILITY_WEIGHT": 0.5,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V35 residual-purified counterfactual world model.  This is a source-level
    # correction: B0 remains frozen, while only the island-delete proposal
    # field is trained against B0 false-positive residual supervision.  The
    # decision world model then scores the purified candidate actions.
    v35_requested = (
        _legacy_version_token_in_filename(filename, "v35")
        or requested_tag.startswith("V35_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v35_residual_purified_world_model"
    )
    if v35_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", True)
        setattr(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", True)
        setattr(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        setattr(m1, "M1_LOSS_VERSION", "v35_residual_purified_world_model")
        if not requested_tag.startswith("V35_"):
            setattr(m1, "RUN_TAG", "V35_B0Frozen_ResidualPurifiedWorldModel_100ep")
        for key, value in {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V32_ISLAND_TYPE_ID": 0,
            "V32_FREEZE_CANDIDATE_BANK": False,
            "V32_DELTA_EPS": 1e-8,
            "V35_RESIDUAL_PROPOSAL_SCALE": 1.0,
            "V35_GAIN_SCALE": 0.06,
            "V35_GAIN_CLIP": 0.06,
            "V35_SOFTMAX_TEMPERATURE": 0.02,
            "V35_HARM_PENALTY": 0.012,
            "V35_DSC_BENEFIT_MARGIN": 0.002,
            "V35_DSC_HARM_MARGIN": 0.002,
            "V35_SOURCE_FOCAL_GAMMA": 2.0,
            "V35_SOURCE_POS_WEIGHT": 6.0,
            "V35_SOURCE_WEIGHT": 1.0,
            "V35_PURIFICATION_WEIGHT": 1.0,
            "V35_TP_PRESERVE_WEIGHT": 2.0,
            "V35_GAIN_WEIGHT": 1.5,
            "V35_OUTCOME_WEIGHT": 1.0,
            "V35_CHOICE_WEIGHT": 1.5,
            "V35_EXPECTED_GAIN_WEIGHT": 1.0,
            "V35_DOWNSIDE_WEIGHT": 1.0,
            "V35_PAIRWISE_WEIGHT": 0.5,
            "V35_PAIRWISE_MARGIN": 0.006,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    # V34 spatial quantile world-model.  V34 intentionally reuses the frozen
    # V32 island-only candidate domain, but replaces the point-score selector
    # with a spatial counterfactual outcome distribution.
    v34_requested = (
        _legacy_version_token_in_filename(filename, "v34")
        or requested_tag.startswith("V34_")
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v34_spatial_quantile_world_model"
    )
    if v34_requested:
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", True)
        setattr(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", True)
        setattr(m1, "M1_LOSS_VERSION", "v34_spatial_quantile_world_model")
        if not requested_tag.startswith("V34_"):
            setattr(m1, "RUN_TAG", "V34_B0Frozen_SpatialQuantileWorldModel_dev10")
        for key, value in {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V32_ISLAND_TYPE_ID": 0,
            "V32_FREEZE_CANDIDATE_BANK": True,
            "V32_DELTA_EPS": 1e-8,
            "V34_SEMANTIC_DIM": 16,
            "V34_SPATIAL_DIM": 96,
            "V34_GAIN_SCALE": 0.06,
            "V34_GAIN_CLIP": 0.06,
            "V34_DEPLOY_QUANTILE_INDEX": 0,
            "V34_SOFTMAX_TEMPERATURE": 0.010,
            "V34_HUBER_BETA": 0.20,
            "V34_DSC_BENEFIT_MARGIN": 0.002,
            "V34_DSC_HARM_MARGIN": 0.002,
            "V34_QUANTILE_WEIGHT": 1.0,
            "V34_MEDIAN_WEIGHT": 1.0,
            "V34_DECISION_WEIGHT": 2.0,
            "V34_DOWNSIDE_WEIGHT": 4.0,
            "V34_PAIRWISE_WEIGHT": 0.5,
            "V34_SEVERE_HARM_WEIGHT": 3.0,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    v32_requested = _legacy_version_token_in_filename(filename, "v32") or requested_tag.startswith("V32_")
    if v32_requested:
        setattr(m1, "V32_ISLAND_PHASEB_POLICY", True)
        setattr(m1, "M1_LOSS_VERSION", "v32_island_phaseb")
        if not requested_tag.startswith("V32_"):
            setattr(m1, "RUN_TAG", "V32_B0Frozen_IslandDeletePhaseB_dev10")
        for key, value in {
            "V32_ISLAND_TYPE_ID": 0,
            "V32_FREEZE_CANDIDATE_BANK": True,
            "V32_TEXT_VETO_WEIGHT": 0.15,
            "V32_DELTA_EPS": 1e-8,
            "V32_DSC_BENEFIT_MARGIN": 0.002,
            "V32_DSC_HARM_MARGIN": 0.002,
            "V32_NEUTRAL_NULL_MARGIN": 0.015,
            "V32_HARMFUL_NULL_MARGIN": 0.075,
            "V32_POSITIVE_CASE_WEIGHT": 2.0,
            "V32_CHOICE_WEIGHT": 2.0,
            "V32_RANK_WEIGHT": 1.0,
            "V32_UTILITY_WEIGHT": 0.50,
            "V32_VETO_WEIGHT": 0.10,
        }.items():
            setattr(m1, key, getattr(m1, key, value))

    v31_requested = _legacy_version_token_in_filename(filename, "v31") or requested_tag.startswith("V31_")
    if v31_requested:
        setattr(m1, "V31_CANDIDATE_CONDITIONED_POLICY", True)
        setattr(m1, "V20_UNIFIED_ACTION_CF", True)
        setattr(m1, "M1_LOSS_VERSION", "v31_competitive_utility")
        if not requested_tag.startswith("V31_"):
            setattr(m1, "RUN_TAG", "V31_B0Frozen_CandidateConditionedUtility_dev10")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V31_HIDDEN_DIM": 128,
            "V31_TYPE_EMBED_DIM": 16,
            "V31_CONTEXT_RADIUS": 3,
            "V31_DROPOUT": 0.10,
            "V31_TEXT_VETO_WEIGHT": 0.25,
            "V31_NULL_MARGIN": 0.05,
            "V31_DSC_BENEFIT_MARGIN": 0.002,
            "V31_DSC_HARM_MARGIN": 0.002,
            "V31_NEUTRAL_NULL_MARGIN": 0.025,
            "V31_HARMFUL_NULL_MARGIN": 0.100,
            "V31_BENEFIT_NULL_MARGIN": 0.050,
            "V31_PROPOSAL_WEIGHT": 1.00,
            "V31_REPAIR_WEIGHT": 0.50,
            "V31_UTILITY_WEIGHT": 1.00,
            "V31_VETO_WEIGHT": 0.50,
            "V31_CHOICE_WEIGHT": 2.00,
            "V31_RANK_WEIGHT": 1.50,
            "V31_FUSION_WEIGHT": 0.50,
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    v27_requested = _legacy_version_token_in_filename(filename, "v27") or requested_tag.startswith("V27_")
    if v27_requested:
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not requested_tag.startswith("V27_"):
            setattr(m1, "RUN_TAG", "V27_B0Frozen_ParetoUtilityPolicy_100ep")
        defaults = {
            "M1_TRAIN_NUM_SAMPLES": 4,
            "V25_HIDDEN_DIM": 128,
            "V25_TYPE_EMBED_DIM": 16,
            "V25_MAX_ACTIONS": 1,
            "V25_BENEFIT_MARGIN": 0.002,
            "V25_HARM_MARGIN": 0.002,
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
        }
        for key, value in defaults.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
        return cfg

    v26_requested = (
        _legacy_version_token_in_filename(filename, "v26")
        or requested_tag.startswith("V26_")
        or _cfg_get(m1, "V26_NULL_MARGIN", None) is not None
    )
    if v26_requested:
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not requested_tag.startswith("V26_"):
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
        or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
        == "v25_type_conditional_utility"
        or requested_tag.startswith("V25_")
    )
    if enabled:
        setattr(m1, "V25_TYPE_CONDITIONAL_UTILITY_BANK", True)
        setattr(m1, "M1_LOSS_VERSION", "v25_type_conditional_utility")
        if not requested_tag.startswith("V25_"):
            setattr(m1, "RUN_TAG", "V25_B0Frozen_TypeConditionalUtilityBank_100ep")
        for key, value in {
            "V25_HIDDEN_DIM": 128,
            "V25_TYPE_EMBED_DIM": 16,
            "V25_BENEFIT_MARGIN": 0.002,
            "V25_HARM_MARGIN": 0.002,
            "V25_BENEFIT_MIN": 0.50,
            "V25_HARM_MAX": 0.30,
            "V25_UTILITY_MARGIN": 0.10,
            "V25_SELECTOR_MARGIN": 0.00,
            "V25_MAX_ACTIONS": 1,
        }.items():
            if _cfg_get(m1, key, None) is None:
                setattr(m1, key, value)
    return cfg

def _validate_semlt_lst_v31_rootfix_runtime(cfg):
    """Fail fast if the v3.1 Root-Fix is contaminated by legacy V31 routing."""
    m1 = _cfg_get(cfg, "M1", None)
    if m1 is None or not bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
        return cfg

    violations = []
    if bool(_cfg_get(m1, "V20_UNIFIED_ACTION_CF", False)):
        violations.append("V20_UNIFIED_ACTION_CF must be false")
    if bool(_cfg_get(m1, "V31_CANDIDATE_CONDITIONED_POLICY", False)):
        violations.append("V31_CANDIDATE_CONDITIONED_POLICY must be false")
    if str(_cfg_get(m1, "M1_LOSS_VERSION", "") or "").strip().lower() == "v31_competitive_utility":
        violations.append("legacy M1_LOSS_VERSION=v31_competitive_utility is forbidden")

    mc_samples = int(_cfg_get(m1, "M1_TRAIN_NUM_SAMPLES", 1))
    if mc_samples != 1:
        violations.append(
            f"M1_TRAIN_NUM_SAMPLES must be 1 for joint-E2E Root-Fix, got {mc_samples}"
        )
    if str(_cfg_get(m1, "CANDIDATE_MODE", "")).strip().lower() != "semlt_autozero_transport":
        violations.append("CANDIDATE_MODE must be semlt_autozero_transport")
    if str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() != "semlt_autozero":
        violations.append("PROTOCOL must be semlt_autozero")

    if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
        if str(_cfg_get(m1, "SEMLT_GATE_LOSS_DOMAIN", "")).strip().lower() != "transition_eligible":
            violations.append(
                "eligible-gate root-fix requires SEMLT_GATE_LOSS_DOMAIN=transition_eligible"
            )
        action_value = bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False))
        expected_loss_version = (
            "semlt_lst_action_value_policy" if action_value
            else "semlt_lst_eligible_gate_rootfix"
        )
        if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != expected_loss_version:
            violations.append(
                f"eligible-gate root-fix requires M1_LOSS_VERSION={expected_loss_version}"
            )

    if violations:
        raise RuntimeError(
            "SemLT-LST v3.1 Root-Fix routing contract violated; refusing to "
            "start a potentially OOM/wrong-method run: " + "; ".join(violations)
        )
    return cfg


def m1_train_mode(cfg):
    mode = str(_cfg_get(_cfg_get(cfg, "M1", None), "TRAIN_MODE", "anchor_student")).lower()
    allowed = {"e2e", "frozen", "anchor_student"}
    if mode not in allowed:
        raise ValueError(f"M1.TRAIN_MODE must be one of {sorted(allowed)}.")
    return mode


def results_name(cfg):
    name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{cfg.MODEL.BACKBONE.replace('/', '-')}"
    if m1_enabled(cfg):
        run_tag = str(_cfg_get(cfg.M1, "RUN_TAG", "M1PSEDirectFusionV4"))
    else:
        run_tag = str(_cfg_get(cfg.TRAIN, "RUN_TAG", "") or "").strip()
    if run_tag:
        name += "_" + run_tag
    return name


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_only(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _official_base_batch_seed(seed: int, epoch: int, batch_index: int) -> int:
    """Independent deterministic RNG stream for each physical Base update."""
    modulus = 2**31 - 1
    return int((int(seed) * 1_000_003 + int(epoch) * 100_003 + int(batch_index) + 97) % modulus)


def _load_base_parity_reference(path: str):
    """Read the completed matched-Base hashes for fail-fast comparison."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise RuntimeError(f"JBT Base parity reference log does not exist: {path}")
    init_pattern = re.compile(r"\[UCFNRT_BASE_PARITY_INIT\].*sha256=([0-9a-f]{64})")
    step_pattern = re.compile(r"\[UCFNRT_BASE_PARITY_STEP1\].*sha256=([0-9a-f]{64})")
    epoch_pattern = re.compile(
        r"\[UCFNRT_BASE_PARITY_EPOCH_END\] epoch=(\d+).*sha256=([0-9a-f]{64})"
    )
    reference = {"epochs": {}}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = init_pattern.search(line)
            if match:
                reference["init"] = match.group(1)
            match = step_pattern.search(line)
            if match:
                reference["step1"] = match.group(1)
            match = epoch_pattern.search(line)
            if match:
                reference["epochs"][int(match.group(1))] = match.group(2)
    if "init" not in reference or "step1" not in reference or not reference["epochs"]:
        raise RuntimeError(
            "Matched-Base parity reference is incomplete; expected init, step1 and epoch hashes: "
            + path
        )
    return reference


def _assert_reference_hash(reference, key, actual: str, epoch: int = 0):
    if not reference:
        return
    expected = reference["epochs"].get(epoch) if key == "epoch" else reference.get(key)
    if expected is None:
        raise RuntimeError(f"Matched-Base parity reference has no {key} hash for epoch={epoch}")
    if actual != expected:
        raise RuntimeError(
            f"[JBT_V633_PARITY_FAIL_FAST] {key} mismatch at epoch={epoch}: "
            f"expected={expected} actual={actual}"
        )


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume the matching experiment")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="B0 best-Dice checkpoint. Required for frozen and anchor_student M1 modes.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use")
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir",
        type=str, default="", help="Output directory"
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Config overrides")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    # Canonical research configuration: map its stable public interface onto
    # the mature reference implementation without exposing historical keys.
    if str(_cfg_get(cfg.M1, "CANDIDATE_MODE", "")).lower() == "mechanism":
        cfg.M1.ACTION_BANK_UNIFIED_ACTION_CF = True
        cfg.M1.V426_MECHANISM_DUAL_EXPERT = True
        train_mode = str(_cfg_get(cfg.M1, "TRAIN_MODE", "frozen")).lower()
        # frozen/reference/cache audit path should stay strict candidate-only;
        # unified/e2e path must allow final fusion/selection training.
        cfg.M1.M1_STRICT_CANDIDATE_ONLY = train_mode in {
            "frozen",
            "candidate_only",
            "anchor_student",
        }
        cfg.M1.EVIDENCE_GUIDED_FECG_ENABLED = True

    cfg.merge_from_list(args.opts)
    cfg.update({key: value for key, value in vars(args).items()})
    cfg = _activate_v25_runtime_compat(cfg, args.config_file)
    cfg = _validate_semlt_lst_v31_rootfix_runtime(cfg)
    return cfg


def logger_config(log_path):
    logger = logging.getLogger(f"MedCLIPSegTrain:{os.path.abspath(log_path)}")
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


def _per_case_fp_tversky_loss(logits, labels, alpha: float, beta: float):
    """Per-case Tversky loss with explicit false-positive emphasis.

    ``alpha > beta`` penalizes background leakage more strongly than missed
    foreground.  This is intentionally computed per case so a CVaR-style tail
    term can focus training on the hardest samples in each mini-batch.
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    labels = (labels > 0.5).to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    tp = (probs * labels).flatten(1).sum(dim=1)
    fp = (probs * (1.0 - labels)).flatten(1).sum(dim=1)
    fn = ((1.0 - probs) * labels).flatten(1).sum(dim=1)
    score = (tp + 1.0e-6) / (tp + alpha * fp + beta * fn + 1.0e-6)
    return 1.0 - score


def _per_case_base_composite_loss(logits, labels, dice_weight: float, ce_weight: float):
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits_view = logits[:, 0]
    else:
        logits_view = logits
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels_view = labels[:, 0]
    else:
        labels_view = labels
    labels_view = (labels_view > 0.5).to(logits_view.dtype)
    probs = torch.sigmoid(logits_view)
    bce = F.binary_cross_entropy_with_logits(
        logits_view, labels_view, reduction="none"
    ).flatten(1).mean(dim=1)
    inter = (probs * labels_view).flatten(1).sum(dim=1)
    den = probs.flatten(1).sum(dim=1) + labels_view.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + 1.0e-6) / (den + 1.0e-6)
    return dice_weight * dice + ce_weight * bce


def _v547_weighted_mean(values, case_weights=None):
    values = values.reshape(-1)
    if case_weights is None:
        return values.mean()
    weights = case_weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if weights.numel() != values.numel():
        raise ValueError(
            f"V547 case weight count {weights.numel()} != batch {values.numel()}"
        )
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)


def _v547_soft_boundary(mask, radius=1):
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask[:, None], kernel, 1, radius)[:, 0]
    eroded = -F.max_pool2d(-mask[:, None], kernel, 1, radius)[:, 0]
    return (dilated - eroded).clamp(0.0, 1.0)


def _slr_true_hr_enabled(cfg):
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        _cfg_get(m1, "ENABLED", False)
        and (
            bool(_cfg_get(m1, "GEOTR_SPARC_HR_ENABLED", False))
            or (
                str(_cfg_get(m1, "GEOTR_AEFR_STAGE", "")).strip().lower() == "sparse_local_rerendering"
                and _cfg_get(m1, "GEOTR_SLR_TRUE_HR_ENABLED", False)
            )
        )
    )

def _v547_semantic_safe_augment(images, masks, cfg, hr_images=None, hr_masks=None):
    if not bool(_cfg_get(cfg.TRAIN, "V547_SEMANTIC_SAFE_AUG_ENABLED", False)):
        if hr_images is not None and hr_masks is not None:
            return images, masks, hr_images, hr_masks
        if hr_images is not None:
            return images, masks, hr_images
        if hr_masks is not None:
            return images, masks, hr_masks
        return images, masks
    return apply_v547_semantic_safe_augmentation(
        images,
        masks,
        probability=float(_cfg_get(cfg.TRAIN, "V547_AUG_PROBABILITY", 0.75)),
        rotation_degrees=float(_cfg_get(cfg.TRAIN, "V547_AUG_ROTATION_DEGREES", 10.0)),
        scale_min=float(_cfg_get(cfg.TRAIN, "V547_AUG_SCALE_MIN", 0.97)),
        scale_max=float(_cfg_get(cfg.TRAIN, "V547_AUG_SCALE_MAX", 1.03)),
        translate_fraction=float(_cfg_get(cfg.TRAIN, "V547_AUG_TRANSLATE_FRACTION", 0.02)),
        contrast_min=float(_cfg_get(cfg.TRAIN, "V547_AUG_CONTRAST_MIN", 0.95)),
        contrast_max=float(_cfg_get(cfg.TRAIN, "V547_AUG_CONTRAST_MAX", 1.05)),
        noise_std=float(_cfg_get(cfg.TRAIN, "V547_AUG_NOISE_STD", 0.01)),
        hr_images=hr_images,
        hr_masks=hr_masks,
    )

def _v547_case_difficulty(logits, labels):
    """Detached per-case difficulty used by the dataset-level memory.

    The score is symmetric: it contains overlap error, FP, FN, foreground-mass
    distortion, and boundary mismatch.  It is used only for relative ranking,
    not as an additional gradient path.
    """
    logit_view = logits[:, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits
    label_view = labels[:, 0] if labels.ndim == 4 and labels.shape[1] == 1 else labels
    gt = (label_view > 0.5).to(logit_view.dtype)
    prob = torch.sigmoid(logit_view)
    tp = (prob * gt).flatten(1).sum(dim=1)
    fp = (prob * (1.0 - gt)).flatten(1).sum(dim=1)
    fn = ((1.0 - prob) * gt).flatten(1).sum(dim=1)
    pred_mass = prob.flatten(1).sum(dim=1)
    gt_mass = gt.flatten(1).sum(dim=1)
    dice_error = 1.0 - (2.0 * tp + 1.0e-6) / (
        pred_mass + gt_mass + 1.0e-6
    )
    fp_rate = fp / pred_mass.clamp_min(1.0)
    fn_rate = fn / gt_mass.clamp_min(1.0)
    mass_error = (
        torch.log((pred_mass + 1.0) / (gt_mass + 1.0)).abs() / 3.0
    ).clamp(0.0, 1.0)
    pred_boundary = _v547_soft_boundary(prob)
    gt_boundary = _v547_soft_boundary(gt)
    boundary_inter = (pred_boundary * gt_boundary).flatten(1).sum(dim=1)
    boundary_den = (
        pred_boundary.flatten(1).sum(dim=1)
        + gt_boundary.flatten(1).sum(dim=1)
    )
    boundary_error = 1.0 - (2.0 * boundary_inter + 1.0e-6) / (
        boundary_den + 1.0e-6
    )
    return (
        0.55 * dice_error
        + 0.15 * fp_rate
        + 0.15 * fn_rate
        + 0.10 * mass_error
        + 0.05 * boundary_error
    ).detach()


def _jbtl_auxiliary_scale(cfg):
    """Epoch-dependent scale for JBT-Lite boundary objectives.

    Supported schedules:
      * ``cosine`` / ``linear``: legacy hold-then-decay schedules.
      * ``hard_cutoff``: full strength through ``RBAL_FULL_WEIGHT_EPOCHS``
        and exactly zero afterwards.
      * ``matched_budget``: reproduce the *effective EDGE update budget* of a
        shorter reference cosine run while the Base follows the longer formal
        scheduler.  For one-based training epoch ``e`` and optimizer-step index
        ``t=e-1``::

            scale(e) = eta_ref(t) / eta_target(t),   e <= active_epochs
            scale(e) = 0,                           e > active_epochs

        where eta_ref is CosineAnnealingLR with
        ``RBAL_BUDGET_REFERENCE_EPOCHS`` (20 by default) and eta_target uses
        the actual ``SCHEDULER_TOTAL_EPOCHS`` (100 in paper100).  Therefore

            eta_target(e) * lambda_edge * scale(e)
              ~= eta_ref(e) * lambda_edge,

        restoring the successful short-run geometric update magnitude without
        weakening the Base's 100-epoch optimization trajectory.

    The current one-based epoch is written into ``cfg`` by the training loop.
    Validation reuses that value but never updates parameters.
    """
    enabled = bool(_cfg_get(cfg.TRAIN, "RBAL_SCHEDULE_ENABLED", False))
    if not enabled:
        return 1.0

    epoch_1based = max(1, int(getattr(cfg, "_jbtl_current_epoch_1based", 1)))
    hold = max(0, int(_cfg_get(cfg.TRAIN, "RBAL_FULL_WEIGHT_EPOCHS", 20)))
    schedule = str(_cfg_get(cfg.TRAIN, "RBAL_SCHEDULE_TYPE", "cosine")).strip().lower()

    if schedule == "always_on":
        return 1.0

    if schedule == "warmup":
        warmup_end = max(
            1, int(_cfg_get(cfg.TRAIN, "RBAL_WARMUP_END_EPOCH", 30))
        )
        start_scale = float(
            _cfg_get(cfg.TRAIN, "RBAL_WARMUP_START_SCALE", 0.0)
        )
        start_scale = min(1.0, max(0.0, start_scale))
        if epoch_1based >= warmup_end:
            return 1.0
        if warmup_end == 1:
            return 1.0
        progress = float(epoch_1based - 1) / float(warmup_end - 1)
        return start_scale + (1.0 - start_scale) * progress

    if schedule == "hard_cutoff":
        return 1.0 if epoch_1based <= hold else 0.0

    if schedule == "matched_budget":
        active_epochs = max(
            1,
            int(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_ACTIVE_EPOCHS", hold or 20)),
        )
        if epoch_1based > active_epochs:
            return 0.0

        reference_epochs = max(
            1,
            int(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_REFERENCE_EPOCHS", active_epochs)),
        )
        target_epochs = max(
            1,
            int(_cfg_get(
                cfg.TRAIN,
                "SCHEDULER_TOTAL_EPOCHS",
                _cfg_get(cfg.TRAIN, "NUM_EPOCHS", 100),
            )),
        )
        base_lr = float(_cfg_get(cfg.TRAIN, "LEARNING_RATE", 3.0e-4))
        eta_min = float(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_ETA_MIN", 1.0e-4))
        max_scale = max(0.0, float(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_MAX_SCALE", 1.0)))

        # During one-based epoch e the optimizer uses the LR after e-1 scheduler
        # steps. This matches PyTorch CosineAnnealingLR in the current training
        # loop, where scheduler.step() occurs after the epoch update.
        t = max(0, epoch_1based - 1)

        def _cosine_lr(total):
            tt = min(float(t), float(total))
            return eta_min + 0.5 * (base_lr - eta_min) * (
                1.0 + math.cos(math.pi * tt / float(total))
            )

        eta_ref = _cosine_lr(reference_epochs)
        eta_target = _cosine_lr(target_epochs)
        if eta_target <= 0.0:
            raise RuntimeError(
                "[JBTL6_MATCHED_BUDGET] target scheduler LR is non-positive; "
                f"epoch={epoch_1based} eta_target={eta_target}."
            )
        scale = eta_ref / eta_target
        if max_scale > 0.0:
            scale = min(scale, max_scale)
        return max(0.0, float(scale))

    decay_end = max(hold + 1, int(_cfg_get(cfg.TRAIN, "RBAL_DECAY_END_EPOCH", 80)))
    if epoch_1based <= hold:
        return 1.0
    if epoch_1based >= decay_end:
        return 0.0
    progress = (epoch_1based - hold) / float(decay_end - hold)
    if schedule == "linear":
        return max(0.0, 1.0 - progress)
    if schedule != "cosine":
        raise ValueError(
            f"Unsupported TRAIN.RBAL_SCHEDULE_TYPE={schedule!r}; "
            "use cosine, linear, hard_cutoff, matched_budget, always_on, or warmup."
        )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _jbtl_effective_weights(cfg):
    scale = _jbtl_auxiliary_scale(cfg)
    edge = float(_cfg_get(cfg.TRAIN, "RBAL_EDGE_WEIGHT", 0.0)) * scale
    normal = float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_WEIGHT", 0.0)) * scale
    return edge, normal, scale


def _jbtl_auxiliary_loss(logits, target, cfg):
    """Dispatch the pre-declared matched auxiliary objective.

    ``surface`` is the untouched Sep14 objective.  The alternatives use the
    same scalar weight, temporal schedule, logits, labels, and gradient scope;
    therefore Table 4 changes only the auxiliary loss definition.
    """
    loss_type = str(
        _cfg_get(cfg.TRAIN, "RBAL_AUX_LOSS_TYPE", "surface")
    ).strip().lower()
    radius = int(_cfg_get(cfg.TRAIN, "RBAL_BOUNDARY_RADIUS_PX", 1))
    if loss_type == "surface":
        return compute_edge_alignment_loss(
            logits, target, boundary_radius_px=radius
        )
    if loss_type == "boundary":
        return compute_boundary_loss(logits, target)
    if loss_type in {"hausdorff", "hausdorff_dt", "hd"}:
        return compute_hausdorff_dt_loss(
            logits,
            target,
            alpha=float(_cfg_get(cfg.TRAIN, "RBAL_HD_ALPHA", 2.0)),
        )
    if loss_type in {"active_contour", "activecontour", "ac"}:
        return compute_active_contour_loss(
            logits,
            target,
            length_weight=float(
                _cfg_get(cfg.TRAIN, "RBAL_ACTIVE_CONTOUR_LENGTH_WEIGHT", 1.0)
            ),
            region_weight=float(
                _cfg_get(cfg.TRAIN, "RBAL_ACTIVE_CONTOUR_REGION_WEIGHT", 1.0)
            ),
        )
    raise ValueError(
        f"Unsupported TRAIN.RBAL_AUX_LOSS_TYPE={loss_type!r}; "
        "use surface, boundary, hausdorff_dt, or active_contour."
    )



def _jbtl_edge_decoder_only_enabled(cfg):
    return (
        str(_cfg_get(cfg.TRAIN, "RBAL_EDGE_GRAD_SCOPE", "all")).strip().lower()
        == "decoder_only"
        and float(_cfg_get(cfg.TRAIN, "RBAL_EDGE_WEIGHT", 0.0)) > 0.0
    )


def _jbtl_decoder_params(model):
    """Return only the geometric segmentation head parameters.

    The Base objective still trains PVL adapters + decoder.  The boundary-only
    auxiliary gradient is deliberately routed only to mask_head/upscale so a
    contour loss cannot rewrite the cross-modal semantic adapters.
    """
    params = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("mask_head.") or name.startswith("upscale."):
            params.append(param)
            names.append(name)
    if not params:
        raise RuntimeError("[JBTL5_EDGEISO] no trainable mask_head/upscale parameters found")
    return params, names


def _jbtl_edge_aux_loss(logits, target, cfg):
    """Compute the effective EDGE objective for decoder-only gradient routing."""
    edge_weight, normal_weight, scale = _jbtl_effective_weights(cfg)
    if normal_weight > 0.0:
        raise RuntimeError(
            "[JBTL5_EDGEISO] decoder-only formal protocol is EDGE-only; "
            "set TRAIN.RBAL_NORMAL_WEIGHT=0."
        )
    if edge_weight <= 0.0:
        zero = logits.sum() * 0.0
        return zero, edge_weight, scale
    logit_view = logits[:, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits
    label_view = target[:, 0] if target.ndim == 4 and target.shape[1] == 1 else target
    label_view = (label_view > 0.5).to(device=logit_view.device, dtype=logit_view.dtype)
    edge_loss, _ = _jbtl_auxiliary_loss(logit_view, label_view, cfg)
    return edge_loss * float(edge_weight), edge_weight, scale

def calc_base_loss(
    logits,
    labels,
    ce_loss,
    dice_loss,
    cfg,
    clip_loss=None,
    case_weights=None,
):
    """Base objective with dataset-level tail weighting and symmetric error control."""
    v547_enabled = bool(
        _cfg_get(cfg.TRAIN, "V547_HARD_CASE_MEMORY_ENABLED", False)
    )
    if v547_enabled:
        per_case_main = _per_case_base_composite_loss(
            logits,
            labels,
            float(cfg.TRAIN.DICE_WEIGHT),
            float(cfg.TRAIN.CE_WEIGHT),
        )
        loss = _v547_weighted_mean(per_case_main, case_weights)
    else:
        loss = (
            cfg.TRAIN.DICE_WEIGHT * dice_loss(logits, labels)
            + cfg.TRAIN.CE_WEIGHT * ce_loss(logits, labels.float())
        )

    tversky_weight = float(_cfg_get(cfg.TRAIN, "BASE_FP_TVERSKY_WEIGHT", 0.0))
    tail_weight = float(_cfg_get(cfg.TRAIN, "BASE_TAIL_CVAR_WEIGHT", 0.0))
    if tversky_weight > 0.0 or tail_weight > 0.0:
        alpha = max(float(_cfg_get(cfg.TRAIN, "BASE_FP_TVERSKY_ALPHA", 0.70)), 0.0)
        beta = max(float(_cfg_get(cfg.TRAIN, "BASE_FP_TVERSKY_BETA", 0.30)), 0.0)
        denom = max(alpha + beta, 1.0e-6)
        alpha, beta = alpha / denom, beta / denom
        per_case = _per_case_fp_tversky_loss(logits, labels, alpha, beta)
        if tversky_weight > 0.0:
            loss = loss + tversky_weight * _v547_weighted_mean(
                per_case, case_weights if v547_enabled else None
            )
        if tail_weight > 0.0:
            fraction = float(_cfg_get(cfg.TRAIN, "BASE_TAIL_CVAR_FRACTION", 0.25))
            fraction = min(max(fraction, 1.0 / max(int(per_case.numel()), 1)), 1.0)
            k = max(1, int(math.ceil(per_case.numel() * fraction)))
            loss = loss + tail_weight * torch.topk(
                per_case, k=k, largest=True
            ).values.mean()

    # V547 symmetric Tversky prevents fixing large false-positive spill by
    # trading it for catastrophic under-segmentation.
    symmetric_weight = float(
        _cfg_get(cfg.TRAIN, "BASE_SYMMETRIC_TVERSKY_WEIGHT", 0.0)
    )
    if symmetric_weight > 0.0:
        fp_heavy = _per_case_fp_tversky_loss(logits, labels, 0.70, 0.30)
        fn_heavy = _per_case_fp_tversky_loss(logits, labels, 0.30, 0.70)
        symmetric = 0.5 * (fp_heavy + fn_heavy)
        loss = loss + symmetric_weight * _v547_weighted_mean(
            symmetric, case_weights if v547_enabled else None
        )

    dro_weight = float(_cfg_get(cfg.TRAIN, "BASE_HARDNESS_DRO_WEIGHT", 0.0))
    if dro_weight > 0.0:
        per_case_base = _per_case_base_composite_loss(
            logits,
            labels,
            float(cfg.TRAIN.DICE_WEIGHT),
            float(cfg.TRAIN.CE_WEIGHT),
        )
        temperature = max(
            float(_cfg_get(cfg.TRAIN, "BASE_HARDNESS_DRO_TEMPERATURE", 0.25)),
            1.0e-4,
        )
        hardness_weights = torch.softmax(
            per_case_base.detach() / temperature, dim=0
        )
        robust_loss = (hardness_weights * per_case_base).sum()
        loss = loss + dro_weight * (robust_loss - per_case_base.mean())

    label_view = labels[:, 0] if labels.ndim == 4 and labels.shape[1] == 1 else labels
    label_view = (label_view > 0.5).to(dtype=logits.dtype)
    logit_view = logits[:, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits
    prob_view = torch.sigmoid(logit_view)

    hard_negative_weight = float(
        _cfg_get(cfg.TRAIN, "BASE_HARD_NEGATIVE_WEIGHT", 0.0)
    )
    if hard_negative_weight > 0.0:
        radius = max(0, int(_cfg_get(cfg.TRAIN, "BASE_HARD_NEGATIVE_DILATION", 3)))
        kernel = 2 * radius + 1
        safe_foreground = F.max_pool2d(
            label_view[:, None], kernel_size=kernel, stride=1, padding=radius
        )[:, 0]
        far_background = (1.0 - safe_foreground).clamp(0.0, 1.0)
        bg_scores = (prob_view * far_background).flatten(1)
        fraction = float(
            _cfg_get(cfg.TRAIN, "BASE_HARD_NEGATIVE_FRACTION", 0.02)
        )
        k = max(1, int(math.ceil(bg_scores.shape[1] * fraction)))
        hard_negative_case = torch.topk(bg_scores, k=k, dim=1).values.mean(dim=1)
        loss = loss + hard_negative_weight * _v547_weighted_mean(
            hard_negative_case, case_weights if v547_enabled else None
        )

    gt_mass = label_view.flatten(1).sum(dim=1)
    pred_mass = prob_view.flatten(1).sum(dim=1)
    nonempty = gt_mass > 0
    spill_weight = float(_cfg_get(cfg.TRAIN, "BASE_SPILL_RATIO_WEIGHT", 0.0))
    if spill_weight > 0.0 and bool(nonempty.any()):
        ratio = pred_mass[nonempty] / gt_mass[nonempty].clamp_min(1.0)
        margin = float(_cfg_get(cfg.TRAIN, "BASE_SPILL_RATIO_MARGIN", 1.75))
        spill_case = F.relu(ratio - margin).pow(2)
        selected_weights = (
            case_weights[nonempty] if v547_enabled and case_weights is not None else None
        )
        loss = loss + spill_weight * _v547_weighted_mean(
            spill_case, selected_weights
        )

    underfill_weight = float(
        _cfg_get(cfg.TRAIN, "BASE_UNDERFILL_RATIO_WEIGHT", 0.0)
    )
    if underfill_weight > 0.0 and bool(nonempty.any()):
        ratio = pred_mass[nonempty] / gt_mass[nonempty].clamp_min(1.0)
        margin = float(_cfg_get(cfg.TRAIN, "BASE_UNDERFILL_RATIO_MARGIN", 0.55))
        underfill_case = F.relu(margin - ratio).pow(2)
        selected_weights = (
            case_weights[nonempty] if v547_enabled and case_weights is not None else None
        )
        loss = loss + underfill_weight * _v547_weighted_mean(
            underfill_case, selected_weights
        )

    boundary_weight = float(
        _cfg_get(cfg.TRAIN, "BASE_BOUNDARY_DICE_WEIGHT", 0.0)
    )
    if boundary_weight > 0.0:
        pred_boundary = _v547_soft_boundary(prob_view)
        gt_boundary = _v547_soft_boundary(label_view)
        inter = (pred_boundary * gt_boundary).flatten(1).sum(dim=1)
        den = (
            pred_boundary.flatten(1).sum(dim=1)
            + gt_boundary.flatten(1).sum(dim=1)
        )
        boundary_case = 1.0 - (2.0 * inter + 1.0e-6) / (den + 1.0e-6)
        loss = loss + boundary_weight * _v547_weighted_mean(
            boundary_case, case_weights if v547_enabled else None
        )

    # JBT-Lite v3: keep the empirically useful surface-alignment term and
    # restore only the successful *principle* from older JBT signed-displacement
    # supervision as a lightweight signed-normal margin loss.
    rbal_edge_weight, rbal_normal_weight, _ = _jbtl_effective_weights(cfg)
    edge_decoder_only = _jbtl_edge_decoder_only_enabled(cfg)
    if rbal_normal_weight > 0.0 and edge_decoder_only:
        raise RuntimeError(
            "[JBTL5_EDGEISO] NORMAL is disabled in decoder-only EDGE protocol."
        )
    if rbal_edge_weight > 0.0 and not edge_decoder_only:
        rbal_edge, _ = _jbtl_auxiliary_loss(logit_view, label_view, cfg)
        loss = loss + rbal_edge_weight * rbal_edge
    if rbal_normal_weight > 0.0:
        rbal_normal, _ = compute_normal_margin_loss(
            logit_view, label_view,
            boundary_radius_px=int(_cfg_get(cfg.TRAIN, "RBAL_BOUNDARY_RADIUS_PX", 1)),
            normal_delta_px=float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_DELTA_PX", 1.5)),
            normal_margin=float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_MARGIN", 1.0)),
            normal_smooth_radius_px=int(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_SMOOTH_RADIUS_PX", 1)),
            normal_min_grad=float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_MIN_GRAD", 0.03)),
            normal_orientation_fg=float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_ORIENTATION_FG", 0.65)),
            normal_orientation_bg=float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_ORIENTATION_BG", 0.35)),
        )
        loss = loss + rbal_normal_weight * rbal_normal

    empty_mask_weight = float(
        _cfg_get(cfg.TRAIN, "BASE_EMPTY_MASK_WEIGHT", 0.0)
    )
    if empty_mask_weight > 0.0:
        empty_case = label_view.flatten(1).sum(dim=1) == 0
        if bool(empty_case.any()):
            loss = loss + empty_mask_weight * prob_view[
                empty_case
            ].flatten(1).mean(dim=1).mean()

    if clip_loss is not None:
        loss = loss + cfg.TRAIN.CLIP_WEIGHT * torch.nan_to_num(
            clip_loss, nan=0.0, posinf=1.0, neginf=1.0
        )
    return loss


def _jbtl14_qabr_deploy_aligned_loss(
    model,
    labels,
    ce_loss,
    dice_loss,
    cfg,
    *,
    case_weights=None,
):
    """Consume QABR's detached-host finite-deployment view.

    This is intentionally *not* part of ``calc_base_loss``: the factual Base/
    Surface branch must keep exactly its original loss graph.  ``qabr.py``
    caches ``host.detach() + finite_corr`` during training, with all QABR side
    inputs detached.  Applying the existing Dice+CE objective to that cached
    tensor therefore trains QABR with the same finite state used at deployment
    while producing zero gradient for Base/PVL/decoder.

    Returns ``(weighted_loss_or_None, raw_loss_or_None)``.
    """
    owner = getattr(model, "module", model)
    qabr = getattr(owner, "qabr", None)
    if qabr is None or not hasattr(qabr, "pop_training_aux_logits"):
        return None, None
    aux_logits = qabr.pop_training_aux_logits()
    if aux_logits is None:
        return None, None

    if aux_logits.ndim == 4 and aux_logits.shape[1] == 1:
        aux_logits = aux_logits[:, 0]
    label_view = labels[:, 0] if labels.ndim == 4 and labels.shape[1] == 1 else labels
    label_view = (label_view > 0.5).to(device=aux_logits.device, dtype=aux_logits.dtype)

    v547_enabled = bool(_cfg_get(cfg.TRAIN, "V547_HARD_CASE_MEMORY_ENABLED", False))
    if v547_enabled:
        per_case = _per_case_base_composite_loss(
            aux_logits,
            label_view,
            float(cfg.TRAIN.DICE_WEIGHT),
            float(cfg.TRAIN.CE_WEIGHT),
        )
        raw_loss = _v547_weighted_mean(per_case, case_weights)
    else:
        raw_loss = (
            float(cfg.TRAIN.DICE_WEIGHT) * dice_loss(aux_logits, label_view)
            + float(cfg.TRAIN.CE_WEIGHT) * ce_loss(aux_logits, label_view.float())
        )

    if not bool(torch.isfinite(raw_loss).all()):
        raise FloatingPointError("[JBTL14_QABR] non-finite deploy-aligned auxiliary loss")
    weight = float(os.environ.get("QABR_V14_AUX_WEIGHT", "1.0"))
    if weight <= 0.0:
        raise ValueError("QABR_V14_AUX_WEIGHT must be > 0")
    return weight * raw_loss, raw_loss.detach()


def _dice_per_case(logits, masks):
    return _dice_per_case_probs(torch.sigmoid(logits), masks)


def _as_single_channel_spatial_map(tensor, *, name):
    """Canonicalize a binary spatial tensor to ``[B,H,W]``.

    Validation outputs are not fully shape-uniform: Base/Fusion paths commonly
    expose ``[B,H,W]``, while the V538/V541 hard composer intentionally retains
    a singleton channel as ``[B,1,H,W]``.  Letting PyTorch broadcast those two
    forms silently creates ``[B,B,H,W]`` tensors.  The old Dice helper then
    failed only at validation-time reduction, after an entire training epoch.

    Accept exactly the two semantically equivalent single-channel layouts and
    reject every ambiguous multi-channel layout instead of broadcasting it.
    """
    if tensor.ndim == 4:
        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must have one channel when 4D; got {tuple(tensor.shape)}"
            )
        tensor = tensor[:, 0]
    elif tensor.ndim != 3:
        raise ValueError(
            f"{name} must be [B,H,W] or [B,1,H,W]; got {tuple(tensor.shape)}"
        )
    return tensor


def _dice_per_case_probs(probabilities, masks):
    probabilities = _as_single_channel_spatial_map(
        probabilities, name="probabilities"
    )
    masks = _as_single_channel_spatial_map(masks, name="masks")
    if probabilities.shape != masks.shape:
        raise ValueError(
            "Probability/mask spatial shape mismatch after channel normalization: "
            f"{tuple(probabilities.shape)} vs {tuple(masks.shape)}"
        )
    masks = (masks > 0.5).float()
    pred = (probabilities > 0.5).float()
    inter = (pred * masks).sum(dim=(1, 2))
    den = pred.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    return (2.0 * inter + 1e-7) / (den + 1e-7)


def _teacher_anchor_loss(student_logits, teacher_logits, cfg):
    """Confident-teacher probabilistic anchor.  Gradients only update Student Base."""
    if teacher_logits.ndim == 4:
        teacher_logits = teacher_logits[:, 0]
    if student_logits.ndim == 4:
        student_logits = student_logits[:, 0]
    teacher_prob = torch.sigmoid(teacher_logits).detach()
    confidence = torch.maximum(teacher_prob, 1.0 - teacher_prob)
    threshold = float(_cfg_get(cfg.M1, "TEACHER_CONFIDENCE_THRESHOLD", 0.80))
    mask = (confidence >= threshold).float()
    per_pixel = F.binary_cross_entropy_with_logits(student_logits, teacher_prob, reduction="none")
    denominator = mask.sum().clamp_min(1.0)
    loss = (per_pixel * mask).sum() / denominator
    agreement = ((torch.sigmoid(student_logits) - teacher_prob).abs() * mask).sum() / denominator
    return loss, {
        "anchor_confident_fraction": mask.mean().detach(),
        "anchor_abs_probability_gap": agreement.detach(),
    }


def _v20_greedy_set_oracle(candidate_probs, masks, action_types, max_actions=4):
    """Val-only greedy oracle set; never used by inference/training selection."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    gt = (masks > 0.5).float()
    b, slots, h, w = candidate_probs.shape
    base = candidate_probs[:, 0]
    current = base.clone()
    selected = torch.zeros((b, slots - 1), dtype=torch.bool, device=base.device)
    def dice(x):
        pred = (x >= 0.5).float()
        inter = (pred * gt).sum(dim=(1,2)); den = pred.sum(dim=(1,2)) + gt.sum(dim=(1,2))
        return (2.0 * inter + 1e-7) / (den + 1e-7)
    current_dice = dice(current)
    for _ in range(max_actions):
        best_gain = torch.zeros_like(current_dice)
        best_index = torch.full((b,), -1, dtype=torch.long, device=base.device)
        best_mask = current
        for k in range(slots - 1):
            cand = candidate_probs[:, k + 1]
            # Candidate action already encodes one edit from Preserve. Compose only
            # its signed local delta with current mask.
            proposal = (current + (cand - base)).clamp(0.0, 1.0)
            gain = dice(proposal) - current_dice
            gain = gain.masked_fill(selected[:, k], -1e6)
            take = gain > best_gain
            best_gain = torch.where(take, gain, best_gain)
            best_index = torch.where(take, torch.full_like(best_index, k), best_index)
            best_mask = torch.where(take[:, None, None], proposal, best_mask)
        accept = best_index >= 0
        if not accept.any():
            break
        selected.scatter_(1, best_index.clamp_min(0)[:, None], accept[:, None])
        current = torch.where(accept[:, None, None], best_mask, current)
        current_dice = dice(current)
    return current_dice, selected


def _surface_dice_proxy(probabilities, masks, tolerance: int = 2):
    """Hard train-resolution surface Dice used for V407 Val/model selection.

    It is not the native-resolution Test evaluator.  It is a fixed diagnostic
    and supervision proxy so selection never trades a visible boundary collapse
    for a small overlap-only Dice gain.
    """
    if masks.ndim == 4:
        masks = masks[:, 0]
    target = (masks > 0.5).float()
    if probabilities.ndim == 3:
        probabilities = probabilities[:, None]
    pred = (probabilities > 0.5).float()
    b, k, h, w = pred.shape
    gt = target[:, None].expand(b, k, h, w)
    x = pred.reshape(b * k, 1, h, w)
    y = gt.reshape(b * k, 1, h, w)
    ex = -F.max_pool2d(-x, 3, 1, 1)
    ey = -F.max_pool2d(-y, 3, 1, 1)
    bx = (x - ex).clamp(0.0, 1.0)
    by = (y - ey).clamp(0.0, 1.0)
    radius = max(0, int(tolerance))
    near_x = F.max_pool2d(bx, 2 * radius + 1, 1, radius)
    near_y = F.max_pool2d(by, 2 * radius + 1, 1, radius)
    close_x = (bx * near_y).sum(dim=(1, 2, 3))
    close_y = (by * near_x).sum(dim=(1, 2, 3))
    denominator = bx.sum(dim=(1, 2, 3)) + by.sum(dim=(1, 2, 3))
    score = (close_x + close_y) / denominator.clamp_min(1e-6)
    area_x = x.sum(dim=(1, 2, 3))
    area_y = y.sum(dim=(1, 2, 3))
    score = torch.where((area_x == 0) & (area_y == 0), torch.ones_like(score), score)
    score = torch.where((area_x == 0) ^ (area_y == 0), torch.zeros_like(score), score)
    return score.reshape(b, k)


def _validation_tail_metrics(values, cfg, prefix: str):
    """Return lower-tail statistics and a pre-declared robust selection score."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_q10": 0.0,
            f"{prefix}_cvar": 0.0,
            f"{prefix}_catastrophic_rate": 1.0,
            f"{prefix}_tail_score": float("-inf"),
        }

    quantile = float(_cfg_get(cfg.TRAIN, "VAL_TAIL_QUANTILE", 0.10))
    quantile = min(max(quantile, 0.0), 1.0)
    tail_fraction = float(_cfg_get(cfg.TRAIN, "VAL_TAIL_FRACTION", 0.10))
    tail_fraction = min(max(tail_fraction, 1.0 / array.size), 1.0)
    catastrophic_threshold = float(
        _cfg_get(cfg.TRAIN, "VAL_CATASTROPHIC_DSC_THRESHOLD", 0.50)
    )

    q_value = float(np.quantile(array, quantile))
    k = max(1, int(math.ceil(array.size * tail_fraction)))
    cvar = float(np.sort(array)[:k].mean())
    catastrophic_rate = float((array < catastrophic_threshold).mean())

    mean_weight = float(_cfg_get(cfg.TRAIN, "VAL_TAIL_SCORE_MEAN_WEIGHT", 0.60))
    q_weight = float(_cfg_get(cfg.TRAIN, "VAL_TAIL_SCORE_Q_WEIGHT", 0.20))
    cvar_weight = float(_cfg_get(cfg.TRAIN, "VAL_TAIL_SCORE_CVAR_WEIGHT", 0.20))
    weight_sum = max(mean_weight + q_weight + cvar_weight, 1.0e-8)
    mean_weight /= weight_sum
    q_weight /= weight_sum
    cvar_weight /= weight_sum
    catastrophic_penalty = float(
        _cfg_get(cfg.TRAIN, "VAL_CATASTROPHIC_RATE_PENALTY", 0.10)
    )

    score = (
        mean_weight * float(array.mean())
        + q_weight * q_value
        + cvar_weight * cvar
        - catastrophic_penalty * catastrophic_rate
    )
    return {
        f"{prefix}_q10": q_value,
        f"{prefix}_cvar": cvar,
        f"{prefix}_catastrophic_rate": catastrophic_rate,
        f"{prefix}_tail_score": score,
    }



def _native_metrics_from_probabilities(
    probability: torch.Tensor,
    mask_names,
    gt_root: str,
    tolerance: int = 2,
) -> tuple[list[float], list[float]]:
    """Compute native DSC and corrected true 2-D NSD used by ``utils/eval.py``."""
    if probability.ndim == 3:
        probability = probability[:, None]
    if probability.ndim != 4:
        raise ValueError(f"Expected [B,K,H,W], got {tuple(probability.shape)}")
    probs = probability.detach().float().cpu().numpy()
    names = list(mask_names)
    dice_scores, nsd_scores = [], []
    for batch_index, name in enumerate(names):
        gt_path = os.path.join(gt_root, str(name))
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Native validation GT missing: {gt_path}")
        gt_binary = gt > 127
        for candidate_index in range(probs.shape[1]):
            pred_small = probs[batch_index, candidate_index] >= 0.5
            pred = cv2.resize(
                pred_small.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            dice, nsd = case_metrics_2d(gt_binary, pred, tolerance=tolerance)
            dice_scores.append(dice)
            nsd_scores.append(nsd)
    batch = probs.shape[0]
    candidates = probs.shape[1]
    dice_array = np.asarray(dice_scores, dtype=np.float64).reshape(batch, candidates)
    nsd_array = np.asarray(nsd_scores, dtype=np.float64).reshape(batch, candidates)
    return dice_array.tolist(), nsd_array.tolist()


def _native_gt_area_ratios(mask_names, gt_root: str) -> list[float]:
    """Read exact native GT masks and return foreground area fractions.

    Used only for validation stratification diagnostics (empty/small/large
    lesion safety); it never participates in training or checkpoint selection.
    """
    ratios = []
    for name in list(mask_names):
        gt_path = os.path.join(gt_root, str(name))
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Native validation GT missing: {gt_path}")
        ratios.append(float((gt > 127).mean()))
    return ratios


def _v521_validation_preserve_warmup_active(model, cfg):
    """Return True when validation should use physical Preserve/Base for M2.

    V521/V522/V523 BUSI-base formal training warms up M1/M2 while the deployed action
    path is still Preserve.  Validation must not evaluate a randomly initialized
    M2 deployment before the pre-declared selection/deploy start epoch, otherwise
    early validation logs show a large artificial M2 degradation although the
    training deployed tensor is still Base.

    This is validation-only.  It does not change training loss, M1/M2 gradients,
    candidate Oracle, PWO diagnostics, or post-warmup deployment.
    """
    if not m1_enabled(cfg):
        return False

    m1_cfg = _cfg_get(cfg, "M1", None)
    v524 = bool(_cfg_get(m1_cfg, "V524_COUNTERFACTUAL_PROMPTED_REGION_ENABLED", False))
    v523 = bool(_cfg_get(m1_cfg, "V523_SEA_LEVEL_UTILITY_COMPOSER_ENABLED", False))
    v522 = bool(_cfg_get(m1_cfg, "V522_PWO_DISTILLED_SEQUENTIAL_LOCAL_ENABLED", False))
    v521 = bool(_cfg_get(m1_cfg, "V521_CANDIDATE_CONDITIONAL_REGION_COMPOSER_ENABLED", False))
    if not (v521 or v522 or v523 or v524):
        return False

    preserve_key = (
        "V524_VAL_PRESERVE_DURING_WARMUP"
        if v524
        else "V523_VAL_PRESERVE_DURING_WARMUP"
        if v523
        else "V522_VAL_PRESERVE_DURING_WARMUP"
        if v522
        else "V521_VAL_PRESERVE_DURING_WARMUP"
    )
    until_key = (
        "V524_VAL_PRESERVE_UNTIL_EPOCH"
        if v524
        else "V523_VAL_PRESERVE_UNTIL_EPOCH"
        if v523
        else "V522_VAL_PRESERVE_UNTIL_EPOCH"
        if v522
        else "V521_VAL_PRESERVE_UNTIL_EPOCH"
    )
    if not bool(_cfg_get(m1_cfg, preserve_key, False)):
        return False

    current_epoch = int(getattr(model, "current_epoch", 0)) + 1

    until_epoch = int(_cfg_get(m1_cfg, until_key, 0))
    if until_epoch <= 0:
        train_cfg = _cfg_get(cfg, "TRAIN", None)
        until_epoch = int(_cfg_get(train_cfg, "VAL_SELECTION_START_EPOCH", 1))

    return current_epoch < until_epoch

def replay_public_repo_validation_rng(model, val_dataloader, device):
    """Replay released MedCLIPSeg validation *RNG consumption* only.

    The public train.py runs validation every epoch.  Its eval forward first
    executes one stochastic PVL pass and then, with num_samples=1, executes a
    second stochastic pass.  Even when the last checkpoint is ultimately used,
    those draws (plus the Val DataLoader iterator seed) change the next epoch's
    RNG trajectory.  This helper reproduces that behavior without using Val to
    select a checkpoint.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            text = batch["text_prompt"]
            # Current model forward collects exactly one pass. Two calls match
            # the released model's discarded regular pass + 1 collected pass.
            model(image=images, text=text, num_samples=1, compute_m1=False)
            model(image=images, text=text, num_samples=1, compute_m1=False)
    if was_training:
        model.train()


def evaluate_validation(model, val_dataloader, device, ce_loss, dice_loss, cfg):
    """Validation under fixed MC seed with DSC and train-resolution NSD audits."""
    model.eval()
    val_losses, base_scores, fusion_scores, oracle_scores, set_oracle_scores = [], [], [], [], []
    m2_scores, m2_nsd_scores = [], []
    shadow_m2_scores, shadow_m2_nsd_scores = [], []
    pwo_scores, pwo_nsd_scores = [], []
    mhcs_global_selected_scores, mhcs_global_selected_nsd_scores = [], []
    mhcs_local_scores, mhcs_local_nsd_scores = [], []
    mhcs_hard_envelope_scores, mhcs_hard_envelope_nsd_scores = [], []
    mhcs_patch_oracle_scores, mhcs_patch_oracle_nsd_scores = [], []
    mhcs_gate_alpha_scores, mhcs_soft_oracle_risk_scores = [], []
    mhcs_effective_rank_scores = []
    deployable_oracle_scores, deployable_oracle_nsd_scores = [], []
    base_nsd_scores, fusion_nsd_scores, oracle_nsd_scores, pareto_dice_scores, pareto_nsd_scores = [], [], [], [], []
    deployable_pareto_dice_scores, deployable_pareto_nsd_scores = [], []
    action_scores = []
    selected_counts = []
    native_enabled = bool(_cfg_get(cfg.TRAIN, "VAL_NATIVE_METRICS", False))
    native_gt_root = os.path.join(str(cfg.DATASET.VAL_PATH), "label")
    native_base_dice, native_base_nsd = [], []
    native_m2_dice, native_m2_nsd = [], []
    native_shadow_m2_dice, native_shadow_m2_nsd = [], []
    native_fusion_dice, native_fusion_nsd = [], []
    native_oracle_dice, native_oracle_nsd = [], []
    native_component_oracle_dice, native_component_oracle_nsd = [], []
    native_mhcs_global_selected_dice, native_mhcs_global_selected_nsd = [], []
    m1_native_scores, m1_native_nsd_scores = [], []
    native_m1_dice, native_m1_nsd = [], []
    # GEOTR v3: paired native metrics for the actual internal operators.
    native_geotr_geometry_dice, native_geotr_geometry_nsd = [], []
    native_geotr_recon_base_dice, native_geotr_recon_base_nsd = [], []
    native_geotr_recon_after_geometry_dice, native_geotr_recon_after_geometry_nsd = [], []
    # Canonical C2R-v2 exact native-resolution WHERE/proposal ceilings.
    native_c2r_context_oracle_dice, native_c2r_context_oracle_nsd = [], []
    native_c2r_candidate_oracle_dice, native_c2r_candidate_oracle_nsd = [], []
    # PC2R-v3.2 exact native deployment-operator decomposition.
    native_pc2r_stage_dice = {k: [] for k in ("canonical", "candidate", "area", "risk", "direction", "spread", "strength")}
    native_pc2r_stage_nsd = {k: [] for k in ("canonical", "candidate", "area", "risk", "direction", "spread", "strength")}
    native_case_area_ratio = []
    # V4C model-resolution GT-only diagnostics. These are validation-only and
    # never participate in checkpoint selection or deployment.
    geotr_v4c_val_diag = {}
    geotr_v4d_val_diag = {}
    geotr_v4e_val_diag = {}
    geotr_v4f_val_diag = {}
    geotr_v4g_val_diag = {}
    sparc_hr_val_diag = {}
    sparc_hr_dice_scores, sparc_hr_nsd_scores = [], []
    sparc_m1_hr_dice_scores, sparc_m1_hr_nsd_scores = [], []
    sparc_base_hr_dice_scores, sparc_base_hr_nsd_scores = [], []
    sparc_lr_hr_gt_dice_scores, sparc_lr_hr_gt_nsd_scores = [], []
    val_mc_samples = (
        int(_cfg_get(cfg.M1, "VAL_NUM_SAMPLES", 10))
        if m1_enabled(cfg)
        else max(1, int(_cfg_get(cfg.TRAIN, "VAL_NUM_SAMPLES", 1)))
    )
    val_seed = (
        int(_cfg_get(cfg.M1, "VAL_MC_SEED", -1))
        if m1_enabled(cfg)
        else int(_cfg_get(cfg.TRAIN, "VAL_MC_SEED", 20260910))
    )
    nsd_tolerance = int(_cfg_get(cfg.M1, "M2_NSD_TOLERANCE_PIXELS", 2)) if m1_enabled(cfg) else 2
    pareto_dsc_eps = float(_cfg_get(cfg.M1, "M2_PARETO_DSC_EPS", 0.001)) if m1_enabled(cfg) else 0.0
    pareto_nsd_eps = float(_cfg_get(cfg.M1, "M2_PARETO_NSD_EPS", 0.001)) if m1_enabled(cfg) else 0.0
    utility_dsc_weight = float(_cfg_get(cfg.M1, "M2_UTILITY_DSC_WEIGHT", 0.60)) if m1_enabled(cfg) else 1.0
    utility_nsd_weight = float(_cfg_get(cfg.M1, "M2_UTILITY_NSD_WEIGHT", 0.40)) if m1_enabled(cfg) else 0.0
    norm = max(utility_dsc_weight + utility_nsd_weight, 1e-6)
    utility_dsc_weight, utility_nsd_weight = utility_dsc_weight / norm, utility_nsd_weight / norm

    state = _capture_rng_state() if val_seed >= 0 else None
    if val_seed >= 0:
        _seed_only(val_seed)
    try:
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation MC={val_mc_samples}"):
                images = batch["image"].to(device)
                images_hr = batch.get("image_hr", None)
                if isinstance(images_hr, torch.Tensor):
                    images_hr = images_hr.to(device)
                masks = batch["ground_truth_mask"].to(device)
                masks_hr = batch.get("ground_truth_mask_hr", None)
                if isinstance(masks_hr, torch.Tensor):
                    masks_hr = masks_hr.to(device)
                if m1_uses_unified_action_cf(cfg):
                    pred = model.predict_m1_diagnostics(
                        images, batch["text_prompt"], num_samples=val_mc_samples, slr_hr_image=images_hr
                    )
                    if (
                        bool(_cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False))
                        and isinstance(masks_hr, torch.Tensor)
                        and isinstance(pred.get("sparc_final_prob_hr"), torch.Tensor)
                    ):
                        _sparc_hr_prob = pred["sparc_final_prob_hr"]
                        sparc_hr_dice_scores.extend(
                            _dice_per_case_probs(_sparc_hr_prob, masks_hr).cpu().tolist()
                        )
                        sparc_hr_nsd_scores.extend(
                            _surface_dice_proxy(
                                _sparc_hr_prob, masks_hr,
                                max(1, 2 * nsd_tolerance),
                            )[:, 0].cpu().tolist()
                        )
                        _sparc_m1_hr = pred.get("sparc_anchor_prob_hr")
                        _sparc_base_hr = pred.get("sparc_base_prob_hr")
                        if isinstance(_sparc_m1_hr, torch.Tensor):
                            sparc_m1_hr_dice_scores.extend(
                                _dice_per_case_probs(_sparc_m1_hr, masks_hr).cpu().tolist()
                            )
                            sparc_m1_hr_nsd_scores.extend(
                                _surface_dice_proxy(
                                    _sparc_m1_hr, masks_hr, max(1, 2 * nsd_tolerance)
                                )[:, 0].cpu().tolist()
                            )
                        if isinstance(_sparc_base_hr, torch.Tensor):
                            sparc_base_hr_dice_scores.extend(
                                _dice_per_case_probs(_sparc_base_hr, masks_hr).cpu().tolist()
                            )
                            sparc_base_hr_nsd_scores.extend(
                                _surface_dice_proxy(
                                    _sparc_base_hr, masks_hr, max(1, 2 * nsd_tolerance)
                                )[:, 0].cpu().tolist()
                            )
                        # A paired-HR metric is meaningful only if the LR/HR
                        # labels describe the same geometry. Log this audit on
                        # every run so a data-registration failure cannot be
                        # misdiagnosed as an M2 architecture failure.
                        _lr_mask_hr = F.interpolate(
                            masks[:, None].float(), size=masks_hr.shape[-2:], mode="nearest"
                        ).to(masks_hr)
                        sparc_lr_hr_gt_dice_scores.extend(
                            _dice_per_case_probs(_lr_mask_hr, masks_hr).cpu().tolist()
                        )
                        sparc_lr_hr_gt_nsd_scores.extend(
                            _surface_dice_proxy(
                                _lr_mask_hr, masks_hr, max(1, 2 * nsd_tolerance)
                            )[:, 0].cpu().tolist()
                        )
                        _sparc_diag = compute_sparc_hr_validation_diagnostics(
                            cfg, masks_hr, pred
                        )
                        for _key, _value in _sparc_diag.items():
                            if isinstance(_value, torch.Tensor):
                                _value = float(_value.detach().mean().cpu())
                            sparc_hr_val_diag.setdefault(_key, []).append(float(_value))
                    if (
                        bool(_cfg_get(cfg.M1, "GEOTR_TYPED_RESIDUAL_ENABLED", False))
                        and not bool(_cfg_get(cfg.M1, "GEOTR_V4D_ROOT_FIX_ENABLED", False))
                        and not bool(_cfg_get(cfg.M1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False))
                        and not bool(_cfg_get(cfg.M1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False))
                        and not bool(_cfg_get(cfg.M1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False))
                    ):
                        _v4c_diag = compute_geotr_v4c_validation_diagnostics(cfg, masks, pred)
                        for _k, _v in _v4c_diag.items():
                            if isinstance(_v, torch.Tensor):
                                _v = float(_v.detach().mean().cpu())
                            geotr_v4c_val_diag.setdefault(_k, []).append(float(_v))
                    if bool(_cfg_get(cfg.M1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False)):
                        _v4g_diag = compute_geotr_v4g_validation_diagnostics(cfg, masks, pred)
                        for _k, _v in _v4g_diag.items():
                            if isinstance(_v, torch.Tensor):
                                _v = float(_v.detach().mean().cpu())
                            geotr_v4g_val_diag.setdefault(_k, []).append(float(_v))
                    elif bool(_cfg_get(cfg.M1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)):
                        _v4f_diag = compute_geotr_v4f_validation_diagnostics(cfg, masks, pred)
                        for _k, _v in _v4f_diag.items():
                            if isinstance(_v, torch.Tensor):
                                _v = float(_v.detach().mean().cpu())
                            geotr_v4f_val_diag.setdefault(_k, []).append(float(_v))
                    elif bool(_cfg_get(cfg.M1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)):
                        _v4e_diag = compute_geotr_v4e_validation_diagnostics(cfg, masks, pred)
                        for _k, _v in _v4e_diag.items():
                            if isinstance(_v, torch.Tensor):
                                _v = float(_v.detach().mean().cpu())
                            geotr_v4e_val_diag.setdefault(_k, []).append(float(_v))
                    elif bool(_cfg_get(cfg.M1, "GEOTR_V4D_ROOT_FIX_ENABLED", False)):
                        _v4d_diag = compute_geotr_v4d_validation_diagnostics(cfg, masks, pred)
                        for _k, _v in _v4d_diag.items():
                            if isinstance(_v, torch.Tensor):
                                _v = float(_v.detach().mean().cpu())
                            geotr_v4d_val_diag.setdefault(_k, []).append(float(_v))
                    base_probs = pred["base_probs"]
                    candidate_probs = pred["candidate_probs"]
                    fused_probs = pred["fused_probs"]

                    # V521/V522 formal BUSI-base warmup:
                    # validation must follow the same physical deployment
                    # curriculum as training.  Before the pre-declared deploy /
                    # selection epoch, M2/Fusion are forced to Preserve/Base.
                    # Candidate oracle, PWO and action diagnostics still use
                    # candidate_probs, so M1 capacity remains auditable.
                    if _v521_validation_preserve_warmup_active(model, cfg):
                        preserve_probs = base_probs
                        fused_probs = preserve_probs
                        pred["fused_probs"] = preserve_probs

                        for _key in (
                            "m2_fused_probs",
                            "m2_proposal_probs",
                            "m2_training_probs",
                            "m2_action_composed_probs",
                            "m2_selected_probs",
                            "direct_fused_probs",
                            "router_fused_probs",
                            "v20_fused_probs",
                            "v20_hard_fused_probs",
                        ):
                            if isinstance(pred.get(_key), torch.Tensor):
                                pred[_key] = preserve_probs

                        for _key in (
                            "v20_selector_hard",
                            "v20_selector_probs",
                            "m1_selector_hard",
                            "m1_selector_soft",
                        ):
                            if isinstance(pred.get(_key), torch.Tensor):
                                pred[_key] = torch.zeros_like(pred[_key])

                        for _key in (
                            "m2_accept",
                            "m2_selected_action",
                            "m2_selection_exact",
                        ):
                            if isinstance(pred.get(_key), torch.Tensor):
                                pred[_key] = torch.zeros_like(pred[_key])
                    base_logits = torch.logit(base_probs.clamp(1e-6, 1.0 - 1e-6))
                    val_losses.append(calc_base_loss(base_logits, masks, ce_loss, dice_loss, cfg).item())

                    base_dice = _dice_per_case_probs(base_probs, masks)
                    slots_dice = torch.stack([
                        _dice_per_case_probs(candidate_probs[:, s], masks)
                        for s in range(candidate_probs.shape[1])
                    ], dim=1)
                    base_nsd = _surface_dice_proxy(base_probs, masks, nsd_tolerance)[:, 0]
                    slots_nsd = _surface_dice_proxy(candidate_probs, masks, nsd_tolerance)
                    fusion_dice = _dice_per_case_probs(fused_probs, masks)
                    fusion_nsd = _surface_dice_proxy(fused_probs, masks, nsd_tolerance)[:, 0]

                    mhcs_global_selected_probs = None
                    mhcs_global_selected_dice_batch = None
                    mhcs_global_selected_nsd_batch = None
                    mhcs_local_probs = None
                    mhcs_local_dice_batch = None
                    mhcs_local_nsd_batch = None
                    mhcs_hard_envelope_dice_batch = None
                    mhcs_hard_envelope_nsd_batch = None
                    mhcs_patch_oracle_dice_batch = None
                    mhcs_patch_oracle_nsd_batch = None
                    if _mhcs(cfg):
                        quality_pred = pred.get("mhcs_quality_pred")
                        if isinstance(quality_pred, torch.Tensor):
                            if quality_pred.ndim != 2 or quality_pred.shape[:2] != candidate_probs.shape[:2]:
                                raise RuntimeError(
                                    "MHCS-R4.6 mean-utility candidate weights must be [B,K] aligned to candidate_probs; "
                                    f"quality={tuple(quality_pred.shape)} candidates={tuple(candidate_probs.shape)}"
                                )
                            selected_idx = quality_pred.argmax(dim=1)
                            mhcs_global_selected_probs = candidate_probs.gather(
                                1,
                                selected_idx[:, None, None, None].expand(
                                    -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
                                ),
                            )[:, 0]
                            mhcs_global_selected_dice_batch = _dice_per_case_probs(
                                mhcs_global_selected_probs, masks
                            )
                            mhcs_global_selected_nsd_batch = _surface_dice_proxy(
                                mhcs_global_selected_probs, masks, nsd_tolerance
                            )[:, 0]

                        mhcs_local_probs = pred.get("mhcs_local_probs")
                        if isinstance(mhcs_local_probs, torch.Tensor):
                            if mhcs_local_probs.ndim == 4 and mhcs_local_probs.shape[1] == 1:
                                mhcs_local_probs = mhcs_local_probs[:, 0]
                            mhcs_local_dice_batch = _dice_per_case_probs(mhcs_local_probs, masks)
                            mhcs_local_nsd_batch = _surface_dice_proxy(
                                mhcs_local_probs, masks, nsd_tolerance
                            )[:, 0]

                        hard_env = pred.get("mhcs_surface_hard_probs")
                        if isinstance(hard_env, torch.Tensor):
                            if hard_env.ndim == 4 and hard_env.shape[1] == 1:
                                hard_env = hard_env[:, 0]
                            mhcs_hard_envelope_dice_batch = _dice_per_case_probs(hard_env, masks)
                            mhcs_hard_envelope_nsd_batch = _surface_dice_proxy(
                                hard_env, masks, nsd_tolerance
                            )[:, 0]

                        # GT-only R4.6 validation diagnostic: optimize all spatial
                        # regions jointly under the same whole-mask BCE+Dice objective.
                        patch_size = 16
                        try:
                            patch_size = int(str(cfg.MODEL.BACKBONE).rsplit("/", 1)[-1])
                        except Exception:
                            pass
                        target3 = _target_3d(masks).to(candidate_probs)
                        joint_oracle = _joint_oracle_envelope(
                            candidate_probs, target3, patch_size,
                            iterations=int(_cfg_get(cfg.M1, "MHCS_JOINT_ORACLE_ITERS", 4)),
                        )
                        joint_oracle_prob = joint_oracle["prob"].to(candidate_probs)
                        mhcs_patch_oracle_dice_batch = _dice_per_case_probs(
                            joint_oracle_prob, masks
                        )
                        mhcs_patch_oracle_nsd_batch = _surface_dice_proxy(
                            joint_oracle_prob, masks, nsd_tolerance
                        )[:, 0]

                        gate_alpha = pred.get("mhcs_gate_alpha")
                        if isinstance(gate_alpha, torch.Tensor):
                            mhcs_gate_alpha_scores.extend(gate_alpha.detach().reshape(-1).cpu().tolist())
                        effective_rank = pred.get("mhcs_effective_rank")
                        if isinstance(effective_rank, torch.Tensor):
                            mhcs_effective_rank_scores.extend(effective_rank.detach().reshape(-1).cpu().tolist())

                        # Threshold-free soft oracle risk: per pixel use the candidate
                        # with minimum detached Bernoulli NLL. This complements HardPWO.
                        gt_soft = masks[:, 0] if masks.ndim == 4 and masks.shape[1] == 1 else masks
                        cp = candidate_probs.clamp(1.0e-4, 1.0 - 1.0e-4)
                        yy = gt_soft[:, None].expand_as(cp)
                        candidate_risk = -(yy * torch.log(cp) + (1.0 - yy) * torch.log1p(-cp))
                        mhcs_soft_oracle_risk_scores.extend(
                            candidate_risk.min(dim=1).values.mean(dim=(-2, -1)).cpu().tolist()
                        )

                    # V488 M2 is the candidate-constrained pixel composer, while
                    # fused_probs is the M3 safe-deployed output.  Track both so
                    # each module's incremental contribution is visible during
                    # checkpoint selection and ablation analysis.
                    if bool(
                        _cfg_get(cfg.M1, "V527_VALIDATION_USE_PROPOSAL", False)
                        or _cfg_get(cfg.M1, "V528_VALIDATION_USE_PROPOSAL", False)
                        or _cfg_get(cfg.M1, "V529_VALIDATION_USE_PROPOSAL", False)
                        or _cfg_get(cfg.M1, "V530_VALIDATION_USE_PROPOSAL", False)
                    ):
                        raw_m2_probs = pred.get(
                            "m2_proposal_probs", pred.get("m2_fused_probs")
                        )
                    else:
                        raw_m2_probs = pred.get("m2_fused_probs")
                    if _mhcs(cfg):
                        # Legacy checkpoint-selection field ``native_m2_dice``
                        # is kept for protocol compatibility, but in MHCS it is
                        # exactly the set-aware compositional final mask.
                        raw_m2_probs = fused_probs
                    if isinstance(raw_m2_probs, torch.Tensor):
                        m2_probs = raw_m2_probs
                        m2_dice = _dice_per_case_probs(m2_probs, masks)
                        m2_nsd = _surface_dice_proxy(
                            m2_probs, masks, nsd_tolerance
                        )[:, 0]
                    else:
                        m2_probs = None
                        m2_dice = None
                        m2_nsd = None

                    raw_shadow_m2_probs = pred.get(
                        "v541_shadow_selected_final_probability"
                    )
                    if isinstance(raw_shadow_m2_probs, torch.Tensor):
                        shadow_m2_probs = raw_shadow_m2_probs
                        shadow_m2_dice = _dice_per_case_probs(
                            shadow_m2_probs, masks
                        )
                        shadow_m2_nsd = _surface_dice_proxy(
                            shadow_m2_probs, masks, nsd_tolerance
                        )[:, 0]
                    else:
                        shadow_m2_probs = None
                        shadow_m2_dice = None
                        shadow_m2_nsd = None

                    if _semlt(cfg):
                        raw_m1_native_probs = pred.get(
                            "geotopo_geometry_probs", pred.get("mhcs_final_probs", base_probs)
                        )
                    elif _mhcs(cfg) and bool(
                        _cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)
                    ):
                        # SPARC is M2.  Its final mask must never be reported as
                        # the M1-native baseline.  In full mode M1 is Geometry;
                        # in residual mode it is the unchanged Base anchor.
                        _sparc_mode = str(
                            _cfg_get(cfg.M1, "GEOTOPO_MODE", "full")
                        ).strip().lower()
                        if _sparc_mode == "full":
                            raw_m1_native_probs = pred.get("geotopo_geometry_probs")
                        else:
                            raw_m1_native_probs = pred.get(
                                "geotopo_base_probs", base_probs
                            )
                    else:
                        raw_m1_native_probs = pred.get(
                            "mhcs_final_probs",
                            pred.get(
                                "v552r4212_m1_native_probability",
                                pred.get(
                                    "v552r4211_m1_native_probability",
                                    pred.get(
                                        "v552r4210_m1_native_probability",
                                        pred.get("v552r4209_m1_native_probability"),
                                    ),
                                ),
                            ),
                        )
                    if isinstance(raw_m1_native_probs, torch.Tensor):
                        m1_native_probs = raw_m1_native_probs
                        m1_native_dice = _dice_per_case_probs(m1_native_probs, masks)
                        m1_native_nsd = _surface_dice_proxy(
                            m1_native_probs, masks, nsd_tolerance
                        )[:, 0]
                        m1_native_scores.extend(m1_native_dice.cpu().tolist())
                        m1_native_nsd_scores.extend(m1_native_nsd.cpu().tolist())
                    else:
                        m1_native_probs = None

                    # Train-resolution PWO diagnostic.  Preserve C0 wherever it
                    # is correct or no M1 candidate can repair its error; where
                    # at least one candidate is correct, use the GT value.  The
                    # formal original-resolution PWO is computed by the separate
                    # evaluate_v488_pwo_upper_bound.py script.
                    gt_hard = masks
                    if gt_hard.ndim == 4 and gt_hard.shape[1] == 1:
                        gt_hard = gt_hard[:, 0]
                    gt_hard = gt_hard >= 0.5
                    slot_hard = candidate_probs >= 0.5
                    base_hard = slot_hard[:, 0]
                    base_wrong = base_hard.ne(gt_hard)
                    repairable = slot_hard[:, 1:].eq(
                        gt_hard[:, None]
                    ).any(dim=1)
                    pwo_hard = torch.where(
                        base_wrong & repairable,
                        gt_hard,
                        base_hard,
                    )
                    pwo_probs = pwo_hard.to(dtype=base_probs.dtype)
                    pwo_dice = _dice_per_case_probs(pwo_probs, masks)
                    pwo_nsd = _surface_dice_proxy(
                        pwo_probs, masks, nsd_tolerance
                    )[:, 0]

                    # All-candidate Oracle: diagnostic upper bound across C0-C8.
                    oracle = slots_dice.max(dim=1).values

                    # Deployable Oracle: only Preserve plus action types allowed
                    # by the fixed M2/M3 deployment rule.
                    raw_action_types = pred.get("v20_action_types")
                    if not isinstance(raw_action_types, torch.Tensor):
                        raise RuntimeError(
                            "V423 requires v20_action_types tensor for deployable Oracle."
                        )

                    action_types = raw_action_types.reshape(-1).to(
                        device=slots_dice.device,
                        dtype=torch.long,
                    )

                    if action_types.numel() != slots_dice.shape[1] - 1:
                        raise RuntimeError(
                            "V423 action-type count mismatch: "
                            f"{action_types.numel()} vs {slots_dice.shape[1] - 1}"
                        )

                    allowed_types_cfg = _cfg_get(
                        cfg.M1,
                        "EVIDENCE_GUIDED_DEPLOY_ALLOWED_TYPES",
                        None,
                    )

                    if allowed_types_cfg is None:
                        deployable_action = torch.ones(
                            action_types.shape[0],
                            dtype=torch.bool,
                            device=slots_dice.device,
                        )
                    else:
                        allowed_types = {int(value) for value in allowed_types_cfg}
                        deployable_action = torch.zeros(
                            action_types.shape[0],
                            dtype=torch.bool,
                            device=slots_dice.device,
                        )
                        for action_type in allowed_types:
                            deployable_action = (
                                deployable_action
                                | action_types.eq(action_type)
                            )

                    deployable_slots = torch.cat(
                        [
                            torch.ones(
                                1,
                                dtype=torch.bool,
                                device=slots_dice.device,
                            ),
                            deployable_action,
                        ],
                        dim=0,
                    )

                    deployable_oracle = slots_dice.masked_fill(
                        ~deployable_slots[None, :],
                        float("-inf"),
                    ).max(dim=1).values

                    deployable_oracle_nsd = slots_nsd.masked_fill(
                        ~deployable_slots[None, :],
                        float("-inf"),
                    ).max(dim=1).values

                    # Pareto Oracle: must improve both DSC and NSD.
                    d_dice = slots_dice[:, 1:] - slots_dice[:, :1]
                    d_nsd = slots_nsd[:, 1:] - slots_nsd[:, :1]
                    pareto = (
                        (d_dice > pareto_dsc_eps)
                        & (d_nsd > pareto_nsd_eps)
                    )

                    utility = (
                        utility_dsc_weight * d_dice
                        + utility_nsd_weight * d_nsd
                    )

                    masked_utility = utility.masked_fill(
                        ~pareto,
                        float("-inf"),
                    )

                    best_utility, best_idx = masked_utility.max(dim=1)
                    has_pareto = torch.isfinite(best_utility)

                    selected_idx = torch.where(
                        has_pareto,
                        best_idx + 1,
                        torch.zeros_like(best_idx),
                    )

                    pareto_dice = slots_dice.gather(
                        1,
                        selected_idx[:, None],
                    )[:, 0]

                    pareto_nsd = slots_nsd.gather(
                        1,
                        selected_idx[:, None],
                    )[:, 0]

                    # Same Pareto criterion, restricted to deployable types.
                    deployable_pareto = pareto & deployable_action[None, :]

                    deployable_utility = utility.masked_fill(
                        ~deployable_pareto,
                        float("-inf"),
                    )

                    deployable_best_utility, deployable_best_idx = (
                        deployable_utility.max(dim=1)
                    )

                    has_deployable_pareto = torch.isfinite(
                        deployable_best_utility
                    )

                    deployable_selected_idx = torch.where(
                        has_deployable_pareto,
                        deployable_best_idx + 1,
                        torch.zeros_like(deployable_best_idx),
                    )

                    deployable_pareto_dice = slots_dice.gather(
                        1,
                        deployable_selected_idx[:, None],
                    )[:, 0]

                    deployable_pareto_nsd = slots_nsd.gather(
                        1,
                        deployable_selected_idx[:, None],
                    )[:, 0]

                    if _mhcs(cfg):
                        # For full-mask hypotheses the meaningful set upper
                        # bound is the pixel-wise oracle, not additive Base
                        # residual composition.
                        set_oracle = pwo_dice
                    else:
                        set_oracle, _ = _v20_greedy_set_oracle(
                            candidate_probs,
                            masks,
                            pred["v20_action_types"],
                            max_actions=int(_cfg_get(cfg.M1, "V20_ORACLE_MAX_ACTIONS", 4)),
                        )

                    base_scores += base_dice.cpu().tolist()
                    fusion_scores += fusion_dice.cpu().tolist()
                    if m2_dice is not None:
                        m2_scores += m2_dice.cpu().tolist()
                        m2_nsd_scores += m2_nsd.cpu().tolist()
                    if shadow_m2_dice is not None:
                        shadow_m2_scores += shadow_m2_dice.cpu().tolist()
                        shadow_m2_nsd_scores += shadow_m2_nsd.cpu().tolist()
                    pwo_scores += pwo_dice.cpu().tolist()
                    pwo_nsd_scores += pwo_nsd.cpu().tolist()
                    if mhcs_global_selected_dice_batch is not None:
                        mhcs_global_selected_scores += mhcs_global_selected_dice_batch.cpu().tolist()
                        mhcs_global_selected_nsd_scores += mhcs_global_selected_nsd_batch.cpu().tolist()
                    if mhcs_local_dice_batch is not None:
                        mhcs_local_scores += mhcs_local_dice_batch.cpu().tolist()
                        mhcs_local_nsd_scores += mhcs_local_nsd_batch.cpu().tolist()
                    if mhcs_hard_envelope_dice_batch is not None:
                        mhcs_hard_envelope_scores += mhcs_hard_envelope_dice_batch.cpu().tolist()
                        mhcs_hard_envelope_nsd_scores += mhcs_hard_envelope_nsd_batch.cpu().tolist()
                    if mhcs_patch_oracle_dice_batch is not None:
                        mhcs_patch_oracle_scores += mhcs_patch_oracle_dice_batch.cpu().tolist()
                        mhcs_patch_oracle_nsd_scores += mhcs_patch_oracle_nsd_batch.cpu().tolist()
                    oracle_scores += oracle.cpu().tolist()
                    set_oracle_scores += set_oracle.cpu().tolist()
                    base_nsd_scores += base_nsd.cpu().tolist()
                    fusion_nsd_scores += fusion_nsd.cpu().tolist()
                    oracle_nsd_scores += slots_nsd.max(dim=1).values.cpu().tolist()
                    pareto_dice_scores += pareto_dice.cpu().tolist()
                    pareto_nsd_scores += pareto_nsd.cpu().tolist()
                    deployable_oracle_scores += deployable_oracle.cpu().tolist()
                    deployable_oracle_nsd_scores += deployable_oracle_nsd.cpu().tolist()
                    deployable_pareto_dice_scores += deployable_pareto_dice.cpu().tolist()
                    deployable_pareto_nsd_scores += deployable_pareto_nsd.cpu().tolist()
                    action_scores.append(slots_dice[:, 1:].mean(dim=0).cpu())
                    selected_counts.append(pred["v20_selector_hard"].sum(dim=1).float().cpu())
                    if native_enabled:
                        mask_names = batch.get("mask_name", batch.get("image_name"))
                        base_native_d, base_native_n = _native_metrics_from_probabilities(
                            base_probs, mask_names, native_gt_root, nsd_tolerance
                        )
                        native_case_area_ratio.extend(_native_gt_area_ratios(mask_names, native_gt_root))
                        fusion_native_d, fusion_native_n = _native_metrics_from_probabilities(
                            fused_probs, mask_names, native_gt_root, nsd_tolerance
                        )
                        if mhcs_global_selected_probs is not None:
                            global_native_d, global_native_n = _native_metrics_from_probabilities(
                                mhcs_global_selected_probs, mask_names, native_gt_root, nsd_tolerance
                            )
                            native_mhcs_global_selected_dice.extend(row[0] for row in global_native_d)
                            native_mhcs_global_selected_nsd.extend(row[0] for row in global_native_n)
                        if m2_probs is not None:
                            m2_native_d, m2_native_n = _native_metrics_from_probabilities(
                                m2_probs, mask_names, native_gt_root, nsd_tolerance
                            )
                        else:
                            m2_native_d, m2_native_n = [], []
                        if shadow_m2_probs is not None:
                            shadow_m2_native_d, shadow_m2_native_n = (
                                _native_metrics_from_probabilities(
                                    shadow_m2_probs,
                                    mask_names,
                                    native_gt_root,
                                    nsd_tolerance,
                                )
                            )
                        else:
                            shadow_m2_native_d, shadow_m2_native_n = [], []
                        if m1_native_probs is not None:
                            m1_native_d, m1_native_n = _native_metrics_from_probabilities(
                                m1_native_probs, mask_names, native_gt_root, nsd_tolerance
                            )
                        else:
                            m1_native_d, m1_native_n = [], []
                        candidate_native_d, candidate_native_n = _native_metrics_from_probabilities(
                            candidate_probs, mask_names, native_gt_root, nsd_tolerance
                        )
                        component_candidate_probs = pred.get(
                            "v541_slot_exact_candidate_probs"
                        )
                        if isinstance(component_candidate_probs, torch.Tensor):
                            # M2 actually selects from the V538 local component
                            # slots, not from the four global action candidates.
                            component_with_base = torch.cat(
                                [base_probs[:, None], component_candidate_probs],
                                dim=1,
                            )
                            (
                                component_native_d,
                                component_native_n,
                            ) = _native_metrics_from_probabilities(
                                component_with_base,
                                mask_names,
                                native_gt_root,
                                nsd_tolerance,
                            )
                        else:
                            component_native_d, component_native_n = [], []
                        native_base_dice.extend(row[0] for row in base_native_d)
                        native_base_nsd.extend(row[0] for row in base_native_n)

                        # GEOTR v3 internal paired outputs from the SAME checkpoint/MC Base.
                        # These are diagnostics only; formal checkpoint selection remains
                        # native_m2_dice/native_m2_nsd == actual deployable Final.
                        _geo_prob = pred.get("geotopo_geometry_probs")
                        if isinstance(_geo_prob, torch.Tensor):
                            _gd, _gn = _native_metrics_from_probabilities(
                                _geo_prob, mask_names, native_gt_root, nsd_tolerance
                            )
                            native_geotr_geometry_dice.extend(row[0] for row in _gd)
                            native_geotr_geometry_nsd.extend(row[0] for row in _gn)
                        _rb_prob = pred.get("geotopo_residual_only_probs")
                        if isinstance(_rb_prob, torch.Tensor):
                            _rd, _rn = _native_metrics_from_probabilities(
                                _rb_prob, mask_names, native_gt_root, nsd_tolerance
                            )
                            native_geotr_recon_base_dice.extend(row[0] for row in _rd)
                            native_geotr_recon_base_nsd.extend(row[0] for row in _rn)
                        _rg_prob = pred.get("geotopo_reconstruction_after_geometry_probs")
                        if isinstance(_rg_prob, torch.Tensor):
                            _rgd, _rgn = _native_metrics_from_probabilities(
                                _rg_prob, mask_names, native_gt_root, nsd_tolerance
                            )
                            native_geotr_recon_after_geometry_dice.extend(row[0] for row in _rgd)
                            native_geotr_recon_after_geometry_nsd.extend(row[0] for row in _rgn)
                        if bool(_cfg_get(cfg.M1, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False)):
                            _mode = str(_cfg_get(cfg.M1, "GEOTOPO_MODE", "full")).strip().lower()
                            _cprefix = (
                                "geotopo_reconstruction_after_geometry"
                                if _mode == "full" else "geotopo_reconstruction_base"
                            )
                            _anchor_prob = _geo_prob if _mode == "full" else base_probs
                            _context_mask = pred.get(_cprefix + "_c2r_region_mask")
                            _candidate_mask = pred.get(_cprefix + "_c2r_candidate_mask")
                            if all(isinstance(x, torch.Tensor) for x in (_anchor_prob, _context_mask, _candidate_mask)):
                                # Exact BCHW shape contract.  The old A2 path could
                                # mix [B,H,W] Base with [B,1,H,W] masks, causing
                                # torch.where to broadcast to [B,B,H,W] and corrupt
                                # NativeContext/CandidateOracle.
                                if _anchor_prob.ndim == 3:
                                    _anchor_prob = _anchor_prob[:, None]
                                if _context_mask.ndim == 3:
                                    _context_mask = _context_mask[:, None]
                                if _candidate_mask.ndim == 3:
                                    _candidate_mask = _candidate_mask[:, None]
                                if _anchor_prob.ndim != 4 or _anchor_prob.shape[1] != 1:
                                    raise RuntimeError(f"C2R native oracle anchor must be [B,1,H,W], got {tuple(_anchor_prob.shape)}")
                                if _context_mask.shape != _anchor_prob.shape or _candidate_mask.shape != _anchor_prob.shape:
                                    raise RuntimeError(
                                        "C2R native oracle shape mismatch: "
                                        f"anchor={tuple(_anchor_prob.shape)} context={tuple(_context_mask.shape)} candidate={tuple(_candidate_mask.shape)}"
                                    )
                                _gt3 = _target_3d(masks).to(_anchor_prob)
                                _gt4 = _gt3[:, None]
                                if _gt4.shape[-2:] != _anchor_prob.shape[-2:]:
                                    _gt4 = F.interpolate(
                                        _gt4.float(), size=_anchor_prob.shape[-2:], mode="nearest"
                                    ).to(_anchor_prob)
                                _context_oracle_prob = torch.where(
                                    _context_mask.detach() > 0.5, _gt4, _anchor_prob.detach()
                                )
                                _candidate_oracle_prob = torch.where(
                                    _candidate_mask.detach() > 0.5, _gt4, _anchor_prob.detach()
                                )
                                _cod, _con = _native_metrics_from_probabilities(
                                    _context_oracle_prob, mask_names, native_gt_root, nsd_tolerance
                                )
                                _cad, _can = _native_metrics_from_probabilities(
                                    _candidate_oracle_prob, mask_names, native_gt_root, nsd_tolerance
                                )
                                native_c2r_context_oracle_dice.extend(row[0] for row in _cod)
                                native_c2r_context_oracle_nsd.extend(row[0] for row in _con)
                                native_c2r_candidate_oracle_dice.extend(row[0] for row in _cad)
                                native_c2r_candidate_oracle_nsd.extend(row[0] for row in _can)

                                if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)):
                                    _stage_keys = {
                                        "canonical": _cprefix + "_c2r_mean_prob",
                                        "candidate": _cprefix + "_pc2r_stage_candidate_prob",
                                        "area": _cprefix + "_pc2r_stage_area_prob",
                                        "risk": _cprefix + "_pc2r_stage_risk_prob",
                                        "direction": _cprefix + "_pc2r_stage_direction_prob",
                                        "spread": _cprefix + "_pc2r_stage_spread_prob",
                                        "strength": _cprefix + "_pc2r_stage_strength_prob",
                                    }
                                    for _stage, _key in _stage_keys.items():
                                        _sp = pred.get(_key)
                                        if isinstance(_sp, torch.Tensor):
                                            _sd, _sn = _native_metrics_from_probabilities(
                                                _sp, mask_names, native_gt_root, nsd_tolerance
                                            )
                                            native_pc2r_stage_dice[_stage].extend(row[0] for row in _sd)
                                            native_pc2r_stage_nsd[_stage].extend(row[0] for row in _sn)
                        if m2_native_d:
                            native_m2_dice.extend(row[0] for row in m2_native_d)
                            native_m2_nsd.extend(row[0] for row in m2_native_n)
                        if shadow_m2_native_d:
                            native_shadow_m2_dice.extend(
                                row[0] for row in shadow_m2_native_d
                            )
                            native_shadow_m2_nsd.extend(
                                row[0] for row in shadow_m2_native_n
                            )
                        if m1_native_d:
                            native_m1_dice.extend(row[0] for row in m1_native_d)
                            native_m1_nsd.extend(row[0] for row in m1_native_n)
                        native_fusion_dice.extend(row[0] for row in fusion_native_d)
                        native_fusion_nsd.extend(row[0] for row in fusion_native_n)
                        native_oracle_dice.extend(max(row) for row in candidate_native_d)
                        native_oracle_nsd.extend(max(row) for row in candidate_native_n)
                        if component_native_d:
                            native_component_oracle_dice.extend(
                                max(row) for row in component_native_d
                            )
                            native_component_oracle_nsd.extend(
                                max(row) for row in component_native_n
                            )

                elif m1_enabled(cfg):
                    prediction = model.predict_m1_diagnostics(
                        images, batch["text_prompt"], num_samples=val_mc_samples, slr_hr_image=images_hr
                    )
                    base_probs = prediction["base_probs"]
                    candidate_probs = prediction["candidate_probs"]
                    fusion_probs = prediction["fused_probs"]
                    base_logits = torch.logit(base_probs.clamp(1e-6, 1.0 - 1e-6))
                    val_losses.append(calc_base_loss(base_logits, masks, ce_loss, dice_loss, cfg).item())
                    base_dice = _dice_per_case_probs(base_probs, masks)
                    role_dice = torch.stack([
                        _dice_per_case_probs(candidate_probs[:, role], masks)
                        for role in range(candidate_probs.shape[1])
                    ], dim=1)
                    fusion_dice = _dice_per_case_probs(fusion_probs, masks)
                    base_nsd = _surface_dice_proxy(base_probs, masks, nsd_tolerance)[:, 0]
                    fusion_nsd = _surface_dice_proxy(fusion_probs, masks, nsd_tolerance)[:, 0]
                    base_scores.extend(base_dice.cpu().tolist())
                    oracle_scores.extend(role_dice.max(dim=1).values.cpu().tolist())
                    fusion_scores.extend(fusion_dice.cpu().tolist())
                    base_nsd_scores.extend(base_nsd.cpu().tolist())
                    fusion_nsd_scores.extend(fusion_nsd.cpu().tolist())
                    oracle_nsd_scores.extend(
                        _surface_dice_proxy(candidate_probs, masks, nsd_tolerance).max(dim=1).values.cpu().tolist()
                    )
                else:
                    # Fair A0 Base-only validation.
                    #
                    # Fusion and Oracle are exactly C0, but native-resolution
                    # metrics must still be computed so A0 follows the same
                    # best-Val checkpoint-selection protocol as all M1 runs.
                    logits, _, _ = model(
                        images,
                        text=batch["text_prompt"],
                        num_samples=val_mc_samples,
                        return_aux=True,
                    )

                    val_losses.append(
                        calc_base_loss(
                            logits,
                            masks,
                            ce_loss,
                            dice_loss,
                            cfg,
                        ).item()
                    )

                    base_dice = _dice_per_case(
                        logits,
                        masks,
                    )

                    base_probs = torch.sigmoid(
                        logits
                    )

                    base_nsd = _surface_dice_proxy(
                        base_probs,
                        masks,
                        nsd_tolerance,
                    )[:, 0]

                    base_scores.extend(
                        base_dice.cpu().tolist()
                    )
                    oracle_scores.extend(
                        base_dice.cpu().tolist()
                    )
                    fusion_scores.extend(
                        base_dice.cpu().tolist()
                    )

                    base_nsd_scores.extend(
                        base_nsd.cpu().tolist()
                    )
                    fusion_nsd_scores.extend(
                        base_nsd.cpu().tolist()
                    )
                    oracle_nsd_scores.extend(
                        base_nsd.cpu().tolist()
                    )

                    if native_enabled:
                        mask_names = batch.get(
                            "mask_name",
                            batch.get("image_name"),
                        )

                        (
                            base_native_d,
                            base_native_n,
                        ) = _native_metrics_from_probabilities(
                            base_probs,
                            mask_names,
                            native_gt_root,
                            nsd_tolerance,
                        )

                        native_base_dice.extend(
                            row[0]
                            for row in base_native_d
                        )
                        native_base_nsd.extend(
                            row[0]
                            for row in base_native_n
                        )

                        # In A0, Final/Fusion/Oracle are all C0.
                        native_fusion_dice.extend(
                            row[0]
                            for row in base_native_d
                        )
                        native_fusion_nsd.extend(
                            row[0]
                            for row in base_native_n
                        )
                        native_oracle_dice.extend(
                            row[0]
                            for row in base_native_d
                        )
                        native_oracle_nsd.extend(
                            row[0]
                            for row in base_native_n
                        )
    finally:
        if state is not None:
            _restore_rng_state(state)
        model.train()

    base_mean = mean(base_scores)
    base_nsd_mean = mean(base_nsd_scores)
    fusion_mean = mean(fusion_scores)
    fusion_nsd_mean = mean(fusion_nsd_scores)
    m2_mean = mean(m2_scores) if m2_scores else fusion_mean
    m2_nsd_mean = mean(m2_nsd_scores) if m2_nsd_scores else fusion_nsd_mean
    shadow_m2_mean = mean(shadow_m2_scores) if shadow_m2_scores else m2_mean
    shadow_m2_nsd_mean = (
        mean(shadow_m2_nsd_scores) if shadow_m2_nsd_scores else m2_nsd_mean
    )
    pwo_mean = mean(pwo_scores) if pwo_scores else mean(oracle_scores)
    pwo_nsd_mean = mean(pwo_nsd_scores) if pwo_nsd_scores else mean(oracle_nsd_scores)
    common = {
        "loss": mean(val_losses),
        "base_dice": base_mean,
        "fusion_dice": fusion_mean,
        "m2_dice": m2_mean,
        "shadow_m2_dice": shadow_m2_mean,
        "pwo_dice": pwo_mean,
        "mhcs_global_selected_dice": (
            mean(mhcs_global_selected_scores) if mhcs_global_selected_scores else base_mean
        ),
        "mhcs_global_selected_nsd": (
            mean(mhcs_global_selected_nsd_scores) if mhcs_global_selected_nsd_scores else base_nsd_mean
        ),
        "mhcs_local_dice": mean(mhcs_local_scores) if mhcs_local_scores else fusion_mean,
        "mhcs_local_nsd": mean(mhcs_local_nsd_scores) if mhcs_local_nsd_scores else fusion_nsd_mean,
        "mhcs_hard_envelope_dice": mean(mhcs_hard_envelope_scores) if mhcs_hard_envelope_scores else fusion_mean,
        "mhcs_hard_envelope_nsd": mean(mhcs_hard_envelope_nsd_scores) if mhcs_hard_envelope_nsd_scores else fusion_nsd_mean,
        "mhcs_patch_oracle_dice": mean(mhcs_patch_oracle_scores) if mhcs_patch_oracle_scores else mean(oracle_scores),
        "mhcs_patch_oracle_nsd": mean(mhcs_patch_oracle_nsd_scores) if mhcs_patch_oracle_nsd_scores else mean(oracle_nsd_scores),
        "mhcs_gate_alpha": mean(mhcs_gate_alpha_scores) if mhcs_gate_alpha_scores else 0.0,
        "mhcs_soft_oracle_risk": mean(mhcs_soft_oracle_risk_scores) if mhcs_soft_oracle_risk_scores else 0.0,
        "mhcs_effective_rank": mean(mhcs_effective_rank_scores) if mhcs_effective_rank_scores else 0.0,
        "base_nsd": base_nsd_mean,
        "fusion_nsd": fusion_nsd_mean,
        "m2_nsd": m2_nsd_mean,
        "shadow_m2_nsd": shadow_m2_nsd_mean,
        "pwo_nsd": pwo_nsd_mean,
        "oracle_dice": mean(oracle_scores),
        "oracle_nsd": mean(oracle_nsd_scores),
        "oracle_gain": mean(oracle_scores) - base_mean,
        "deployable_oracle_dice": (
            mean(deployable_oracle_scores)
            if deployable_oracle_scores else mean(oracle_scores)
        ),
        "deployable_oracle_nsd": (
            mean(deployable_oracle_nsd_scores)
            if deployable_oracle_nsd_scores else mean(oracle_nsd_scores)
        ),
        "deployable_oracle_gain": (
            mean(deployable_oracle_scores) - base_mean
            if deployable_oracle_scores else mean(oracle_scores) - base_mean
        ),
        "m1_native_dice": mean(m1_native_scores) if m1_native_scores else base_mean,
        "m1_native_nsd": mean(m1_native_nsd_scores) if m1_native_nsd_scores else base_nsd_mean,
        "m1_native_gain": (mean(m1_native_scores) - base_mean) if m1_native_scores else 0.0,
        "fusion_gain": fusion_mean - base_mean,
        "fusion_nsd_gain": fusion_nsd_mean - base_nsd_mean,
        "m2_gain": m2_mean - base_mean,
        "m2_nsd_gain": m2_nsd_mean - base_nsd_mean,
        "shadow_m2_gain": shadow_m2_mean - base_mean,
        "shadow_m2_nsd_gain": shadow_m2_nsd_mean - base_nsd_mean,
        "m3_gain_vs_m2": fusion_mean - m2_mean,
        "m3_nsd_gain_vs_m2": fusion_nsd_mean - m2_nsd_mean,
        "pwo_gain": pwo_mean - base_mean,
        "pwo_gap_over_global_oracle": pwo_mean - mean(oracle_scores),
        "m2_gap_to_pwo": pwo_mean - m2_mean,
        "final_gap_to_pwo": pwo_mean - fusion_mean,
        "mc_samples": val_mc_samples,
        "mc_seed": val_seed,
    }
    if geotr_v4c_val_diag:
        common.update({
            _k: mean(_vals) for _k, _vals in geotr_v4c_val_diag.items() if _vals
        })
    if sparc_hr_dice_scores:
        common.update({
            "native_sparc_hr_dice": mean(sparc_hr_dice_scores),
            "native_sparc_hr_nsd": mean(sparc_hr_nsd_scores),
            "native_sparc_hr_cases": len(sparc_hr_dice_scores),
            "native_sparc_m1_hr_dice": mean(sparc_m1_hr_dice_scores),
            "native_sparc_m1_hr_nsd": mean(sparc_m1_hr_nsd_scores),
            "native_sparc_base_hr_dice": mean(sparc_base_hr_dice_scores),
            "native_sparc_base_hr_nsd": mean(sparc_base_hr_nsd_scores),
            "sparc_lr_hr_gt_dice": mean(sparc_lr_hr_gt_dice_scores),
            "sparc_lr_hr_gt_nsd": mean(sparc_lr_hr_gt_nsd_scores),
        })
    if sparc_hr_val_diag:
        for _key, _values in sparc_hr_val_diag.items():
            if not _values:
                continue
            common[_key] = (
                sum(_values)
                if (_key.endswith("_count") or _key.endswith("_sum"))
                else mean(_values)
            )
    if geotr_v4g_val_diag:
        # V4G/R4 diagnostics mix additive counts with batch-level means.  Sum
        # counts first, retain historical batch-macro ratios explicitly, then
        # recompute the public precision/recall/rates from global counts.
        for _k, _vals in geotr_v4g_val_diag.items():
            if not _vals:
                continue
            common[_k] = sum(_vals) if (_k.endswith("_count") or _k.endswith("_sum")) else mean(_vals)
        for _k in (
            "val_geotr_v4g_selection_precision",
            "val_geotr_v4g_selection_recall",
            "val_geotr_v4g_anchor_error_rate",
            "val_geotr_v4g_correction_recall",
            "val_geotr_v4g_introduction_rate",
        ):
            if _k in common:
                common[_k + "_macro_batch"] = common[_k]
        _sel_err = common.get("val_geotr_v4g_selected_error_count", 0.0)
        _sel_cor = common.get("val_geotr_v4g_selected_correct_count", 0.0)
        _sel_all = common.get("val_geotr_v4g_selected_count", _sel_err + _sel_cor)
        _tot_err = common.get("val_geotr_v4g_total_error_count", 0.0)
        _tot_pix = common.get("val_geotr_v4g_total_pixel_count", 0.0)
        if _sel_all > 0:
            common["val_geotr_v4g_selection_precision"] = _sel_err / _sel_all
        if _tot_err > 0:
            common["val_geotr_v4g_selection_recall"] = _sel_err / _tot_err
        if _tot_pix > 0:
            common["val_geotr_v4g_anchor_error_rate"] = _tot_err / _tot_pix
        _corr = common.get("val_geotr_v4g_corrected_error_count", 0.0)
        _intro = common.get("val_geotr_v4g_introduced_error_count", 0.0)
        common["val_geotr_v4g_net_correction_count"] = _corr - _intro
        if _tot_err > 0:
            common["val_geotr_v4g_correction_recall"] = _corr / _tot_err
        if _sel_all > 0:
            common["val_geotr_v4g_introduction_rate"] = _intro / _sel_all
        # Canonical C2R public intervention metrics use global micro counts.
        # Preserve historical mean-of-batch ratios under *_macro_batch.
        if "val_geotr_c2r_edit_precision" in common:
            common["val_geotr_c2r_edit_precision_macro_batch"] = common["val_geotr_c2r_edit_precision"]
        if (_corr + _intro) > 0:
            common["val_geotr_c2r_edit_precision"] = _corr / (_corr + _intro)
        _cand_comp = common.get("val_geotr_c2r_candidate_component_count", 0.0)
        _comm_comp = common.get("val_geotr_c2r_committed_component_count", 0.0)
        _cand_area = common.get("val_geotr_c2r_candidate_component_area_total_sum", 0.0)
        _comm_area = common.get("val_geotr_c2r_committed_component_area_total_sum", 0.0)
        if _cand_comp > 0:
            common["val_geotr_c2r_candidate_component_area_mean"] = _cand_area / _cand_comp
        if _comm_comp > 0:
            common["val_geotr_c2r_committed_component_area_mean"] = _comm_area / _comm_comp
        _c2r_cases = common.get("val_geotr_c2r_case_count", 0.0)
        if _c2r_cases > 0:
            common["val_geotr_c2r_harm_case_rate"] = common.get("val_geotr_c2r_harm_case_count", 0.0) / _c2r_cases
            common["val_geotr_c2r_benefit_case_rate"] = common.get("val_geotr_c2r_benefit_case_count", 0.0) / _c2r_cases
        _reach = common.get("val_pc2r_reachable_region_error_count", 0.0)
        _unreach = common.get("val_pc2r_unreachable_region_error_count", 0.0)
        if (_reach + _unreach) > 0:
            common["val_pc2r_reachable_region_error_fraction"] = _reach / (_reach + _unreach)
        for _name in ("error", "correct", "candidate"):
            _anum = common.get(f"val_geotr_c2r_agreement_{_name}_count", 0.0)
            _aden = common.get(f"val_geotr_c2r_agreement_{_name}_total_count", 0.0)
            if _aden > 0:
                common[f"val_geotr_c2r_agreement_{_name}"] = _anum / _aden
            _ssum = common.get(f"val_geotr_c2r_spread_{_name}_sum", 0.0)
            _scnt = common.get(f"val_geotr_c2r_spread_{_name}_count", 0.0)
            if _scnt > 0:
                common[f"val_geotr_c2r_spread_{_name}"] = _ssum / _scnt
        _f_tp = common.get("val_geotr_v4g_r4_flip_tp_count", 0.0)
        _f_fp = common.get("val_geotr_v4g_r4_flip_fp_count", 0.0)
        if (_f_tp + _f_fp) > 0:
            common["val_geotr_v4g_r4_flip_precision"] = _f_tp / (_f_tp + _f_fp)
        if _sel_err > 0:
            common["val_geotr_v4g_r4_flip_recall_selected"] = _f_tp / _sel_err
        if _sel_cor > 0:
            common["val_geotr_v4g_r4_false_flip_rate"] = _f_fp / _sel_cor
        if _sel_all > 0:
            common["val_geotr_v4g_r4_effective_edit_rate"] = (_f_tp + _f_fp) / _sel_all
    if geotr_v4f_val_diag:
        for _k, _vals in geotr_v4f_val_diag.items():
            if _vals:
                common[_k] = mean(_vals)
    if geotr_v4e_val_diag:
        for _k, _vals in geotr_v4e_val_diag.items():
            if _vals:
                common[_k] = mean(_vals)
    if geotr_v4d_val_diag:
        # Count fields must be summed across validation batches; all other
        # diagnostics are batch means.  This lets us report an exact global
        # pixel F1 in addition to the historical mean-of-batch F1.
        for _k, _vals in geotr_v4d_val_diag.items():
            if not _vals:
                continue
            common[_k] = sum(_vals) if _k.endswith("_count") else mean(_vals)
        def _v4d_f1(tp, fp, fn):
            den = 2.0 * tp + fp + fn
            return (2.0 * tp / den) if den > 0.0 else 0.0
        _fn_f1 = _v4d_f1(
            common.get("val_geotr_v4d_fn_tp_count", 0.0),
            common.get("val_geotr_v4d_fn_fp_count", 0.0),
            common.get("val_geotr_v4d_fn_fn_count", 0.0),
        )
        _fp_f1 = _v4d_f1(
            common.get("val_geotr_v4d_fp_tp_count", 0.0),
            common.get("val_geotr_v4d_fp_fp_count", 0.0),
            common.get("val_geotr_v4d_fp_fn_count", 0.0),
        )
        common["val_geotr_v4d_global_fn_f1"] = _fn_f1
        common["val_geotr_v4d_global_fp_f1"] = _fp_f1
        common["val_geotr_v4d_global_typed_macro_f1"] = 0.5 * (_fn_f1 + _fp_f1)
    common.update(_validation_tail_metrics(base_scores, cfg, "base_dice"))
    common.update(_validation_tail_metrics(fusion_scores, cfg, "fusion_dice"))
    if native_enabled and native_base_dice:
        native_base_dice_mean = mean(native_base_dice)
        native_base_nsd_mean = mean(native_base_nsd)
        native_fusion_dice_mean = mean(native_fusion_dice)
        native_fusion_nsd_mean = mean(native_fusion_nsd)
        native_m2_dice_mean = (
            mean(native_m2_dice) if native_m2_dice else native_fusion_dice_mean
        )
        native_m2_nsd_mean = (
            mean(native_m2_nsd) if native_m2_nsd else native_fusion_nsd_mean
        )
        native_shadow_m2_dice_mean = (
            mean(native_shadow_m2_dice)
            if native_shadow_m2_dice else native_m2_dice_mean
        )
        native_shadow_m2_nsd_mean = (
            mean(native_shadow_m2_nsd)
            if native_shadow_m2_nsd else native_m2_nsd_mean
        )
        native_m1_dice_mean = (
            mean(native_m1_dice) if native_m1_dice else native_base_dice_mean
        )
        native_m1_nsd_mean = (
            mean(native_m1_nsd) if native_m1_nsd else native_base_nsd_mean
        )
        if bool(
            _cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)
        ):
            # Correct scientific meaning for the new experiments: M1 is the
            # Geometry operator in full mode and Base in residual-only A2.
            _v32_mode = str(_cfg_get(cfg.M1, "GEOTOPO_MODE", "full")).strip().lower()
            if _v32_mode == "full" and native_geotr_geometry_dice:
                native_m1_dice_mean = mean(native_geotr_geometry_dice)
                native_m1_nsd_mean = mean(native_geotr_geometry_nsd)
            elif _v32_mode == "residual":
                native_m1_dice_mean = native_base_dice_mean
                native_m1_nsd_mean = native_base_nsd_mean
        native_oracle_dice_mean = mean(native_oracle_dice)
        native_oracle_nsd_mean = mean(native_oracle_nsd)
        native_component_oracle_dice_mean = (
            mean(native_component_oracle_dice)
            if native_component_oracle_dice
            else native_oracle_dice_mean
        )
        native_component_oracle_nsd_mean = (
            mean(native_component_oracle_nsd)
            if native_component_oracle_nsd
            else native_oracle_nsd_mean
        )
        native_dsc_weight = float(_cfg_get(cfg.TRAIN, "VAL_NATIVE_DSC_WEIGHT", 0.5))
        native_nsd_weight = float(_cfg_get(cfg.TRAIN, "VAL_NATIVE_NSD_WEIGHT", 0.5))
        native_norm = max(native_dsc_weight + native_nsd_weight, 1.0e-8)
        native_dsc_weight /= native_norm
        native_nsd_weight /= native_norm
        common.update(_validation_tail_metrics(
            native_base_dice, cfg, "native_base_dice"
        ))
        native_m1_case_values = (
            native_m1_dice if native_m1_dice else native_base_dice
        )
        common.update(_validation_tail_metrics(
            native_m1_case_values, cfg, "native_m1_dice"
        ))
        common.update(_validation_tail_metrics(
            native_fusion_dice, cfg, "native_fusion_dice"
        ))
        native_m2_tail_values = native_m2_dice if native_m2_dice else native_fusion_dice
        common.update(_validation_tail_metrics(
            native_m2_tail_values, cfg, "native_m2_dice"
        ))
        if native_shadow_m2_dice:
            common.update(_validation_tail_metrics(
                native_shadow_m2_dice, cfg, "native_shadow_m2_dice"
            ))

        # R4.8 pre-declared case-level safety audit.  Mean improvement alone is
        # insufficient when rare harmful edits are about twice as large as
        # beneficial edits.  These statistics are validation-only and never
        # inspect Test during training.
        native_m2_for_safety = native_m2_dice if native_m2_dice else native_fusion_dice
        if len(native_m1_case_values) == len(native_base_dice):
            m1_gain_arr = (
                np.asarray(native_m1_case_values, dtype=np.float64)
                - np.asarray(native_base_dice, dtype=np.float64)
            )
            m1_positive = m1_gain_arr[m1_gain_arr > 0.0]
            m1_negative = m1_gain_arr[m1_gain_arr < 0.0]
            common.update({
                "native_m1_beneficial_case_rate": float((m1_gain_arr > 0.0).mean()),
                "native_m1_equal_case_rate": float((m1_gain_arr == 0.0).mean()),
                "native_m1_harmful_case_rate": float((m1_gain_arr < 0.0).mean()),
                "native_m1_harmful_case_count": int((m1_gain_arr < 0.0).sum()),
                "native_m1_mean_positive_gain": (
                    float(m1_positive.mean()) if m1_positive.size else 0.0
                ),
                "native_m1_mean_harmful_change": (
                    float(m1_negative.mean()) if m1_negative.size else 0.0
                ),
                "native_m1_worst_case_gain": float(m1_gain_arr.min()),
                "native_m1_p10_case_gain": float(np.quantile(m1_gain_arr, 0.10)),
                "native_m1_median_case_gain": float(np.median(m1_gain_arr)),
                "native_m1_best_case_gain": float(m1_gain_arr.max()),
            })
        if bool(
            _cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False)
            or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)
        ):
            _v32_mode = str(_cfg_get(cfg.M1, "GEOTOPO_MODE", "full")).strip().lower()
            if _v32_mode == "full" and len(native_geotr_recon_after_geometry_dice) == len(native_geotr_geometry_dice) and native_geotr_geometry_dice:
                gain_arr = np.asarray(native_geotr_recon_after_geometry_dice, dtype=np.float64) - np.asarray(native_geotr_geometry_dice, dtype=np.float64)
            elif _v32_mode == "residual" and len(native_geotr_recon_base_dice) == len(native_base_dice) and native_geotr_recon_base_dice:
                gain_arr = np.asarray(native_geotr_recon_base_dice, dtype=np.float64) - np.asarray(native_base_dice, dtype=np.float64)
            else:
                gain_arr = np.asarray(native_m2_for_safety, dtype=np.float64) - np.asarray(native_base_dice, dtype=np.float64)
        else:
            gain_arr = np.asarray(native_m2_for_safety, dtype=np.float64) - np.asarray(native_base_dice, dtype=np.float64)
        pos_arr = gain_arr[gain_arr > 0.0]
        neg_arr = gain_arr[gain_arr < 0.0]
        mean_positive_gain = float(pos_arr.mean()) if pos_arr.size else 0.0
        mean_harmful_change = float(neg_arr.mean()) if neg_arr.size else 0.0
        mean_harm_magnitude = abs(mean_harmful_change)
        harm_to_benefit = (
            mean_harm_magnitude / max(mean_positive_gain, 1.0e-12)
            if mean_harm_magnitude > 0.0 else 0.0
        )
        common.update({
            "native_m2_beneficial_case_rate": float((gain_arr > 0.0).mean()),
            "native_m2_equal_case_rate": float((gain_arr == 0.0).mean()),
            "native_m2_harmful_case_rate": float((gain_arr < 0.0).mean()),
            "native_m2_mean_positive_gain": mean_positive_gain,
            "native_m2_mean_harmful_change": mean_harmful_change,
            "native_m2_mean_harm_magnitude": mean_harm_magnitude,
            "native_m2_harm_to_benefit_ratio": harm_to_benefit,
        })
        if native_geotr_geometry_dice:
            _geo_d = mean(native_geotr_geometry_dice)
            _geo_n = mean(native_geotr_geometry_nsd)
            _rb_d = mean(native_geotr_recon_base_dice) if native_geotr_recon_base_dice else native_base_dice_mean
            _rb_n = mean(native_geotr_recon_base_nsd) if native_geotr_recon_base_nsd else native_base_nsd_mean
            _rg_d = mean(native_geotr_recon_after_geometry_dice) if native_geotr_recon_after_geometry_dice else _geo_d
            _rg_n = mean(native_geotr_recon_after_geometry_nsd) if native_geotr_recon_after_geometry_nsd else _geo_n
            _geotr_mode = str(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTOPO_MODE", "full")).strip().lower()
            if _geotr_mode == "residual" and native_geotr_recon_base_dice and len(native_geotr_recon_base_dice) == len(native_base_dice):
                _stage2_native = np.asarray(native_geotr_recon_base_dice, dtype=np.float64) - np.asarray(native_base_dice, dtype=np.float64)
                _stage2_label = "base"
            elif native_geotr_recon_after_geometry_dice and len(native_geotr_recon_after_geometry_dice) == len(native_geotr_geometry_dice):
                _stage2_native = np.asarray(native_geotr_recon_after_geometry_dice, dtype=np.float64) - np.asarray(native_geotr_geometry_dice, dtype=np.float64)
                _stage2_label = "geometry"
            else:
                _stage2_native = None
                _stage2_label = "unavailable"
            if _stage2_native is not None:
                _stage2_pos = _stage2_native[_stage2_native > 0.0]
                _stage2_neg = _stage2_native[_stage2_native < 0.0]
                common.update({
                    "native_geotr_stage2_gain_vs_geometry": float(_stage2_native.mean()),
                    "native_geotr_stage2_beneficial_case_rate": float((_stage2_native > 0.0).mean()),
                    "native_geotr_stage2_harmful_case_rate": float((_stage2_native < 0.0).mean()),
                    "native_geotr_stage2_mean_positive_gain": float(_stage2_pos.mean()) if _stage2_pos.size else 0.0,
                    "native_geotr_stage2_mean_harmful_change": float(_stage2_neg.mean()) if _stage2_neg.size else 0.0,
                    "native_geotr_stage2_anchor_is_base": 1.0 if _stage2_label == "base" else 0.0,
                })
            common.update({
                "native_geotr_geometry_dice": _geo_d,
                "native_geotr_geometry_nsd": _geo_n,
                "native_geotr_geometry_gain": _geo_d - native_base_dice_mean,
                "native_geotr_geometry_nsd_gain": _geo_n - native_base_nsd_mean,
                "native_geotr_recon_base_dice": _rb_d,
                "native_geotr_recon_base_nsd": _rb_n,
                "native_geotr_recon_base_gain": _rb_d - native_base_dice_mean,
                "native_geotr_recon_base_nsd_gain": _rb_n - native_base_nsd_mean,
                "native_geotr_recon_after_geometry_dice": _rg_d,
                "native_geotr_recon_after_geometry_nsd": _rg_n,
                "native_geotr_recon_after_geometry_gain_vs_geometry": _rg_d - _geo_d,
                "native_geotr_recon_after_geometry_nsd_gain_vs_geometry": _rg_n - _geo_n,
            })

        if native_c2r_context_oracle_dice:
            _ctx_d = mean(native_c2r_context_oracle_dice)
            _ctx_n = mean(native_c2r_context_oracle_nsd)
            _cand_d = mean(native_c2r_candidate_oracle_dice) if native_c2r_candidate_oracle_dice else _ctx_d
            _cand_n = mean(native_c2r_candidate_oracle_nsd) if native_c2r_candidate_oracle_nsd else _ctx_n
            _mode = str(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTOPO_MODE", "full")).strip().lower()
            _anchor_native_d = mean(native_geotr_geometry_dice) if (_mode == "full" and native_geotr_geometry_dice) else native_base_dice_mean
            _anchor_native_n = mean(native_geotr_geometry_nsd) if (_mode == "full" and native_geotr_geometry_nsd) else native_base_nsd_mean
            common.update({
                "native_c2r_context_oracle_dice": _ctx_d,
                "native_c2r_context_oracle_nsd": _ctx_n,
                "native_c2r_context_oracle_gain": _ctx_d - _anchor_native_d,
                "native_c2r_context_oracle_nsd_gain": _ctx_n - _anchor_native_n,
                "native_c2r_candidate_oracle_dice": _cand_d,
                "native_c2r_candidate_oracle_nsd": _cand_n,
                "native_c2r_candidate_oracle_gain": _cand_d - _anchor_native_d,
                "native_c2r_candidate_oracle_nsd_gain": _cand_n - _anchor_native_n,
            })

        if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)):
            _v32_mode = str(_cfg_get(cfg.M1, "GEOTOPO_MODE", "full")).strip().lower()
            _anchor_d = mean(native_geotr_geometry_dice) if (_v32_mode == "full" and native_geotr_geometry_dice) else native_base_dice_mean
            _anchor_n = mean(native_geotr_geometry_nsd) if (_v32_mode == "full" and native_geotr_geometry_nsd) else native_base_nsd_mean
            for _stage in ("canonical", "candidate", "area", "risk", "direction", "spread", "strength"):
                if native_pc2r_stage_dice[_stage]:
                    _sd = mean(native_pc2r_stage_dice[_stage])
                    _sn = mean(native_pc2r_stage_nsd[_stage])
                    common[f"native_pc2r_stage_{_stage}_dice"] = _sd
                    common[f"native_pc2r_stage_{_stage}_nsd"] = _sn
                    common[f"native_pc2r_stage_{_stage}_gain"] = _sd - _anchor_d
                    common[f"native_pc2r_stage_{_stage}_nsd_gain"] = _sn - _anchor_n

            # Exact native case-size safety audit for the true Stage-2 anchor.
            if native_case_area_ratio:
                _areas = np.asarray(native_case_area_ratio, dtype=np.float64)
                if _v32_mode == "full" and len(native_geotr_recon_after_geometry_dice) == len(native_geotr_geometry_dice) == len(_areas):
                    _sg = np.asarray(native_geotr_recon_after_geometry_dice, dtype=np.float64) - np.asarray(native_geotr_geometry_dice, dtype=np.float64)
                elif _v32_mode == "residual" and len(native_geotr_recon_base_dice) == len(native_base_dice) == len(_areas):
                    _sg = np.asarray(native_geotr_recon_base_dice, dtype=np.float64) - np.asarray(native_base_dice, dtype=np.float64)
                else:
                    _sg = None
                if _sg is not None:
                    _empty = _areas <= 0.0
                    _nonempty = ~_empty
                    common["native_pc2r_empty_case_count"] = int(_empty.sum())
                    common["native_pc2r_empty_case_mean_gain"] = float(_sg[_empty].mean()) if _empty.any() else 0.0
                    common["native_pc2r_empty_case_harm_rate"] = float((_sg[_empty] < 0.0).mean()) if _empty.any() else 0.0
                    common["native_pc2r_nonempty_case_mean_gain"] = float(_sg[_nonempty].mean()) if _nonempty.any() else 0.0
                    if _nonempty.any():
                        _q = np.quantile(_areas[_nonempty], [0.25, 0.50, 0.75])
                        _bins = [
                            ("q1_small", _nonempty & (_areas <= _q[0])),
                            ("q2", _nonempty & (_areas > _q[0]) & (_areas <= _q[1])),
                            ("q3", _nonempty & (_areas > _q[1]) & (_areas <= _q[2])),
                            ("q4_large", _nonempty & (_areas > _q[2])),
                        ]
                        for _name, _mask in _bins:
                            common[f"native_pc2r_{_name}_count"] = int(_mask.sum())
                            common[f"native_pc2r_{_name}_mean_gain"] = float(_sg[_mask].mean()) if _mask.any() else 0.0
                            common[f"native_pc2r_{_name}_harm_rate"] = float((_sg[_mask] < 0.0).mean()) if _mask.any() else 0.0

        common.update({
            "native_base_dice": native_base_dice_mean,
            "native_base_nsd": native_base_nsd_mean,
            "native_mhcs_global_selected_dice": (
                mean(native_mhcs_global_selected_dice)
                if native_mhcs_global_selected_dice else native_base_dice_mean
            ),
            "native_mhcs_global_selected_nsd": (
                mean(native_mhcs_global_selected_nsd)
                if native_mhcs_global_selected_nsd else native_base_nsd_mean
            ),
            "native_fusion_dice": native_fusion_dice_mean,
            "native_fusion_nsd": native_fusion_nsd_mean,
            "native_m1_dice": native_m1_dice_mean,
            "native_m1_nsd": native_m1_nsd_mean,
            "native_m1_gain": native_m1_dice_mean - native_base_dice_mean,
            "native_m2_dice": native_m2_dice_mean,
            "native_m2_nsd": native_m2_nsd_mean,
            "native_shadow_m2_dice": native_shadow_m2_dice_mean,
            "native_shadow_m2_nsd": native_shadow_m2_nsd_mean,
            # Historical Oracle is the global action-candidate Oracle.
            "native_oracle_dice": native_oracle_dice_mean,
            "native_oracle_nsd": native_oracle_nsd_mean,
            "native_action_oracle_dice": native_oracle_dice_mean,
            "native_action_oracle_nsd": native_oracle_nsd_mean,
            # In C2R the candidate set is simply Base/Final; expose the exact
            # meaning instead of interpreting this historical field as ROI oracle.
            "native_case_choice_oracle_dice": native_oracle_dice_mean,
            "native_case_choice_oracle_nsd": native_oracle_nsd_mean,
            # V543B Oracle is the candidate set M2 can actually select.
            "native_component_oracle_dice": (
                native_component_oracle_dice_mean
            ),
            "native_component_oracle_nsd": (
                native_component_oracle_nsd_mean
            ),
            "native_component_oracle_gain": (
                native_component_oracle_dice_mean
                - native_base_dice_mean
            ),
            "native_fusion_gain": native_fusion_dice_mean - native_base_dice_mean,
            "native_fusion_nsd_gain": native_fusion_nsd_mean - native_base_nsd_mean,
            "native_m2_gain": native_m2_dice_mean - native_base_dice_mean,
            "native_m2_nsd_gain": native_m2_nsd_mean - native_base_nsd_mean,
            "native_shadow_m2_gain": (
                native_shadow_m2_dice_mean - native_base_dice_mean
            ),
            "native_shadow_m2_nsd_gain": (
                native_shadow_m2_nsd_mean - native_base_nsd_mean
            ),
            "native_shadow_m2_score": (
                native_dsc_weight * native_shadow_m2_dice_mean
                + native_nsd_weight * native_shadow_m2_nsd_mean
            ),
            "native_m3_gain_vs_m2": native_fusion_dice_mean - native_m2_dice_mean,
            "native_m3_nsd_gain_vs_m2": native_fusion_nsd_mean - native_m2_nsd_mean,
            "native_oracle_gain": native_oracle_dice_mean - native_base_dice_mean,
            "native_fusion_score": (
                native_dsc_weight * native_fusion_dice_mean
                + native_nsd_weight * native_fusion_nsd_mean
            ),
        })
    if m1_uses_unified_action_cf(cfg):
        action_mean = torch.stack(action_scores).mean(dim=0).tolist() if action_scores else []
        selected_mean = torch.cat(selected_counts).mean().item() if selected_counts else 0.0
        common.update({
            "set_oracle_dice": mean(set_oracle_scores),
            "set_oracle_gain": mean(set_oracle_scores) - base_mean,
            "pareto_oracle_dice": mean(pareto_dice_scores) if pareto_dice_scores else base_mean,
            "pareto_oracle_nsd": mean(pareto_nsd_scores) if pareto_nsd_scores else base_nsd_mean,
            "pareto_oracle_gain": (
                mean(pareto_dice_scores) - base_mean if pareto_dice_scores else 0.0
            ),
            "deployable_pareto_oracle_dice": (
                mean(deployable_pareto_dice_scores)
                if deployable_pareto_dice_scores else base_mean
            ),
            "deployable_pareto_oracle_nsd": (
                mean(deployable_pareto_nsd_scores)
                if deployable_pareto_nsd_scores else base_nsd_mean
            ),
            "deployable_pareto_oracle_gain": (
                mean(deployable_pareto_dice_scores) - base_mean
                if deployable_pareto_dice_scores else 0.0
            ),
            "action_dice": action_mean,
            "selected_actions": selected_mean,
            "oracle_preserve": 0,
            "oracle_shrink": 0,
            "oracle_expand": 0,
        })
        return common

    common.update({
        "trim_dice": base_mean,
        "component_dice": base_mean,
        "trim_gain": 0.0,
        "component_gain": 0.0,
        "oracle_preserve": 0,
        "oracle_shrink": 0,
        "oracle_expand": 0,
    })
    return common


def evaluate_mhcs_fixed_train(model, dataloader, device, cfg):
    """Fixed-model, eval-mode Train subset audit for MHCS.

    Unlike streaming M1_DIAG, every case here is evaluated by the exact same
    end-of-epoch model in eval/no_grad mode. This makes Train-vs-Val Bank capacity
    comparisons scientifically interpretable without touching Test.
    """
    if dataloader is None or not _mhcs(cfg):
        return None
    was_training = model.training
    model.eval()
    base_all, best_all, pwo_all = [], [], []
    patch_all, hard_env_all = [], []
    global_all, local_all, final_all = [], [], []
    soft_oracle_all, gate_all, erank_all = [], [], []
    pc2r_train_canonical_gain_all, pc2r_train_final_gain_all, pc2r_train_candidate_oracle_gain_all = [], [], []
    state = _capture_rng_state()
    try:
        _seed_only(int(_cfg_get(cfg.M1, "VAL_MC_SEED", 42)))
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="MHCS FixedTrainDiag"):
                images = batch["image"].to(device)
                images_hr = batch.get("image_hr", None)
                if isinstance(images_hr, torch.Tensor):
                    images_hr = images_hr.to(device)
                masks = batch["ground_truth_mask"].to(device)
                pred = model.predict_m1_diagnostics(
                    images, batch["text_prompt"],
                    num_samples=int(_cfg_get(cfg.M1, "VAL_NUM_SAMPLES", 10)),
                    slr_hr_image=images_hr,
                )
                base = pred["base_probs"]
                cp = pred["candidate_probs"]
                final = pred["mhcs_final_probs"]
                local = pred.get("mhcs_local_probs", final)
                glob = pred.get("mhcs_global_selected_probs")
                if not isinstance(glob, torch.Tensor):
                    q = pred.get("mhcs_quality_probs")
                    idx = q.argmax(1)
                    glob = cp.gather(1, idx[:, None, None, None].expand(-1,1,cp.shape[-2],cp.shape[-1]))[:,0]
                if final.ndim == 4 and final.shape[1] == 1:
                    final = final[:, 0]
                if local.ndim == 4 and local.shape[1] == 1:
                    local = local[:, 0]
                if glob.ndim == 4 and glob.shape[1] == 1:
                    glob = glob[:, 0]
                base_d = _dice_per_case_probs(base, masks)
                slots_d = torch.stack([_dice_per_case_probs(cp[:, i], masks) for i in range(cp.shape[1])], 1)
                gt = masks[:, 0] if masks.ndim == 4 and masks.shape[1] == 1 else masks
                gt_h = gt >= 0.5
                hard = cp >= 0.5
                repairable = hard.eq(gt_h[:, None]).any(1)
                pwo_h = torch.where(repairable, gt_h, hard[:, 0]).to(base.dtype)
                pwo_d = _dice_per_case_probs(pwo_h, masks)
                p = cp.clamp(1e-4, 1.0-1e-4)
                y = gt[:, None].expand_as(p)
                risk = -(y*torch.log(p) + (1-y)*torch.log1p(-p))
                soft_oracle = risk.min(1).values.mean((-2,-1))
                patch_size = 16
                try:
                    patch_size = int(str(cfg.MODEL.BACKBONE).rsplit("/", 1)[-1])
                except Exception:
                    pass
                target3 = _target_3d(masks).to(cp)
                joint_oracle = _joint_oracle_envelope(
                    cp, target3, patch_size,
                    iterations=int(_cfg_get(cfg.M1, "MHCS_JOINT_ORACLE_ITERS", 4)),
                )
                patch_prob = joint_oracle["prob"].to(cp)
                hard_env = pred.get("mhcs_surface_hard_probs")
                if isinstance(hard_env, torch.Tensor) and hard_env.ndim == 4 and hard_env.shape[1] == 1:
                    hard_env = hard_env[:, 0]

                base_all += base_d.cpu().tolist()
                best_all += slots_d.max(1).values.cpu().tolist()
                pwo_all += pwo_d.cpu().tolist()
                patch_all += _dice_per_case_probs(patch_prob, masks).cpu().tolist()
                if isinstance(hard_env, torch.Tensor):
                    hard_env_all += _dice_per_case_probs(hard_env, masks).cpu().tolist()
                global_all += _dice_per_case_probs(glob, masks).cpu().tolist()
                local_all += _dice_per_case_probs(local, masks).cpu().tolist()
                final_all += _dice_per_case_probs(final, masks).cpu().tolist()
                soft_oracle_all += soft_oracle.cpu().tolist()
                gate = pred.get("mhcs_gate_alpha")
                if isinstance(gate, torch.Tensor): gate_all += gate.reshape(-1).cpu().tolist()
                erank = pred.get("mhcs_effective_rank")
                if isinstance(erank, torch.Tensor): erank_all += erank.reshape(-1).cpu().tolist()

                if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False)):
                    mode = str(_cfg_get(cfg.M1, "GEOTOPO_MODE", "full")).strip().lower()
                    if mode in {"residual", "full"}:
                        prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
                        anchor_pc = pred.get("geotopo_geometry_probs") if mode == "full" else pred.get("geotopo_base_probs", base)
                        canonical_pc = pred.get(prefix + "_c2r_mean_prob")
                        candidate_pc = pred.get(prefix + "_c2r_candidate_mask")
                        final_pc = pred.get("geotopo_final_probs", final)
                        if all(isinstance(x, torch.Tensor) for x in (anchor_pc, canonical_pc, candidate_pc, final_pc)):
                            if anchor_pc.ndim == 4 and anchor_pc.shape[1] == 1: anchor_pc = anchor_pc[:,0]
                            if canonical_pc.ndim == 4 and canonical_pc.shape[1] == 1: canonical_pc = canonical_pc[:,0]
                            if candidate_pc.ndim == 4 and candidate_pc.shape[1] == 1: candidate_pc = candidate_pc[:,0]
                            if final_pc.ndim == 4 and final_pc.shape[1] == 1: final_pc = final_pc[:,0]
                            gt_pc = _target_3d(masks).to(anchor_pc)
                            if gt_pc.shape[-2:] != anchor_pc.shape[-2:]:
                                gt_pc = F.interpolate(gt_pc[:,None].float(), size=anchor_pc.shape[-2:], mode="nearest")[:,0].to(anchor_pc)
                            ad = _dice_per_case_probs(anchor_pc, gt_pc)
                            cd = _dice_per_case_probs(canonical_pc, gt_pc)
                            fd = _dice_per_case_probs(final_pc, gt_pc)
                            oracle_pc = torch.where(candidate_pc > 0.5, gt_pc, anchor_pc)
                            od = _dice_per_case_probs(oracle_pc, gt_pc)
                            pc2r_train_canonical_gain_all += (cd-ad).cpu().tolist()
                            pc2r_train_final_gain_all += (fd-ad).cpu().tolist()
                            pc2r_train_candidate_oracle_gain_all += (od-ad).cpu().tolist()
    finally:
        _restore_rng_state(state)
        model.train(was_training)
    return {
        "base": mean(base_all), "best_single": mean(best_all), "hard_pwo": mean(pwo_all),
        "patch_oracle": mean(patch_all) if patch_all else mean(best_all),
        "hard_envelope": mean(hard_env_all) if hard_env_all else mean(final_all),
        "global": mean(global_all), "local": mean(local_all), "final": mean(final_all),
        "soft_oracle_risk": mean(soft_oracle_all),
        "gate_alpha": mean(gate_all) if gate_all else 0.0,
        "effective_rank": mean(erank_all) if erank_all else 0.0,
        "pc2r_canonical_gain": mean(pc2r_train_canonical_gain_all) if pc2r_train_canonical_gain_all else 0.0,
        "pc2r_final_gain": mean(pc2r_train_final_gain_all) if pc2r_train_final_gain_all else 0.0,
        "pc2r_candidate_oracle_gain": mean(pc2r_train_candidate_oracle_gain_all) if pc2r_train_candidate_oracle_gain_all else 0.0,
    }


def build_model(cfg):
    clip_model = str(cfg.MODEL.CLIP_MODEL).lower()
    if clip_model != "unimedclip":
        raise ValueError(
            "This V410-only project supports MODEL.CLIP_MODEL=unimedclip only; "
            f"got {cfg.MODEL.CLIP_MODEL!r}."
        )
    return build_medclipseg_unimedclip(cfg)

def worker_init_fn_factory(seed):
    """Seed Python/NumPy from PyTorch's epoch-specific worker seed.

    DataLoader creates a new ``base_seed`` whenever a new iterator/epoch is
    created. The previous ``seed + worker_id`` implementation reset every
    re-created worker to the same augmentation stream each epoch. Reading the
    seed assigned by DataLoader keeps runs reproducible while allowing fresh,
    deterministic augmentations across epochs.
    """
    del seed  # retained in the signature for backward compatibility

    def worker_init_fn(worker_id):
        del worker_id
        worker_seed = int(torch.initial_seed() % (2 ** 32))
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return worker_init_fn



def official_medclipseg_worker_init_fn_factory(seed):
    """Exact worker seeding used by the public CVPR-2026 MedCLIPSeg train.py."""
    def worker_init_fn(worker_id):
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return worker_init_fn

def _candidate_ratio(cfg, epoch):
    """Nominal auxiliary-loss schedule for stable joint end-to-end training.

    V470 started at ``maximum / ramp`` and increased aggressively.  With a
    proposal loss around 4--5 and a Base loss around 0.5, that made the
    auxiliary objective dominate from the first few epochs.  V471 supports an
    explicit non-zero start value and linearly reaches the configured maximum.
    All task modules remain trainable from epoch one.
    """
    if not m1_enabled(cfg):
        return 0.0
    if _semlt(cfg):
        # M1 observes detached factual Base logits; its single deployed
        # Geometry objective is live at full weight exactly as in legacy MHCS.
        return 1.0
    if _mhcs(cfg):
        # MHCS uses its own learned uncertainty balance between bank coverage
        # and final composition.  There is no additional hand-tuned M1 warmup.
        return 1.0
    if _clean_dynamic_component_set(cfg):
        # CLEAN has no hand-tuned auxiliary warm-up schedule. All downstream
        # modules are live from epoch one; the single Base-relative safety
        # budget is applied later by _v490_split_objective.
        return 1.0

    # V530/V529/V528 M2-only stages freeze Base/PVL/M1/M3.  Their only live
    # objective is the pixel-composer auxiliary loss, so an M1-side warmup must
    # never suppress the auxiliary graph on epoch 0.  Scheduler warmup remains
    # controlled independently by TRAIN.WARMUP_EPOCHS.
    if _v519_is_m2_only(cfg):
        return max(
            0.0,
            float(_cfg_get(cfg.M1, "CANDIDATE_LOSS_WEIGHT", 1.0)),
        )

    maximum = float(_cfg_get(cfg.M1, "CANDIDATE_LOSS_WEIGHT", 0.10))
    start = float(_cfg_get(cfg.M1, "CANDIDATE_LOSS_START_WEIGHT", 0.005))
    warmup = int(_cfg_get(cfg.M1, "WARMUP_EPOCHS", 0))
    v507_aux_start = int(_cfg_get(cfg.M1, "V507_AUX_START_EPOCH", warmup))
    ramp = int(_cfg_get(cfg.M1, "RAMP_EPOCHS", 20))

    if epoch < max(warmup, v507_aux_start):
        return 0.0

    maximum = max(0.0, maximum)
    start = min(max(0.0, start), maximum)
    effective_start = max(warmup, v507_aux_start)
    if ramp <= 1:
        return maximum

    step = max(0, epoch - effective_start)
    progress = min(1.0, float(step) / float(ramp - 1))
    return start + (maximum - start) * progress


def _effective_candidate_ratio(cfg, nominal_ratio, base_loss, proposal_loss):
    """Cap the detached auxiliary contribution relative to the Base loss.

    This is loss-scale balancing, not gradient detachment: gradients still
    propagate through ``proposal_loss`` to Base, M1 and M2 according to the
    configured non-zero cross-module gradient scales.  Only the scalar weight
    is computed from detached loss magnitudes, preventing a large auxiliary
    loss from overwhelming the segmentation objective.
    """
    nominal = float(max(0.0, nominal_ratio))
    if nominal == 0.0:
        return 0.0
    if str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "semlt_autozero":
        # AutoZero has no auxiliary-to-Base balancing coefficient. The Base
        # and M1 objectives own disjoint parameter sets because Base/semantic
        # evidence is detached inside M1, so a scalar trade-off is unnecessary.
        return nominal
    # JBT-v3 separates two fundamentally different controls:
    #   (1) full-strength residual-branch learning for JBT parameters;
    #   (2) a small cross-gradient scale on Base/PVL inputs.
    # v2 used V471 to scale the *whole* proposal loss to <=30% of Base, which
    # unintentionally weakened M1 itself even though Base gradients were already
    # controlled by _scale_gradient.  In v3 the branch keeps its declared loss
    # weight while Base safety remains exclusively a gradient-routing contract.
    if bool(_cfg_get(cfg.M1, "JBT_DECOUPLE_BRANCH_LOSS_FROM_BASE_CAP", False)):
        return nominal * max(0.0, float(
            _cfg_get(cfg.M1, "JBT_BRANCH_LOSS_WEIGHT", 1.0)
        ))

    if _mhcs(cfg):
        if not bool(_cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)):
            # Legacy MHCS uses learned homoscedastic balancing.
            return nominal
        # SPARC-HR2 is still trained end-to-end from scratch in one run, but its
        # exact HR counterfactual objective contains many more terms than Base.
        # A detached magnitude cap preserves gradient flow while preventing an
        # early, numerically large critic/selector loss from rotating the shared
        # Base trajectory.  This is scale control, not staged freezing.
        cap = float(
            _cfg_get(cfg.M1, "GEOTR_SPARC_AUX_TO_BASE_LOSS_RATIO_CAP", 0.75)
        )
        if cap <= 0.0:
            return nominal
        base_value = float(base_loss.detach().abs().clamp_min(1.0e-8).cpu())
        proposal_value = float(
            proposal_loss.detach().abs().clamp_min(1.0e-8).cpu()
        )
        return min(nominal, cap * base_value / proposal_value)

    cap = float(_cfg_get(cfg.M1, "V471_AUX_TO_BASE_LOSS_RATIO_CAP", 0.35))
    if cap <= 0.0:
        return nominal

    base_value = float(base_loss.detach().abs().clamp_min(1.0e-8).cpu())
    proposal_value = float(proposal_loss.detach().abs().clamp_min(1.0e-8).cpu())
    cap_weight = cap * base_value / proposal_value
    minimum = float(_cfg_get(cfg.M1, "V471_MIN_EFFECTIVE_CANDIDATE_WEIGHT", 0.0))
    return max(minimum, min(nominal, cap_weight))



def _v490_split_objective(
    cfg,
    diagnostics,
    candidate_ratio_nominal,
    base_loss,
):
    """Route live M1/M2/M3 objectives with an explicit Base-relative contract.

    V502/V503 ratios are magnitude contracts, not raw multipliers. Each module
    objective is normalized by its detached current magnitude and then scaled to
    the requested fraction of the detached Base loss. This preserves gradients
    while preventing numerically large auxiliary losses from changing Base
    optimization geometry.
    """
    if not _v490_is_end_to_end(cfg):
        return None
    m1_objective = diagnostics.pop("_v490_m1_objective", None)
    if m1_objective is None:
        return None
    m2_objective = diagnostics.pop("_v490_m2_objective", None)
    m3_objective = diagnostics.pop("_v490_m3_objective", None)
    if not all(isinstance(value, torch.Tensor) for value in (
        m1_objective, m2_objective, m3_objective
    )):
        raise RuntimeError(
            "V490 loss routing requires live M1/M2/M3 objectives from v484_loss.py"
        )

    clean_dynamic_component_set = _clean_dynamic_component_set(cfg)
    v538_enabled = clean_dynamic_component_set or bool(
        _cfg_get(cfg.M1, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False)
    )
    v538_independent = clean_dynamic_component_set or bool(
        _cfg_get(cfg.M1, "V538_INDEPENDENT_OBJECTIVE_ROUTING", False)
    )
    v538_m1_objective = diagnostics.pop("_v538_m1_objective", None)
    v538_m2_objective = diagnostics.pop("_v538_m2_objective", None)

    # V552-R3 exposes Editor, Outcome and Composer as three live objectives.
    # Pop them even for legacy R2 configs so graph-carrying tensors never leak
    # into the generic diagnostic logger.  R3 routes each objective with its
    # own Base-relative budget; a transient Composer spike can therefore no
    # longer collapse Editor/Outcome gradients.
    r3_editor_objective = diagnostics.pop("_v552r3_editor_objective", None)
    r3_outcome_objective = diagnostics.pop("_v552r3_outcome_objective", None)
    r3_composer_objective = diagnostics.pop("_v552r3_composer_objective", None)
    r3_enabled = bool(
        _cfg_get(cfg.M1, "V552R3_ROOT_CALIBRATED_ROUTING_ENABLED", False)
    )

    if clean_dynamic_component_set:
        if not all(isinstance(value, torch.Tensor) for value in (
            v538_m1_objective, v538_m2_objective
        )):
            raise RuntimeError(
                "CLEAN routing requires live component M1 and M2 objectives"
            )
        # One global safety budget is the only outer loss-scale control.
        # Within that budget, M1/M2 are magnitude-normalized and M2 receives a
        # data-adaptive share equal to current factual SetOracle realization.
        # No per-loss hand multipliers are used here.
        base_magnitude = base_loss.detach().abs().clamp_min(1.0e-8)
        budget_ratio = max(0.0, float(_cfg_get(
            cfg.M1, "AUX_TO_BASE_MAX_RATIO", 0.25
        )))
        realization = diagnostics.get(
            "v561_set_oracle_realization_ratio", base_loss.new_zeros(())
        )
        if not isinstance(realization, torch.Tensor):
            realization = base_loss.new_tensor(float(realization))
        realization = realization.detach().clamp(0.0, 1.0)
        m1_share = base_loss.new_ones(())
        m2_share = realization
        share_sum = (m1_share + m2_share).clamp_min(1.0e-8)
        m1_ratio = budget_ratio * m1_share / share_sum
        m2_ratio = budget_ratio * m2_share / share_sum
        m1_magnitude = v538_m1_objective.detach().abs().clamp_min(1.0e-8)
        m2_magnitude = v538_m2_objective.detach().abs().clamp_min(1.0e-8)
        m1_weight = m1_ratio * base_magnitude / m1_magnitude
        m2_weight = m2_ratio * base_magnitude / m2_magnitude
        proposal_objective = m1_weight * v538_m1_objective + m2_weight * v538_m2_objective
        actual_m1_ratio = (m1_weight * v538_m1_objective).detach().abs() / base_magnitude
        actual_m2_ratio = (m2_weight * v538_m2_objective).detach().abs() / base_magnitude
        total_ratio_tensor = actual_m1_ratio + actual_m2_ratio
        diagnostics.update({
            "clean_outer_budget_ratio": base_loss.new_tensor(budget_ratio),
            "clean_m2_realization_share": realization,
            "clean_m1_effective_weight": m1_weight.detach(),
            "clean_m2_effective_weight": m2_weight.detach(),
            "clean_m1_routed_ratio": actual_m1_ratio.detach(),
            "clean_m2_routed_ratio": actual_m2_ratio.detach(),
            "clean_total_routed_ratio": total_ratio_tensor.detach(),
            # Compatibility diagnostics consumed by existing gradient logging.
            "v538_independent_route_enabled": base_loss.new_ones(()),
            "v538_m1_effective_weight": m1_weight.detach(),
            "v538_m2_effective_weight": m2_weight.detach(),
            "v538_m1_uncapped_effective_weight": m1_weight.detach(),
            "v538_m2_uncapped_effective_weight": m2_weight.detach(),
            "v538_m1_weight_capped": base_loss.new_zeros(()),
            "v538_m2_weight_capped": base_loss.new_zeros(()),
            "v538_m1_weighted_objective": (m1_weight * v538_m1_objective).detach(),
            "v538_m2_weighted_objective": (m2_weight * v538_m2_objective).detach(),
            "v538_m1_nominal_routed_ratio": m1_ratio.detach(),
            "v538_m2_nominal_routed_ratio": m2_ratio.detach(),
            "v538_m1_routed_ratio": actual_m1_ratio.detach(),
            "v538_m2_routed_ratio": actual_m2_ratio.detach(),
            "v538_total_routed_ratio": total_ratio_tensor.detach(),
            "v506_aux_budget_ratio": total_ratio_tensor.detach(),
            "v506_m1_routed_ratio": actual_m1_ratio.detach(),
            "v506_m2_routed_ratio": actual_m2_ratio.detach(),
            "v506_m3_routed_ratio": base_loss.new_zeros(()),
        })
        return proposal_objective, float(total_ratio_tensor.detach().cpu())

    if v538_enabled and v538_independent and r3_enabled:
        live = (
            v538_m1_objective,
            r3_editor_objective,
            r3_outcome_objective,
            r3_composer_objective,
        )
        if not all(isinstance(value, torch.Tensor) for value in live):
            raise RuntimeError(
                "V552-R3 independent routing requires live M1, Editor, "
                "Outcome and Composer objective tensors"
            )

        def _r3_scale_value(key, default=1.0):
            value = diagnostics.get(key, base_loss.new_tensor(default))
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        ratios = (
            max(0.0, float(_cfg_get(
                cfg.M1, "V538_M1_TO_BASE_OBJECTIVE_RATIO", 0.08
            ))) * max(0.0, _r3_scale_value("v538_m1_train_scale", 1.0)),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_EDITOR_TO_BASE_OBJECTIVE_RATIO", 0.015
            ))) * max(0.0, _r3_scale_value("v551_editor_train_scale", 1.0)),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_OUTCOME_TO_BASE_OBJECTIVE_RATIO", 0.015
            ))) * max(0.0, _r3_scale_value("v552_outcome_train_scale", 1.0)),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_COMPOSER_TO_BASE_OBJECTIVE_RATIO", 0.010
            ))) * max(0.0, _r3_scale_value("v552_composer_train_scale", 1.0)),
        )
        maximum_weights = (
            max(0.0, float(_cfg_get(
                cfg.M1, "V538_M1_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_EDITOR_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_OUTCOME_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
            max(0.0, float(_cfg_get(
                cfg.M1, "V552R3_COMPOSER_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
        )

        base_magnitude = base_loss.detach().abs().clamp_min(1.0e-8)
        aux_reference_floor = max(
            0.0,
            float(_cfg_get(cfg.M1, "V543_AUX_BASE_REFERENCE_FLOOR", 0.0)),
        )
        if aux_reference_floor > 0.0:
            base_magnitude = torch.maximum(
                base_magnitude,
                base_magnitude.new_tensor(aux_reference_floor),
            )

        weighted = []
        effective_weights = []
        uncapped_weights = []
        actual_ratios = []
        for ratio, objective, maximum_weight in zip(
            ratios, live, maximum_weights
        ):
            if ratio <= 0.0:
                uncapped_weight = base_magnitude.new_zeros(())
                effective_weight = base_magnitude.new_zeros(())
                weighted_objective = objective * 0.0
            else:
                magnitude = objective.detach().abs().clamp_min(1.0e-8)
                uncapped_weight = ratio * base_magnitude / magnitude
                effective_weight = (
                    uncapped_weight.clamp(max=maximum_weight)
                    if maximum_weight > 0.0 else uncapped_weight
                )
                weighted_objective = effective_weight * objective
            actual_ratio = (
                weighted_objective.detach().abs() / base_magnitude
            ).clamp_min(0.0)
            weighted.append(weighted_objective)
            effective_weights.append(effective_weight.detach())
            uncapped_weights.append(uncapped_weight.detach())
            actual_ratios.append(actual_ratio.detach())

        proposal_objective = sum(weighted)
        m2_weighted = weighted[1] + weighted[2] + weighted[3]
        m2_actual_ratio = actual_ratios[1] + actual_ratios[2] + actual_ratios[3]
        total_ratio_tensor = actual_ratios[0] + m2_actual_ratio
        total_ratio = float(total_ratio_tensor.detach().cpu())
        zero = base_loss.new_zeros(())
        diagnostics.update({
            "v552r3_independent_routing_enabled": base_loss.new_ones(()),
            "v543_aux_reference_magnitude": base_magnitude.detach(),
            "v552r3_m1_effective_weight": effective_weights[0],
            "v552r3_editor_effective_weight": effective_weights[1],
            "v552r3_outcome_effective_weight": effective_weights[2],
            "v552r3_composer_effective_weight": effective_weights[3],
            "v552r3_m1_uncapped_effective_weight": uncapped_weights[0],
            "v552r3_editor_uncapped_effective_weight": uncapped_weights[1],
            "v552r3_outcome_uncapped_effective_weight": uncapped_weights[2],
            "v552r3_composer_uncapped_effective_weight": uncapped_weights[3],
            "v552r3_m1_weighted_objective": weighted[0].detach(),
            "v552r3_editor_weighted_objective": weighted[1].detach(),
            "v552r3_outcome_weighted_objective": weighted[2].detach(),
            "v552r3_composer_weighted_objective": weighted[3].detach(),
            "v552r3_m1_nominal_routed_ratio": base_loss.new_tensor(ratios[0]),
            "v552r3_editor_nominal_routed_ratio": base_loss.new_tensor(ratios[1]),
            "v552r3_outcome_nominal_routed_ratio": base_loss.new_tensor(ratios[2]),
            "v552r3_composer_nominal_routed_ratio": base_loss.new_tensor(ratios[3]),
            "v552r3_m1_routed_ratio": actual_ratios[0],
            "v552r3_editor_routed_ratio": actual_ratios[1],
            "v552r3_outcome_routed_ratio": actual_ratios[2],
            "v552r3_composer_routed_ratio": actual_ratios[3],
            "v552r3_total_routed_ratio": total_ratio_tensor,
            # Existing log/gradient code expects the V538 compatibility keys.
            "v538_independent_route_enabled": base_loss.new_ones(()),
            "v538_m1_uncapped_effective_weight": uncapped_weights[0],
            "v538_m2_uncapped_effective_weight": (
                uncapped_weights[1] + uncapped_weights[2] + uncapped_weights[3]
            ),
            "v548_stable_m2_routing_enabled": zero,
            "v548_m2_raw_uncapped_effective_weight": zero,
            "v548_m2_ema_objective_magnitude": zero,
            "v538_m1_effective_weight": effective_weights[0],
            "v538_m2_effective_weight": (
                effective_weights[1] + effective_weights[2] + effective_weights[3]
            ),
            "v538_m1_weight_capped": (
                uncapped_weights[0] > effective_weights[0] + 1.0e-12
            ).to(base_loss.dtype),
            "v538_m2_weight_capped": zero,
            "v538_m1_weighted_objective": weighted[0].detach(),
            "v538_m2_weighted_objective": m2_weighted.detach(),
            "v538_m1_nominal_routed_ratio": base_loss.new_tensor(ratios[0]),
            "v538_m2_nominal_routed_ratio": base_loss.new_tensor(
                ratios[1] + ratios[2] + ratios[3]
            ),
            "v538_m1_routed_ratio": actual_ratios[0],
            "v538_m2_routed_ratio": m2_actual_ratio,
            "v538_total_routed_ratio": total_ratio_tensor,
            "v490_m1_effective_weight": effective_weights[0],
            "v490_m2_direct_weight": (
                effective_weights[1] + effective_weights[2] + effective_weights[3]
            ),
            "v490_m3_direct_weight": zero,
            "v490_m1_weighted_objective": weighted[0].detach(),
            "v490_m2_weighted_objective": m2_weighted.detach(),
            "v490_m3_weighted_objective": zero,
            "v491_scaled_aux_to_base_ratio": total_ratio_tensor,
            "v506_aux_budget_ratio": total_ratio_tensor,
            "v506_m1_routed_ratio": actual_ratios[0],
            "v506_m2_routed_ratio": m2_actual_ratio,
            "v506_m3_routed_ratio": zero,
        })
        return proposal_objective, total_ratio

    if v538_enabled and v538_independent:
        if not all(isinstance(value, torch.Tensor) for value in (
            v538_m1_objective, v538_m2_objective
        )):
            raise RuntimeError(
                "V538 independent routing requires live _v538_m1_objective and "
                "_v538_m2_objective tensors from utils/v484_loss.py"
            )

        def _scale_value(key, default=1.0):
            value = diagnostics.get(key, base_loss.new_tensor(default))
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        m1_ratio = max(0.0, float(_cfg_get(
            cfg.M1, "V538_M1_TO_BASE_OBJECTIVE_RATIO", 0.10
        ))) * max(0.0, _scale_value("v538_m1_train_scale", 1.0))
        m2_ratio = max(0.0, float(_cfg_get(
            cfg.M1, "V538_M2_TO_BASE_OBJECTIVE_RATIO", 0.03
        ))) * max(0.0, _scale_value("v538_m2_train_scale", 1.0))
        ratios = (m1_ratio, m2_ratio)
        objectives = (v538_m1_objective, v538_m2_objective)
        maximum_weights = (
            max(0.0, float(_cfg_get(
                cfg.M1, "V538_M1_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
            max(0.0, float(_cfg_get(
                cfg.M1, "V538_M2_MAX_EFFECTIVE_WEIGHT", 5.0
            ))),
        )
        base_magnitude = base_loss.detach().abs().clamp_min(1.0e-8)
        # V543B: M1/M2 should not lose nearly all absolute gradient merely
        # because the Base converges earlier.  The floor is set from the Base
        # loss observed when M1 becomes fully active, preserving the intended
        # auxiliary scale instead of tying it to a vanishing current loss.
        aux_reference_floor = max(
            0.0,
            float(_cfg_get(cfg.M1, "V543_AUX_BASE_REFERENCE_FLOOR", 0.0)),
        )
        if aux_reference_floor > 0.0:
            base_magnitude = torch.maximum(
                base_magnitude,
                base_magnitude.new_tensor(aux_reference_floor),
            )
        weighted = []
        effective_weights = []
        uncapped_weights = []
        actual_ratios = []
        v548_stable_m2_routing = bool(
            _cfg_get(cfg.M1, "V548_STABLE_M2_ROUTING_ENABLED", False)
        )
        v548_routing_decay = float(
            _cfg_get(cfg.M1, "V548_ROUTING_EMA_DECAY", 0.98)
        )
        v548_raw_uncapped_m2 = base_magnitude.new_zeros(())
        v548_m2_ema_magnitude = base_magnitude.new_zeros(())
        for objective_index, (ratio, objective, maximum_weight) in enumerate(zip(
            ratios, objectives, maximum_weights
        )):
            if ratio <= 0.0:
                uncapped_weight = base_magnitude.new_zeros(())
                weight = base_magnitude.new_zeros(())
                weighted_objective = objective * 0.0
            else:
                objective_magnitude = objective.detach().abs().clamp_min(1.0e-8)
                if objective_index == 1 and v548_stable_m2_routing:
                    routing_state = getattr(
                        cfg, "_v548_m2_routing_state", None
                    )
                    if not isinstance(routing_state, dict):
                        routing_state = {}
                        cfg._v548_m2_routing_state = routing_state
                    (
                        weight,
                        raw_uncapped_weight,
                        uncapped_weight,
                        ema_magnitude,
                    ) = stable_m2_effective_weight(
                        ratio=ratio,
                        base_magnitude=base_magnitude,
                        objective_magnitude=objective_magnitude,
                        maximum_weight=maximum_weight,
                        state=routing_state,
                        decay=v548_routing_decay,
                    )
                    v548_raw_uncapped_m2 = raw_uncapped_weight
                    v548_m2_ema_magnitude = ema_magnitude
                else:
                    uncapped_weight = ratio * base_magnitude / objective_magnitude
                    if maximum_weight > 0.0:
                        weight = uncapped_weight.clamp(max=maximum_weight)
                    else:
                        weight = uncapped_weight
                weighted_objective = weight * objective
            actual_ratio = (
                weighted_objective.detach().abs() / base_magnitude
            ).clamp_min(0.0)
            weighted.append(weighted_objective)
            effective_weights.append(weight.detach())
            uncapped_weights.append(uncapped_weight.detach())
            actual_ratios.append(actual_ratio.detach())

        proposal_objective = weighted[0] + weighted[1]
        total_ratio_tensor = actual_ratios[0] + actual_ratios[1]
        total_ratio = float(total_ratio_tensor.detach().cpu())
        zero = base_loss.new_zeros(())
        diagnostics.update({
            "v538_independent_route_enabled": base_loss.new_ones(()),
            "v543_aux_reference_magnitude": base_magnitude.detach(),
            "v538_m1_uncapped_effective_weight": uncapped_weights[0],
            "v538_m2_uncapped_effective_weight": uncapped_weights[1],
            "v548_stable_m2_routing_enabled": base_loss.new_tensor(
                1.0 if v548_stable_m2_routing else 0.0
            ),
            "v548_m2_raw_uncapped_effective_weight": v548_raw_uncapped_m2,
            "v548_m2_ema_objective_magnitude": v548_m2_ema_magnitude,
            "v538_m1_effective_weight": effective_weights[0],
            "v538_m2_effective_weight": effective_weights[1],
            "v538_m1_weight_capped": (
                uncapped_weights[0] > effective_weights[0] + 1.0e-12
            ).to(base_loss.dtype),
            "v538_m2_weight_capped": (
                uncapped_weights[1] > effective_weights[1] + 1.0e-12
            ).to(base_loss.dtype),
            "v538_m1_weighted_objective": weighted[0].detach(),
            "v538_m2_weighted_objective": weighted[1].detach(),
            "v538_m1_nominal_routed_ratio": base_loss.new_tensor(m1_ratio),
            "v538_m2_nominal_routed_ratio": base_loss.new_tensor(m2_ratio),
            "v538_m1_routed_ratio": actual_ratios[0],
            "v538_m2_routed_ratio": actual_ratios[1],
            "v538_total_routed_ratio": total_ratio_tensor,
            # Compatibility diagnostics used by the existing log parser.
            "v490_m1_effective_weight": effective_weights[0],
            "v490_m2_direct_weight": effective_weights[1],
            "v490_m3_direct_weight": zero,
            "v490_m1_weighted_objective": weighted[0].detach(),
            "v490_m2_weighted_objective": weighted[1].detach(),
            "v490_m3_weighted_objective": zero,
            "v491_scaled_aux_to_base_ratio": total_ratio_tensor,
            "v506_aux_budget_ratio": total_ratio_tensor,
            "v506_m1_routed_ratio": actual_ratios[0],
            "v506_m2_routed_ratio": actual_ratios[1],
            "v506_m3_routed_ratio": zero,
        })
        return proposal_objective, total_ratio

    # Legacy frozen-M1 V519 compatibility mode.  The corrected joint V519 does
    # not enter this branch: it uses the bounded Base-relative M1/M2 budget so
    # both objectives remain active while M2 inputs stay gradient-isolated.
    if _v519_is_m2_only(cfg):
        weight = float(_cfg_get(cfg.M1, "V519_M2_OBJECTIVE_WEIGHT", 1.0))
        proposal_objective = weight * m2_objective
        zero = base_loss.new_zeros(())
        diagnostics.update({
            "v490_m1_effective_weight": zero,
            "v490_m2_direct_weight": base_loss.new_tensor(weight),
            "v490_m3_direct_weight": zero,
            "v490_m1_weighted_objective": zero,
            "v490_m2_weighted_objective": proposal_objective.detach(),
            "v490_m3_weighted_objective": zero,
            "v491_scaled_aux_to_base_ratio": zero,
            "v506_aux_budget_ratio": zero,
            "v506_m1_routed_ratio": zero,
            "v506_m2_routed_ratio": base_loss.new_tensor(weight),
            "v506_m3_routed_ratio": zero,
        })
        return proposal_objective, weight

    use_base_relative = bool(
        _cfg_get(cfg.M1, "V502_HIERARCHICAL_UTILITY_SOFT_ROUTER_ENABLED", False)
        or _cfg_get(cfg.M1, "V503_FACTUAL_ATOMIC_CAUSAL_ENABLED", False)
    )
    if use_base_relative:
        v531_enabled = bool(
            _cfg_get(cfg.M1, "V531_TYPED_SPARSE_REFINER_ENABLED", False)
        )
        v505_enabled = bool(
            _cfg_get(cfg.M1, "V505_INTERACTIVE_REGION_CAUSAL_ENABLED", False)
        )
        v532_enabled = bool(
            _cfg_get(cfg.M1, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False)
        )
        if v532_enabled:
            m1_scale_value = diagnostics.get(
                "v532_m1_scale", base_loss.new_ones(())
            )
            refiner_scale_value = diagnostics.get(
                "v532_refiner_global_scale", base_loss.new_ones(())
            )
            if isinstance(m1_scale_value, torch.Tensor):
                m1_scale_value = float(m1_scale_value.detach().cpu())
            else:
                m1_scale_value = float(m1_scale_value)
            if isinstance(refiner_scale_value, torch.Tensor):
                refiner_scale_value = float(refiner_scale_value.detach().cpu())
            else:
                refiner_scale_value = float(refiner_scale_value)
            ratios = (
                max(0.0, float(_cfg_get(
                    cfg.M1, "V532_M1_TO_BASE_OBJECTIVE_RATIO", 0.15
                ))) * max(0.0, m1_scale_value),
                max(0.0, float(_cfg_get(
                    cfg.M1, "V532_REFINER_TO_BASE_OBJECTIVE_RATIO", 0.25
                ))) * max(0.0, refiner_scale_value),
                0.0,
            )
        elif v531_enabled:
            # V531 uses one from-scratch run with loss curricula.  Preserve the
            # configured ramp after magnitude normalization by applying the
            # current curriculum scale to the Base-relative objective budget.
            m1_scale_value = diagnostics.get(
                "v531_m1_scale", base_loss.new_ones(())
            )
            m2_scale_value = diagnostics.get(
                "v531_m2_global_scale", base_loss.new_ones(())
            )
            if isinstance(m1_scale_value, torch.Tensor):
                m1_scale_value = float(m1_scale_value.detach().cpu())
            else:
                m1_scale_value = float(m1_scale_value)
            if isinstance(m2_scale_value, torch.Tensor):
                m2_scale_value = float(m2_scale_value.detach().cpu())
            else:
                m2_scale_value = float(m2_scale_value)
            ratios = (
                max(0.0, float(_cfg_get(
                    cfg.M1, "V531_M1_TO_BASE_OBJECTIVE_RATIO", 0.25
                ))) * max(0.0, m1_scale_value),
                max(0.0, float(_cfg_get(
                    cfg.M1, "V531_M2_TO_BASE_OBJECTIVE_RATIO", 0.50
                ))) * max(0.0, m2_scale_value),
                0.0,
            )
        elif v505_enabled:
            # V505 previously normalised each already-ramped objective back to
            # a fixed 0.35/0.50/0.25 Base ratio.  That exactly cancelled the
            # M1/M2/M3 ramp and produced an abrupt total auxiliary/Base ratio
            # of 1.10.  V506 assigns one bounded total budget and preserves each
            # module's ramp fraction after magnitude normalisation.
            total_budget = max(
                0.0,
                float(_cfg_get(cfg.M1, "V506_TOTAL_AUX_TO_BASE_RATIO", 0.35)),
            )
            raw_shares = [
                max(0.0, float(_cfg_get(cfg.M1, "V506_M1_AUX_SHARE", 0.50))),
                max(0.0, float(_cfg_get(cfg.M1, "V506_M2_AUX_SHARE", 0.30))),
                max(0.0, float(_cfg_get(cfg.M1, "V506_M3_AUX_SHARE", 0.20))),
            ]
            share_sum = max(sum(raw_shares), 1.0e-8)
            shares = [value / share_sum for value in raw_shares]
            scale_keys = ("v505_m1_scale", "v505_m2_scale", "v505_m3_scale")
            final_keys = (
                "V505_M1_FINAL_WEIGHT",
                "V505_M2_FINAL_WEIGHT",
                "V505_M3_FINAL_WEIGHT",
            )
            ramp_fractions = []
            for scale_key, final_key in zip(scale_keys, final_keys):
                scale_value = diagnostics.get(scale_key, base_loss.new_zeros(()))
                if isinstance(scale_value, torch.Tensor):
                    scale_float = float(scale_value.detach().cpu())
                else:
                    scale_float = float(scale_value)
                final_value = max(
                    float(_cfg_get(cfg.M1, final_key, 1.0)),
                    1.0e-8,
                )
                ramp_fractions.append(
                    min(1.0, max(0.0, scale_float / final_value))
                )
            ratios = tuple(
                total_budget * share * ramp
                for share, ramp in zip(shares, ramp_fractions)
            )
        else:
            ratios = (
                float(_cfg_get(cfg.M1, "V502_M1_TO_BASE_OBJECTIVE_RATIO", 0.35)),
                float(_cfg_get(cfg.M1, "V502_M2_TO_BASE_OBJECTIVE_RATIO", 0.50)),
                float(_cfg_get(cfg.M1, "V502_M3_TO_BASE_OBJECTIVE_RATIO", 0.25)),
            )

        objectives = (m1_objective, m2_objective, m3_objective)
        base_magnitude = base_loss.detach().abs().clamp_min(1.0e-8)
        weighted = []
        effective_weights = []
        for ratio, objective in zip(ratios, objectives):
            if float(ratio) <= 0.0:
                weight = base_magnitude.new_zeros(())
                weighted_objective = objective * 0.0
            else:
                objective_magnitude = objective.detach().abs().clamp_min(1.0e-8)
                weight = float(ratio) * base_magnitude / objective_magnitude
                weighted_objective = weight * objective
            weighted.append(weighted_objective)
            effective_weights.append(weight.detach())
        proposal_objective = sum(weighted)
        diagnostics.update({
            "v490_m1_effective_weight": effective_weights[0],
            "v490_m2_direct_weight": effective_weights[1],
            "v490_m3_direct_weight": effective_weights[2],
            "v490_m1_weighted_objective": weighted[0].detach(),
            "v490_m2_weighted_objective": weighted[1].detach(),
            "v490_m3_weighted_objective": weighted[2].detach(),
            "v491_scaled_aux_to_base_ratio": base_loss.new_tensor(sum(ratios)),
            "v506_aux_budget_ratio": base_loss.new_tensor(
                float(_cfg_get(cfg.M1, "V506_TOTAL_AUX_TO_BASE_RATIO", sum(ratios)))
                if v505_enabled else sum(ratios)
            ),
            "v506_m1_routed_ratio": base_loss.new_tensor(ratios[0]),
            "v506_m2_routed_ratio": base_loss.new_tensor(ratios[1]),
            "v506_m3_routed_ratio": base_loss.new_tensor(ratios[2]),
        })
        return proposal_objective, float(ratios[0])

    m1_ratio = _effective_candidate_ratio(
        cfg, candidate_ratio_nominal, base_loss, m1_objective
    )
    m2_direct = float(_cfg_get(cfg.M1, "V490_M2_DIRECT_OBJECTIVE_WEIGHT", 1.0))
    m3_direct = float(_cfg_get(cfg.M1, "V490_M3_DIRECT_OBJECTIVE_WEIGHT", 1.0))
    proposal_objective = (
        m1_ratio * m1_objective
        + m2_direct * m2_objective
        + m3_direct * m3_objective
    )
    diagnostics.update({
        "v490_m1_effective_weight": base_loss.new_tensor(m1_ratio),
        "v490_m2_direct_weight": base_loss.new_tensor(m2_direct),
        "v490_m3_direct_weight": base_loss.new_tensor(m3_direct),
        "v490_m1_weighted_objective": (m1_ratio * m1_objective).detach(),
        "v490_m2_weighted_objective": (m2_direct * m2_objective).detach(),
        "v490_m3_weighted_objective": (m3_direct * m3_objective).detach(),
    })
    return proposal_objective, m1_ratio


def _v552r4201_assert_objective_route(
    cfg,
    *,
    compute_m1: bool,
    proposal_loss,
    effective_candidate_ratio: float,
    diagnostics,
    epoch: int,
):
    """Fail closed if the CLEAN graph computes M1 but silently routes it to zero.

    R4.20.1 deliberately keeps the legacy V532 direct objective ratios at zero
    because V538 owns the current component M1/M2 objectives.  CLEAN originally
    pruned the V538 routing keys, causing a valid non-zero M1 objective to be
    multiplied by zero for every batch.  This contract makes that state
    impossible to run silently again.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1_cfg, "V552R4201_ROOTFIX_ENABLED", False)):
        return
    if not compute_m1:
        return
    if not bool(_cfg_get(m1_cfg, "V538_INDEPENDENT_OBJECTIVE_ROUTING", False)):
        raise RuntimeError(
            "[V552R4201_OBJECTIVE_ROUTE] V538 independent objective routing is disabled; "
            "the clean V532 direct ratios are zero, so M1 would receive no gradient."
        )

    m1_scale_value = diagnostics.get("v538_m1_train_scale", 1.0)
    if isinstance(m1_scale_value, torch.Tensor):
        m1_scale = float(m1_scale_value.detach().cpu())
    else:
        m1_scale = float(m1_scale_value)
    configured_m1_ratio = max(
        0.0, float(_cfg_get(m1_cfg, "V538_M1_TO_BASE_OBJECTIVE_RATIO", 0.0))
    )
    route_expected = configured_m1_ratio > 0.0 and m1_scale > 0.0
    if not route_expected:
        diagnostics["v552r4201_objective_route_live"] = proposal_loss.new_zeros(())
        return

    routed_value = diagnostics.get("v538_m1_routed_ratio", None)
    if isinstance(routed_value, torch.Tensor):
        routed_ratio = float(routed_value.detach().cpu())
    elif routed_value is None:
        routed_ratio = 0.0
    else:
        routed_ratio = float(routed_value)
    proposal_magnitude = float(proposal_loss.detach().abs().cpu())
    problems = []
    if float(effective_candidate_ratio) <= 0.0:
        problems.append(f"effective_candidate_ratio={float(effective_candidate_ratio):.8f}")
    if routed_ratio <= 0.0:
        problems.append(f"v538_m1_routed_ratio={routed_ratio:.8f}")
    if proposal_magnitude <= 0.0:
        problems.append(f"proposal_loss_abs={proposal_magnitude:.8f}")
    if not bool(proposal_loss.requires_grad):
        problems.append("proposal_loss.requires_grad=False")
    if problems:
        raise RuntimeError(
            "[V552R4201_OBJECTIVE_ROUTE] live M1 objective was silently disconnected "
            f"at epoch={epoch + 1}: " + ", ".join(problems)
        )
    diagnostics["v552r4201_objective_route_live"] = proposal_loss.new_ones(())
    diagnostics["v552r4201_m1_routed_ratio"] = proposal_loss.new_tensor(routed_ratio)
    diagnostics["v552r4201_proposal_loss_abs"] = proposal_loss.detach().abs()


def _anchor_ratio(cfg):
    return float(_cfg_get(cfg.M1, "TEACHER_ANCHOR_WEIGHT", 0.15))


def _v541_is_component_m2_parameter(name: str) -> bool:
    """Return True for V541 selector-only parameters.

    M1 owns component generation.  The detached candidate-outcome encoder and
    all risk/calibration heads belong to M2 and must use the M2 optimizer and
    gradient clipping groups.
    """
    if name.startswith("module."):
        name = name[len("module."):]
    prefix = "v484_pipeline.pixel_composer.component_slot_generator."
    if not name.startswith(prefix):
        return False
    relative = name[len(prefix):]
    return relative.startswith((
        "m2_outcome_encoder.",
        "m2_selector_trunk.",
        "benefit_head.",
        "harm_head.",
        "rank_head.",
        "gain_head.",
        "gain_logvar_head.",
        "clean_gain_head.",
        # V551 region editor is part of M2, not the M1 proposal generator.
        "editor_route_context_adapter.",
        "editor_route_head.",
        "editor_dose_adjust_head.",
        "editor_relative_gain_head.",
        "editor_safety_outcome_head.",
        "editor_safety_benefit_head.",
        "editor_safety_harm_head.",
        "route_feature_adapter.",
        "candidate_utility_outcome_head.",
        "candidate_absolute_gain_head.",
        "safety_route_feature_adapter.",
        "utility_route_feature_adapter.",
        "safety_route_visual_adapter.",
        "utility_route_visual_adapter.",
        "safety_value_feature_adapter.",
        "utility_value_feature_adapter.",
        "safety_value_visual_adapter.",
        "utility_value_visual_adapter.",
        "editor_benefit_magnitude_head.",
        "editor_harm_magnitude_head.",
        "candidate_benefit_magnitude_head.",
        "candidate_harm_magnitude_head.",
        "atom_quality_head.",
        "r2_outcome_head.",
        "r2_benefit_magnitude_head.",
        "r2_harm_magnitude_head.",
        "editor_feature_adapter.",
        "composer_state_encoder.",
        "composer_candidate_adapter.",
        "composer_marginal_head.",
        "composer_stop_head.",
        "local_residual_head.",
    ))


def _v490_gradient_group(name: str) -> str:
    """Use exactly the same module partition as the V490 optimizer.

    Historical clipping recognised only ``m1_pse`` and ``ccv_m2``.  All V490
    parameters therefore fell into the Base group and the large Base norm
    scaled down M2/M3 gradients before every optimizer step.  The grouping below
    follows the actual ``v484_pipeline`` architecture.
    """
    if name.startswith("module."):
        name = name[len("module."):]
    if name.startswith(("vision_model.", "text_model.")):
        return "encoders"
    if name.startswith("pvl_adapters."):
        return "pvl"
    if name.startswith("v484_pipeline.safe_deployer."):
        return "m3"
    if name.startswith(
        "v484_pipeline.pixel_composer.component_slot_generator."
    ):
        return "m2" if _v541_is_component_m2_parameter(name) else "m1"
    if name.startswith("v484_pipeline.pixel_composer."):
        return "m2"
    if name.startswith("v484_pipeline."):
        return "m1"
    if name.startswith("ccv_m2."):
        return "m2"
    if name.startswith("m1_pse.v4g_refiner."):
        return "m2"
    if name.startswith("m1_pse."):
        return "m1"
    return "base"



_MHCS_BANK_PREFIXES = (
    "m1_pse.image_stem.",
    "m1_pse.semantic_proj.",
    "m1_pse.text_proj.",
    "m1_pse.pixel_fuse.",
    "m1_pse.global_context.",
    "m1_pse.context_film.",
    "m1_pse.distribution_trunk.",
    "m1_pse.mean_head.",
    "m1_pse.factor_head.",
    "m1_pse.diag_std_head.",
)
_MHCS_M2_PREFIXES = (
    "m1_pse.m2_surface.",
    # GEOTR-V4G is the active Stage-2 refiner under MHCS.  It must share the
    # structured M2 optimizer/LR/clipping ownership rather than falling through
    # the fail-closed unowned-m1_pse guard.
    "m1_pse.v4g_refiner.",
)


def _mhcs_root_gradient_group(name: str) -> str:
    """Fail-closed R4.7 ownership: Base/PVL/protected-M1-bank/outcome-regret-M2."""
    if name.startswith("module."):
        name = name[len("module."):]
    if name.startswith(("vision_model.", "text_model.")):
        return "encoders"
    if name.startswith("pvl_adapters."):
        return "pvl"
    if (
        name.startswith(_MHCS_BANK_PREFIXES)
        or name == "m1_pse.context_gate"
        or name == "m1_pse.m1_distribution_log_var"
    ):
        return "mhcs_bank"
    if name.startswith(_MHCS_M2_PREFIXES):
        return "mhcs_m2"
    if name.startswith("m1_pse."):
        raise RuntimeError(
            "MHCS-R4.8 found an unowned m1_pse parameter; add it explicitly to "
            f"protected M1-bank or structured M2 ownership before training: {name}"
        )
    return "base"


def _mhcs_no_decay(name: str, parameter) -> bool:
    clean = name[len("module."):] if name.startswith("module.") else name
    low = clean.lower()
    return bool(
        parameter.ndim <= 1
        or clean.endswith(".bias")
        or "norm" in low
        or clean == "m1_pse.m1_distribution_log_var"
        or clean == "m1_pse.context_gate"
    )


def _grad_l2(parameters):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return None
    device = params[0].grad.device
    sq = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in params:
        sq = sq + parameter.grad.detach().float().pow(2).sum()
    return torch.sqrt(sq)


def _v538_component_grad_diagnostics(model, reference):
    """Return component and V552-R2 head-specific live gradient norms."""
    m1_sq = reference.new_zeros(())
    m2_sq = reference.new_zeros(())
    composer_sq = reference.new_zeros(())
    state_sq = reference.new_zeros(())
    stop_sq = reference.new_zeros(())
    marginal_sq = reference.new_zeros(())
    r4204_occupancy_sq = reference.new_zeros(())
    m1_count = 0
    m2_count = 0
    prefix = "v484_pipeline.pixel_composer.component_slot_generator."
    for name, parameter in model.named_parameters():
        clean_name = name[len("module."):] if name.startswith("module.") else name
        if not clean_name.startswith(prefix) or parameter.grad is None:
            continue
        relative = clean_name[len(prefix):]
        grad = parameter.grad.detach().float()
        value = grad.pow(2).sum().to(reference.device)
        if _v541_is_component_m2_parameter(clean_name):
            m2_sq = m2_sq + value
            m2_count += int(parameter.numel())
        else:
            m1_sq = m1_sq + value
            m1_count += int(parameter.numel())
        if relative.startswith((
            "composer_state_encoder.", "composer_candidate_adapter.",
            "composer_marginal_head.", "composer_stop_head.",
        )):
            composer_sq = composer_sq + value
        if relative.startswith("composer_state_encoder."):
            state_sq = state_sq + value
        if relative.startswith("composer_stop_head."):
            stop_sq = stop_sq + value
        if relative.startswith("composer_marginal_head."):
            marginal_sq = marginal_sq + value
        if relative.startswith("r4204_residual_occupancy_head."):
            r4204_occupancy_sq = r4204_occupancy_sq + value
    m1_norm = torch.sqrt(m1_sq).detach()
    m2_norm = torch.sqrt(m2_sq).detach()
    return {
        # Historical names are explicitly pre-clip values.
        "v538_m1_grad_norm": m1_norm,
        "v538_m2_grad_norm": m2_norm,
        "v538_m1_grad_norm_preclip": m1_norm,
        "v538_m2_grad_norm_preclip": m2_norm,
        "v538_m1_grad_parameter_count": reference.new_tensor(float(m1_count)),
        "v538_m2_grad_parameter_count": reference.new_tensor(float(m2_count)),
        "v552_composer_grad_norm": torch.sqrt(composer_sq).detach(),
        "v552_state_encoder_grad_norm": torch.sqrt(state_sq).detach(),
        "v552_stop_head_grad_norm": torch.sqrt(stop_sq).detach(),
        "v552_marginal_head_grad_norm": torch.sqrt(marginal_sq).detach(),
        "v552r4204_occupancy_head_grad_norm": torch.sqrt(r4204_occupancy_sq).detach(),
    }


def _check_and_clip_grads(model, cfg):
    """Validate/clip gradients with fail-closed MHCS-R4.8 ownership.

    MHCS-R4.9 uses four independent groups (Base, PVL, Base-conditioned residual refiner, explicit verifier).
    This is still one joint optimisation step; independent clipping only prevents
    one task's norm from silently changing another group's effective step size.
    """
    exact_geotr_m1 = bool(
        _semlt(cfg)
        and str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "geotr_m1"
    )
    if exact_geotr_m1:
        # Preserve the successful MHCS Stage-1 optimization geometry after
        # physically extracting M1: Base, PVL and Transport are clipped
        # independently.  A single global norm would let the large Base/PVL
        # gradient silently shrink the M1 update even though the parameter
        # groups and learning rates are otherwise identical.
        grouped = {"base": [], "pvl": [], "m1": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            clean = name[len("module."):] if name.startswith("module.") else name
            if clean.startswith(("vision_model.", "text_model.")):
                raise RuntimeError(
                    f"GEOTR-M1 exact frozen encoder unexpectedly has gradient: {name}"
                )
            if clean.startswith("pvl_adapters."):
                grouped["pvl"].append(parameter)
            elif clean.startswith("m1_pse."):
                grouped["m1"].append(parameter)
            else:
                grouped["base"].append(parameter)

        limits = {
            "base": float(_cfg_get(cfg.TRAIN, "BASE_GRAD_CLIP", 1.0)),
            "pvl": float(_cfg_get(cfg.TRAIN, "PVL_GRAD_CLIP", 1.0)),
            "m1": float(
                _cfg_get(
                    cfg.TRAIN,
                    "M1_GRAD_CLIP",
                    _cfg_get(cfg.TRAIN, "MHCS_BANK_GRAD_CLIP", 1.0),
                )
            ),
        }
        diagnostics = {}
        for group_name, parameters in grouped.items():
            if not parameters:
                continue
            pre = _grad_l2(parameters)
            if pre is None or not torch.isfinite(pre):
                raise FloatingPointError(
                    f"Non-finite GEOTR-M1 exact {group_name} gradient norm: {pre}"
                )
            if limits[group_name] > 0.0:
                torch.nn.utils.clip_grad_norm_(parameters, limits[group_name])
            post = _grad_l2(parameters)
            if post is None or not torch.isfinite(post):
                raise FloatingPointError(
                    f"Non-finite clipped GEOTR-M1 exact {group_name} gradient norm: {post}"
                )
            diagnostics[f"geotr_m1_grad_{group_name}_preclip"] = pre.detach()
            diagnostics[f"geotr_m1_grad_{group_name}_postclip"] = post.detach()
        return diagnostics

    if _mhcs(cfg):
        grouped = {
            "base": [],
            "pvl": [],
            "mhcs_bank": [],
            "mhcs_m2": [],
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            group = _mhcs_root_gradient_group(name)
            if group == "encoders":
                raise RuntimeError(
                    f"MHCS-R4.8 frozen encoder unexpectedly has gradient: {name}"
                )
            grouped[group].append(parameter)

        limits = {
            "base": float(_cfg_get(cfg.TRAIN, "BASE_GRAD_CLIP", 1.0)),
            "pvl": float(_cfg_get(cfg.TRAIN, "PVL_GRAD_CLIP", 1.0)),
            "mhcs_bank": float(_cfg_get(cfg.TRAIN, "MHCS_BANK_GRAD_CLIP", 1.0)),
            "mhcs_m2": float(_cfg_get(cfg.TRAIN, "MHCS_M2_GRAD_CLIP", _cfg_get(cfg.TRAIN, "MHCS_ROUTER_GRAD_CLIP", 1.0))),
        }
        diagnostics = {}
        for group_name, parameters in grouped.items():
            if not parameters:
                continue
            pre = _grad_l2(parameters)
            if pre is None or not torch.isfinite(pre):
                raise FloatingPointError(
                    f"Non-finite {group_name} gradient norm before clipping: {pre}"
                )
            limit = limits[group_name]
            if limit > 0:
                torch.nn.utils.clip_grad_norm_(parameters, limit)
            post = _grad_l2(parameters)
            if post is None or not torch.isfinite(post):
                raise FloatingPointError(
                    f"Non-finite {group_name} gradient norm after clipping: {post}"
                )
            diagnostics[f"mhcs_grad_{group_name}_preclip"] = pre.detach()
            diagnostics[f"mhcs_grad_{group_name}_postclip"] = post.detach()
        return diagnostics

    strict_joint = bool(
        m1_enabled(cfg)
        and _cfg_get(cfg.M1, "V470_STRICT_JOINT_E2E", False)
    )
    if strict_joint:
        grouped = {
            "base": [],
            "pvl": [],
            "m1": [],
            "m2": [],
            "m3": [],
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            group = _v490_gradient_group(name)
            if group == "encoders":
                raise RuntimeError(
                    f"Frozen encoder parameter unexpectedly has a gradient: {name}"
                )
            grouped[group].append(parameter)

        clip_values = {
            "base": float(_cfg_get(cfg.TRAIN, "BASE_GRAD_CLIP", 1.0)),
            "pvl": float(
                _cfg_get(
                    cfg.TRAIN,
                    "PVL_GRAD_CLIP",
                    _cfg_get(cfg.TRAIN, "BASE_GRAD_CLIP", 1.0),
                )
            ),
            "m1": float(_cfg_get(cfg.TRAIN, "M1_GRAD_CLIP", 1.0)),
            "m2": float(_cfg_get(cfg.TRAIN, "M2_GRAD_CLIP", 1.0)),
            "m3": float(
                _cfg_get(
                    cfg.TRAIN,
                    "M3_GRAD_CLIP",
                    _cfg_get(cfg.TRAIN, "M2_GRAD_CLIP", 1.0),
                )
            ),
        }
        for group_name, parameters in grouped.items():
            if not parameters:
                continue
            limit = clip_values[group_name]
            if limit > 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, limit)
            else:
                norm = torch.linalg.vector_norm(
                    torch.stack(
                        [parameter.grad.detach().norm(2) for parameter in parameters]
                    ),
                    2,
                )
            if not torch.isfinite(norm):
                raise FloatingPointError(
                    f"Non-finite {group_name} gradient norm: {norm}"
                )
        return {}

    grad_clip = float(_cfg_get(cfg.TRAIN, "GRAD_CLIP", 0.0))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm before optimizer step: {grad_norm}"
            )
    return {}


def _v481_pcgrad_backward(
    model,
    base_objective,
    aux_objective,
    grad_accum_steps: int,
):
    """Two-objective PCGrad for Base-vs-M1 conflict.

    This is not freezing.  All trainable parameters can receive gradients.
    When the M1 auxiliary gradient conflicts with the Base objective gradient,
    only the conflicting component is projected away.
    """
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    g_base = torch.autograd.grad(
        base_objective,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    g_aux = torch.autograd.grad(
        aux_objective,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    dot = None
    base_norm_sq = None
    aux_norm_sq = None

    for gb, ga in zip(g_base, g_aux):
        if gb is None or ga is None:
            continue
        d = (gb.detach() * ga.detach()).sum()
        bn = gb.detach().pow(2).sum()
        an = ga.detach().pow(2).sum()
        dot = d if dot is None else dot + d
        base_norm_sq = bn if base_norm_sq is None else base_norm_sq + bn
        aux_norm_sq = an if aux_norm_sq is None else aux_norm_sq + an

    if dot is None:
        return {
            "cem_v481_pcgrad_cos": 0.0,
            "cem_v481_pcgrad_conflict": 0.0,
        }

    denom = (
        base_norm_sq.clamp_min(1.0e-12).sqrt()
        * aux_norm_sq.clamp_min(1.0e-12).sqrt()
    )
    cos = dot / denom.clamp_min(1.0e-12)
    conflict = bool(dot.item() < 0.0)

    projected_aux = []
    if conflict:
        coeff = dot / base_norm_sq.clamp_min(1.0e-12)
        for gb, ga in zip(g_base, g_aux):
            if ga is None:
                projected_aux.append(None)
            elif gb is None:
                projected_aux.append(ga)
            else:
                projected_aux.append(ga - coeff * gb)
    else:
        projected_aux = list(g_aux)

    scale = 1.0 / float(max(1, int(grad_accum_steps)))

    for p, gb, ga in zip(params, g_base, projected_aux):
        if gb is None and ga is None:
            continue
        total = None
        if gb is not None:
            total = gb
        if ga is not None:
            total = ga if total is None else total + ga
        if total is None:
            continue
        total = total.detach() * scale
        if p.grad is None:
            p.grad = total.clone()
        else:
            p.grad.add_(total)

    return {
        "cem_v481_pcgrad_cos": float(cos.detach().cpu()),
        "cem_v481_pcgrad_conflict": float(conflict),
    }



# V484_RUNTIME_AUX_INJECTION_BEGIN
def _v484_loss_active(cfg):
    if not m1_enabled(cfg):
        return False
    m1 = _cfg_get(cfg, "M1", None)
    loss_version = str(
        _cfg_get(
            m1,
            "LOSS_MODE",
            _cfg_get(m1, "M1_LOSS_VERSION", ""),
        )
    ).strip().lower()
    return loss_version in {
        "v484_error_state_causal",
        "v485_error_state_causal",
        "v485_v484only_m1local",
        "v486_fixedbase_m1local_repair",
        "v487_basesafe_gated_repair",
        "v488_pixel_composer_safe_deployer",
        "v489_end_to_end_sparse_region_expert",
        "v490_root_cause_intervention_risk",
        "v531_typed_sparse_refiner",
        "v532_unified_sparse_refiner",
        "clean_dynamic_component_set",
        "tc_drcs",
    }


def _normalize_v484_output(v484_out, base_logits):
    """Normalize V484 pipeline output to a dict.

    Supported returns:
      1) dict
      2) (candidate_logits_or_probs, aux_dict)
      3) (aux_dict, candidate_logits_or_probs)
      4) tuple/list containing one dict and one tensor
    """
    if isinstance(v484_out, dict):
        out = dict(v484_out)
    elif isinstance(v484_out, (tuple, list)):
        out = {}
        tensor_payload = None

        for item in v484_out:
            if isinstance(item, dict):
                out.update(item)
            elif torch.is_tensor(item):
                # Prefer [B,K,H,W] candidate tensor.
                if item.ndim >= 3:
                    tensor_payload = item

        if tensor_payload is not None:
            t = tensor_payload
            if t.ndim == 3:
                t = t[:, None]

            # Heuristic:
            # - logits may be outside [0,1]
            # - probs are usually already [0,1]
            if bool((t.detach().min() >= 0.0).item()) and bool((t.detach().max() <= 1.0).item()):
                out.setdefault("candidate_probs", t.clamp(1.0e-6, 1.0 - 1.0e-6))
                out.setdefault(
                    "candidates",
                    torch.logit(out["candidate_probs"].clamp(1.0e-6, 1.0 - 1.0e-6)),
                )
            else:
                out.setdefault("candidates", t)
                out.setdefault(
                    "candidate_probs",
                    torch.sigmoid(t).clamp(1.0e-6, 1.0 - 1.0e-6),
                )

        if not out:
            raise RuntimeError(
                "V484 pipeline returned tuple/list, but no dict or tensor "
                "payload could be extracted. tuple_types="
                + str([type(x) for x in v484_out])
            )
    else:
        raise RuntimeError(
            "V484 pipeline must return dict or tuple/list, got: "
            + str(type(v484_out))
        )

    # Always ensure factual C0 exists and is detached.
    if "c0_prob_detached" not in out:
        b = base_logits
        if b.ndim == 3:
            b = b[:, None]
        out["c0_prob_detached"] = torch.sigmoid(
            b.detach()
        ).clamp(1.0e-6, 1.0 - 1.0e-6)

    # Normalize candidate fields.
    if "candidate_probs" in out:
        cp = out["candidate_probs"]
        if cp.ndim == 3:
            cp = cp[:, None]
        out["candidate_probs"] = cp.clamp(1.0e-6, 1.0 - 1.0e-6)
        out["candidates"] = torch.logit(
            out["candidate_probs"].clamp(1.0e-6, 1.0 - 1.0e-6)
        )

    elif "local_candidate_logits" in out:
        logits = out["local_candidate_logits"]
        if logits.ndim == 3:
            logits = logits[:, None]
        out["local_candidate_logits"] = logits
        out["candidates"] = logits
        out["candidate_probs"] = torch.sigmoid(
            logits
        ).clamp(1.0e-6, 1.0 - 1.0e-6)

    elif "candidates" in out:
        logits = out["candidates"]
        if logits.ndim == 3:
            logits = logits[:, None]
        out["candidates"] = logits
        out["candidate_probs"] = torch.sigmoid(
            logits
        ).clamp(1.0e-6, 1.0 - 1.0e-6)

    else:
        raise RuntimeError(
            "V484 output has no candidate_probs/local_candidate_logits/candidates. "
            "keys=" + str(sorted(out.keys()))
        )

    # If local logits are missing but candidates exist, expose candidates as local
    # logits so v484_loss can still read local_candidate_logits.
    if "local_candidate_logits" not in out:
        out["local_candidate_logits"] = out["candidates"]

    # Minimal safe defaults for optional fields used by v484_loss.
    cp = out["candidate_probs"]
    bsz, slots, h, w = cp.shape
    local_k = max(slots - 1, 1)
    device = cp.device
    dtype = cp.dtype

    if "local_active_mask" not in out:
        out["local_active_mask"] = torch.ones(
            bsz,
            local_k,
            device=device,
            dtype=dtype,
        )

    if "local_supports" not in out:
        c0 = out["c0_prob_detached"]
        if c0.ndim == 3:
            c0 = c0[:, None]
        diff = (cp[:, 1:] - c0).abs() if slots > 1 else torch.zeros(
            bsz, 1, h, w, device=device, dtype=dtype
        )
        out["local_supports"] = diff

    if "local_deltas" not in out:
        c0 = out["c0_prob_detached"]
        if c0.ndim == 3:
            c0 = c0[:, None]
        delta = (cp[:, 1:] - c0) if slots > 1 else torch.zeros(
            bsz, 1, h, w, device=device, dtype=dtype
        )
        out["local_deltas"] = delta

    if "global_active_mask" not in out:
        out["global_active_mask"] = torch.zeros(
            bsz,
            0,
            device=device,
            dtype=dtype,
        )

    return out


def _inject_v484_aux_if_needed(model, cfg, images, base_logits, aux):
    """Run registered V484 pipeline and inject normalized outputs into aux."""
    if not _v484_loss_active(cfg):
        return aux

    if aux is None:
        aux = {}
    if not isinstance(aux, dict):
        raise RuntimeError(
            "V484 requires aux to be a dict, got: "
            + str(type(aux))
        )

    if (
        "c0_prob_detached" in aux
        and (
            "candidate_probs" in aux
            or "candidates" in aux
            or "local_candidate_logits" in aux
        )
    ):
        normalized = _normalize_v484_output(aux, base_logits)
        merged = dict(aux)
        merged.update(normalized)
        return merged

    pipeline = None
    for name in (
        "v484_pipeline",
        "v484_causal_pipeline",
        "v484_error_state_pipeline",
        "v484",
    ):
        module = getattr(model, name, None)
        if module is not None:
            pipeline = module
            break

    if pipeline is None:
        m1_pse = getattr(model, "m1_pse", None)
        if m1_pse is not None:
            for name in (
                "v484_pipeline",
                "v484_causal_pipeline",
                "v484_error_state_pipeline",
                "v484",
            ):
                module = getattr(m1_pse, name, None)
                if module is not None:
                    pipeline = module
                    break

    if pipeline is None:
        raise RuntimeError(
            "V484 loss is active, but model has no registered V484 pipeline."
        )

    try:
        v484_out = pipeline(
            image=images,
            base_logits=base_logits,
            image_features=None,
            text_features=None,
            semantic_map=None,
        )
    except TypeError:
        v484_out = pipeline(images, base_logits)

    v484_dict = _normalize_v484_output(v484_out, base_logits)

    merged = dict(aux)
    merged.update(v484_dict)

    missing = [
        key
        for key in ("c0_prob_detached", "candidate_probs", "candidates", "local_candidate_logits")
        if key not in merged
    ]
    if missing:
        raise RuntimeError(
            "V484 aux injection failed. Missing keys: " + str(missing)
            + " available=" + str(sorted(merged.keys()))
        )

    return merged
# V484_RUNTIME_AUX_INJECTION_END





# V487_M1_ONLY_ABLATION_HELPER_BEGIN
def _v487_is_m1_only_ablation(cfg):
    return bool(
        m1_enabled(cfg)
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V487_M1_ONLY_ABLATION", False))
    )


def _v488_is_m2m3_only(cfg):
    """True only for the V488 stage that freezes Base and the complete M1 bank."""
    return bool(
        m1_enabled(cfg)
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V488_M2M3_ONLY", False))
    )

def _v519_is_m2_only(cfg):
    """Fixed Base/M1 candidate-bank protocol for regional M2 training."""
    m1_cfg = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and (
            bool(_cfg_get(m1_cfg, "V519_M2_ONLY", False))
            or bool(_cfg_get(m1_cfg, "V524_M2_ONLY", False))
        )
    )


def _v519_is_joint_m1_m2(cfg):
    """V519/V520/V521/V522/V523 protocol: train Base/PVL/M1/M2 and bypass M3.

    M1 keeps its validated V518 architecture and receives its own factual/bank
    objective.  M2 observes detached Base/M1 candidates, so its gradients can
    never alter Base or M1.  V520/V521/V522/V523 retain this loading/optimizer contract while
    replacing the collapsed flat router with a gate + conditional selector.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and (
            bool(_cfg_get(m1_cfg, "V519_JOINT_M1_M2", False))
            or bool(_cfg_get(m1_cfg, "V520_JOINT_M1_M2", False))
            or bool(_cfg_get(m1_cfg, "V521_JOINT_M1_M2", False))
            or bool(_cfg_get(m1_cfg, "V522_JOINT_M1_M2", False))
            or bool(_cfg_get(m1_cfg, "V523_JOINT_M1_M2", False))
            or bool(_cfg_get(m1_cfg, "V524_JOINT_M1_M2", False))
        )
    )


def _v519_uses_v518_source(cfg):
    """Return True only when the run actually requires a trained V518 source.

    V519/V520/V521/V522/V523 may also run from the official BUSI Base checkpoint.
    When V515_OFFICIAL_BASE_INIT is enabled, only Base/PVL tensors are loaded;
    M1 and M2 are initialized by the current model and trained jointly.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    official_base_init = bool(
        _cfg_get(m1_cfg, "V515_OFFICIAL_BASE_INIT", False)
    )

    return bool(
        (
            _v519_is_m2_only(cfg)
            or _v519_is_joint_m1_m2(cfg)
        )
        and not official_base_init
    )


def _v531_uses_source_checkpoint(cfg):
    """Whether V531 requires a validated Base+M1 initialization checkpoint."""
    m1_cfg = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and bool(_cfg_get(m1_cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False))
        and bool(_cfg_get(m1_cfg, "V531_SOURCE_REQUIRES_M1", True))
    )


def _v489_is_end_to_end(cfg):
    """V489: freeze only image/text encoders; train every downstream module."""
    return bool(
        m1_enabled(cfg)
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V489_END_TO_END_ENABLED", False))
    )


def _v490_is_end_to_end(cfg):
    """Unified downstream task graph trained jointly with the coarse decoder."""
    m1_cfg = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and (
            _clean_dynamic_component_set(cfg)
            or bool(_cfg_get(m1_cfg, "V490_ROOT_CAUSE_ENABLED", False))
            or bool(_cfg_get(m1_cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False))
        )
    )


def _v489_or_v490_end_to_end(cfg):
    return _v489_is_end_to_end(cfg) or _v490_is_end_to_end(cfg)


def _v487_apply_m1_only_ablation_freeze(model, cfg, logger=None):
    """V487A: M1-only ablation.

    Keep Base-safe E2E training, but remove M2/M3 confounding:
      - train M1 error-state head and candidate generators only;
      - freeze local verifier / rejector / CCV / text verifier;
      - final deployment remains Preserve/Base by config.
    """
    if not _v487_is_m1_only_ablation(cfg):
        return

    trainable_m1_prefixes = (
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
        "v484_pipeline.global_generator.",
    )
    frozen_prefixes = (
        "ccv_m2.",
        "m2_text_verifier.",
    )
    frozen_name_tokens = (
        "local_m2",
        "local_verifier",
        "verifier",
        "rejector",
        "m3",
    )

    trainable_count = 0
    frozen_count = 0

    for name, parameter in model.named_parameters():
        if name.startswith(frozen_prefixes):
            parameter.requires_grad_(False)
            frozen_count += parameter.numel()
            continue

        if name.startswith("v484_pipeline."):
            allow = any(name.startswith(prefix) for prefix in trainable_m1_prefixes)
            if any(token in name.lower() for token in frozen_name_tokens):
                allow = False
            parameter.requires_grad_(bool(allow))
            if allow:
                trainable_count += parameter.numel()
            else:
                frozen_count += parameter.numel()

    message = (
        "[V487A_M1_ONLY_ABLATION] enabled | "
        f"trainable_m1_params={trainable_count/1e6:.3f}M | "
        f"frozen_m2_m3_params={frozen_count/1e6:.3f}M"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)
# V487_M1_ONLY_ABLATION_HELPER_END


def _compute_proposal_loss(loss_version: str, cfg, aux, masks, base_logits, epoch):
    """Dispatch to the correct M1 proposal loss function based on loss_version."""
    if loss_version in {"semlt_autozero", "semlt_autozero_transport"}:
        return compute_semlt_autozero_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    if loss_version in {"geotr_m1", "exact_geometry_transport"}:
        return compute_geotr_m1_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    if loss_version in {"semlt", "semlt_logit_transport"}:
        return compute_semlt_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    # Current semantic loss names after source-file cleanup.
    if loss_version in {"mhcs", "multi_hypothesis_composition"}:
        return compute_multi_hypothesis_composition_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    if loss_version in {"v484_error_state_causal", "v485_error_state_causal", "v485_v484only_m1local", "v486_fixedbase_m1local_repair", "v487_basesafe_gated_repair", "v488_pixel_composer_safe_deployer", "v489_end_to_end_sparse_region_expert", "v490_root_cause_intervention_risk", "v531_typed_sparse_refiner", "v532_unified_sparse_refiner", "clean_dynamic_component_set", "tc_drcs"}:
        from utils.v484_loss import compute_v484_loss
        return compute_v484_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    if loss_version == "reference_quality":
        return compute_reference_candidate_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    if loss_version in {
        "text_prompted_hypothesis",
        "compositional_error_modes",
        "cem_candidates",
        "v474_cem",
    }:
        return compute_text_prompted_hypothesis_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    if loss_version == "v463_residual_ccv_joint":
        return compute_v463_residual_ccv_joint_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    if loss_version == "unified_m1_safe_fusion":
        return compute_unified_m1_safe_fusion_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    if loss_version == "safe_residual_candidates":
        return compute_safe_residual_candidate_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
        )
    elif loss_version == "evidence_guided_candidate_control":
        return compute_evidence_guided_candidate_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    elif loss_version == "candidate_edit_control":
        return compute_candidate_edit_control_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
            epoch=epoch,
        )
    elif loss_version == "candidate_calibration":
        return compute_candidate_calibration_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
        )
    elif loss_version == "candidate_consensus":
        return compute_candidate_consensus_loss(
            cfg,
            aux["candidates"],
            masks,
            aux,
        )

    if loss_version == "v383_conservative_action_value":
        return compute_m1_v383_conservative_action_value_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v396_fecg":
        return compute_m1_v396_fecg_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    elif loss_version == "v393_preserve_aware_edit_control":
        return compute_m1_v393_preserve_aware_edit_control_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    elif loss_version == "v394_counterfactual_value_rank":
        return compute_m1_v394_counterfactual_value_rank_loss(
            cfg, aux["candidates"], masks, aux, epoch=epoch
        )
    elif loss_version in {"v381_lesion_background_calibrated_atomic", "v382_action_conditional_quantile_atomic", "v391_lesion_background_mlp_calibrated_atomic", "v392_dense_patch_text_falsification"}:
        return compute_m1_v381_lesion_background_calibrated_atomic_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v38_casewise_falsified_delta_consensus":
        return compute_m1_v38_casewise_falsified_delta_consensus_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v37_text_falsified_structural_consensus":
        return compute_m1_v37_text_falsified_structural_consensus_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v36_casewise_plackett_luce":
        return compute_m1_v36_casewise_plackett_luce_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v35_residual_purified_world_model":
        return compute_m1_v35_residual_purified_world_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v34_spatial_quantile_world_model":
        return compute_m1_v34_spatial_quantile_world_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v32_island_phaseb":
        if bool(_cfg_get(cfg.M1, "V33_SIGNED_GAIN_REGRESSION", False)):
            return compute_m1_v33_signed_gain_regression_loss(cfg, aux["candidates"], masks, aux)
        return compute_m1_v32_island_phaseb_loss(cfg, aux["candidates"], masks, aux)
    elif loss_version == "v31_competitive_utility":
        return compute_m1_v31_competitive_utility_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)
    elif loss_version == "v25_type_conditional_utility":
        return compute_m1_v25_type_conditional_utility_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)
    elif loss_version == "v20_unified_action_cf_set":
        return compute_m1_v20_unified_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)
    elif loss_version == "v19_action_bank":
        return compute_m1_v19_action_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)
    elif loss_version == "v18_atomic":
        return compute_m1_v18_atomic_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)
    else:
        return compute_trainable_pse_loss(base_logits, aux["candidates"], aux, masks, cfg, epoch=epoch)


def _load_checkpoint_state(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def _checkpoint_model_state_with_deployed_ema(path):
    """Return the exact deployed checkpoint weights (EMA when selected as EMA)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = dict(state)
    if (
        isinstance(checkpoint, dict)
        and str(checkpoint.get("weight_source", "")).lower() == "ema"
        and isinstance(checkpoint.get("ema_shadow"), dict)
    ):
        for key, value in checkpoint["ema_shadow"].items():
            if key in state and tuple(state[key].shape) == tuple(value.shape):
                state[key] = value
    return state


def _load_v488_m1_source_checkpoint(model, path, logger):
    """Load V487A Base+M1 exactly, leaving only newly introduced V488 M2/M3 missing."""
    state_dict = _checkpoint_model_state_with_deployed_ema(path)
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_buffers = {
        "text_model.transformer.embeddings.position_ids",
    }
    allowed_new_prefixes = (
        "v484_pipeline.pixel_composer.",
        "v484_pipeline.safe_deployer.",
    )
    illegal_missing = [
        key for key in result.missing_keys
        if key not in allowed_missing_buffers
        and not key.startswith(allowed_new_prefixes)
    ]
    if illegal_missing or result.unexpected_keys:
        raise RuntimeError(
            "V488 source checkpoint is incompatible. "
            f"Illegal missing keys: {illegal_missing}; unexpected keys: {result.unexpected_keys}"
        )

    required_prefixes = (
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
    )
    source_keys = set(state_dict)
    absent = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in source_keys)]
    if absent:
        raise RuntimeError(
            "V488 requires a trained V487A M1 checkpoint; missing source prefixes: "
            + str(absent)
        )
    logger.info(
        "[V488_SOURCE] Loaded deployed V487A Base+M1 from %s; only new M2/M3 tensors are initialized. "
        "missing_new=%d",
        path,
        len(result.missing_keys),
    )


def _load_v489_source_checkpoint(model, path, logger):
    """Load deployed V487A Base+M1 into the new V489 architecture.

    Shape-compatible tensors are loaded, V489 M2/M3 tensors remain newly
    initialized, and every loaded downstream tensor is subsequently trainable.
    This is initialization, not freezing.
    """
    source = _checkpoint_model_state_with_deployed_ema(path)
    destination = model.state_dict()
    compatible = {}
    shape_mismatch = []
    for key, value in source.items():
        if key not in destination:
            continue
        if tuple(destination[key].shape) != tuple(value.shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(destination[key].shape)))
            continue
        # V489 deliberately reinitializes the redesigned M2/M3 even when the
        # source checkpoint happens to contain an older V488 implementation.
        if key.startswith((
            "v484_pipeline.pixel_composer.",
            "v484_pipeline.safe_deployer.",
        )):
            continue
        compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    required_prefixes = (
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
        "pvl_adapters.",
        "mask_head.",
        "upscale.",
    )
    loaded_keys = set(compatible)
    absent = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in loaded_keys)]
    if absent:
        raise RuntimeError(
            "V489 requires a deployed V487A Base+M1 checkpoint; missing loaded prefixes: "
            + str(absent)
        )
    logger.info(
        "[V489_SOURCE] Loaded shape-compatible deployed V487A Base+M1 from %s | "
        "loaded=%d new_or_missing=%d ignored_shape_mismatch=%d. All downstream "
        "tensors will be trainable after the encoder-only freeze contract.",
        path,
        len(compatible),
        len(result.missing_keys),
        len(shape_mismatch),
    )
    if shape_mismatch:
        logger.info("[V489_SOURCE] shape-mismatch preview=%s", shape_mismatch[:12])




def _load_v531_source_checkpoint(model, path, logger, cfg):
    """Load a validated Base+V503/V518 M1 into V531.

    Router-stage runs reinitialize the new V531 refiner.  Joint-stage runs may
    continue from a trained V531 router by setting
    ``V531_REINITIALIZE_REFINER=false``.  Historical candidate-ranker/M3
    tensors are never required.
    """
    source = _checkpoint_model_state_with_deployed_ema(path)
    destination = model.state_dict()
    reinitialize_refiner = bool(
        _cfg_get(_cfg_get(cfg, "M1", None), "V531_REINITIALIZE_REFINER", True)
    )
    compatible = {}
    shape_mismatch = []
    skipped_refiner = 0
    for key, value in source.items():
        key2 = key[7:] if key.startswith("module.") else key
        if reinitialize_refiner and key2.startswith("v484_pipeline.pixel_composer."):
            skipped_refiner += 1
            continue
        if key2.startswith("v484_pipeline.safe_deployer."):
            continue
        if key2 not in destination:
            continue
        if tuple(destination[key2].shape) != tuple(value.shape):
            shape_mismatch.append(
                (key2, tuple(value.shape), tuple(destination[key2].shape))
            )
            continue
        compatible[key2] = value

    result = model.load_state_dict(compatible, strict=False)
    required_prefixes = (
        "pvl_adapters.",
        "mask_head.",
        "upscale.",
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
    )
    loaded = set(compatible)
    absent = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in loaded)
    ]
    if absent:
        raise RuntimeError(
            "V531 requires a validated Base+typed-M1 checkpoint; missing prefixes: "
            + str(absent)
        )
    if not reinitialize_refiner and not any(
        key.startswith("v484_pipeline.pixel_composer.") for key in loaded
    ):
        raise RuntimeError(
            "V531_REINITIALIZE_REFINER=false requires a V531 checkpoint that "
            "contains v484_pipeline.pixel_composer.* tensors."
        )
    logger.info(
        "[V531_SOURCE] Loaded Base/PVL/typed-M1 from %s | loaded=%d "
        "new_or_missing=%d reinitialize_refiner=%s skipped_refiner=%d "
        "shape_mismatch=%d",
        path, len(compatible), len(result.missing_keys),
        reinitialize_refiner, skipped_refiner, len(shape_mismatch),
    )
    if shape_mismatch:
        logger.info("[V531_SOURCE] shape mismatch preview=%s", shape_mismatch[:12])


def _load_v519_source_checkpoint(model, path, logger):
    """Load validated V518 Base/PVL/M1 and reinitialize the regional M2.

    Old V505 M2/M3 tensors are intentionally skipped because the family-aware
    regional composer has a different parameterization.  In corrected V519,
    the loaded Base/PVL/M1 tensors remain trainable; M2 sees detached candidate
    observations so its gradients cannot corrupt them.
    """
    source = _checkpoint_model_state_with_deployed_ema(path)
    destination = model.state_dict()
    compatible = {}
    shape_mismatch = []
    for key, value in source.items():
        key2 = key[7:] if key.startswith("module.") else key
        if key2.startswith((
            "v484_pipeline.pixel_composer.",
            "v484_pipeline.safe_deployer.",
        )):
            continue
        if key2 not in destination:
            continue
        if tuple(destination[key2].shape) != tuple(value.shape):
            shape_mismatch.append(
                (key2, tuple(value.shape), tuple(destination[key2].shape))
            )
            continue
        compatible[key2] = value

    result = model.load_state_dict(compatible, strict=False)
    required_prefixes = (
        "pvl_adapters.",
        "mask_head.",
        "upscale.",
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
    )
    loaded = set(compatible)
    absent = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in loaded)
    ]
    if absent:
        raise RuntimeError(
            "V519 requires a validated V518 Base+M1 checkpoint; missing prefixes: "
            + str(absent)
        )
    illegal_missing = [
        key for key in result.missing_keys
        if key != "text_model.transformer.embeddings.position_ids"
        and not key.startswith((
            "v484_pipeline.pixel_composer.",
            "v484_pipeline.safe_deployer.",
        ))
    ]
    if illegal_missing or result.unexpected_keys:
        raise RuntimeError(
            "V519 source checkpoint is incompatible. "
            f"illegal_missing={illegal_missing[:40]} "
            f"unexpected={result.unexpected_keys[:40]}"
        )
    logger.info(
        "[V519_SOURCE] Loaded validated V518 Base/PVL/M1 initialization from %s | "
        "loaded=%d new_m2m3=%d shape_mismatch=%d",
        path,
        len(compatible),
        len(result.missing_keys),
        len(shape_mismatch),
    )
    if shape_mismatch:
        logger.info("[V519_SOURCE] shape mismatch preview=%s", shape_mismatch[:12])


def _load_v515_official_base_checkpoint(model, path, logger):
    """Load only official MedCLIPSeg Base/PVL tensors.

    This is for official-checkpoint ablation only:
      loaded: pvl_adapters.*, mask_head.*, upscale.*
      ignored: image/text encoders, M1/M2/M3, incompatible tensors.
    """
    source = _checkpoint_model_state_with_deployed_ema(path)
    destination = model.state_dict()

    keep_prefixes = (
        "pvl_adapters.",
        "mask_head.",
        "upscale.",
    )

    compatible = {}
    mismatched = []
    ignored = 0

    for key, value in source.items():
        key2 = key[7:] if key.startswith("module.") else key
        if not key2.startswith(keep_prefixes):
            ignored += 1
            continue
        if key2 not in destination:
            ignored += 1
            continue
        if tuple(destination[key2].shape) != tuple(value.shape):
            mismatched.append((key2, tuple(value.shape), tuple(destination[key2].shape)))
            continue
        compatible[key2] = value

    required = ("pvl_adapters.", "mask_head.", "upscale.")
    missing_required = [
        prefix for prefix in required
        if not any(k.startswith(prefix) for k in compatible)
    ]
    if missing_required:
        raise RuntimeError(
            "V515 official Base init failed; missing required loaded prefixes: "
            + str(missing_required)
            + f" loaded_keys_preview={list(compatible)[:20]}"
        )

    result = model.load_state_dict(compatible, strict=False)

    logger.info(
        "[V515_OFFICIAL_BASE_INIT] Selectively loaded official Base/PVL from %s | "
        "loaded=%d ignored=%d missing_after_partial_load=%d mismatched=%d",
        path,
        len(compatible),
        ignored,
        len(result.missing_keys),
        len(mismatched),
    )
    if mismatched:
        logger.info("[V515_OFFICIAL_BASE_INIT] mismatch preview=%s", mismatched[:12])

def _load_initial_base_checkpoint(model, path, logger):
    state_dict = _load_checkpoint_state(path)
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_buffers = {
        "text_model.transformer.embeddings.position_ids",
    }
    illegal_missing = [
        key for key in result.missing_keys
        if not (
            key.startswith("m1_pse.")
            or key.startswith("m2_text_verifier.")
            or key.startswith("m2_tide_repair_head.")
            or key.startswith("ccv_m2.")
            or key in allowed_missing_buffers
        )
    ]
    if illegal_missing or result.unexpected_keys:
        raise RuntimeError(
            "Base checkpoint is incompatible with the current M1 model. "
            f"Missing non-M1 keys: {illegal_missing}; unexpected keys: {result.unexpected_keys}"
        )
    logger.info(
        "Loaded B0 checkpoint into Student. Missing new M1 keys: %s",
        result.missing_keys,
    )



def _load_frozen_m1_checkpoint(model, path, logger):
    """Load a verified pretrained M1 candidate bank; never freeze random M1."""
    if not path:
        raise ValueError(
            "V404 requires M1.TIDE_M1_CHECKPOINT. It must be the Val-selected "
            "M1 checkpoint whose IndividualOracle is higher than Preserve."
        )

    state = _load_checkpoint_state(path)

    source = {
        key: value
        for key, value in state.items()
        if key.startswith("m1_pse.")
        and ".m2_tide_repair_head." not in key
    }

    if not source:
        raise RuntimeError(
            f"No pretrained m1_pse.* tensors found in TIDE_M1_CHECKPOINT: {path}"
        )

    destination = model.state_dict()
    compatible = {}
    shape_mismatch = []

    for key, value in source.items():
        if key not in destination:
            continue

        if tuple(destination[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(destination[key].shape))
            )
            continue

        compatible[key] = value

    core_prefixes = (
        "m1_pse.trunk.",
        "m1_pse.actionness_heads.",
        "m1_pse.delta_heads.",
    )

    expected_core = [
        key
        for key in destination
        if key.startswith(core_prefixes)
    ]

    missing_core = [
        key
        for key in expected_core
        if key not in compatible
    ]

    if missing_core or shape_mismatch:
        preview = missing_core[:8] + [
            f"{key}: {src_shape} != {dst_shape}"
            for key, src_shape, dst_shape in shape_mismatch[:8]
        ]

        raise RuntimeError(
            "TIDE_M1_CHECKPOINT is not architecture-compatible with this V404 "
            "config. Use the exact config family that generated the validated "
            f"M1 checkpoint. Examples: {preview}"
        )

    result = model.load_state_dict(compatible, strict=False)

    unexpected = [
        key
        for key in result.unexpected_keys
        if key.startswith("m1_pse.")
    ]

    if unexpected:
        raise RuntimeError(
            f"Unexpected M1 keys while loading TIDE_M1_CHECKPOINT: {unexpected[:8]}"
        )

    logger.info(
        "[V404] Loaded verified pretrained M1 tensors=%d from: %s",
        len(compatible),
        path,
    )


def _load_safe_boundary_reference_checkpoint(model, path, logger):
    """Load exactly the validated A1 C9 branch, never A2 C10 tensors.

    This is intentionally narrow: the checkpoint must supply the original C9
    semantic projection/refiner/boundary head with matching shapes. New A3
    C10 modules remain freshly initialised and are the only trainable tensors.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "M1.SAFE_CONTEXT_REFERENCE_CHECKPOINT is missing: "
            + str(path)
        )

    state = _load_checkpoint_state(path)
    destination = model.state_dict()
    prefixes = (
        "m1_pse.safe_residual_semantic_proj.",
        "m1_pse.safe_residual_refiner.",
        "m1_pse.safe_boundary_residual_head.",
    )
    expected = [
        key for key in destination
        if key.startswith(prefixes)
    ]
    if not expected:
        raise RuntimeError(
            "Current model has no A1-compatible C9 safe-residual tensors."
        )

    missing = []
    mismatch = []
    for key in expected:
        if key not in state:
            missing.append(key)
        elif tuple(state[key].shape) != tuple(destination[key].shape):
            mismatch.append(
                (key, tuple(state[key].shape), tuple(destination[key].shape))
            )

    if missing or mismatch:
        raise RuntimeError(
            "A1 C9 reference checkpoint is incompatible with the current "
            "safe-residual architecture. missing="
            + str(missing[:8])
            + " mismatch="
            + str(mismatch[:4])
        )

    with torch.no_grad():
        for key in expected:
            destination[key].copy_(state[key])

    logger.info(
        "[SAFE_CONTEXT_CERTIFIED_ONLY] loaded frozen A1 C9 tensors=%d from: %s",
        len(expected),
        path,
    )

def _build_frozen_teacher(cfg, checkpoint_path, logger):
    """Build a strict B0-only frozen reference teacher.

    The Student remains V383-E2E trainable.  The Teacher must instead be a
    true B0 architecture with M1/M2 disabled, because the recovered checkpoint
    intentionally contains only original B0 tensors.
    """
    teacher_cfg = copy.deepcopy(cfg)
    m1 = teacher_cfg.M1

    # Disable M1 structurally and erase every historical V20-V383 dispatch
    # trigger.  Otherwise CustomCLIP may identify this teacher as V383 only
    # from RUN_TAG / M1_LOSS_VERSION despite M1.ENABLED=False.
    m1.ENABLED = False
    m1.TRAIN_MODE = "frozen"
    m1.INFERENCE_MODE = "preserve"
    m1.RUN_TAG = "B0_TEACHER_REFERENCE"
    m1.M1_LOSS_VERSION = "baseline_teacher"
    m1.M1_TRAIN_NUM_SAMPLES = 1
    m1.TEXT_VERIFIER_ENABLED = False

    disable_flags = (
        "V20_UNIFIED_ACTION_CF",
        "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID",
        "V24_TEXT_RANKED_STRUCTURAL_MEDOID",
        "V25_TYPE_CONDITIONAL_UTILITY_BANK",
        "V31_CANDIDATE_CONDITIONED_POLICY",
        "V32_ISLAND_PHASEB_POLICY",
        "V33_SIGNED_GAIN_REGRESSION",
        "V34_SPATIAL_QUANTILE_WORLD_MODEL",
        "V35_RESIDUAL_PURIFIED_WORLD_MODEL",
        "V36_CASEWISE_PLACKETT_LUCE",
        "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS",
        "V37_CONTEXT_CLEAN_CONTROLS",
        "V38_CASEWISE_FALSIFIED_DELTA_CONSENSUS",
        "V381_LESION_BACKGROUND_CALIBRATED_ATOMIC",
        "V382_ACTION_CONDITIONAL_QUANTILE_ATOMIC",
        "V383_CONSERVATIVE_ACTION_VALUE",
        "V383_E2E_REFERENCE_ANCHORED",
        "V383_REFERENCE_ANCHOR_ONLY",
        "V383_STUDENT_BASE_TRAINABLE",
    )
    for key in disable_flags:
        setattr(m1, key, False)

    teacher = build_model(teacher_cfg).to(cfg.MODEL.DEVICE)

    if bool(getattr(teacher, "m1_enabled", True)):
        raise RuntimeError("B0 teacher construction failed: M1 is still enabled.")
    if getattr(teacher, "m1_pse", None) is not None:
        raise RuntimeError("B0 teacher construction failed: m1_pse still exists.")

    active_policy_flags = [
        name for name in (
            "v383_conservative_action_value",
            "v382_action_conditional_quantile_atomic",
            "v381_lesion_background_calibrated_atomic",
            "v38_casewise_falsified_delta_consensus",
            "v37_text_falsified_structural_consensus",
        )
        if bool(getattr(teacher, name, False))
    ]
    if active_policy_flags:
        raise RuntimeError(
            "B0 teacher construction failed: V-series policy flags remain active: "
            f"{active_policy_flags}"
        )

    # Official B0 checkpoints may omit this HuggingFace persistent buffer.
    # It is regenerated by the model and must not invalidate strict B0 testing.
    teacher_result = teacher.load_state_dict(
        _load_checkpoint_state(checkpoint_path),
        strict=False,
    )

    allowed_missing_buffers = {
        "text_model.transformer.embeddings.position_ids",
    }

    illegal_missing = [
        key for key in teacher_result.missing_keys
        if key not in allowed_missing_buffers
    ]

    if illegal_missing or teacher_result.unexpected_keys:
        raise RuntimeError(
            "B0 teacher checkpoint is incompatible. "
            f"Missing non-buffer keys: {illegal_missing}; "
            f"unexpected keys: {teacher_result.unexpected_keys}"
        )

    if teacher_result.missing_keys:
        logger.info(
            "B0 teacher allowed missing buffers: %s",
            teacher_result.missing_keys,
        )

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    logger.info("Built strict frozen B0 Teacher from: %s", checkpoint_path)
    return teacher





def _is_v487_base_safe_e2e(cfg):
    """Whether the V487 Base/PVL split-backward contract is active.

    V519/V524 M2-only freezes Base/PVL/M1/M3 and trains only the regional
    pixel composer. Its Base objective is intentionally gradient-free, so it
    must never enter the V487 branch that calls base_objective.backward().
    """
    return bool(
        m1_enabled(cfg)
        and not _v519_is_m2_only(cfg)
        and bool(
            _cfg_get(
                _cfg_get(cfg, "M1", None),
                "V487_BASE_SAFE_E2E",
                False,
            )
        )
    )


def _v487_protected_base_prefixes():
    return (
        "pvl_adapters.",
        "mask_head.",
        "upscale.",
    )


def _v552r4204_base_fingerprint(model):
    """SHA256 fingerprint of the task-specific Base/PVL initialization.

    A same-seed ablation is not fair if an architectural branch changes random
    initialization of the shared Base.  Hash only the protected Base/PVL tensors
    (not M1/M2) so A0/A1/A2 can be checked before the first optimizer step.
    """
    prefixes = _v487_protected_base_prefixes()
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        clean_name = name[len("module."):] if name.startswith("module.") else name
        if not any(clean_name.startswith(prefix) for prefix in prefixes):
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(clean_name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        # ``parameter`` can legitimately be a scalar (0-D), e.g. PVL attention gates.
        # PyTorch forbids dtype-reinterpret ``view(torch.uint8)`` directly on a
        # 0-D tensor because the element size changes.  Flatten first so both
        # scalars and higher-rank tensors have a valid 1-D storage view while
        # preserving the exact raw parameter bits used by the fingerprint.
        raw_bytes = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
        count += 1
    if count <= 0:
        raise RuntimeError("V552-R4.20.4 Base fingerprint found no protected Base/PVL tensors")
    return digest.hexdigest(), count


def _ucfnrt_tensor_sha256(tensor):
    """Bitwise SHA256 for parity auditing without changing model state."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _ucfnrt_protected_grad_fingerprint(model):
    """SHA256 of Base/PVL gradients after the official Base backward."""
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        clean_name = name[len("module."):] if name.startswith("module.") else name
        if not any(clean_name.startswith(prefix) for prefix in _v487_protected_base_prefixes()):
            continue
        digest.update(clean_name.encode("utf-8"))
        grad = parameter.grad
        if grad is None:
            digest.update(b"<NONE>")
        else:
            value = grad.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    if count <= 0:
        raise RuntimeError("UC-FNRT parity audit found no protected Base/PVL tensors")
    return digest.hexdigest(), count


def _ucfnrt_assert_protected_grads_unchanged(model, before, logger=None):
    """Fail closed if the detached M1 backward changes any Base/PVL gradient."""
    leaks = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        old = before[name]
        new = parameter.grad
        if old is None and new is None:
            continue
        if old is None and new is not None:
            delta = float(new.detach().abs().max().cpu())
        elif old is not None and new is None:
            delta = float(old.detach().abs().max().cpu())
        else:
            delta = float((new.detach() - old.detach()).abs().max().cpu())
        if delta > 1.0e-10:
            leaks.append((name, delta))
    if leaks:
        message = "UC-FNRT detached M1 gradient leaked into Base/PVL: " + str(leaks[:8])
        if logger is not None:
            logger.error(message)
        raise RuntimeError(message)


def _v487_m1_prefixes():
    return (
        "v484_pipeline.",
    )


def _v487_snapshot_protected_grads(model):
    prefixes = _v487_protected_base_prefixes()
    saved = {}
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            saved[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return saved


def _v487_restore_protected_grads(model, saved):
    for name, parameter in model.named_parameters():
        if name in saved:
            if saved[name] is None:
                parameter.grad = None
            else:
                parameter.grad = saved[name].detach().clone()


def _jbt_v6_owns_m1_pse_grads(cfg):
    """True when the current JBT-v6 auxiliary optimizer physically owns m1_pse.*.

    V487 predates JBT-v6 and originally treated every ``m1_pse.*`` tensor as a
    legacy/forbidden branch.  Reusing that cleanup unchanged silently erased the
    complete JBT-v6 gradient immediately before ``optimizer.step()``.  In v6.1
    the Base remains protected by snapshot/restore, while the separate auxiliary
    Adam is explicitly allowed to keep the gradients it owns.
    """
    m1 = _cfg_get(cfg, "M1", None)
    return bool(
        m1_enabled(cfg)
        and bool(_cfg_get(m1, "V487_BASE_SAFE_E2E", False))
        and bool(_cfg_get(m1, "JBT_V6_SEPARATE_BASE_OPTIMIZER", False))
        and bool(_cfg_get(m1, "JBT_V6_DIRECT_SIGNED_FLOW", False))
    )


def _v487_clear_forbidden_m1_grads(model, cfg=None):
    # JBT-v6.1: m1_pse.* is the CURRENT method and is owned by the physically
    # separate JBT Adam.  The Base/PVL safety contract is already enforced by
    # _v487_assert_no_proposal_grad_leak + _v487_restore_protected_grads, so
    # clearing m1_pse here would disable learning rather than improve safety.
    if cfg is not None and _jbt_v6_owns_m1_pse_grads(cfg):
        return 0

    # Historical V487 compatibility: only V484/V486 was a valid repair branch.
    allowed = _v487_m1_prefixes()
    forbidden = (
        "m1_pse.",
        "ccv_m2.",
        "m2_text_verifier.",
        "m2_tide_repair_head.",
    )
    cleared = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in forbidden) and not any(name.startswith(prefix) for prefix in allowed):
            if parameter.grad is not None:
                cleared += 1
            parameter.grad = None
    return cleared


def _jbt_v6_real_train_grad_health(
    model, cfg, signed_owner_fraction=None,
):
    """Fail closed if JBT-v6 reaches optimizer.step() without real gradients.

    This guard runs on the *actual train.py path*, after V487 Base-gradient
    restoration/cleanup.  It closes the coverage hole where the old synthetic
    contract proved that the local loss graph had gradients even though the
    outer training loop erased them before the optimizer could consume them.
    """
    if not _jbt_v6_owns_m1_pse_grads(cfg):
        return None

    component_prefixes = {
        "flow": "m1_pse.mean_head.",
        "error": "m1_pse.error_head.",
        "feedback": "m1_pse.feature_feedback_head.",
        "utility": "m1_pse.v6_candidate_utility_head.",
    }
    total_sq = 0.0
    tensor_count = 0
    component_sq = {key: 0.0 for key in component_prefixes}
    component_count = {key: 0 for key in component_prefixes}

    for name, parameter in model.named_parameters():
        if not name.startswith("m1_pse.") or not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            continue
        value = grad.detach().float()
        sq = float(value.square().sum().cpu())
        total_sq += sq
        tensor_count += 1
        for key, prefix in component_prefixes.items():
            if name.startswith(prefix):
                component_sq[key] += sq
                component_count[key] += 1

    total_norm = math.sqrt(max(total_sq, 0.0))
    flow_norm = math.sqrt(max(component_sq["flow"], 0.0))
    if tensor_count <= 0 or not math.isfinite(total_norm) or total_norm <= 1.0e-12:
        raise RuntimeError(
            "JBT-v6.1 real-train gradient guard failed: m1_pse.* has no live "
            "gradient immediately before optimizer.step(). Check outer-loop "
            "gradient ownership/cleanup before changing model thresholds."
        )
    if bool(_cfg_get(_cfg_get(cfg, "M1", None), "JBT_V6_SIGNED_DISPLACEMENT_SUPERVISION", False)):
        flow_is_zero = bool(
            component_count["flow"] <= 0
            or not math.isfinite(flow_norm)
            or flow_norm <= 1.0e-12
        )
        # A complete physical batch can legitimately contain no normal-ray
        # displacement owners (empty GT / no reachable Base contour).  In that
        # case the signed-flow objective is exactly constant and its zero
        # gradient is correct.  Fail only when an owner exists but the graph is
        # nevertheless disconnected.  Legacy callers omit the owner statistic
        # and therefore retain the previous fail-closed behaviour.
        owner_is_known_empty = bool(
            signed_owner_fraction is not None
            and math.isfinite(float(signed_owner_fraction))
            and float(signed_owner_fraction) <= 0.0
        )
        if flow_is_zero and not owner_is_known_empty:
            raise RuntimeError(
                "JBT-v6.1 real-train gradient guard failed: signed-flow head "
                "has zero gradient although signed displacement supervision is enabled."
            )

    return {
        "tensor_count": float(tensor_count),
        "total_norm": float(total_norm),
        **{
            f"{key}_norm": float(math.sqrt(max(component_sq[key], 0.0)))
            for key in component_prefixes
        },
    }


def _v487_assert_no_proposal_grad_leak(model, before, logger=None):
    leaks = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        old = before[name]
        new = parameter.grad
        if old is None and new is None:
            continue
        if old is None and new is not None:
            val = float(new.detach().abs().max().cpu())
            if val > 1.0e-10:
                leaks.append((name, val))
        elif old is not None and new is None:
            val = float(old.detach().abs().max().cpu())
            if val > 1.0e-10:
                leaks.append((name, val))
        else:
            val = float((new.detach() - old.detach()).abs().max().cpu())
            if val > 1.0e-10:
                leaks.append((name, val))
    if leaks:
        message = "V487 proposal gradient leaked into Base/PVL/mask/upscale: " + str(leaks[:8])
        if logger is not None:
            logger.error(message)
        raise RuntimeError(message)

def _v488_keep_frozen_modules_eval(model, cfg):
    """Keep every frozen observer deterministic while trainable heads learn.

    CAUSAL56 freezes one common Base/PVL checkpoint for every M1 ablation.
    ``requires_grad=False`` alone is insufficient because ``model.train()``
    would still enable Dropout and update BatchNorm buffers in a frozen
    observer.  Reusing the established V488 mechanism makes both weights and
    forward state invariant across the ablation matrix.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    v532_enabled = bool(
        _clean_dynamic_component_set(cfg)
        or (m1_enabled(cfg) and bool(_cfg_get(m1_cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False)))
    )
    v531_enabled = bool(
        m1_enabled(cfg)
        and bool(_cfg_get(m1_cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False))
    )
    unified_refiner_enabled = v532_enabled or v531_enabled
    if not (
        _v488_is_m2m3_only(cfg)
        or _v519_is_m2_only(cfg)
        or unified_refiner_enabled
        or _geotr_m1_causal_ablation(cfg)
    ):
        return
    # model.train() is needed so the V488 pipeline uses train-time MC count and
    # straight-through M3 gating.  Immediately switch every subtree without a
    # trainable tensor back to eval, including parameter-free Dropout children.
    for module in model.modules():
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()
    pipeline = getattr(model, "v484_pipeline", None)
    if pipeline is not None:
        pipeline.train()
        if getattr(pipeline, "pixel_composer", None) is not None:
            pipeline.pixel_composer.train()
        if getattr(pipeline, "safe_deployer", None) is not None:
            pipeline.safe_deployer.train()
        # Re-assert deterministic frozen M1 after pipeline.train() recursively
        # toggles all children back to train mode.  V531 joint training keeps
        # the factual error head and atomic generator trainable.
        refiner_stage = str(
            _cfg_get(
                m1_cfg,
                "V532_TRAIN_STAGE" if v532_enabled else "V531_TRAIN_STAGE",
                "full" if v532_enabled else "router",
            )
        ).strip().lower()
        for name in (
            "error_state_head",
            "local_generator",
            "global_generator",
            "local_verifier",
            "rejector",
        ):
            module = getattr(pipeline, name, None)
            if module is None:
                continue
            v536_source_frozen = bool(
                v532_enabled
                and (
                    bool(_cfg_get(m1_cfg, "V536_PRIOR_ALIGNED_CASE_COMPONENT_ENABLED", False))
                    or bool(_cfg_get(m1_cfg, "V537_COMPONENT_UTILITY_RANKER_ENABLED", False))
                )
                and bool(
                    _cfg_get(
                        m1_cfg,
                        "V537_FREEZE_SOURCE_AT_RANKER",
                        _cfg_get(m1_cfg, "V536_FREEZE_SOURCE_AT_DEPLOY", True),
                    )
                )
                and int(getattr(pipeline, "current_epoch", 0))
                    >= _v536_deploy_start_epoch(cfg)
            )
            if (
                unified_refiner_enabled
                and refiner_stage in {"joint", "full"}
                and name in {"error_state_head", "local_generator"}
                and not v536_source_frozen
            ):
                module.train()
            else:
                module.eval()



def _v536_deploy_start_epoch(cfg):
    m1_cfg = _cfg_get(cfg, "M1", None)
    configured = None
    if bool(_cfg_get(m1_cfg, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False)):
        configured = _cfg_get(m1_cfg, "V538_DEPLOY_START_EPOCH", None)
    if configured is None:
        configured = _cfg_get(m1_cfg, "V537_RANKER_START_EPOCH", None)
    if configured is None:
        configured = _cfg_get(m1_cfg, "V536_DEPLOY_START_EPOCH", None)
    if configured is not None:
        return max(int(configured), 0)
    return max(
        int(_cfg_get(m1_cfg, "V532_M1_START_EPOCH", 0))
        + int(_cfg_get(m1_cfg, "V532_M1_RAMP_EPOCHS", 10)),
        0,
    )


def _v538_update_quality_gate(model, cfg, means, logger=None):
    """Persistently open deployment from independent epoch-level evidence.

    V552-R4.4 has a dedicated audit trace.  It is forced on one physically
    valid top-ranked candidate per training case and never becomes the model
    output.  This removes the historical circular dependency in which the
    quality gate required successful Shadow executions while Shadow itself was
    filtered by the unopened quality gate.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1_cfg, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False)):
        return False
    patience = max(int(_cfg_get(m1_cfg, "V538_M2_READY_PATIENCE", 3)), 1)
    core = model.module if hasattr(model, "module") else model
    pipeline = getattr(core, "v484_pipeline", None)
    refiner = getattr(pipeline, "pixel_composer", None)
    if refiner is None or not hasattr(refiner, "update_v538_quality_ready"):
        raise RuntimeError(
            "V538 is enabled but v484_pipeline.pixel_composer does not expose "
            "update_v538_quality_ready()."
        )

    r44_enabled = bool(
        _cfg_get(m1_cfg, "V552R44_AUDIT_GATE_ROOTFIX_ENABLED", False)
        and _cfg_get(
            m1_cfg, "V552R44_ACTION_AWARE_QUALITY_GATE_ENABLED", False
        )
    )
    r45_enabled = bool(
        _cfg_get(m1_cfg, "V552R45_ROOTFIX_ENABLED", False)
        and _cfg_get(m1_cfg, "V552R45_DIRECT_AUDIT_GATE_ENABLED", True)
    )
    r46_enabled = bool(
        _cfg_get(m1_cfg, "V552R46_ROOTFIX_ENABLED", False)
        and _cfg_get(m1_cfg, "V552R46_FORMAL_POLICY_AUDIT_ENABLED", True)
    )
    r47_enabled = bool(
        _cfg_get(m1_cfg, "V552R47_ROOTFIX_ENABLED", False)
        and _cfg_get(m1_cfg, "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED", False)
    )
    r48_enabled = bool(
        _cfg_get(m1_cfg, "V552R48_ROOTFIX_ENABLED", False)
        and _cfg_get(m1_cfg, "V552R48_ITERATIVE_BINDING_ENABLED", False)
    )
    r49_enabled = bool(
        _cfg_get(m1_cfg, "V552R49_ROOTFIX_ENABLED", False)
        and _cfg_get(m1_cfg, "V552R49_CONTENT_SELECTIVE_ATTENTION_ENABLED", False)
    )
    r410_enabled = bool(
        _cfg_get(m1_cfg, "V552R410_ROOTFIX_ENABLED", False)
    )
    r411_enabled = bool(
        _cfg_get(m1_cfg, "V552R411_ROOTFIX_ENABLED", False)
    )
    r412_enabled = bool(
        _cfg_get(m1_cfg, "V552R412_ROOTFIX_ENABLED", False)
    )
    r413_enabled = bool(
        _cfg_get(m1_cfg, "V552R413_ROOTFIX_ENABLED", False)
    )
    r414_enabled = bool(
        _cfg_get(m1_cfg, "V552R414_ROOTFIX_ENABLED", False)
    )
    r415_enabled = bool(
        _cfg_get(m1_cfg, "V552R415_ROOTFIX_ENABLED", False)
    )
    r416_enabled = bool(
        _cfg_get(m1_cfg, "V552R416_ROOTFIX_ENABLED", False)
    )
    r417_enabled = bool(
        _cfg_get(m1_cfg, "V552R417_ROOTFIX_ENABLED", False)
    )
    r418_enabled = bool(
        _cfg_get(m1_cfg, "V552R418_ROOTFIX_ENABLED", False)
    )
    r418_paired_enabled_cfg = r418_enabled and bool(
        _cfg_get(m1_cfg, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", True)
    )
    r419_enabled = r418_enabled and bool(
        _cfg_get(m1_cfg, "V552R419_ROOTFIX_ENABLED", False)
    )
    r420_enabled = r418_enabled and bool(
        _cfg_get(m1_cfg, "V552R420_ROOTFIX_ENABLED", False)
    )
    if r44_enabled:
        candidate_executed = float(
            means.get("v552r44_audit_execute_count_epoch", 0.0)
        )
        candidate_improved = float(
            means.get("v552r44_audit_improved_count_epoch", 0.0)
        )
        candidate_harmful = float(
            means.get("v552r44_audit_harmful_count_epoch", 0.0)
        )
        candidate_gain_sum = float(
            means.get("v552r44_audit_selected_gain_sum_epoch", 0.0)
        )
        candidate_case_count = float(
            means.get("v552r44_audit_case_count_epoch", 0.0)
        )
        policy_executed = float(
            means.get("v552r46_formal_policy_audit_execute_count_epoch" if r46_enabled else "v552r44_policy_audit_execute_count_epoch", 0.0)
        )
        policy_improved = float(
            means.get("v552r46_formal_policy_audit_improved_count_epoch" if r46_enabled else "v552r44_policy_audit_improved_count_epoch", 0.0)
        )
        policy_harmful = float(
            means.get("v552r46_formal_policy_audit_harmful_count_epoch" if r46_enabled else "v552r44_policy_audit_harmful_count_epoch", 0.0)
        )
        policy_gain_sum = float(
            means.get("v552r46_formal_policy_audit_selected_gain_sum_epoch" if r46_enabled else "v552r44_policy_audit_selected_gain_sum_epoch", 0.0)
        )
        policy_case_count = float(
            means.get("v552r46_formal_policy_audit_case_count_epoch" if r46_enabled else "v552r44_policy_audit_case_count_epoch", 0.0)
        )
        min_candidate_executed = max(
            int(_cfg_get(m1_cfg, "V552R44_QUALITY_MIN_AUDITED_CASES", 16)), 1
        )
        min_policy_executed = max(
            int(_cfg_get(m1_cfg, "V552R44_QUALITY_MIN_POLICY_CASES", 16)), 1
        )
        evidence_ready = bool(
            candidate_executed >= float(min_candidate_executed)
            and policy_executed >= float(min_policy_executed)
        )
        candidate_precision = candidate_improved / max(candidate_executed, 1.0)
        candidate_harm_rate = candidate_harmful / max(candidate_executed, 1.0)
        candidate_mean_gain = candidate_gain_sum / max(candidate_executed, 1.0)
        candidate_coverage = candidate_executed / max(candidate_case_count, 1.0)
        precision = policy_improved / max(policy_executed, 1.0)
        harm_rate = policy_harmful / max(policy_executed, 1.0)
        mean_gain = policy_gain_sum / max(policy_executed, 1.0)
        coverage = policy_executed / max(policy_case_count, 1.0)

        outcome_balanced_accuracy = float(
            means.get("v552_outcome_balanced_accuracy", 0.0)
        )
        harm_recall = float(means.get("v552_harm_recall", 0.0))
        benefit_recall = float(means.get("v552_benefit_recall", 0.0))
        if r46_enabled:
            safety_benefit_recall = float(
                means.get("v552r4_editor_safety_benefit_recall", 0.0)
            )
            safety_harm_recall = float(
                means.get("v552r4_editor_safety_harm_recall", 0.0)
            )
            signed_gain_benefit = float(
                means.get("v552r45_utility_signed_gain_mean_benefit", 0.0)
            )
            signed_gain_harm = float(
                means.get("v552r45_utility_signed_gain_mean_harm", 0.0)
            )
            selector_ready = bool(
                evidence_ready
                and harm_rate <= float(_cfg_get(
                    m1_cfg, "V552R46_QUALITY_MAX_POLICY_HARM_RATE", 0.15
                ))
                and mean_gain > float(_cfg_get(
                    m1_cfg, "V552R46_QUALITY_MIN_POLICY_GAIN", 0.0
                ))
                and coverage >= float(_cfg_get(
                    m1_cfg, "V552R46_QUALITY_MIN_POLICY_COVERAGE", 0.05
                ))
                and safety_benefit_recall >= float(_cfg_get(
                    m1_cfg, "V552R46_MIN_SAFETY_BENEFIT_RECALL", 0.35
                ))
                and safety_harm_recall >= float(_cfg_get(
                    m1_cfg, "V552R46_MIN_SAFETY_HARM_RECALL", 0.55
                ))
                and signed_gain_benefit > 0.0
                and signed_gain_harm < 0.0
                and policy_improved > policy_harmful
            )
        elif r45_enabled:
            safety_benefit_recall = float(
                means.get("v552r4_editor_safety_benefit_recall", 0.0)
            )
            safety_harm_recall = float(
                means.get("v552r4_editor_safety_harm_recall", 0.0)
            )
            signed_gain_benefit = float(
                means.get("v552r45_utility_signed_gain_mean_benefit", 0.0)
            )
            signed_gain_harm = float(
                means.get("v552r45_utility_signed_gain_mean_harm", 0.0)
            )
            selector_ready = bool(
                evidence_ready
                and harm_rate <= float(
                    _cfg_get(m1_cfg, "V552R45_QUALITY_MAX_POLICY_HARM_RATE", 0.15)
                )
                and mean_gain > float(
                    _cfg_get(m1_cfg, "V552R45_QUALITY_MIN_POLICY_GAIN", 0.0)
                )
                and coverage >= float(
                    _cfg_get(m1_cfg, "V552R45_QUALITY_MIN_POLICY_COVERAGE", 0.05)
                )
                and safety_benefit_recall >= float(
                    _cfg_get(m1_cfg, "V552R45_MIN_SAFETY_BENEFIT_RECALL", 0.35)
                )
                and safety_harm_recall >= float(
                    _cfg_get(m1_cfg, "V552R45_MIN_SAFETY_HARM_RECALL", 0.55)
                )
                and outcome_balanced_accuracy >= float(
                    _cfg_get(m1_cfg, "V552R45_MIN_UTILITY_BALANCED_ACCURACY", 0.55)
                )
                and harm_recall >= float(
                    _cfg_get(m1_cfg, "V552R45_MIN_UTILITY_HARM_RECALL", 0.50)
                )
                and benefit_recall >= float(
                    _cfg_get(m1_cfg, "V552R45_MIN_UTILITY_BENEFIT_RECALL", 0.50)
                )
                and signed_gain_benefit > 0.0
                and signed_gain_harm < 0.0
                and policy_improved > policy_harmful
            )
        else:
            safety_benefit_recall = 0.0
            safety_harm_recall = 0.0
            signed_gain_benefit = 0.0
            signed_gain_harm = 0.0
            selector_ready = bool(
                evidence_ready
                and precision >= float(
                    _cfg_get(m1_cfg, "V552R44_QUALITY_MIN_AUDIT_PRECISION", 0.55)
                )
                and harm_rate <= float(
                    _cfg_get(m1_cfg, "V552R44_QUALITY_MAX_AUDIT_HARM_RATE", 0.15)
                )
                and mean_gain > float(
                    _cfg_get(m1_cfg, "V552R44_QUALITY_MIN_AUDIT_GAIN", 0.0)
                )
                and coverage >= float(
                    _cfg_get(m1_cfg, "V552R44_QUALITY_MIN_AUDIT_COVERAGE", 0.05)
                )
                and outcome_balanced_accuracy >= float(
                    _cfg_get(m1_cfg, "V552R44_MIN_UTILITY_BALANCED_ACCURACY", 0.55)
                )
                and harm_recall >= float(
                    _cfg_get(m1_cfg, "V552R44_MIN_UTILITY_HARM_RECALL", 0.45)
                )
                and benefit_recall >= float(
                    _cfg_get(m1_cfg, "V552R44_MIN_UTILITY_BENEFIT_RECALL", 0.45)
                )
                and policy_improved > policy_harmful
            )

        current_epoch = int(means.get("v549_current_epoch", 0.0))
        action_precision = float(
            means.get("v552r43_atom_correction_precision", 0.0)
        )
        action_capture = float(
            means.get("v552r43_atom_corrected_pixel_capture", 0.0)
        )
        useful_atoms = float(means.get("v552r3_useful_atom_count", 0.0))
        predicted_active = float(
            means.get("v552r3_predicted_active_atom_count", 0.0)
        )
        m1_oracle = float(means.get("v551_m1_atom_oracle_gain", 0.0))
        editor_oracle = float(means.get("v551_editor_atom_oracle_gain", 0.0))
        component_oracle = float(means.get("v538_component_oracle_gain", 0.0))
        teacher_oracle = float(
            means.get("v538_action_realizable_teacher_oracle_gain", 0.0)
        )
        teacher_realization = float(
            means.get("v552r46_teacher_realization_ratio", 0.0)
        )
        r47_mask_purity = float(means.get("v544_mask_soft_purity", 0.0))
        r47_component_capture = float(means.get("v538_component_capture_ratio", 0.0))
        r49_dn_reconstruction_dice = float(
            means.get("v552r48_dn_reconstruction_dice", 0.0)
        )
        r49_attention_entropy_ratio = float(
            means.get("v552r49_attention_entropy_ratio", 1.0)
        )
        r49_attention_max_weight = float(
            means.get("v552r49_attention_max_weight", 0.0)
        )
        r412_box_iou = float(means.get("v552r412_box_iou", 0.0))
        r412_oracle_box_dice = float(
            means.get("v552r412_oracle_box_canonical_dice", 0.0)
        )
        r412_native_box_dice = float(
            means.get("v552r412_native_box_canonical_dice", 0.0)
        )
        r413_proposal_box_iou = float(
            means.get("v552r413_proposal_box_iou", 0.0)
        )
        r413_box_drift_l1 = float(
            means.get("v552r413_box_drift_l1", 1.0)
        )
        r415_identity_match_rate = float(
            means.get("v552r415_identity_match_rate", 0.0)
        )
        r415_center_error_p90 = float(
            means.get("v552r415_center_error_px_p90", 1.0e9)
        )
        r416_pre_topk_peak_recall = float(
            means.get("v552r416_pre_topk_peak_recall", 0.0)
        )
        r416_edge_offset_mae_px = float(
            means.get("v552r416_edge_offset_mae_px", 1.0e9)
        )
        r416_ltrb_enabled = bool(_cfg_get(
            m1_cfg, "V552R416_ASYMMETRIC_LTRB_ENABLED", False
        ))
        r417_pre_topk_peak_recall = float(
            means.get("v552r417_pre_topk_peak_recall", 0.0)
        )
        r417_selected_spatial_coverage = float(
            means.get("v552r417_selected_spatial_coverage", 0.0)
        )
        r417_location_offset_mae_px = float(
            means.get("v552r417_location_offset_mae_px", 1.0e9)
        )
        r417_shared_offset_enabled = bool(_cfg_get(
            m1_cfg, "V552R417_SHARED_OFFSET_ENABLED", True
        ))
        r418_paired_mask_dice = float(means.get("v552r418_paired_mask_dice", 0.0))
        r418_paired_target_consistency = float(means.get("v552r418_paired_target_consistency", 0.0))
        r418_native_mask_matching_dice = float(means.get("v552r418_native_mask_matching_dice", 0.0))
        r419_seed_support_fraction = float(means.get("v552r419_seed_support_fraction", 0.0))
        r419_final_support_fraction = float(means.get("v552r419_final_support_fraction", 0.0))
        r419_outside_mask_probability = float(means.get("v552r419_outside_mask_probability", 1.0))
        r419_native_mask_soft_purity = float(means.get("v552r419_native_mask_soft_purity", 0.0))
        r420_native_mask_matching_dice = float(means.get("v552r420_native_mask_matching_dice", 0.0))
        r420_native_mask_soft_purity = float(means.get("v552r420_native_mask_soft_purity", 0.0))
        r420_native_mask_soft_coverage = float(means.get("v552r420_native_mask_soft_coverage", 0.0))
        r420_relative_coord_mean_abs = float(means.get("v552r420_relative_coord_mean_abs", 0.0))
        r412_oracle_retention = (
            component_oracle / teacher_oracle
            if teacher_oracle > 1.0e-12 else 0.0
        )
        if r46_enabled:
            candidate_ready = bool(
                current_epoch >= int(_cfg_get(
                    m1_cfg, "V552R44_QUALITY_START_EPOCH", 12
                ))
                and candidate_executed >= float(min_candidate_executed)
                and candidate_mean_gain > float(_cfg_get(
                    m1_cfg, "V552R46_MIN_CANDIDATE_AUDIT_GAIN", 0.0
                ))
                and candidate_harm_rate <= float(_cfg_get(
                    m1_cfg, "V552R46_MAX_CANDIDATE_AUDIT_HARM", 0.20
                ))
                and m1_oracle >= float(_cfg_get(
                    m1_cfg, "V552R44_MIN_M1_ORACLE_GAIN", 0.003
                ))
                and editor_oracle >= m1_oracle - float(_cfg_get(
                    m1_cfg, "V552R44_EDITOR_ORACLE_TOLERANCE", 1.0e-4
                ))
                and component_oracle >= float(_cfg_get(
                    m1_cfg, "V552R44_MIN_COMPONENT_ORACLE_GAIN", 0.002
                ))
                and teacher_oracle > 0.0
                and teacher_realization > 0.0
                and (
                    (not r47_enabled)
                    or (
                        teacher_realization >= float(_cfg_get(
                            m1_cfg, "V552R47_MIN_TEACHER_REALIZATION_RATIO", 0.15
                        ))
                        and r47_mask_purity >= float(_cfg_get(
                            m1_cfg, "V552R47_MIN_MASK_SOFT_PURITY", 0.12
                        ))
                        and r47_component_capture >= float(_cfg_get(
                            m1_cfg, "V552R47_MIN_COMPONENT_CAPTURE", 0.15
                        ))
                    )
                )
                and (
                    (not r412_enabled)
                    or (
                        r412_box_iou >= float(_cfg_get(
                            m1_cfg, "V552R412_MIN_BOX_IOU", 0.45
                        ))
                        and r412_oracle_box_dice >= float(_cfg_get(
                            m1_cfg, "V552R412_MIN_ORACLE_BOX_DICE", 0.60
                        ))
                        and r412_native_box_dice >= float(_cfg_get(
                            m1_cfg, "V552R412_MIN_NATIVE_BOX_DICE", 0.40
                        ))
                        and r412_oracle_retention >= float(_cfg_get(
                            m1_cfg, "V552R412_MIN_ORACLE_RETENTION", 0.20
                        ))
                    )
                )
                and (
                    (not r413_enabled)
                    or (
                        r413_proposal_box_iou >= float(_cfg_get(
                            m1_cfg, "V552R413_MIN_PROPOSAL_BOX_IOU", 0.45
                        ))
                        and r413_box_drift_l1 <= float(_cfg_get(
                            m1_cfg, "V552R413_MAX_BOX_DRIFT_L1", 1.0e-4
                        ))
                    )
                )
                and (
                    (not r415_enabled)
                    or (
                        r415_identity_match_rate >= float(_cfg_get(
                            m1_cfg, "V552R415_MIN_IDENTITY_MATCH_RATE", 0.50
                        ))
                        and r415_center_error_p90 <= float(_cfg_get(
                            m1_cfg, "V552R415_MAX_MATCHED_CENTER_ERROR_P90_PX", 10.0
                        ))
                    )
                )
                and (
                    (not r416_enabled)
                    or (
                        r416_pre_topk_peak_recall >= float(_cfg_get(
                            m1_cfg, "V552R416_MIN_PRE_TOPK_PEAK_RECALL", 0.80
                        ))
                        and (
                            (not r416_ltrb_enabled)
                            or r416_edge_offset_mae_px <= float(_cfg_get(
                                m1_cfg, "V552R416_MAX_EDGE_OFFSET_MAE_PX", 12.0
                            ))
                        )
                    )
                )
                and (
                    (not r417_enabled)
                    or r418_enabled
                    or (
                        r417_pre_topk_peak_recall >= float(_cfg_get(
                            m1_cfg, "V552R417_MIN_PRE_TOPK_PEAK_RECALL", 0.75
                        ))
                        and r417_selected_spatial_coverage >= float(_cfg_get(
                            m1_cfg, "V552R417_MIN_SELECTED_SPATIAL_COVERAGE", 0.60
                        ))
                        and (
                            (not r417_shared_offset_enabled)
                            or r417_location_offset_mae_px <= float(_cfg_get(
                                m1_cfg, "V552R417_MAX_LOCATION_OFFSET_MAE_PX", 6.0
                            ))
                        )
                    )
                )
                and (
                    (not r418_enabled)
                    or (
                        (
                            (not r418_paired_enabled_cfg)
                            or (
                                r418_paired_target_consistency >= 0.999
                                and r418_paired_mask_dice >= float(_cfg_get(
                                    m1_cfg, "V552R418_MIN_PAIRED_MASK_DICE", 0.50
                                ))
                            )
                        )
                        and r418_native_mask_matching_dice >= float(_cfg_get(
                            m1_cfg, "V552R418_MIN_NATIVE_MATCH_DICE", 0.20
                        ))
                    )
                )
                and (
                    (not r49_enabled)
                    or r411_enabled
                    or (
                        r49_dn_reconstruction_dice >= float(_cfg_get(
                            m1_cfg, "V552R49_MIN_DN_RECONSTRUCTION_DICE", 0.50
                        ))
                        and r49_attention_entropy_ratio <= float(_cfg_get(
                            m1_cfg, "V552R49_MAX_ATTENTION_ENTROPY_RATIO", 0.95
                        ))
                        and r49_attention_max_weight >= float(_cfg_get(
                            m1_cfg, "V552R49_MIN_ATTENTION_MAX_WEIGHT", 0.16
                        ))
                    )
                )
            )
        else:
            candidate_ready = bool(
            current_epoch >= int(
                _cfg_get(m1_cfg, "V552R44_QUALITY_START_EPOCH", 12)
            )
            and candidate_executed >= float(min_candidate_executed)
            and candidate_mean_gain > float(
                _cfg_get(m1_cfg, "V552R45_MIN_CANDIDATE_AUDIT_GAIN", 0.0) if r45_enabled else _cfg_get(m1_cfg, "V552R44_MIN_CANDIDATE_AUDIT_GAIN", 0.0)
            )
            and candidate_harm_rate <= float(
                _cfg_get(m1_cfg, "V552R45_MAX_CANDIDATE_AUDIT_HARM", 0.20) if r45_enabled else _cfg_get(m1_cfg, "V552R44_MAX_CANDIDATE_AUDIT_HARM", 0.25)
            )
            and action_precision >= float(
                _cfg_get(
                    m1_cfg,
                    "V552R45_MIN_ACTION_CORRECTION_PRECISION" if r45_enabled else "V552R44_MIN_ACTION_CORRECTION_PRECISION",
                    0.65 if r45_enabled else 0.55,
                )
            )
            and (
                r45_enabled
                or action_capture >= float(
                    _cfg_get(
                        m1_cfg,
                        "V552R44_MIN_ACTION_CORRECTED_CAPTURE",
                        0.20,
                    )
                )
            )
            and useful_atoms >= float(
                _cfg_get(m1_cfg, "V552R44_MIN_USEFUL_ATOMS", 0.75)
            )
            and predicted_active >= float(
                _cfg_get(m1_cfg, "V552R44_MIN_PREDICTED_ACTIVE_ATOMS", 0.50)
            )
            and predicted_active <= float(
                _cfg_get(m1_cfg, "V552R44_MAX_PREDICTED_ACTIVE_ATOMS", 3.50)
            )
            and m1_oracle >= float(
                _cfg_get(m1_cfg, "V552R44_MIN_M1_ORACLE_GAIN", 0.003)
            )
            and editor_oracle >= m1_oracle - float(
                _cfg_get(m1_cfg, "V552R44_EDITOR_ORACLE_TOLERANCE", 1.0e-4)
            )
            and component_oracle >= float(
                _cfg_get(m1_cfg, "V552R44_MIN_COMPONENT_ORACLE_GAIN", 0.002)
            )
        )
        epoch_ready = selector_ready and candidate_ready
        close_patience = max(
            int(_cfg_get(m1_cfg, "V552_QUALITY_CLOSE_PATIENCE", 2)), 1
        )
        opened = bool(
            refiner.update_v538_quality_ready(
                epoch_ready,
                patience=patience,
                close_patience=close_patience,
            )
        )
        means.update({
            "v552r44_quality_evidence_ready": 1.0 if evidence_ready else 0.0,
            "v552r44_quality_candidate_audit_precision": candidate_precision,
            "v552r44_quality_candidate_audit_harm_rate": candidate_harm_rate,
            "v552r44_quality_candidate_audit_mean_gain": candidate_mean_gain,
            "v552r44_quality_candidate_audit_coverage": candidate_coverage,
            "v552r44_quality_audit_precision": precision,
            "v552r44_quality_audit_harm_rate": harm_rate,
            "v552r44_quality_audit_mean_gain": mean_gain,
            "v552r44_quality_audit_coverage": coverage,
            "v552r44_quality_selector_ready": 1.0 if selector_ready else 0.0,
            "v552r44_quality_candidate_ready": 1.0 if candidate_ready else 0.0,
            "v552r44_action_precision_gate": action_precision,
            "v552r44_action_capture_gate": action_capture,
            "v552r45_quality_selector_ready": 1.0 if (r45_enabled and selector_ready) else 0.0,
            "v552r45_quality_candidate_ready": 1.0 if (r45_enabled and candidate_ready) else 0.0,
            "v552r45_quality_safety_benefit_recall": safety_benefit_recall,
            "v552r45_quality_safety_harm_recall": safety_harm_recall,
            "v552r45_quality_signed_gain_benefit": signed_gain_benefit,
            "v552r45_quality_signed_gain_harm": signed_gain_harm,
            "v552r45_quality_action_capture_readiness": action_capture,
            "v552r46_quality_selector_ready": 1.0 if (r46_enabled and selector_ready) else 0.0,
            "v552r46_quality_candidate_ready": 1.0 if (r46_enabled and candidate_ready) else 0.0,
            "v552r46_quality_formal_policy_harm_rate": harm_rate if r46_enabled else 0.0,
            "v552r46_quality_formal_policy_gain": mean_gain if r46_enabled else 0.0,
            "v552r46_quality_formal_policy_coverage": coverage if r46_enabled else 0.0,
            "v552r46_quality_teacher_oracle": teacher_oracle if r46_enabled else 0.0,
            "v552r46_quality_teacher_realization_ratio": teacher_realization if r46_enabled else 0.0,
            "v552r47_quality_mask_purity": r47_mask_purity if r47_enabled else 0.0,
            "v552r47_quality_component_capture": r47_component_capture if r47_enabled else 0.0,
            "v552r47_quality_teacher_realization_ratio": teacher_realization if r47_enabled else 0.0,
            "v552r49_quality_dn_reconstruction_dice": r49_dn_reconstruction_dice if r49_enabled else 0.0,
            "v552r49_quality_attention_entropy_ratio": r49_attention_entropy_ratio if r49_enabled else 0.0,
            "v552r49_quality_attention_max_weight": r49_attention_max_weight if r49_enabled else 0.0,
            "v552r412_quality_box_iou": r412_box_iou if r412_enabled else 0.0,
            "v552r412_quality_oracle_box_dice": r412_oracle_box_dice if r412_enabled else 0.0,
            "v552r412_quality_native_box_dice": r412_native_box_dice if r412_enabled else 0.0,
            "v552r412_quality_oracle_retention": r412_oracle_retention if r412_enabled else 0.0,
            "v552r413_quality_proposal_box_iou": r413_proposal_box_iou if r413_enabled else 0.0,
            "v552r413_quality_box_drift_l1": r413_box_drift_l1 if r413_enabled else 0.0,
            "v552r415_quality_identity_match_rate": r415_identity_match_rate if r415_enabled else 0.0,
            "v552r415_quality_center_error_p90_px": r415_center_error_p90 if r415_enabled else 0.0,
            "v552r416_quality_pre_topk_peak_recall": r416_pre_topk_peak_recall if r416_enabled else 0.0,
            "v552r416_quality_edge_offset_mae_px": r416_edge_offset_mae_px if (r416_enabled and r416_ltrb_enabled) else 0.0,
            "v552r417_quality_pre_topk_peak_recall": r417_pre_topk_peak_recall if r417_enabled else 0.0,
            "v552r417_quality_selected_spatial_coverage": r417_selected_spatial_coverage if r417_enabled else 0.0,
            "v552r417_quality_location_offset_mae_px": r417_location_offset_mae_px if (r417_enabled and r417_shared_offset_enabled) else 0.0,
            "v552r418_quality_paired_mask_dice": r418_paired_mask_dice if r418_enabled else 0.0,
            "v552r418_quality_paired_target_consistency": r418_paired_target_consistency if r418_enabled else 0.0,
            "v552r418_quality_native_mask_matching_dice": r418_native_mask_matching_dice if r418_enabled else 0.0,
            "v552r419_quality_seed_support_fraction": r419_seed_support_fraction if r419_enabled else 0.0,
            "v552r419_quality_final_support_fraction": r419_final_support_fraction if r419_enabled else 0.0,
            "v552r419_quality_outside_mask_probability": r419_outside_mask_probability if r419_enabled else 0.0,
            "v552r419_quality_native_mask_soft_purity": r419_native_mask_soft_purity if r419_enabled else 0.0,
            "v552r420_quality_native_mask_matching_dice": r420_native_mask_matching_dice if r420_enabled else 0.0,
            "v552r420_quality_native_mask_soft_purity": r420_native_mask_soft_purity if r420_enabled else 0.0,
            "v552r420_quality_native_mask_soft_coverage": r420_native_mask_soft_coverage if r420_enabled else 0.0,
            "v552r420_quality_relative_coord_mean_abs": r420_relative_coord_mean_abs if r420_enabled else 0.0,
            "v538_quality_ready_epoch_fraction": 1.0 if epoch_ready else 0.0,
            "v538_quality_ready": 1.0 if opened else 0.0,
            "v538_quality_ready_streak": float(
                refiner.v538_quality_ready_streak.item()
            ),
            "v552_quality_bad_streak": float(
                refiner.v552_quality_bad_streak.item()
            ),
            "v552_outcome_balanced_accuracy_gate": outcome_balanced_accuracy,
            "v552_harm_sign_recall_gate": harm_recall,
            "v552_benefit_recall_gate": benefit_recall,
        })
        if logger is not None:
            gate_label = (
                "[V552R420_QUALITY_GATE] " if r420_enabled else
                "[V552R419_QUALITY_GATE] " if r419_enabled else
                "[V552R418_QUALITY_GATE] " if r418_enabled else
                "[V552R417_QUALITY_GATE] " if r417_enabled else
                "[V552R416_QUALITY_GATE] " if r416_enabled else
                "[V552R415_QUALITY_GATE] " if r415_enabled else
                "[V552R414_QUALITY_GATE] " if r414_enabled else
                "[V552R413_QUALITY_GATE] " if r413_enabled else
                "[V552R412_QUALITY_GATE] " if r412_enabled else
                "[V552R411_QUALITY_GATE] " if r411_enabled else
                "[V552R410_QUALITY_GATE] " if r410_enabled else
                "[V552R49_QUALITY_GATE] " if r49_enabled else
                "[V552R48_QUALITY_GATE] " if r48_enabled else
                "[V552R47_QUALITY_GATE] " if r47_enabled else
                "[V552R46_QUALITY_GATE] " if r46_enabled else
                "[V552R45_QUALITY_GATE] " if r45_enabled else
                "[V552R44_QUALITY_GATE] "
            )
            logger.info(
                gate_label
                + "candidate=%.0f/%.0f policy=%.0f/%.0f "
                "min=%d/%d candidate_gain=%.6f candidate_harm=%.4f "
                "precision=%.4f harm=%.4f gain=%.6f coverage=%.4f "
                "action_precision=%.4f action_capture=%.4f "
                "candidate_ready=%s selector_ready=%s streak=%d/%d opened=%s",
                candidate_executed,
                candidate_case_count,
                policy_executed,
                policy_case_count,
                min_candidate_executed,
                min_policy_executed,
                candidate_mean_gain,
                candidate_harm_rate,
                precision,
                harm_rate,
                mean_gain,
                coverage,
                action_precision,
                action_capture,
                candidate_ready,
                selector_ready,
                int(refiner.v538_quality_ready_streak.item()),
                patience,
                opened,
            )
        return opened

    epoch_gate = bool(_cfg_get(m1_cfg, "V549_EPOCH_QUALITY_GATE_ENABLED", False))
    if epoch_gate:
        executed = float(means.get("v549_shadow_execute_count_epoch", 0.0))
        improved = float(means.get("v549_shadow_improved_count_epoch", 0.0))
        harmful = float(means.get("v549_shadow_harmful_count_epoch", 0.0))
        gain_sum = float(means.get("v549_shadow_selected_gain_sum_epoch", 0.0))
        min_executed = max(
            int(_cfg_get(m1_cfg, "V549_QUALITY_MIN_EXECUTED_CASES", 8)), 1
        )
        evidence_ready = executed >= float(min_executed)
        precision = improved / max(executed, 1.0)
        harm_rate = harmful / max(executed, 1.0)
        mean_gain = gain_sum / max(executed, 1.0)
        r2_enabled = bool(_cfg_get(m1_cfg, "V552R2_TEACHER_DECOUPLED_ENABLED", False))
        outcome_balanced_accuracy = float(
            means.get(
                "v552_outcome_balanced_accuracy",
                means.get("v544_outcome_balanced_accuracy_global", 0.0),
            )
        )
        harm_sign_recall = float(
            means.get(
                "v552_harm_recall",
                means.get("v544_harm_gain_negative_rate_global", 0.0),
            )
        )
        benefit_recall = float(means.get("v552_benefit_recall", 0.0))
        selector_ready = bool(
            evidence_ready
            and precision >= float(
                _cfg_get(m1_cfg, "V541_QUALITY_MIN_EXECUTE_PRECISION", 0.50)
            )
            and harm_rate <= float(
                _cfg_get(m1_cfg, "V541_QUALITY_MAX_HARMFUL_CASE_RATE", 0.10)
            )
            and mean_gain > float(
                _cfg_get(m1_cfg, "V541_QUALITY_MIN_COMPOSER_GAIN", 0.0)
            )
            and outcome_balanced_accuracy >= float(
                _cfg_get(m1_cfg, "V552_MIN_OUTCOME_BALANCED_ACCURACY", 0.60)
            )
            and harm_sign_recall >= float(
                _cfg_get(m1_cfg, "V552_MIN_HARM_SIGN_RECALL", 0.50)
            )
            and (
                (not r2_enabled)
                or benefit_recall >= float(
                    _cfg_get(m1_cfg, "V552_QUALITY_MIN_BENEFIT_RECALL", 0.50)
                )
            )
            and policy_improved > policy_harmful
        )
        teacher_gain = float(means.get("v538_teacher_component_oracle_gain", 0.0))
        component_gain = float(means.get("v538_component_oracle_gain", 0.0))
        teacher_floor = max(float(
            _cfg_get(m1_cfg, "V540_M2_MIN_TEACHER_ORACLE_GAIN", 0.0)
        ), 0.0)
        absolute_floor = max(float(
            _cfg_get(
                m1_cfg,
                "V540_M2_MIN_ABSOLUTE_ORACLE_GAIN",
                _cfg_get(m1_cfg, "V538_M2_MIN_COMPONENT_ORACLE_GAIN", 0.0),
            )
        ), 0.0)
        teacher_fraction = max(float(
            _cfg_get(m1_cfg, "V540_M2_MIN_TEACHER_CAPTURE_FRACTION", 0.0)
        ), 0.0)
        adaptive_oracle_floor = max(
            absolute_floor, teacher_fraction * teacher_gain
        )
        current_epoch = int(means.get("v549_current_epoch", 0.0))
        candidate_ready = bool(
            current_epoch >= int(_cfg_get(m1_cfg, "V538_M2_MIN_START_EPOCH", 0))
            and teacher_gain >= teacher_floor
            and component_gain >= adaptive_oracle_floor
            and float(means.get("v538_component_capture_ratio", 0.0))
                >= float(_cfg_get(m1_cfg, "V540_M2_MIN_CAPTURE_RATIO", 0.0))
            and float(means.get("v538_positive_component_rate", 0.0))
                >= float(_cfg_get(m1_cfg, "V538_M2_MIN_POSITIVE_COMPONENT_RATE", 0.0))
            and float(means.get("v538_component_purity", 0.0))
                >= float(_cfg_get(m1_cfg, "V538_M2_MIN_COMPONENT_PURITY", 0.0))
        )
        epoch_ready = selector_ready and candidate_ready
        close_patience = max(
            int(_cfg_get(m1_cfg, "V552_QUALITY_CLOSE_PATIENCE", 1)), 1
        )
        opened = bool(
            refiner.update_v538_quality_ready(
                epoch_ready,
                patience=patience,
                close_patience=close_patience,
            )
        )
        means.update({
            "v549_quality_evidence_ready": 1.0 if evidence_ready else 0.0,
            "v549_quality_execute_precision": precision,
            "v549_quality_harm_rate": harm_rate,
            "v549_quality_mean_selected_gain": mean_gain,
            "v549_quality_selector_ready": 1.0 if selector_ready else 0.0,
            "v549_quality_candidate_ready": 1.0 if candidate_ready else 0.0,
            "v538_quality_ready_epoch_fraction": 1.0 if epoch_ready else 0.0,
            "v538_quality_ready": 1.0 if opened else 0.0,
            "v538_quality_ready_streak": float(
                refiner.v538_quality_ready_streak.item()
            ),
            "v552_quality_bad_streak": float(
                refiner.v552_quality_bad_streak.item()
            ),
            "v552_outcome_balanced_accuracy_gate": outcome_balanced_accuracy,
            "v552_harm_sign_recall_gate": harm_sign_recall,
            "v552_benefit_recall_gate": benefit_recall,
        })
        if logger is not None:
            logger.info(
                "[V549_QUALITY_GATE] executed=%.0f min=%d precision=%.4f "
                "harm=%.4f gain=%.6f candidate_ready=%s selector_ready=%s "
                "streak=%d/%d opened=%s",
                executed, min_executed, precision, harm_rate, mean_gain,
                candidate_ready, selector_ready,
                int(refiner.v538_quality_ready_streak.item()), patience, opened,
            )
        return opened

    ready_fraction = float(means.get("v538_m2_ready", 0.0))
    required_fraction = float(
        _cfg_get(m1_cfg, "V538_M2_READY_EPOCH_FRACTION", 0.50)
    )
    opened = bool(
        refiner.update_v538_quality_ready(
            ready_fraction >= required_fraction,
            patience=patience,
            close_patience=max(
                int(_cfg_get(m1_cfg, "V552_QUALITY_CLOSE_PATIENCE", 1)), 1
            ),
        )
    )
    means["v538_quality_ready_epoch_fraction"] = ready_fraction
    means["v538_quality_ready"] = 1.0 if opened else 0.0
    means["v538_quality_ready_streak"] = float(
        refiner.v538_quality_ready_streak.item()
    )
    if logger is not None:
        logger.info(
            "[V538_QUALITY_GATE] ready_fraction=%.4f required=%.4f "
            "streak=%d/%d opened=%s",
            ready_fraction, required_fraction,
            int(refiner.v538_quality_ready_streak.item()), patience, opened,
        )
    return opened


def _v536_apply_source_freeze(model, cfg, epoch, logger=None):
    """Freeze Base/PVL/M1 exactly when V536 deployment begins.

    The selector must learn against a stationary candidate distribution.  This
    is an execution-stage contract rather than a tuned loss weight.  The
    optimizer may still hold the frozen tensors, but they receive no gradients
    and their modules are switched to eval by _v488_keep_frozen_modules_eval.
    """
    m1_cfg = _cfg_get(cfg, "M1", None)
    v537_enabled = bool(_cfg_get(m1_cfg, "V537_COMPONENT_UTILITY_RANKER_ENABLED", False))
    enabled = bool(
        m1_enabled(cfg)
        and (
            bool(_cfg_get(m1_cfg, "V536_PRIOR_ALIGNED_CASE_COMPONENT_ENABLED", False))
            or v537_enabled
        )
        and bool(
            _cfg_get(
                m1_cfg,
                "V537_FREEZE_SOURCE_AT_RANKER",
                _cfg_get(m1_cfg, "V536_FREEZE_SOURCE_AT_DEPLOY", True),
            )
        )
    )
    if not enabled or int(epoch) < _v536_deploy_start_epoch(cfg):
        return False

    prefixes = (
        "mask_head.",
        "upscale.",
        "pvl_adapters.",
        "v484_pipeline.error_state_head.",
        "v484_pipeline.local_generator.",
    )
    frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                parameter.grad = None
            frozen += parameter.numel()

    if not bool(getattr(model, "_v536_source_frozen", False)):
        setattr(model, "_v536_source_frozen", True)
        if logger is not None:
            logger.info(
                ("[V537_STAGE]" if v537_enabled else "[V536_STAGE]")
                + " epoch=%d fixed Base/PVL/M1 source at ranker/deployment "
                "start=%d | frozen_parameters=%.6fM | trainable target=M2 only",
                int(epoch) + 1,
                _v536_deploy_start_epoch(cfg) + 1,
                frozen / 1.0e6,
            )
    return True


def _force_unified_e2e_trainable(model, cfg, logger):
    """Apply the actual trainable-parameter contract for unified M1 runs.

    V490/V490.2 uses ``v484_pipeline`` as the only active M1/M2/M3 path.
    Historical ``m1_pse``/``ccv_m2``/text-verifier modules can still be
    registered for checkpoint compatibility, but they are not called by the
    V490 forward graph.  Marking those compatibility modules trainable creates
    optimizer/EMA parameters that can never receive gradients.  They are
    therefore excluded from the V490 task graph rather than being described as
    frozen active modules.
    """
    # V518_FIXED_BASE_M1_ONLY_CONTRACT
    #
    # Official BUSI Base/PVL is an immutable factual reference.
    # Train only the V518 factual error locator and candidate generators.
    # This branch must run before V490_ACTIVE_E2E, otherwise V490 re-enables
    # Base/PVL and violates V469_FREEZE_BASE.
    m1_cfg = _cfg_get(cfg, "M1", None)

    # AutoZero has a deliberately different, minimal M1 architecture.
    #
    # It must NOT be validated against the legacy GEOTR/SemLT owners such as
    # pixel_fuse/global_context/context_film/distribution_trunk/flow_head.
    # AutoZero owns exactly:
    #   image_stem + semantic_proj + fuse + controller
    #
    # Base decoder/PVL remains trainable in the matched-budget E2E protocol;
    # UniMedCLIP/BERT encoders remain frozen.
    autozero = (
        m1_enabled(cfg)
        and str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower()
        == "semlt_autozero"
    )

    if autozero:
        uc_fnrt = bool(_cfg_get(m1_cfg, "SEMLT_UC_FNRT", False))
        uc_offset_head_registered = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_OFFSET_HEAD_REGISTERED",
                _cfg_get(m1_cfg, "SEMLT_UC_OFFSET_DISTRIBUTION", False),
            )
        )
        uc_offset_distribution = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_OFFSET_DISTRIBUTION", False)
        )
        uc_hrcv = bool(_cfg_get(m1_cfg, "SEMLT_UC_HRCV", False))
        uc_hrcv_registered = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV_REGISTERED", uc_hrcv)
        )
        uc_hrcv_candidate_conditioned = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV_CANDIDATE_CONDITIONED", uc_hrcv)
        )
        uc_mrm_registered = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_REGISTERED", False)
        )
        uc_mrm_relational = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_RELATIONAL_COST", False)
        )
        uc_mrm_ordered = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_ORDERED_AGGREGATION", False)
        )
        uc_mrm_dominant = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_DOMINANT_MODE", False)
        )
        uc_operator_aligned = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_OPERATOR_ALIGNED_CANDIDATES", False)
        )
        uc_ordered_cdf = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_ORDERED_CDF_LOSS", False)
        )
        uc_reachable_match_only = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_REACHABLE_MATCH_ONLY", False)
        )
        uc_hierarchical_ordinal = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HIERARCHICAL_ORDINAL_LOSS", False)
        )
        uc_hierarchical_joint_proper = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_HIERARCHICAL_JOINT_PROPER_SCORE",
                False,
            )
        )
        uc_mrm_structured = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_STRUCTURED_LOSS", False)
        )
        if uc_offset_distribution and not uc_offset_head_registered:
            raise RuntimeError(
                "DNR trainability contract: distribution enabled but matched "
                "offset head is not registered."
            )
        if uc_hrcv and not uc_offset_distribution:
            raise RuntimeError("HRCV trainability contract requires DNR distribution")
        if uc_hrcv and not uc_hrcv_registered:
            raise RuntimeError("HRCV enabled but matched HRCV modules are not registered")
        if uc_hrcv_candidate_conditioned and not uc_hrcv:
            raise RuntimeError("candidate-conditioned HRCV requires HRCV enabled")
        if uc_mrm_registered and not uc_hrcv_registered:
            raise RuntimeError("MRM registration requires HRCV registration")
        if (uc_mrm_relational or uc_mrm_ordered) and not uc_mrm_registered:
            raise RuntimeError("MRM relational/ordered paths require registered modules")
        if uc_mrm_ordered and not uc_mrm_relational:
            raise RuntimeError("MRM ordered aggregation requires relational cost")
        if uc_mrm_dominant and not uc_offset_distribution:
            raise RuntimeError("MRM dominant-mode decoder requires DNR distribution")
        if uc_operator_aligned and not uc_hrcv_candidate_conditioned:
            raise RuntimeError(
                "operator-aligned candidate evidence requires candidate-conditioned HRCV"
            )
        if (uc_ordered_cdf or uc_reachable_match_only) and not uc_offset_distribution:
            raise RuntimeError(
                "ordered/reachable distribution supervision requires DNR distribution"
            )
        if uc_hierarchical_ordinal and not uc_offset_distribution:
            raise RuntimeError(
                "hierarchical ordinal supervision requires DNR distribution"
            )
        if uc_hierarchical_ordinal and (uc_ordered_cdf or uc_mrm_structured):
            raise RuntimeError(
                "hierarchical ordinal supervision replaces joint ordered CDF/MRM "
                "structured supervision; enable exactly one"
            )
        if uc_hierarchical_joint_proper and not uc_hierarchical_ordinal:
            raise RuntimeError(
                "hierarchical joint proper score requires hierarchical ordinal supervision"
            )

        # UC-FNRT no longer uses the archived AutoZero ``fuse + controller``
        # predictor.  Its physical forward path is:
        #   image_stem / semantic_proj
        #   -> posterior-aware uc_context_fuse
        #   -> compact bilateral normal-ray projections/fusion
        #   -> direction_controller + magnitude_controller.
        # Keep a route-specific whitelist so compatibility modules that remain
        # registered in the state dict cannot silently enter the optimizer as
        # dead trainable parameters.  Legacy AutoZero/BNW/SDF/PS-OMW keeps its
        # original ``fuse + controller`` contract unchanged.
        fixed_base_refinement = bool(
            _cfg_get(m1_cfg, "SEMLT_FIXED_BASE_REFINEMENT", False)
        )
        base_prefixes = () if fixed_base_refinement else (
            "mask_head.",
            "upscale.",
            "pvl_adapters.",
        )
        if uc_fnrt:
            m1_active_prefixes = (
                "m1_pse.image_stem.",
                "m1_pse.semantic_proj.",
                "m1_pse.uc_context_fuse.",
                "m1_pse.uc_ray_image_proj.",
                "m1_pse.uc_ray_semantic_proj.",
                "m1_pse.uc_ray_cue_proj.",
                "m1_pse.uc_ray_fuse.",
                "m1_pse.direction_controller.",
                "m1_pse.magnitude_controller.",
            )
            if uc_offset_head_registered:
                m1_active_prefixes = m1_active_prefixes + (
                    "m1_pse.offset_controller.",
                )
            if uc_hrcv_registered:
                m1_active_prefixes = m1_active_prefixes + (
                    "m1_pse.uc_hrcv_context.",
                    "m1_pse.uc_hrcv_cue_proj.",
                    "m1_pse.uc_hrcv_scorer.",
                )
            if uc_mrm_registered:
                m1_active_prefixes = m1_active_prefixes + (
                    "m1_pse.uc_mrm_rel_embed.",
                    "m1_pse.uc_mrm_unary_head.",
                    "m1_pse.uc_mrm_aggregate.",
                    "m1_pse.uc_mrm_head.",
                )
        else:
            m1_active_prefixes = (
                "m1_pse.image_stem.",
                "m1_pse.semantic_proj.",
                "m1_pse.fuse.",
                "m1_pse.controller.",
            )

        allowed_prefixes = base_prefixes + m1_active_prefixes
        active_names = []

        for name, parameter in model.named_parameters():
            trainable = name.startswith(allowed_prefixes)

            # Pretrained encoders must remain frozen exactly as Base protocol.
            if name.startswith(("vision_model.", "text_model.")):
                trainable = False

            parameter.requires_grad_(trainable)

            if trainable:
                active_names.append(name)

        required = m1_active_prefixes

        missing = [
            prefix
            for prefix in required
            if not any(name.startswith(prefix) for name in active_names)
        ]

        if missing:
            raise RuntimeError(
                "AutoZero trainability contract missing active modules: "
                + str(missing)
            )

        # AutoZero is M1-only. Any active Stage-2/3 module is a real contract
        # violation rather than something to silently tolerate.
        forbidden_tokens = (
            "m2_surface",
            "v4g_refiner",
            "ccv_m2",
            "m2_text_verifier",
            "m2_tide_repair",
            "safe_deployer",
            "m3_policy",
            "flow_head",
            "pixel_fuse",
            "global_context",
            "context_film",
            "distribution_trunk",
            "text_proj",
        )

        leaks = [
            name
            for name in active_names
            if any(token in name for token in forbidden_tokens)
        ]

        if leaks:
            raise RuntimeError(
                "AutoZero trainability contract leaked legacy/M2/M3 parameters: "
                + str(leaks[:50])
            )

        if uc_fnrt:
            controller_prefixes = [
                "m1_pse.direction_controller.",
                "m1_pse.magnitude_controller.",
            ]
            if uc_offset_head_registered:
                controller_prefixes.append("m1_pse.offset_controller.")
            controller_names = [
                name for name in active_names
                if name.startswith(tuple(controller_prefixes))
            ]
            direction_names = [
                name for name in active_names
                if name.startswith("m1_pse.direction_controller.")
            ]
            magnitude_names = [
                name for name in active_names
                if name.startswith("m1_pse.magnitude_controller.")
            ]
            offset_names = [
                name for name in active_names
                if name.startswith("m1_pse.offset_controller.")
            ]
            if not direction_names or not magnitude_names:
                raise RuntimeError(
                    "UC-FNRT factorized controllers are absent from optimizer graph: "
                    f"direction={len(direction_names)} magnitude={len(magnitude_names)}"
                )
            if uc_offset_head_registered and not offset_names:
                raise RuntimeError(
                    "DNR matched offset controller is absent from optimizer graph."
                )
            if (not uc_offset_head_registered) and offset_names:
                raise RuntimeError(
                    "DNR offset controller unexpectedly entered the optimizer graph."
                )
            route_label = (
                "UC-OACD"
                if (uc_operator_aligned or uc_ordered_cdf)
                else
                "UC-MRM"
                if (uc_mrm_relational or uc_mrm_ordered or uc_mrm_dominant)
                else "UC-HRCV"
                if uc_hrcv
                else "UC-DNR"
                if uc_offset_distribution
                else "UC-FNRT"
            )
        else:
            controller_names = [
                name for name in active_names
                if name.startswith("m1_pse.controller.")
            ]
            if not controller_names:
                raise RuntimeError(
                    "AutoZero controller is absent from optimizer graph."
                )
            route_label = "AutoZero"

        logger.info(
            "[%s TRAINABLE] %s%s only | "
            "encoders frozen | trainable_tensors=%d "
            "trainable_parameters=%.6fM | controller_tensors=%d",
            route_label,
            ("Fixed Base/PVL + " if fixed_base_refinement else "Base/PVL + "),
            route_label,
            len(active_names),
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ) / 1.0e6,
            len(controller_names),
        )
        logger.info(
            "[AUTOZERO TRAINABLE] active preview=%s",
            active_names[:40],
        )
        return

    # Legacy SemLT / GEOTR-M1 transport graph.
    # Only decoder/PVL and Stage-1 Transport may enter optimizer or EMA;
    # no Stage-2/M2 parameter may exist.
    if _semlt(cfg):
        causal_ablation = _geotr_m1_causal_ablation(cfg)
        use_semantic = bool(
            _cfg_get(m1_cfg, "GEOTR_M1_USE_SEMANTIC_CONDITIONING", True)
        )
        use_text = bool(
            _cfg_get(m1_cfg, "GEOTR_M1_USE_TEXT_CONDITIONING", True)
        )
        allowed_prefixes = ("mask_head.", "upscale.", "pvl_adapters.", "m1_pse.")
        active_names = []
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith("m1_pse.")
                if causal_ablation
                else name.startswith(allowed_prefixes)
            )
            if name.startswith(("vision_model.", "text_model.")):
                trainable = False
            # Optional branches stay in the state dict for a strict ablation
            # architecture, but an inactive branch is not counted or sent to
            # the optimizer.  This also fails visibly if a supposedly disabled
            # conditioner starts receiving gradients in a future refactor.
            if not use_semantic and name.startswith("m1_pse.semantic_proj."):
                trainable = False
            if not use_text and name.startswith("m1_pse.text_proj."):
                trainable = False
            parameter.requires_grad_(trainable)
            if trainable:
                active_names.append(name)
        exact_m1 = str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "geotr_m1"
        required = [
            "m1_pse.image_stem.",
            "m1_pse.pixel_fuse.", "m1_pse.global_context.", "m1_pse.context_film.",
            "m1_pse.distribution_trunk.",
        ]
        if use_semantic:
            required.append("m1_pse.semantic_proj.")
        if use_text:
            required.append("m1_pse.text_proj.")
        if exact_m1:
            required.extend([
                "m1_pse.context_gate", "m1_pse.mean_head.",
                "m1_pse.diag_std_head.", "m1_pse.m1_distribution_log_var",
            ])
        else:
            required.append("m1_pse.flow_head.")
        missing = [prefix for prefix in required if not any(name.startswith(prefix) for name in active_names)]
        if missing:
            raise RuntimeError("M1-only trainability contract missing modules: " + str(missing))
        forbidden_tokens = (
            "m2_surface", "v4g_refiner", "ccv_m2", "m2_text_verifier",
            "m2_tide_repair", "safe_deployer", "m3_policy",
        )
        leaks = [name for name in active_names if any(token in name for token in forbidden_tokens)]
        if leaks:
            raise RuntimeError("M1-only graph leaked M2/M3 parameters: " + str(leaks[:50]))
        logger.info(
            "[%s TRAINABLE] %s | "
            "trainable_tensors=%d trainable_parameters=%.6fM | M2/M3=absent",
            "GEOTR-M1 EXACT" if exact_m1 else "SemLT-M1",
            (
                "common Base/PVL frozen; exact Stage-1 Transport only"
                if causal_ablation
                else "Base/PVL + exact Stage-1 Transport only"
            ),
            len(active_names),
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.0e6,
        )
        return

    # MHCS formal trainability contract: Base/PVL + the new full-mask
    # hypothesis generator/composer only.  Historical residual/component/M2/M3
    # modules are registered for checkpoint compatibility but cannot train.
    mhcs_trainability = bool(
        str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "mhcs"
        or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
        in {"mhcs", "multi_hypothesis_composition"}
    )
    if mhcs_trainability:
        allowed_prefixes = (
            "mask_head.",
            "upscale.",
            "pvl_adapters.",
            "m1_pse.",
        )
        active_names = []
        v4g_sparse_direct_refiner = bool(_cfg_get(m1_cfg, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False))
        v4f_selective_intervention = bool(_cfg_get(m1_cfg, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)) and not v4g_sparse_direct_refiner
        v4e_operator_consistent = bool(_cfg_get(m1_cfg, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)) and not v4f_selective_intervention and not v4g_sparse_direct_refiner
        for name, parameter in model.named_parameters():
            trainable = name.startswith(allowed_prefixes)
            if name.startswith(("vision_model.", "text_model.")):
                trainable = False
            # V4E removes the continuous severity branch from both deployment
            # and objective.  Do not leave dead compatibility tensors in the
            # optimizer/EMA just because they remain in the state dict.
            if v4e_operator_consistent and name.startswith((
                "m1_pse.m2_surface.severity_body.",
                "m1_pse.m2_surface.severity_out.",
            )):
                trainable = False
            if v4g_sparse_direct_refiner and name.startswith("m1_pse.m2_surface."):
                trainable = False
            parameter.requires_grad_(trainable)
            if trainable:
                active_names.append(name)
        # Architecture-aware MHCS/GEOTR trainability contract.
        #
        # Historical GEOTR-V3 used ``factor_head.residual_context`` as a
        # trainable residual feature owner. GEOTR-V4B intentionally removes
        # that path: ``factor_head.residual_context`` is now ``nn.Identity``
        # and every Stage-2 WHERE/HOW parameter lives under ``m2_surface.*``.
        # Requiring a parameter under ``factor_head`` would therefore reject
        # the correct V4B architecture before optimizer construction.
        geotr_v4b_error_localized = bool(
            _cfg_get(m1_cfg, "GEOTOPO_REFINEMENT_ENABLED", False)
            and _cfg_get(m1_cfg, "GEOTR_ERROR_LOCALIZATION_ENABLED", False)
        )

        required = [
            # Stage-1 Transport evidence/geometry owners.
            "m1_pse.image_stem.",
            "m1_pse.semantic_proj.",
            "m1_pse.text_proj.",
            "m1_pse.pixel_fuse.",
            "m1_pse.global_context.",
            "m1_pse.context_film.",
            "m1_pse.context_gate",
            "m1_pse.distribution_trunk.",
            "m1_pse.mean_head.",
            "m1_pse.diag_std_head.",
            "m1_pse.m1_distribution_log_var",
        ]
        if v4g_sparse_direct_refiner:
            required.append("m1_pse.v4g_refiner.")
        else:
            required.append("m1_pse.m2_surface.")
        if geotr_v4b_error_localized and not v4g_sparse_direct_refiner:
            # V4B Stage-2 must contain real WHERE/HOW owners.  This is stronger
            # than the historical factor_head check and fails closed if the
            # error-localizer/correction implementation is accidentally absent.
            required.extend([
                "m1_pse.m2_surface.image_stem.",
                "m1_pse.m2_surface.semantic_proj.",
                "m1_pse.m2_surface.text_proj.",
                "m1_pse.m2_surface.error_out.",
                "m1_pse.m2_surface.correction_out.",
            ])
            if bool(_cfg_get(m1_cfg, "GEOTR_V4D_ROOT_FIX_ENABLED", False)) and not v4f_selective_intervention:
                required.extend([
                    "m1_pse.m2_surface.severity_body.",
                    "m1_pse.m2_surface.severity_out.",
                ])
            if v4f_selective_intervention:
                required.extend([
                    "m1_pse.m2_surface.v4f_trace_fuse.",
                    "m1_pse.m2_surface.v4f_policy_body.",
                    "m1_pse.m2_surface.v4f_edit_out.",
                    "m1_pse.m2_surface.v4f_direction_out.",
                ])
        elif not v4g_sparse_direct_refiner:
            # Legacy MHCS / GEOTR-V3 compatibility.
            required.append("m1_pse.factor_head.")

        missing = [r for r in required if not any(n.startswith(r) for n in active_names)]
        if missing:
            raise RuntimeError("MHCS trainability contract missing modules: " + str(missing))
        forbidden = (
            "v484_pipeline.", "ccv_m2.", "m2_text_verifier.",
            "m2_tide_repair_head.",
        )
        leaks = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(forbidden)
        ]
        if leaks:
            raise RuntimeError("MHCS leaked historical M1/M2/M3 parameters: " + str(leaks[:50]))
        active_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if bool(_cfg_get(m1_cfg, "GEOTOPO_REFINEMENT_ENABLED", False)):
            if v4g_sparse_direct_refiner:
                if bool(_cfg_get(m1_cfg, "GEOTR_SPARC_HR_ENABLED", False)):
                    logger.info(
                        "[SPARC-HR TRAINABLE] Base/PVL + validated Transport M1 + GT-free HR atomic posterior composer; "
                        "same replacement operator for utility supervision/deployment; benefit-set STOP + direct lower quantile; "
                        "true paired 2x HR image/mask | trainable_tensors=%d trainable_parameters=%.6fM",
                        len(active_names), active_parameters / 1.0e6,
                    )
                elif bool(_cfg_get(m1_cfg, "GEOTR_C2R_ENABLED", False)):
                    if bool(_cfg_get(m1_cfg, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False)):
                        if bool(_cfg_get(m1_cfg, "GEOTR_AEFR_ENABLED", False)):
                            logger.info(
                                "[GEOTR-AEFR TRAINABLE] Base/PVL + unchanged Transport + factual residual refiner; "
                                "stage=%s joint_geometry_grad=%s transition_aware=%s raw_flow_evidence=%s; "
                                "typed controls remain reproducible; IFR factorizes hard residual intervention into dense WHERE, editness, ADD/REMOVE direction and operator-specific magnitude; "
                                "posterior samples are diagnostic probes only | "
                                "trainable_tensors=%d trainable_parameters=%.6fM",
                                str(_cfg_get(m1_cfg, "GEOTR_AEFR_STAGE", "single_atomic")),
                                bool(_cfg_get(m1_cfg, "GEOTR_AEFR_JOINT_GEOMETRY_GRAD_ENABLED", False)),
                                bool(_cfg_get(m1_cfg, "GEOTR_AEFR_TRANSITION_AWARE_ENABLED", False)),
                                bool(_cfg_get(m1_cfg, "GEOTR_AEFR_USE_RAW_FLOW_EVIDENCE", True)),
                                len(active_names), active_parameters / 1.0e6,
                            )
                        elif bool(_cfg_get(m1_cfg, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False)):
                            logger.info(
                                "[GEOTR-PC2R-V3.2 TRAINABLE] Base/PVL + unchanged Transport + PC2R-v3.1 canonical-coordinate posterior refiner; "
                                "Stage-2 objective is operator-aligned 0.5*canonical + 0.5*actual-deployed-final BCE/Dice; "
                                "posterior view losses are diagnostic-only; native operator/reachability/boundary audits enabled | "
                                "trainable_tensors=%d trainable_parameters=%.6fM",
                                len(active_names), active_parameters / 1.0e6,
                            )
                        elif bool(_cfg_get(m1_cfg, "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED", False)):
                            logger.info(
                                "[GEOTR-PC2R-V3.1 TRAINABLE] Base/PVL + unchanged Transport + actual Geometry-aligned MC posterior ROI conditions + "
                                "canonical-coordinate bounded logit residuals + directional/risk atomic component certificate; "
                                "all posterior-conditioned branches supervise the same factual Z0 correction coordinate | "
                                "trainable_tensors=%d trainable_parameters=%.6fM",
                                len(active_names), active_parameters / 1.0e6,
                            )
                        elif bool(_cfg_get(m1_cfg, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False)):
                            logger.info(
                                "[GEOTR-PC2R-V3 TRAINABLE] Base/PVL + unchanged Transport + actual Geometry-aligned MC posterior ROI conditions + "
                                "per-posterior bounded logit residuals + canonical residual aggregation + directional/risk atomic certificate | "
                                "trainable_tensors=%d trainable_parameters=%.6fM",
                                len(active_names), active_parameters / 1.0e6,
                            )
                        else:
                            logger.info(
                                "[GEOTR-C2R-V2 TRAINABLE] Base/PVL + unchanged Transport + true non-overlapping uncertainty ROIs + "
                                "canonical factual-reference erode/factual/dilate reconstruction + atomic connected-component certificate/commit; "
                                "hard commit is diagnostic/deployment-only, training uses view+canonical BCE/Dice | "
                                "trainable_tensors=%d trainable_parameters=%.6fM",
                                len(active_names), active_parameters / 1.0e6,
                            )
                    else:
                        logger.info(
                            "[GEOTR-C2R TRAINABLE] Base/PVL + unchanged Transport + deterministic uncertainty ROIs + "
                            "shared factual/erode/dilate regional reconstruction + GT-free counterfactual consensus commit; "
                            "no synthetic action labels, no learned safety gate | trainable_tensors=%d trainable_parameters=%.6fM",
                            len(active_names), active_parameters / 1.0e6,
                        )
                elif bool(_cfg_get(m1_cfg, "GEOTR_V4G_R4_EXOGENOUS_PATCH_FLIP_ENABLED", False)):
                    logger.info(
                        "[GEOTR-V4G-R4 TRAINABLE] Base/PVL + unchanged Transport + top-K inspection + local patch FLIP/KEEP head + "
                        "stationary exogenous structured corruption; deterministic minimal flip; no factual residual target | "
                        "trainable_tensors=%d trainable_parameters=%.6fM",
                        len(active_names), active_parameters / 1.0e6,
                    )
                elif bool(_cfg_get(m1_cfg, "GEOTR_V4G_R3_MINIMAL_INTERVENTION_ENABLED", False)):
                    logger.info(
                        "[GEOTR-V4G-R3 TRAINABLE] Base/PVL + unchanged Transport + top-K sparse point refiner + "
                        "transport-aligned MC/fine evidence + natural-prevalence minimal signed delta regression; DN=OFF | "
                        "trainable_tensors=%d trainable_parameters=%.6fM",
                        len(active_names), active_parameters / 1.0e6,
                    )
                elif bool(_cfg_get(m1_cfg, "GEOTR_V4G_R2_CORRECTION_PRESERVE_ENABLED", False)):
                    logger.info(
                        "[GEOTR-V4G-R2 TRAINABLE] Base/PVL + unchanged Transport + correction/preserve sparse point refiner + "
                        "transport-aligned MC evidence + detached decoder fine feature + structured denoising | "
                        "trainable_tensors=%d trainable_parameters=%.6fM",
                        len(active_names), active_parameters / 1.0e6,
                    )
                else:
                    logger.info(
                        "[GEOTR-V4G TRAINABLE] Base/PVL + Transport + uncertainty-guided sparse direct GT refiner + factual/exogenous denoising + MC evidence | "
                        "trainable_tensors=%d trainable_parameters=%.6fM",
                        len(active_names), active_parameters / 1.0e6,
                    )
            elif v4f_selective_intervention:
                logger.info(
                    "[GEOTR-V4F TRAINABLE] Base/PVL + Transport + high-recall proposal + selective execution + bounded conditional dose + Transport trace | "
                    "trainable_tensors=%d trainable_parameters=%.6fM",
                    len(active_names), active_parameters / 1.0e6,
                )
            elif bool(_cfg_get(m1_cfg, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)):
                logger.info(
                    "[GEOTR-V4E TRAINABLE] Base/PVL + Transport + hard typed SUPPORT + full-dose HOW; legacy severity branch frozen | "
                    "trainable_tensors=%d trainable_parameters=%.6fM",
                    len(active_names), active_parameters / 1.0e6,
                )
            else:
                logger.info(
                    "[GEOTR TRAINABLE] Base/PVL + Transport + typed WHERE + continuous SEVERITY + teacher-decoupled HOW Stage-2 | "
                    "trainable_tensors=%d trainable_parameters=%.6fM",
                    len(active_names), active_parameters / 1.0e6,
                )
        else:
            logger.info(
                "[MHCS-R5.2 TRAINABLE] Base/PVL + unchanged R5.1 bank + per-candidate Harm/Neutral/Benefit safety router | "
                "trainable_tensors=%d trainable_parameters=%.6fM",
                len(active_names), active_parameters / 1.0e6,
            )
        logger.info("[MHCS_TRAINABLE] active preview=%s", active_names[:50])
        return

    # V531/V532 unified sparse refiner trainable contract.
    # Legacy static marker retained: [V531_TRAINABLE]
    v532_enabled = bool(
        _clean_dynamic_component_set(cfg)
        or (m1_enabled(cfg) and bool(_cfg_get(m1_cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False)))
    )
    v531_enabled = bool(
        m1_enabled(cfg)
        and bool(_cfg_get(m1_cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False))
    )
    if v532_enabled or v531_enabled:
        clean_dynamic_component_set = _clean_dynamic_component_set(cfg)
        tc_drcs = _tc_drcs(cfg)
        version = "TC_DRCS" if tc_drcs else ("CLEAN_COMPONENT_SET" if clean_dynamic_component_set else ("V532" if v532_enabled else "V531"))
        stage_key = "CLEAN_STAGE" if clean_dynamic_component_set else ("V532_TRAIN_STAGE" if v532_enabled else "V531_TRAIN_STAGE")
        freeze_key = "CLEAN_FREEZE_BASE" if clean_dynamic_component_set else ("V532_FREEZE_BASE" if v532_enabled else "V531_FREEZE_BASE")
        stage = (
            "full" if clean_dynamic_component_set else
            str(_cfg_get(m1_cfg, stage_key, "full" if v532_enabled else "router")).strip().lower()
        )
        if stage not in {"router", "joint", "full"}:
            raise ValueError(
                f"{stage_key} must be one of router/joint/full, got {stage!r}"
            )
        freeze_base = False if clean_dynamic_component_set else bool(_cfg_get(m1_cfg, freeze_key, not v532_enabled))
        if clean_dynamic_component_set:
            # Clean-family formal path: only scientific owners that receive
            # live M1/M2 gradients are optimizer/EMA parameters.
            c = "v484_pipeline.pixel_composer.component_slot_generator."
            allowed_prefixes = [
                "v484_pipeline.pixel_composer.context_encoder.",
                "v484_pipeline.pixel_composer.semantic_proj.",
                "v484_pipeline.pixel_composer.semantic_fuse.",
                c + "r47_pixel_encoder.",
                c + "r47_slot_queries.",
                c + "v561_global_decoder.",
                c + "v561_query_action_head.",
                c + "v561_mask_embed.",
                c + "v561_mask_bias",
                c + "v562_residual_head.",
                c + "v562_query_presence_head.",
                c + "clean_loss_log_vars",
                c + "pyramid_blocks.",
                c + "scale_router.",
                c + "pyramid_fuse.",
                # Minimal M2 is intentionally unchanged in TC-DRCS.
                c + "m2_outcome_encoder.",
                c + "m2_selector_trunk.",
                c + "clean_gain_head.",
            ]
            if tc_drcs:
                allowed_prefixes.extend([
                    c + "tc_mask_pixel_fuse.",
                    c + "tc_pilot_query_proj.",
                ])
            else:
                allowed_prefixes.extend([
                    c + "v562_anchor_feature_proj.",
                    c + "v562_anchor_pos_mlp.",
                    c + "clean_mask_pixel_fuse.",
                    c + "clean_attention_precision_head.",
                    c + "clean_mask_precision_head.",
                ])
        else:
            allowed_prefixes = ["v484_pipeline.pixel_composer."]
            if stage in {"joint", "full"}:
                allowed_prefixes.extend(
                    [
                        "v484_pipeline.error_state_head.",
                        "v484_pipeline.local_generator.",
                    ]
                )
        if stage == "full" and not freeze_base:
            allowed_prefixes.extend(["pvl_adapters.", "mask_head.", "upscale."])
        allowed_prefixes = tuple(allowed_prefixes)

        active_names = []
        for name, parameter in model.named_parameters():
            trainable = any(name.startswith(prefix) for prefix in allowed_prefixes)
            parameter.requires_grad_(trainable)
            if trainable:
                active_names.append(name)

        if clean_dynamic_component_set:
            required = [
                "v484_pipeline.pixel_composer.context_encoder.",
                "v484_pipeline.pixel_composer.component_slot_generator.v562_residual_head.",
                "v484_pipeline.pixel_composer.component_slot_generator.clean_gain_head.",
            ]
            required.append(
                "v484_pipeline.pixel_composer.component_slot_generator.tc_mask_pixel_fuse."
                if tc_drcs else
                "v484_pipeline.pixel_composer.component_slot_generator.clean_mask_pixel_fuse."
            )
        else:
            required = ["v484_pipeline.pixel_composer."]
            if stage in {"joint", "full"}:
                required.extend(
                    [
                        "v484_pipeline.error_state_head.",
                        "v484_pipeline.local_generator.",
                    ]
                )
        missing = [
            prefix for prefix in required
            if not any(name.startswith(prefix) for name in active_names)
        ]
        if missing:
            raise RuntimeError(
                f"{version} trainable contract is missing required modules: {missing}"
            )

        forbidden_prefixes = (
            "vision_model.",
            "text_model.",
            "m1_pse.",
            "ccv_m2.",
            "m2_text_verifier.",
            "m2_tide_repair_head.",
            "v484_pipeline.safe_deployer.",
            "v484_pipeline.global_generator.",
            "v484_pipeline.local_verifier.",
            "v484_pipeline.rejector.",
        )
        leaks = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(forbidden_prefixes)
        ]
        if leaks:
            raise RuntimeError(
                f"{version} contract leaked encoder/legacy/M3 tensors: {leaks[:50]}"
            )
        if clean_dynamic_component_set:
            clean_forbidden = (
                ".benefit_head.", ".harm_head.", ".rank_head.", ".gain_head.",
                ".gain_logvar_head.", ".editor_", ".composer_", ".r48_",
                ".v561_typed_decoder.", ".candidate_utility_outcome_head.",
                ".candidate_absolute_gain_head.", ".atom_quality_head.",
                "v484_pipeline.error_state_head.", "v484_pipeline.local_generator.",
                "v484_pipeline.pixel_composer.utility_policy_head.",
            )
            if tc_drcs:
                clean_forbidden = clean_forbidden + (
                    ".v562_anchor_feature_proj.", ".v562_anchor_pos_mlp.",
                    ".clean_mask_pixel_fuse.", ".clean_attention_precision_head.",
                    ".clean_mask_precision_head.",
                )
            clean_leaks = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(token in name for token in clean_forbidden)
            ]
            if clean_leaks:
                raise RuntimeError(
                    "CLEAN formal optimizer leaked historical parameters: "
                    + str(clean_leaks[:50])
                )
        if freeze_base:
            base_leaks = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and name.startswith(
                    ("pvl_adapters.", "mask_head.", "upscale.")
                )
            ]
            if base_leaks:
                raise RuntimeError(
                    f"{version} fixed-Base contract leaked Base/PVL tensors: "
                    + str(base_leaks[:50])
                )

        active_parameters = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "[%s_TRAINABLE] stage=%s freeze_base=%s | "
            "unified typed sparse refiner active | trainable_tensors=%d "
            "trainable_parameters=%.6fM",
            version, stage, freeze_base, len(active_names), active_parameters / 1.0e6,
        )
        logger.info("[%s_TRAINABLE] active preview=%s", version, active_names[:50])
        return

    v518_fixed_base_m1_only = bool(
        m1_enabled(cfg)
        and bool(_cfg_get(m1_cfg, "V518_ENABLED", False))
        and bool(_cfg_get(m1_cfg, "V469_FREEZE_BASE", False))
        and bool(_cfg_get(m1_cfg, "V515_OFFICIAL_BASE_INIT", False))
        and not _v519_is_m2_only(cfg)
        and not _v519_is_joint_m1_m2(cfg)
    )

    if v518_fixed_base_m1_only:
        allowed_prefixes = (
            "v484_pipeline.error_state_head.",
            "v484_pipeline.local_generator.",
            "v484_pipeline.global_generator.",
        )

        active_names = []
        frozen_names = []

        for name, parameter in model.named_parameters():
            trainable = name.startswith(allowed_prefixes)
            parameter.requires_grad_(trainable)

            if trainable:
                active_names.append(name)
            else:
                frozen_names.append(name)

        required_prefixes = (
            "v484_pipeline.error_state_head.",
            "v484_pipeline.local_generator.",
        )

        missing_required = [
            prefix
            for prefix in required_prefixes
            if not any(
                name.startswith(prefix)
                for name in active_names
            )
        ]

        if missing_required:
            raise RuntimeError(
                "V518 fixed-Base M1-only contract is missing "
                "required modules: "
                + str(missing_required)
            )

        forbidden_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not name.startswith(allowed_prefixes)
        ]

        if forbidden_trainable:
            raise RuntimeError(
                "V518 fixed-Base M1-only contract leaked "
                "non-M1 trainable tensors: "
                + str(forbidden_trainable[:50])
            )

        base_leaks = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith(
                (
                    "pvl_adapters.",
                    "mask_head.",
                    "upscale.",
                    "vision_model.",
                    "text_model.",
                )
            )
        ]

        if base_leaks:
            raise RuntimeError(
                "V518 fixed-Base contract found trainable "
                "Base/PVL/encoder tensors: "
                + str(base_leaks[:50])
            )

        m2_m3_leaks = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith(
                (
                    "v484_pipeline.pixel_composer.",
                    "v484_pipeline.safe_deployer.",
                    "v484_pipeline.local_verifier.",
                    "v484_pipeline.rejector.",
                    "m1_pse.",
                    "ccv_m2.",
                    "m2_text_verifier.",
                    "m2_tide_repair_head.",
                )
            )
        ]

        if m2_m3_leaks:
            raise RuntimeError(
                "V518 M1-only contract leaked M2/M3/legacy tensors: "
                + str(m2_m3_leaks[:50])
            )

        active_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        logger.info(
            "[V518_FIXED_BASE_M1_ONLY] official Base/PVL frozen | "
            "only error-state locator and candidate generators trainable | "
            "trainable_tensors=%d trainable_parameters=%.6fM "
            "frozen_tensors=%d",
            len(active_names),
            active_parameters / 1.0e6,
            len(frozen_names),
        )

        logger.info(
            "[V518_FIXED_BASE_M1_ONLY] active preview=%s",
            active_names[:50],
        )

        return

    if _v519_is_joint_m1_m2(cfg):
        frozen_prefixes = (
            "vision_model.",
            "text_model.",
            "_v396_observer_vision.",
            "m1_pse.",
            "ccv_m2.",
            "m2_text_verifier.",
            "m2_tide_repair_head.",
            "v484_pipeline.safe_deployer.",
        )
        frozen_exact = {"logit_scale"}
        active = []
        frozen = []
        for name, parameter in model.named_parameters():
            should_freeze = name in frozen_exact or name.startswith(frozen_prefixes)
            parameter.requires_grad_(not should_freeze)
            (frozen if should_freeze else active).append(name)

        required = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
            "v484_pipeline.error_state_head.",
            "v484_pipeline.local_generator.",
            "v484_pipeline.pixel_composer.",
        )
        absent = [p for p in required if not any(n.startswith(p) for n in active)]
        if absent:
            raise RuntimeError(
                "V519/V520/V521/V522/V523 joint Base/PVL/M1/M2 contract missing active prefixes: "
                + str(absent)
            )
        leaked_m3 = [n for n in active if n.startswith("v484_pipeline.safe_deployer.")]
        if leaked_m3:
            raise RuntimeError("V519/V520/V521/V522/V523 joint contract leaked M3 tensors: " + str(leaked_m3[:20]))
        logger.info(
            "[V519_JOINT_M1_M2] Base/PVL/M1/M2 trainable; image/text/M3 frozen | "
            "M2 receives detached Base/M1 candidate observations | active_tensors=%d",
            len(active),
        )
        logger.info("[V519_JOINT_M1_M2] active preview=%s", active[:40])
        return

    if _v519_is_m2_only(cfg):
        allowed_prefix = "v484_pipeline.pixel_composer."
        v528_enabled = bool(
            _cfg_get(cfg.M1, "V528_OUTCOME_COMPOSER_ENABLED", False)
        )
        v529_enabled = bool(
            _cfg_get(cfg.M1, "V529_CALIBRATION_FIRST_OUTCOME_ENABLED", False)
        )
        v530_enabled = bool(
            _cfg_get(cfg.M1, "V530_PROBABILITY_CALIBRATED_OUTCOME_ENABLED", False)
        )
        shared_feature_prefixes = (
            "context_encoder.",
            "semantic_proj.",
            "candidate_encoder.",
            "family_embedding.",
            "action_embedding.",
            "metadata_proj.",
            "candidate_fuse.",
        )
        set_feature_prefixes = (
            "set_encoder.",
            "preserve_token",
            "region_context_proj.",
        )
        v528_trainable_prefixes = shared_feature_prefixes + set_feature_prefixes + (
            "v528_base_confusion_head.",
            "v528_outcome_correctness_head.",
            "v528_outcome_logvar_head.",
        )
        v529_uncertainty_enabled = bool(
            _cfg_get(cfg.M1, "V529_UNCERTAINTY_ENABLED", False)
        )
        v529_trainable_prefixes = shared_feature_prefixes + (
            "v529_base_error_head.",
            "v529_outcome_pixel_head.",
        ) + (
            set_feature_prefixes + ("v529_outcome_logvar_head.",)
            if v529_uncertainty_enabled else ()
        )
        v530_uncertainty_enabled = bool(
            _cfg_get(cfg.M1, "V530_UNCERTAINTY_ENABLED", False)
        )
        v530_trainable_prefixes = shared_feature_prefixes + (
            "v530_base_error_pixel_head.",
            "v530_outcome_coarse_head.",
            "v530_outcome_refine_head.",
        ) + (
            set_feature_prefixes + ("v530_outcome_logvar_head.",)
            if v530_uncertainty_enabled else ()
        )
        common_forbidden_prefixes = (
            "case_verifier.",
            "family_selector.",
            "prompt_head.",
            "refiner.",
            "route_head.",
            "editability_head.",
            "conditional_score_head.",
            "dense_utility_head.",
            "dense_logvar_head.",
            "dense_harm_head.",
            "region_utility_head.",
            "region_logvar_head.",
            "region_harm_head.",
        )
        v528_forbidden_prefixes = common_forbidden_prefixes + (
            "v529_base_error_head.",
            "v529_outcome_pixel_head.",
            "v529_outcome_logvar_head.",
        )
        v529_forbidden_prefixes = common_forbidden_prefixes + (
            "v528_base_confusion_head.",
            "v528_outcome_correctness_head.",
            "v528_outcome_logvar_head.",
            "v530_base_error_pixel_head.",
            "v530_outcome_coarse_head.",
            "v530_outcome_refine_head.",
            "v530_outcome_logvar_head.",
        ) + (
            () if v529_uncertainty_enabled
            else set_feature_prefixes + ("v529_outcome_logvar_head.",)
        )
        v530_forbidden_prefixes = common_forbidden_prefixes + (
            "v528_base_confusion_head.",
            "v528_outcome_correctness_head.",
            "v528_outcome_logvar_head.",
            "v529_base_error_head.",
            "v529_outcome_pixel_head.",
            "v529_outcome_logvar_head.",
        ) + (
            () if v530_uncertainty_enabled
            else set_feature_prefixes + ("v530_outcome_logvar_head.",)
        )
        active = []
        leaked_outcome = []
        for name, parameter in model.named_parameters():
            local_name = (
                name[len(allowed_prefix):]
                if name.startswith(allowed_prefix)
                else ""
            )
            if v530_enabled:
                trainable_prefixes = v530_trainable_prefixes
                forbidden_prefixes = v530_forbidden_prefixes
            elif v529_enabled:
                trainable_prefixes = v529_trainable_prefixes
                forbidden_prefixes = v529_forbidden_prefixes
            elif v528_enabled:
                trainable_prefixes = v528_trainable_prefixes
                forbidden_prefixes = v528_forbidden_prefixes
            else:
                trainable_prefixes = ()
                forbidden_prefixes = ()

            if v530_enabled or v529_enabled or v528_enabled:
                trainable = (
                    name.startswith(allowed_prefix)
                    and any(
                        local_name == prefix
                        or local_name.startswith(prefix)
                        for prefix in trainable_prefixes
                    )
                    and not any(
                        local_name.startswith(prefix)
                        for prefix in forbidden_prefixes
                    )
                )
            else:
                trainable = name.startswith(allowed_prefix)
            parameter.requires_grad_(trainable)
            if trainable:
                active.append(name)
                if (v530_enabled or v529_enabled or v528_enabled) and any(
                    local_name.startswith(prefix)
                    for prefix in forbidden_prefixes
                ):
                    leaked_outcome.append(name)
        if not active:
            raise RuntimeError("V519 M2-only contract found no composer parameters.")
        forbidden = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(allowed_prefix)
        ]
        if forbidden:
            raise RuntimeError(
                "V519 M2-only contract leaked trainable tensors: "
                + str(forbidden[:30])
            )
        if leaked_outcome:
            version = "V530" if v530_enabled else ("V529" if v529_enabled else "V528")
            raise RuntimeError(
                f"{version} outcome-composer contract leaked obsolete heads: "
                + str(leaked_outcome[:30])
            )
        logger.info(
            "[V519_M2_ONLY] frozen Base/PVL/M1/M3 | "
            "trainable_m2_tensors=%d trainable_m2_parameters=%.3fM",
            len(active),
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
        if v530_enabled:
            logger.info(
                "[V530_PROBABILITY_CALIBRATED] strict full-resolution outcome whitelist active | "
                "v528_v529_ranker_case_prompt_heads_frozen=true"
            )
        elif v529_enabled:
            logger.info(
                "[V529_CALIBRATION_FIRST] strict dense-outcome whitelist active | "
                "legacy_ranker_v528_case_prompt_heads_frozen=true"
            )
        elif v528_enabled:
            logger.info(
                "[V528_OUTCOME_ONLY] strict outcome whitelist active | "
                "obsolete_ranker_case_prompt_heads_frozen=true"
            )
        logger.info("[V519_M2_ONLY] M2 preview=%s", active[:30])
        return

    if _v490_is_end_to_end(cfg):
        frozen_prefixes = (
            "vision_model.",
            "text_model.",
            "_v396_observer_vision.",
            # Inactive historical task paths; V490 uses v484_pipeline only.
            "m1_pse.",
            "ccv_m2.",
            "m2_text_verifier.",
            "m2_tide_repair_head.",
        )
        frozen_exact = {"logit_scale"}
        trainable_names = []
        frozen_names = []
        for name, parameter in model.named_parameters():
            freeze = name in frozen_exact or name.startswith(frozen_prefixes)
            parameter.requires_grad_(not freeze)
            (frozen_names if freeze else trainable_names).append(name)

        required_active_prefixes = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
            "v484_pipeline.error_state_head.",
            "v484_pipeline.local_generator.",
            "v484_pipeline.pixel_composer.",
            "v484_pipeline.safe_deployer.",
        )
        missing_active = [
            prefix for prefix in required_active_prefixes
            if not any(name.startswith(prefix) for name in trainable_names)
        ]
        if missing_active:
            raise RuntimeError(
                "V490 active end-to-end contract is incomplete; missing trainable "
                "prefixes: " + str(missing_active)
            )

        legacy_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                name.startswith((
                    "m1_pse.",
                    "ccv_m2.",
                    "m2_text_verifier.",
                    "m2_tide_repair_head.",
                    "_v396_observer_vision.",
                ))
                or name == "logit_scale"
            )
        ]
        if legacy_trainable:
            raise RuntimeError(
                "V490 inactive compatibility tensors are unexpectedly trainable: "
                + str(legacy_trainable[:30])
            )

        encoder_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (name.startswith(("vision_model.", "text_model.")) or name == "logit_scale")
        ]
        if encoder_trainable:
            raise RuntimeError(
                "V490 pretrained encoder freeze failed: "
                + str(encoder_trainable[:20])
            )

        logger.info(
            "[V490_ACTIVE_E2E] pretrained encoders + inactive legacy paths frozen | "
            "active_trainable_tensors=%d active_trainable_parameters=%.3fM | "
            "frozen_tensors=%d",
            len(trainable_names),
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
            len(frozen_names),
        )
        logger.info("[V490_ACTIVE_E2E] trainable preview=%s", trainable_names[:40])
        logger.info("[V490_ACTIVE_E2E] inactive/frozen preview=%s", frozen_names[:40])
        return

    if _v489_is_end_to_end(cfg):
        frozen_encoder_prefixes = ("vision_model.", "text_model.")
        trainable_names = []
        frozen_names = []
        for name, parameter in model.named_parameters():
            freeze = name.startswith(frozen_encoder_prefixes)
            parameter.requires_grad_(not freeze)
            (frozen_names if freeze else trainable_names).append(name)
        if not any(name.startswith("v484_pipeline.error_state_head.") for name in trainable_names):
            raise RuntimeError("V489 contract failed: M1 error-state head is not trainable.")
        if not any(name.startswith("v484_pipeline.pixel_composer.") for name in trainable_names):
            raise RuntimeError("V489 contract failed: M2 composer is not trainable.")
        if not any(name.startswith("v484_pipeline.safe_deployer.") for name in trainable_names):
            raise RuntimeError("V489 contract failed: M3 selector is not trainable.")
        unexpected_frozen = [
            name for name, parameter in model.named_parameters()
            if not parameter.requires_grad and not name.startswith(frozen_encoder_prefixes)
        ]
        if unexpected_frozen:
            raise RuntimeError(
                "V489 allows freezing only image/text encoders; unexpected frozen tensors: "
                + str(unexpected_frozen[:30])
            )
        encoder_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(frozen_encoder_prefixes)
        ]
        if encoder_trainable:
            raise RuntimeError("V489 encoder freeze failed: " + str(encoder_trainable[:20]))
        logger.info(
            "[V489_END_TO_END] encoder-only freeze | trainable_tensors=%d "
            "trainable_parameters=%.3fM | frozen_encoder_tensors=%d",
            len(trainable_names),
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
            len(frozen_names),
        )
        logger.info("[V489_END_TO_END] trainable preview=%s", trainable_names[:40])
        return

    if _v488_is_m2m3_only(cfg):
        # Strict experiment contract: neither Base nor any M1 tensor may update.
        allowed_prefixes = (
            "v484_pipeline.pixel_composer.",
            "v484_pipeline.safe_deployer.",
        )
        active = []
        for name, parameter in model.named_parameters():
            trainable = name.startswith(allowed_prefixes)
            parameter.requires_grad_(trainable)
            if trainable:
                active.append(name)
        has_m2 = any(name.startswith("v484_pipeline.pixel_composer.") for name in active)
        has_m3 = any(name.startswith("v484_pipeline.safe_deployer.") for name in active)
        if not has_m2 or not has_m3:
            raise RuntimeError(
                "V488_M2M3_ONLY contract failed: both pixel_composer and safe_deployer "
                f"must be trainable (has_m2={has_m2}, has_m3={has_m3})."
            )
        forbidden = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(allowed_prefixes)
        ]
        if forbidden:
            raise RuntimeError("V488 freeze contract violated: " + str(forbidden[:20]))
        logger.info(
            "[V488_M2M3_ONLY] Base and complete M1 frozen; trainable M2/M3 tensors=%d "
            "parameters=%d preview=%s",
            len(active),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            active[:30],
        )
        return

    # V469 treats B0 as an immutable counterfactual reference.  In that mode
    # the function deliberately does not follow the historical broad-keyword
    # E2E behaviour.
    if m1_enabled(cfg) and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V486_FIXED_BASE_PROBE", False)):
        # V486 diagnostic/proof mode: C0/Base is fixed and only the local
        # intervention pipeline learns.  This guarantees proposal loss cannot
        # update PVL, mask_head, upscale, legacy M1, M2 or M3.
        active = []
        frozen = []
        for name, parameter in model.named_parameters():
            trainable = name.startswith("v484_pipeline.")
            parameter.requires_grad_(trainable)
            if trainable:
                active.append(name)
            else:
                frozen.append(name)
        if not active:
            raise RuntimeError("V486_FIXED_BASE_PROBE enabled but v484_pipeline has no trainable parameters.")
        forbidden = [
            name for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("v484_pipeline.")
        ]
        if forbidden:
            raise RuntimeError("V486 fixed-base contract violated; non-V486 tensors trainable: " + str(forbidden[:20]))
        logger.info(
            "[V486_FIXED_BASE_PROBE] Base/PVL/mask/upscale/legacy M1/M2/M3 frozen; "
            "trainable_v484_tensors=%d parameters=%d preview=%s",
            len(active),
            sum(p.numel() for n, p in model.named_parameters() if n.startswith("v484_pipeline.") and p.requires_grad),
            active[:20],
        )
        return

    if _is_v487_base_safe_e2e(cfg):
        active = []
        frozen = []
        for name, parameter in model.named_parameters():
            trainable = name.startswith((
                "pvl_adapters.",
                "mask_head.",
                "upscale.",
                "v484_pipeline.",
            ))
            # Explicitly exclude legacy M1/M2/M3 from the V487 M1-local repair
            # run.  V487 still trains Base end-to-end through base_loss, but the
            # proposal loss can only train v484_pipeline.
            if name.startswith((
                "m1_pse.",
                "ccv_m2.",
                "m2_text_verifier.",
                "m2_tide_repair_head.",
                "vision_model.",
                "text_model.",
            )) or name == "logit_scale":
                trainable = False
            parameter.requires_grad_(trainable)
            if trainable:
                active.append(name)
            else:
                frozen.append(name)
        has_base = any(name.startswith(("pvl_adapters.", "mask_head.", "upscale.")) for name in active)
        has_m1 = any(name.startswith("v484_pipeline.") for name in active)
        if not has_base or not has_m1:
            raise RuntimeError(
                "V487_BASE_SAFE_E2E contract failed: requires trainable Base/PVL and v484_pipeline. "
                f"has_base={has_base} has_v484={has_m1}"
            )
        logger.info(
            "[V487_BASE_SAFE_E2E] trainable Base via base_loss + trainable v484_pipeline via proposal_loss | tensors=%d parameters=%d preview=%s",
            len(active),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            active[:30],
        )
        logger.info(
            "[V487_BASE_SAFE_E2E] proposal-gradient guard enabled: protected prefixes=%s",
            _v487_protected_base_prefixes(),
        )
        return

    if bool(_cfg_get(cfg.MODEL, "FREEZE_IMAGE_TEXT_ENCODERS", False)):
        # V480 fair-ablation contract:
        #
        # A0 Base-only:
        #   - external image/text encoders remain frozen;
        #   - PVL adapters and segmentation Base remain trainable;
        #   - M1/M2/M3 are absent or explicitly frozen.
        #
        # M1-enabled experiments:
        #   - preserve the original V479 trainable scope unchanged.
        base_only = not m1_enabled(cfg)

        disabled_task_prefixes = (
            "m1_pse.",
            "ccv_m2.",
            "m2_text_verifier.",
            "m2_tide_repair_head.",
        )

        active = []
        frozen_image = []
        frozen_text = []
        frozen_ablation = []

        for name, parameter in model.named_parameters():
            if name.startswith("vision_model."):
                parameter.requires_grad_(False)
                frozen_image.append(name)

            elif (
                name.startswith("text_model.")
                or name == "logit_scale"
            ):
                parameter.requires_grad_(False)
                frozen_text.append(name)

            elif name.startswith("_v396_observer_vision."):
                parameter.requires_grad_(False)

            elif (
                base_only
                and name.startswith(
                    disabled_task_prefixes
                )
            ):
                parameter.requires_grad_(False)
                frozen_ablation.append(name)

            else:
                parameter.requires_grad_(True)
                active.append(name)

        groups = {
            "pvl": [
                name
                for name in active
                if name.startswith("pvl_adapters.")
            ],
            "base": [
                name
                for name in active
                if name.startswith(
                    ("mask_head.", "upscale.")
                )
            ],
            "m1": [
                name
                for name in active
                if name.startswith(("m1_pse.", "v484_pipeline."))
            ],
            "m2m3": [
                name
                for name in active
                if name.startswith(
                    (
                        "m1_pse.m2_counterfactual.",
                        "m1_pse.m3_policy.",
                        "m1_pse.candidate_embedding.",
                        "m1_pse.failure_head.",
                        "ccv_m2.",
                        "m2_text_verifier.",
                        "m2_tide_repair_head.",
                    )
                )
            ],
        }

        # Both A0 and M1-enabled experiments require the same
        # trainable segmentation path.
        if not groups["pvl"] or not groups["base"]:
            raise RuntimeError(
                "Frozen-encoder contract failed: "
                "PVL and Base must remain trainable."
            )

        # Only M1-enabled experiments require trainable M1.
        if (
            not base_only
            and not groups["m1"]
        ):
            raise RuntimeError(
                "V479 frozen-encoder contract failed: "
                "an M1-enabled run has no trainable "
                "M1 tensors."
            )

        # A0 must not silently train any candidate or selector
        # parameter.
        if (
            base_only
            and (
                groups["m1"]
                or groups["m2m3"]
            )
        ):
            raise RuntimeError(
                "V480 A0 Base-only contract failed: "
                "M1/M2/M3 tensors are trainable."
            )

        contract = (
            "V480_A0_BASE_ONLY"
            if base_only
            else "V479_FROZEN_ENCODERS"
        )

        logger.info(
            "[%s] image=%d text=%d frozen | "
            "task tensors=%d | parameters=%d",
            contract,
            len(frozen_image),
            len(frozen_text),
            len(active),
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )

        logger.info(
            "[%s] pvl=%d base=%d m1=%d "
            "m2m3=%d ablation_frozen=%d | "
            "preview=%s",
            contract,
            len(groups["pvl"]),
            len(groups["base"]),
            len(groups["m1"]),
            len(groups["m2m3"]),
            len(frozen_ablation),
            active[:30],
        )

        if str(
            getattr(
                cfg,
                "init_checkpoint",
                "",
            )
            or ""
        ).strip():
            raise RuntimeError(
                "V479/V480 starts from the public "
                "pretrained encoders and forbids a "
                "task-trained checkpoint."
            )

        return
    if bool(_cfg_get(cfg.M1, "V478_FULL_END_TO_END", False)):
        active = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)
            active.append(name)
        logger.info(
            "[V478_FULL_END_TO_END] all named parameters trainable | tensors=%d | parameters=%d",
            len(active),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        logger.info("[V478_FULL_END_TO_END] preview=%s", active[:30])
        return
    if not m1_enabled(cfg):
        return

    mode = m1_train_mode(cfg)
    if mode != "e2e":
        return

    if bool(_cfg_get(cfg.M1, "M1_ONLY_CANDIDATE_PRETRAIN", False)):
        return

    strict_joint = bool(_cfg_get(cfg.M1, "V470_STRICT_JOINT_E2E", False))
    if strict_joint:
        # Fair task-level E2E contract: only external pretrained encoders stay
        # frozen.  No task-trained Base/B0 checkpoint or segmentation branch is
        # frozen.  Obsolete compatibility heads that do not participate in the
        # V470 forward/loss are kept fixed to avoid counting unused parameters.
        base_prefixes = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
        )
        last_adapter_count = int(
            _cfg_get(cfg.M1, "V475_TRAIN_LAST_ADAPTERS", 0)
        )
        total_adapter_count = len(getattr(model, "pvl_adapters", []))
        if last_adapter_count > 0 and total_adapter_count > 0:
            last_adapter_count = min(last_adapter_count, total_adapter_count)
            first_trainable_adapter = total_adapter_count - last_adapter_count
        else:
            first_trainable_adapter = 0
        candidate_mode = str(
            _cfg_get(cfg.M1, "CANDIDATE_MODE", "") or ""
        ).strip().lower()
        cem_mode = candidate_mode in {
            "compositional_error_modes",
            "cem_candidates",
            "v474_cem",
        }

        # V474 CEM is a self-contained candidate generator.  Every tensor
        # under m1_pse participates in its forward/loss graph and must remain
        # trainable.  The historical allow-list below only covers the old
        # action-bank modules and would silently freeze image/text/state fusion,
        # mode queries, signed-delta heads, quality prediction and failure gate.
        if cem_mode:
            m1_prefixes = ("m1_pse.",)
        else:
            m1_prefixes = (
                "m1_pse.trunk.",
                "m1_pse.semantic_proj.",
                "m1_pse.actionness_heads.",
                "m1_pse.delta_heads.",
                "m1_pse.v463_residual_head.",
            )
        m2_prefixes = ("ccv_m2.",)

        base_names, m1_names, m2_names = [], [], []
        frozen_early_adapter_names = []
        for name, parameter in model.named_parameters():
            is_base = name.startswith(base_prefixes)
            if name.startswith("pvl_adapters.") and total_adapter_count > 0:
                try:
                    adapter_index = int(name.split(".", 2)[1])
                except (IndexError, ValueError):
                    adapter_index = total_adapter_count
                if adapter_index < first_trainable_adapter:
                    is_base = False
                    frozen_early_adapter_names.append(name)
            is_m1 = name.startswith(m1_prefixes)
            is_m2 = name.startswith(m2_prefixes)
            trainable = is_base or is_m1 or is_m2
            parameter.requires_grad_(trainable)
            if is_base:
                base_names.append(name)
            elif is_m1:
                m1_names.append(name)
            elif is_m2:
                m2_names.append(name)

        if cem_mode:
            all_cem_names = [
                name for name, _ in model.named_parameters()
                if name.startswith("m1_pse.")
            ]
            frozen_cem_names = [
                name for name, parameter in model.named_parameters()
                if name.startswith("m1_pse.") and not parameter.requires_grad
            ]
            required_cem_prefixes = (
                "m1_pse.image_stem.",
                "m1_pse.semantic_proj.",
                "m1_pse.text_proj.",
                "m1_pse.negative_text_proj.",
                "m1_pse.state_proj.",
                "m1_pse.fusion.",
                "m1_pse.mode_queries.",
                "m1_pse.mode_film.",
                "m1_pse.mode_trunk.",
                "m1_pse.attention_head.",
                "m1_pse.delta_head.",
                "m1_pse.mode_gate.",
                "m1_pse.quality_head.",
                "m1_pse.failure_head.",
            )
            missing_cem_prefixes = [
                prefix for prefix in required_cem_prefixes
                if not any(name.startswith(prefix) for name in all_cem_names)
            ]
            if missing_cem_prefixes:
                raise RuntimeError(
                    "V474 CEM construction is incomplete; missing parameter "
                    "prefixes: " + str(missing_cem_prefixes)
                )
            if frozen_cem_names:
                raise RuntimeError(
                    "V474 CEM trainable contract violated; frozen tensors: "
                    + str(frozen_cem_names[:40])
                )

        if not base_names:
            raise RuntimeError(
                "V470 strict joint E2E has no trainable segmentation/Base tensors."
            )
        if not m1_names:
            raise RuntimeError(
                "V470 strict joint E2E has no trainable candidate-generator tensors."
            )
        if bool(_cfg_get(cfg.M1, "V463_CCV_ENABLED", False)) and not m2_names:
            raise RuntimeError(
                "V470 CCV is enabled but no ccv_m2 tensors are trainable."
            )

        logger.info(
            "[V470_JOINT_E2E] base_trainable_tensors=%d | "
            "m1_trainable_tensors=%d | m2_trainable_tensors=%d | "
            "task_checkpoint_loaded=%s",
            len(base_names),
            len(m1_names),
            len(m2_names),
            bool(str(getattr(cfg, "init_checkpoint", "") or "").strip()),
        )
        logger.info(
            "[V470_JOINT_E2E] base_preview=%s | m1_preview=%s | m2_preview=%s",
            base_names[:20], m1_names[:20], m2_names[:20],
        )
        if frozen_early_adapter_names:
            logger.info(
                "[V475_TAIL_ROBUST] frozen_early_adapter_tensors=%d | "
                "train_last_adapters=%d/%d",
                len(frozen_early_adapter_names),
                last_adapter_count,
                total_adapter_count,
            )
        if str(getattr(cfg, "init_checkpoint", "") or "").strip():
            raise RuntimeError(
                "V470 fair-from-one-run protocol forbids a task-trained "
                "M1.INIT_CHECKPOINT/--init-checkpoint."
            )
        return

    freeze_base = bool(_cfg_get(cfg.M1, "V469_FREEZE_BASE", False))
    if freeze_base:
        active = []
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith("m1_pse.")
                or name.startswith("ccv_m2.")
            )
            # Historical identity observer remains fixed even inside M1.
            if name == "m2_text_verifier.patch_adapter.weight":
                trainable = False
            parameter.requires_grad_(trainable)
            if trainable:
                active.append(name)

        m1_names = [name for name in active if name.startswith("m1_pse.")]
        m2_names = [name for name in active if name.startswith("ccv_m2.")]
        if not m1_names:
            raise RuntimeError(
                "V469 frozen-reference run has no trainable M1 parameters."
            )
        if bool(_cfg_get(cfg.M1, "V463_CCV_ENABLED", False)) and not m2_names:
            raise RuntimeError(
                "V469 CCV is enabled but no ccv_m2 parameters are trainable."
            )

        logger.info(
            "[V469_FROZEN_REFERENCE] base_trainable_tensors=0 | "
            "m1_trainable_tensors=%d | m2_trainable_tensors=%d",
            len(m1_names),
            len(m2_names),
        )
        logger.info(
            "[V469_FROZEN_REFERENCE] m1_preview=%s | m2_preview=%s",
            m1_names[:20],
            m2_names[:20],
        )
        return

    blocked_prefixes = (
        "text_model.",
        "_v396_observer_vision.",
    )
    full_vision = bool(_cfg_get(cfg.M1, "V395_FULL_VISION_E2E", False))
    base_keywords = (
        "pvl_adapters",
        "mask_head",
        "upscale",
        "fusion",
        "fuse",
        "decoder",
        "seg",
        "refine",
        "adapter",
        "projection",
        "proj",
        "gate",
        "head",
    )

    active = []
    for name, parameter in model.named_parameters():
        if name.startswith(blocked_prefixes):
            parameter.requires_grad_(False)
            continue
        if name.startswith("vision_model.") and not full_vision:
            parameter.requires_grad_(False)
            continue
        if name == "m2_text_verifier.patch_adapter.weight":
            parameter.requires_grad_(False)
            continue

        is_m1 = (
            name.startswith("m1_pse.")
            or name.startswith("m2_text_verifier.")
            or name.startswith("m2_tide_repair_head.")
            or name.startswith("ccv_m2.")
        )
        is_base_head = any(key in name for key in base_keywords)
        if is_m1 or is_base_head:
            parameter.requires_grad_(True)
        if parameter.requires_grad:
            active.append(name)

    base_active = [
        name for name in active
        if not (
            name.startswith("m1_pse.")
            or name.startswith("m2_text_verifier.")
            or name.startswith("m2_tide_repair_head.")
            or name.startswith("ccv_m2.")
        )
    ]
    m1_active_names = [
        name for name in active
        if (
            name.startswith("m1_pse.")
            or name.startswith("m2_text_verifier.")
            or name.startswith("m2_tide_repair_head.")
            or name.startswith("ccv_m2.")
        )
    ]
    if not base_active:
        preview = [name for name, _ in model.named_parameters()][:80]
        raise RuntimeError(
            "Unified E2E requested, but no B0-head/base parameters were made "
            "trainable. First parameter names preview: " + str(preview)
        )
    if not m1_active_names:
        raise RuntimeError(
            "Unified E2E requested, but no M1 parameters were made trainable."
        )
    logger.info(
        "[FORCE_UNIFIED_E2E] base_head_trainable_tensors=%d | "
        "m1_trainable_tensors=%d",
        len(base_active),
        len(m1_active_names),
    )
    logger.info("[FORCE_UNIFIED_E2E] base_head_preview=%s", base_active[:30])

def _enforce_v552_geometry_owner_trainability(model, cfg, logger=None):
    """Re-apply the R4.13/R4.14/R4.15 single-geometry-owner invariant.

    Some historical trainability contracts intentionally turn every non-encoder
    task parameter back on.  That would silently re-enable obsolete dense-size
    and iterative-anchor heads after the component generator froze them.  This
    post-contract guard runs immediately before optimizer construction so those
    legacy geometry parameters cannot re-enter the optimizer.
    """
    enabled = bool(_cfg_get(cfg.M1, "V552R413_ROOTFIX_ENABLED", False))
    if not enabled:
        return []
    forbidden = (
        "component_slot_generator.r411_size_head.",
        "component_slot_generator.r48_anchor_heads.",
    )
    frozen = []
    extent_trainable = []
    for name, parameter in model.named_parameters():
        if any(token in name for token in forbidden):
            parameter.requires_grad_(False)
            frozen.append(name)
        if "component_slot_generator.r413_extent_head." in name and parameter.requires_grad:
            extent_trainable.append(name)
    if not extent_trainable:
        raise RuntimeError(
            "V552-R4.13/R4.14/R4.15 geometry-owner contract failed: contextual/query extent head is not trainable"
        )
    if logger is not None:
        logger.info(
            "[V552_GEOMETRY_OWNER_TRAINABILITY] legacy_geometry_frozen=%d extent_trainable=%d",
            len(frozen), len(extent_trainable),
        )
    return frozen


def _enforce_v552r4201_clean_trainability(model, cfg, logger=None):
    """Remove historical heads from the active R4.20.1 optimizer graph.

    ``_force_unified_e2e_trainable`` deliberately re-enables the whole
    ``pixel_composer`` namespace for legacy unified runs.  That broad rule is
    incompatible with the R4.18+ box-free contract because it silently turns
    old box/ROI/iterative-decoder parameters back on after the component
    generator has frozen them.  R4.20.1 is a clean protocol, so re-apply a
    *current-path* ownership contract immediately before optimizer creation.

    This guard does not freeze Base/PVL or the active M1/M2 path.  It removes
    only modules that are provably not consumers/owners in the clean dynamic
    residual-mask graph.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V552R4201_ROOTFIX_ENABLED", False)):
        return []

    # Historical geometry / ROI / iterative-decoder owners.  In R4.20.1 the
    # physical R4.17 peak is the sole center, and R4.20 dynamic convs own mask
    # realization.  DN is off, so the R4.8 decoder stack has no forward user.
    forbidden_tokens = (
        "component_slot_generator.r411_size_head.",
        "component_slot_generator.r411_offset_head.",
        "component_slot_generator.r417_location_offset_head.",
        "component_slot_generator.r411_box_query_mlp.",
        "component_slot_generator.r411_roi_mask_head.",
        "component_slot_generator.r413_extent_head.",
        "component_slot_generator.r48_anchor_heads.",
        "component_slot_generator.r48_offset_heads.",
        "component_slot_generator.r48_context_projs.",
        "component_slot_generator.r48_query_norm1.",
        "component_slot_generator.r48_query_norm2.",
        "component_slot_generator.r48_ffns.",
        "component_slot_generator.r48_mask_query_proj.",
        "component_slot_generator.r48_dn_query_encoder.",
        "component_slot_generator.r47_base_anchor_logits",
        # R4.20 overwrites the historical multiscale parent mask before any
        # downstream use.  Keep pyramid feature routing/fusion trainable, but
        # remove the obsolete parent mask renderers themselves.
        "component_slot_generator.mask_encoder.",
        "component_slot_generator.mask_head.",
        "component_slot_generator.scale_mask_heads.",
    )
    r4203_enabled = bool(_cfg_get(m1, "V552R4203_ROOTFIX_ENABLED", False))
    r4204_enabled = r4203_enabled and bool(_cfg_get(m1, "V552R4204_ROOTFIX_ENABLED", False))
    if bool(_cfg_get(m1, "V552R420_TYPE_DECOUPLED_MASK_ENABLED", False)) or r4203_enabled:
        forbidden_tokens = forbidden_tokens + (
            "component_slot_generator.r411_type_embedding.",
        )
    if r4203_enabled:
        # Dense competitive set prediction replaces the point-conditioned
        # Dynamic Mask owner completely.  If a stale checkpoint/config ever
        # reintroduces those tensors, fail closed at optimizer construction.
        forbidden_tokens = forbidden_tokens + (
            "component_slot_generator.r420_mask_feature_proj.",
            "component_slot_generator.r420_dynamic_controller.",
        )

    frozen = []
    for name, parameter in model.named_parameters():
        if any(token in name for token in forbidden_tokens):
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                parameter.grad = None
            frozen.append(name)

    required_active = (
        "component_slot_generator.r47_pixel_encoder.",
        "component_slot_generator.r47_slot_queries.",
        "component_slot_generator.r411_proposal_stem.",
        "component_slot_generator.r411_center_head.",
        "component_slot_generator.r417_location_head.",
    )
    if not r4203_enabled:
        required_active = required_active + (
            "component_slot_generator.r420_mask_feature_proj.",
            "component_slot_generator.r420_dynamic_controller.",
        )
    if r4204_enabled:
        required_active = required_active + (
            "component_slot_generator.r4204_residual_occupancy_head.",
        )
    if bool(_cfg_get(m1, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)):
        required_active = required_active + (
            "component_slot_generator.r4210_overflow_gate_head.",
        )
    active_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    missing = [
        token for token in required_active
        if not any(token in name for name in active_names)
    ]
    if missing:
        raise RuntimeError(
            "V552-R4.20.1 clean trainability contract lost active modules: "
            + str(missing)
        )

    leaks = [
        name for name in active_names
        if any(token in name for token in forbidden_tokens)
    ]
    if leaks:
        raise RuntimeError(
            "V552-R4.20.1 clean trainability contract leaked obsolete heads: "
            + str(leaks[:50])
        )

    if logger is not None:
        logger.info(
            "[V552R4201_CLEAN_TRAINABILITY] obsolete_tensors_removed=%d "
            "active_component_tensors=%d",
            len(frozen),
            sum("component_slot_generator." in name for name in active_names),
        )
    return frozen


def _make_optimizer_and_scheduler(model, cfg):
    jbt_v6_dual_optimizer = False
    jbt_v6_aux_optimizer = None
    base_params = []
    pvl_params = []
    vision_params = []
    text_params = []
    m1_params = []
    m2_params = []
    m3_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("v484_pipeline.safe_deployer.") or name.startswith("m1_pse.m3_policy."):
            m3_params.append(parameter)
        elif name.startswith(
            "v484_pipeline.pixel_composer.component_slot_generator."
        ):
            if _v541_is_component_m2_parameter(name):
                m2_params.append(parameter)
            else:
                m1_params.append(parameter)
        elif (
            name.startswith("v484_pipeline.pixel_composer.")
            or name.startswith("m2_tide_repair_head.")
            or name.startswith("m1_pse.m2_tide_repair_head.")
            or name.startswith("ccv_m2.")
            or name.startswith("m1_pse.m2_counterfactual.")
            or name.startswith("m1_pse.candidate_embedding.")
            or name == "m1_pse.candidate_reliability_bias"
            or name.startswith("m1_pse.failure_head.")
            or name.startswith("m1_pse.v4g_refiner.")
        ):
            if _clean_dynamic_component_set(cfg) and name.startswith("v484_pipeline.pixel_composer."):
                m1_params.append(parameter)
            else:
                m2_params.append(parameter)
        elif "m1_pse" in name or "m2_text_verifier" in name or name.startswith("v484_pipeline."):
            m1_params.append(parameter)
        elif name.startswith("pvl_adapters."):
            pvl_params.append(parameter)
        elif name.startswith("vision_model."):
            vision_params.append(parameter)
        elif name.startswith("text_model."):
            text_params.append(parameter)
        else:
            base_params.append(parameter)
    if not any((base_params, pvl_params, vision_params, text_params, m1_params, m2_params, m3_params)):
        raise RuntimeError("No trainable parameters found.")

    pure_b0 = not m1_enabled(cfg)
    official_contract = bool(
        _cfg_get(_cfg_get(cfg, "M1", None), "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False)
    )
    if official_contract:
        # JBT-v6 exact public-Base contract.  The released code uses
        #   Adam(filter(trainable Base params), lr=3e-4)
        # with a single parameter list and default Adam execution policy.
        # v5 forced foreach=False and, in joint mode, put Base and JBT in the
        # same optimizer.  Both are unnecessary deviations.  v6 keeps Base in
        # its own exact Adam and gives JBT an independent Adam facade.
        official_base_params = []
        official_aux_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("m1_pse."):
                official_aux_params.append(parameter)
            else:
                official_base_params.append(parameter)
        if not official_base_params:
            raise RuntimeError("Official MedCLIPSeg contract has no trainable Base parameters.")
        if str(_cfg_get(cfg.TRAIN, "OPTIMIZER", "adam")).strip().lower() != "adam":
            raise ValueError("Official MedCLIPSeg Base contract requires Adam.")
        # Intentionally do not pass foreach/fused: this matches the public repo.
        # In v6.3.6 recovery the already-completed matched Base is the immutable
        # causal host, so its physical Adam has exactly zero learning rate.
        fixed_base_recovery = (
            os.environ.get("JBT_FIXED_BASE_RECOVERY", "0").strip() == "1"
        )
        official_base_lr = (
            0.0 if fixed_base_recovery else float(cfg.TRAIN.LEARNING_RATE)
        )
        base_optimizer = torch.optim.Adam(
            official_base_params,
            lr=official_base_lr,
        )
        base_optimizer.param_groups[0]["name"] = "official_base_exact"
        if fixed_base_recovery:
            print(
                "[JBT_V636_FIXED_BASE_OPTIMIZER] official_base_exact lr=0; "
                "only jbt_aux_exact can update parameters"
            )
        if m1_enabled(cfg):
            if not official_aux_params:
                raise RuntimeError("Official Base+JBT requires a non-empty M1 parameter set.")
            separate = bool(_cfg_get(cfg.M1, "JBT_V6_SEPARATE_BASE_OPTIMIZER", False))
            if separate:
                jbt_v6_aux_optimizer = torch.optim.Adam(
                    official_aux_params,
                    lr=float(_cfg_get(cfg.M1, "GEOTR_M1_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE)),
                )
                jbt_v6_aux_optimizer.param_groups[0]["name"] = "jbt_aux_exact"
                jbt_v6_dual_optimizer = True
            else:
                # Backward-compatible single-optimizer path for old configs.
                base_optimizer.add_param_group({
                    "params": official_aux_params,
                    "lr": float(_cfg_get(cfg.M1, "GEOTR_M1_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE)),
                    "name": "m1",
                })
        elif official_aux_params:
            raise RuntimeError("Official Base-only run unexpectedly has trainable M1 parameters.")
    elif pure_b0:
        if m1_params or m2_params or m3_params:
            raise RuntimeError("Pure B0 contract violated: M1/M2/M3 parameters are trainable.")
        b0_params = base_params + pvl_params + vision_params + text_params
        if not b0_params:
            raise RuntimeError("Pure B0 has no trainable parameters.")
        groups = [{"params": b0_params, "lr": float(cfg.TRAIN.LEARNING_RATE), "name": "b0_all"}]
        optimizer_name = str(_cfg_get(cfg.TRAIN, "OPTIMIZER", "adam")).strip().lower()
        if optimizer_name == "adamw":
            base_optimizer = torch.optim.AdamW(
                groups,
                weight_decay=float(_cfg_get(cfg.TRAIN, "WEIGHT_DECAY", 0.0)),
            )
        elif optimizer_name == "adam":
            base_optimizer = torch.optim.Adam(
                groups,
                weight_decay=float(_cfg_get(cfg.TRAIN, "WEIGHT_DECAY", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported TRAIN.OPTIMIZER={optimizer_name!r}; use 'adam' or 'adamw'.")
    else:
        if _semlt(cfg):
            if vision_params or text_params:
                raise RuntimeError("SemLT keeps the pretrained image/text encoders frozen.")
            causal_ablation = _geotr_m1_causal_ablation(cfg)
            fixed_base_refinement = bool(
                _cfg_get(cfg.M1, "SEMLT_FIXED_BASE_REFINEMENT", False)
            )
            if fixed_base_refinement:
                if base_params or pvl_params:
                    raise RuntimeError(
                        "Fixed-Base OACD leaked trainable Base/PVL parameters."
                    )
                if not m1_params:
                    raise RuntimeError(
                        "Fixed-Base OACD has no trainable M1 transport parameters."
                    )
            elif causal_ablation:
                if base_params or pvl_params:
                    raise RuntimeError(
                        "GEOTR-M1 causal ablation received trainable Base/PVL parameters."
                    )
                if not m1_params:
                    raise RuntimeError(
                        "GEOTR-M1 causal ablation has no trainable M1 transport parameters."
                    )
            elif not base_params or not pvl_params or not m1_params:
                raise RuntimeError("SemLT requires non-empty Base, PVL and M1 transport groups.")
            if m2_params or m3_params:
                raise RuntimeError("SemLT M1-only optimizer received M2/M3 parameters.")
            logical = {"base": [], "pvl": [], "semlt": []}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith("pvl_adapters."):
                    logical["pvl"].append((name, parameter))
                elif name.startswith("m1_pse."):
                    logical["semlt"].append((name, parameter))
                else:
                    logical["base"].append((name, parameter))
            base_lr = float(_cfg_get(cfg.M1, "BASE_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE))
            pvl_lr = float(_cfg_get(cfg.M1, "PVL_LEARNING_RATE", base_lr))
            semlt_lr = float(_cfg_get(
                cfg.M1,
                "GEOTR_M1_LEARNING_RATE",
                _cfg_get(cfg.M1, "SEMLT_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE),
            ))
            lrs = {"base": base_lr, "pvl": pvl_lr, "semlt": semlt_lr}
            weight_decay = float(_cfg_get(cfg.TRAIN, "WEIGHT_DECAY", 0.0))
            groups = []
            for logical_name in ("base", "pvl", "semlt"):
                decay, nodecay = [], []
                for name, parameter in logical[logical_name]:
                    (nodecay if _mhcs_no_decay(name, parameter) else decay).append(parameter)
                if decay:
                    groups.append({
                        "params": decay, "lr": lrs[logical_name],
                        "weight_decay": weight_decay, "name": logical_name + "_decay",
                    })
                if nodecay:
                    groups.append({
                        "params": nodecay, "lr": lrs[logical_name],
                        "weight_decay": 0.0, "name": logical_name + "_nodecay",
                    })
            optimizer_name = str(_cfg_get(cfg.TRAIN, "OPTIMIZER", "adamw")).strip().lower()
            optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
            if optimizer_cls is None:
                raise ValueError(f"Unsupported TRAIN.OPTIMIZER={optimizer_name!r}; use 'adam' or 'adamw'.")
            strict_parity = bool(
                _cfg_get(cfg.M1, "SEMLT_STRICT_CAUSAL_PARITY", False)
            )
            optimizer_kwargs = {"weight_decay": 0.0}
            if strict_parity:
                # Keep optimizer execution single-tensor and explicit while
                # diagnosing exact cross-variant Base/PVL parity.  Parameter
                # groups and mathematical Adam updates remain unchanged.
                optimizer_kwargs.update({"foreach": False, "fused": False})
            base_optimizer = optimizer_cls(groups, **optimizer_kwargs)
            exact_m1 = str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "geotr_m1"
            if fixed_base_refinement:
                print(
                    "[FIXED-BASE OACD OPTIMIZER] Base/PVL frozen | "
                    f"Transport-only lr={semlt_lr:.2e} | M2/M3=absent"
                )
            elif causal_ablation:
                print(
                    "[GEOTR-M1 CAUSAL OPTIMIZER] common Base/PVL frozen | "
                    f"Transport-only lr={semlt_lr:.2e} | M2/M3=absent"
                )
            else:
                print(
                    f"[{'GEOTR-M1 EXACT' if exact_m1 else 'SemLT-M1'} OPTIMIZER] "
                    "independent Base/PVL/Transport groups | "
                    f"lr(base/pvl/semlt)={base_lr:.2e}/{pvl_lr:.2e}/{semlt_lr:.2e} | M2/M3=absent"
                )
        elif _mhcs(cfg):
            # MHCS-R4.6 uses explicit Base/PVL/protected-M1-bank/M2-envelope ownership and no-decay
            # subgroups. This is one AdamW and one scheduler; the separation is
            # only parameter ownership, LR and gradient clipping.
            logical = {"base": [], "pvl": [], "mhcs_bank": [], "mhcs_m2": []}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                group_name = _mhcs_root_gradient_group(name)
                if group_name == "encoders":
                    raise RuntimeError(
                        f"MHCS-R4.6 requires frozen image/text encoders, but {name} is trainable."
                    )
                logical[group_name].append((name, parameter))
            if not logical["base"] or not logical["pvl"]:
                raise RuntimeError("MHCS-R4.6 requires trainable Base and PVL groups.")
            if not logical["mhcs_bank"] or not logical["mhcs_m2"]:
                raise RuntimeError("MHCS-R4.6 requires non-empty M1-bank and M2-envelope groups.")

            base_lr = float(_cfg_get(cfg.M1, "BASE_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE))
            pvl_lr = float(_cfg_get(cfg.M1, "V489_PVL_LEARNING_RATE", base_lr))
            bank_lr = float(_cfg_get(cfg.M1, "MHCS_BANK_LEARNING_RATE", base_lr))
            m2_lr = float(
                _cfg_get(
                    cfg.M1,
                    "MHCS_M2_LEARNING_RATE",
                    _cfg_get(
                        cfg.M1, "MHCS_ROUTER_LEARNING_RATE",
                        _cfg_get(cfg.M1, "CANDIDATE_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE),
                    ),
                )
            )
            lrs = {
                "base": base_lr,
                "pvl": pvl_lr,
                "mhcs_bank": bank_lr,
                "mhcs_m2": m2_lr,
            }
            wd = float(_cfg_get(cfg.TRAIN, "WEIGHT_DECAY", 0.0))
            groups = []
            for logical_name in ("base", "pvl", "mhcs_bank", "mhcs_m2"):
                decay, nodecay = [], []
                for name, parameter in logical[logical_name]:
                    (nodecay if _mhcs_no_decay(name, parameter) else decay).append(parameter)
                if decay:
                    groups.append({
                        "params": decay, "lr": lrs[logical_name],
                        "weight_decay": wd, "name": logical_name + "_decay",
                    })
                if nodecay:
                    groups.append({
                        "params": nodecay, "lr": lrs[logical_name],
                        "weight_decay": 0.0, "name": logical_name + "_nodecay",
                    })

            optimizer_name = str(_cfg_get(cfg.TRAIN, "OPTIMIZER", "adamw")).strip().lower()
            optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
            if optimizer_cls is None:
                raise ValueError(
                    f"Unsupported TRAIN.OPTIMIZER={optimizer_name!r}; use 'adam' or 'adamw'."
                )
            # Per-group decay is explicit above; global decay must stay zero.
            base_optimizer = optimizer_cls(groups, weight_decay=0.0)
            print(
                "[MHCS-R5.2 OPTIMIZER] independent Base/PVL/unchanged-bank/action-safety-router groups | "
                f"lr(base/pvl/bank/m2)={base_lr:.2e}/{pvl_lr:.2e}/{bank_lr:.2e}/{m2_lr:.2e} | "
                "bias/norm/log_vars no-decay"
            )
        else:
            groups = []
            strict_joint = bool(_cfg_get(cfg.M1, "V470_STRICT_JOINT_E2E", False))
            if strict_joint:
                if not (base_params or pvl_params):
                    raise RuntimeError("V470 requires a trainable Base/PVL parameter group.")
                if not m1_params:
                    raise RuntimeError("V470 requires a trainable M1 parameter group.")
                if bool(_cfg_get(cfg.M1, "V463_CCV_ENABLED", False)) and not m2_params:
                    raise RuntimeError("V470 requires a trainable M2/CCV parameter group.")
                if vision_params:
                    raise RuntimeError("V470 keeps the external vision encoder frozen.")
            freeze_base = bool(_cfg_get(cfg.M1, "V469_FREEZE_BASE", False))
            if freeze_base and (base_params or pvl_params or vision_params):
                raise RuntimeError("V469 frozen-reference contract violated before optimizer construction.")

            base_lr = float(_cfg_get(cfg.M1, "BASE_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE))
            pvl_lr = float(_cfg_get(cfg.M1, "V489_PVL_LEARNING_RATE", base_lr))
            # ── V503: Down-weight PVL LR to prevent M1/M2 gradient dominance ──
            # M1/M2 losses are 300× larger than base loss. With equal PVL LR,
            # PVL adapters (shared by base + aux modules) are optimized for
            # M1/M2 objectives, corrupting base segmentation quality.
            # Reference: Misra et al. ECCV 2016 — cross-stitch task decoupling.
            _v503_pvl_ratio = float(_cfg_get(cfg.M1, "V503_PVL_LR_RATIO", 1.0))
            if _v503_pvl_ratio < 1.0 and m1_enabled(cfg):
                pvl_lr = pvl_lr * _v503_pvl_ratio
                print(
                    f"[V503] PVL LR scaled by {_v503_pvl_ratio:.3f}:"
                    f" {pvl_lr:.2e} (prevents M1/M2 gradient dominance over shared PVL)"
                )
            m1_lr = float(_cfg_get(cfg.M1, "CANDIDATE_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE))
            m2_lr = float(_cfg_get(cfg.M1, "M2_LEARNING_RATE", cfg.TRAIN.LEARNING_RATE))
            m3_lr = float(_cfg_get(cfg.M1, "M3_LEARNING_RATE", m2_lr))

            if base_params:
                groups.append({"params": base_params, "lr": base_lr, "name": "base"})
            if pvl_params:
                groups.append({"params": pvl_params, "lr": pvl_lr, "name": "pvl"})
            if vision_params:
                groups.append({
                    "params": vision_params,
                    "lr": float(_cfg_get(cfg.M1, "V395_VISION_LEARNING_RATE", 1.0e-6)),
                    "name": "vision",
                })
            if text_params:
                groups.append({
                    "params": text_params,
                    "lr": float(_cfg_get(cfg.M1, "V478_TEXT_LEARNING_RATE", 1.0e-6)),
                    "name": "text",
                })
            if m1_params:
                groups.append({"params": m1_params, "lr": m1_lr, "name": "m1"})
            if m2_params:
                groups.append({"params": m2_params, "lr": m2_lr, "name": "m2"})
            if m3_params:
                groups.append({"params": m3_params, "lr": m3_lr, "name": "m3"})

            optimizer_name = str(_cfg_get(cfg.TRAIN, "OPTIMIZER", "adamw")).strip().lower()
            optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
            if optimizer_cls is None:
                raise ValueError(f"Unsupported TRAIN.OPTIMIZER={optimizer_name!r}; use 'adam' or 'adamw'.")
            base_optimizer = optimizer_cls(
                groups,
                weight_decay=float(_cfg_get(cfg.TRAIN, "WEIGHT_DECAY", 0.0)),
            )

    use_sam = bool(_cfg_get(cfg.TRAIN, "USE_SAM", False))
    if use_sam:
        if jbt_v6_dual_optimizer:
            raise RuntimeError("JBT-v6 exact dual optimizer requires TRAIN.USE_SAM=false")
        optimizer = SAM(
            base_optimizer,
            rho=float(_cfg_get(cfg.TRAIN, "SAM_RHO", 0.05)),
            adaptive=bool(_cfg_get(cfg.TRAIN, "SAM_ADAPTIVE", False)),
        )
    else:
        optimizer = (
            _JBTDualOptimizer(base_optimizer, jbt_v6_aux_optimizer)
            if jbt_v6_dual_optimizer else base_optimizer
        )

    use_ema = bool(_cfg_get(cfg.TRAIN, "USE_EMA", False))
    ema = ModelEMA(model, decay=float(_cfg_get(cfg.TRAIN, "EMA_DECAY", 0.999))) if use_ema else None

    min_ratio = float(_cfg_get(cfg.TRAIN, "MIN_LR_RATIO", 1.0 / 3.0))
    total_epochs = max(
        1,
        int(_cfg_get(cfg.TRAIN, "SCHEDULER_TOTAL_EPOCHS", cfg.TRAIN.NUM_EPOCHS)),
    )
    if bool(_cfg_get(cfg.TRAIN, "V507_STRICT_SCHEDULER_CONTRACT", False)):
        configured_epochs = int(cfg.TRAIN.NUM_EPOCHS)
        if total_epochs != configured_epochs:
            raise ValueError(
                "V507 scheduler contract failed: TRAIN.SCHEDULER_TOTAL_EPOCHS "
                f"must equal TRAIN.NUM_EPOCHS, got {total_epochs} vs "
                f"{configured_epochs}."
            )
    warmup_epochs = max(0, int(_cfg_get(cfg.TRAIN, "WARMUP_EPOCHS", 0)))

    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max(min_ratio, float(epoch + 1) / float(warmup_epochs))
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(cosine_epochs)))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # For the v6 dual path the protected Base scheduler must see only the
    # exact Base Adam.  The auxiliary scheduler is advanced in lockstep by a
    # small facade created below.
    scheduler_optimizer = base_optimizer if (use_sam or jbt_v6_dual_optimizer) else optimizer
    official_base_contract = bool(
        _cfg_get(
            _cfg_get(cfg, "M1", None),
            "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT",
            False,
        )
    )
    paper100_schedule = bool(
        official_base_contract
        or (
            _semlt(cfg)
            and str(_cfg_get(cfg.M1, "GEOTR_M1_FORMAL_PROTOCOL", "")).strip().lower()
            == "paper100"
        )
    )
    v537_fresh_m2 = bool(
        m1_enabled(cfg)
        and bool(_cfg_get(cfg.M1, "V537_COMPONENT_UTILITY_RANKER_ENABLED", False))
        and bool(_cfg_get(cfg.M1, "V537_M2_FRESH_SCHEDULE", True))
    )
    fixed_base_recovery = (
        os.environ.get("JBT_FIXED_BASE_RECOVERY", "0").strip() == "1"
    )
    base_trajectory_lock = bool(
        m1_enabled(cfg)
        and bool(_cfg_get(cfg.M1, "BASE_TRAJECTORY_LOCK", False))
    ) or fixed_base_recovery
    if paper100_schedule:
        if warmup_epochs != 0 or abs(min_ratio) > 1.0e-12:
            raise ValueError(
                "paper100 scheduler requires WARMUP_EPOCHS=0 and MIN_LR_RATIO=0"
            )
        # Public MedCLIPSeg uses CosineAnnealingLR(T_max=100, eta_min=1e-4).
        # The previous R3 paper100 path incorrectly annealed to zero.
        base_scheduler_exact = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer,
            T_max=total_epochs,
            eta_min=(0.0 if base_trajectory_lock else (1.0e-4 if official_base_contract else 0.0)),
        )
        if jbt_v6_dual_optimizer:
            aux_eta_min = float(_cfg_get(cfg.M1, "JBT_V6_AUX_ETA_MIN", 1.0e-4))
            aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                jbt_v6_aux_optimizer, T_max=total_epochs, eta_min=aux_eta_min
            )
            scheduler = _JBTDualScheduler(base_scheduler_exact, aux_scheduler)
        else:
            scheduler = base_scheduler_exact
    elif v537_fresh_m2:
        ranker_start = _v536_deploy_start_epoch(cfg)
        ranker_epochs = max(1, total_epochs - ranker_start)
        ranker_min_ratio = float(_cfg_get(cfg.M1, "V537_M2_MIN_LR_RATIO", min_ratio))

        def m2_schedule(epoch):
            if epoch < ranker_start:
                return 0.0
            progress = min(1.0, max(0.0, float(epoch - ranker_start) / float(ranker_epochs)))
            return ranker_min_ratio + (1.0 - ranker_min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        lr_lambdas = [
            (m2_schedule if group.get("name") == "m2" else schedule)
            for group in scheduler_optimizer.param_groups
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            scheduler_optimizer,
            lr_lambda=lr_lambdas,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            scheduler_optimizer,
            lr_lambda=schedule,
        )
    return (
        optimizer,
        scheduler,
        ema,
        len(base_params),
        len(pvl_params),
        len(m1_params),
        len(m2_params),
        len(m3_params),
    )

def _checkpoint_state(model, optimizer, scheduler, epoch, best_dice, best_fusion, best_oracle, run_name, cfg, phase_b_started=False, ema=None, weight_source="raw", hard_case_memory=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "best_fusion_dice": best_fusion,
        "best_oracle_dice": best_oracle,
        "run_name": run_name,
        "m1_train_mode": m1_train_mode(cfg) if m1_enabled(cfg) else "disabled",
        "phase_b_started": bool(phase_b_started),
        "weight_source": weight_source,
        "m1_contract": (
            "V20 joint contract: atomic action pool, matched local counterfactual text verification, and sparse set selection are trained together. "
            "No phase freeze/reload; inference remains GT/Oracle/test-label-free."
        ),
    }
    if ema is not None:
        state["ema_shadow"] = {k: v.clone() for k, v in ema.shadow.items()}
    if hard_case_memory is not None:
        state["v547_hard_case_memory"] = hard_case_memory.state_dict()
    routing_state = getattr(cfg, "_v548_m2_routing_state", None)
    if isinstance(routing_state, dict):
        state["v548_m2_routing_state"] = dict(routing_state)
    return state



def _validate_v550_protocol(cfg):
    """Fail fast on stale-replay or deployed-utility supervision regressions."""
    m1 = _cfg_get(cfg, "M1", None)
    protocol = str(_cfg_get(m1, "V550_PROTOCOL_VERSION", "")).strip()
    if not protocol:
        return
    errors = []
    # V552-R4 intentionally replaces the legacy V549 factorized deployment
    # intersection with one unified safety gate.  The old V550 validator used
    # to require V549_FACTORIZED_DEPLOYMENT_ENABLED=true unconditionally,
    # while the R4 protocol correctly requires it to be false.  Dispatch the
    # gate requirement by protocol version instead of imposing both contracts.
    r4_enabled = bool(
        _cfg_get(m1, "V552R4_UNIFIED_REFERENCE_CONTRACT_ENABLED", False)
    ) or str(
        _cfg_get(m1, "V552_PROTOCOL_VERSION", "")
    ).strip() in {
        "unified_reference_contract_v5",
        "decoupled_critic_contract_v6",
        "spatial_evidence_action_aware_contract_v7",
    }

    required_true = (
        "V547_FACTORIZED_OUTCOME_ENABLED",
        "V548_FACTORIZED_DIRECTION_ZERO_INIT",
        "V548_EXPLICIT_SIGN_LOSS_ENABLED",
        "V548_STABLE_M2_ROUTING_ENABLED",
        "V549_EPOCH_QUALITY_GATE_ENABLED",
        "V550_CURRENT_SEMANTIC_BALANCE_ENABLED",
        "V550_CANDIDATE_QUALITY_ROUTING_ENABLED",
        "V544_MINIMAL_M2_OBJECTIVE_ENABLED",
        "V544_USE_GAIN_AS_DECISION_SCORE",
    )
    if not r4_enabled:
        required_true = required_true + (
            "V549_FACTORIZED_DEPLOYMENT_ENABLED",
        )

    for key in required_true:
        if not bool(_cfg_get(m1, key, False)):
            errors.append(f"M1.{key} must be true")

    if r4_enabled and bool(
        _cfg_get(m1, "V549_FACTORIZED_DEPLOYMENT_ENABLED", True)
    ):
        errors.append(
            "M1.V549_FACTORIZED_DEPLOYMENT_ENABLED must be false for "
            "V552-R4 unified deployment"
        )
    if bool(_cfg_get(m1, "V549_CROSS_BATCH_BALANCED_QUEUE_ENABLED", False)):
        errors.append(
            "M1.V549_CROSS_BATCH_BALANCED_QUEUE_ENABLED must be false; "
            "V550 forbids stale post-trunk feature/label replay"
        )
    # V552-R2 deliberately disables the legacy V550 explicit Sign BCE.
    # R2 replaces the mutually inconsistent Benefit/Harm/Direction/Sign heads
    # with one three-class Outcome head and a sign-constrained magnitude
    # decomposition.  Requiring V550_M2_SIGN_WEIGHT > 0 here would both reject
    # the correct R2 config and encourage re-enabling the diagnosed conflict.
    r2_enabled = bool(
        _cfg_get(m1, "V552R2_TEACHER_DECOUPLED_ENABLED", False)
    )
    r42_enabled = bool(
        _cfg_get(m1, "V552R42_DECOUPLED_CRITIC_ENABLED", False)
    )
    positive_float = (
        "V550_GAIN_LISTWISE_TEMPERATURE",
        "V550_MAX_CLASS_WEIGHT",
        "V550_QUALITY_TARGET_POSITIVE_RATE",
        "V550_QUALITY_TARGET_PURITY",
        "V550_QUALITY_TARGET_CAPTURE",
    )
    for key in positive_float:
        if float(_cfg_get(m1, key, 0.0)) <= 0.0:
            errors.append(f"M1.{key} must be > 0")

    legacy_sign_weight = float(_cfg_get(m1, "V550_M2_SIGN_WEIGHT", 0.0))
    if r2_enabled:
        if abs(legacy_sign_weight) > 1.0e-12:
            errors.append(
                "M1.V550_M2_SIGN_WEIGHT must be 0 for V552-R2; "
                "the unified V552 Outcome/Gain objective owns sign calibration"
            )
        if r42_enabled:
            if abs(float(_cfg_get(m1, "V552_SIGNED_GAIN_WEIGHT", 0.0))) > 1.0e-12:
                errors.append(
                    "M1.V552_SIGNED_GAIN_WEIGHT must be 0 for V552-R4.2; "
                    "conditional magnitudes own gain calibration"
                )
        elif float(_cfg_get(m1, "V552_SIGNED_GAIN_WEIGHT", 0.0)) <= 0.0:
            errors.append(
                "M1.V552_SIGNED_GAIN_WEIGHT must be > 0 for V552-R2"
            )
    elif legacy_sign_weight <= 0.0:
        errors.append("M1.V550_M2_SIGN_WEIGHT must be > 0")

    # Pairwise/Listwise are optional during controlled ablations.
    # Negative weights remain invalid, while zero cleanly disables a term.
    nonnegative_float = (
        "V550_M2_GAIN_PAIRWISE_WEIGHT",
        "V550_M2_GAIN_LISTWISE_WEIGHT",
    )
    for key in nonnegative_float:
        if float(_cfg_get(m1, key, 0.0)) < 0.0:
            errors.append(f"M1.{key} must be >= 0")
    min_class = float(_cfg_get(m1, "V550_MIN_CLASS_WEIGHT", 0.0))
    max_class = float(_cfg_get(m1, "V550_MAX_CLASS_WEIGHT", 0.0))
    if min_class <= 0.0 or min_class > max_class:
        errors.append(
            "M1.V550_MIN_CLASS_WEIGHT must be > 0 and <= V550_MAX_CLASS_WEIGHT"
        )
    quality_floor = float(_cfg_get(m1, "V550_MIN_QUALITY_TRAIN_SCALE", -1.0))
    if not (0.0 <= quality_floor <= 1.0):
        errors.append("M1.V550_MIN_QUALITY_TRAIN_SCALE must be in [0,1]")
    sign_margin = float(_cfg_get(m1, "V542_GAIN_SIGN_MARGIN", -1.0))
    deploy_gain = float(_cfg_get(m1, "V538_COMPOSER_MIN_GAIN", -2.0))
    if abs(sign_margin - deploy_gain) > 1.0e-12:
        errors.append(
            "M1.V542_GAIN_SIGN_MARGIN must equal M1.V538_COMPOSER_MIN_GAIN"
        )
    if float(_cfg_get(m1, "V541_DEPLOY_LCB_BETA", 0.0)) != 0.0:
        errors.append("M1.V541_DEPLOY_LCB_BETA must be 0")
    if errors:
        raise ValueError(
            "V550 protocol validation failed:\n- " + "\n- ".join(errors)
        )


def _validate_v549_protocol(cfg):
    """Fail fast if V549 is not semantically closed from training to deployment."""
    m1 = _cfg_get(cfg, "M1", None)
    protocol = str(_cfg_get(m1, "V549_PROTOCOL_VERSION", "")).strip()
    if not protocol:
        return
    required_true = (
        "V547_FACTORIZED_OUTCOME_ENABLED",
        "V548_FACTORIZED_DIRECTION_ZERO_INIT",
        "V548_EXPLICIT_SIGN_LOSS_ENABLED",
        "V548_STABLE_M2_ROUTING_ENABLED",
        "V549_FACTORIZED_DEPLOYMENT_ENABLED",
        "V549_CROSS_BATCH_BALANCED_QUEUE_ENABLED",
        "V549_EPOCH_QUALITY_GATE_ENABLED",
        "V544_MINIMAL_M2_OBJECTIVE_ENABLED",
        "V544_USE_GAIN_AS_DECISION_SCORE",
    )
    errors = [
        f"M1.{key} must be true"
        for key in required_true
        if not bool(_cfg_get(m1, key, False))
    ]
    for key in (
        "V549_DEPLOY_EDITABILITY_THRESHOLD",
        "V549_DEPLOY_DIRECTION_THRESHOLD",
    ):
        value = float(_cfg_get(m1, key, -1.0))
        if not (0.0 < value < 1.0):
            errors.append(f"M1.{key} must be in (0,1)")
    if float(_cfg_get(m1, "V549_M2_SIGN_WEIGHT", 0.0)) <= 0.0:
        errors.append("M1.V549_M2_SIGN_WEIGHT must be > 0")
    if int(_cfg_get(m1, "V549_QUEUE_SIZE_PER_CLASS", 0)) <= 0:
        errors.append("M1.V549_QUEUE_SIZE_PER_CLASS must be > 0")
    if int(_cfg_get(m1, "V549_QUEUE_MAX_SAMPLES_PER_CLASS", 0)) <= 0:
        errors.append("M1.V549_QUEUE_MAX_SAMPLES_PER_CLASS must be > 0")
    if float(_cfg_get(m1, "V541_DEPLOY_LCB_BETA", 0.0)) != 0.0:
        errors.append("M1.V541_DEPLOY_LCB_BETA must be 0")
    if bool(_cfg_get(m1, "V546_GAIN_SIGN_SHADOW_DEPLOY_ENABLED", False)):
        errors.append(
            "M1.V546_GAIN_SIGN_SHADOW_DEPLOY_ENABLED must be false so Shadow "
            "uses the exact V549 deployment contract"
        )
    sign_margin = float(_cfg_get(m1, "V542_GAIN_SIGN_MARGIN", -1.0))
    deploy_gain = float(_cfg_get(m1, "V538_COMPOSER_MIN_GAIN", -2.0))
    if abs(sign_margin - deploy_gain) > 1.0e-12:
        errors.append(
            "M1.V542_GAIN_SIGN_MARGIN must equal M1.V538_COMPOSER_MIN_GAIN"
        )
    if errors:
        raise ValueError(
            "V549 protocol validation failed:\n- " + "\n- ".join(errors)
        )


def _validate_v548_protocol(cfg):
    """Fail fast if a V548 run reopens the diagnosed M2 collapse paths."""
    m1 = _cfg_get(cfg, "M1", None)
    protocol = str(_cfg_get(m1, "V548_PROTOCOL_VERSION", "")).strip()
    if not protocol:
        return
    required_true = (
        "V547_FACTORIZED_OUTCOME_ENABLED",
        "V544_MINIMAL_M2_OBJECTIVE_ENABLED",
        "V544_USE_GAIN_AS_DECISION_SCORE",
        "V548_FACTORIZED_DIRECTION_ZERO_INIT",
        "V548_CLASS_COMPLETE_DIRECTION_UPDATE",
        "V548_EXPLICIT_SIGN_LOSS_ENABLED",
        "V548_SIGN_LOSS_REQUIRES_BOTH_CLASSES",
        "V548_STABLE_M2_ROUTING_ENABLED",
    )
    errors = [
        f"M1.{key} must be true"
        for key in required_true
        if not bool(_cfg_get(m1, key, False))
    ]
    if float(_cfg_get(m1, "V543_M2_SIGN_MARGIN_WEIGHT", 0.0)) <= 0.0:
        errors.append("M1.V543_M2_SIGN_MARGIN_WEIGHT must be > 0")
    if float(_cfg_get(m1, "V541_DEPLOY_LCB_BETA", 0.0)) != 0.0:
        errors.append("M1.V541_DEPLOY_LCB_BETA must be 0 for deterministic signed Gain")
    if errors:
        raise ValueError(
            "V548 protocol validation failed:\n- " + "\n- ".join(errors)
        )


def _validate_v547_protocol(cfg):
    """Fail fast on combinations that recreate the diagnosed V546 failures."""
    m1 = _cfg_get(cfg, "M1", None)
    train = _cfg_get(cfg, "TRAIN", None)
    protocol = str(_cfg_get(m1, "V547_PROTOCOL_VERSION", "")).strip()
    if not protocol:
        return
    errors = []

    # V552-R4.6 intentionally supersedes only the *paired residual replay*
    # part of the historical V547 protocol.  The legacy validator used to
    # require V547_PAIRED_RESIDUAL_REPLAY_ENABLED=true unconditionally, while
    # R4.6's single-native contract correctly requires it false.  Dispatch
    # that one invariant by the active top-level protocol instead of imposing
    # mutually exclusive contracts before training can start.
    v552_protocol = str(_cfg_get(m1, "V552_PROTOCOL_VERSION", "")).strip()
    r46_single_native = bool(
        _cfg_get(m1, "V552R46_ROOTFIX_ENABLED", False)
        and _cfg_get(m1, "V552R46_NATIVE_STATE_ONLY_ENABLED", False)
    ) or v552_protocol in {"single_native_action_realizable_contract_v10", "spatially_anchored_query_mask_contract_v11", "iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15", "paired_stable_box_free_residual_mask_set_contract_v18", "location_conditioned_dynamic_residual_mask_contract_v20"}

    required_true = [
        "V538_ONLINE_COMPONENT_REFINER_ENABLED",
        "V547_FACTORIZED_OUTCOME_ENABLED",
        "V546_SLOT_COMPETITION_ENABLED",
        "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED",
        "V544_USE_GAIN_AS_DECISION_SCORE",
    ]
    if not r46_single_native:
        required_true.append("V547_PAIRED_RESIDUAL_REPLAY_ENABLED")

    for key in required_true:
        if not bool(_cfg_get(m1, key, False)):
            errors.append(f"M1.{key} must be true")

    required_false = [
        "V538_RESIDUAL_REPLAY_ENABLED",
        "V546_STREAMING_BALANCED_SOFTMAX_ENABLED",
        "V546_GAIN_SIGN_SHADOW_DEPLOY_ENABLED",
        "V546_VAL_USE_SHADOW_M2_FOR_SELECTION",
    ]
    if r46_single_native:
        required_false.append("V547_PAIRED_RESIDUAL_REPLAY_ENABLED")

    for key in required_false:
        if bool(_cfg_get(m1, key, False)):
            errors.append(f"M1.{key} must be false")
    selection_metric = str(
        _cfg_get(train, "VAL_SELECTION_METRIC", "")
    )
    if not selection_metric.startswith("native_m2_"):
        errors.append(
            "TRAIN.VAL_SELECTION_METRIC must select actual native_m2_* deployment"
        )
    deploy_start = int(_cfg_get(m1, "V538_DEPLOY_START_EPOCH", 999999))
    total_epochs = int(_cfg_get(train, "NUM_EPOCHS", 0))
    if deploy_start >= total_epochs:
        errors.append(
            f"M1.V538_DEPLOY_START_EPOCH={deploy_start} must be < NUM_EPOCHS={total_epochs}"
        )
    selection_start = int(_cfg_get(train, "VAL_SELECTION_START_EPOCH", 1))
    if selection_start <= deploy_start:
        errors.append(
            "TRAIN.VAL_SELECTION_START_EPOCH is one-based and must be greater "
            "than the zero-based V538_DEPLOY_START_EPOCH"
        )
    batch_size = int(_cfg_get(train, "BATCH_SIZE", 1))
    if batch_size <= 2 and float(_cfg_get(train, "BASE_TAIL_CVAR_WEIGHT", 0.0)) > 0.0:
        errors.append("Batch<=2 forbids mini-batch BASE_TAIL_CVAR_WEIGHT; use V547 memory")
    if not bool(_cfg_get(train, "V547_HARD_CASE_MEMORY_ENABLED", False)):
        errors.append("TRAIN.V547_HARD_CASE_MEMORY_ENABLED must be true")
    if not bool(_cfg_get(train, "V547_SEMANTIC_SAFE_AUG_ENABLED", False)):
        errors.append("TRAIN.V547_SEMANTIC_SAFE_AUG_ENABLED must be true")
    if errors:
        raise ValueError(
            "V547 protocol contract failed:\n  - " + "\n  - ".join(errors)
        )


def _validate_v551_rootfix_protocol(cfg):
    """Fail fast if V551 regresses to the CPU/double-encoding implementation."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V551_MULTISCALE_TYPED_EDITOR_ENABLED", False)):
        return
    errors = []
    protocol = str(_cfg_get(m1, "V551_PROTOCOL_VERSION", "")).strip()
    if protocol not in {
        "gpu_sparse_single_pass_v1",
        "safe_calibrated_sparse_editor_v1",
    }:
        errors.append(
            "M1.V551_PROTOCOL_VERSION must be gpu_sparse_single_pass_v1 "
            "or safe_calibrated_sparse_editor_v1"
        )
    if protocol == "safe_calibrated_sparse_editor_v1":
        v552_protocol = str(_cfg_get(m1, "V552_PROTOCOL_VERSION", "")).strip()
        allowed_v552_protocols = {
            "safe_calibrated_multicandidate_composer_v2",
            "teacher_decoupled_outcome_calibrated_composer_v3",
            "root_calibrated_teacher_forced_composer_v4",
            "unified_reference_contract_v5",
            "decoupled_critic_contract_v6",
            "spatial_evidence_action_aware_contract_v7",
            "audit_gate_class_value_contract_v8",
            "error_aware_factorized_safe_contract_v9",
            "single_native_action_realizable_contract_v10",
            "spatially_anchored_query_mask_contract_v11",
            "iterative_denoising_component_decoder_contract_v12",
            "content_selective_residual_decoder_contract_v13",
            "evidence_proposed_local_reconstruction_contract_v14",
            "typed_native_residual_set_refiner_contract_v15",
            "paired_stable_box_free_residual_mask_set_contract_v18",
            "location_conditioned_dynamic_residual_mask_contract_v20",
            "location_conditioned_dynamic_residual_mask_clean_contract_v21",
            "dense_competitive_residual_set_contract_v22",
            "factorized_residual_existence_identity_contract_v24",
            "capacity_consistent_residual_factorization_contract_v25",
            "factorization_aligned_conditional_identity_contract_v26",
            "dynamic_visual_instance_binding_contract_v27",
            "normalized_visual_instance_binding_contract_v28_a2",
            "persistent_visual_instance_binding_contract_v28_a3",
            "seed_consistent_persistent_binding_contract_v28_a4",
            "goal_driven_peak_binding_contract_v29_b1",
            "goal_driven_seed_matching_contract_v29_b2",
            "goal_driven_full_m1_contract_v29_b3",
            "instance_valid_anchor_control_contract_v30_c0",
            "interior_anchor_contract_v30_c1",
            "variable_cardinality_seed_contract_v30_c2",
            "independent_overflow_contract_v30_c3",
            "deployable_m1_alignment_contract_v30_c4",
            "hard_seed_gate_control_contract_v31_d0",
            "proposal_existence_decoupling_contract_v31_d1",
            "geometry_overflow_decoupling_contract_v31_d2",
            "deployable_alignment_decoupled_contract_v31_d3",
        }
        if v552_protocol not in allowed_v552_protocols:
            errors.append(
                "M1.V552_PROTOCOL_VERSION must be one of "
                + ", ".join(sorted(allowed_v552_protocols))
            )

        # R4.20.3/V22 removes the Dense->TopK->Point->DynamicMask
        # bottleneck from the Native residual-set path.  The dense R4.17 field
        # remains supervised, but slots form by differentiable pixel ownership
        # competition; no hard point, box, radius or dynamic-channel renderer
        # is allowed to own residual geometry.
        if v552_protocol in {
            "dense_competitive_residual_set_contract_v22",
            "factorized_residual_existence_identity_contract_v24",
            "capacity_consistent_residual_factorization_contract_v25",
            "factorization_aligned_conditional_identity_contract_v26",
            "dynamic_visual_instance_binding_contract_v27",
            "normalized_visual_instance_binding_contract_v28_a2",
            "persistent_visual_instance_binding_contract_v28_a3",
            "seed_consistent_persistent_binding_contract_v28_a4",
            "goal_driven_peak_binding_contract_v29_b1",
            "goal_driven_seed_matching_contract_v29_b2",
            "goal_driven_full_m1_contract_v29_b3",
            "instance_valid_anchor_control_contract_v30_c0",
            "interior_anchor_contract_v30_c1",
            "variable_cardinality_seed_contract_v30_c2",
            "independent_overflow_contract_v30_c3",
            "deployable_m1_alignment_contract_v30_c4",
            "hard_seed_gate_control_contract_v31_d0",
            "proposal_existence_decoupling_contract_v31_d1",
            "geometry_overflow_decoupling_contract_v31_d2",
            "deployable_alignment_decoupled_contract_v31_d3",
        }:
            required_true = (
                "CEM_V484_ENABLED",
                "V532_UNIFIED_SPARSE_REFINER_ENABLED",
                "V538_ONLINE_COMPONENT_REFINER_ENABLED",
                "V538_INDEPENDENT_OBJECTIVE_ROUTING",
                "V551_MULTISCALE_TYPED_EDITOR_ENABLED",
                "V552R47_ROOTFIX_ENABLED",
                "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
                "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
                "V552R48_ROOTFIX_ENABLED",
                "V552R48_ITERATIVE_BINDING_ENABLED",
                "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
                "V552R411_ROOTFIX_ENABLED",
                "V552R411_TYPED_PROPOSAL_ENABLED",
                "V552R411_USE_RAW_NATIVE_MASKS",
                "V552R417_ROOTFIX_ENABLED",
                "V552R418_ROOTFIX_ENABLED",
                "V552R4201_ROOTFIX_ENABLED",
                "V552R4203_ROOTFIX_ENABLED",
                "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED",
            )
            for key in required_true:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20.3 requires M1.{key}=true")

            required_false = (
                "V552R47_POINT_MASK_SUPERVISION_ENABLED",
                "V552R47_HYBRID_MATCHING_ENABLED",
                "V552R48_DEEP_SUPERVISION_ENABLED",
                "V552R48_DN_COMPONENT_QUERY_ENABLED",
                "V552R49_ROOTFIX_ENABLED",
                "V552R410_ROOTFIX_ENABLED",
                "V552R411_LOCAL_ROI_DECODER_ENABLED",
                "V552R412_ROOTFIX_ENABLED",
                "V552R413_ROOTFIX_ENABLED",
                "V552R414_ROOTFIX_ENABLED",
                "V552R415_ROOTFIX_ENABLED",
                "V552R416_ROOTFIX_ENABLED",
                "V552R417_SHARED_OFFSET_ENABLED",
                "V552R418_PAIRED_STABLE_TEACHER_ENABLED",
                "V552R419_ROOTFIX_ENABLED",
                "V552R420_ROOTFIX_ENABLED",
            )
            for key in required_false:
                if bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20.3 requires M1.{key}=false")

            zero_weights = (
                "V552R47_POINT_MASK_WEIGHT",
                "V552R47_ANCHOR_REG_WEIGHT",
                "V552R48_DEEP_MASK_WEIGHT",
                "V552R48_DN_MASK_WEIGHT",
                "V552R48_DN_ANCHOR_WEIGHT",
                "V552R411_PROPOSAL_OFFSET_WEIGHT",
                "V552R411_PROPOSAL_SIZE_WEIGHT",
            )
            for key in zero_weights:
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(f"V552-R4.20.3 requires M1.{key}=0")

            if float(_cfg_get(m1, "V538_M1_TO_BASE_OBJECTIVE_RATIO", 0.0)) <= 0.0:
                errors.append("V552-R4.20.3 requires V538_M1_TO_BASE_OBJECTIVE_RATIO>0")
            if float(_cfg_get(m1, "V538_M2_TO_BASE_OBJECTIVE_RATIO", -1.0)) < 0.0:
                errors.append("V552-R4.20.3 requires V538_M2_TO_BASE_OBJECTIVE_RATIO>=0")
            if float(_cfg_get(m1, "V538_M1_MAX_EFFECTIVE_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.3 requires V538_M1_MAX_EFFECTIVE_WEIGHT>0")
            if float(_cfg_get(m1, "V538_M2_MAX_EFFECTIVE_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.3 requires V538_M2_MAX_EFFECTIVE_WEIGHT>0")
            if int(_cfg_get(m1, "V538_M1_ROUTE_START_EPOCH", -1)) != 0:
                errors.append("V552-R4.20.3 requires V538_M1_ROUTE_START_EPOCH=0")
            if int(_cfg_get(m1, "V538_M1_ROUTE_RAMP_EPOCHS", 0)) < 1:
                errors.append("V552-R4.20.3 requires V538_M1_ROUTE_RAMP_EPOCHS>=1")
            if float(_cfg_get(m1, "CANDIDATE_LOSS_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.3 requires CANDIDATE_LOSS_WEIGHT>0")

            if v552_protocol in {
                "factorized_residual_existence_identity_contract_v24",
                "capacity_consistent_residual_factorization_contract_v25",
                "factorization_aligned_conditional_identity_contract_v26",
                "dynamic_visual_instance_binding_contract_v27",
            "normalized_visual_instance_binding_contract_v28_a2",
            "persistent_visual_instance_binding_contract_v28_a3",
            "seed_consistent_persistent_binding_contract_v28_a4",
            "goal_driven_peak_binding_contract_v29_b1",
            "goal_driven_seed_matching_contract_v29_b2",
            "goal_driven_full_m1_contract_v29_b3",
            "instance_valid_anchor_control_contract_v30_c0",
            "interior_anchor_contract_v30_c1",
            "variable_cardinality_seed_contract_v30_c2",
            "independent_overflow_contract_v30_c3",
            "deployable_m1_alignment_contract_v30_c4",
            "hard_seed_gate_control_contract_v31_d0",
            "proposal_existence_decoupling_contract_v31_d1",
            "geometry_overflow_decoupling_contract_v31_d2",
            "deployable_alignment_decoupled_contract_v31_d3",
            }:
                if not bool(_cfg_get(m1, "V552R4204_ROOTFIX_ENABLED", False)):
                    errors.append("V552-R4.20.4/5 requires M1.V552R4204_ROOTFIX_ENABLED=true")
                if not bool(_cfg_get(m1, "V552R4204_BASE_RNG_ISOLATION_ENABLED", False)):
                    errors.append("V552-R4.20.4/5 requires M1.V552R4204_BASE_RNG_ISOLATION_ENABLED=true")
                if not bool(_cfg_get(m1, "V487_BASE_SAFE_E2E", False)):
                    errors.append("V552-R4.20.4/5 requires M1.V487_BASE_SAFE_E2E=true for exact Base/PVL auxiliary-gradient isolation")
                forbidden_r4204 = (
                    "V552R419_ROOTFIX_ENABLED",
                    "V552R420_ROOTFIX_ENABLED",
                )
                for key in forbidden_r4204:
                    if bool(_cfg_get(m1, key, False)):
                        errors.append(f"V552-R4.20.4/5 requires M1.{key}=false")
                if v552_protocol in {
                    "capacity_consistent_residual_factorization_contract_v25",
                    "factorization_aligned_conditional_identity_contract_v26",
                    "dynamic_visual_instance_binding_contract_v27",
            "normalized_visual_instance_binding_contract_v28_a2",
            "persistent_visual_instance_binding_contract_v28_a3",
            "seed_consistent_persistent_binding_contract_v28_a4",
                    "goal_driven_peak_binding_contract_v29_b1",
                    "goal_driven_seed_matching_contract_v29_b2",
                    "goal_driven_full_m1_contract_v29_b3",
                    "instance_valid_anchor_control_contract_v30_c0",
                    "interior_anchor_contract_v30_c1",
                    "variable_cardinality_seed_contract_v30_c2",
                    "independent_overflow_contract_v30_c3",
                    "deployable_m1_alignment_contract_v30_c4",
                    "hard_seed_gate_control_contract_v31_d0",
                    "proposal_existence_decoupling_contract_v31_d1",
                    "geometry_overflow_decoupling_contract_v31_d2",
                    "deployable_alignment_decoupled_contract_v31_d3",
                }:
                    if not bool(_cfg_get(m1, "V552R4205_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.5/6 requires M1.V552R4205_ROOTFIX_ENABLED=true")
                    if bool(_cfg_get(m1, "V552R4204_SPATIAL_IDENTITY_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.5/6 requires M1.V552R4204_SPATIAL_IDENTITY_ENABLED=false; "
                            "the self-derived centroid/variance feedback failed causally and numerically"
                        )
                if v552_protocol == "factorization_aligned_conditional_identity_contract_v26":
                    if not bool(_cfg_get(m1, "V552R4206_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.6 requires M1.V552R4206_ROOTFIX_ENABLED=true")
                    if not bool(_cfg_get(m1, "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.6 requires V546_OPTIMAL_COMPONENT_MATCHING_ENABLED=true; "
                            "conditional slot labels must follow one globally optimal permutation"
                        )
                    if bool(_cfg_get(m1, "V552R412_ROOTFIX_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.6 requires V552R412_ROOTFIX_ENABLED=false so canonical BCE "
                            "cannot confound the CE+Dice identity/shape objective"
                        )
                    if bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.6 requires V552R418_PAIRED_STABLE_TEACHER_ENABLED=false "
                            "for a single Native conditional-identity intervention"
                        )
                if v552_protocol == "dynamic_visual_instance_binding_contract_v27":
                    if not bool(_cfg_get(m1, "V552R4207_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.7 requires M1.V552R4207_ROOTFIX_ENABLED=true")
                    if bool(_cfg_get(m1, "V552R4206_ROOTFIX_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.7 requires V552R4206_ROOTFIX_ENABLED=false; "
                            "query binding must be isolated from conditional-CE objective changes"
                        )
                    if not bool(_cfg_get(m1, "V552R417_ROOTFIX_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.7 requires V552R417_ROOTFIX_ENABLED=true for inference-visible visual seeds"
                        )
                    if bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.7 requires V552R418_PAIRED_STABLE_TEACHER_ENABLED=false "
                            "so only Native query initialization changes"
                        )
                    if bool(_cfg_get(m1, "V552R412_ROOTFIX_ENABLED", False)):
                        errors.append(
                            "V552-R4.20.7 requires V552R412_ROOTFIX_ENABLED=false; "
                            "no box/canonical geometry owner may confound instance binding"
                        )
                if v552_protocol in {
                    "normalized_visual_instance_binding_contract_v28_a2",
                    "persistent_visual_instance_binding_contract_v28_a3",
                    "seed_consistent_persistent_binding_contract_v28_a4",
                    "goal_driven_peak_binding_contract_v29_b1",
                    "goal_driven_seed_matching_contract_v29_b2",
                    "goal_driven_full_m1_contract_v29_b3",
                    "instance_valid_anchor_control_contract_v30_c0",
                    "interior_anchor_contract_v30_c1",
                    "variable_cardinality_seed_contract_v30_c2",
                    "independent_overflow_contract_v30_c3",
                    "deployable_m1_alignment_contract_v30_c4",
                    "hard_seed_gate_control_contract_v31_d0",
                    "proposal_existence_decoupling_contract_v31_d1",
                    "geometry_overflow_decoupling_contract_v31_d2",
                    "deployable_alignment_decoupled_contract_v31_d3",
                }:
                    if not bool(_cfg_get(m1, "V552R4208_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.8 requires M1.V552R4208_ROOTFIX_ENABLED=true")
                    if not bool(_cfg_get(m1, "V552R4208_NORMALIZED_FUSION_ENABLED", False)):
                        errors.append("V552-R4.20.8 requires normalized visual-query fusion")
                    if bool(_cfg_get(m1, "V552R4207_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.8 forbids simultaneous R4.20.7 one-shot query intervention")
                    if bool(_cfg_get(m1, "V552R4206_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.8 requires R4.20.6 CE disabled for causal isolation")
                    expected_persistent = v552_protocol in {
                        "persistent_visual_instance_binding_contract_v28_a3",
                        "seed_consistent_persistent_binding_contract_v28_a4",
                        "goal_driven_peak_binding_contract_v29_b1",
                        "goal_driven_seed_matching_contract_v29_b2",
                        "goal_driven_full_m1_contract_v29_b3",
                        "instance_valid_anchor_control_contract_v30_c0",
                        "interior_anchor_contract_v30_c1",
                        "variable_cardinality_seed_contract_v30_c2",
                        "independent_overflow_contract_v30_c3",
                        "deployable_m1_alignment_contract_v30_c4",
                        "hard_seed_gate_control_contract_v31_d0",
                        "proposal_existence_decoupling_contract_v31_d1",
                        "geometry_overflow_decoupling_contract_v31_d2",
                        "deployable_alignment_decoupled_contract_v31_d3",
                    }
                    if bool(_cfg_get(m1, "V552R4208_PERSISTENT_IDENTITY_ENABLED", False)) != expected_persistent:
                        errors.append(
                            f"V552-R4.20.8 protocol {v552_protocol} persistent-identity flag mismatch"
                        )
                    expected_matching = v552_protocol in {
                        "seed_consistent_persistent_binding_contract_v28_a4",
                        "goal_driven_seed_matching_contract_v29_b2",
                        "goal_driven_full_m1_contract_v29_b3",
                    }
                    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)) != expected_matching:
                        errors.append(
                            f"V552-R4.20.8 protocol {v552_protocol} seed-matching flag mismatch"
                        )
                if v552_protocol in {
                    "goal_driven_peak_binding_contract_v29_b1",
                    "goal_driven_seed_matching_contract_v29_b2",
                    "goal_driven_full_m1_contract_v29_b3",
                    "instance_valid_anchor_control_contract_v30_c0",
                    "interior_anchor_contract_v30_c1",
                    "variable_cardinality_seed_contract_v30_c2",
                    "independent_overflow_contract_v30_c3",
                    "deployable_m1_alignment_contract_v30_c4",
                    "hard_seed_gate_control_contract_v31_d0",
                    "proposal_existence_decoupling_contract_v31_d1",
                    "geometry_overflow_decoupling_contract_v31_d2",
                    "deployable_alignment_decoupled_contract_v31_d3",
                }:
                    if not bool(_cfg_get(m1, "V552R4209_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.9 requires V552R4209_ROOTFIX_ENABLED=true")
                    if not bool(_cfg_get(m1, "V552R4209_CENTER_PEAK_FOCAL_ENABLED", False)):
                        errors.append("V552-R4.20.9 requires standard center-peak focal supervision")
                    if not bool(_cfg_get(m1, "V552R4208_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.20.9 requires R4.20.8 normalized binding")
                    if not bool(_cfg_get(m1, "V552R4208_PERSISTENT_IDENTITY_ENABLED", False)):
                        errors.append("V552-R4.20.9 requires persistent identity")
                    expected_seed_matching_v29 = v552_protocol in {
                        "goal_driven_seed_matching_contract_v29_b2",
                        "goal_driven_full_m1_contract_v29_b3",
                    }
                    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)) != expected_seed_matching_v29:
                        errors.append(
                            f"V552-R4.20.9 protocol {v552_protocol} seed-matching flag mismatch"
                        )
                    expected_overflow_v29 = v552_protocol == "goal_driven_full_m1_contract_v29_b3"
                    if bool(_cfg_get(m1, "V552R4209_BALANCED_OVERFLOW_ENABLED", False)) != expected_overflow_v29:
                        errors.append(
                            f"V552-R4.20.9 protocol {v552_protocol} balanced-overflow flag mismatch"
                        )
                    target_dsc = float(_cfg_get(m1, "V552R4209_M1_TARGET_DSC", 0.86))
                    if not (0.0 < target_dsc <= 1.0):
                        errors.append("V552-R4.20.9 M1 target DSC must be in (0,1]")

                r4210_protocols = {
                    "instance_valid_anchor_control_contract_v30_c0",
                    "interior_anchor_contract_v30_c1",
                    "variable_cardinality_seed_contract_v30_c2",
                    "independent_overflow_contract_v30_c3",
                    "deployable_m1_alignment_contract_v30_c4",
                }
                if v552_protocol in r4210_protocols:
                    if not bool(_cfg_get(m1, "V552R4210_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.21.0 requires V552R4210_ROOTFIX_ENABLED=true")
                    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)):
                        errors.append("V552-R4.21.0 forbids hard seed-consistent matching; mask geometry stays permutation-invariant")
                    if bool(_cfg_get(m1, "V552R4209_BALANCED_OVERFLOW_ENABLED", False)):
                        errors.append("V552-R4.21.0 forbids shared-simplex balanced overflow")
                    if bool(_cfg_get(m1, "V552R4206_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.21.0 forbids R4.20.6 global K+1 CE")
                    order = [
                        "instance_valid_anchor_control_contract_v30_c0",
                        "interior_anchor_contract_v30_c1",
                        "variable_cardinality_seed_contract_v30_c2",
                        "independent_overflow_contract_v30_c3",
                        "deployable_m1_alignment_contract_v30_c4",
                    ]
                    stage = order.index(v552_protocol)
                    expected = {
                        "V552R4210_INTERIOR_ANCHOR_ENABLED": stage >= 1,
                        "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED": stage >= 2,
                        "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED": stage >= 3,
                        "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED": stage >= 4,
                    }
                    for key, want in expected.items():
                        got = bool(_cfg_get(m1, key, False))
                        if got != want:
                            errors.append(f"V552-R4.21.0 {v552_protocol} requires M1.{key}={str(want).lower()}")
                    target_dsc_4210 = float(_cfg_get(m1, "V552R4210_M1_TARGET_DSC", 0.86))
                    if not (0.0 < target_dsc_4210 <= 1.0):
                        errors.append("V552-R4.21.0 M1 target DSC must be in (0,1]")

                r4211_protocols = {
                    "hard_seed_gate_control_contract_v31_d0",
                    "proposal_existence_decoupling_contract_v31_d1",
                    "geometry_overflow_decoupling_contract_v31_d2",
                    "deployable_alignment_decoupled_contract_v31_d3",
                }
                if v552_protocol in r4211_protocols:
                    if not bool(_cfg_get(m1, "V552R4210_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.21.1 requires R4.21.0 root scaffold")
                    if not bool(_cfg_get(m1, "V552R4210_INTERIOR_ANCHOR_ENABLED", False)):
                        errors.append("V552-R4.21.1 requires the guaranteed-inside interior anchor")
                    if not bool(_cfg_get(m1, "V552R4211_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.21.1 requires V552R4211_ROOTFIX_ENABLED=true")
                    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)):
                        errors.append("V552-R4.21.1 forbids hard seed-consistent matching")
                    if bool(_cfg_get(m1, "V552R4209_BALANCED_OVERFLOW_ENABLED", False)):
                        errors.append("V552-R4.21.1 forbids legacy shared-simplex overflow")
                    if bool(_cfg_get(m1, "V552R4206_ROOTFIX_ENABLED", False)):
                        errors.append("V552-R4.21.1 forbids global K+1 CE")
                    order4211 = [
                        "hard_seed_gate_control_contract_v31_d0",
                        "proposal_existence_decoupling_contract_v31_d1",
                        "geometry_overflow_decoupling_contract_v31_d2",
                        "deployable_alignment_decoupled_contract_v31_d3",
                    ]
                    stage4211 = order4211.index(v552_protocol)
                    expected4211 = {
                        "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED": stage4211 == 0,
                        "V552R4211_PROPOSAL_EXISTENCE_DECOUPLING_ENABLED": stage4211 >= 1,
                        "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED": stage4211 >= 2,
                        "V552R4211_GEOMETRY_OVERFLOW_DECOUPLING_ENABLED": stage4211 >= 2,
                        "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED": stage4211 >= 3,
                    }
                    for key, want in expected4211.items():
                        got = bool(_cfg_get(m1, key, False))
                        if got != want:
                            errors.append(
                                f"V552-R4.21.1 {v552_protocol} requires M1.{key}={str(want).lower()}"
                            )
                    if stage4211 >= 1 and bool(_cfg_get(m1, "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED", False)):
                        errors.append("V552-R4.21.1 D1+ forbids location-confidence hard gating of instance existence")
                    if stage4211 >= 2 and not bool(_cfg_get(m1, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)):
                        errors.append("V552-R4.21.1 D2+ requires independent Bernoulli overflow")
                    target_dsc_4211 = float(_cfg_get(m1, "V552R4211_M1_TARGET_DSC", 0.86))
                    if not (0.0 < target_dsc_4211 <= 1.0):
                        errors.append("V552-R4.21.1 M1 target DSC must be in (0,1]")

                # V552-R4.21.2 is layered on the exact D0 scaffold so that all
                # E0-E5 runs share the historical parameter topology/RNG.  Its
                # own stage key validates only the new causal interventions.
                if bool(_cfg_get(m1, "V552R4212_ROOTFIX_ENABLED", False)):
                    if v552_protocol != "hard_seed_gate_control_contract_v31_d0":
                        errors.append("V552-R4.21.2 requires the exact R4.21.1 D0 scaffold protocol")
                    if not bool(_cfg_get(m1, "V552R4210_INTERIOR_ANCHOR_ENABLED", False)):
                        errors.append("V552-R4.21.2 requires guaranteed-inside interior anchors")
                    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)):
                        errors.append("V552-R4.21.2 forbids hard seed-consistent matching")
                    if bool(_cfg_get(m1, "V552R4209_BALANCED_OVERFLOW_ENABLED", False)):
                        errors.append("V552-R4.21.2 forbids legacy balanced overflow")
                    if bool(_cfg_get(m1, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)):
                        errors.append("V552-R4.21.2 keeps overflow audit-only; executable overflow gate must be disabled")
                    if bool(_cfg_get(m1, "V552R4211_GEOMETRY_OVERFLOW_DECOUPLING_ENABLED", False)):
                        errors.append("V552-R4.21.2 forbids executable geometry/overflow gating")
                    if bool(_cfg_get(m1, "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED", False)):
                        errors.append("V552-R4.21.2 isolates candidate-bank alignment from legacy M1Native alignment")
                    if int(_cfg_get(m1, "V538_COMPOSER_MAX_STEPS", 0)) != 1:
                        errors.append("V552-R4.21.2 requires one-step M2")
                    stage4212 = int(_cfg_get(m1, "V552R4212_STAGE", 0))
                    if stage4212 not in {1,2,3,4,5}:
                        errors.append("V552-R4.21.2 stage must be one of 1..5")
                    expected4212 = {
                        "V552R4212_INDEPENDENT_CANDIDATE_SET_ENABLED": stage4212 >= 1,
                        "V552R4212_EXISTENCE_NO_OBJECT_ENABLED": stage4212 >= 1,
                        "V552R4212_DISABLE_VISUAL_SEED_IDENTITY_ENABLED": stage4212 >= 2,
                        "V552R4212_CANDIDATE_ALIGNMENT_ENABLED": stage4212 >= 3,
                        "V552R4212_DIRECT_DELTA_UTILITY_ENABLED": stage4212 >= 4,
                        "V552R4212_ZERO_STOP_ONE_STEP_ENABLED": stage4212 >= 5,
                    }
                    for key, want in expected4212.items():
                        got = bool(_cfg_get(m1, key, False))
                        if got != want:
                            errors.append(
                                f"V552-R4.21.2 stage E{stage4212} requires M1.{key}={str(want).lower()}"
                            )
                    if stage4212 >= 1 and bool(_cfg_get(m1, "V546_SLOT_COMPETITION_ENABLED", False)):
                        errors.append("V552-R4.21.2 independent set forbids winner-take-all slot competition")
            if abs(float(_cfg_get(m1, "V532_M1_TO_BASE_OBJECTIVE_RATIO", 0.0))) > 1.0e-12:
                errors.append("V552-R4.20.3 keeps V532_M1_TO_BASE_OBJECTIVE_RATIO=0; V538 owns M1 routing")
            if abs(float(_cfg_get(m1, "V532_REFINER_TO_BASE_OBJECTIVE_RATIO", 0.0))) > 1.0e-12:
                errors.append("V552-R4.20.3 keeps V532_REFINER_TO_BASE_OBJECTIVE_RATIO=0; V538 owns M2 routing")
            if bool(_cfg_get(m1, "V532_FREEZE_BASE", True)):
                errors.append("V552-R4.20.3 keeps Base trainable from scratch")
            if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
                errors.append("V552-R4.20.3 forbids a task/M1 INIT_CHECKPOINT")
            if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(
                _cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)
            ):
                errors.append("V552-R4.20.3 requires NUM_COMPONENT_SLOTS == MAX_ACTIVE_ATOMS")
            if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
                errors.append("V552-R4.20.3 requires V551_MAX_ATOMS_PER_SLOT=1")
            teacher_min = int(_cfg_get(m1, "V538_TEACHER_MIN_PIXELS", -1))
            atom_min = int(_cfg_get(m1, "V551_ATOM_MIN_PIXELS", -2))
            if teacher_min < 1 or teacher_min != atom_min:
                errors.append("V552-R4.20.3 requires one physical min-pixel teacher/atom contract")
            selection_metric = str(
                _cfg_get(_cfg_get(cfg, "TRAIN", None), "VAL_SELECTION_METRIC", "")
            ).strip().lower()
            if selection_metric != "native_m2_dice":
                errors.append("V552-R4.20.3 requires TRAIN.VAL_SELECTION_METRIC=native_m2_dice")

            if errors:
                raise ValueError(
                    "V552-R4.20.3 dense-set protocol validation failed:\n- "
                    + "\n- ".join(errors)
                )
            return

        # R4.20.1/V21 is intentionally self-contained.  Do not route it
        # through R2->R4.16 historical validator cascades: those cascades are
        # exactly why a current experiment inherited ~1,700 stale M1 knobs.
        # The clean contract declares only invariants that still own the
        # current forward/loss/deployment graph.
        if v552_protocol == "location_conditioned_dynamic_residual_mask_clean_contract_v21":
            required_true = (
                "CEM_V484_ENABLED",
                "V532_UNIFIED_SPARSE_REFINER_ENABLED",
                "V538_ONLINE_COMPONENT_REFINER_ENABLED",
                "V538_INDEPENDENT_OBJECTIVE_ROUTING",
                "V551_MULTISCALE_TYPED_EDITOR_ENABLED",
                "V552R47_ROOTFIX_ENABLED",
                "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
                "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
                "V552R48_ROOTFIX_ENABLED",
                "V552R48_ITERATIVE_BINDING_ENABLED",
                "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
                "V552R411_ROOTFIX_ENABLED",
                "V552R411_TYPED_PROPOSAL_ENABLED",
                "V552R411_USE_RAW_NATIVE_MASKS",
                "V552R417_ROOTFIX_ENABLED",
                "V552R418_ROOTFIX_ENABLED",
                "V552R420_ROOTFIX_ENABLED",
                "V552R4201_ROOTFIX_ENABLED",
                "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED",
            )
            for key in required_true:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20.1 clean requires M1.{key}=true")

            required_false = (
                "V552R47_POINT_MASK_SUPERVISION_ENABLED",
                "V552R47_HYBRID_MATCHING_ENABLED",
                "V552R48_DEEP_SUPERVISION_ENABLED",
                "V552R48_DN_COMPONENT_QUERY_ENABLED",
                "V552R49_ROOTFIX_ENABLED",
                "V552R410_ROOTFIX_ENABLED",
                "V552R411_LOCAL_ROI_DECODER_ENABLED",
                "V552R412_ROOTFIX_ENABLED",
                "V552R413_ROOTFIX_ENABLED",
                "V552R414_ROOTFIX_ENABLED",
                "V552R415_ROOTFIX_ENABLED",
                "V552R416_ROOTFIX_ENABLED",
                "V552R417_SHARED_OFFSET_ENABLED",
                "V552R418_PAIRED_STABLE_TEACHER_ENABLED",
                "V552R419_ROOTFIX_ENABLED",
            )
            for key in required_false:
                if bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20.1 clean requires M1.{key}=false")

            zero_weights = (
                "V552R47_POINT_MASK_WEIGHT",
                "V552R47_ANCHOR_REG_WEIGHT",
                "V552R48_DEEP_MASK_WEIGHT",
                "V552R48_DN_MASK_WEIGHT",
                "V552R48_DN_ANCHOR_WEIGHT",
                "V552R411_PROPOSAL_OFFSET_WEIGHT",
                "V552R411_PROPOSAL_SIZE_WEIGHT",
            )
            for key in zero_weights:
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(f"V552-R4.20.1 clean requires M1.{key}=0")

            # R4.20.1 objective ownership contract.  The CLEAN pruning must not
            # remove the V538 independent router: V532_M1_TO_BASE_OBJECTIVE_RATIO
            # and V532_REFINER_TO_BASE_OBJECTIVE_RATIO are intentionally zero so
            # the legacy V532 bundle cannot become a second objective owner.
            # Therefore V538 is the *single* live owner that routes the current
            # component-mask/location M1 objective and M2 supervision objective.
            if float(_cfg_get(m1, "V538_M1_TO_BASE_OBJECTIVE_RATIO", 0.0)) <= 0.0:
                errors.append("V552-R4.20.1 clean requires V538_M1_TO_BASE_OBJECTIVE_RATIO>0")
            if float(_cfg_get(m1, "V538_M2_TO_BASE_OBJECTIVE_RATIO", -1.0)) < 0.0:
                errors.append("V552-R4.20.1 clean requires V538_M2_TO_BASE_OBJECTIVE_RATIO>=0")
            if float(_cfg_get(m1, "V538_M1_MAX_EFFECTIVE_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.1 clean requires V538_M1_MAX_EFFECTIVE_WEIGHT>0")
            if float(_cfg_get(m1, "V538_M2_MAX_EFFECTIVE_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.1 clean requires V538_M2_MAX_EFFECTIVE_WEIGHT>0")
            if int(_cfg_get(m1, "V538_M1_ROUTE_START_EPOCH", -1)) != 0:
                errors.append("V552-R4.20.1 clean requires V538_M1_ROUTE_START_EPOCH=0")
            if int(_cfg_get(m1, "V538_M1_ROUTE_RAMP_EPOCHS", 0)) < 1:
                errors.append("V552-R4.20.1 clean requires V538_M1_ROUTE_RAMP_EPOCHS>=1")
            if float(_cfg_get(m1, "CANDIDATE_LOSS_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.20.1 clean requires CANDIDATE_LOSS_WEIGHT>0 so M1 forward/loss is computed")
            if abs(float(_cfg_get(m1, "V532_M1_TO_BASE_OBJECTIVE_RATIO", 0.0))) > 1.0e-12:
                errors.append("V552-R4.20.1 clean keeps V532_M1_TO_BASE_OBJECTIVE_RATIO=0; V538 owns M1 routing")
            if abs(float(_cfg_get(m1, "V532_REFINER_TO_BASE_OBJECTIVE_RATIO", 0.0))) > 1.0e-12:
                errors.append("V552-R4.20.1 clean keeps V532_REFINER_TO_BASE_OBJECTIVE_RATIO=0; V538 owns M2 routing")

            if bool(_cfg_get(m1, "V532_FREEZE_BASE", True)):
                errors.append("V552-R4.20.1 clean keeps Base trainable from scratch")
            if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
                errors.append("V552-R4.20.1 clean forbids a task/M1 INIT_CHECKPOINT")
            if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(
                _cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)
            ):
                errors.append("V552-R4.20.1 requires NUM_COMPONENT_SLOTS == MAX_ACTIVE_ATOMS")
            if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
                errors.append("V552-R4.20.1 requires V551_MAX_ATOMS_PER_SLOT=1")
            teacher_min = int(_cfg_get(m1, "V538_TEACHER_MIN_PIXELS", -1))
            atom_min = int(_cfg_get(m1, "V551_ATOM_MIN_PIXELS", -2))
            if teacher_min < 1 or teacher_min != atom_min:
                errors.append("V552-R4.20.1 requires one physical min-pixel teacher/atom contract")
            channels420 = int(_cfg_get(m1, "V552R420_DYNAMIC_CHANNELS", 0))
            if not 4 <= channels420 <= 32:
                errors.append("V552-R4.20.1 DYNAMIC_CHANNELS must be in [4,32]")
            if (
                not bool(_cfg_get(m1, "V552R4201_ABLATION_MODE", False))
                and not bool(_cfg_get(m1, "V552R420_TYPE_DECOUPLED_MASK_ENABLED", False))
            ):
                errors.append("V552-R4.20.1 FULL requires type/shape decoupling")
            selection_metric = str(
                _cfg_get(_cfg_get(cfg, "TRAIN", None), "VAL_SELECTION_METRIC", "")
            ).strip().lower()
            if selection_metric != "native_m2_dice":
                errors.append("V552-R4.20.1 requires TRAIN.VAL_SELECTION_METRIC=native_m2_dice")

            if errors:
                raise ValueError(
                    "V552-R4.20.1 clean protocol validation failed:\n- "
                    + "\n- ".join(errors)
                )
            return
        if v552_protocol in {
            "teacher_decoupled_outcome_calibrated_composer_v3",
            "root_calibrated_teacher_forced_composer_v4",
            "unified_reference_contract_v5",
            "decoupled_critic_contract_v6",
            "spatial_evidence_action_aware_contract_v7",
            "audit_gate_class_value_contract_v8",
            "error_aware_factorized_safe_contract_v9",
            "single_native_action_realizable_contract_v10",
            "spatially_anchored_query_mask_contract_v11",
            "iterative_denoising_component_decoder_contract_v12",
            "content_selective_residual_decoder_contract_v13",
            "evidence_proposed_local_reconstruction_contract_v14",
            "typed_native_residual_set_refiner_contract_v15",
            "paired_stable_box_free_residual_mask_set_contract_v18",
            "location_conditioned_dynamic_residual_mask_contract_v20",
        }:
            if not bool(_cfg_get(m1, "V552R2_TEACHER_DECOUPLED_ENABLED", False)):
                errors.append("V552-R2/R3 requires V552R2_TEACHER_DECOUPLED_ENABLED=true")
            teacher_pool = int(_cfg_get(m1, "V552_COMPOSER_TEACHER_POOL_SIZE", 0))
            deploy_pool = int(_cfg_get(m1, "V552_COMPOSER_DEPLOY_POOL_SIZE", 0))
            if not 1 <= deploy_pool <= teacher_pool <= 6:
                errors.append("V552-R2/R3 requires 1 <= deploy pool <= teacher pool <= 6")
            if float(_cfg_get(m1, "V551_EDITOR_PRESERVE_BIAS", 0.0)) < 2.0:
                errors.append("V552-R2/R3 requires V551_EDITOR_PRESERVE_BIAS >= 2.0")

        if v552_protocol == "root_calibrated_teacher_forced_composer_v4":
            if not bool(_cfg_get(
                m1, "V552R3_ROOT_CALIBRATED_ROUTING_ENABLED", False
            )):
                errors.append(
                    "V552-R3 requires V552R3_ROOT_CALIBRATED_ROUTING_ENABLED=true"
                )
            positive_r3 = (
                "V552R3_EDITOR_TO_BASE_OBJECTIVE_RATIO",
                "V552R3_OUTCOME_TO_BASE_OBJECTIVE_RATIO",
                "V552R3_COMPOSER_TO_BASE_OBJECTIVE_RATIO",
                "V552R3_GAIN_UNIT",
                "V552R3_BENEFIT_SIGN_MARGIN",
                "V552R3_HARM_SIGN_MARGIN",
            )
            for key in positive_r3:
                if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                    errors.append(f"V552-R3 requires M1.{key} > 0")
            max_useful = int(_cfg_get(m1, "V552R3_MAX_USEFUL_ATOMS", 0))
            max_active_r3 = int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", 0))
            if not 1 <= max_useful <= max_active_r3:
                errors.append(
                    "V552-R3 requires 1 <= V552R3_MAX_USEFUL_ATOMS "
                    "<= V551_MAX_ACTIVE_ATOMS"
                )
        if v552_protocol in {
            "unified_reference_contract_v5",
            "decoupled_critic_contract_v6",
            "spatial_evidence_action_aware_contract_v7",
            "audit_gate_class_value_contract_v8",
            "error_aware_factorized_safe_contract_v9",
            "single_native_action_realizable_contract_v10",
            "spatially_anchored_query_mask_contract_v11",
            "iterative_denoising_component_decoder_contract_v12",
            "content_selective_residual_decoder_contract_v13",
            "evidence_proposed_local_reconstruction_contract_v14",
            "typed_native_residual_set_refiner_contract_v15",
            "paired_stable_box_free_residual_mask_set_contract_v18",
            "location_conditioned_dynamic_residual_mask_contract_v20",
        }:
            if not bool(_cfg_get(
                m1, "V552R4_UNIFIED_REFERENCE_CONTRACT_ENABLED", False
            )):
                errors.append(
                    "V552-R4 requires V552R4_UNIFIED_REFERENCE_CONTRACT_ENABLED=true"
                )
            if not bool(_cfg_get(
                m1, "V552R3_ROOT_CALIBRATED_ROUTING_ENABLED", False
            )):
                errors.append(
                    "V552-R4 requires independent root-calibrated routing"
                )
            if bool(_cfg_get(
                m1, "V549_FACTORIZED_DEPLOYMENT_ENABLED", True
            )):
                errors.append(
                    "V552-R4 forbids intersection with V549 factorized deployment"
                )
            if float(_cfg_get(
                m1, "V551_EDITOR_POSITIVE_WEIGHT", 1.0
            )) > 2.0:
                errors.append(
                    "V552-R4 requires V551_EDITOR_POSITIVE_WEIGHT <= 2.0"
                )
            queue_capacity = int(_cfg_get(
                m1, "V552R4_CRITIC_QUEUE_CAPACITY", 0
            ))
            if queue_capacity < 64:
                errors.append(
                    "V552-R4 requires V552R4_CRITIC_QUEUE_CAPACITY >= 64"
                )
            max_useful_r4 = int(_cfg_get(
                m1, "V552R4_MAX_USEFUL_ATOMS", 0
            ))
            max_active_r4 = int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", 0))
            if not 1 <= max_useful_r4 <= max_active_r4:
                errors.append(
                    "V552-R4 requires 1 <= V552R4_MAX_USEFUL_ATOMS "
                    "<= V551_MAX_ACTIVE_ATOMS"
                )
            if float(_cfg_get(
                m1, "V552_EDITOR_RELATIVE_MARGIN", 0.0
            )) <= 0.0:
                errors.append(
                    "V552-R4 requires a positive Editor relative margin"
                )
        if v552_protocol in {
            "decoupled_critic_contract_v6",
            "spatial_evidence_action_aware_contract_v7",
            "audit_gate_class_value_contract_v8",
            "error_aware_factorized_safe_contract_v9",
            "single_native_action_realizable_contract_v10",
            "spatially_anchored_query_mask_contract_v11",
            "iterative_denoising_component_decoder_contract_v12",
            "content_selective_residual_decoder_contract_v13",
            "evidence_proposed_local_reconstruction_contract_v14",
            "typed_native_residual_set_refiner_contract_v15",
            "paired_stable_box_free_residual_mask_set_contract_v18",
            "location_conditioned_dynamic_residual_mask_contract_v20",
        }:
            required_r42 = (
                "V552R42_DECOUPLED_CRITIC_ENABLED",
                "V552R42_EQUAL_CLASS_MEAN_ENABLED",
                "V552R42_COMPOSER_CRITIC_GRAD_ISOLATION_ENABLED",
                "V552R42_QUOTA_REPLAY_ENABLED",
                "V552R42_COMPOSER_FACTORIZED_STOP_ENABLED",
            )
            for key in required_r42:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.2 requires M1.{key}=true")
            minimum_replay = int(_cfg_get(
                m1, "V552R42_REPLAY_MIN_PER_CLASS", 0
            ))
            queue_capacity = int(_cfg_get(
                m1, "V552R4_CRITIC_QUEUE_CAPACITY", 0
            ))
            if not 1 <= minimum_replay <= queue_capacity:
                errors.append(
                    "V552-R4.2 requires 1 <= V552R42_REPLAY_MIN_PER_CLASS "
                    "<= V552R4_CRITIC_QUEUE_CAPACITY"
                )
            for key in (
                "V552R42_CRITIC_OUTCOME_CE_WEIGHT",
                "V552R42_CRITIC_CONDITIONAL_MAGNITUDE_WEIGHT",
                "V552R42_CRITIC_EXPECTED_GAIN_WEIGHT",
                "V552R42_QUEUE_AUX_WEIGHT",
            ):
                if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                    errors.append(f"V552-R4.2 requires M1.{key} > 0")
            asymmetric_keys = (
                "V552R4_EDITOR_BENEFIT_SIGN_WEIGHT",
                "V552R4_EDITOR_HARM_SIGN_WEIGHT",
                "V552R4_UTILITY_BENEFIT_SIGN_WEIGHT",
                "V552R4_UTILITY_HARM_SIGN_WEIGHT",
            )
            for key in asymmetric_keys:
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(
                        f"V552-R4.2 requires M1.{key}=0; sign is structural"
                    )
        if v552_protocol in {
            "spatial_evidence_action_aware_contract_v7",
            "audit_gate_class_value_contract_v8",
            "error_aware_factorized_safe_contract_v9",
        }:
            required_r43 = (
                "V552R43_ROOTFIX_ENABLED",
                "V552R43_SPATIAL_ROUTE_EVIDENCE_ENABLED",
                "V552R43_ACTION_AWARE_ATOM_TARGET_ENABLED",
                "V552R43_BALANCED_PRESENCE_ENABLED",
                "V552R43_QUEUE_CE_ONLY_ENABLED",
            )
            for key in required_r43:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.3 requires M1.{key}=true")
            safety_margin = float(_cfg_get(m1, "V552R43_SAFETY_CLASS_MARGIN", 0.0))
            utility_margin = float(_cfg_get(m1, "V552R43_UTILITY_CLASS_MARGIN", 0.0))
            if safety_margin <= 0.0 or utility_margin <= 0.0:
                errors.append("V552-R4.3 requires positive Safety/Utility class margins")
            if utility_margin > safety_margin:
                errors.append("V552-R4.3 requires Utility margin <= Safety margin")
            magnitude_start = int(_cfg_get(m1, "V552R43_MAGNITUDE_START_EPOCH", -1))
            expected_start = int(_cfg_get(m1, "V552R43_EXPECTED_GAIN_START_EPOCH", -1))
            outcome_start = int(_cfg_get(m1, "V552_OUTCOME_START_EPOCH", 0))
            if not outcome_start <= magnitude_start <= expected_start:
                errors.append(
                    "V552-R4.3 requires Outcome CE start <= Magnitude start "
                    "<= Expected-Gain start"
                )
            if float(_cfg_get(m1, "V552R42_QUEUE_AUX_WEIGHT", 1.0)) > 0.25:
                errors.append("V552-R4.3 requires V552R42_QUEUE_AUX_WEIGHT <= 0.25")

        if v552_protocol in {"single_native_action_realizable_contract_v10", "spatially_anchored_query_mask_contract_v11", "iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15", "paired_stable_box_free_residual_mask_set_contract_v18", "location_conditioned_dynamic_residual_mask_contract_v20"}:
            required_r46 = (
                "V552R46_ROOTFIX_ENABLED",
                "V552R46_NATIVE_STATE_ONLY_ENABLED",
                "V552R46_SINGLE_M1_TEACHER_ENABLED",
                "V552R46_ONE_STEP_COMPOSER_ENABLED",
                "V552R46_FORMAL_POLICY_AUDIT_ENABLED",
                "V552R45_FACTORIZED_SAFETY_ENABLED",
                "V552R45_DIRECT_SIGNED_UTILITY_ENABLED",
                "V552R45_FACTORIZED_COMPOSER_ENABLED",
            )
            for key in required_r46:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.6 requires M1.{key}=true")
            if bool(_cfg_get(m1, "V547_PAIRED_RESIDUAL_REPLAY_ENABLED", True)):
                errors.append("V552-R4.6 requires paired residual replay disabled")
            if bool(_cfg_get(m1, "V538_RESIDUAL_REPLAY_ENABLED", True)):
                errors.append("V552-R4.6 requires residual replay disabled")
            if bool(_cfg_get(m1, "V552R45_ERROR_AWARE_ENABLED", True)):
                errors.append("V552-R4.6 removes R4.5 binary EPR from the main path")
            for key in (
                "V552R3_USEFUL_ATOM_PRESENCE_WEIGHT",
                "V552R3_CARDINALITY_WEIGHT",
                "V552R3_FALSE_ATOM_WEIGHT",
                "V552R43_FALSE_ATOM_WEIGHT",
                "V551_SCALE_LOSS_WEIGHT",
                "V552_ATOM_QUALITY_WEIGHT",
                "V552R45_UTILITY_SIGN_WEIGHT",
            ):
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(f"V552-R4.6 requires M1.{key}=0")
            if int(_cfg_get(m1, "V538_COMPOSER_MAX_STEPS", -1)) != 1:
                errors.append("V552-R4.6 requires V538_COMPOSER_MAX_STEPS=1")
            max_atoms = int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", 0))
            if int(_cfg_get(m1, "V552_COMPOSER_TEACHER_POOL_SIZE", -1)) != max_atoms:
                errors.append("V552-R4.6 teacher pool must equal V551_MAX_ACTIVE_ATOMS")
            if int(_cfg_get(m1, "V552_COMPOSER_DEPLOY_POOL_SIZE", -1)) != max_atoms:
                errors.append("V552-R4.6 deploy pool must equal V551_MAX_ACTIVE_ATOMS")
            if bool(_cfg_get(m1, "V552R44_SEMANTIC_DEPLOYMENT_ENABLED", True)):
                errors.append("V552-R4.6 requires legacy 3-way semantic deployment disabled")
            if bool(_cfg_get(m1, "V552_UNIFIED_DEPLOYMENT_GATE", True)):
                errors.append("V552-R4.6 requires legacy unified Benefit/Harm gate disabled")
            if float(_cfg_get(m1, "V552R46_QUALITY_MAX_POLICY_HARM_RATE", 2.0)) > 0.15:
                errors.append("V552-R4.6 formal policy harm budget must remain <=0.15")

        if v552_protocol in {"spatially_anchored_query_mask_contract_v11", "iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15"}:
            required_r47 = (
                "V552R47_ROOTFIX_ENABLED",
                "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
                "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
                "V552R47_POINT_MASK_SUPERVISION_ENABLED",
                "V552R47_HYBRID_MATCHING_ENABLED",
            )
            for key in required_r47:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.7 requires M1.{key}=true")
            if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(
                _cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)
            ):
                errors.append(
                    "V552-R4.7 requires one query == one deployable component: "
                    "V538_NUM_COMPONENT_SLOTS must equal V551_MAX_ACTIVE_ATOMS"
                )
            if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
                errors.append("V552-R4.7 requires V551_MAX_ATOMS_PER_SLOT=1")
            if float(_cfg_get(m1, "V552R47_POINT_MASK_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.7 requires positive point-mask supervision")
            if float(_cfg_get(m1, "V552R47_ANCHOR_REG_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.7 requires positive anchor regression supervision")
            if float(_cfg_get(m1, "V552R47_MATCH_ANCHOR_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.7 requires positive hybrid matching anchor weight")
            point_count = int(_cfg_get(m1, "V552R47_POINT_COUNT", 0))
            if point_count < 256:
                errors.append("V552-R4.7 requires at least 256 supervised mask points")
            if bool(_cfg_get(m1, "V552R45_ERROR_AWARE_ENABLED", True)):
                errors.append("V552-R4.7 keeps the obsolete binary EPR out of the main path")

        if v552_protocol in {"iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15"}:
            required_r48 = (
                "V552R48_ROOTFIX_ENABLED",
                "V552R48_ITERATIVE_BINDING_ENABLED",
                "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
                "V552R48_DEEP_SUPERVISION_ENABLED",
                "V552R48_DN_COMPONENT_QUERY_ENABLED",
            )
            for key in required_r48:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.8 requires M1.{key}=true")
            layers = int(_cfg_get(m1, "V552R48_DECODER_LAYERS", 0))
            if layers < 2:
                errors.append("V552-R4.8 requires at least two iterative decoder layers")
            grid = int(_cfg_get(m1, "V552R48_LOCAL_GRID_SIZE", 0))
            if grid < 3 or grid % 2 == 0:
                errors.append("V552-R4.8 requires odd V552R48_LOCAL_GRID_SIZE >= 3")
            if int(_cfg_get(m1, "V552R48_DN_GROUPS", 0)) < 1:
                errors.append("V552-R4.8 requires at least one DN group")
            noise = float(_cfg_get(m1, "V552R48_DN_NOISE_SCALE", -1.0))
            if not (0.0 < noise <= 0.75):
                errors.append("V552-R4.8 requires 0 < DN noise scale <= 0.75")
            for key in (
                "V552R48_DEEP_MASK_WEIGHT",
                "V552R48_DN_MASK_WEIGHT",
                "V552R48_DN_ANCHOR_WEIGHT",
            ):
                if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                    errors.append(f"V552-R4.8 requires positive M1.{key}")
            if bool(_cfg_get(m1, "V538_RESIDUAL_REPLAY_ENABLED", True)):
                errors.append("V552-R4.8 DN supervision must use Native Base, not residual replay")
            if bool(_cfg_get(m1, "V547_PAIRED_RESIDUAL_REPLAY_ENABLED", True)):
                errors.append("V552-R4.8 DN supervision must use Native Base, not paired replay")

        if v552_protocol in {"content_selective_residual_decoder_contract_v13", "typed_native_residual_set_refiner_contract_v15"}:
            required_r49 = (
                "V552R49_ROOTFIX_ENABLED",
                "V552R49_CONTENT_SELECTIVE_ATTENTION_ENABLED",
                "V552R49_ANCHOR_SAMPLING_ONLY_ENABLED",
                "V552R49_DN_CURRICULUM_ENABLED",
            )
            for key in required_r49:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.9 requires M1.{key}=true")
            # R4.9's defining contract: a reference box can decide *where to
            # sample* but cannot write a rectangular segmentation prior.
            if abs(float(_cfg_get(m1, "V552R48_WINDOW_PRIOR_SCALE", 0.0))) > 1.0e-12:
                errors.append("V552-R4.9 requires V552R48_WINDOW_PRIOR_SCALE=0")
            init_scale = float(_cfg_get(m1, "V552R49_ATTENTION_LOGIT_SCALE_INIT", 0.0))
            max_scale = float(_cfg_get(m1, "V552R49_ATTENTION_LOGIT_SCALE_MAX", 0.0))
            if not (1.0 <= init_scale <= max_scale <= 100.0):
                errors.append(
                    "V552-R4.9 requires 1 <= attention logit-scale init <= max <= 100"
                )
            noise_start = float(_cfg_get(m1, "V552R49_DN_NOISE_START", -1.0))
            noise_final = float(_cfg_get(m1, "V552R49_DN_NOISE_FINAL", -1.0))
            if not (0.0 <= noise_start < noise_final <= 0.50):
                errors.append("V552-R4.9 requires 0 <= DN noise start < final <= 0.50")
            if int(_cfg_get(m1, "V552R49_DN_NOISE_RAMP_EPOCHS", 0)) < 1:
                errors.append("V552-R4.9 requires a positive DN noise curriculum length")
            # Base remains jointly trainable, but its residual target must not
            # move faster than the component decoder can learn it.
            if float(_cfg_get(m1, "BASE_LEARNING_RATE", 1.0)) > 3.0e-5 + 1.0e-12:
                errors.append("V552-R4.9 requires BASE_LEARNING_RATE <= 3e-5")

        if v552_protocol == "evidence_proposed_local_reconstruction_contract_v14":
            required_r410 = (
                "V552R410_ROOTFIX_ENABLED",
            )
            for key in required_r410:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.10 requires M1.{key}=true")
            ablation = bool(_cfg_get(m1, "V552R410_ABLATION_MODE", False))
            if not ablation:
                for key in (
                    "V552R410_EVIDENCE_PROPOSAL_ENABLED",
                    "V552R410_SUPPORT_ONLY_LOCAL_READOUT_ENABLED",
                    "V552R410_DN_CLEAN_CURRICULUM_ENABLED",
                ):
                    if not bool(_cfg_get(m1, key, False)):
                        errors.append(f"V552-R4.10 full protocol requires M1.{key}=true")
            nms = int(_cfg_get(m1, "V552R410_PROPOSAL_NMS_KERNEL", 0))
            if nms < 3 or nms % 2 == 0:
                errors.append("V552-R4.10 requires odd proposal NMS kernel >= 3")
            score_threshold = float(_cfg_get(m1, "V552R410_PROPOSAL_SCORE_THRESHOLD", -1.0))
            if not (0.0 <= score_threshold <= 1.0):
                errors.append("V552-R4.10 proposal score threshold must be in [0,1]")
            if float(_cfg_get(m1, "V552R410_SUPPORT_EXPAND", 0.0)) < 1.0:
                errors.append("V552-R4.10 support domain must expand, not shrink, the reference box")
            if float(_cfg_get(m1, "V552R410_SUPPORT_MAX_PENALTY", 0.0)) <= 0.0:
                errors.append("V552-R4.10 requires a positive outside-support suppression penalty")
            if int(_cfg_get(m1, "V552R410_DN_CLEAN_EPOCHS", -1)) < 0:
                errors.append("V552-R4.10 DN clean epochs must be >= 0")
            dn_final = float(_cfg_get(m1, "V552R410_DN_NOISE_FINAL", -1.0))
            if not (0.0 <= dn_final <= 0.30):
                errors.append("V552-R4.10 DN final noise must stay in [0,0.30]")
            if int(_cfg_get(m1, "V552R410_DN_NOISE_RAMP_EPOCHS", 0)) < 1:
                errors.append("V552-R4.10 DN noise ramp must be positive")
            # R4.9 showed anchor matching cost was comparable to the actual
            # low Dice similarities.  Keep geometry as a mild tie-break only.
            if float(_cfg_get(m1, "V552R47_MATCH_ANCHOR_WEIGHT", 1.0)) > 0.15 + 1.0e-12:
                errors.append("V552-R4.10 requires V552R47_MATCH_ANCHOR_WEIGHT <= 0.15")
            if float(_cfg_get(m1, "BASE_LEARNING_RATE", 1.0)) > 3.0e-5 + 1.0e-12:
                errors.append("V552-R4.10 keeps BASE_LEARNING_RATE <= 3e-5")

        if v552_protocol == "paired_stable_box_free_residual_mask_set_contract_v18":
            # R4.18 deliberately reuses the proven R4.8 query-mask decoder,
            # but removes all box/ROI ownership from the native residual mask.
            # Location is only a query seed; one query predicts one full mask.
            required_r418 = (
                "V552R47_ROOTFIX_ENABLED",
                "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
                "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
                "V552R48_ROOTFIX_ENABLED",
                "V552R48_ITERATIVE_BINDING_ENABLED",
                "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
                "V552R48_DN_COMPONENT_QUERY_ENABLED",
                "V552R411_ROOTFIX_ENABLED",
                "V552R411_TYPED_PROPOSAL_ENABLED",
                "V552R411_USE_RAW_NATIVE_MASKS",
                "V552R417_ROOTFIX_ENABLED",
                "V552R418_ROOTFIX_ENABLED",
                "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED",
            )
            for key in required_r418:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.18 requires M1.{key}=true")
            if bool(_cfg_get(m1, "V552R411_LOCAL_ROI_DECODER_ENABLED", True)):
                errors.append("V552-R4.18 requires V552R411_LOCAL_ROI_DECODER_ENABLED=false")
            if bool(_cfg_get(m1, "V552R417_SHARED_OFFSET_ENABLED", True)):
                errors.append("V552-R4.18 requires V552R417_SHARED_OFFSET_ENABLED=false")
            if bool(_cfg_get(m1, "V552R47_POINT_MASK_SUPERVISION_ENABLED", False)):
                errors.append("V552-R4.18 removes sparse point-mask supervision from the main mask objective")
            if bool(_cfg_get(m1, "V552R47_HYBRID_MATCHING_ENABLED", False)):
                errors.append("V552-R4.18 requires mask-only identity matching; hybrid anchor matching must be false")
            if bool(_cfg_get(m1, "V552R48_DEEP_SUPERVISION_ENABLED", False)):
                errors.append("V552-R4.18 uses only the final direct full-mask set objective")
            if bool(_cfg_get(m1, "V552R49_ROOTFIX_ENABLED", False)):
                errors.append("V552-R4.18 bypasses R4.9 anchor-local attention; V552R49_ROOTFIX_ENABLED must be false")
            for key in (
                "V552R412_ROOTFIX_ENABLED",
                "V552R413_ROOTFIX_ENABLED",
                "V552R414_ROOTFIX_ENABLED",
                "V552R415_ROOTFIX_ENABLED",
                "V552R416_ROOTFIX_ENABLED",
            ):
                if bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.18 box-free isolation requires M1.{key}=false")
            for key in (
                "V552R47_POINT_MASK_WEIGHT",
                "V552R47_ANCHOR_REG_WEIGHT",
                "V552R48_DEEP_MASK_WEIGHT",
                "V552R48_DN_MASK_WEIGHT",
                "V552R48_DN_ANCHOR_WEIGHT",
                "V552R411_PROPOSAL_OFFSET_WEIGHT",
            ):
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(f"V552-R4.18 requires M1.{key}=0")
            if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)):
                errors.append("V552-R4.18 requires V538_NUM_COMPONENT_SLOTS == V551_MAX_ACTIVE_ATOMS")
            if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
                errors.append("V552-R4.18 requires V551_MAX_ATOMS_PER_SLOT=1")
            teacher_min = int(_cfg_get(m1, "V538_TEACHER_MIN_PIXELS", -1))
            atom_min = int(_cfg_get(m1, "V551_ATOM_MIN_PIXELS", -2))
            if teacher_min < 1 or teacher_min != atom_min:
                errors.append("V552-R4.18 requires V538_TEACHER_MIN_PIXELS == V551_ATOM_MIN_PIXELS >= 1")
            if (
                not bool(_cfg_get(m1, "V552R418_ABLATION_MODE", False))
                and not bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", False))
            ):
                errors.append("V552-R4.18 FULL requires V552R418_PAIRED_STABLE_TEACHER_ENABLED=true")
            if bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", False)):
                if int(_cfg_get(m1, "V552R418_PAIRED_MIN_PIXELS", 0)) < 1:
                    errors.append("V552-R4.18 requires V552R418_PAIRED_MIN_PIXELS>=1")
                min_pair = float(_cfg_get(m1, "V552R418_MIN_PAIRED_MASK_DICE", -1.0))
                min_native = float(_cfg_get(m1, "V552R418_MIN_NATIVE_MATCH_DICE", -1.0))
                if not (0.0 < min_pair <= 1.0):
                    errors.append("V552-R4.18 MIN_PAIRED_MASK_DICE must be in (0,1]")
                if not (0.0 < min_native <= 1.0):
                    errors.append("V552-R4.18 MIN_NATIVE_MATCH_DICE must be in (0,1]")
            if bool(_cfg_get(m1, "V552R419_ROOTFIX_ENABLED", False)):
                radius = float(_cfg_get(m1, "V552R419_SEED_RADIUS_PX", 0.0))
                kernel = int(_cfg_get(m1, "V552R419_SUPPORT_DILATE_KERNEL", 0))
                penalty = float(_cfg_get(m1, "V552R419_OUTSIDE_LOGIT_PENALTY", 0.0))
                threshold = float(_cfg_get(m1, "V552R419_MASK_THRESHOLD", -1.0))
                if radius < 2.0 or radius > 64.0:
                    errors.append("V552-R4.19 SEED_RADIUS_PX must be in [2,64]")
                if kernel < 1 or kernel > 31 or kernel % 2 == 0:
                    errors.append("V552-R4.19 SUPPORT_DILATE_KERNEL must be odd and in [1,31]")
                if penalty <= 0.0:
                    errors.append("V552-R4.19 requires a positive OUTSIDE_LOGIT_PENALTY")
                if not (0.0 < threshold < 1.0):
                    errors.append("V552-R4.19 MASK_THRESHOLD must be in (0,1)")

        if v552_protocol == "location_conditioned_dynamic_residual_mask_contract_v20":
            # R4.20 root isolation: preserve the proven R4.17 independent
            # location seed and R4.18 box-free set assignment, but replace the
            # R4.19 hard-threshold/frontier mask realization with a CondInst-
            # style location-conditioned dynamic mask head.  Paired supervision
            # remains off in this version so the decoder mechanism is tested as
            # one causal variable before any train/inference conditioning change.
            for key in (
                "V552R47_ROOTFIX_ENABLED",
                "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
                "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
                "V552R48_ROOTFIX_ENABLED",
                "V552R48_ITERATIVE_BINDING_ENABLED",
                "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
                "V552R411_ROOTFIX_ENABLED",
                "V552R411_TYPED_PROPOSAL_ENABLED",
                "V552R411_USE_RAW_NATIVE_MASKS",
                "V552R417_ROOTFIX_ENABLED",
                "V552R418_ROOTFIX_ENABLED",
                "V552R420_ROOTFIX_ENABLED",
                "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED",
            ):
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20 requires M1.{key}=true")
            if bool(_cfg_get(m1, "V552R419_ROOTFIX_ENABLED", False)):
                errors.append("V552-R4.20 replaces the R4.19 hard-frontier decoder; V552R419_ROOTFIX_ENABLED must be false")
            if bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", False)):
                errors.append("V552-R4.20 root-isolation run requires paired teacher=false; paired conditioning is a later causal experiment")
            if bool(_cfg_get(m1, "V552R411_LOCAL_ROI_DECODER_ENABLED", True)):
                errors.append("V552-R4.20 requires V552R411_LOCAL_ROI_DECODER_ENABLED=false")
            if bool(_cfg_get(m1, "V552R417_SHARED_OFFSET_ENABLED", True)):
                errors.append("V552-R4.20 requires V552R417_SHARED_OFFSET_ENABLED=false")
            if bool(_cfg_get(m1, "V552R47_POINT_MASK_SUPERVISION_ENABLED", False)):
                errors.append("V552-R4.20 keeps sparse point-mask supervision off")
            if bool(_cfg_get(m1, "V552R47_HYBRID_MATCHING_ENABLED", False)):
                errors.append("V552-R4.20 requires mask-only identity matching")
            if bool(_cfg_get(m1, "V552R48_DEEP_SUPERVISION_ENABLED", False)):
                errors.append("V552-R4.20 isolates the final dynamic mask objective; deep supervision must be false")
            if bool(_cfg_get(m1, "V552R49_ROOTFIX_ENABLED", False)):
                errors.append("V552-R4.20 does not stack the historical R4.9 attention path")
            for key in (
                "V552R412_ROOTFIX_ENABLED", "V552R413_ROOTFIX_ENABLED",
                "V552R414_ROOTFIX_ENABLED", "V552R415_ROOTFIX_ENABLED",
                "V552R416_ROOTFIX_ENABLED",
            ):
                if bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.20 box-free isolation requires M1.{key}=false")
            for key in (
                "V552R47_POINT_MASK_WEIGHT", "V552R47_ANCHOR_REG_WEIGHT",
                "V552R48_DEEP_MASK_WEIGHT", "V552R48_DN_MASK_WEIGHT",
                "V552R48_DN_ANCHOR_WEIGHT", "V552R411_PROPOSAL_OFFSET_WEIGHT",
                "V552R411_PROPOSAL_SIZE_WEIGHT",
            ):
                if abs(float(_cfg_get(m1, key, 0.0))) > 1.0e-12:
                    errors.append(f"V552-R4.20 requires M1.{key}=0")
            channels420 = int(_cfg_get(m1, "V552R420_DYNAMIC_CHANNELS", 0))
            if channels420 < 4 or channels420 > 32:
                errors.append("V552-R4.20 DYNAMIC_CHANNELS must stay in [4,32]")
            if (
                not bool(_cfg_get(m1, "V552R420_ABLATION_MODE", False))
                and not bool(_cfg_get(m1, "V552R420_TYPE_DECOUPLED_MASK_ENABLED", False))
            ):
                errors.append("V552-R4.20 FULL requires type/shape decoupling; set ABLATION_MODE=true only for A1")
            if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)):
                errors.append("V552-R4.20 requires one bounded query slot per active atom")
            if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
                errors.append("V552-R4.20 requires V551_MAX_ATOMS_PER_SLOT=1")

        if v552_protocol == "typed_native_residual_set_refiner_contract_v15":
            for key in (
                "V552R411_ROOTFIX_ENABLED",
                "V552R411_TYPED_PROPOSAL_ENABLED",
                "V552R411_LOCAL_ROI_DECODER_ENABLED",
                "V552R411_USE_RAW_NATIVE_MASKS",
                "V552R411_UTILITY_AWARE_TEACHER_TOPK",
            ):
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.11 requires M1.{key}=true")
            teacher_min = int(_cfg_get(m1, "V538_TEACHER_MIN_PIXELS", -1))
            atom_min = int(_cfg_get(m1, "V551_ATOM_MIN_PIXELS", -2))
            if teacher_min < 1 or teacher_min != atom_min:
                errors.append(
                    "V552-R4.11 requires one physical residual contract: "
                    "V538_TEACHER_MIN_PIXELS == V551_ATOM_MIN_PIXELS >= 1"
                )
            if float(_cfg_get(m1, "V552R47_ANCHOR_MIN_SIZE", 1.0)) > 0.01:
                errors.append(
                    "V552-R4.11 requires V552R47_ANCHOR_MIN_SIZE <= 0.01 "
                    "so tiny residuals remain representable"
                )
            if bool(_cfg_get(m1, "V552R410_EVIDENCE_PROPOSAL_ENABLED", False)):
                errors.append("V552-R4.11 forbids the detached R4.10 evidence proposal")
            if bool(_cfg_get(m1, "V552R410_SUPPORT_ONLY_LOCAL_READOUT_ENABLED", False)):
                errors.append("V552-R4.11 forbids the R4.10 support-only global readout")
            if int(_cfg_get(m1, "V552R411_PROPOSAL_NMS_KERNEL", 0)) < 3:
                errors.append("V552-R4.11 requires proposal NMS kernel >= 3")
            if int(_cfg_get(m1, "V552R411_ROI_SIZE", 0)) < 24:
                errors.append("V552-R4.11 requires ROI_SIZE >= 24 for tiny residual shape reconstruction")
            if float(_cfg_get(m1, "V552R411_ROI_EXPAND", 0.0)) < 1.0:
                errors.append("V552-R4.11 requires ROI_EXPAND >= 1")
            for key in (
                "V552R411_PROPOSAL_CENTER_WEIGHT",
                "V552R411_PROPOSAL_OFFSET_WEIGHT",
            ):
                if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                    errors.append(f"V552-R4.11 requires positive M1.{key}")
            r413_geometry_lock = bool(_cfg_get(m1, "V552R413_ROOTFIX_ENABLED", False))
            size_weight = float(_cfg_get(m1, "V552R411_PROPOSAL_SIZE_WEIGHT", 0.0))
            if r413_geometry_lock:
                if abs(size_weight) > 1.0e-12:
                    errors.append(
                        "V552-R4.13 requires V552R411_PROPOSAL_SIZE_WEIGHT=0; "
                        "selected-instance log extent is the sole size owner"
                    )
            elif size_weight <= 0.0:
                errors.append("V552-R4.11 requires positive M1.V552R411_PROPOSAL_SIZE_WEIGHT")
            if float(_cfg_get(m1, "V552R411_MATCH_TYPE_WEIGHT", 0.0)) <= 0.0:
                errors.append("V552-R4.11 requires positive V552R411_MATCH_TYPE_WEIGHT")
            if float(_cfg_get(m1, "V552R48_DN_MASK_WEIGHT", 1.0)) > 0.5 + 1.0e-12:
                errors.append(
                    "V552-R4.11 keeps DN auxiliary: V552R48_DN_MASK_WEIGHT must be <= 0.5"
                )
            if bool(_cfg_get(m1, "V552R412_ROOTFIX_ENABLED", False)):
                # R4.12 is a strict post-localization extension of the v15
                # Native-residual contract.  It keeps the successful typed
                # locator and makes geometry/shape independently identifiable.
                if int(_cfg_get(m1, "V552R412_CANONICAL_ROI_SIZE", 0)) < 48:
                    errors.append("V552-R4.12 requires CANONICAL_ROI_SIZE >= 48")
                support_strength = float(_cfg_get(m1, "V552R412_ACTION_SUPPORT_STRENGTH", -1.0))
                if not (0.0 < support_strength <= 2.0):
                    errors.append("V552-R4.12 action-support strength must be in (0,2]")
                support_floor = float(_cfg_get(m1, "V552R412_ACTION_SUPPORT_FLOOR", -1.0))
                if not (0.0 < support_floor <= 0.25):
                    errors.append("V552-R4.12 action-support floor must be in (0,0.25]")
                band = int(_cfg_get(m1, "V552R412_BOUNDARY_BAND_KERNEL", 0))
                if band < 3 or band % 2 == 0:
                    errors.append("V552-R4.12 boundary-band kernel must be odd and >=3")
                for key in (
                    "V552R412_CANONICAL_MASK_WEIGHT",
                    "V552R412_BOX_GIOU_WEIGHT",
                    "V552R412_ORACLE_SHAPE_WEIGHT",
                    "V552R412_MATCH_CENTER_WEIGHT",
                    "V552R412_MATCH_BOX_WEIGHT",
                    "V552R412_MATCH_GIOU_WEIGHT",
                ):
                    if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                        errors.append(f"V552-R4.12 requires positive M1.{key}")
                global_weight = float(_cfg_get(m1, "V552R412_GLOBAL_MASK_WEIGHT", -1.0))
                if not (0.0 <= global_weight <= 0.5):
                    errors.append("V552-R4.12 GLOBAL_MASK_WEIGHT must stay in [0,0.5]")
                mask_match = float(_cfg_get(m1, "V552R412_MATCH_MASK_WEIGHT", -1.0))
                if not (0.0 <= mask_match <= 0.5):
                    errors.append("V552-R4.12 MATCH_MASK_WEIGHT must stay in [0,0.5]")
                if not bool(_cfg_get(m1, "V552R48_DN_COMPONENT_QUERY_ENABLED", False)):
                    errors.append("V552-R4.12 requires training-only oracle-geometry DN queries")
                if bool(_cfg_get(m1, "V552R413_ROOTFIX_ENABLED", False)):
                    if not bool(_cfg_get(m1, "V552R413_QUERY_EXTENT_ENABLED", False)):
                        errors.append("V552-R4.13 requires selected-instance query extent enabled")
                    if float(_cfg_get(m1, "V552R413_LOG_SIZE_WEIGHT", 0.0)) <= 0.0:
                        errors.append("V552-R4.13 requires positive log-size supervision")
                    if float(_cfg_get(m1, "V552R413_MIN_PROPOSAL_BOX_IOU", 0.0)) <= 0.0:
                        errors.append("V552-R4.13 requires positive proposal-box IoU readiness")
                    max_drift = float(_cfg_get(m1, "V552R413_MAX_BOX_DRIFT_L1", -1.0))
                    if not (0.0 <= max_drift <= 1.0e-3):
                        errors.append("V552-R4.13 requires MAX_BOX_DRIFT_L1 in [0,1e-3]")
                    if bool(_cfg_get(m1, "V552R414_ROOTFIX_ENABLED", False)):
                        grid414 = int(_cfg_get(m1, "V552R414_CONTEXT_GRID_SIZE", 0))
                        if grid414 < 5 or grid414 % 2 == 0:
                            errors.append("V552-R4.14 requires odd CONTEXT_GRID_SIZE >= 5")
                        radius414 = float(_cfg_get(m1, "V552R414_CONTEXT_RADIUS", -1.0))
                        if not (0.01 <= radius414 <= 0.25):
                            errors.append("V552-R4.14 requires CONTEXT_RADIUS in [0.01,0.25]")
                        extent_type_weight = float(_cfg_get(m1, "V552R414_EXTENT_MATCH_TYPE_WEIGHT", -1.0))
                        if not (0.0 < extent_type_weight <= 0.5):
                            errors.append("V552-R4.14 requires type to be a positive <=0.5 tie-break in center-first extent matching")
                        if bool(_cfg_get(m1, "V552R415_ROOTFIX_ENABLED", False)):
                            min_gate = float(_cfg_get(m1, "V552R415_MIN_CENTER_GATE_PX", -1.0))
                            max_gate = float(_cfg_get(m1, "V552R415_MAX_CENTER_GATE_PX", -1.0))
                            diag_ratio = float(_cfg_get(m1, "V552R415_CENTER_GATE_DIAG_RATIO", -1.0))
                            type_penalty = float(_cfg_get(m1, "V552R415_TYPE_MISMATCH_PENALTY", -1.0))
                            min_match = float(_cfg_get(m1, "V552R415_MIN_IDENTITY_MATCH_RATE", -1.0))
                            max_p90 = float(_cfg_get(m1, "V552R415_MAX_MATCHED_CENTER_ERROR_P90_PX", -1.0))
                            if not (1.0 <= min_gate <= max_gate <= 32.0):
                                errors.append("V552-R4.15 requires 1<=MIN_CENTER_GATE_PX<=MAX_CENTER_GATE_PX<=32")
                            if not (0.1 <= diag_ratio <= 2.0):
                                errors.append("V552-R4.15 requires CENTER_GATE_DIAG_RATIO in [0.1,2.0]")
                            if not (0.0 <= type_penalty < 1.0):
                                errors.append("V552-R4.15 requires TYPE_MISMATCH_PENALTY in [0,1)")
                            if not (0.1 <= min_match <= 0.95):
                                errors.append("V552-R4.15 requires MIN_IDENTITY_MATCH_RATE in [0.1,0.95]")
                            if not (min_gate <= max_p90 <= 2.0 * max_gate):
                                errors.append("V552-R4.15 matched-center P90 readiness must be physically consistent with the center gate")
                            if bool(_cfg_get(m1, "V552R416_ROOTFIX_ENABLED", False)):
                                if not bool(_cfg_get(m1, "V552R414_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.16 requires R4.14 contextual extent")
                                unique416 = bool(_cfg_get(m1, "V552R416_UNIQUE_POINT_TOPK_ENABLED", False))
                                ltrb416 = bool(_cfg_get(m1, "V552R416_ASYMMETRIC_LTRB_ENABLED", False))
                                radius416 = float(_cfg_get(m1, "V552R416_CROSS_TYPE_NMS_RADIUS_PX", -1.0))
                                if unique416 and not (1.0 <= radius416 <= 12.0):
                                    errors.append("V552-R4.16 cross-type NMS radius must be in [1,12] px")
                                if ltrb416 and float(_cfg_get(m1, "V552R416_LTRB_WEIGHT", 0.0)) <= 0.0:
                                    errors.append("V552-R4.16 asymmetric LTRB requires positive LTRB_WEIGHT")
                                min_pre = float(_cfg_get(m1, "V552R416_MIN_PRE_TOPK_PEAK_RECALL", -1.0))
                                if not (0.0 < min_pre <= 1.0):
                                    errors.append("V552-R4.16 MIN_PRE_TOPK_PEAK_RECALL must be in (0,1]")
                                max_edge = float(_cfg_get(m1, "V552R416_MAX_EDGE_OFFSET_MAE_PX", -1.0))
                                if ltrb416 and not (1.0 <= max_edge <= 64.0):
                                    errors.append("V552-R4.16 MAX_EDGE_OFFSET_MAE_PX must be in [1,64]")
                            if bool(_cfg_get(m1, "V552R417_ROOTFIX_ENABLED", False)):
                                r418_cfg = bool(_cfg_get(m1, "V552R418_ROOTFIX_ENABLED", False))
                                if (not r418_cfg) and not bool(_cfg_get(m1, "V552R415_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.17 requires R4.15 rejectable identity matching unless R4.18 box-free mask-set mode is enabled")
                                if (not r418_cfg) and not bool(_cfg_get(m1, "V552R414_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.17 requires R4.14 contextual extent unless R4.18 box-free mask-set mode is enabled")
                                if bool(_cfg_get(m1, "V552R416_UNIQUE_POINT_TOPK_ENABLED", False)):
                                    errors.append("V552-R4.17 location-first proposal recovery must disable R4.16 cross-type UniqueTopK")
                                if bool(_cfg_get(m1, "V552R416_ASYMMETRIC_LTRB_ENABLED", False)):
                                    errors.append("V552-R4.17 proposal-recovery isolation must disable R4.16 LTRB")
                                nms417 = int(_cfg_get(m1, "V552R417_LOCATION_NMS_KERNEL", 3))
                                over417 = int(_cfg_get(m1, "V552R417_LOCATION_OVERSAMPLE_FACTOR", 4))
                                dedup417 = float(_cfg_get(m1, "V552R417_LOCATION_DEDUP_RADIUS_PX", 2.0))
                                if nms417 < 1 or nms417 % 2 == 0 or nms417 > 9:
                                    errors.append("V552-R4.17 LOCATION_NMS_KERNEL must be odd in [1,9]")
                                if not (1 <= over417 <= 16):
                                    errors.append("V552-R4.17 LOCATION_OVERSAMPLE_FACTOR must be in [1,16]")
                                if not (0.0 <= dedup417 <= 8.0):
                                    errors.append("V552-R4.17 LOCATION_DEDUP_RADIUS_PX must be in [0,8]")
                                min_peak417 = float(_cfg_get(m1, "V552R417_MIN_PRE_TOPK_PEAK_RECALL", -1.0))
                                min_cov417 = float(_cfg_get(m1, "V552R417_MIN_SELECTED_SPATIAL_COVERAGE", -1.0))
                                max_off417 = float(_cfg_get(m1, "V552R417_MAX_LOCATION_OFFSET_MAE_PX", -1.0))
                                if not (0.0 < min_peak417 <= 1.0):
                                    errors.append("V552-R4.17 MIN_PRE_TOPK_PEAK_RECALL must be in (0,1]")
                                if not (0.0 < min_cov417 <= 1.0):
                                    errors.append("V552-R4.17 MIN_SELECTED_SPATIAL_COVERAGE must be in (0,1]")
                                if bool(_cfg_get(m1, "V552R417_SHARED_OFFSET_ENABLED", True)) and not (0.5 <= max_off417 <= 32.0):
                                    errors.append("V552-R4.17 MAX_LOCATION_OFFSET_MAE_PX must be in [0.5,32]")
                            if bool(_cfg_get(m1, "V552R418_ROOTFIX_ENABLED", False)):
                                if not bool(_cfg_get(m1, "V552R48_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.18 requires the R4.8 query-mask decoder")
                                if not bool(_cfg_get(m1, "V552R411_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.18 requires R4.11 typed/location query seeds")
                                if not bool(_cfg_get(m1, "V552R417_ROOTFIX_ENABLED", False)):
                                    errors.append("V552-R4.18 requires R4.17 independent location seeding")
                                if bool(_cfg_get(m1, "V552R417_SHARED_OFFSET_ENABLED", True)):
                                    errors.append("V552-R4.18 must disable the failed R4.17 shared offset")
                                if bool(_cfg_get(m1, "V552R411_LOCAL_ROI_DECODER_ENABLED", True)):
                                    errors.append("V552-R4.18 must disable ROI/box-owned mask rendering")
                                for key in ("V552R412_ROOTFIX_ENABLED", "V552R413_ROOTFIX_ENABLED", "V552R414_ROOTFIX_ENABLED", "V552R415_ROOTFIX_ENABLED", "V552R416_ROOTFIX_ENABLED"):
                                    if bool(_cfg_get(m1, key, False)):
                                        errors.append(f"V552-R4.18 box-free isolation requires M1.{key}=false")
                                if not bool(_cfg_get(m1, "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED", False)):
                                    errors.append("V552-R4.18 requires one-to-one optimal mask-set matching")
                                if (
                                    not bool(_cfg_get(m1, "V552R418_ABLATION_MODE", False))
                                    and not bool(_cfg_get(m1, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", True))
                                ):
                                    errors.append("V552-R4.18 FULL requires paired stable teacher supervision")

        if v552_protocol == "error_aware_factorized_safe_contract_v9":
            required_r45 = (
                "V552R44_AUDIT_GATE_ROOTFIX_ENABLED",
                "V552R44_ACTION_AWARE_QUALITY_GATE_ENABLED",
                "V552R44_CLASS_VALUE_DECOUPLING_ENABLED",
                "V552R44_SEMANTIC_DEPLOYMENT_ENABLED",
                "V552R45_ROOTFIX_ENABLED",
                "V552R45_ERROR_AWARE_ENABLED",
                "V552R45_FACTORIZED_SAFETY_ENABLED",
                "V552R45_DIRECT_SIGNED_UTILITY_ENABLED",
                "V552R45_FACTORIZED_COMPOSER_ENABLED",
                "V552R45_DIRECT_AUDIT_GATE_ENABLED",
            )
            for key in required_r45:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.5 requires M1.{key}=true")
            alpha = float(_cfg_get(m1, "V552R45_TVERSKY_ALPHA", -1.0))
            beta = float(_cfg_get(m1, "V552R45_TVERSKY_BETA", -1.0))
            if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and beta > alpha):
                errors.append(
                    "V552-R4.5 requires 0<=Tversky alpha<beta<=1 for the current low-recall failure"
                )
            for key in (
                "V552R45_ERROR_BCE_WEIGHT",
                "V552R45_ERROR_TVERSKY_WEIGHT",
                "V552R45_SAFETY_BENEFIT_WEIGHT",
                "V552R45_SAFETY_HARM_WEIGHT",
                "V552R45_UTILITY_SIGN_WEIGHT",
                "V552R45_UTILITY_VALUE_WEIGHT",
                "V552R45_COMPOSER_EXECUTE_WEIGHT",
                "V552R45_COMPOSER_CHOICE_WEIGHT",
                "V552R45_COMPOSER_VALUE_WEIGHT",
            ):
                if float(_cfg_get(m1, key, 0.0)) <= 0.0:
                    errors.append(f"V552-R4.5 requires M1.{key}>0")
            if bool(_cfg_get(m1, "V552R44_COMPOSER_SELECTIVE_RISK_ENABLED", True)):
                errors.append(
                    "V552-R4.5 requires R4.4 selective-risk objective disabled; risk remains diagnostic only"
                )
            if float(_cfg_get(m1, "V552R45_QUALITY_MAX_POLICY_HARM_RATE", 2.0)) > 0.15:
                errors.append("V552-R4.5 policy harm budget must remain <=0.15")

        if v552_protocol == "audit_gate_class_value_contract_v8":
            required_r44 = (
                "V552R44_AUDIT_GATE_ROOTFIX_ENABLED",
                "V552R44_ACTION_AWARE_QUALITY_GATE_ENABLED",
                "V552R44_CLASS_VALUE_DECOUPLING_ENABLED",
                "V552R44_SEMANTIC_DEPLOYMENT_ENABLED",
                "V552R44_COMPOSER_SELECTIVE_RISK_ENABLED",
            )
            for key in required_r44:
                if not bool(_cfg_get(m1, key, False)):
                    errors.append(f"V552-R4.4 requires M1.{key}=true")
            audit_start = int(_cfg_get(
                m1, "V552R44_AUDIT_SHADOW_START_EPOCH", -1
            ))
            quality_start = int(_cfg_get(
                m1, "V552R44_QUALITY_START_EPOCH", -1
            ))
            if audit_start < 0 or quality_start < audit_start:
                errors.append(
                    "V552-R4.4 requires 0 <= Audit start <= Quality start"
                )
            if int(_cfg_get(m1, "V552R44_AUDIT_SHADOW_TOPK", 0)) != 1:
                errors.append("V552-R4.4 audit must force exactly Top-1")
            if float(_cfg_get(
                m1, "V552R44_MIN_ACTION_CORRECTION_PRECISION", 0.0
            )) <= 0.0:
                errors.append("V552-R4.4 requires positive action precision gate")
            if float(_cfg_get(
                m1, "V552R44_COMPOSER_TARGET_COVERAGE", 0.0
            )) <= 0.0:
                errors.append("V552-R4.4 requires positive Composer coverage target")
            if int(_cfg_get(
                m1, "V552R44_QUALITY_MIN_POLICY_CASES", 0
            )) <= 0:
                errors.append("V552-R4.4 requires positive Policy-Audit sample count")
            policy_harm = float(_cfg_get(
                m1, "V552R44_QUALITY_MAX_AUDIT_HARM_RATE", -1.0
            ))
            candidate_harm = float(_cfg_get(
                m1, "V552R44_MAX_CANDIDATE_AUDIT_HARM", -1.0
            ))
            if not 0.0 <= policy_harm <= candidate_harm <= 1.0:
                errors.append(
                    "V552-R4.4 requires 0 <= Policy harm budget "
                    "<= Candidate harm budget <= 1"
                )
            if float(_cfg_get(
                m1, "V552R44_COMPOSER_SELECTIVE_RISK_WEIGHT", 0.0
            )) <= 0.0:
                errors.append("V552-R4.4 requires positive selective-risk weight")
            if float(_cfg_get(
                m1, "V552R44_COMPOSER_COVERAGE_WEIGHT", 0.0
            )) <= 0.0:
                errors.append("V552-R4.4 requires positive coverage weight")

        if (
            v552_protocol not in {"single_native_action_realizable_contract_v10", "spatially_anchored_query_mask_contract_v11", "iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15", "paired_stable_box_free_residual_mask_set_contract_v18", "location_conditioned_dynamic_residual_mask_contract_v20"}
            and not bool(_cfg_get(m1, "V552_UNIFIED_DEPLOYMENT_GATE", False))
        ):
            errors.append("M1.V552_UNIFIED_DEPLOYMENT_GATE must be true")
        if int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", 99)) > 6:
            errors.append("V552 requires V551_MAX_ACTIVE_ATOMS <= 6")
        if float(_cfg_get(m1, "V551_EDITOR_DOSE_ADJUST_MAX", 9.0)) > 1.15:
            errors.append("V552 requires V551_EDITOR_DOSE_ADJUST_MAX <= 1.15")
        if not bool(_cfg_get(m1, "V552_MULTICANDIDATE_COMPOSER_ENABLED", False)):
            errors.append("V552-R1 requires V552_MULTICANDIDATE_COMPOSER_ENABLED=true")
        composer_steps = int(_cfg_get(m1, "V538_COMPOSER_MAX_STEPS", 0))
        if v552_protocol in {"single_native_action_realizable_contract_v10", "spatially_anchored_query_mask_contract_v11", "iterative_denoising_component_decoder_contract_v12", "content_selective_residual_decoder_contract_v13", "evidence_proposed_local_reconstruction_contract_v14", "typed_native_residual_set_refiner_contract_v15", "paired_stable_box_free_residual_mask_set_contract_v18", "location_conditioned_dynamic_residual_mask_contract_v20"}:
            if composer_steps != 1:
                errors.append("V552-R4.6 requires V538_COMPOSER_MAX_STEPS=1")
        elif composer_steps < 2 or composer_steps > 3:
            errors.append("V552-R1 requires 2 <= V538_COMPOSER_MAX_STEPS <= 3")
    if not bool(_cfg_get(m1, "V551_GPU_ATOMIZER_ENABLED", False)):
        errors.append("M1.V551_GPU_ATOMIZER_ENABLED must be true")
    if not bool(_cfg_get(m1, "V551_SINGLE_PASS_EDITOR", False)):
        errors.append("M1.V551_SINGLE_PASS_EDITOR must be true")
    max_active = int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", 0))
    if not 2 <= max_active <= 12:
        errors.append("M1.V551_MAX_ACTIVE_ATOMS must be in [2,12]")
    spatial = int(_cfg_get(m1, "V541_SELECTOR_SPATIAL_SIZE", 16))
    if spatial > 8:
        errors.append("M1.V541_SELECTOR_SPATIAL_SIZE must be <= 8 for V551")
    if errors:
        raise ValueError(
            "V551 root-fix protocol failed:\n  - " + "\n  - ".join(errors)
        )


def _validate_v560_clean_core_protocol(cfg):
    """Minimal fail-fast contract for the Clean Residual Set + Utility core.

    V560 intentionally does not inherit the historical R4.x readiness/seed/
    overflow protocol tree.  It keeps the same Base and joint E2E training, but
    assigns exactly one owner to each scientific responsibility: direct query
    masks for M1 geometry, presence for no-object, action for correction type,
    and raw signed DeltaDice for M2 selection.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V560_CLEAN_CORE_ENABLED", False)):
        return
    errors = []
    def need_true(key):
        if not bool(_cfg_get(m1, key, False)):
            errors.append(f"M1.{key} must be true")
    need_true("V532_UNIFIED_SPARSE_REFINER_ENABLED")
    need_true("V535_ADAPTIVE_UTILITY_POLICY_ENABLED")
    need_true("V538_ONLINE_COMPONENT_REFINER_ENABLED")
    need_true("V551_MULTISCALE_TYPED_EDITOR_ENABLED")
    need_true("V551_GPU_ATOMIZER_ENABLED")
    need_true("V551_SINGLE_PASS_EDITOR")
    need_true("V546_OPTIMAL_COMPONENT_MATCHING_ENABLED")
    if str(_cfg_get(m1, "V551_PROTOCOL_VERSION", "")) != "gpu_sparse_single_pass_v1":
        errors.append("V560 requires V551_PROTOCOL_VERSION=gpu_sparse_single_pass_v1")
    if int(_cfg_get(m1, "V538_COMPOSER_MAX_STEPS", 0)) != 1:
        errors.append("V560 requires one-step selection: V538_COMPOSER_MAX_STEPS=1")
    slots = int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0))
    atoms = int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1))
    if slots <= 0 or slots != atoms:
        errors.append("V560 requires V538_NUM_COMPONENT_SLOTS == V551_MAX_ACTIVE_ATOMS > 0")
    if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
        errors.append("V560 requires V551_MAX_ATOMS_PER_SLOT=1")
    if bool(_cfg_get(m1, "V546_SLOT_COMPETITION_ENABLED", False)):
        errors.append("V560 forbids WTA slot competition")
    if (
        bool(_cfg_get(m1, "V544_REGION_BALANCED_MASK_ENABLED", False))
        and not bool(_cfg_get(m1, "V562_ROOTFIX_ENABLED", False))
    ):
        errors.append("V560 uses standard BCEWithLogits+Dice unless V562 owns sparse balanced mask supervision")
    # Historical geometry owners must not silently regain deployment authority.
    for key in (
        "V552R47_ROOTFIX_ENABLED",
        "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED",
        "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED",
        "V552R48_ROOTFIX_ENABLED",
        "V552R48_ITERATIVE_BINDING_ENABLED",
        "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED",
        "V552R419_ROOTFIX_ENABLED", "V552R420_ROOTFIX_ENABLED",
        "V552R4203_DENSE_COMPETITIVE_SET_ENABLED",
        "V552R4204_RESIDUAL_EXISTENCE_IDENTITY_ENABLED",
        "V552R4205_CAPACITY_CONSISTENT_FACTORIZATION_ENABLED",
        "V552R4206_CONDITIONAL_IDENTITY_ENABLED",
        "V552R4207_DYNAMIC_VISUAL_BINDING_ENABLED",
        # V560 owns query identity and mask geometry directly.  Legacy R4.20.8--
        # R4.21.1 subfeature switches must therefore be false even when their
        # historical root switch is false; V551 consumes several of these
        # subfeatures directly at construction time.
        "V552R4208_ROOTFIX_ENABLED",
        "V552R4208_NORMALIZED_FUSION_ENABLED",
        "V552R4208_PERSISTENT_IDENTITY_ENABLED",
        "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED",
        "V552R4209_ROOTFIX_ENABLED",
        "V552R4209_CENTER_PEAK_FOCAL_ENABLED",
        "V552R4209_BALANCED_OVERFLOW_ENABLED",
        "V552R4210_ROOTFIX_ENABLED",
        "V552R4210_INTERIOR_ANCHOR_ENABLED",
        "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED",
        "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED",
        "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED",
        "V552R4211_ROOTFIX_ENABLED",
        "V552R4211_PROPOSAL_EXISTENCE_DECOUPLING_ENABLED",
        "V552R4211_GEOMETRY_OVERFLOW_DECOUPLING_ENABLED",
        "V552R4212_ROOTFIX_ENABLED",
    ):
        if bool(_cfg_get(m1, key, False)):
            errors.append(f"V560 clean path forbids historical owner M1.{key}=true")
    if bool(_cfg_get(m1, "V538_RESIDUAL_REPLAY_ENABLED", True)):
        errors.append("V560 requires factual current-Base Teachers; V538_RESIDUAL_REPLAY_ENABLED must be false")
    if str(_cfg_get(_cfg_get(cfg, "VAL", None), "SELECTION_METRIC", "native_m2_dice")) not in {"native_m2_dice", ""}:
        errors.append("V560 formal selection must remain native_m2_dice")
    if errors:
        raise ValueError("V560 Clean Core protocol failed:\n  - " + "\n  - ".join(errors))



def _validate_v561_bcrs_protocol(cfg):
    """Fail-fast contract for Base-Conditioned Residual Correction Set M1."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V561_BCRS_ENABLED", False)):
        return
    errors = []
    if not bool(_cfg_get(m1, "V560_CLEAN_CORE_ENABLED", False)):
        errors.append("V561 requires V560_CLEAN_CORE_ENABLED=true for the physical one-step M2 contract")
    variant = str(_cfg_get(m1, "V561_BCRS_VARIANT", "")).strip().lower()
    if variant not in {"static", "image", "typed", "rootfix", "persistent"}:
        errors.append("V561_BCRS_VARIANT must be one of: static, image, typed, rootfix, persistent")
    if str(_cfg_get(m1, "V561_PROTOCOL_VERSION", "")) != "base_conditioned_residual_correction_set_v1":
        errors.append("V561_PROTOCOL_VERSION must be base_conditioned_residual_correction_set_v1")
    if bool(_cfg_get(m1, "V552R4212_CANDIDATE_ALIGNMENT_ENABLED", False)):
        errors.append("V561 forbids whole-image candidate-alignment ownership")
    if bool(_cfg_get(m1, "V538_RESIDUAL_REPLAY_ENABLED", True)):
        errors.append("V561 uses factual current-Base residual Teachers; replay must be false")
    for key in (
        "V552R47_ROOTFIX_ENABLED",
        "V552R48_ROOTFIX_ENABLED",
        "V552R48_ITERATIVE_BINDING_ENABLED",
        "V552R419_ROOTFIX_ENABLED",
        "V552R420_ROOTFIX_ENABLED",
        "V552R4203_ROOTFIX_ENABLED",
        "V552R4204_ROOTFIX_ENABLED",
        "V552R4205_ROOTFIX_ENABLED",
        "V552R4206_ROOTFIX_ENABLED",
        "V552R4207_ROOTFIX_ENABLED",
        "V552R4208_ROOTFIX_ENABLED",
        "V552R4209_ROOTFIX_ENABLED",
        "V552R4210_ROOTFIX_ENABLED",
        "V552R4211_ROOTFIX_ENABLED",
        "V552R4212_ROOTFIX_ENABLED",
    ):
        if bool(_cfg_get(m1, key, False)):
            errors.append(f"V561 BCRS forbids historical geometry owner M1.{key}=true")
    if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(
        _cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)
    ):
        errors.append("V561 requires slots == max active atoms")
    if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
        errors.append("V561 requires one executable correction per query slot")
    if errors:
        raise ValueError("V561 BCRS protocol failed:\n  - " + "\n  - ".join(errors))


def _validate_v562_rootfix_protocol(cfg):
    """Fail-fast contract for V562 M1 root repair."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V562_ROOTFIX_ENABLED", False)):
        return
    errors = []
    if not bool(_cfg_get(m1, "V561_BCRS_ENABLED", False)):
        errors.append("V562 requires V561_BCRS_ENABLED=true as the clean set scaffold")
    if str(_cfg_get(m1, "V561_BCRS_VARIANT", "")).strip().lower() not in {"rootfix", "persistent"}:
        errors.append("V562 requires V561_BCRS_VARIANT=rootfix or persistent")
    if str(_cfg_get(m1, "V562_PROTOCOL_VERSION", "")) != "residual_anchor_direct_execution_v1":
        errors.append("V562_PROTOCOL_VERSION must be residual_anchor_direct_execution_v1")
    if bool(_cfg_get(m1, "V538_RESIDUAL_REPLAY_ENABLED", True)):
        errors.append("V562 requires factual current-Base residual Teachers; replay must be false")
    if bool(_cfg_get(m1, "V552R4212_CANDIDATE_ALIGNMENT_ENABLED", False)):
        errors.append("V562 forbids whole-image candidate alignment")
    if int(_cfg_get(m1, "V538_NUM_COMPONENT_SLOTS", 0)) != int(_cfg_get(m1, "V551_MAX_ACTIVE_ATOMS", -1)):
        errors.append("V562 requires one active atom per query slot")
    if int(_cfg_get(m1, "V551_MAX_ATOMS_PER_SLOT", 99)) != 1:
        errors.append("V562 requires V551_MAX_ATOMS_PER_SLOT=1")
    if float(_cfg_get(m1, "V562_MATCH_ACTION_WEIGHT", 0.10)) > 0.25:
        errors.append("V562 action matching weight must stay <=0.25 so geometry owns identity")
    if float(_cfg_get(m1, "V562_MATCH_CENTER_WEIGHT", 0.50)) <= 0.0:
        errors.append("V562 requires positive proposal-center matching weight")
    if float(_cfg_get(m1, "V562_RESIDUAL_PROPOSAL_WEIGHT", 1.0)) <= 0.0:
        errors.append("V562 residual proposal supervision must be active")
    if float(_cfg_get(m1, "V562_LOCAL_EXECUTION_WEIGHT", 1.0)) <= 0.0:
        errors.append("V562 local executable-correction supervision must be active")
    if errors:
        raise ValueError("V562 root-fix protocol failed:\n  - " + "\n  - ".join(errors))


def _validate_v563_rootfix_protocol(cfg):
    """Fail-fast contract for persistent spatial instance binding."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V563_ROOTFIX_ENABLED", False)):
        return
    errors = []
    if not bool(_cfg_get(m1, "V562_ROOTFIX_ENABLED", False)):
        errors.append("V563 extends V562 proposal/executor and requires V562_ROOTFIX_ENABLED=true")
    if str(_cfg_get(m1, "V561_BCRS_VARIANT", "")).strip().lower() != "persistent":
        errors.append("V563 requires V561_BCRS_VARIANT=persistent")
    if str(_cfg_get(m1, "V563_PROTOCOL_VERSION", "")) != "persistent_anchor_hard_local_v1":
        errors.append("V563_PROTOCOL_VERSION must be persistent_anchor_hard_local_v1")
    attention_radius = float(_cfg_get(m1, "V563_ATTENTION_RADIUS", 0.24))
    mask_radius = float(_cfg_get(m1, "V563_MASK_RADIUS", 0.20))
    if not (0.05 <= mask_radius <= attention_radius <= 0.50):
        errors.append("V563 requires 0.05 <= MASK_RADIUS <= ATTENTION_RADIUS <= 0.50")
    identity_mix = float(_cfg_get(m1, "V563_IDENTITY_MIX", 0.60))
    if not (0.30 <= identity_mix <= 0.90):
        errors.append("V563_IDENTITY_MIX must stay in [0.30,0.90] to preserve anchor identity")
    residual_scale = float(_cfg_get(m1, "V563_QUERY_RESIDUAL_SCALE", 0.15))
    if not (0.0 < residual_scale <= 0.30):
        errors.append("V563_QUERY_RESIDUAL_SCALE must stay in (0,0.30]")
    if float(_cfg_get(m1, "V563_OUTSIDE_LOGIT_PENALTY", 12.0)) < 8.0:
        errors.append("V563_OUTSIDE_LOGIT_PENALTY must be >=8 for hard extent safety")
    if float(_cfg_get(m1, "V563_OUTSIDE_MASK_WEIGHT", 1.0)) <= 0.0:
        errors.append("V563_OUTSIDE_MASK_WEIGHT must be positive")
    if bool(_cfg_get(m1, "V544_REGION_BALANCED_MASK_ENABLED", False)):
        errors.append("V563 disables V562 whole-image region-balanced BCE; use local-window BCE+Dice")
    if errors:
        raise ValueError("V563 root-fix protocol failed:\n  - " + "\n  - ".join(errors))



def _validate_v564_rootfix_protocol(cfg):
    """Fail-fast contract for ownership-consistent instance correction."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V564_ROOTFIX_ENABLED", False)):
        return
    errors = []
    if not bool(_cfg_get(m1, "V563_ROOTFIX_ENABLED", False)):
        errors.append("V564 extends V563 hard-local safety and requires V563_ROOTFIX_ENABLED=true")
    if str(_cfg_get(m1, "V561_BCRS_VARIANT", "")).strip().lower() != "persistent":
        errors.append("V564 requires V561_BCRS_VARIANT=persistent")
    if str(_cfg_get(m1, "V564_PROTOCOL_VERSION", "")) != "ownership_consistent_dual_stream_v1":
        errors.append("V564_PROTOCOL_VERSION must be ownership_consistent_dual_stream_v1")
    if abs(float(_cfg_get(m1, "V562_MATCH_ACTION_WEIGHT", 0.0))) > 1.0e-12:
        errors.append("V564 forbids action-dependent teacher ownership; set V562_MATCH_ACTION_WEIGHT=0")
    if (not bool(_cfg_get(m1, "V565_ROOTFIX_ENABLED", False))) and float(_cfg_get(m1, "V564_COMPONENT_SEED_WEIGHT", 0.50)) <= 0.0:
        errors.append("V564_COMPONENT_SEED_WEIGHT must be positive unless V565 replaces it")
    if float(_cfg_get(m1, "V564_PROPOSAL_SHAPE_SCALE", 0.50)) <= 0.0:
        errors.append("V564_PROPOSAL_SHAPE_SCALE must be positive so dense proposal shape reaches the mask")
    identity_scale = float(_cfg_get(m1, "V564_ATTENTION_IDENTITY_SCALE", 0.35))
    if not (0.05 <= identity_scale <= 0.80):
        errors.append("V564_ATTENTION_IDENTITY_SCALE must stay in [0.05,0.80]")
    min_radius = float(_cfg_get(m1, "V564_MIN_MASK_RADIUS", 0.05))
    max_radius = float(_cfg_get(m1, "V564_MAX_MASK_RADIUS", 0.20))
    attention_radius = float(_cfg_get(m1, "V563_ATTENTION_RADIUS", 0.24))
    if not (0.02 <= min_radius <= max_radius <= attention_radius):
        errors.append("V564 requires 0.02 <= MIN_MASK_RADIUS <= MAX_MASK_RADIUS <= V563_ATTENTION_RADIUS")
    if bool(_cfg_get(m1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)):
        errors.append("V564 owns matching directly; legacy R4.20.8 partial fallback matching must be disabled")
    if bool(_cfg_get(m1, "V544_REGION_BALANCED_MASK_ENABLED", False)):
        errors.append("V564 retains V563 local-window BCE+Dice and forbids whole-image balanced mask BCE")
    if errors:
        raise ValueError("V564 complete root-fix protocol failed:\n  - " + "\n  - ".join(errors))

def _validate_v565_rootfix_protocol(cfg):
    """Fail-fast contract for instance-coverage residual seed set."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "V565_ROOTFIX_ENABLED", False)):
        return
    errors = []
    if not bool(_cfg_get(m1, "V564_ROOTFIX_ENABLED", False)):
        errors.append("V565 requires V564 strict ownership / feasible local supervision")
    if str(_cfg_get(m1, "V565_PROTOCOL_VERSION", "")) != "instance_coverage_seed_set_v1":
        errors.append("V565_PROTOCOL_VERSION must be instance_coverage_seed_set_v1")
    if float(_cfg_get(m1, "V564_COMPONENT_SEED_WEIGHT", 0.0)) != 0.0:
        errors.append("V565 replaces V564 component-peak BCE; set V564_COMPONENT_SEED_WEIGHT=0")
    if float(_cfg_get(m1, "V565_SEED_HEATMAP_WEIGHT", 1.0)) <= 0.0:
        errors.append("V565_SEED_HEATMAP_WEIGHT must be positive")
    if float(_cfg_get(m1, "V565_SEED_RANK_WEIGHT", 0.25)) <= 0.0:
        errors.append("V565_SEED_RANK_WEIGHT must be positive")
    if float(_cfg_get(m1, "V565_SEED_RANK_MARGIN", 0.50)) <= 0.0:
        errors.append("V565_SEED_RANK_MARGIN must be positive")
    nms = float(_cfg_get(m1, "V565_SEED_NMS_RADIUS", 0.025))
    min_r = float(_cfg_get(m1, "V564_MIN_MASK_RADIUS", 0.05))
    if not (0.005 <= nms <= min_r):
        errors.append("V565_SEED_NMS_RADIUS must be small and <= V564_MIN_MASK_RADIUS")
    rel = float(_cfg_get(m1, "V565_SUPPORT_RELATIVE_THRESHOLD", 0.35))
    if not (0.10 <= rel <= 0.90):
        errors.append("V565_SUPPORT_RELATIVE_THRESHOLD must stay in [0.10,0.90]")
    quant = float(_cfg_get(m1, "V565_EXTENT_QUANTILE", 0.90))
    if not (0.50 <= quant <= 0.99):
        errors.append("V565_EXTENT_QUANTILE must stay in [0.50,0.99]")
    max_att = float(_cfg_get(m1, "V565_MAX_ATTENTION_RADIUS", 0.20))
    legacy_att = float(_cfg_get(m1, "V563_ATTENTION_RADIUS", 0.24))
    if not (min_r <= max_att <= legacy_att):
        errors.append("V565_MAX_ATTENTION_RADIUS must be within [MIN_MASK_RADIUS,V563_ATTENTION_RADIUS]")
    if float(_cfg_get(m1, "V565_SHAPE_SCALE", 0.75)) <= 0.0:
        errors.append("V565_SHAPE_SCALE must be positive")
    if abs(float(_cfg_get(m1, "V562_MATCH_ACTION_WEIGHT", 0.0))) > 1.0e-12:
        errors.append("V565 retains V564 strict ownership: V562_MATCH_ACTION_WEIGHT must remain 0")
    if errors:
        raise ValueError("V565 root-fix protocol failed:\n  - " + "\n  - ".join(errors))


def _validate_clean_dynamic_component_set_protocol(cfg):
    """Minimal fail-fast contract for the legacy non-versioned CLEAN path."""
    if not _clean_dynamic_component_set(cfg) or _tc_drcs(cfg):
        return
    m1 = _cfg_get(cfg, "M1", None)
    errors = []
    keys = list(m1.keys()) if hasattr(m1, "keys") else list(vars(m1).keys())
    historical = [str(k) for k in keys if __import__("re").match(r"^V[0-9]", str(k).upper())]
    if historical:
        errors.append("historical Vxxx keys are forbidden on CLEAN formal path: " + ", ".join(historical[:20]))
    if str(_cfg_get(m1, "LOSS_MODE", "")).strip().lower() != "clean_dynamic_component_set":
        errors.append("M1.LOSS_MODE must be clean_dynamic_component_set")
    if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
        errors.append("CLEAN is from-scratch and forbids task-trained INIT_CHECKPOINT")
    if int(_cfg_get(m1, "NUM_COMPONENT_SLOTS", 0)) <= 0:
        errors.append("NUM_COMPONENT_SLOTS must be positive")
    budget = float(_cfg_get(m1, "AUX_TO_BASE_MAX_RATIO", 0.25))
    if not (0.0 < budget <= 1.0):
        errors.append("AUX_TO_BASE_MAX_RATIO must be in (0,1]")
    forbidden_geometry = {
        "CENTER_SIGMA_MIN_PX", "CENTER_SIGMA_MAX_PX", "SEED_NMS_RADIUS",
        "SUPPORT_RELATIVE_THRESHOLD", "EXTENT_QUANTILE",
        "ATTENTION_RADIUS_SCALE", "MAX_ATTENTION_RADIUS", "SHAPE_SCALE",
        "SEED_RANK_MARGIN", "SEED_RANK_WEIGHT", "SEED_HEATMAP_WEIGHT",
        "MASK_RADIUS", "ATTENTION_RADIUS",
    }
    present_geometry = sorted(forbidden_geometry.intersection({str(k).upper() for k in keys}))
    if present_geometry:
        errors.append("hand-tuned spatial/seed constants are forbidden: " + ", ".join(present_geometry))
    task_weight_keys = [
        str(k) for k in keys
        if str(k).upper().endswith("_WEIGHT")
        and str(k).upper() not in {"BASE_LEARNING_RATE", "CANDIDATE_LEARNING_RATE", "M2_LEARNING_RATE"}
    ]
    if task_weight_keys:
        errors.append("per-task hand loss weights are forbidden; use learned uncertainty: " + ", ".join(task_weight_keys[:20]))
    if errors:
        raise ValueError("CLEAN dynamic component-set protocol failed:\n  - " + "\n  - ".join(errors))



def _validate_tc_drcs_protocol(cfg):
    """Fail-fast contract for Teacher-Complete Differentiable Residual Component Set."""
    if not _tc_drcs(cfg):
        return
    m1 = _cfg_get(cfg, "M1", None)
    errors = []
    keys = list(m1.keys()) if hasattr(m1, "keys") else list(vars(m1).keys())
    historical = [str(k) for k in keys if __import__("re").match(r"^V[0-9]", str(k).upper())]
    if historical:
        errors.append("historical Vxxx keys are forbidden on TC-DRCS formal path: " + ", ".join(historical[:20]))
    for key in ("PROTOCOL", "CANDIDATE_MODE", "LOSS_MODE"):
        if str(_cfg_get(m1, key, "")).strip().lower() != "tc_drcs":
            errors.append(f"M1.{key} must be tc_drcs")
    if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
        errors.append("TC-DRCS is from-scratch and forbids task-trained INIT_CHECKPOINT")
    if int(_cfg_get(m1, "NUM_COMPONENT_SLOTS", 0)) <= 0:
        errors.append("NUM_COMPONENT_SLOTS must be positive")
    budget = float(_cfg_get(m1, "AUX_TO_BASE_MAX_RATIO", 0.25))
    if not (0.0 < budget <= 1.0):
        errors.append("AUX_TO_BASE_MAX_RATIO must be in (0,1]")
    forbidden_tokens = (
        "RADIUS", "SIGMA", "NMS", "CENTER_WEIGHT", "ACTION_WEIGHT", "MATCH_MARGIN",
        "SHAPE_SCALE", "IDENTITY_MIX", "OUTSIDE_LOGIT", "PILOT_WEIGHT", "MASK_WEIGHT",
    )
    upper_keys = [str(k).upper() for k in keys]
    bad = [k for k in upper_keys if any(token in k for token in forbidden_tokens)]
    if bad:
        errors.append("hard proposal/spatial or hand task-weight knobs are forbidden: " + ", ".join(bad[:20]))
    if errors:
        raise ValueError("TC-DRCS protocol failed:\n  - " + "\n  - ".join(errors))



def _validate_semlt_protocol(cfg):
    """Fail closed unless the run is a genuine physical M1-only experiment."""
    if not _semlt(cfg):
        return
    m1 = _cfg_get(cfg, "M1", None)
    train = _cfg_get(cfg, "TRAIN", None)
    errors = []
    protocol_name = str(_cfg_get(m1, "PROTOCOL", "")).strip().lower()
    autozero = protocol_name == "semlt_autozero"
    if autozero:
        expected = {
            "PROTOCOL": "semlt_autozero",
            "CANDIDATE_MODE": "semlt_autozero_transport",
            "LOSS_MODE": "semlt_autozero",
        }
        for key, value in expected.items():
            if str(_cfg_get(m1, key, "")).strip().lower() != value:
                errors.append(f"M1.{key} must be {value}")
        if not bool(_cfg_get(m1, "SEMLT_LST_V3_MAIN", False)):
            errors.append("SEMLT_LST_V3_MAIN must be true for the calibrated formal run")
        fixed_base_refinement = bool(
            _cfg_get(m1, "SEMLT_FIXED_BASE_REFINEMENT", False)
        )
        base_trajectory_lock = bool(
            _cfg_get(m1, "BASE_TRAJECTORY_LOCK", False)
        )
        has_init_checkpoint = bool(
            str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip()
            or str(getattr(cfg, "init_checkpoint", "") or "").strip()
        )
        if fixed_base_refinement:
            if not has_init_checkpoint:
                errors.append(
                    "SEMLT_FIXED_BASE_REFINEMENT requires a trained Base checkpoint "
                    "via --init-checkpoint or M1.INIT_CHECKPOINT"
                )
            if not bool(_cfg_get(m1, "V469_FREEZE_BASE", False)):
                errors.append(
                    "SEMLT_FIXED_BASE_REFINEMENT requires V469_FREEZE_BASE=true"
                )
            if not bool(_cfg_get(m1, "V515_OFFICIAL_BASE_INIT", False)):
                errors.append(
                    "SEMLT_FIXED_BASE_REFINEMENT requires V515_OFFICIAL_BASE_INIT=true "
                    "so only Base/PVL tensors are imported"
                )
        else:
            if has_init_checkpoint:
                errors.append(
                    "from-scratch SemLT-AutoZero forbids task-trained checkpoints"
                )
            if bool(_cfg_get(m1, "V469_FREEZE_BASE", False)):
                errors.append(
                    "from-scratch SemLT-AutoZero requires V469_FREEZE_BASE=false"
                )

        if base_trajectory_lock:
            if fixed_base_refinement:
                errors.append(
                    "BASE_TRAJECTORY_LOCK and SEMLT_FIXED_BASE_REFINEMENT are mutually exclusive"
                )
            if bool(_cfg_get(m1, "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False)):
                errors.append(
                    "BASE_TRAJECTORY_LOCK reproduces the validated historical C3 Base and "
                    "requires OFFICIAL_MEDCLIPSEG_BASE_CONTRACT=false"
                )
            if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
                errors.append("BASE_TRAJECTORY_LOCK requires MHCS_RNG_ISOLATION=true")
            if not bool(_cfg_get(m1, "SEMLT_STRICT_CAUSAL_PARITY", False)):
                errors.append("BASE_TRAJECTORY_LOCK requires SEMLT_STRICT_CAUSAL_PARITY=true")
            if int(_cfg_get(train, "NUM_WORKERS", -1)) != 2:
                errors.append("BASE_TRAJECTORY_LOCK requires TRAIN.NUM_WORKERS=2")
        if not bool(_cfg_get(m1, "SEMLT_AUTOZERO_DETACH_CONDITIONERS", True)):
            errors.append("SemLT-AutoZero requires detached Base/semantic conditioners")
        boundary_normal_warp = bool(_cfg_get(m1, "SEMLT_BOUNDARY_NORMAL_WARP", False))
        sdf_operator_matched_warp = bool(_cfg_get(m1, "SEMLT_SDF_OPERATOR_MATCHED_WARP", False))
        posterior_stable_operator_warp = bool(_cfg_get(m1, "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP", False))
        uc_fnrt = bool(_cfg_get(m1, "SEMLT_UC_FNRT", False))
        if sum(int(v) for v in (boundary_normal_warp, sdf_operator_matched_warp, posterior_stable_operator_warp, uc_fnrt)) > 1:
            errors.append(
                "SEMLT_BOUNDARY_NORMAL_WARP, SEMLT_SDF_OPERATOR_MATCHED_WARP, "
                "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP and SEMLT_UC_FNRT are mutually exclusive"
            )
        formal_protocol = str(_cfg_get(m1, "GEOTR_M1_FORMAL_PROTOCOL", "")).strip().lower()
        if formal_protocol not in {"paper100", "restore150", "diag20"}:
            errors.append(
                "SemLT-LST requires GEOTR_M1_FORMAL_PROTOCOL=paper100, restore150 or diag20"
            )
        if formal_protocol == "diag20" and not (
            bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False))
            or boundary_normal_warp
            or sdf_operator_matched_warp
            or posterior_stable_operator_warp
            or uc_fnrt
        ):
            errors.append(
                "diag20 is reserved for v3.1 diagnostics or an explicit SemLT geometry-warp route"
            )
        radius = int(_cfg_get(m1, "SEMLT_LOCAL_RADIUS_PX", 0))
        if not 1 <= radius <= 16:
            errors.append("SEMLT_LOCAL_RADIUS_PX must be a pre-declared architecture choice in [1,16]")
        if boundary_normal_warp:
            if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
                errors.append("Boundary-Normal Warp forbids SEMLT_LST_V31_ROOTFIX: no Gate exists")
            if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
                errors.append("Boundary-Normal Warp forbids eligibility Gate root-fix: no Gate exists")
            if bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)):
                errors.append("Boundary-Normal Warp forbids Action-Value policy: signed displacement is the action")
            if bool(_cfg_get(m1, "SEMLT_TRANSITION_BAND_ENABLED", False)):
                errors.append("Boundary-Normal Warp uses a Base-boundary support band, not legacy transition eligibility")
            if bool(_cfg_get(m1, "SEMLT_HARD_DEPLOY", False)):
                errors.append("Boundary-Normal Warp has no hard Gate; SEMLT_HARD_DEPLOY must be false")
            if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != "semlt_boundary_normal_warp":
                errors.append("Boundary-Normal Warp requires M1_LOSS_VERSION=semlt_boundary_normal_warp")
            if int(_cfg_get(m1, "M1_TRAIN_NUM_SAMPLES", 1)) != 1:
                errors.append("Boundary-Normal Warp requires M1_TRAIN_NUM_SAMPLES=1")

        if sdf_operator_matched_warp:
            if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
                errors.append("SDF operator-matched warp forbids SEMLT_LST_V31_ROOTFIX: no Gate exists")
            if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
                errors.append("SDF operator-matched warp forbids eligibility Gate root-fix: no Gate exists")
            if bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)):
                errors.append("SDF operator-matched warp forbids Action-Value policy")
            if bool(_cfg_get(m1, "SEMLT_TRANSITION_BAND_ENABLED", False)):
                errors.append("SDF operator-matched warp uses contour-owned swept support, not transition eligibility")
            if bool(_cfg_get(m1, "SEMLT_HARD_DEPLOY", False)):
                errors.append("SDF operator-matched warp has no hard Gate; SEMLT_HARD_DEPLOY must be false")
            if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != "semlt_sdf_operator_matched_warp":
                errors.append("SDF operator-matched warp requires M1_LOSS_VERSION=semlt_sdf_operator_matched_warp")
            if int(_cfg_get(m1, "M1_TRAIN_NUM_SAMPLES", 1)) != 1:
                errors.append("SDF operator-matched warp requires M1_TRAIN_NUM_SAMPLES=1")
            if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
                errors.append("SDF operator-matched warp requires MHCS_RNG_ISOLATION=true for paired Base/PVL stochastic trajectories")

        if posterior_stable_operator_warp:
            if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
                errors.append("Posterior-stable operator warp forbids SEMLT_LST_V31_ROOTFIX")
            if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
                errors.append("Posterior-stable operator warp forbids eligibility Gate root-fix")
            if bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)):
                errors.append("Posterior-stable operator warp forbids Action-Value policy")
            if bool(_cfg_get(m1, "SEMLT_TRANSITION_BAND_ENABLED", False)):
                errors.append("Posterior-stable operator warp uses fixed smooth narrow-band extension, not transition eligibility")
            if bool(_cfg_get(m1, "SEMLT_HARD_DEPLOY", False)):
                errors.append("Posterior-stable operator warp has no hard Gate")
            if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != "semlt_posterior_stable_operator_warp":
                errors.append("Posterior-stable operator warp requires M1_LOSS_VERSION=semlt_posterior_stable_operator_warp")
            if int(_cfg_get(m1, "M1_TRAIN_NUM_SAMPLES", 1)) != 1:
                errors.append("Posterior-stable operator warp keeps official Base single-forward; M1_TRAIN_NUM_SAMPLES must remain 1")
            if int(_cfg_get(m1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 0)) != 10:
                errors.append("Posterior-stable operator warp locks GEOTR_TRAIN_POSTERIOR_SAMPLES=10 to match ValMC10 mean-then-refine")
            if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
                errors.append("Posterior-stable operator warp requires MHCS_RNG_ISOLATION=true")

        if uc_fnrt:
            run_tag = str(_cfg_get(m1, "RUN_TAG", "") or "").strip().upper()
            r21_calibrated = "OACD_R21_CALIBRATED" in run_tag
            r22_joint_proper = "OACD_R22_JOINT_PROPER" in run_tag
            r3_hcrv = "OACD_R3_HCRV" in run_tag
            r3_ablation = str(
                _cfg_get(m1, "SEMLT_UC_HCRV_ABLATION", "full") or "full"
            ).strip().lower()
            allowed_r3_ablations = {
                "full",
                "r8_search_range",
                "no_censored_supervision",
                "no_relational_volume",
                "no_hierarchical_decision",
            }
            if r3_ablation not in allowed_r3_ablations:
                errors.append(
                    "SEMLT_UC_HCRV_ABLATION must be one categorical value from "
                    + ", ".join(sorted(allowed_r3_ablations))
                )
            uc_ablation = str(
                _cfg_get(m1, "SEMLT_UC_FNRT_ABLATION", "full") or "full"
            ).strip().lower()
            allowed_uc_ablations = {
                "full", "no_posterior_uncertainty", "no_normal_ray_evidence",
                "direct_signed", "segmentation_only",
            }
            if uc_ablation not in allowed_uc_ablations:
                errors.append(
                    "SEMLT_UC_FNRT_ABLATION must be one categorical value from "
                    + ", ".join(sorted(allowed_uc_ablations))
                )
            if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
                errors.append("UC-FNRT forbids SEMLT_LST_V31_ROOTFIX")
            if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
                errors.append("UC-FNRT forbids eligibility Gate root-fix")
            if bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)):
                errors.append("UC-FNRT forbids Action-Value policy")
            if bool(_cfg_get(m1, "SEMLT_TRANSITION_BAND_ENABLED", False)):
                errors.append("UC-FNRT uses fixed cosine normal-ray transport, not transition eligibility")
            if bool(_cfg_get(m1, "SEMLT_HARD_DEPLOY", False)):
                errors.append("UC-FNRT has no hard Gate; SEMLT_HARD_DEPLOY must be false")
            if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != "semlt_uc_fnrt":
                errors.append("UC-FNRT requires M1_LOSS_VERSION=semlt_uc_fnrt")
            if int(_cfg_get(m1, "M1_TRAIN_NUM_SAMPLES", 1)) != 1:
                errors.append("UC-FNRT requires M1_TRAIN_NUM_SAMPLES=1")
            if int(_cfg_get(m1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 0)) != 10:
                errors.append("UC-FNRT locks GEOTR_TRAIN_POSTERIOR_SAMPLES=10")
            if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
                errors.append("UC-FNRT requires MHCS_RNG_ISOLATION=true")
            if not bool(_cfg_get(m1, "SEMLT_GRADIENT_AUDIT", False)):
                errors.append("UC-FNRT requires SEMLT_GRADIENT_AUDIT=true")
            if r21_calibrated:
                if not bool(_cfg_get(m1, "SEMLT_UC_HIERARCHICAL_ORDINAL_LOSS", False)):
                    errors.append("OACD-R2.1-CAL requires hierarchical ordinal loss")
                if not bool(_cfg_get(m1, "SEMLT_UC_CASE_BALANCED_OWNERS", False)):
                    errors.append("OACD-R2.1-CAL requires case-balanced owner reduction")
                if bool(_cfg_get(m1, "SEMLT_UC_SIGN_CLASS_BALANCED_LEGACY", False)):
                    errors.append(
                        "OACD-R2.1-CAL forbids legacy class-equalized sign loss; "
                        "set SEMLT_UC_SIGN_CLASS_BALANCED_LEGACY=false"
                    )
            if r22_joint_proper:
                if not bool(_cfg_get(m1, "SEMLT_UC_HIERARCHICAL_ORDINAL_LOSS", False)):
                    errors.append("OACD-R2.2-JPS requires hierarchical ordinal loss")
                if not bool(_cfg_get(m1, "SEMLT_UC_CASE_BALANCED_OWNERS", False)):
                    errors.append("OACD-R2.2-JPS requires case-balanced owner reduction")
                if bool(_cfg_get(m1, "SEMLT_UC_SIGN_CLASS_BALANCED_LEGACY", False)):
                    errors.append(
                        "OACD-R2.2-JPS forbids legacy class-equalized sign loss"
                    )
                if not bool(
                    _cfg_get(
                        m1,
                        "SEMLT_UC_HIERARCHICAL_JOINT_PROPER_SCORE",
                        False,
                    )
                ):
                    errors.append(
                        "OACD-R2.2-JPS requires the full signed joint proper-score group"
                    )
                if bool(_cfg_get(m1, "SEMLT_UC_ORDERED_CDF_LOSS", False)):
                    errors.append(
                        "OACD-R2.2-JPS keeps the standalone ordered-CDF branch disabled"
                    )
            if r3_hcrv:
                if not bool(_cfg_get(m1, "SEMLT_UC_OPERATOR_ALIGNED_CANDIDATES", False)):
                    errors.append("OACD-R3-HCRV requires operator-aligned candidates")
                if bool(_cfg_get(m1, "SEMLT_UC_MRM_DOMINANT_MODE", False)):
                    errors.append(
                        "OACD-R3-HCRV forbids the old bin-level dominant-mode decoder"
                    )
                if not bool(_cfg_get(m1, "SEMLT_UC_HIERARCHICAL_ORDINAL_LOSS", False)):
                    errors.append(
                        "OACD-R3-HCRV requires hierarchical sign/magnitude supervision"
                    )
                if bool(
                    _cfg_get(m1, "SEMLT_UC_HIERARCHICAL_JOINT_PROPER_SCORE", False)
                ):
                    errors.append(
                        "OACD-R3-HCRV uses the R2.1 hierarchical objective; "
                        "the R2.2 joint group reduced the observed Train gain"
                    )

                # BUSI formal ablations remove one conceptual method component
                # at a time.  The categorical selector prevents accidental
                # multi-switch variants from being presented as a clean
                # ablation.  All data, optimizer, schedule, MC budget, physical
                # operator and Base/PVL gradient-isolation settings stay fixed.
                r3_expected = {
                    "full": {
                        "radius": 16, "relational": True, "ordered": True,
                        "hierarchical": True, "keep_prior": True,
                        "censored": True, "reachable_only": False,
                    },
                    "r8_search_range": {
                        "radius": 8, "relational": True, "ordered": True,
                        "hierarchical": True, "keep_prior": True,
                        "censored": True, "reachable_only": False,
                    },
                    "no_censored_supervision": {
                        "radius": 16, "relational": True, "ordered": True,
                        "hierarchical": True, "keep_prior": True,
                        "censored": False, "reachable_only": True,
                    },
                    "no_relational_volume": {
                        "radius": 16, "relational": False, "ordered": False,
                        "hierarchical": True, "keep_prior": True,
                        "censored": True, "reachable_only": False,
                    },
                    "no_hierarchical_decision": {
                        "radius": 16, "relational": True, "ordered": True,
                        "hierarchical": False, "keep_prior": False,
                        "censored": True, "reachable_only": False,
                    },
                }[r3_ablation]
                r3_observed = {
                    "radius": int(_cfg_get(m1, "SEMLT_LOCAL_RADIUS_PX", 0)),
                    "relational": bool(_cfg_get(m1, "SEMLT_UC_MRM_RELATIONAL_COST", False)),
                    "ordered": bool(_cfg_get(m1, "SEMLT_UC_MRM_ORDERED_AGGREGATION", False)),
                    "hierarchical": bool(_cfg_get(m1, "SEMLT_UC_HIERARCHICAL_CONFIDENCE_DECODER", False)),
                    "keep_prior": bool(_cfg_get(m1, "SEMLT_UC_RADIUS_BALANCED_KEEP_PRIOR", False)),
                    "censored": bool(_cfg_get(m1, "SEMLT_UC_CENSORED_ENDPOINT_ACTION", False)),
                    "reachable_only": bool(_cfg_get(m1, "SEMLT_UC_REACHABLE_MATCH_ONLY", True)),
                }
                mismatches = [
                    f"{key}={r3_observed[key]!r} (expected {value!r})"
                    for key, value in r3_expected.items()
                    if r3_observed[key] != value
                ]
                if mismatches:
                    errors.append(
                        f"OACD-R3-HCRV ablation {r3_ablation!r} contract mismatch: "
                        + "; ".join(mismatches)
                    )

        if bool(_cfg_get(m1, "SEMLT_LST_V31_ROOTFIX", False)):
            if not bool(_cfg_get(m1, "SEMLT_TRANSITION_BAND_ENABLED", False)):
                errors.append("v3.1 requires SEMLT_TRANSITION_BAND_ENABLED=true")
            if not bool(_cfg_get(m1, "SEMLT_HARD_DEPLOY", False)):
                errors.append("v3.1 requires SEMLT_HARD_DEPLOY=true")
            threshold = float(_cfg_get(m1, "SEMLT_DEPLOY_THRESHOLD", -1.0))
            if abs(threshold - 0.5) > 1.0e-12:
                errors.append("v3.1 diagnostic/formal protocol fixes SEMLT_DEPLOY_THRESHOLD=0.5")
            if float(_cfg_get(m1, "SEMLT_UTILITY_MARGIN", -1.0)) != 0.0:
                errors.append("v3.1 root-fix fixes SEMLT_UTILITY_MARGIN=0.0 before any tuning")
            if int(_cfg_get(m1, "SEMLT_GATE_TEACHER_WARMUP_EPOCHS", -1)) != 3:
                errors.append("v3.1 root-fix fixes SEMLT_GATE_TEACHER_WARMUP_EPOCHS=3")
            if bool(_cfg_get(m1, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)):
                if str(_cfg_get(m1, "SEMLT_GATE_LOSS_DOMAIN", "")).strip().lower() != "transition_eligible":
                    errors.append(
                        "eligible-gate root-fix requires SEMLT_GATE_LOSS_DOMAIN=transition_eligible"
                    )
                action_value = bool(_cfg_get(m1, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False))
                expected_loss_version = (
                    "semlt_lst_action_value_policy" if action_value
                    else "semlt_lst_eligible_gate_rootfix"
                )
                if str(_cfg_get(m1, "M1_LOSS_VERSION", "")).strip().lower() != expected_loss_version:
                    errors.append(
                        f"eligible-gate root-fix requires M1_LOSS_VERSION={expected_loss_version}"
                    )
        if formal_protocol in {"paper100", "diag20"}:
            model_cfg = _cfg_get(cfg, "MODEL", None)
            test_cfg = _cfg_get(cfg, "TEST", None)
            if abs(float(_cfg_get(train, "LEARNING_RATE", 0.0)) - 3.0e-4) > 1.0e-12:
                errors.append("paper100 requires TRAIN.LEARNING_RATE=3e-4")
            if int(_cfg_get(train, "BATCH_SIZE", 0)) != 24:
                errors.append("paper100 requires physical TRAIN.BATCH_SIZE=24")
            if int(_cfg_get(train, "GRAD_ACCUMULATION_STEPS", 1)) != 1:
                errors.append("paper100 requires TRAIN.GRAD_ACCUMULATION_STEPS=1")
            if abs(float(_cfg_get(train, "CE_WEIGHT", -1.0)) - 0.5) > 1.0e-12:
                errors.append("paper100 requires TRAIN.CE_WEIGHT=0.5")
            if abs(float(_cfg_get(train, "DICE_WEIGHT", -1.0)) - 0.5) > 1.0e-12:
                errors.append("paper100 requires TRAIN.DICE_WEIGHT=0.5")
            if abs(float(_cfg_get(train, "CLIP_WEIGHT", -1.0)) - 0.1) > 1.0e-12:
                errors.append("paper100 requires TRAIN.CLIP_WEIGHT=0.1")
            if int(_cfg_get(train, "WARMUP_EPOCHS", -1)) != 0:
                errors.append("paper100 uses plain cosine annealing and requires TRAIN.WARMUP_EPOCHS=0")
            if abs(float(_cfg_get(train, "MIN_LR_RATIO", -1.0))) > 1.0e-12:
                errors.append("paper100 cosine annealing requires TRAIN.MIN_LR_RATIO=0")
            if abs(float(_cfg_get(train, "WEIGHT_DECAY", -1.0))) > 1.0e-12:
                errors.append("paper100 requires TRAIN.WEIGHT_DECAY=0")
            if abs(float(_cfg_get(model_cfg, "BETA", -1.0)) - 2.35) > 1.0e-12:
                errors.append("paper100 requires MODEL.BETA=2.35")
            if int(_cfg_get(model_cfg, "ADAPTER_DIM", 0)) != 256:
                errors.append("paper100 requires MODEL.ADAPTER_DIM=256")
            if int(_cfg_get(model_cfg, "NUM_UPSCALE", 0)) != 2:
                errors.append("paper100 requires MODEL.NUM_UPSCALE=2")
            if abs(float(_cfg_get(model_cfg, "TEMPERATURE", -1.0)) - 0.2) > 1.0e-12:
                errors.append("paper100 requires MODEL.TEMPERATURE=0.2")
            if not bool(_cfg_get(test_cfg, "USE_LATEST", False)):
                errors.append("paper100 requires TEST.USE_LATEST=true so Test uses the physical last epoch")
            if int(_cfg_get(test_cfg, "NUM_SAMPLES", 0)) != 30:
                errors.append("paper100 performance inference requires TEST.NUM_SAMPLES=30")
        if str(_cfg_get(m1, "INFERENCE_MODE", "")).strip().lower() != "unified_action_cf_selection":
            errors.append("M1.INFERENCE_MODE must use unified_action_cf_selection")
        if not bool(_cfg_get(m1, "GEOTOPO_REFINEMENT_ENABLED", False)):
            errors.append("SemLT-AutoZero requires GEOTOPO_REFINEMENT_ENABLED=true")
        auto_adapter = _cfg_get(cfg, "AUTO_ADAPTER", None)
        if bool(_cfg_get(auto_adapter, "ENABLED", False)):
            errors.append("AUTO_ADAPTER must be disabled: AutoZero does not tune weights from diagnostics")

        # Loss lambdas remain forbidden. The local radius is an explicit,
        # interpretable hypothesis-class choice selected on Train/Val coverage.
        forbidden_scalar_keys = {
            "BASE_LEARNING_RATE", "PVL_LEARNING_RATE", "GEOTR_M1_LEARNING_RATE", "SEMLT_LEARNING_RATE",
            "M2_LEARNING_RATE", "M3_LEARNING_RATE", "M2_LOSS_WEIGHT", "CANDIDATE_LOSS_WEIGHT",
            "V471_AUX_TO_BASE_LOSS_RATIO_CAP", "MHCS_HIDDEN_DIM", "SEMANTIC_CHANNELS", "MHCS_TEXT_DIM",
            "GEOTOPO_FLOW_INIT_SCALE_PX", "GEOTR_M1_MAX_FLOW_PX", "GEOTR_M1_GATE_LOW",
            "GEOTR_M1_GATE_HIGH", "GEOTR_M1_GATE_GAMMA", "GEOTR_M1_NORMAL_EPS",
            "GEOTR_M1_MAX_RESIDUAL_LOGIT", "GEOTR_M1_CONTEXT_GATE_INIT", "GEOTR_M1_BCE_WEIGHT",
            "GEOTR_M1_DICE_WEIGHT", "GEOTR_M1_BOUNDARY_WEIGHT", "GEOTOPO_SMOOTHNESS_WEIGHT",
            "GEOTR_M1_FOLDING_WEIGHT", "GEOTR_M1_MIN_JACOBIAN", "GEOTR_M1_EDGE_STIFFNESS_POWER",
            "GEOTR_M1_EDGE_STIFFNESS_FLOOR", "GEOTR_M1_UNIFIED_WEIGHT", "GEOTR_M1_UNIFIED_SUPPORT_RATIO",
            "GEOTR_M1_UNIFIED_TANGENT_RATIO", "GEOTR_M1_UNIFIED_SMOOTH_RATIO",
            "GEOTR_M1_UNIFIED_SUPPORT_POWER", "GEOTR_M1_UNIFIED_NORMAL_SMOOTH_KERNEL",
            "GEOTR_M1_FN_TEACHER_WEIGHT", "GEOTR_M1_FN_TEACHER_RADIUS_PX",
            "GEOTR_M1_FN_TEACHER_THRESHOLD", "SEMLT_MAX_FLOW_PX", "SEMLT_BOUNDARY_WEIGHT",
            "SEMLT_MAGNITUDE_WEIGHT", "SEMLT_TV_WEIGHT", "SEMLT_FOLD_WEIGHT",
        }
        present = sorted(key for key in forbidden_scalar_keys if _cfg_get(m1, key, None) is not None)
        if present:
            errors.append("manual M1 scalar tuning keys are forbidden: " + ", ".join(present))

        forbidden_true = (
            "GEOTR_SPARC_HR_ENABLED", "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED",
            "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED",
            "GEOTR_V4D_ROOT_FIX_ENABLED", "GEOTR_C2R_ENABLED", "GEOTR_C2R_CANONICAL_ROI_ENABLED",
            "GEOTR_PC2R_POSTERIOR_V3_ENABLED", "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED",
            "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", "GEOTR_AEFR_ENABLED",
            "GEOTR_SLR_TRUE_HR_ENABLED", "GEOTR_TYPED_RESIDUAL_ENABLED",
        )
        enabled_forbidden = [key for key in forbidden_true if bool(_cfg_get(m1, key, False))]
        if enabled_forbidden:
            errors.append("M2/residual flags must be false: " + ", ".join(enabled_forbidden))
        errors.extend(validate_geotr_m1_checkpoint_protocol(m1, train))
        if errors:
            raise ValueError("SemLT-AutoZero protocol failed:\n  - " + "\n  - ".join(errors))
        return

    exact_m1 = protocol_name == "geotr_m1"
    causal_ablation = bool(
        exact_m1 and _cfg_get(m1, "GEOTR_M1_CAUSAL_ABLATION", False)
    )
    main_e2e100 = bool(
        exact_m1 and _cfg_get(m1, "GEOTR_M1_MAIN_E2E100", False)
    )
    expected = {
        "PROTOCOL": "geotr_m1" if exact_m1 else "semlt",
        "CANDIDATE_MODE": "exact_geometry_transport" if exact_m1 else "semlt_logit_transport",
        "LOSS_MODE": "geotr_m1" if exact_m1 else "semlt",
        "GEOTOPO_MODE": "geometry",
    }
    for key, value in expected.items():
        if str(_cfg_get(m1, key, "")).strip().lower() != value:
            errors.append(f"M1.{key} must be {value}")
    configured_init = str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip()
    cli_init = str(getattr(cfg, "init_checkpoint", "") or "").strip()
    # The three checkpoint modes below are mutually exclusive.  Keep them in
    # one if/elif chain so a declared common-Base causal ablation is not
    # accidentally reclassified by the generic from-scratch checkpoint guard.
    #
    # Regression fixed in SEMLT_LOCKED_ABLATION_RUNTIME_FIX_R1:
    # the old code used ``if causal_ablation: ...`` followed by a *second*
    # ``if main_e2e100: ... elif configured_init or cli_init: ...``.  Therefore
    # every legitimate causal ablation (causal=True, main=False, checkpoint
    # present) still entered the trailing elif and failed before epoch 1.
    if causal_ablation and main_e2e100:
        errors.append(
            "GEOTR_M1_MAIN_E2E100 and GEOTR_M1_CAUSAL_ABLATION are mutually exclusive"
        )

    if causal_ablation:
        if not (configured_init or cli_init):
            errors.append(
                "GEOTR_M1_CAUSAL_ABLATION requires a common Base checkpoint "
                "through --init-checkpoint or M1.INIT_CHECKPOINT"
            )
        if not bool(_cfg_get(m1, "V469_FREEZE_BASE", False)):
            errors.append(
                "GEOTR_M1_CAUSAL_ABLATION requires M1.V469_FREEZE_BASE=true"
            )
        if not bool(_cfg_get(m1, "GEOTR_M1_DETACH_CONDITIONERS", True)):
            errors.append(
                "GEOTR_M1_CAUSAL_ABLATION requires detached conditioning evidence"
            )
    elif main_e2e100:
        if configured_init or cli_init:
            errors.append(
                "GEOTR_M1_MAIN_E2E100 is from-scratch and forbids task-trained checkpoints"
            )
        if bool(_cfg_get(m1, "V469_FREEZE_BASE", False)):
            errors.append(
                "GEOTR_M1_MAIN_E2E100 requires V469_FREEZE_BASE=false"
            )
        if not bool(_cfg_get(m1, "GEOTR_M1_DETACH_CONDITIONERS", True)):
            errors.append(
                "GEOTR_M1_MAIN_E2E100 requires detached Base/conditioner evidence"
            )
        if int(_cfg_get(train, "NUM_EPOCHS", 0)) != 100:
            errors.append("GEOTR_M1_MAIN_E2E100 requires TRAIN.NUM_EPOCHS=100")
        if int(_cfg_get(train, "SCHEDULER_TOTAL_EPOCHS", 0)) != 100:
            errors.append(
                "GEOTR_M1_MAIN_E2E100 requires TRAIN.SCHEDULER_TOTAL_EPOCHS=100"
            )
        if bool(_cfg_get(train, "USE_VALIDATION_SELECTION", False)):
            errors.append(
                "GEOTR_M1_MAIN_E2E100 uses the physical epoch-100 checkpoint and forbids validation selection"
            )
        if str(_cfg_get(m1, "GEOTR_M1_OPERATOR", "")).strip().lower() != "free_2d":
            errors.append("GEOTR_M1_MAIN_E2E100 requires the locked free_2d operator")
        if str(_cfg_get(m1, "GEOTR_M1_TRANSPORT_SPACE", "")).strip().lower() != "logit":
            errors.append("GEOTR_M1_MAIN_E2E100 requires logit transport")
        if str(_cfg_get(m1, "GEOTR_M1_GATE_MODE", "")).strip().lower() != "none":
            errors.append("GEOTR_M1_MAIN_E2E100 requires gate_mode=none")
        if str(_cfg_get(m1, "GEOTR_M1_DEFORM_MODE", "")).strip().lower() != "unified":
            errors.append("GEOTR_M1_MAIN_E2E100 requires unified anisotropic geometry")
        if abs(float(_cfg_get(m1, "GEOTR_M1_UNIFIED_SUPPORT_RATIO", -1.0)) - 1.0) > 1e-12:
            errors.append("GEOTR_M1_MAIN_E2E100 requires unified support ratio=1")
        if abs(float(_cfg_get(m1, "GEOTR_M1_UNIFIED_TANGENT_RATIO", -1.0)) - 1.0) > 1e-12:
            errors.append("GEOTR_M1_MAIN_E2E100 requires unified tangent ratio=1")
        if abs(float(_cfg_get(m1, "GEOTR_M1_UNIFIED_SMOOTH_RATIO", -1.0))) > 1e-12:
            errors.append("GEOTR_M1_MAIN_E2E100 requires unified smooth ratio=0")
    elif configured_init or cli_init:
        errors.append(
            "from-scratch M1-only training forbids task checkpoints; declare "
            "GEOTR_M1_CAUSAL_ABLATION=true only for common-Base ablations"
        )
    if str(_cfg_get(m1, "INFERENCE_MODE", "")).strip().lower() != "unified_action_cf_selection":
        errors.append("M1.INFERENCE_MODE must use the canonical unified_action_cf_selection dispatcher")
    if not bool(_cfg_get(m1, "GEOTOPO_REFINEMENT_ENABLED", False)):
        errors.append("M1-only Transport requires GEOTOPO_REFINEMENT_ENABLED=true")
    if float(_cfg_get(m1, "M2_LEARNING_RATE", 0.0)) != 0.0:
        errors.append("M2_LEARNING_RATE must be zero")
    if float(_cfg_get(m1, "M3_LEARNING_RATE", 0.0)) != 0.0:
        errors.append("M3_LEARNING_RATE must be zero")
    forbidden_true = (
        "GEOTR_SPARC_HR_ENABLED", "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED",
        "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED",
        "GEOTR_V4D_ROOT_FIX_ENABLED", "GEOTR_C2R_ENABLED",
        "GEOTR_C2R_CANONICAL_ROI_ENABLED", "GEOTR_PC2R_POSTERIOR_V3_ENABLED",
        "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED", "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED",
        "GEOTR_AEFR_ENABLED", "GEOTR_SLR_TRUE_HR_ENABLED", "GEOTR_TYPED_RESIDUAL_ENABLED",
    )
    enabled_forbidden = [key for key in forbidden_true if bool(_cfg_get(m1, key, False))]
    if enabled_forbidden:
        errors.append("M2/residual flags must be false: " + ", ".join(enabled_forbidden))
    if exact_m1:
        if float(_cfg_get(m1, "GEOTOPO_FLOW_INIT_SCALE_PX", 0.0)) <= 0.0:
            errors.append("GEOTOPO_FLOW_INIT_SCALE_PX must be positive")
        if float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", -1.0)) < 0.0:
            errors.append("GEOTOPO_SMOOTHNESS_WEIGHT must be non-negative")
        if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
            errors.append("MHCS_RNG_ISOLATION must be true for the controlled M1 trajectory")
    else:
        if float(_cfg_get(m1, "SEMLT_MAX_FLOW_PX", 0.0)) <= 0.0:
            errors.append("SEMLT_MAX_FLOW_PX must be positive")
        for key in ("SEMLT_BOUNDARY_WEIGHT", "SEMLT_MAGNITUDE_WEIGHT", "SEMLT_TV_WEIGHT", "SEMLT_FOLD_WEIGHT"):
            if float(_cfg_get(m1, key, -1.0)) < 0.0:
                errors.append(f"{key} must be non-negative")
    errors.extend(validate_geotr_m1_checkpoint_protocol(m1, train))
    if bool(_cfg_get(train, "VAL_REQUIRE_M2_NONDEGRADATION_OVER_BASE", False)):
        errors.append("M2-over-Base validation gate must be false")
    if bool(_cfg_get(train, "VAL_REQUIRE_M2_NONDEGRADATION_OVER_M1", False)):
        errors.append("M2-over-M1 validation gate must be false")
    if bool(_cfg_get(train, "VAL_REQUIRE_M3_NONDEGRADATION_OVER_M2", False)):
        errors.append("M3 validation gate must be false")
    if errors:
        raise ValueError("Physical M1-only protocol failed:\n  - " + "\n  - ".join(errors))


def _validate_mhcs_protocol(cfg):
    """Fail-fast R5.2 Per-Candidate Counterfactual Safety Router contract.

    R5.2 deliberately keeps the R5.1 H0..H5 candidate bank and replaces only M2.
    Every concrete local H1..H5 action is classified Harm/Neutral/Benefit from its
    counterfactual DSC gain. Deployment uses the same action unit and is fail-closed.
    GT/predicted eligibility masking, binary EditGate+Selector routing, ST, signed-risk
    regression and inverse-gradient balancing are forbidden.
    """
    if not _mhcs(cfg):
        return
    m1 = _cfg_get(cfg, "M1", None)
    errors = []

    # Geometry--Residual A0--A3 protocol.  This intentionally reuses the MHCS
    # integration point but removes the historical H1..H5/oracle/safety contract.
    if bool(_cfg_get(m1, "GEOTOPO_REFINEMENT_ENABLED", False)):
        if str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() != "mhcs":
            errors.append("Geometry--Residual experiments require M1.PROTOCOL=mhcs for integration compatibility")
        for key in ("CANDIDATE_MODE", "LOSS_MODE"):
            if str(_cfg_get(m1, key, "")).strip().lower() not in {"mhcs", "multi_hypothesis_composition"}:
                errors.append(f"M1.{key} must remain multi_hypothesis_composition for the existing dispatcher")
        mode = str(_cfg_get(m1, "GEOTOPO_MODE", "")).strip().lower()
        if mode not in {"base", "geometry", "residual", "full"}:
            errors.append("GEOTOPO_MODE must be base, geometry, residual or full")
        if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
            errors.append("A0--A3 are from-scratch and forbid task-trained INIT_CHECKPOINT")
        if float(_cfg_get(m1, "M2_LEARNING_RATE", 0.0)) != 0.0:
            errors.append("Historical M2_LEARNING_RATE must be 0")
        if float(_cfg_get(m1, "M3_LEARNING_RATE", 0.0)) != 0.0:
            errors.append("Historical M3_LEARNING_RATE must be 0")
        if float(_cfg_get(m1, "MHCS_BANK_LEARNING_RATE", 0.0)) <= 0.0:
            errors.append("MHCS_BANK_LEARNING_RATE must be positive")
        if float(_cfg_get(m1, "MHCS_M2_LEARNING_RATE", 0.0)) <= 0.0:
            errors.append("MHCS_M2_LEARNING_RATE must be positive for the residual-head optimizer owner")
        if int(_cfg_get(m1, "MHCS_HIDDEN_DIM", 0)) < 32:
            errors.append("MHCS_HIDDEN_DIM must be >=32")
        if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
            errors.append("A0--A3 require MHCS_RNG_ISOLATION=true for fair shared-Base trajectories")
        smooth = float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", 0.0))
        if smooth < 0.0:
            errors.append("GEOTOPO_SMOOTHNESS_WEIGHT must be non-negative")
        if not bool(_cfg_get(m1, "GEOTR_SEQUENTIAL_RECONSTRUCTION", False)):
            errors.append("GEOTR v3 requires GEOTR_SEQUENTIAL_RECONSTRUCTION=true")
        if not bool(_cfg_get(m1, "GEOTR_BOUNDED_RECONSTRUCTION", False)):
            errors.append("GEOTR v3 requires GEOTR_BOUNDED_RECONSTRUCTION=true")
        bank_lr = float(_cfg_get(m1, "MHCS_BANK_LEARNING_RATE", 0.0))
        recon_lr = float(_cfg_get(m1, "MHCS_M2_LEARNING_RATE", 0.0))
        if bool(_cfg_get(m1, "GEOTR_SPARC_HR_ENABLED", False)):
            if recon_lr <= 0.0 or recon_lr > bank_lr:
                errors.append(
                    "SPARC-HR3.1 requires 0 < MHCS_M2_LEARNING_RATE <= "
                    "MHCS_BANK_LEARNING_RATE"
                )
        elif abs(bank_lr - recon_lr) > 1.0e-12:
            errors.append("GEOTR causal A0-A3 requires equal Transport/Recon learning rates: MHCS_BANK_LEARNING_RATE == MHCS_M2_LEARNING_RATE")
        train_cfg = _cfg_get(cfg, "TRAIN", None)
        if not bool(_cfg_get(train_cfg, "USE_VALIDATION_SELECTION", False)):
            errors.append("GEOTR formal protocol requires TRAIN.USE_VALIDATION_SELECTION=true")
        sparc_hr = bool(_cfg_get(m1, "GEOTR_SPARC_HR_ENABLED", False))
        expected_selection = "native_sparc_hr_dice" if sparc_hr else "native_m2_dice"
        expected_tiebreak = "native_sparc_hr_nsd" if sparc_hr else "native_m2_nsd"
        if str(_cfg_get(train_cfg, "VAL_SELECTION_METRIC", "")).strip() != expected_selection:
            errors.append(
                f"GEOTR checkpoint selection must use TRAIN.VAL_SELECTION_METRIC={expected_selection}"
            )
        if str(_cfg_get(train_cfg, "VAL_TIEBREAK_METRIC", "")).strip() != expected_tiebreak:
            errors.append(
                f"GEOTR checkpoint tie-break must use TRAIN.VAL_TIEBREAK_METRIC={expected_tiebreak}"
            )
        if int(_cfg_get(train_cfg, "VAL_INTERVAL", 1)) != 1:
            errors.append("GEOTR protocol requires TRAIN.VAL_INTERVAL=1")
        if sparc_hr:
            if not bool(_cfg_get(m1, "GEOTR_SPARC_DENSE_ROUTER_ENABLED", False)):
                errors.append("SPARC-HR3 requires GEOTR_SPARC_DENSE_ROUTER_ENABLED=true")
            if int(_cfg_get(m1, "GEOTR_SPARC_MAX_STEPS", 1)) != 1:
                errors.append("SPARC-HR3 is one-shot and requires GEOTR_SPARC_MAX_STEPS=1")
            if int(_cfg_get(m1, "GEOTR_SPARC_ROUTER_GRID_SIZE", 112)) < 56:
                errors.append("SPARC-HR3 ROUTER_GRID_SIZE must be >=56")
            if float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_TEMPERATURE", 0.5)) <= 0.0:
                errors.append("SPARC-HR3 ROUTER_TEMPERATURE must be positive")
            if float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_ANCHOR_PRIOR", 1.0)) <= 0.0:
                errors.append("SPARC-HR3 ROUTER_ANCHOR_PRIOR must be positive")
            if float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_SOFT_ADVANTAGE_WEIGHT", 0.0)) != 0.0:
                errors.append("SPARC-HR3.1 requires ROUTER_SOFT_ADVANTAGE_WEIGHT=0")
            if float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_MIN_ADVANTAGE", 0.0)) <= 0.0:
                errors.append("SPARC-HR3.1 ROUTER_MIN_ADVANTAGE must be positive")
            if int(_cfg_get(m1, "GEOTR_SPARC_BOUNDARY_BAND_RADIUS_HR", 12)) < 4:
                errors.append("SPARC-HR3.1 BOUNDARY_BAND_RADIUS_HR must be >=4")
            if float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_EDIT_LOGIT_MARGIN", 1.5)) <= 0.0:
                errors.append("SPARC-HR3.1 EDIT_LOGIT_MARGIN must be positive")
            edit_probability = float(
                _cfg_get(m1, "GEOTR_SPARC_ROUTER_EDIT_PROBABILITY_THRESHOLD", 0.70)
            )
            if edit_probability <= 0.5 or edit_probability >= 1.0:
                errors.append("SPARC-HR3.1 EDIT_PROBABILITY_THRESHOLD must be in (0.5,1)")
            actionable_weight = float(
                _cfg_get(m1, "GEOTR_SPARC_ROUTER_ACTIONABLE_WEIGHT", 0.40)
            )
            if actionable_weight <= 0.0 or actionable_weight >= 0.5:
                errors.append("SPARC-HR3.1 ROUTER_ACTIONABLE_WEIGHT must be in (0,0.5)")
            if not bool(_cfg_get(m1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False)):
                errors.append("SPARC-HR requires GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED=true")
            if bool(_cfg_get(m1, "GEOTR_C2R_ENABLED", False)) or bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False)):
                errors.append("SPARC-HR is an alternative M2; C2R and AEFR must be false")
            if int(_cfg_get(m1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 0)) < 3:
                errors.append("SPARC-HR requires GEOTR_TRAIN_POSTERIOR_SAMPLES>=3")
            if int(_cfg_get(m1, "VAL_NUM_SAMPLES", 0)) < 3:
                errors.append("SPARC-HR requires VAL_NUM_SAMPLES>=3")
            if int(_cfg_get(m1, "GEOTR_SLR_HR_SIZE", 448)) != 2 * int(_cfg_get(_cfg_get(cfg, "DATASET", None), "SIZE", 224)):
                errors.append("SPARC-HR requires GEOTR_SLR_HR_SIZE == 2 * DATASET.SIZE")
            if int(_cfg_get(m1, "GEOTR_SPARC_NUM_REGIONS", 4)) < 1:
                errors.append("GEOTR_SPARC_NUM_REGIONS must be positive")
            region_size = int(_cfg_get(m1, "GEOTR_SPARC_REGION_SIZE_LR", 49))
            if region_size < 3 or region_size % 2 == 0:
                errors.append("GEOTR_SPARC_REGION_SIZE_LR must be odd and >=3")
            if int(_cfg_get(m1, "GEOTR_SPARC_MIN_CENTER_DISTANCE_LR", region_size)) < region_size:
                errors.append("SPARC-HR2 requires MIN_CENTER_DISTANCE_LR >= REGION_SIZE_LR")
            steps = int(_cfg_get(m1, "GEOTR_SPARC_MAX_STEPS", 3))
            if steps < 1 or steps > int(_cfg_get(m1, "GEOTR_SPARC_NUM_REGIONS", 4)):
                errors.append("GEOTR_SPARC_MAX_STEPS must be in [1, NUM_REGIONS]")
            overlap = float(_cfg_get(m1, "GEOTR_SPARC_MAX_REGION_OVERLAP", 0.10))
            if overlap < 0.0 or overlap > 0.25:
                errors.append("SPARC-HR2 MAX_REGION_OVERLAP must be in [0,0.25]")
            crossing = float(_cfg_get(m1, "GEOTR_SPARC_MINIMUM_CROSSING_MARGIN", 0.05))
            if crossing <= 0.0 or crossing >= 0.25:
                errors.append("SPARC-HR2 MINIMUM_CROSSING_MARGIN must be in (0,0.25)")
            if int(_cfg_get(m1, "GEOTR_SPARC_CRITIC_GRID_SIZE", 112)) < 56:
                errors.append("SPARC-HR2 CRITIC_GRID_SIZE must be >=56")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_TEMPERATURE", 1.0)) <= 0.0:
                errors.append("SPARC-HR2 POLICY_TEMPERATURE must be positive")
            neutral_margin = float(_cfg_get(m1, "GEOTR_SPARC_NEUTRAL_UTILITY_MARGIN", 2.0e-4))
            if neutral_margin < 0.0:
                errors.append("SPARC-HR2 NEUTRAL_UTILITY_MARGIN must be non-negative")
            if abs(float(_cfg_get(m1, "GEOTR_SPARC_STOP_MARGIN", neutral_margin)) - neutral_margin) > 1.0e-12:
                errors.append("SPARC-HR2.6 STOP_MARGIN must equal NEUTRAL_UTILITY_MARGIN")
            if abs(float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_CLASS_SCORE_SCALE", 0.0))) > 1.0e-12:
                errors.append("SPARC-HR2.6 forbids mixing class probability into numeric utility")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_TEACHER_TEMPERATURE", 0.002)) <= 0.0:
                errors.append("SPARC-HR2.6 POLICY_TEACHER_TEMPERATURE must be positive")
            policy_margin = float(_cfg_get(m1, "GEOTR_SPARC_POLICY_MARGIN", 0.0))
            if policy_margin < 0.0:
                errors.append("SPARC-HR2.6 POLICY_MARGIN must be non-negative")
            positive_mix = float(
                _cfg_get(m1, "GEOTR_SPARC_POLICY_POSITIVE_UNIFORM_MIX", 0.50)
            )
            if positive_mix < 0.0 or positive_mix > 1.0:
                errors.append("SPARC-HR2.6 POLICY_POSITIVE_UNIFORM_MIX must be in [0,1]")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_BENEFIT_AUX_WEIGHT", 0.25)) < 0.0:
                errors.append("SPARC-HR2.6 POLICY_BENEFIT_AUX_WEIGHT must be non-negative")
            if int(_cfg_get(m1, "GEOTR_SPARC_CRITIC_PATCH_SIZE", 12)) < 6:
                errors.append("SPARC-HR2.6 CRITIC_PATCH_SIZE must be >=6")
            if int(_cfg_get(m1, "GEOTR_SPARC_SOURCE_EMBED_DIM", 12)) < 4:
                errors.append("SPARC-HR2.6 SOURCE_EMBED_DIM must be >=4")
            if int(_cfg_get(m1, "GEOTR_SPARC_POLICY_SET_LAYERS", 2)) < 1:
                errors.append("SPARC-HR2.6 POLICY_SET_LAYERS must be >=1")
            if int(_cfg_get(m1, "GEOTR_SPARC_POLICY_SET_HEADS", 4)) < 1:
                errors.append("SPARC-HR2.6 POLICY_SET_HEADS must be >=1")
            if int(_cfg_get(m1, "GEOTR_SPARC_MIN_ACTION_CHANGE_HR_PIXELS", 4)) < 1:
                errors.append("SPARC-HR2.6 MIN_ACTION_CHANGE_HR_PIXELS must be >=1")
            if not bool(_cfg_get(m1, "GEOTR_SPARC_PRUNE_DUPLICATE_ACTIONS", True)):
                errors.append("SPARC-HR2.6 requires duplicate-action pruning")
            for _key in (
                "GEOTR_SPARC_POLICY_SOFT_TARGET_WEIGHT",
                "GEOTR_SPARC_POLICY_HARD_TARGET_WEIGHT",
                "GEOTR_SPARC_POLICY_HARD_NEGATIVE_WEIGHT",
                "GEOTR_SPARC_POLICY_HARM_BELOW_STOP_WEIGHT",
            ):
                if float(_cfg_get(m1, _key, 0.0)) <= 0.0:
                    errors.append(f"SPARC-HR2.6 {_key} must be positive")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_HARD_NEGATIVE_MARGIN", 0.50)) < 0.0:
                errors.append("SPARC-HR2.6 POLICY_HARD_NEGATIVE_MARGIN must be non-negative")
            for _key in (
                "GEOTR_SPARC_POLICY_STRUCTURED_REGRET_WEIGHT",
                "GEOTR_SPARC_POLICY_EXPECTED_REGRET_WEIGHT",
                "GEOTR_SPARC_POLICY_REGRET_SCALE",
            ):
                if float(_cfg_get(m1, _key, 0.0)) <= 0.0:
                    errors.append(f"SPARC-HR2.6 {_key} must be positive")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_REGRET_CAP", 5.0)) < 1.0:
                errors.append("SPARC-HR2.6 POLICY_REGRET_CAP must be >=1")
            learner_weight = float(
                _cfg_get(m1, "GEOTR_SPARC_LEARNER_POLICY_LOSS_WEIGHT", 0.50)
            )
            if learner_weight < 0.0 or learner_weight > 1.0:
                errors.append("SPARC-HR2.6 LEARNER_POLICY_LOSS_WEIGHT must be in [0,1]")
            quantile_tau = float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_QUANTILE_TAU", 0.10))
            if quantile_tau <= 0.0 or quantile_tau >= 0.5:
                errors.append("SPARC-HR2.6 UTILITY_QUANTILE_TAU must be in (0,0.5)")
            benefit_threshold = float(
                _cfg_get(m1, "GEOTR_SPARC_BENEFIT_PROB_THRESHOLD", 0.50)
            )
            if benefit_threshold <= 0.0 or benefit_threshold >= 1.0:
                errors.append("SPARC-HR2.6 compatibility BENEFIT_PROB_THRESHOLD must be in (0,1)")
            if float(_cfg_get(m1, "GEOTR_SPARC_POLICY_RANK_LOSS_WEIGHT", 0.25)) < 0.0:
                errors.append("SPARC-HR2.6 POLICY_RANK_LOSS_WEIGHT must be non-negative")
            if float(_cfg_get(m1, "GEOTR_SPARC_CLASS_BALANCE_MAX", 8.0)) < 1.0:
                errors.append("SPARC-HR2.6 CLASS_BALANCE_MAX must be >=1")
            if float(_cfg_get(m1, "GEOTR_SPARC_QUANTILE_LOSS_WEIGHT", 0.0)) < 0.0:
                errors.append("SPARC-HR2.6 QUANTILE_LOSS_WEIGHT must be non-negative")
            aux_cap = float(_cfg_get(m1, "GEOTR_SPARC_AUX_TO_BASE_LOSS_RATIO_CAP", 0.75))
            if aux_cap <= 0.0 or aux_cap > 1.0:
                errors.append("SPARC-HR2 AUX_TO_BASE_LOSS_RATIO_CAP must be in (0,1]")
            hr_tol = int(_cfg_get(m1, "GEOTR_SPARC_SURFACE_TOLERANCE_HR_PX", 4))
            lr_tol = int(_cfg_get(m1, "M2_NSD_TOLERANCE_PIXELS", 2))
            if hr_tol != 2 * lr_tol:
                errors.append("SPARC-HR2 HR surface tolerance must equal 2 * M2_NSD_TOLERANCE_PIXELS")
        c2r = bool(_cfg_get(m1, "GEOTR_C2R_ENABLED", False))
        if c2r:
            if not bool(_cfg_get(m1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False)):
                errors.append("C2R requires GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED=true for M2 ownership")
            if bool(_cfg_get(m1, "GEOTR_V4G_R3_MINIMAL_INTERVENTION_ENABLED", False)):
                errors.append("C2R forbids the R3 point-delta actuator")
            if bool(_cfg_get(m1, "GEOTR_V4G_R4_EXOGENOUS_PATCH_FLIP_ENABLED", False)):
                errors.append("C2R forbids R4/R4.1 synthetic FLIP/KEEP action supervision")
            if bool(_cfg_get(m1, "GEOTR_V4G_R41_SELECTION_CONSISTENT_ENABLED", False)):
                errors.append("C2R forbids the R4.1 point-action path")
            rs = int(_cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33))
            if rs < 3 or rs % 2 == 0:
                errors.append("GEOTR_C2R_REGION_SIZE must be odd and >=3")
            if int(_cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 4)) <= 0:
                errors.append("GEOTR_C2R_NUM_REGIONS must be positive")
            if int(_cfg_get(m1, "GEOTR_C2R_COUNTERFACTUAL_RADIUS", 1)) <= 0:
                errors.append("GEOTR_C2R_COUNTERFACTUAL_RADIUS must be positive")
            if str(_cfg_get(m1, "GEOTR_C2R_SELECTION_SCORE", "margin")).strip().lower() not in {"margin", "mc_std", "mc_disagreement", "entropy", "hybrid_max"}:
                errors.append("GEOTR_C2R_SELECTION_SCORE is invalid")
            if float(_cfg_get(m1, "GEOTR_C2R_VIEW_LOSS_WEIGHT", 1.0)) <= 0.0:
                errors.append("GEOTR_C2R_VIEW_LOSS_WEIGHT must be positive")
            if float(_cfg_get(m1, "GEOTR_C2R_FINAL_SEG_WEIGHT", 0.25)) < 0.0:
                errors.append("GEOTR_C2R_FINAL_SEG_WEIGHT must be non-negative")
            c2r_v2 = bool(_cfg_get(m1, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False))
            if c2r_v2:
                if int(_cfg_get(m1, "GEOTR_C2R_MIN_CENTER_DISTANCE", rs)) < rs:
                    errors.append("C2R-v2 requires GEOTR_C2R_MIN_CENTER_DISTANCE >= GEOTR_C2R_REGION_SIZE")
                if int(_cfg_get(m1, "GEOTR_C2R_SEED_RADIUS", 2)) < 0:
                    errors.append("GEOTR_C2R_SEED_RADIUS must be non-negative")
                agree = float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_AGREEMENT_THRESHOLD", 0.90))
                spread = float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_SPREAD_THRESHOLD", 0.12))
                conf = float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_CONFIDENCE_THRESHOLD", 0.10))
                if not (0.0 <= agree <= 1.0):
                    errors.append("GEOTR_C2R_COMPONENT_AGREEMENT_THRESHOLD must be in [0,1]")
                if spread < 0.0:
                    errors.append("GEOTR_C2R_COMPONENT_SPREAD_THRESHOLD must be >=0")
                if not (0.0 <= conf <= 0.5):
                    errors.append("GEOTR_C2R_COMPONENT_CONFIDENCE_THRESHOLD must be in [0,0.5]")
                if int(_cfg_get(m1, "GEOTR_C2R_COMPONENT_MIN_AREA", 2)) <= 0:
                    errors.append("GEOTR_C2R_COMPONENT_MIN_AREA must be positive")
                if float(_cfg_get(m1, "GEOTR_C2R_CANONICAL_RESIDUAL_SCALE", 1.0)) <= 0.0:
                    errors.append("GEOTR_C2R_CANONICAL_RESIDUAL_SCALE must be >0")
                if float(_cfg_get(m1, "GEOTR_C2R_CANONICAL_SEG_WEIGHT", 0.5)) <= 0.0:
                    errors.append("GEOTR_C2R_CANONICAL_SEG_WEIGHT must be positive")
                pc2r_v3 = bool(_cfg_get(m1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False))
                pc2r_v31 = bool(_cfg_get(m1, "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED", False))
                pc2r_v32 = bool(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False))
                pc2r_v32_audit = bool(_cfg_get(m1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False))
                aefr = bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False))
                if aefr:
                    stage = str(_cfg_get(m1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower()
                    if stage not in {"single_atomic", "single_continuous", "hybrid_geometry", "typed_state_taylor", "typed_state_exact", "soft_ownership_exact", "intervention_factorized", "selective_minimal_intervention", "sparse_local_rerendering"}:
                        errors.append("GEOTR_AEFR_STAGE is invalid")
                    if pc2r_v3 or pc2r_v31 or pc2r_v32:
                        errors.append("AEFR is an alternative Stage-2 path; disable PC2R-v3/v3.1/v3.2 flags")
                    if int(_cfg_get(m1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 10)) < 3:
                        errors.append("AEFR posterior diagnostics require GEOTR_TRAIN_POSTERIOR_SAMPLES>=3")
                    if bool(_cfg_get(m1, "GEOTR_AEFR_POSTERIOR_STABILITY_ENABLED", True)) and int(_cfg_get(m1, "VAL_NUM_SAMPLES", 10)) < 3:
                        errors.append("AEFR posterior stability requires VAL_NUM_SAMPLES>=3")
                    if int(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_RADIUS_PX", 5)) < 1:
                        errors.append("GEOTR_AEFR_BOUNDARY_RADIUS_PX must be >=1")
                    if float(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX", 4.0)) <= 0.0:
                        errors.append("GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX must be >0")
                    if float(_cfg_get(m1, "GEOTR_AEFR_INTERIOR_CROSSING_MARGIN", 0.5)) <= 0.0:
                        errors.append("GEOTR_AEFR_INTERIOR_CROSSING_MARGIN must be >0")
                    if float(_cfg_get(m1, "GEOTR_AEFR_TRANSITION_LOGIT_SCALE", 2.0)) <= 0.0:
                        errors.append("GEOTR_AEFR_TRANSITION_LOGIT_SCALE must be >0")
                    if stage in {"intervention_factorized", "selective_minimal_intervention"}:
                        if not bool(_cfg_get(m1, "GEOTR_AEFR_TRANSITION_AWARE_ENABLED", False)):
                            errors.append(f"{stage} requires GEOTR_AEFR_TRANSITION_AWARE_ENABLED=true")
                        ep = float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_ERROR_PRIOR", 0.03))
                        ip = float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_EDIT_PRIOR", 0.03))
                        if not (0.0 < ep < 0.5):
                            errors.append("GEOTR_AEFR_INTERVENTION_ERROR_PRIOR must be in (0,0.5)")
                        if not (0.0 < ip < 0.5):
                            errors.append("GEOTR_AEFR_INTERVENTION_EDIT_PRIOR must be in (0,0.5)")
                        if stage == "intervention_factorized" and float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_ERROR_POS_WEIGHT", 6.0)) <= 0.0:
                            errors.append("GEOTR_AEFR_INTERVENTION_ERROR_POS_WEIGHT must be >0")
                        if stage == "selective_minimal_intervention":
                            tau = float(_cfg_get(m1, "GEOTR_AEFR_SMI_COMMIT_THRESHOLD", 0.50))
                            if not (0.0 < tau < 1.0):
                                errors.append("GEOTR_AEFR_SMI_COMMIT_THRESHOLD must be in (0,1)")
                            dtau = float(_cfg_get(m1, "GEOTR_AEFR_SMI_DIRECTION_CONFIDENCE_THRESHOLD", 0.50))
                            if not (0.0 <= dtau < 1.0):
                                errors.append("GEOTR_AEFR_SMI_DIRECTION_CONFIDENCE_THRESHOLD must be in [0,1)")
                            for key in ("GEOTR_AEFR_SMI_LOCALIZER_WEIGHT", "GEOTR_AEFR_SMI_DIRECTION_WEIGHT", "GEOTR_AEFR_SMI_BOUNDARY_MAG_WEIGHT", "GEOTR_AEFR_SMI_DEPLOY_WEIGHT"):
                                if float(_cfg_get(m1, key, 0.0)) < 0.0:
                                    errors.append(f"{key} must be non-negative")
                    if stage == "sparse_local_rerendering":
                        rs_slr = int(_cfg_get(m1, "GEOTR_SLR_REGION_SIZE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)))
                        if rs_slr < 3 or rs_slr % 2 == 0:
                            errors.append("GEOTR_SLR_REGION_SIZE must be odd and >=3")
                        if int(_cfg_get(m1, "GEOTR_SLR_NUM_REGIONS", 8)) < 1:
                            errors.append("GEOTR_SLR_NUM_REGIONS must be >=1")
                        d_slr = int(_cfg_get(m1, "GEOTR_SLR_MIN_CENTER_DISTANCE", max(1, rs_slr // 2)))
                        if d_slr < 1:
                            errors.append("GEOTR_SLR_MIN_CENTER_DISTANCE must be >=1")
                        taper_slr = int(_cfg_get(m1, "GEOTR_SLR_BLEND_TAPER_PX", 4))
                        if taper_slr < 0 or taper_slr > rs_slr // 2:
                            errors.append("GEOTR_SLR_BLEND_TAPER_PX must be in [0, REGION_SIZE//2]")
                        if int(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8)) < 1:
                            errors.append("GEOTR_SLR_SDF_RADIUS_PX must be >=1")
                        if int(_cfg_get(m1, "GEOTR_SLR_TRAIN_POSITIVE_REGIONS", 4)) < 1:
                            errors.append("GEOTR_SLR_TRAIN_POSITIVE_REGIONS must be >=1")
                        if int(_cfg_get(m1, "GEOTR_SLR_TRAIN_CLEAN_REGIONS", 4)) < 1:
                            errors.append("GEOTR_SLR_TRAIN_CLEAN_REGIONS must be >=1")
                        clean_err = float(_cfg_get(m1, "GEOTR_SLR_CLEAN_MAX_ERROR_FRACTION", 0.02))
                        if not (0.0 <= clean_err < 1.0):
                            errors.append("GEOTR_SLR_CLEAN_MAX_ERROR_FRACTION must be in [0,1)")
                        ucdrt = bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_ENABLED", False))
                        if ucdrt:
                            _ucdrt_br = int(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5))
                            if _ucdrt_br < 1:
                                errors.append("GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX must be >=1")
                            _ucdrt_sdf_r = int(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8))
                            if _ucdrt_sdf_r <= _ucdrt_br:
                                errors.append("GEOTR-SLR-UCDRT requires GEOTR_SLR_SDF_RADIUS_PX > GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX so ADD/REMOVE have nonempty interior support")
                            if float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX", 4.0)) <= 0.0:
                                errors.append("GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX must be >0")
                            if float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP", 4.0)) <= 0.0:
                                errors.append("GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP must be >0")
                            if float(_cfg_get(m1, "GEOTR_SLR_UCDRT_UTILITY_TEMPERATURE", 0.25)) <= 0.0:
                                errors.append("GEOTR_SLR_UCDRT_UTILITY_TEMPERATURE must be >0")
                            if int(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_REGIONS", 4)) < 1:
                                errors.append("GEOTR_SLR_UCDRT_PAIRED_REGIONS must be >=1")
                            margin = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_TARGET_MARGIN", 0.05))
                            if not (0.0 < margin < 0.5):
                                errors.append("GEOTR_SLR_UCDRT_TARGET_MARGIN must be in (0,0.5)")
                            ucdrt_weight_keys = (
                                "GEOTR_SLR_UCDRT_SELECTOR_WEIGHT",
                                "GEOTR_SLR_UCDRT_STATE_WEIGHT",
                                "GEOTR_SLR_UCDRT_UTILITY_WEIGHT",
                                "GEOTR_SLR_UCDRT_ACTION_WEIGHT",
                                "GEOTR_SLR_UCDRT_FINAL_WEIGHT",
                                "GEOTR_SLR_UCDRT_PAIRED_WEIGHT",
                            )
                            for key in ucdrt_weight_keys:
                                if float(_cfg_get(m1, key, 0.0)) < 0.0:
                                    errors.append(f"{key} must be non-negative")
                            ucdrt_wsum = sum(float(_cfg_get(m1, key, 0.0)) for key in ucdrt_weight_keys)
                            if abs(ucdrt_wsum - 1.0) > 1.0e-6:
                                errors.append("GEOTR-UCDRT objective weights must sum to 1")
                        else:
                            action_margin = float(_cfg_get(m1, "GEOTR_SLR_ACTION_MARGIN_PROB", 0.05))
                            if not (0.0 < action_margin < 0.5):
                                errors.append("GEOTR_SLR_ACTION_MARGIN_PROB must be in (0,0.5)")
                            for key in ("GEOTR_SLR_SELECTOR_WEIGHT", "GEOTR_SLR_PATCH_WEIGHT", "GEOTR_SLR_FINAL_WEIGHT", "GEOTR_SLR_SDF_WEIGHT"):
                                if float(_cfg_get(m1, key, 0.0)) < 0.0:
                                    errors.append(f"{key} must be non-negative")
                            slr_wsum = sum(float(_cfg_get(m1, key, 0.0)) for key in (
                                "GEOTR_SLR_SELECTOR_WEIGHT", "GEOTR_SLR_PATCH_WEIGHT", "GEOTR_SLR_FINAL_WEIGHT", "GEOTR_SLR_SDF_WEIGHT"
                            ))
                            if abs(slr_wsum - 1.0) > 1.0e-6:
                                errors.append("GEOTR-SLR stable objective weights must sum to 1")
                    transition_aware = bool(_cfg_get(m1, "GEOTR_AEFR_TRANSITION_AWARE_ENABLED", False))
                    if stage == "sparse_local_rerendering" and not transition_aware:
                        errors.append("GEOTR-SLR requires GEOTR_AEFR_TRANSITION_AWARE_ENABLED=true")
                    if (
                        stage in {"single_atomic", "single_continuous", "hybrid_geometry"}
                        and (not transition_aware)
                        and (not bool(_cfg_get(m1, "GEOTR_AEFR_USE_RAW_FLOW_EVIDENCE", True)))
                    ):
                        errors.append("Legacy AEFR stages may disable raw flow only in the T2 transition-aware ablation")
                    joint = bool(_cfg_get(m1, "GEOTR_AEFR_JOINT_GEOMETRY_GRAD_ENABLED", False))
                    stopgrad = bool(_cfg_get(m1, "GEOTR_STAGE2_STOPGRAD_TRANSPORT", True))
                    if joint == stopgrad:
                        errors.append("AEFR requires STOPGRAD_TRANSPORT == (not JOINT_GEOMETRY_GRAD_ENABLED)")
                if pc2r_v31 and not pc2r_v3:
                    errors.append("PC2R-v3.1 requires GEOTR_PC2R_POSTERIOR_V3_ENABLED=true")
                if (pc2r_v32 or pc2r_v32_audit) and (not aefr) and not pc2r_v31:
                    errors.append("PC2R-v3.2 root audit/operator alignment requires PC2R-v3.1 canonical coordinates")
                if pc2r_v32 and not pc2r_v32_audit:
                    errors.append("PC2R-v3.2 Operator-Aligned WHAT requires GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED=true")
                if pc2r_v32:
                    cw = float(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_CANONICAL_WEIGHT", 0.5))
                    fw = float(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_FINAL_WEIGHT", 0.5))
                    if cw <= 0.0 or fw <= 0.0:
                        errors.append("PC2R-v3.2 requires positive canonical/final operator loss weights")
                if pc2r_v3:
                    if int(_cfg_get(m1, "GEOTR_PC2R_NUM_POSTERIOR_VIEWS", 3)) != 3:
                        errors.append("PC2R-v3 requires GEOTR_PC2R_NUM_POSTERIOR_VIEWS=3")
                    if int(_cfg_get(m1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 10)) < 3:
                        errors.append("PC2R-v3 requires GEOTR_TRAIN_POSTERIOR_SAMPLES>=3")
                    if int(_cfg_get(m1, "VAL_NUM_SAMPLES", 10)) < 3:
                        errors.append("PC2R-v3 requires VAL_NUM_SAMPLES>=3")
                    if float(_cfg_get(m1, "GEOTR_PC2R_RESIDUAL_LOGIT_SCALE", 2.0)) <= 0.0:
                        errors.append("GEOTR_PC2R_RESIDUAL_LOGIT_SCALE must be >0")
                    rf = float(_cfg_get(m1, "GEOTR_PC2R_RISK_TOP_FRACTION", 0.20))
                    if not (0.0 < rf <= 1.0):
                        errors.append("GEOTR_PC2R_RISK_TOP_FRACTION must be in (0,1]")
                    da = float(_cfg_get(m1, "GEOTR_PC2R_DIRECTION_AGREEMENT_THRESHOLD", 0.80))
                    if not (0.0 <= da <= 1.0):
                        errors.append("GEOTR_PC2R_DIRECTION_AGREEMENT_THRESHOLD must be in [0,1]")
                    for key, default in (("GEOTR_PC2R_RESIDUAL_SPREAD_THRESHOLD",1.0),("GEOTR_PC2R_RESIDUAL_STRENGTH_THRESHOLD",0.05),("GEOTR_PC2R_RAW_CORRECTION_THRESHOLD",0.02)):
                        if float(_cfg_get(m1, key, default)) < 0.0:
                            errors.append(key + " must be non-negative")
        if errors:
            raise ValueError("Geometry--Residual A0--A3 protocol failed:\n  - " + "\n  - ".join(errors))
        return
    if str(_cfg_get(m1, "PROTOCOL", "")).strip().lower() != "mhcs":
        errors.append("M1.PROTOCOL must be mhcs")
    for key in ("CANDIDATE_MODE", "LOSS_MODE"):
        if str(_cfg_get(m1, key, "")).strip().lower() not in {"mhcs", "multi_hypothesis_composition"}:
            errors.append(f"M1.{key} must be multi_hypothesis_composition")
    if str(_cfg_get(m1, "INIT_CHECKPOINT", "") or "").strip():
        errors.append("R5.2 is from-scratch and forbids task-trained INIT_CHECKPOINT")
    if not bool(_cfg_get(m1, "MHCS_ROOT_CAUSAL_RESIDUAL", False)):
        errors.append("R5.2 requires MHCS_ROOT_CAUSAL_RESIDUAL=true")
    if not bool(_cfg_get(m1, "MHCS_BASE_CONDITIONED", False)):
        errors.append("R5.2 keeps the R5.1 Base-conditioned H1..H5 bank")
    if bool(_cfg_get(m1, "MHCS_BASE_FREE_ESCAPE", False)):
        errors.append("R5.2 causal M2 test forbids Base-free Escape")
    if bool(_cfg_get(m1, "MHCS_STRUCTURAL_EXPERT", False)):
        errors.append("R5.2 causal M2 test forbids Structural expert")
    if not bool(_cfg_get(m1, "MHCS_LOCAL_COMPOSER", False)):
        errors.append("R5.2 requires MHCS_LOCAL_COMPOSER=true")
    if bool(_cfg_get(m1, "MHCS_HIERARCHICAL_ROUTER", False)):
        errors.append("R5.2 forbids the R5.1 EditGate+conditional-selector router")
    if not bool(_cfg_get(m1, "MHCS_COUNTERFACTUAL_SAFETY_ROUTER", False)):
        errors.append("R5.2 requires MHCS_COUNTERFACTUAL_SAFETY_ROUTER=true")
    if not bool(_cfg_get(m1, "MHCS_LOCAL_ACTION_VERIFIER", False)):
        errors.append("R5.2 requires MHCS_LOCAL_ACTION_VERIFIER=true")
    if bool(_cfg_get(m1, "MHCS_TYPED_ELIGIBILITY", False)):
        errors.append("R5.2 M2 forbids GT/predicted typed eligibility masking")
    if not bool(_cfg_get(m1, "MHCS_ANNEALED_ASSIGNMENT", False)):
        errors.append("R5.2 retains the R5.1 candidate-bank annealed MCL objective")
    if not bool(_cfg_get(m1, "MHCS_ERROR_AS_EVIDENCE", False)):
        errors.append("R5.2 requires MHCS_ERROR_AS_EVIDENCE=true")
    if bool(_cfg_get(m1, "MHCS_EXPLICIT_VERIFIER", False)):
        errors.append("R5.2 has no whole-image verifier; MHCS_EXPLICIT_VERIFIER=false")
    if int(_cfg_get(m1, "MHCS_ERROR_STATE_CLASSES", 0)) != 3:
        errors.append("R5.2 requires Preserve/FN/FP error evidence")
    if int(_cfg_get(m1, "MHCS_ACTION_SAFETY_CLASSES", 0)) != 3:
        errors.append("R5.2 requires MHCS_ACTION_SAFETY_CLASSES=3 (Harm/Neutral/Benefit)")
    if int(_cfg_get(m1, "MHCS_NUM_HYPOTHESES", 0)) != 5:
        errors.append("R5.2 requires the unchanged five generated candidates H1..H5")
    if int(_cfg_get(m1, "MHCS_ROUTER_GRID_SIZE", 0)) < 2:
        errors.append("R5.2 requires MHCS_ROUTER_GRID_SIZE>=2")
    if float(_cfg_get(m1, "M2_LEARNING_RATE", 0.0)) != 0.0:
        errors.append("R5.2 has no historical M2 module; M2_LEARNING_RATE must be 0")
    if float(_cfg_get(m1, "M3_LEARNING_RATE", 0.0)) != 0.0:
        errors.append("R5.2 has no historical M3 module; M3_LEARNING_RATE must be 0")
    if float(_cfg_get(m1, "MHCS_M2_LEARNING_RATE", 0.0)) <= 0.0:
        errors.append("R5.2 requires positive MHCS_M2_LEARNING_RATE for the safety router")
    if float(_cfg_get(m1, "MHCS_BANK_LEARNING_RATE", 0.0)) <= 0.0:
        errors.append("R5.2 keeps a trainable R5.1 candidate bank")
    if not bool(_cfg_get(m1, "MHCS_RNG_ISOLATION", False)):
        errors.append("R5.2 requires MHCS_RNG_ISOLATION=true")
    if not bool(_cfg_get(m1, "MHCS_EXACT_PRESERVE", False)):
        errors.append("R5.2 requires MHCS_EXACT_PRESERVE=true")
    if bool(_cfg_get(m1, "MHCS_TRAIN_STRAIGHT_THROUGH_ROUTE", False)):
        errors.append("R5.2 forbids straight-through routing")
    if bool(_cfg_get(m1, "MHCS_SIGN_BALANCED_RISK", False)):
        errors.append("R5.2 forbids signed-risk regression")
    if str(_cfg_get(m1, "MHCS_M2_GRAD_BALANCE", "none")).strip().lower() != "none":
        errors.append("R5.2 requires MHCS_M2_GRAD_BALANCE=none")
    if float(_cfg_get(m1, "MHCS_MAX_RESIDUAL_LOGIT", 0.0)) <= 0.0:
        errors.append("R5.2 keeps positive R5.1 residual capacity")
    for key in ("MHCS_ERROR_FOCAL_GAMMA", "MHCS_SUPPORT_FOCAL_GAMMA", "MHCS_ACTION_SAFETY_FOCAL_GAMMA"):
        if float(_cfg_get(m1, key, -1.0)) < 0.0:
            errors.append(f"R5.2 requires {key}>=0")
    for key in ("MHCS_ACTION_BENEFIT_MARGIN", "MHCS_ACTION_HARM_MARGIN", "MHCS_ACTION_SAFETY_MARGIN"):
        if float(_cfg_get(m1, key, -1.0)) < 0.0:
            errors.append(f"R5.2 requires {key}>=0")
    benefit_thr = float(_cfg_get(m1, "MHCS_ACTION_BENEFIT_THRESHOLD", -1.0))
    harm_thr = float(_cfg_get(m1, "MHCS_ACTION_HARM_THRESHOLD", -1.0))
    if not (0.0 < benefit_thr < 1.0):
        errors.append("R5.2 requires 0<MHCS_ACTION_BENEFIT_THRESHOLD<1")
    if not (0.0 <= harm_thr < 1.0):
        errors.append("R5.2 requires 0<=MHCS_ACTION_HARM_THRESHOLD<1")
    m2_dropout = float(_cfg_get(m1, "MHCS_M2_DROPOUT", 0.0))
    if not (0.0 <= m2_dropout < 0.5):
        errors.append("R5.2 requires 0<=MHCS_M2_DROPOUT<0.5")
    t0 = float(_cfg_get(m1, "MHCS_ASSIGNMENT_TEMPERATURE_START", 0.0))
    tmin = float(_cfg_get(m1, "MHCS_ASSIGNMENT_TEMPERATURE_MIN", 0.0))
    decay = float(_cfg_get(m1, "MHCS_ASSIGNMENT_TEMPERATURE_DECAY", 0.0))
    if not (t0 > 0.0 and tmin > 0.0 and t0 >= tmin):
        errors.append("R5.2 keeps valid R5.1 candidate-bank annealing temperatures")
    if not (0.0 < decay <= 1.0):
        errors.append("R5.2 requires 0<MHCS_ASSIGNMENT_TEMPERATURE_DECAY<=1")
    if errors:
        raise ValueError("MHCS-R5.2 protocol failed:\n  - " + "\n  - ".join(errors))

def _build_v479_sampler(dataset, cfg):
    if not bool(_cfg_get(cfg.TRAIN, "USE_STRATIFIED_SAMPLER", False)):
        return None
    pairs = getattr(dataset, "data_pairs", None)
    if not pairs:
        return None
    malignant_weight = float(_cfg_get(cfg.TRAIN, "MALIGNANT_SAMPLE_WEIGHT", 1.5))
    midsize_weight = float(_cfg_get(cfg.TRAIN, "MID_SIZE_SAMPLE_WEIGHT", 1.3))
    max_weight = float(_cfg_get(cfg.TRAIN, "MAX_SAMPLE_WEIGHT", 2.0))
    weights = []
    for pair in pairs:
        image_path, mask_path = str(pair[0]), str(pair[1])
        name = os.path.basename(image_path).lower()
        weight = malignant_weight if "malignant" in name else 1.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            area = float((mask > 127).mean())
            if 0.01 <= area < 0.08:
                weight *= midsize_weight
        weights.append(min(max_weight, weight))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )

def main():
    cfg = get_arguments()
    _validate_v547_protocol(cfg)
    _validate_v548_protocol(cfg)
    _validate_v549_protocol(cfg)
    _validate_v550_protocol(cfg)
    _validate_v551_rootfix_protocol(cfg)
    _validate_v560_clean_core_protocol(cfg)
    _validate_v561_bcrs_protocol(cfg)
    _validate_v562_rootfix_protocol(cfg)
    _validate_v563_rootfix_protocol(cfg)
    _validate_v564_rootfix_protocol(cfg)
    _validate_v565_rootfix_protocol(cfg)
    _validate_clean_dynamic_component_set_protocol(cfg)
    _validate_tc_drcs_protocol(cfg)
    _validate_semlt_protocol(cfg)
    _validate_mhcs_protocol(cfg)

    # V398: record the BUSI B0 initialization source in the experiment config.
    # A command-line --init-checkpoint still takes precedence, but when it is
    # omitted the configured M1.INIT_CHECKPOINT is loaded after the V398 model
    # is built and before optimizer/EMA construction.
    configured_init_checkpoint = str(
        _cfg_get(_cfg_get(cfg, "M1", None), "INIT_CHECKPOINT", "") or ""
    ).strip()
    if not str(getattr(cfg, "init_checkpoint", "") or "").strip() and configured_init_checkpoint:
        cfg.init_checkpoint = configured_init_checkpoint

    if cfg.data_percentage != 100:
        cfg.DATASET.NAME = f"{cfg.DATASET.NAME}_{cfg.data_percentage}"

    mode = m1_train_mode(cfg) if m1_enabled(cfg) else "disabled"
    if (
        m1_enabled(cfg)
        and bool(_cfg_get(cfg.M1, "V470_STRICT_JOINT_E2E", False))
        and str(getattr(cfg, "init_checkpoint", "") or "").strip()
    ):
        raise ValueError(
            "V470 strict joint E2E forbids loading a task-trained checkpoint. "
            "Use only the pretrained UniMedCLIP/BiomedBERT files declared in MODEL."
        )
    if m1_enabled(cfg) and mode in {"frozen", "anchor_student"} and not cfg.init_checkpoint:
        raise ValueError(
            f"M1.TRAIN_MODE={mode!r} requires --init-checkpoint pointing to the B0 best-Dice checkpoint."
        )

    if _v531_uses_source_checkpoint(cfg) and not str(
        getattr(cfg, "init_checkpoint", "") or ""
    ).strip():
        raise ValueError(
            "V531 requires --init-checkpoint pointing to a Val-selected "
            "V518 typed-M1 checkpoint (or a previous V531 checkpoint for "
            "joint continuation). Set M1.V531_SOURCE_REQUIRES_M1=false only "
            "for a declared from-scratch ablation."
        )

    if _v519_uses_v518_source(cfg) and not str(getattr(cfg, "init_checkpoint", "") or "").strip():
        raise ValueError(
            "V519 requires --init-checkpoint pointing to the Val-selected "
            "V518 Base/PVL/M1 checkpoint."
        )

    if _v488_is_m2m3_only(cfg) and not str(getattr(cfg, "init_checkpoint", "") or "").strip():
        raise ValueError(
            "M1.V488_M2M3_ONLY=true requires --init-checkpoint pointing to the "
            "Val-selected V487A M1 checkpoint. Random or modified M1 is forbidden."
        )

    if (
        _v490_is_end_to_end(cfg)
        and str(getattr(cfg, "init_checkpoint", "") or "").strip()
        and not (
            m1_enabled(cfg)
            and (
                bool(_cfg_get(_cfg_get(cfg, "M1", None), "V515_OFFICIAL_BASE_INIT", False))
                or _v519_uses_v518_source(cfg)
                or bool(_cfg_get(_cfg_get(cfg, "M1", None), "V531_TYPED_SPARSE_REFINER_ENABLED", False))
            )
        )
    ):
        raise ValueError(
            "V490 is a task-level from-scratch protocol. Remove M1.INIT_CHECKPOINT "
            "and do not pass --init-checkpoint. Only the pretrained image/text "
            "backbones declared in MODEL are allowed."
        )
    if _v489_is_end_to_end(cfg) and not str(getattr(cfg, "init_checkpoint", "") or "").strip():
        raise ValueError(
            "M1.V489_END_TO_END_ENABLED=true requires INIT_CHECKPOINT or "
            "--init-checkpoint pointing to the Val-selected V487A Base+M1 checkpoint. "
            "The checkpoint is only initialization; all downstream modules are then trainable."
        )

    if (
        m1_enabled(cfg)
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V486_FIXED_BASE_PROBE", False))
        and not str(getattr(cfg, "init_checkpoint", "") or "").strip()
    ):
        raise ValueError(
            "M1.V486_FIXED_BASE_PROBE=true requires --init-checkpoint or "
            "M1.INIT_CHECKPOINT pointing to a stable A0/Base checkpoint. "
            "This mode is explicitly designed to protect Base by freezing it."
        )

    if (
        m1_enabled(cfg)
        and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V487_BASE_SAFE_E2E", False))
        and str(getattr(cfg, "init_checkpoint", "") or "").strip()
        and not bool(_cfg_get(_cfg_get(cfg, "M1", None), "V515_OFFICIAL_BASE_INIT", False))
        and not _v519_uses_v518_source(cfg)
        and not bool(_cfg_get(_cfg_get(cfg, "M1", None), "V531_TYPED_SPARSE_REFINER_ENABLED", False))
    ):
        raise ValueError(
            "M1.V487_BASE_SAFE_E2E=true is the final no-external-B0 end-to-end mode; "
            "do not pass --init-checkpoint unless this is the declared V519 "
            "V518-source initialization protocol. Base is trained only by "
            "base_loss while proposal gradients are isolated."
        )

    run_name = results_name(cfg)
    checkpoint_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logger_config(os.path.join(checkpoint_dir, "log.txt"))
    logger.info("************\n** Config **\n************\n%s", cfg)
    logger.info("Experiment name: %s", run_name)
    _rbal_edge_w = float(_cfg_get(cfg.TRAIN, "RBAL_EDGE_WEIGHT", 0.0))
    _rbal_normal_w = float(_cfg_get(cfg.TRAIN, "RBAL_NORMAL_WEIGHT", 0.0))
    if _rbal_edge_w > 0.0 or _rbal_normal_w > 0.0:
        _rbal_schedule_name = (
            str(_cfg_get(cfg.TRAIN, "RBAL_SCHEDULE_TYPE", "off"))
            if bool(_cfg_get(cfg.TRAIN, "RBAL_SCHEDULE_ENABLED", False)) else "off"
        )
        logger.info(
            "[JBTL6_GLOBAL_GEOMETRY] edge_weight=%.4f normal_weight=%.4f boundary_radius=%d "
            "loss_type=%s schedule=%s hold=%d decay_end=%d grad_scope=%s; "
            "M1/candidate/selector disabled when M1.ENABLED=false.",
            _rbal_edge_w,
            _rbal_normal_w,
            int(_cfg_get(cfg.TRAIN, "RBAL_BOUNDARY_RADIUS_PX", 1)),
            str(_cfg_get(cfg.TRAIN, "RBAL_AUX_LOSS_TYPE", "surface")),
            _rbal_schedule_name,
            int(_cfg_get(cfg.TRAIN, "RBAL_FULL_WEIGHT_EPOCHS", 20)),
            int(_cfg_get(cfg.TRAIN, "RBAL_DECAY_END_EPOCH", 80)),
            str(_cfg_get(cfg.TRAIN, "RBAL_EDGE_GRAD_SCOPE", "all")),
        )
        if _rbal_schedule_name.strip().lower() == "matched_budget":
            logger.info(
                "[JBTL6_MATCHED_BUDGET] active=%d reference_T=%d target_T=%d "
                "base_lr=%.8f eta_min=%.8f max_scale=%.4f",
                int(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_ACTIVE_EPOCHS", 20)),
                int(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_REFERENCE_EPOCHS", 20)),
                int(_cfg_get(cfg.TRAIN, "SCHEDULER_TOTAL_EPOCHS", 100)),
                float(_cfg_get(cfg.TRAIN, "LEARNING_RATE", 3.0e-4)),
                float(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_ETA_MIN", 1.0e-4)),
                float(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_MAX_SCALE", 1.0)),
            )
    if cfg.init_checkpoint:
        logger.info("Initialization checkpoint (loaded before optimizer/EMA): %s", cfg.init_checkpoint)
    elif _v490_is_end_to_end(cfg):
        logger.info(
            "[V490_FROM_SCRATCH] No task-trained Base/M1/M2/M3 checkpoint is loaded. "
            "Only the pretrained image/text backbones declared in MODEL are used."
        )
    if m1_enabled(cfg):
        if _semlt(cfg):
            if bool(_cfg_get(cfg.TRAIN, "USE_VALIDATION_SELECTION", False)):
                logger.info(
                    "SemLT M1-only mode=%s; Base/PVL and bounded logit Transport are "
                    "optimized in one run. M2/M3 are absent. Validation is used only "
                    "for native_m1_dice/native_m1_nsd checkpoint selection; Test remains unopened.",
                    mode,
                )
            else:
                logger.info(
                    "SemLT M1-only mode=%s; Base/PVL and bounded logit Transport are "
                    "optimized in one run. M2/M3 are absent. Val/Test remain unopened "
                    "during training; the physical last-epoch checkpoint is pre-declared.",
                    mode,
                )
        else:
            logger.info(
                "Unified mode=%s; Preserve/Base, typed candidates, causal gate and "
                "CCV are optimized in one run. Validation may be opened only for "
                "pre-declared checkpoint selection; Test remains unopened.",
                mode,
            )
        if bool(_cfg_get(cfg.M1, "GEOTOPO_REFINEMENT_ENABLED", False)):
            posterior_order = str(
                _cfg_get(cfg.M1, "GEOTR_POSTERIOR_INFERENCE_ORDER", "mean_then_refine")
            ).strip().lower()
            aliases = {
                "samplewise": "refine_then_mean",
                "samplewise_refine": "refine_then_mean",
                "mean_first": "mean_then_refine",
            }
            posterior_order = aliases.get(posterior_order, posterior_order)
            if posterior_order not in {"mean_then_refine", "refine_then_mean"}:
                raise ValueError(
                    "M1.GEOTR_POSTERIOR_INFERENCE_ORDER must be mean_then_refine "
                    f"or refine_then_mean, got {posterior_order!r}."
                )
            val_mc = int(_cfg_get(cfg.M1, "VAL_NUM_SAMPLES", 10))
            test_mc = int(_cfg_get(cfg.TEST, "NUM_SAMPLES", val_mc))
            if bool(_cfg_get(cfg.M1, "GEOTR_REQUIRE_VAL_TEST_MC_MATCH", False)) and val_mc != test_mc:
                raise ValueError(
                    "GEOTR posterior protocol requires VAL_NUM_SAMPLES == TEST.NUM_SAMPLES; "
                    f"got {val_mc} vs {test_mc}."
                )
            checkpoint_policy = (
                "validation-only selection"
                if bool(_cfg_get(cfg.TRAIN, "USE_VALIDATION_SELECTION", False))
                else "physical last epoch"
            )
            logger.info(
                "[GEOTR_POSTERIOR_PROTOCOL] order=%s | ValMC=%d TestMC=%d | "
                "checkpoint=%s; Test unopened.",
                posterior_order, val_mc, test_mc, checkpoint_policy,
            )

    if cfg.seed >= 0:
        logger.info("Setting fixed seed: %d", cfg.seed)
        set_random_seed(cfg.seed)
        public_cudnn_parity = bool(
            _cfg_get(_cfg_get(cfg, "M1", None), "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False)
            and _cfg_get(cfg.TRAIN, "REPLAY_PUBLIC_REPO_CUDNN_FLAGS", False)
        )
        if public_cudnn_parity:
            # Released MedCLIPSeg train.py explicitly sets deterministic=True
            # and benchmark=False in set_random_seed().  Keep those exact flags
            # when a formal Base-parity run requests public-repo replay.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            logger.info(
                "[PUBLIC_REPO_CUDNN_PARITY] benchmark=False deterministic=True"
            )
        elif bool(_cfg_get(cfg.TRAIN, "DETERMINISTIC", False)):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            logger.info(
                "Deterministic cuDNN mode enabled for reproducible formal runs."
            )

    strict_causal_parity = bool(
        m1_enabled(cfg)
        and _cfg_get(cfg.M1, "SEMLT_STRICT_CAUSAL_PARITY", False)
    )
    if strict_causal_parity:
        # cudnn.deterministic alone does not cover every CUDA backward kernel.
        # The observed A0/A1 mismatch occurred immediately after the identical
        # Base loss backward and before M1 ran, so fail closed on any remaining
        # request deterministic kernels wherever PyTorch provides them instead
        # of misattributing the divergence to M1 RNG.
        # ``grid_sample`` CUDA backward has no switchable deterministic path in
        # current PyTorch, so warn (and verify the actual Base hashes) rather
        # than making the physical transport objective impossible to execute.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            # R2.3-DP: ``use_deterministic_algorithms(..., warn_only=True)``
            # does not prevent PyTorch from selecting the memory-efficient
            # scaled-dot-product-attention backward.  That kernel emitted an
            # explicit nondeterminism warning in the R2.1/R2.2 logs and the
            # protected Base/PVL hashes consequently separated at epoch 4.
            # Force the reference math backend before model construction.  The
            # four calls are guarded for compatibility with older torch builds.
            # This changes only the attention implementation, not its equation,
            # parameters, data order, optimizer, or objective.
            sdp_switches = (
                ("enable_flash_sdp", False),
                ("enable_mem_efficient_sdp", False),
                ("enable_cudnn_sdp", False),
                ("enable_math_sdp", True),
            )
            for switch_name, enabled in sdp_switches:
                switch = getattr(torch.backends.cuda, switch_name, None)
                if switch is not None:
                    switch(enabled)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except AttributeError:
            pass
        sdp_state = {}
        if torch.cuda.is_available():
            for state_name in (
                "flash_sdp_enabled",
                "mem_efficient_sdp_enabled",
                "cudnn_sdp_enabled",
                "math_sdp_enabled",
            ):
                state = getattr(torch.backends.cuda, state_name, None)
                sdp_state[state_name] = None if state is None else bool(state())
        logger.info(
            "[UCFNRT_STRICT_CAUSAL_PARITY] deterministic algorithms enabled "
            "(unsupported CUDA ops warn); TF32 disabled; math-only SDPA "
            "requested before model construction; CUBLAS_WORKSPACE_CONFIG=%s; "
            "sdpa_state=%s; actual Base hashes remain mandatory; M1 "
            "grid_sample CUDA backward remains statistically replicated, not "
            "claimed bitwise deterministic",
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            sdp_state,
        )

    # JBT-Lite v8: strict reproducibility is independent of M1.
    #
    # v6 DIAG40 and v7 FORMAL100 used the same seed, data order, optimizer,
    # scheduler horizon, and the same first-40-epoch EDGE schedule, yet their
    # protected Base/PVL hashes diverged.  The historical strict CUDA block
    # above is gated by M1.ENABLED, so it never ran for JBT-Lite (M1 is off).
    # This flag closes that reproducibility hole for Base/EDGE ablations.
    jbtl_strict_repro = bool(
        _cfg_get(cfg.TRAIN, "STRICT_REPRODUCIBILITY", False)
    )
    if jbtl_strict_repro:
        expected_hash_seed = str(int(cfg.seed))
        actual_hash_seed = os.environ.get("PYTHONHASHSEED", "")
        if actual_hash_seed != expected_hash_seed:
            raise RuntimeError(
                "[JBTL8_STRICT_REPRO] PYTHONHASHSEED must be exported before "
                f"Python starts; expected {expected_hash_seed!r}, got "
                f"{actual_hash_seed!r}. Use the v8 launcher."
            )

        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        if workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "[JBTL8_STRICT_REPRO] CUBLAS_WORKSPACE_CONFIG must be exported "
                "before Python starts as ':4096:8' (recommended) or ':16:8'; "
                f"got {workspace!r}. Use the v8 launcher."
            )

        # Fail closed instead of silently accepting a nondeterministic CUDA
        # kernel.  NORMAL/grid_sample and all legacy M1 branches are disabled
        # in the v8 ablation, so unsupported operators indicate a real protocol
        # violation rather than an expected component of the method.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            for switch_name, enabled in (
                ("enable_flash_sdp", False),
                ("enable_mem_efficient_sdp", False),
                ("enable_cudnn_sdp", False),
                ("enable_math_sdp", True),
            ):
                switch = getattr(torch.backends.cuda, switch_name, None)
                if switch is not None:
                    switch(enabled)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except AttributeError:
            pass

        sdp_state = {}
        if torch.cuda.is_available():
            for state_name in (
                "flash_sdp_enabled",
                "mem_efficient_sdp_enabled",
                "cudnn_sdp_enabled",
                "math_sdp_enabled",
            ):
                state = getattr(torch.backends.cuda, state_name, None)
                sdp_state[state_name] = None if state is None else bool(state())
        logger.info(
            "[JBTL8_STRICT_REPRO] enabled | deterministic_algorithms=True "
            "fail_closed=True | cudnn benchmark=False deterministic=True | "
            "TF32=False | CUBLAS_WORKSPACE_CONFIG=%s | PYTHONHASHSEED=%s | "
            "sdpa_state=%s",
            workspace, actual_hash_seed, sdp_state,
        )

    if (
        not strict_causal_parity
        and not jbtl_strict_repro
        and bool(_cfg_get(cfg.TRAIN, "ALLOW_TF32", False))
        and torch.cuda.is_available()
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        logger.info("TF32 tensor-core math enabled for this speed-optimised run.")

    ce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")
    official_base_contract = bool(
        _cfg_get(
            _cfg_get(cfg, "M1", None),
            "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT",
            False,
        )
    )
    # Official MedCLIPSeg uses only its original RandomGenerator: 50% random
    # rotation in [-20,20] followed by resize/CLIP normalization.  Do not pass
    # the project-specific cfg in official-base mode so later augmentation
    # extensions cannot silently alter the Base training distribution.
    if official_base_contract:
        train_transform = transforms.Compose([
            RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
        ])
    else:
        train_transform = transforms.Compose([
            RandomGenerator(
                output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE],
                cfg=cfg,
            )
        ])

    train_text_file = (
        f"Train_text_{cfg.data_percentage}.xlsx"
        if cfg.data_percentage != 100
        else "Train_text.xlsx"
    )

    train_text = read_text(
        os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, train_text_file)
    )

    worker_init_fn = (
        official_medclipseg_worker_init_fn_factory(cfg.seed)
        if official_base_contract
        else worker_init_fn_factory(cfg.seed)
    )

    # Public MedCLIPSeg train.py hard-codes eight workers.
    num_workers = 8 if official_base_contract else int(_cfg_get(cfg.TRAIN, "NUM_WORKERS", 8))

    generator = None
    official_batch_rng_lock = bool(
        _cfg_get(
            _cfg_get(cfg, "M1", None),
            "OFFICIAL_BASE_BATCH_RNG_LOCK",
            False,
        )
    )
    if official_batch_rng_lock:
        # v6.3.5: keep the shuffled sample order independent of variant-specific
        # model construction and set_epoch hooks.  With generator=None the
        # official DataLoader draws both its iterator base_seed and RandomSampler
        # seed from the process-global torch RNG.  A JBT-only module can then
        # shift the first physical Base batch even though Base weights and the
        # per-batch forward RNG are identical.  A private generator seeded with
        # the experiment seed reproduces the clean seed-N stream while making
        # it impossible for model-side RNG consumption to alter data order.
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))
        logger.info(
            "[JBT_V635_DATALOADER_RNG_LOCK] seed=%d private shuffle/iterator stream",
            int(cfg.seed),
        )
    elif not official_base_contract:
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))

    slr_true_hr = _slr_true_hr_enabled(cfg)
    if slr_true_hr:
        train_dataset = SLRPairedResolutionDataset(
            cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text,
            image_size=cfg.DATASET.SIZE,
            hr_size=int(_cfg_get(cfg.M1, "GEOTR_SLR_HR_SIZE", 448)),
            training=True,
            cfg=cfg,
        )
        logger.info("[GEOTR-SLR-HR] true paired-resolution train data active: LR=%d HR=%d", int(cfg.DATASET.SIZE), int(_cfg_get(cfg.M1, "GEOTR_SLR_HR_SIZE", 448)))
    else:
        train_dataset = DatasetSegmentation(
            cfg.DATASET.TRAIN_PATH,
            cfg.DATASET.NAME,
            train_text,
            train_transform,
            image_size=cfg.DATASET.SIZE,
        )
    train_sampler = _build_v479_sampler(train_dataset, cfg)
    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=True if official_base_contract else torch.cuda.is_available(),
        drop_last=False if official_base_contract else bool(_cfg_get(cfg.TRAIN, "DROP_LAST", False)),
    )
    # The public loader normally does not pass a torch.Generator.  The audited
    # paired path is the sole exception: it needs a private stream so matched
    # Base and JBT see the same physical batches after different model builds.
    if generator is not None:
        train_loader_kwargs["generator"] = generator
    train_dataloader = DataLoader(**train_loader_kwargs)

    use_validation_selection = bool(
        _cfg_get(cfg.TRAIN, "USE_VALIDATION_SELECTION", False)
    )
    replay_public_val_rng = bool(
        official_base_contract
        and _cfg_get(cfg.TRAIN, "REPLAY_PUBLIC_REPO_VALIDATION_RNG", False)
    )
    val_dataloader = None
    if use_validation_selection or replay_public_val_rng:
        val_text = read_text(
            os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, "Val_text.xlsx")
        )
        val_transform = transforms.Compose([
            ValGenerator(
                output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]
            )
        ])
        val_dataset = (
            SLRPairedResolutionDataset(
                cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text,
                image_size=cfg.DATASET.SIZE,
                hr_size=int(_cfg_get(cfg.M1, "GEOTR_SLR_HR_SIZE", 448)),
                training=False,
                cfg=cfg,
            ) if slr_true_hr else DatasetSegmentation(
                cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_transform, image_size=cfg.DATASET.SIZE
            )
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(
                _cfg_get(cfg.TRAIN, "VAL_BATCH_SIZE", cfg.TRAIN.BATCH_SIZE)
            ),
            shuffle=False,
            worker_init_fn=(worker_init_fn if official_base_contract else None),
            num_workers=num_workers,
            pin_memory=(True if official_base_contract else torch.cuda.is_available()),
        )
        if replay_public_val_rng and not use_validation_selection:
            logger.info(
                "[PUBLIC_REPO_RNG_REPLAY] Val is not used for checkpoint selection, "
                "but its loader + stochastic PVL forwards are replayed every epoch "
                "to match the released MedCLIPSeg training RNG trajectory."
            )
        logger.info(
            "[FORMAL_PROTOCOL] Validation checkpoint selection enabled: "
            "metric=%s interval=%d; Test remains unopened.",
            str(_cfg_get(cfg.TRAIN, "VAL_SELECTION_METRIC", "fusion_dice")),
            int(_cfg_get(cfg.TRAIN, "VAL_INTERVAL", 1)),
        )

    # MHCS-R4.8 FixedTrainDiag: deterministic, augmentation-free Train subset.
    # It is constructed independently from the shuffled/augmented training loader
    # and never touches Test. By default it matches Val-set cardinality and uses
    # systematic indices over the full Train set rather than a cherry-picked prefix.
    fixed_train_dataloader = None
    if _mhcs(cfg) and bool(_cfg_get(cfg.TRAIN, "MHCS_FIXED_TRAIN_DIAG_ENABLED", False)):
        fixed_transform = transforms.Compose([
            ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
        ])
        fixed_dataset_full = (
            SLRPairedResolutionDataset(
                cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text,
                image_size=cfg.DATASET.SIZE,
                hr_size=int(_cfg_get(cfg.M1, "GEOTR_SLR_HR_SIZE", 448)),
                training=False,
                cfg=cfg,
            ) if slr_true_hr else DatasetSegmentation(
                cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, fixed_transform, image_size=cfg.DATASET.SIZE
            )
        )
        requested_cases = int(_cfg_get(cfg.TRAIN, "MHCS_FIXED_TRAIN_DIAG_CASES", 0))
        if requested_cases <= 0:
            requested_cases = (
                len(val_dataloader.dataset)
                if val_dataloader is not None else min(78, len(fixed_dataset_full))
            )
        n_fixed = max(1, min(requested_cases, len(fixed_dataset_full)))
        if n_fixed == len(fixed_dataset_full):
            fixed_indices = list(range(n_fixed))
        elif n_fixed == 1:
            fixed_indices = [len(fixed_dataset_full) // 2]
        else:
            fixed_indices = torch.linspace(
                0, len(fixed_dataset_full) - 1, steps=n_fixed
            ).round().long().unique(sorted=True).tolist()
            # Rounding can only reduce cardinality in pathological tiny datasets.
            if len(fixed_indices) < n_fixed:
                used = set(fixed_indices)
                for idx in range(len(fixed_dataset_full)):
                    if idx not in used:
                        fixed_indices.append(idx)
                        used.add(idx)
                    if len(fixed_indices) == n_fixed:
                        break
                fixed_indices = sorted(fixed_indices)
        fixed_train_dataloader = DataLoader(
            Subset(fixed_dataset_full, fixed_indices),
            batch_size=int(_cfg_get(cfg.TRAIN, "VAL_BATCH_SIZE", cfg.TRAIN.BATCH_SIZE)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(
            "[MHCS-R4.8 FIXED_TRAIN_DIAG] cases=%d/%d augmentation=off systematic_indices=true Test=unopened",
            len(fixed_indices), len(fixed_dataset_full),
        )

    model = build_model(cfg).to(cfg.MODEL.DEVICE)
    if _v487_is_m1_only_ablation(cfg):
        logger.info(
            "[V487A_M1_ONLY_CONFIG_ONLY] enabled | "
            "M2/M3 are disabled by config/loss/deploy switches; "
            "no extra parameter freezing is applied."
        )
    if cfg.init_checkpoint:
        if bool(_cfg_get(_cfg_get(cfg, "M1", None), "V531_TYPED_SPARSE_REFINER_ENABLED", False)):
            _load_v531_source_checkpoint(model, cfg.init_checkpoint, logger, cfg)
        elif _v519_uses_v518_source(cfg):
            _load_v519_source_checkpoint(model, cfg.init_checkpoint, logger)
        elif _v489_is_end_to_end(cfg):
            _load_v489_source_checkpoint(model, cfg.init_checkpoint, logger)
        elif _v488_is_m2m3_only(cfg):
            _load_v488_m1_source_checkpoint(model, cfg.init_checkpoint, logger)
        elif m1_enabled(cfg) and bool(_cfg_get(_cfg_get(cfg, "M1", None), "V515_OFFICIAL_BASE_INIT", False)):
            _load_v515_official_base_checkpoint(model, cfg.init_checkpoint, logger)
        else:
            _load_initial_base_checkpoint(model, cfg.init_checkpoint, logger)

    # Safe residual M1-only protocol: B0 is loaded first, then the verified
    # legacy candidate bank is loaded strictly for A1--A8 preservation.  Only
    # m1_pse.safe_* tensors remain trainable; V410 is not imported or changed.
    if m1_enabled(cfg) and bool(
        _cfg_get(cfg.M1, "SAFE_RESIDUAL_ONLY_TRAINING", False)
    ):
        safe_legacy_checkpoint = str(
            _cfg_get(
                cfg.M1,
                "SAFE_RESIDUAL_BASE_M1_CHECKPOINT",
                "",
            )
            or ""
        ).strip()
        if not safe_legacy_checkpoint:
            raise ValueError(
                "M1.SAFE_RESIDUAL_BASE_M1_CHECKPOINT is required when "
                "SAFE_RESIDUAL_ONLY_TRAINING=true."
            )
        _load_frozen_m1_checkpoint(
            model,
            safe_legacy_checkpoint,
            logger,
        )
        logger.info(
            "[SAFE_RESIDUAL_M1_ONLY] verified legacy A1--A8 loaded from: %s",
            safe_legacy_checkpoint,
        )

    # A3: load C9 only from the accepted A1 fixed-endpoint checkpoint.
    # The builder has already frozen C9; new certified C10 tensors remain
    # independent and trainable.
    if m1_enabled(cfg) and bool(
        _cfg_get(cfg.M1, "SAFE_CONTEXT_CERTIFIED_ONLY", False)
    ):
        c9_reference_checkpoint = str(
            _cfg_get(
                cfg.M1,
                "SAFE_CONTEXT_REFERENCE_CHECKPOINT",
                "",
            )
            or ""
        ).strip()
        if not c9_reference_checkpoint:
            raise ValueError(
                "M1.SAFE_CONTEXT_REFERENCE_CHECKPOINT is required when "
                "SAFE_CONTEXT_CERTIFIED_ONLY=true."
            )
        _load_safe_boundary_reference_checkpoint(
            model,
            c9_reference_checkpoint,
            logger,
        )

    # V423 strict M1 candidate-only pretraining:
    # Only candidate proposal/repair tensors are trainable.
    # B0, M2 factual-control verifier, M3 selector, calibrators and
    # historical auxiliary policy heads stay frozen.
    if m1_enabled(cfg) and bool(
        _cfg_get(cfg.M1, "M1_ONLY_CANDIDATE_PRETRAIN", False)
    ):
        active_names = []

        candidate_generator_prefixes = (
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.v35_residual_head.",
        )

        for name, parameter in model.named_parameters():
            trainable = name.startswith(candidate_generator_prefixes)
            parameter.requires_grad_(trainable)

            if trainable:
                active_names.append(name)

        forbidden_tokens = (
            "cf_verifier",
            "selector",
            "v23_",
            "v38_",
            "v381_",
            "v393_",
            "v394_",
            "v396_",
            "safe_",
            "m2_tide_repair",
        )

        forbidden_active = [
            name for name in active_names
            if any(token in name for token in forbidden_tokens)
        ]

        if forbidden_active:
            raise RuntimeError(
                "V423 strict M1 contract violated; non-generator tensors "
                f"remain trainable: {forbidden_active[:12]}"
            )

        logger.info(
            "[V423 strict M1 candidate-only] "
            "B0/M2/M3/calibrators/safe branches frozen; "
            "candidate-generator tensors=%d; parameters=%d",
            len(active_names),
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )

    # V404 mode switch:
    # - Joint-E2E: B0 initializes the model; M1/M2 start from scratch and train together.
    # - Frozen mode: load an existing validated M1 checkpoint and train M2 only.
    if m1_enabled(cfg) and bool(_cfg_get(cfg.M1, "M2_TIDE_REPAIR_ENABLED", False)):
        joint_e2e = bool(
            _cfg_get(cfg.M1, "M2_TIDE_REPAIR_JOINT_E2E", False)
        )

        if joint_e2e:
            logger.info(
                "[V404 Joint-E2E] B0 initialized; M1 and M2 start from scratch. "
                "Skipping TIDE_M1_CHECKPOINT loading and skipping B0/M1 freezing."
            )
        else:
            m1_checkpoint = str(
                _cfg_get(cfg.M1, "TIDE_M1_CHECKPOINT", "") or ""
            ).strip()

            _load_frozen_m1_checkpoint(model, m1_checkpoint, logger)

            if hasattr(model, "set_m2_tide_repair_trainable"):
                model.set_m2_tide_repair_trainable()
                logger.info(
                    "[V404 Frozen] B0 + verified M1 frozen; "
                    "only FN/FP/Boundary M2 heads trainable."
                )

    teacher = None
    if m1_enabled(cfg) and mode == "anchor_student":
        teacher = _build_frozen_teacher(cfg, cfg.init_checkpoint, logger)

    _force_unified_e2e_trainable(model, cfg, logger)
    _enforce_v552_geometry_owner_trainability(model, cfg, logger)
    _enforce_v552r4201_clean_trainability(model, cfg, logger)

    # v6.3.6 recovery: use the completed matched Base as an immutable semantic
    # host. The old run stored hashes but not per-batch augmentation/Adam state,
    # so an already-diverged from-scratch trajectory cannot be reconstructed.
    jbt_fixed_base_recovery = (
        os.environ.get("JBT_FIXED_BASE_RECOVERY", "0").strip() == "1"
    )
    if jbt_fixed_base_recovery:
        fixed_base_checkpoint = os.environ.get(
            "JBT_FIXED_BASE_CHECKPOINT", ""
        ).strip()
        if not fixed_base_checkpoint:
            raise RuntimeError(
                "JBT_FIXED_BASE_RECOVERY=1 requires JBT_FIXED_BASE_CHECKPOINT"
            )
        _load_v515_official_base_checkpoint(
            model, fixed_base_checkpoint, logger
        )
        logger.info(
            "[JBT_V636_FIXED_BASE_LOAD] checkpoint=%s | protected Base/PVL "
            "loaded before optimizer construction; Base lr will be exactly zero",
            fixed_base_checkpoint,
        )

    # ROOTCAUSE-A2: same-code Base parity audit.  This is deliberately placed
    # after all trainability contracts but before optimizer construction/steps.
    # It lets C0/C1/C2/C3/F1 prove that shared Base/PVL initialization is
    # bit-identical rather than comparing absolute Test scores across code
    # revisions.
    ucfnrt_base_parity_audit = bool(
        (m1_enabled(cfg) and bool(_cfg_get(cfg.M1, "SEMLT_BASE_PARITY_AUDIT", False)))
        or bool(_cfg_get(cfg.M1, "OFFICIAL_BASE_PARITY_AUDIT", False))
    )
    if ucfnrt_base_parity_audit:
        parity_init_hash, parity_init_count = _v552r4204_base_fingerprint(model)
        logger.info(
            "[UCFNRT_BASE_PARITY_INIT] sha256=%s protected_tensors=%d",
            parity_init_hash, parity_init_count,
        )

    if bool(_cfg_get(cfg.M1, "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False)):
        logger.info(
            "[OFFICIAL_MEDCLIPSEG_BASE_CONTRACT] Base/PVL uses official augmentation, "
            "worker seeding, Adam 3e-4 and cosine eta_min=1e-4; R3 loss is hard-isolated "
            "to m1_pse.* and cannot modify protected Base/PVL gradients."
        )
    if bool(_cfg_get(cfg.M1, "BASE_TRAJECTORY_LOCK", False)):
        logger.info(
            "[BASE_TRAJECTORY_LOCK] historical C3 Base trajectory: cfg-aware RandomGenerator, "
            "workers=2, independent DataLoader generator, post-construction RNG reset, "
            "strict causal parity, Adam 3e-4, cosine eta_min=0."
        )
    if bool(_cfg_get(cfg.M1, "SEMLT_FIXED_BASE_REFINEMENT", False)):
        logger.info(
            "[FIXED_BASE_REFINEMENT] imported Base/PVL is immutable; only m1_pse.* trains. "
            "The correspondence target is stationary across refinement epochs."
        )
    optimizer, scheduler, ema, base_count, pvl_count, m1_count, m2_count, m3_count = _make_optimizer_and_scheduler(model, cfg)
    if bool(_cfg_get(cfg.M1, "V552R4204_BASE_RNG_ISOLATION_ENABLED", False)):
        base_fingerprint, base_fingerprint_count = _v552r4204_base_fingerprint(model)
        logger.info(
            "[V552R4204_BASE_FINGERPRINT] sha256=%s protected_tensors=%d",
            base_fingerprint,
            base_fingerprint_count,
        )
        # Re-seed *after* variant-specific module construction.  Otherwise the
        # extra occupancy head in A1/A2 consumes RNG during initialization and
        # shifts the first Base/PVL dropout masks even when all shared Base
        # weights are identical.  DataLoader order already uses its own seeded
        # Generator, so this makes the model-side stochastic trajectory fair.
        if int(cfg.seed) >= 0:
            set_random_seed(int(cfg.seed))
            logger.info(
                "[V552R4204_BASE_RNG_RESET] seed=%d after model construction",
                int(cfg.seed),
            )
    if (
        (_mhcs(cfg) or _semlt(cfg))
        and bool(_cfg_get(cfg.M1, "MHCS_RNG_ISOLATION", False))
        and int(cfg.seed) >= 0
        and not bool(_cfg_get(cfg.M1, "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False))
    ):
        # DataLoader shuffling has an independent torch.Generator.  Re-seeding
        # here makes Base/PVL/M1 model-side stochasticity independent of how
        # many parameters the M2 architecture instantiated.  R4.6 M2 itself is
        # deterministic (MHCS_M2_DROPOUT=0), so it cannot shift later RNG draws.
        # Use _seed_only rather than set_random_seed: the latter changes
        # cudnn.benchmark and would silently undo TRAIN.DETERMINISTIC=true.
        _seed_only(int(cfg.seed))
        logger.info(
            "[M1 RNG_RESET] seed=%d after model/optimizer construction; "
            "DataLoader order remains on its independent generator and cuDNN flags are preserved",
            int(cfg.seed),
        )

    logger.info(
        "Trainable tensor groups: base=%d, pvl=%d, m1=%d, m2=%d, m3=%d",
        base_count, pvl_count, m1_count, m2_count, m3_count,
    )
    logger.info(
        "Optimizer groups: %s",
        [(group.get("name", "unnamed"), group.get("lr")) for group in optimizer.param_groups],
    )
    if ema is not None:
        logger.info("EMA enabled with decay=%.4f", ema.decay)
    use_sam = bool(_cfg_get(cfg.TRAIN, "USE_SAM", False))
    use_r_drop = bool(_cfg_get(cfg.TRAIN, "USE_RDROP", False))
    if use_sam:
        logger.info(
            "SAM enabled (rho=%.4f, adaptive=%s)",
            float(_cfg_get(cfg.TRAIN, "SAM_RHO", 0.05)),
            str(_cfg_get(cfg.TRAIN, "SAM_ADAPTIVE", False)),
        )
    if use_r_drop:
        logger.info(
            "R-Drop enabled (weight=%.4f, temperature=%.2f)",
            float(_cfg_get(cfg.TRAIN, "RDROP_WEIGHT", 0.1)),
            float(_cfg_get(cfg.TRAIN, "RDROP_KL_TEMPERATURE", 2.0)),
        )
    trainable_names = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    if bool(_cfg_get(cfg.M1, "V552R4201_ROOTFIX_ENABLED", False)):
        # The historical code printed hundreds of tensor names every launch,
        # making dead-owner leaks hard to see.  Clean mode reports ownership
        # counts by default; set V552R4201_VERBOSE_TRAINABLE_LIST=1 only when a
        # full tensor-name audit is explicitly needed.
        clean_owner_tokens = (
            "r47_pixel_encoder.", "r47_slot_queries.",
            "r411_proposal_stem.", "r411_center_head.",
            "r417_location_head.",
        )
        if not bool(_cfg_get(cfg.M1, "V552R4203_ROOTFIX_ENABLED", False)):
            clean_owner_tokens = clean_owner_tokens + (
                "r420_mask_feature_proj.", "r420_dynamic_controller.",
            )
        if bool(_cfg_get(cfg.M1, "V552R4204_ROOTFIX_ENABLED", False)):
            clean_owner_tokens = clean_owner_tokens + (
                "r4204_residual_occupancy_head.",
            )
        if bool(_cfg_get(cfg.M1, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)):
            clean_owner_tokens = clean_owner_tokens + (
                "r4210_overflow_gate_head.",
            )
        clean_owner_names = [
            name for name in trainable_names
            if any(token in name for token in clean_owner_tokens)
        ]
        logger.info(
            "[V552R4201_CLEAN_OWNERS] active_owner_tensors=%d active_owners=%s",
            len(clean_owner_names),
            sorted({token.rstrip('.') for token in clean_owner_tokens if any(token in n for n in clean_owner_names)}),
        )
        if os.environ.get("V552R4201_VERBOSE_TRAINABLE_LIST", "0") == "1":
            logger.info("Parameters to be updated: %s", trainable_names)
        else:
            logger.info(
                "Parameters to be updated: %d tensors (full list suppressed in clean mode; "
                "set V552R4201_VERBOSE_TRAINABLE_LIST=1 to print it)",
                len(trainable_names),
            )
    else:
        logger.info("Parameters to be updated: %s", trainable_names)
    logger.info("Number of trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # AUTO_PARAM_ADAPTER_INIT_BEGIN
    auto_adapter = AutoParamAdapter(cfg, logger=logger)
    # AUTO_PARAM_ADAPTER_INIT_END

    hard_case_memory = None
    if bool(_cfg_get(cfg.TRAIN, "V547_HARD_CASE_MEMORY_ENABLED", False)):
        hard_case_memory = V547HardCaseMemory(
            momentum=float(
                _cfg_get(cfg.TRAIN, "V547_HARD_CASE_MEMORY_MOMENTUM", 0.90)
            ),
            max_boost=float(
                _cfg_get(cfg.TRAIN, "V547_HARD_CASE_MAX_BOOST", 1.0)
            ),
            warmup_observations=int(
                _cfg_get(cfg.TRAIN, "V547_HARD_CASE_WARMUP_OBSERVATIONS", 1)
            ),
        )
        logger.info(
            "[V547_HARD_CASE_MEMORY] enabled momentum=%.3f max_boost=%.3f warmup=%d",
            hard_case_memory.momentum,
            hard_case_memory.max_boost,
            hard_case_memory.warmup_observations,
        )

    resume_path = os.path.join(
        checkpoint_dir,
        f"{run_name}_latest.pth",
    )

    start_epoch = 0
    best_dice = float("-inf")
    best_fusion = float("-inf")
    best_oracle = float("-inf")
    best_selection_value = float("-inf")
    best_selection_tiebreak = float("-inf")
    best_selection_epoch = -1
    best_native_base_dice = float("-inf")
    best_native_base_nsd = float("-inf")
    best_native_base_catastrophic_rate = float("inf")
    # M1-only ablations need their own validation-selected checkpoint. The main
    # system uses its configured deployed resolution (448 for SPARC-HR2.6).
    # This is selection on Val only; Test remains unopened until the explicit
    # launch_v552r4209_m1_test.sh step.
    best_native_m1_dice = float("-inf")
    best_native_m1_nsd = float("-inf")
    best_native_m1_epoch = -1
    best_selection_path = os.path.join(
        checkpoint_dir, f"{run_name}_best_val.pth"
    )
    best_base_selection_path = os.path.join(
        checkpoint_dir, f"{run_name}_best_base_val.pth"
    )
    best_m1_selection_path = os.path.join(
        checkpoint_dir, f"{run_name}_best_m1_native_val.pth"
    )

    if cfg.resume and os.path.isfile(resume_path):
        checkpoint = torch.load(
            resume_path,
            map_location=cfg.MODEL.DEVICE,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(
            checkpoint.get("scheduler", scheduler.state_dict())
        )

        if ema is not None and "ema_shadow" in checkpoint:
            ema.shadow = checkpoint["ema_shadow"]
        if hard_case_memory is not None and "v547_hard_case_memory" in checkpoint:
            hard_case_memory.load_state_dict(checkpoint["v547_hard_case_memory"])
            logger.info(
                "[V547_HARD_CASE_MEMORY] restored %d cases",
                len(hard_case_memory.values),
            )
        if "v548_m2_routing_state" in checkpoint:
            cfg._v548_m2_routing_state = dict(
                checkpoint["v548_m2_routing_state"]
            )
            logger.info(
                "[V548_STABLE_ROUTING] restored EMA magnitude=%s",
                cfg._v548_m2_routing_state.get(
                    "ema_objective_magnitude", "n/a"
                ),
            )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_selection_value = float(
            checkpoint.get("best_selection_value", best_selection_value)
        )
        best_selection_tiebreak = float(
            checkpoint.get("best_selection_tiebreak", best_selection_tiebreak)
        )
        best_selection_epoch = int(
            checkpoint.get("best_selection_epoch", best_selection_epoch)
        )
        best_native_base_dice = float(
            checkpoint.get("best_native_base_dice", best_native_base_dice)
        )
        best_native_base_nsd = float(
            checkpoint.get("best_native_base_nsd", best_native_base_nsd)
        )
        best_native_base_catastrophic_rate = float(
            checkpoint.get(
                "best_native_base_catastrophic_rate",
                best_native_base_catastrophic_rate,
            )
        )
        best_native_m1_dice = float(
            checkpoint.get("best_native_m1_dice", best_native_m1_dice)
        )
        best_native_m1_nsd = float(
            checkpoint.get("best_native_m1_nsd", best_native_m1_nsd)
        )
        best_native_m1_epoch = int(
            checkpoint.get("best_native_m1_epoch", best_native_m1_epoch)
        )

        if use_validation_selection:
            logger.info(
                "Resumed validation-selected run from epoch %d. "
                "Test remains unopened.",
                start_epoch,
            )
        else:
            logger.info(
                "Resumed Train-only run from epoch %d. "
                "Val and Test remain unopened.",
                start_epoch,
            )

    ucfnrt_parity_step1_logged = False
    parity_reference = _load_base_parity_reference(
        os.environ.get("JBT_BASE_PARITY_REFERENCE_LOG", "").strip()
    )
    if parity_reference:
        if jbt_fixed_base_recovery:
            final_reference_epoch = max(parity_reference["epochs"])
            fixed_reference_hash = parity_reference["epochs"][
                final_reference_epoch
            ]
            if parity_init_hash != fixed_reference_hash:
                raise RuntimeError(
                    "[JBT_V636_FIXED_BASE_LOAD_FAIL] loaded Base hash does not "
                    f"equal matched Base epoch {final_reference_epoch}: "
                    f"expected={fixed_reference_hash} actual={parity_init_hash}"
                )
            parity_reference = {
                "init": fixed_reference_hash,
                "step1": fixed_reference_hash,
                "epochs": {
                    epoch_index: fixed_reference_hash
                    for epoch_index in range(
                        1, int(cfg.TRAIN.NUM_EPOCHS) + 1
                    )
                },
            }
            logger.info(
                "[JBT_V636_FIXED_BASE_REFERENCE] matched Base epoch=%d "
                "sha256=%s; init/step1/all epochs must remain identical",
                final_reference_epoch, fixed_reference_hash,
            )
        _assert_reference_hash(parity_reference, "init", parity_init_hash)
        logger.info(
            "[JBT_V633_PARITY_REFERENCE] matched Base init verified; "
            "step1 and every epoch will fail fast on divergence"
        )
    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        cfg._jbtl_current_epoch_1based = int(epoch) + 1
        # JBT-Lite v12: expose normalized training progress to QABR.
        # This removes the brittle BUSI-specific fixed-step handoff used by v11
        # and keeps the exact deployment state inside the selected checkpoint.
        _jbtl12_owner = getattr(model, "module", model)
        _jbtl12_qabr = getattr(_jbtl12_owner, "qabr", None)
        if _jbtl12_qabr is not None and hasattr(_jbtl12_qabr, "set_training_progress"):
            _jbtl12_qabr.set_training_progress(
                int(epoch) + 1, int(cfg.TRAIN.NUM_EPOCHS)
            )
        _jbtl_edge_eff, _jbtl_normal_eff, _jbtl_scale = _jbtl_effective_weights(cfg)
        _hold_epoch = int(_cfg_get(cfg.TRAIN, "RBAL_FULL_WEIGHT_EPOCHS", 20))
        _decay_end_epoch = int(_cfg_get(cfg.TRAIN, "RBAL_DECAY_END_EPOCH", 80))
        _active_epoch = int(_cfg_get(cfg.TRAIN, "RBAL_BUDGET_ACTIVE_EPOCHS", _hold_epoch))
        _milestones = {1, 5, 10, 15, _hold_epoch, _hold_epoch + 1, _active_epoch, _active_epoch + 1, 40, 50, 60, _decay_end_epoch, int(cfg.TRAIN.NUM_EPOCHS)}
        if (epoch + 1) in _milestones or epoch == start_epoch:
            logger.info(
                "[JBTL_V4_AUX_SCHEDULE] epoch=%03d scale=%.6f edge_eff=%.6f normal_eff=%.6f",
                epoch + 1, _jbtl_scale, _jbtl_edge_eff, _jbtl_normal_eff,
            )
        if (
            ema is not None
            and bool(_cfg_get(cfg.M1, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False))
        ):
            switch_epoch = max(
                int(_cfg_get(cfg.M1, "V538_EMA_DECAY_SWITCH_EPOCH", 30)), 0
            )
            early_decay = float(_cfg_get(cfg.M1, "V538_EMA_DECAY_EARLY", 0.990))
            late_decay = float(_cfg_get(cfg.M1, "V538_EMA_DECAY_LATE", 0.997))
            ema.decay = late_decay if epoch >= switch_epoch else early_decay
            if epoch == start_epoch or epoch == switch_epoch:
                logger.info(
                    "[V538_EMA_TEACHER] epoch=%d decay=%.6f switch_epoch=%d",
                    epoch + 1,
                    ema.decay,
                    switch_epoch + 1,
                )
        model.train()
        model.set_epoch(epoch)
        _v536_apply_source_freeze(model, cfg, epoch, logger)
        _v488_keep_frozen_modules_eval(model, cfg)
        candidate_ratio_nominal = _candidate_ratio(cfg, epoch)
        # AUTO_PARAM_ADAPTER_EPOCH_BEGIN
        if auto_adapter is not None and auto_adapter.enabled:
            candidate_ratio_nominal = auto_adapter.on_epoch_begin(
                epoch, model, optimizer, candidate_ratio_nominal
            )
        # AUTO_PARAM_ADAPTER_EPOCH_END
        # During V396 warmup the candidate term has exactly zero weight. Skip
        # its frozen observer/control pipeline entirely instead of spending a
        # full auxiliary forward on tensors that cannot affect gradients.
        force_m2_only_aux = _v519_is_m2_only(cfg)
        compute_m1_this_epoch = bool(
            m1_enabled(cfg) and (
                force_m2_only_aux
                or candidate_ratio_nominal > 0.0
                or bool(_cfg_get(cfg.M1, "M2_TIDE_REPAIR_ENABLED", False))
            )
        )
        if force_m2_only_aux and candidate_ratio_nominal <= 0.0:
            raise RuntimeError(
                "M2-only contract produced a zero candidate ratio. "
                "M2-only stages freeze Base/PVL/M1/M3 and therefore require "
                "a live auxiliary objective from epoch 1. Set "
                "M1.WARMUP_EPOCHS=0 and keep CANDIDATE_LOSS_WEIGHT>0."
            )
        if bool(_cfg_get(cfg.M1, "V507_BASE_FIRST_CURRICULUM", False)):
            m1_start = int(_cfg_get(cfg.M1, "V505_M1_START_EPOCH", 0))
            m2_start = int(_cfg_get(cfg.M1, "V505_M2_START_EPOCH", 0))
            m3_start = int(_cfg_get(cfg.M1, "V505_M3_START_EPOCH", 0))
            if epoch == 0 or epoch in {m1_start, m2_start, m3_start}:
                phase = (
                    "BASE_ONLY" if epoch < m1_start else
                    "BASE_PLUS_M1" if epoch < m2_start else
                    "BASE_PLUS_M1_M2" if epoch < m3_start else
                    "FULL_M1_M2_M3"
                )
                logger.info(
                    "[V507_CURRICULUM] epoch=%d phase=%s compute_aux=%s "
                    "starts=(M1:%d M2:%d M3:%d)",
                    epoch + 1, phase, compute_m1_this_epoch,
                    m1_start + 1, m2_start + 1, m3_start + 1,
                )
        anchor_ratio = _anchor_ratio(cfg) if teacher is not None else 0.0
        sums, epoch_losses = {}, []
        mhcs_grad_diag_counts = {}
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cfg.TRAIN.NUM_EPOCHS}")
        grad_accum_steps = max(
            1, int(_cfg_get(cfg.TRAIN, "GRAD_ACCUMULATION_STEPS", 1))
        )
        if use_sam and grad_accum_steps > 1:
            raise RuntimeError(
                "Gradient accumulation >1 is not supported together with SAM."
            )
        if not use_sam:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(bar):

            if bool(_cfg_get(cfg.M1, "OFFICIAL_BASE_BATCH_RNG_LOCK", False)):
                batch_rng_seed = _official_base_batch_seed(
                    int(cfg.seed), int(epoch), int(batch_index)
                )
                _seed_only(batch_rng_seed)
                if batch_index == 0:
                    logger.info(
                        "[JBT_V633_BASE_BATCH_RNG_LOCK] epoch=%03d seed=%d "
                        "physical Base stochastic stream isolated",
                        epoch + 1, batch_rng_seed,
                    )

            images = batch["image"].to(cfg.MODEL.DEVICE, non_blocking=True)
            images_hr = batch.get("image_hr", None)
            if isinstance(images_hr, torch.Tensor):
                images_hr = images_hr.to(cfg.MODEL.DEVICE, non_blocking=True)
            masks = batch["ground_truth_mask"].to(cfg.MODEL.DEVICE, non_blocking=True)
            masks_hr = batch.get("ground_truth_mask_hr", None)
            if isinstance(masks_hr, torch.Tensor):
                masks_hr = masks_hr.to(cfg.MODEL.DEVICE, non_blocking=True)
            aug_out = _v547_semantic_safe_augment(
                images, masks, cfg, hr_images=images_hr, hr_masks=masks_hr
            )
            if images_hr is not None and masks_hr is not None:
                images, masks, images_hr, masks_hr = aug_out
            elif images_hr is not None:
                images, masks, images_hr = aug_out
            elif masks_hr is not None:
                images, masks, masks_hr = aug_out
            else:
                images, masks = aug_out
            batch_case_names = batch.get(
                "mask_name", batch.get("image_name", None)
            )
            v547_case_weights = None
            if hard_case_memory is not None:
                v547_case_weights = hard_case_memory.weights(
                    batch_case_names,
                    batch_size=int(images.shape[0]),
                    device=images.device,
                    dtype=images.dtype,
                )

            # JBT-v6.3 exact-Base low-memory route.
            #
            # The released Base uses one real B=24 Adam update.  Replacing it by
            # B=4 with six accumulated backwards is NOT Adam-equivalent and was
            # the direct reason that the protected trajectory diverged at step 1.
            # Conversely, expanding the dense four-candidate JBT graph at B=24
            # exceeds a 48-GB card.  This route therefore performs exactly one
            # public-Base forward/backward on the full physical batch, frees that
            # graph, and accumulates the detached JBT objective in small chunks.
            # Both physical Adam optimizers still step exactly once per public
            # batch.  Since M1 observes only no-grad Base evidence, its chunks
            # cannot alter the protected gradient or the outer Base dropout RNG.
            v63_aux_microbatch = int(
                _cfg_get(cfg.M1, "JBT_V63_AUX_MICROBATCH_SIZE", 0)
            ) if m1_enabled(cfg) else 0
            if v63_aux_microbatch > 0 and compute_m1_this_epoch:
                if not bool(_cfg_get(cfg.M1, "JBT_V63_ENABLED", False)):
                    raise RuntimeError("JBT_V63_AUX_MICROBATCH_SIZE requires JBT_V63_ENABLED=true")
                if not isinstance(optimizer, _JBTDualOptimizer):
                    raise RuntimeError("JBT-v6.3 auxiliary microbatching requires the physical dual Adam")
                if grad_accum_steps != 1 or use_sam or use_r_drop:
                    raise RuntimeError("JBT-v6.3 exact-Base route requires accumulation=1, SAM=false, RDrop=false")
                if teacher is not None or float(anchor_ratio) != 0.0:
                    raise RuntimeError("JBT-v6.3 exact-Base route does not support an external anchor teacher")
                if images_hr is not None or masks_hr is not None:
                    raise RuntimeError("JBT-v6.3 exact-Base route currently supports the released 224px protocol only")
                if hard_case_memory is not None or v547_case_weights is not None:
                    raise RuntimeError("JBT-v6.3 exact-Base route requires the released unweighted sampler")

                batch_size = int(images.shape[0])
                if batch_index == 0:
                    logger.info(
                        "[JBT_V63_EXACT_BASE_AUX_MICROBATCH] physical_base_batch=%d "
                        "aux_microbatch=%d one_base_adam_step=true",
                        batch_size, v63_aux_microbatch,
                    )

                # Exact released-Base objective on the full physical batch.
                base_logits, clip_loss, base_aux = model(
                    image=images,
                    text=batch["text_prompt"],
                    target=masks,
                    return_aux=True,
                    compute_m1=False,
                )
                base_loss = calc_base_loss(
                    base_logits, masks, ce_loss, dice_loss, cfg, clip_loss,
                    case_weights=None,
                )
                if not bool(base_loss.requires_grad):
                    raise RuntimeError("JBT-v6.3 full-batch Base objective has no grad_fn")
                base_loss.backward()
                protected_before = _v487_snapshot_protected_grads(model)
                detached_official_logits = base_logits.detach()
                del base_logits, base_aux
                if isinstance(clip_loss, torch.Tensor):
                    clip_loss = clip_loss.detach()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                loss_version = str(
                    _cfg_get(cfg.M1, "LOSS_MODE", _cfg_get(cfg.M1, "M1_LOSS_VERSION", ""))
                ).lower()
                weighted_diag_sums = {}
                proposal_loss_value = 0.0
                effective_ratio_value = 0.0
                text_prompts = batch["text_prompt"]

                # JBT-v6.3.2 strict stochastic isolation.
                #
                # The detached auxiliary pass contains repeated stochastic Base
                # posterior forwards and may also contain dropout in trainable M1
                # heads.  ``forward_geotr_train_mc_detached_base`` isolates the
                # posterior draws, but it cannot isolate randomness consumed by
                # the downstream M1 forward/backward.  If that outer state is not
                # restored, the next official Base batch sees different dropout
                # masks and the protected trajectory diverges even though the M1
                # gradient is exactly zero on Base/PVL.  Capture every public RNG
                # source after the factual Base backward and restore it before the
                # physical optimizers step.  M1 gradients are already materialized,
                # so restoring RNG does not alter the auxiliary update.
                v632_aux_rng_state = _capture_rng_state()

                for micro_start in range(0, batch_size, v63_aux_microbatch):
                    micro_end = min(batch_size, micro_start + v63_aux_microbatch)
                    micro_count = micro_end - micro_start
                    micro_weight = float(micro_count) / float(batch_size)
                    micro_text = text_prompts[micro_start:micro_end]
                    mean_base_logits, micro_aux = model.forward_geotr_train_mc_detached_base(
                        image=images[micro_start:micro_end],
                        text=micro_text,
                        num_samples=max(2, int(_cfg_get(
                            cfg.M1, "JBT_TRAIN_UNCERTAINTY_MC_SAMPLES", 3
                        ))),
                        supervision_masks=masks[micro_start:micro_end],
                    )
                    micro_aux["official_single_base_logits"] = detached_official_logits[
                        micro_start:micro_end
                    ]
                    raw_proposal, micro_diag = _compute_proposal_loss(
                        loss_version,
                        cfg,
                        micro_aux,
                        masks[micro_start:micro_end],
                        mean_base_logits,
                        epoch,
                    )
                    if raw_proposal is None:
                        raw_proposal = mean_base_logits.sum() * 0.0
                    micro_effective_ratio = _effective_candidate_ratio(
                        cfg, candidate_ratio_nominal, base_loss.detach(), raw_proposal
                    )
                    micro_objective = raw_proposal * float(micro_effective_ratio) * micro_weight
                    if bool(micro_objective.requires_grad):
                        micro_objective.backward()
                    _v487_assert_no_proposal_grad_leak(model, protected_before, logger)
                    _v487_restore_protected_grads(model, protected_before)
                    _v487_clear_forbidden_m1_grads(model, cfg)

                    proposal_loss_value += float(micro_objective.detach().cpu())
                    effective_ratio_value += float(micro_effective_ratio) * micro_weight
                    for key, value in micro_diag.items():
                        if value is None:
                            continue
                        try:
                            scalar = float(
                                value.detach().float().mean().cpu()
                                if torch.is_tensor(value) else value
                            )
                        except (TypeError, ValueError):
                            continue
                        weighted_diag_sums[key] = weighted_diag_sums.get(key, 0.0) + micro_weight * scalar
                    del mean_base_logits, micro_aux, raw_proposal, micro_objective
                    gc.collect()

                _restore_rng_state(v632_aux_rng_state)

                signed_owner_fraction = float(weighted_diag_sums.get(
                    "jbt_v6_signed_disp_owner_fraction", 0.0
                ))
                grad_health = _jbt_v6_real_train_grad_health(
                    model, cfg, signed_owner_fraction=signed_owner_fraction
                )
                if grad_health is None:
                    raise RuntimeError("JBT-v6.3 auxiliary optimizer has no owned gradient contract")
                optimizer.step()
                if ucfnrt_base_parity_audit and not ucfnrt_parity_step1_logged:
                    parity_step_hash, parity_step_count = _v552r4204_base_fingerprint(model)
                    _assert_reference_hash(parity_reference, "step1", parity_step_hash)
                    logger.info(
                        "[UCFNRT_BASE_PARITY_STEP1] sha256=%s protected_tensors=%d",
                        parity_step_hash, parity_step_count,
                    )
                    ucfnrt_parity_step1_logged = True
                optimizer.zero_grad(set_to_none=True)

                proposal_loss = base_loss.detach().new_tensor(proposal_loss_value)
                effective_candidate_ratio = effective_ratio_value
                diagnostics = {
                    **{
                        key: base_loss.detach().new_tensor(value)
                        for key, value in weighted_diag_sums.items()
                    },
                    "jbt_v63_exact_base_aux_microbatch": base_loss.detach().new_ones(()),
                    "jbt_v63_aux_microbatch_size": base_loss.detach().new_tensor(
                        float(v63_aux_microbatch)
                    ),
                    "jbt_v63_zero_flow_no_owner_batch": base_loss.detach().new_tensor(
                        1.0 if (
                            signed_owner_fraction <= 0.0
                            and float(grad_health["flow_norm"]) <= 1.0e-12
                        ) else 0.0
                    ),
                    "jbt_v61_real_grad_total_norm": base_loss.detach().new_tensor(
                        grad_health["total_norm"]
                    ),
                    "jbt_v61_real_grad_flow_norm": base_loss.detach().new_tensor(
                        grad_health["flow_norm"]
                    ),
                    "jbt_v61_real_grad_error_norm": base_loss.detach().new_tensor(
                        grad_health["error_norm"]
                    ),
                    "jbt_v61_real_grad_feedback_norm": base_loss.detach().new_tensor(
                        grad_health["feedback_norm"]
                    ),
                    "jbt_v61_real_grad_utility_norm": base_loss.detach().new_tensor(
                        grad_health["utility_norm"]
                    ),
                }
                if batch_index == 0:
                    logger.info(
                        "[JBT_V6_1_REAL_GRAD] epoch=%03d total=%.6g flow=%.6g "
                        "error=%.6g feedback=%.6g utility=%.6g tensors=%d",
                        epoch + 1,
                        grad_health["total_norm"], grad_health["flow_norm"],
                        grad_health["error_norm"], grad_health["feedback_norm"],
                        grad_health["utility_norm"], int(grad_health["tensor_count"]),
                    )
                anchor_loss = base_loss.detach() * 0.0
                anchor_diag = {}
                loss = base_loss.detach() + proposal_loss
                epoch_losses.append(float(loss.cpu()))
                values = {
                    "base_loss": base_loss.detach(),
                    "anchor_loss": anchor_loss,
                    "proposal_loss": proposal_loss,
                    **diagnostics,
                }
                for key, value in values.items():
                    scalar = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(value)
                    sums[key] = sums.get(key, 0.0) + scalar
                bar.set_postfix(
                    loss=f"{float(loss):.4f}",
                    base=f"{float(base_loss.detach()):.4f}",
                    anchor="0.00",
                    cand=f"{effective_candidate_ratio:.4f}",
                )
                continue

            # V27 must train on the same MC-mean Base distribution that is used
            # by validation and Test inference.  The previous V25/V26 guard only
            # enabled MC for legacy M2 modes, silently leaving
            # unified_action_cf_selection on single-sample candidates.
            mc_train_samples = (
                int(_cfg_get(cfg.M1, "M1_TRAIN_NUM_SAMPLES", _cfg_get(cfg.M1, "M2_TRAIN_NUM_SAMPLES", 1)))
                if m1_enabled(cfg) else 1
            )
            m1_candidate_mode = str(
                _cfg_get(cfg.M1, "CANDIDATE_MODE", "")
            ).strip().lower() if m1_enabled(cfg) else ""
            use_mc_train = (
                m1_uses_unified_action_cf(cfg)
                and mc_train_samples > 1
            )
            v4e_enabled_runtime = bool(_cfg_get(cfg.M1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)) if m1_enabled(cfg) else False
            v4f_enabled_runtime = bool(_cfg_get(cfg.M1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)) if m1_enabled(cfg) else False
            v4g_enabled_runtime = bool(_cfg_get(cfg.M1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False)) if m1_enabled(cfg) else False
            posterior_stable_operator_runtime = bool(
                _cfg_get(cfg.M1, "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP", False)
            ) if m1_enabled(cfg) else False
            uc_fnrt_runtime = bool(_cfg_get(cfg.M1, "SEMLT_UC_FNRT", False)) if m1_enabled(cfg) else False
            geotr_detached_mc_runtime = (
                v4e_enabled_runtime or v4f_enabled_runtime or v4g_enabled_runtime
                or posterior_stable_operator_runtime or uc_fnrt_runtime
            )
            geotr_mc_samples = int(_cfg_get(cfg.M1, "GEOTR_TRAIN_POSTERIOR_SAMPLES", 10)) if geotr_detached_mc_runtime else 1
            uc_fnrt_memory_split = bool(
                uc_fnrt_runtime
                and geotr_detached_mc_runtime
                and compute_m1_this_epoch
                and geotr_mc_samples > 1
            )
            base_loss_prebackward_done = False
            ucfnrt_parity_protected_before = None
            if geotr_detached_mc_runtime and compute_m1_this_epoch and geotr_mc_samples > 1:
                # Official Base/PVL remains the original single-forward training
                # problem.  GEOTR alone observes an RNG-isolated detached MC mean,
                # matching mean_then_refine Val/Test without changing Base gradients.
                base_logits, clip_loss, base_aux = model(
                    image=images, text=batch["text_prompt"], target=masks,
                    return_aux=True, compute_m1=False, slr_hr_image=images_hr,
                    sparc_hr_mask=masks_hr,
                )

                if uc_fnrt_memory_split:
                    # UC-FNRT R2 exact two-graph backward.
                    #
                    # Base/PVL and UC-FNRT have disjoint trainable parameters and
                    # the M1 route consumes only detached Base/semantic
                    # conditioners.  Therefore
                    #
                    #   backward(L_base + L_m1); step()
                    #
                    # is gradient-identical to
                    #
                    #   backward(L_base); free Base graph;
                    #   backward(L_m1); step()
                    #
                    # while the latter never keeps the large trainable Base/PVL
                    # graph alive during MC10 + dense normal-ray refinement.
                    if use_sam:
                        raise RuntimeError(
                            "UC-FNRT memory-split route requires TRAIN.USE_SAM=false"
                        )
                    if grad_accum_steps != 1:
                        raise RuntimeError(
                            "UC-FNRT memory-split route currently requires "
                            "GRAD_ACCUMULATION_STEPS=1 to preserve its exact "
                            "single-step optimizer contract"
                        )
                    if teacher is not None or float(anchor_ratio) != 0.0:
                        raise RuntimeError(
                            "UC-FNRT memory-split route requires no external teacher/anchor"
                        )
                    if use_r_drop:
                        raise RuntimeError(
                            "UC-FNRT memory-split route requires TRAIN.USE_RDROP=false"
                        )
                    if bool(_cfg_get(cfg.M1, "M2_TIDE_REPAIR_ENABLED", False)):
                        raise RuntimeError(
                            "UC-FNRT is M1-only; M2_TIDE_REPAIR_ENABLED must be false"
                        )

                    base_loss = calc_base_loss(
                        base_logits,
                        masks,
                        ce_loss,
                        dice_loss,
                        cfg,
                        clip_loss,
                        case_weights=v547_case_weights,
                    )
                    fixed_base_refinement_runtime = bool(
                        _cfg_get(cfg.M1, "SEMLT_FIXED_BASE_REFINEMENT", False)
                    )
                    if fixed_base_refinement_runtime:
                        # The imported semantic host is intentionally immutable.
                        # We still compute Base loss for diagnostics, but it is not
                        # an optimization objective and must not own a backward graph.
                        if ucfnrt_base_parity_audit and epoch == 0 and batch_index == 0:
                            logger.info(
                                "[FIXED_BASE_FORWARD] logits_sha256=%s base_loss=%.12f",
                                _ucfnrt_tensor_sha256(base_logits),
                                float(base_loss.detach().float().cpu()),
                            )
                        base_loss_prebackward_done = True
                    else:
                        if not bool(base_loss.requires_grad):
                            raise RuntimeError(
                                "[UC_FNRT_MEMORY_SPLIT] trainable Base loss has no grad_fn"
                            )
                        if ucfnrt_base_parity_audit and epoch == 0 and batch_index == 0:
                            logger.info(
                                "[UCFNRT_BASE_PARITY_FORWARD] logits_sha256=%s base_loss=%.12f",
                                _ucfnrt_tensor_sha256(base_logits),
                                float(base_loss.detach().float().cpu()),
                            )
                        base_loss.backward()
                        if ucfnrt_base_parity_audit and epoch == 0 and batch_index == 0:
                            ucfnrt_parity_protected_before = _v487_snapshot_protected_grads(model)
                            grad_sha, grad_count = _ucfnrt_protected_grad_fingerprint(model)
                            logger.info(
                                "[UCFNRT_BASE_PARITY_BASE_GRAD] sha256=%s protected_tensors=%d",
                                grad_sha, grad_count,
                            )
                        base_loss_prebackward_done = True

                    # ---------------------------------------------------------
                    # UC-FNRT exact low-memory handoff.
                    #
                    # Base/PVL backward is already complete here. UC-FNRT M1
                    # consumes detached Base/semantic conditioners and its
                    # gradient-isolation contract forbids M1 gradients from
                    # returning to Base/PVL. Therefore keeping the old Base
                    # autograd roots alive cannot contribute any useful
                    # gradient, but can increase the peak before MC10 + dense
                    # semantic refinement.
                    #
                    # Forward values / losses / optimizer step semantics are
                    # unchanged. Only dead graph references and allocator cache
                    # are released earlier.
                    # ---------------------------------------------------------
                    # Base backward has already been executed above.
                    #
                    # base_logits is only a detached conditioner/diagnostic from
                    # this point onward, so releasing its graph is safe.
                    base_logits = base_logits.detach()

                    # IMPORTANT:
                    # Do NOT detach base_loss here.
                    #
                    # The UC-FNRT two-graph route never backpropagates through
                    # base_loss a second time; later it backpropagates only the
                    # M1-side late_objective.  However, the generic
                    # TRAIN_GRAPH_CONTRACT intentionally uses base_loss's
                    # requires_grad state as evidence that the official Base
                    # objective for this mini-batch was a valid trainable
                    # objective.  HRCV can legitimately have a zero/no-grad M1
                    # objective on a batch with zero contour owners, in which
                    # case the correct update is Base-only.
                    #
                    # backward() above has already freed Base saved activations,
                    # so retaining this scalar graph sentinel has negligible
                    # activation-memory cost.
                    if isinstance(clip_loss, torch.Tensor):
                        clip_loss = clip_loss.detach()

                    del base_aux

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if (
                        ucfnrt_base_parity_audit
                        and epoch == 0
                        and batch_index == 0
                    ):
                        logger.info(
                            "[UC_FNRT_LOW_MEMORY_HANDOFF] "
                            "Base graph detached + GC + CUDA cache release "
                            "before detached posterior MC"
                        )

                _, aux = model.forward_geotr_train_mc_detached_base(
                    image=images, text=batch["text_prompt"],
                    num_samples=geotr_mc_samples, supervision_masks=masks, slr_hr_image=images_hr,
                    sparc_hr_mask=masks_hr,
                )
                aux["official_single_base_logits"] = base_logits.detach()
            elif use_mc_train:
                base_logits, clip_loss, aux = model.forward_m1_train_mc(
                    image=images,
                    text=batch["text_prompt"],
                    num_samples=mc_train_samples,
                    supervision_masks=masks,
                    slr_hr_image=images_hr,
                    sparc_hr_mask=masks_hr,
                )
            else:
                base_logits, clip_loss, aux = model(
                    image=images,
                    text=batch["text_prompt"],
                    target=masks,
                    return_aux=True,
                    compute_m1=compute_m1_this_epoch,
                    slr_hr_image=images_hr,
                    sparc_hr_mask=masks_hr,
                )
            if not base_loss_prebackward_done:
                base_loss = calc_base_loss(
                    base_logits,
                    masks,
                    ce_loss,
                    dice_loss,
                    cfg,
                    clip_loss,
                    case_weights=v547_case_weights,
                )
                qabr_aux_loss, qabr_aux_raw = _jbtl14_qabr_deploy_aligned_loss(
                    model,
                    masks,
                    ce_loss,
                    dice_loss,
                    cfg,
                    case_weights=v547_case_weights,
                )
                if qabr_aux_loss is not None:
                    base_loss = base_loss + qabr_aux_loss
                    if aux is None:
                        aux = {}
                    if isinstance(aux, dict):
                        aux["jbtl14_qabr_aux_loss"] = qabr_aux_raw
            edge_iso_objective = None
            edge_iso_weight = 0.0
            edge_iso_scale = 0.0
            if _jbtl_edge_decoder_only_enabled(cfg):
                if m1_enabled(cfg):
                    raise RuntimeError(
                        "[JBTL5_EDGEISO] decoder-only EDGE is defined for M1.ENABLED=false."
                    )
                edge_iso_objective, edge_iso_weight, edge_iso_scale = _jbtl_edge_aux_loss(
                    base_logits, masks, cfg
                )

            if hard_case_memory is not None:
                v547_difficulty = _v547_case_difficulty(base_logits, masks)
                hard_case_memory.update(batch_case_names, v547_difficulty)

            if compute_m1_this_epoch:
                aux = _inject_v484_aux_if_needed(
                    model,
                    cfg,
                    images,
                    base_logits,
                    aux,
                )
            elif aux is None:
                aux = {}

            # V538 one-run lagged residual teacher.  The same model is evaluated
            # under EMA weights with M1 disabled, then immediately restored.
            # Only the detached Base probability is exposed to the component
            # teacher; no teacher gradient enters the student graph.
            if (
                bool(_cfg_get(cfg.M1, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False))
                and not bool(_cfg_get(cfg.M1, "V552R46_ROOTFIX_ENABLED", False))
            ):
                paired_replay = bool(
                    _cfg_get(
                        cfg.M1,
                        "V547_PAIRED_RESIDUAL_REPLAY_ENABLED",
                        False,
                    )
                )
                if paired_replay:
                    paired_anchor = aux.get("v547_refiner_anchor_prob")
                    if not isinstance(paired_anchor, torch.Tensor):
                        raise RuntimeError(
                            "V547 paired replay requires v547_refiner_anchor_prob "
                            "from the V532 pipeline."
                        )
                    aux["v538_teacher_base_prob"] = paired_anchor.detach()
                else:
                    if ema is None:
                        raise RuntimeError(
                            "V538 requires TRAIN.USE_EMA=true for the lagged residual teacher."
                        )
                    was_training = bool(model.training)
                    with torch.no_grad():
                        with ema.apply_ema():
                            model.eval()
                            teacher_base_logits, _, _ = model(
                                image=images,
                                text=batch["text_prompt"],
                                target=masks,
                                return_aux=True,
                                compute_m1=False,
                            )
                    if was_training:
                        model.train()
                        model.set_epoch(epoch)
                        _v488_keep_frozen_modules_eval(model, cfg)
                    aux["v538_teacher_base_prob"] = torch.sigmoid(
                        teacher_base_logits.detach()
                    )

            anchor_loss = base_loss.detach() * 0.0
            anchor_diag = {}
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, _, _ = teacher(image=images, text=batch["text_prompt"], return_aux=True)
                anchor_loss, anchor_diag = _teacher_anchor_loss(base_logits, teacher_logits, cfg)

            proposal_loss = base_loss.detach() * 0.0
            diagnostics = {}
            if compute_m1_this_epoch:
                loss_version = str(
                    _cfg_get(
                        cfg.M1,
                        "LOSS_MODE",
                        _cfg_get(cfg.M1, "M1_LOSS_VERSION", ""),
                    )
                ).lower()
                proposal_loss, diagnostics = _compute_proposal_loss(
                    loss_version, cfg, aux, masks, base_logits, epoch
                )
                if proposal_loss is None:
                    proposal_loss = base_loss.detach() * 0.0

            # ── M2 TIDE-Repair loss ──
            m2_loss = base_loss.detach() * 0.0
            m2_diagnostics = {}
            if m1_enabled(cfg) and bool(_cfg_get(cfg.M1, "M2_TIDE_REPAIR_ENABLED", False)):
                try:
                    scorer = None
                    if hasattr(model, "_get_m2_tide_repair_head"):
                        head = model._get_m2_tide_repair_head()
                        if head is not None:
                            scorer = getattr(head, "candidate_scorer", None)
                    m2_loss, m2_diagnostics = compute_m2_tide_action_loss(
                        cfg, aux["candidates"], masks, aux, epoch=epoch,
                        scorer=scorer,
                    )
                except Exception as e:
                    log_string = (
                        f"M2 TIDE-Repair loss computation failed at epoch {epoch+1}: {e}"
                    )
                    print(log_string)
                    m2_loss = base_loss.detach() * 0.0
                diagnostics.update(m2_diagnostics)

            v490_split = _v490_split_objective(
                cfg,
                diagnostics,
                candidate_ratio_nominal,
                base_loss,
            )
            if v490_split is not None:
                proposal_loss, effective_candidate_ratio = v490_split
            else:
                effective_candidate_ratio = _effective_candidate_ratio(
                    cfg,
                    candidate_ratio_nominal,
                    base_loss,
                    proposal_loss,
                )
                proposal_loss = effective_candidate_ratio * proposal_loss

            _v552r4201_assert_objective_route(
                cfg,
                compute_m1=compute_m1_this_epoch,
                proposal_loss=proposal_loss,
                effective_candidate_ratio=effective_candidate_ratio,
                diagnostics=diagnostics,
                epoch=epoch,
            )

            m2_loss_weight = float(_cfg_get(cfg.M1, "M2_LOSS_WEIGHT", 1.0))
            joint_boundary_training = bool(
                m1_enabled(cfg)
                and _cfg_get(cfg.M1, "GEOTR_M1_JOINT_BASE_INTEGRATION", False)
            )
            base_aux_weight = (
                float(_cfg_get(cfg.M1, "JBT_BASE_AUX_WEIGHT", 0.35))
                if joint_boundary_training else 1.0
            )
            loss = (
                base_aux_weight * base_loss
                + anchor_ratio * anchor_loss
                + proposal_loss
                + m2_loss_weight * m2_loss
            )
            diagnostics["jbt_base_aux_weight"] = base_loss.detach().new_tensor(
                float(base_aux_weight)
            )

            proposal_contribution = proposal_loss.detach().abs()
            aux_to_base_loss_ratio = proposal_contribution / (
                base_loss.detach().abs().clamp_min(1.0e-8)
            )
            diagnostics.update({
                "v471_candidate_ratio_nominal": candidate_ratio_nominal,
                "v471_candidate_ratio_effective": effective_candidate_ratio,
                "v471_aux_to_base_loss_ratio": aux_to_base_loss_ratio,
                "jbt_v6_separate_base_optimizer": base_loss.detach().new_tensor(
                    1.0 if bool(_cfg_get(cfg.M1, "JBT_V6_SEPARATE_BASE_OPTIMIZER", False)) else 0.0
                ),
            })

            # V390: R-Drop KL consistency (Liang et al., NeurIPS 2021)
            if use_r_drop and m1_enabled(cfg) and use_mc_train and mc_train_samples >= 2:
                # Re-run MC forward with different dropout to get consistency pair
                with torch.no_grad():
                    rdrop_logits2, _, _ = model.forward_m1_train_mc(
                        image=images,
                        text=batch["text_prompt"],
                        num_samples=mc_train_samples,
                        supervision_masks=masks,
                        slr_hr_image=images_hr,
                        sparc_hr_mask=masks_hr,
                    )
                rdrop_weight = float(_cfg_get(cfg.TRAIN, "RDROP_WEIGHT", 0.1))
                rdrop_temp = float(_cfg_get(cfg.TRAIN, "RDROP_KL_TEMPERATURE", 2.0))
                rdrop_loss = rdrop_kl_loss(base_logits, rdrop_logits2, temperature=rdrop_temp)
                loss = loss + rdrop_weight * rdrop_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss}")
            if not bool(loss.requires_grad):
                trainable_preview = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ][:20]
                raise RuntimeError(
                    "[TRAIN_GRAPH_CONTRACT] Total loss has no grad_fn. "
                    f"epoch={epoch + 1} m2_only={_v519_is_m2_only(cfg)} "
                    f"candidate_ratio={candidate_ratio_nominal:.6f} "
                    f"compute_aux={compute_m1_this_epoch} "
                    f"base_requires_grad={bool(base_loss.requires_grad)} "
                    f"proposal_requires_grad={bool(proposal_loss.requires_grad)} "
                    f"trainable_preview={trainable_preview}"
                )

            if use_sam:
                # SAM first step: ascent to worst-case neighborhood
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _check_and_clip_grads(model, cfg)
                optimizer.first_step(zero_grad=True)

                # Recompute loss at w + eps for SAM second step
                if use_mc_train:
                    base_logits_sam, clip_loss_sam, aux_sam = model.forward_m1_train_mc(
                        image=images,
                        text=batch["text_prompt"],
                        num_samples=mc_train_samples,
                        supervision_masks=masks,
                        slr_hr_image=images_hr,
                        sparc_hr_mask=masks_hr,
                    )
                else:
                    base_logits_sam, clip_loss_sam, aux_sam = model(
                        image=images,
                        text=batch["text_prompt"],
                        target=masks,
                        return_aux=True,
                        compute_m1=compute_m1_this_epoch,
                        slr_hr_image=images_hr,
                        sparc_hr_mask=masks_hr,
                    )
                base_loss_sam = calc_base_loss(
                    base_logits_sam,
                    masks,
                    ce_loss,
                    dice_loss,
                    cfg,
                    clip_loss_sam,
                    case_weights=v547_case_weights,
                )
                qabr_aux_loss_sam, _ = _jbtl14_qabr_deploy_aligned_loss(
                    model,
                    masks,
                    ce_loss,
                    dice_loss,
                    cfg,
                    case_weights=v547_case_weights,
                )
                if qabr_aux_loss_sam is not None:
                    base_loss_sam = base_loss_sam + qabr_aux_loss_sam

                if compute_m1_this_epoch:
                    aux_sam = _inject_v484_aux_if_needed(
                        model,
                        cfg,
                        images,
                        base_logits_sam,
                        aux_sam,
                    )
                elif aux_sam is None:
                    aux_sam = {}

                proposal_loss_sam = base_loss_sam.detach() * 0.0
                diagnostics_sam = {}
                if compute_m1_this_epoch:
                    loss_version = str(
                        _cfg_get(
                            cfg.M1,
                            "LOSS_MODE",
                            _cfg_get(cfg.M1, "M1_LOSS_VERSION", ""),
                        )
                    ).lower()
                    proposal_loss_sam, diagnostics_sam = _compute_proposal_loss(
                        loss_version, cfg, aux_sam, masks, base_logits_sam, epoch
                    )
                    if proposal_loss_sam is None:
                        proposal_loss_sam = base_loss_sam.detach() * 0.0

                anchor_loss_sam = base_loss_sam.detach() * 0.0
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits_sam, _, _ = teacher(
                            image=images,
                            text=batch["text_prompt"],
                            return_aux=True,
                        )
                    anchor_loss_sam, _ = _teacher_anchor_loss(
                        base_logits_sam,
                        teacher_logits_sam,
                        cfg,
                    )

                # ── M2 TIDE-Repair loss (SAM second step) ──
                m2_loss_sam = base_loss_sam.detach() * 0.0
                if m1_enabled(cfg) and bool(_cfg_get(cfg.M1, "M2_TIDE_REPAIR_ENABLED", False)):
                    try:
                        scorer_sam = None
                        if hasattr(model, "_get_m2_tide_repair_head"):
                            head_sam = model._get_m2_tide_repair_head()
                            if head_sam is not None:
                                scorer_sam = getattr(head_sam, "candidate_scorer", None)
                        m2_loss_sam, _ = compute_m2_tide_action_loss(
                            cfg, aux_sam["candidates"], masks, aux_sam, epoch=epoch,
                            scorer=scorer_sam,
                        )
                    except Exception:
                        m2_loss_sam = base_loss_sam.detach() * 0.0

                v490_split_sam = _v490_split_objective(
                    cfg,
                    diagnostics_sam,
                    candidate_ratio_nominal,
                    base_loss_sam,
                )
                if v490_split_sam is not None:
                    proposal_loss_sam, effective_candidate_ratio_sam = v490_split_sam
                else:
                    effective_candidate_ratio_sam = _effective_candidate_ratio(
                        cfg,
                        candidate_ratio_nominal,
                        base_loss_sam,
                        proposal_loss_sam,
                    )
                    proposal_loss_sam = (
                        effective_candidate_ratio_sam * proposal_loss_sam
                    )
                _v552r4201_assert_objective_route(
                    cfg,
                    compute_m1=compute_m1_this_epoch,
                    proposal_loss=proposal_loss_sam,
                    effective_candidate_ratio=effective_candidate_ratio_sam,
                    diagnostics=diagnostics_sam,
                    epoch=epoch,
                )
                joint_boundary_training_sam = bool(
                    m1_enabled(cfg)
                    and _cfg_get(cfg.M1, "GEOTR_M1_JOINT_BASE_INTEGRATION", False)
                )
                base_aux_weight_sam = (
                    float(_cfg_get(cfg.M1, "JBT_BASE_AUX_WEIGHT", 0.35))
                    if joint_boundary_training_sam else 1.0
                )
                loss_sam = (
                    base_aux_weight_sam * base_loss_sam
                    + anchor_ratio * anchor_loss_sam
                    + proposal_loss_sam
                    + m2_loss_weight * m2_loss_sam
                )
                loss_sam.backward()
                _check_and_clip_grads(model, cfg)
                optimizer.second_step(zero_grad=True)
            else:
                should_step = (
                    (batch_index + 1) % grad_accum_steps == 0
                    or (batch_index + 1) == len(train_dataloader)
                )

                use_v481_pcgrad = bool(
                    m1_enabled(cfg)
                    and bool(_cfg_get(cfg.M1, "CEM_V481_PCGRAD", False))
                    and float(effective_candidate_ratio) > 0.0
                    and isinstance(proposal_loss, torch.Tensor)
                    and bool(proposal_loss.requires_grad)
                )

                if _is_v487_base_safe_e2e(cfg):
                    if use_sam:
                        raise RuntimeError(
                            "V487_BASE_SAFE_E2E does not support SAM; "
                            "set TRAIN.USE_SAM=false."
                        )

                    # V519/V487 exact gradient isolation with accumulation:
                    #
                    #   1. Scale both objectives by the same accumulation factor.
                    #   2. Accumulate Base/PVL gradients from base_objective.
                    #   3. Snapshot the cumulative protected gradients.
                    #   4. Backpropagate the M1/M2 auxiliary objective.
                    #   5. Assert that the auxiliary objective did not alter any
                    #      protected Base/PVL gradient, then restore the snapshot.
                    #
                    # This preserves the requested effective batch size while
                    # retaining the exact V487 no-proposal-gradient-to-Base
                    # contract. M1/M2 gradients remain accumulated normally.
                    accumulation_scale = 1.0 / float(max(int(grad_accum_steps), 1))
                    base_objective = (
                        base_loss + anchor_ratio * anchor_loss
                    ) * accumulation_scale
                    aux_objective = proposal_loss * accumulation_scale

                    aux_requires_grad = bool(
                        isinstance(aux_objective, torch.Tensor)
                        and bool(aux_objective.requires_grad)
                    )
                    base_objective.backward(retain_graph=aux_requires_grad)
                    protected_before = _v487_snapshot_protected_grads(model)

                    if aux_requires_grad:
                        aux_objective.backward()
                        _v487_assert_no_proposal_grad_leak(
                            model,
                            protected_before,
                            logger,
                        )
                        _v487_restore_protected_grads(
                            model,
                            protected_before,
                        )
                        _v487_clear_forbidden_m1_grads(model, cfg)
                        if _jbt_v6_owns_m1_pse_grads(cfg) and batch_index == 0:
                            grad_health = _jbt_v6_real_train_grad_health(model, cfg)
                            if grad_health is not None:
                                diagnostics.update({
                                    "jbt_v61_real_grad_tensor_count": base_loss.detach().new_tensor(
                                        grad_health["tensor_count"]
                                    ),
                                    "jbt_v61_real_grad_total_norm": base_loss.detach().new_tensor(
                                        grad_health["total_norm"]
                                    ),
                                    "jbt_v61_real_grad_flow_norm": base_loss.detach().new_tensor(
                                        grad_health["flow_norm"]
                                    ),
                                    "jbt_v61_real_grad_error_norm": base_loss.detach().new_tensor(
                                        grad_health["error_norm"]
                                    ),
                                    "jbt_v61_real_grad_feedback_norm": base_loss.detach().new_tensor(
                                        grad_health["feedback_norm"]
                                    ),
                                    "jbt_v61_real_grad_utility_norm": base_loss.detach().new_tensor(
                                        grad_health["utility_norm"]
                                    ),
                                })
                                logger.info(
                                    "[JBT_V6_1_REAL_GRAD] epoch=%03d total=%.6g flow=%.6g "
                                    "error=%.6g feedback=%.6g utility=%.6g tensors=%d",
                                    epoch + 1,
                                    grad_health["total_norm"],
                                    grad_health["flow_norm"],
                                    grad_health["error_norm"],
                                    grad_health["feedback_norm"],
                                    grad_health["utility_norm"],
                                    int(grad_health["tensor_count"]),
                                )
                    else:
                        diagnostics["v487_aux_objective_requires_grad"] = (
                            base_loss.new_tensor(0.0)
                        )
                elif use_v481_pcgrad:
                    aux_objective = proposal_loss
                    base_objective = loss - aux_objective
                    pcgrad_diag = _v481_pcgrad_backward(
                        model,
                        base_objective,
                        aux_objective,
                        grad_accum_steps,
                    )
                    diagnostics.update({
                        key: base_loss.new_tensor(value)
                        for key, value in pcgrad_diag.items()
                    })
                else:
                    if uc_fnrt_memory_split and base_loss_prebackward_done:
                        # Base/PVL gradients are already accumulated and its
                        # graph has been freed before MC10.  Backpropagate only
                        # the detached M1-side objective, then perform the same
                        # single optimizer step below.
                        late_objective = (
                            anchor_ratio * anchor_loss
                            + proposal_loss
                            + m2_loss_weight * m2_loss
                        )
                        late_requires_grad = bool(
                            isinstance(late_objective, torch.Tensor)
                            and bool(late_objective.requires_grad)
                        )
                        if not late_requires_grad:
                            # HRCV is intentionally sparse: it constructs its
                            # cost volume only on Base contour owners.  Early in
                            # joint E2E training a complete mini-batch can have
                            # no predicted Base contour at all.  In that case
                            # HRCV has no physically defined action and the exact
                            # M1 objective is a legitimate constant zero.  This
                            # is NOT a graph-wiring failure: the scientifically
                            # correct update is Base-only for that mini-batch.
                            #
                            # Keep the fail-closed contract for every other
                            # no-grad state.  We waive it only when the actual
                            # HRCV contour-owner diagnostic proves that the
                            # physical action domain is empty.
                            owner_value = diagnostics.get(
                                "geotr_m1_contour_owner_fraction", None
                            )
                            if isinstance(owner_value, torch.Tensor):
                                owner_fraction = float(
                                    owner_value.detach().float().mean().cpu()
                                )
                            elif owner_value is None:
                                owner_fraction = float("nan")
                            else:
                                owner_fraction = float(owner_value)
                            legitimate_null_hrcv_batch = bool(
                                str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower()
                                == "semlt_autozero"
                                and bool(_cfg_get(cfg.M1, "SEMLT_UC_HRCV", False))
                                and math.isfinite(owner_fraction)
                                and owner_fraction <= 0.0
                            )
                            if not legitimate_null_hrcv_batch:
                                raise RuntimeError(
                                    "[UC_FNRT_MEMORY_SPLIT] M1 objective has no grad_fn "
                                    f"outside the legitimate empty-owner HRCV case; "
                                    f"owner_fraction={owner_fraction!r}"
                                )
                            diagnostics[
                                "geotr_m1_memory_split_null_owner_batch"
                            ] = base_loss.detach().new_ones(())
                            diagnostics[
                                "geotr_m1_memory_split_backward"
                            ] = base_loss.detach().new_zeros(())
                        else:
                            hard_m1_isolation = bool(
                                _cfg_get(
                                    cfg.M1,
                                    "SEMLT_HARD_M1_GRAD_ISOLATION",
                                    False,
                                )
                            )
                            if hard_m1_isolation:
                                m1_grad_tensors, m1_grad_norm = (
                                    backward_named_prefix_only(
                                        model,
                                        late_objective,
                                        prefix="m1_pse.",
                                    )
                                )
                                diagnostics[
                                    "geotr_m1_hard_grad_isolation"
                                ] = base_loss.detach().new_ones(())
                                diagnostics[
                                    "geotr_m1_hard_grad_tensor_count"
                                ] = base_loss.detach().new_tensor(
                                    float(m1_grad_tensors)
                                )
                                diagnostics[
                                    "geotr_m1_hard_grad_norm"
                                ] = m1_grad_norm.detach().to(base_loss)
                            else:
                                late_objective.backward()
                            if (
                                ucfnrt_base_parity_audit
                                and epoch == 0
                                and batch_index == 0
                                and ucfnrt_parity_protected_before is not None
                            ):
                                _ucfnrt_assert_protected_grads_unchanged(
                                    model, ucfnrt_parity_protected_before, logger
                                )
                                logger.info(
                                    "[UCFNRT_BASE_PARITY_M1_GRAD_ISOLATION] PASS max_delta<=1e-10"
                                )
                            diagnostics[
                                "geotr_m1_memory_split_null_owner_batch"
                            ] = base_loss.detach().new_zeros(())
                            diagnostics["geotr_m1_memory_split_backward"] = (
                                base_loss.detach().new_ones(())
                            )
                    else:
                        if edge_iso_objective is not None and bool(edge_iso_objective.requires_grad):
                            if use_sam or use_r_drop:
                                raise RuntimeError(
                                    "[JBTL5_EDGEISO] decoder-only EDGE requires SAM=false and RDrop=false."
                                )
                            decoder_params, decoder_names = _jbtl_decoder_params(model)
                            edge_grads = torch.autograd.grad(
                                edge_iso_objective / float(grad_accum_steps),
                                decoder_params,
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True,
                            )
                            (loss / float(grad_accum_steps)).backward()
                            edge_grad_sq = 0.0
                            edge_grad_tensors = 0
                            for param, grad in zip(decoder_params, edge_grads):
                                if grad is None:
                                    continue
                                g = grad.detach()
                                edge_grad_sq += float(g.float().pow(2).sum().cpu())
                                edge_grad_tensors += 1
                                if param.grad is None:
                                    param.grad = g.clone()
                                else:
                                    param.grad.add_(g)
                            diagnostics["jbtl5_edgeiso_weight"] = base_loss.detach().new_tensor(float(edge_iso_weight))
                            diagnostics["jbtl5_edgeiso_scale"] = base_loss.detach().new_tensor(float(edge_iso_scale))
                            diagnostics["jbtl5_edgeiso_grad_norm"] = base_loss.detach().new_tensor(edge_grad_sq ** 0.5)
                            diagnostics["jbtl5_edgeiso_grad_tensors"] = base_loss.detach().new_tensor(float(edge_grad_tensors))
                            if batch_index == 0:
                                logger.info(
                                    "[JBTL5_EDGEISO] epoch=%03d weight=%.6f scale=%.4f decoder_grad_norm=%.6g tensors=%d scope=mask_head+upscale",
                                    epoch + 1, float(edge_iso_weight), float(edge_iso_scale), edge_grad_sq ** 0.5, edge_grad_tensors,
                                )
                        else:
                            (loss / float(grad_accum_steps)).backward()

                if bool(_cfg_get(cfg.M1, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False)):
                    diagnostics.update(
                        _v538_component_grad_diagnostics(model, base_loss)
                    )

                if should_step:
                    grad_clip_diag = _check_and_clip_grads(model, cfg)
                    if isinstance(grad_clip_diag, dict):
                        diagnostics.update(grad_clip_diag)
                    if bool(_cfg_get(
                        cfg.M1, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False
                    )):
                        postclip = _v538_component_grad_diagnostics(
                            model, base_loss
                        )
                        diagnostics.update({
                            "v538_m1_grad_norm_postclip": postclip[
                                "v538_m1_grad_norm"
                            ],
                            "v538_m2_grad_norm_postclip": postclip[
                                "v538_m2_grad_norm"
                            ],
                        })
                    optimizer.step()
                    if ucfnrt_base_parity_audit and not ucfnrt_parity_step1_logged:
                        parity_step_hash, parity_step_count = _v552r4204_base_fingerprint(model)
                        _assert_reference_hash(parity_reference, "step1", parity_step_hash)
                        if bool(_cfg_get(cfg.M1, "SEMLT_FIXED_BASE_REFINEMENT", False)):
                            logger.info(
                                "[FIXED_BASE_STEP1] sha256=%s protected_tensors=%d (must equal init)",
                                parity_step_hash, parity_step_count,
                            )
                        else:
                            logger.info(
                                "[UCFNRT_BASE_PARITY_STEP1] sha256=%s protected_tensors=%d",
                                parity_step_hash, parity_step_count,
                            )
                        ucfnrt_parity_step1_logged = True
                    optimizer.zero_grad(set_to_none=True)

            # V390: update EMA only after an optimizer step.
            if ema is not None and (
                use_sam
                or (batch_index + 1) % grad_accum_steps == 0
                or (batch_index + 1) == len(train_dataloader)
            ):
                ema.update()

            epoch_losses.append(float(loss.detach().cpu()))
            values = {
                "base_loss": base_loss.detach(),
                "anchor_loss": anchor_loss.detach(),
                "proposal_loss": proposal_loss.detach(),
                **anchor_diag,
                **diagnostics,
            }
            for key, value in values.items():
                if value is None:
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 0:
                        continue
                    scalar = float(value.detach().float().mean().cpu())
                else:
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        continue
                sums[key] = sums.get(key, 0.0) + scalar
                if _mhcs(cfg) and str(key).startswith("mhcs_grad_"):
                    mhcs_grad_diag_counts[key] = mhcs_grad_diag_counts.get(key, 0) + 1
            bar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                base=f"{float(base_loss.detach()):.4f}",
                anchor=f"{anchor_ratio:.2f}",
                cand=f"{effective_candidate_ratio:.4f}",
            )

        scheduler.step()

        _replay_after_selection = bool(
            _cfg_get(cfg.TRAIN, "REPLAY_PUBLIC_REPO_VALIDATION_RNG_AFTER_SELECTION", False)
        )
        if (
            replay_public_val_rng
            and val_dataloader is not None
            and (not use_validation_selection or _replay_after_selection)
        ):
            # Validation-based checkpoint selection snapshots/restores the RNG
            # state.  For formal MedCLIPSeg parity we then replay the exact
            # released validation RNG consumption so the next training epoch
            # follows the same stochastic trajectory as the public recipe.
            replay_public_repo_validation_rng(
                model, val_dataloader, cfg.MODEL.DEVICE
            )
            logger.info(
                "[PUBLIC_REPO_RNG_REPLAY] epoch=%03d validation RNG trajectory replayed; "
                "checkpoint_policy=%s.",
                epoch + 1,
                "best_validation" if use_validation_selection else "physical_last_epoch",
            )

        # HRCV fairness audit: initialization/first-step parity is necessary but
        # not sufficient for a strict architectural causal comparison.  Record
        # the complete protected Base/PVL parameter fingerprint after every
        # physical epoch.  The H0/H1 collector compares the two trajectories;
        # this is logging-only and never changes gradients, RNG, or optimizer.
        if ucfnrt_base_parity_audit:
            parity_epoch_hash, parity_epoch_count = _v552r4204_base_fingerprint(model)
            _assert_reference_hash(
                parity_reference, "epoch", parity_epoch_hash, epoch=epoch + 1
            )
            logger.info(
                "[UCFNRT_BASE_PARITY_EPOCH_END] epoch=%03d sha256=%s protected_tensors=%d",
                epoch + 1, parity_epoch_hash, parity_epoch_count,
            )

        means = {
            key: value / max(
                1,
                mhcs_grad_diag_counts.get(key, len(epoch_losses))
                if (_mhcs(cfg) and str(key).startswith("mhcs_grad_"))
                else len(epoch_losses),
            )
            for key, value in sums.items()
        }
        # A mean of per-batch ratios is undefined for batches containing no
        # correctable pixels and produced the misleading value 1439.08 even
        # though the epoch-level fractions were 0.003624 / 0.032224 = 0.1125.
        # Reconstruct the diagnostic from the aggregated epoch fractions. This
        # changes logging only; neither loss nor gradients are affected.
        if _semlt(cfg):
            predicted_edit = float(
                means.get("geotr_m1_predicted_edit_fraction", 0.0)
            )
            target_edit = float(
                means.get("geotr_m1_target_edit_fraction", 0.0)
            )
            means["geotr_m1_overedit_ratio"] = (
                predicted_edit / target_edit if target_edit > 0.0 else 0.0
            )
        if bool(_cfg_get(cfg.M1, "V552R4201_ROOTFIX_ENABLED", False)):
            m1_route_scale = float(means.get("v538_m1_train_scale", 0.0))
            if compute_m1_this_epoch and m1_route_scale > 0.0:
                route_live = float(means.get("v552r4201_objective_route_live", 0.0))
                proposal_mean = float(means.get("proposal_loss", 0.0))
                m1_grad_mean = float(means.get("v538_m1_grad_norm", 0.0))
                if route_live < 0.5 or proposal_mean <= 0.0 or m1_grad_mean <= 0.0:
                    raise RuntimeError(
                        "[V552R4201_EPOCH_ROUTE_CONTRACT] M1 was scheduled as live but "
                        "did not receive a usable training route: "
                        f"epoch={epoch + 1} route_live={route_live:.6f} "
                        f"proposal_loss={proposal_mean:.8f} "
                        f"m1_grad_norm={m1_grad_mean:.8f}. "
                        "Stop before validation/formal training; do not loosen gates."
                    )
        r4212_independent_epoch = bool(
            _cfg_get(cfg.M1, "V552R4212_ROOTFIX_ENABLED", False)
        ) and bool(_cfg_get(cfg.M1, "V552R4212_INDEPENDENT_CANDIDATE_SET_ENABLED", False))
        if bool(_cfg_get(cfg.M1, "V552R4203_ROOTFIX_ENABLED", False)) and not r4212_independent_epoch:
            dense_active = float(means.get("v552r4203_dense_competitive_set_enabled", 0.0))
            point_used = float(means.get("v552r4203_point_bottleneck_used", 1.0))
            ownership_error = float(means.get("v552r4203_ownership_sum_error", 1.0))
            if dense_active < 0.5 or abs(point_used) > 1.0e-8 or ownership_error > 1.0e-4:
                raise RuntimeError(
                    "[V552R4203_EPOCH_SET_CONTRACT] dense competitive residual-set "
                    "ownership was not respected: "
                    f"epoch={epoch + 1} dense_active={dense_active:.6f} "
                    f"point_bottleneck_used={point_used:.6f} "
                    f"ownership_sum_error={ownership_error:.8f}. "
                    "Stop before validation/formal training."
                )
        if bool(_cfg_get(cfg.M1, "V552R4204_ROOTFIX_ENABLED", False)) and not r4212_independent_epoch:
            r4204_active = float(means.get("v552r4204_rootfix_enabled", 0.0))
            location_as_occupancy = float(means.get("v552r4204_location_as_occupancy_used", 1.0))
            mass_error = float(means.get("v552r4204_residual_mass_conservation_error", 1.0))
            occupancy_loss = float(means.get("v552r4204_occupancy_loss", 0.0))
            occupancy_grad = float(means.get("v552r4204_occupancy_head_grad_norm", 0.0))
            occupancy_target_mae = float(
                means.get("v552r4204_occupancy_target_teacher_error_mae", 1.0)
            )
            if (
                r4204_active < 0.5
                or abs(location_as_occupancy) > 1.0e-8
                or mass_error > 1.0e-4
                or occupancy_target_mae > 1.0e-8
                or occupancy_loss <= 0.0
                or occupancy_grad <= 0.0
            ):
                raise RuntimeError(
                    "[V552R4204_EPOCH_FACTORIZATION_CONTRACT] residual existence/identity "
                    "factorisation was not trainable or mass-conserving: "
                    f"epoch={epoch + 1} active={r4204_active:.6f} "
                    f"location_as_occupancy={location_as_occupancy:.6f} "
                    f"mass_error={mass_error:.8f} occupancy_target_mae={occupancy_target_mae:.8f} "
                    f"occupancy_loss={occupancy_loss:.8f} occupancy_head_grad={occupancy_grad:.8f}. "
                    "Stop before validation/formal training; do not loosen deployment gates."
                )
        if bool(_cfg_get(cfg.M1, "V552R4205_ROOTFIX_ENABLED", False)) and not r4212_independent_epoch:
            r4205_active = float(means.get("v552r4205_rootfix_enabled", 0.0))
            target_decomp = float(means.get("v552r4205_target_decomposition_error", 1.0))
            pred_decomp = float(means.get("v552r4205_prediction_decomposition_error", 1.0))
            finite_fraction = float(means.get("v552r4205_final_logits_finite_fraction", 0.0))
            spatial_identity = float(means.get("v552r4204_spatial_identity_enabled", 1.0))
            if (
                r4205_active < 0.5
                or target_decomp > 1.0e-8
                or pred_decomp > 1.0e-4
                or finite_fraction < 1.0 - 1.0e-8
                or abs(spatial_identity) > 1.0e-8
            ):
                raise RuntimeError(
                    "[V552R4205_EPOCH_CAPACITY_CONTRACT] overflow factorisation is not "
                    "target-consistent, probability-conserving, finite, or spatially isolated: "
                    f"epoch={epoch + 1} active={r4205_active:.6f} "
                    f"target_decomp={target_decomp:.8f} pred_decomp={pred_decomp:.8f} "
                    f"finite_fraction={finite_fraction:.8f} spatial_identity={spatial_identity:.6f}. "
                    "Stop before validation/formal training; do not tune M2 gates."
                )
        if bool(_cfg_get(cfg.M1, "V552R4206_ROOTFIX_ENABLED", False)):
            r4206_active = float(means.get("v552r4206_rootfix_enabled", 0.0))
            conditional_loss = float(means.get("v552r4206_conditional_identity_loss", 0.0))
            target_decomp = float(means.get("v552r4206_target_decomposition_error", 1.0))
            overlap_rate = float(means.get("v552r4206_teacher_overlap_rate", 1.0))
            supervised_fraction = float(means.get("v552r4206_supervised_residual_fraction", 0.0))
            unsupervised_pixels = float(
                means.get("v552r4206_unsupervised_residual_pixels", 1.0)
            )
            empty_residual_batch = float(
                means.get("v552r4206_empty_residual_batch", 0.0)
            )
            legacy_bce_active = float(
                means.get("v552r4206_legacy_independent_mask_bce_active", 1.0)
            )
            if (
                r4206_active < 0.5
                or conditional_loss <= 0.0
                or target_decomp > 1.0e-8
                or overlap_rate > 1.0e-8
                or supervised_fraction < 1.0 - 1.0e-8
                or unsupervised_pixels > 1.0e-8
                or abs(legacy_bce_active) > 1.0e-8
            ):
                raise RuntimeError(
                    "[V552R4206_EPOCH_CONDITIONAL_IDENTITY_CONTRACT] categorical identity "
                    "supervision is not live, exact, disjoint, or fully residual-scoped: "
                    f"epoch={epoch + 1} active={r4206_active:.6f} "
                    f"ce={conditional_loss:.8f} target_decomp={target_decomp:.8f} "
                    f"overlap={overlap_rate:.8f} supervised={supervised_fraction:.8f} "
                    f"unsupervised_pixels={unsupervised_pixels:.8f} "
                    f"empty_residual_batch_fraction={empty_residual_batch:.8f} "
                    f"legacy_bce_active={legacy_bce_active:.8f}. "
                    "Stop before validation/formal training; do not compensate with loss weights or M2 gates."
                )
        if bool(_cfg_get(cfg.M1, "V552R4207_ROOTFIX_ENABLED", False)):
            r4207_active = float(means.get("v552r4207_rootfix_enabled", 0.0))
            seed_finite = float(means.get("v552r4207_seed_feature_finite_fraction", 0.0))
            full_image = float(means.get("v552r4207_full_image_assignment_enabled", 0.0))
            hard_support = float(means.get("v552r4207_hard_spatial_support_used", 1.0))
            r4206_active = float(means.get("v552r4206_rootfix_enabled", 0.0))
            q0_cos = float(means.get("v552r4207_q0_pairwise_cosine", 0.0))
            q1_cos = float(means.get("v552r4207_q1_pairwise_cosine", 0.0))
            if (
                r4207_active < 0.5
                or seed_finite < 1.0 - 1.0e-8
                or full_image < 1.0 - 1.0e-8
                or abs(hard_support) > 1.0e-8
                or abs(r4206_active) > 1.0e-8
                or not (-1.000001 <= q0_cos <= 1.000001)
                or not (-1.000001 <= q1_cos <= 1.000001)
            ):
                raise RuntimeError(
                    "[V552R4207_EPOCH_DYNAMIC_VISUAL_BINDING_CONTRACT] visual instance binding "
                    "is not live, finite, full-image, or causally isolated: "
                    f"epoch={epoch + 1} active={r4207_active:.6f} "
                    f"seed_finite={seed_finite:.8f} full_image={full_image:.6f} "
                    f"hard_support={hard_support:.6f} r4206_active={r4206_active:.6f} "
                    f"q0_cos={q0_cos:.6f} q1_cos={q1_cos:.6f}. "
                    "Stop before formal training; do not compensate with loss weights, spatial radii, or M2 gates."
                )

        if bool(_cfg_get(cfg.M1, "V552R4208_ROOTFIX_ENABLED", False)):
            r4208_active = float(means.get("v552r4208_rootfix_enabled", 0.0))
            normalized = float(means.get("v552r4208_normalized_fusion_enabled", 0.0))
            persistent = float(means.get("v552r4208_persistent_identity_enabled", 0.0))
            matching = float(means.get("v552r4208_seed_consistent_matching_enabled", 0.0))
            expected_persistent = bool(_cfg_get(cfg.M1, "V552R4208_PERSISTENT_IDENTITY_ENABLED", False))
            expected_matching = bool(_cfg_get(cfg.M1, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False))
            variable_seed_r4210 = bool(_cfg_get(cfg.M1, "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED", False))
            valid_seed_count_r4210 = float(means.get("v552r4210_valid_seed_count", 0.0))
            r4207_active = float(means.get("v552r4207_rootfix_enabled", 0.0))
            r4206_active = float(means.get("v552r4206_rootfix_enabled", 0.0))
            seed_finite = float(means.get("v552r4207_seed_feature_finite_fraction", 0.0))
            learned_norm = float(means.get("v552r4208_learned_query_norm", 0.0))
            seed_norm = float(means.get("v552r4208_seed_feature_norm", 0.0))
            norm_ratio = float(means.get("v552r4208_seed_to_learned_norm_ratio", 0.0))
            q0_id = float(means.get("v552r4208_q0_seed_identity_cosine", 0.0))
            q1_id = float(means.get("v552r4208_q1_seed_identity_cosine", 0.0))
            full_image = float(means.get("v552r4207_full_image_assignment_enabled", 0.0))
            hard_support = float(means.get("v552r4207_hard_spatial_support_used", 1.0))
            finite_values = all(
                math.isfinite(x)
                for x in (learned_norm, seed_norm, norm_ratio, q0_id, q1_id)
            )
            if (
                r4208_active < 0.5
                or normalized < 0.5
                or abs(r4207_active) > 1.0e-8
                or abs(r4206_active) > 1.0e-8
                or seed_finite < 1.0 - 1.0e-8
                or ((not variable_seed_r4210 or valid_seed_count_r4210 > 1.0e-8) and learned_norm <= 0.0)
                or ((not variable_seed_r4210 or valid_seed_count_r4210 > 1.0e-8) and seed_norm <= 0.0)
                or not finite_values
                or full_image < 1.0 - 1.0e-8
                or abs(hard_support) > 1.0e-8
                or (persistent >= 0.5) != expected_persistent
                or (matching >= 0.5) != expected_matching
            ):
                raise RuntimeError(
                    "[V552R4208_EPOCH_PERSISTENT_BINDING_CONTRACT] normalized/persistent "
                    "instance binding is not live, finite, isolated, or full-image: "
                    f"epoch={epoch + 1} active={r4208_active:.6f} normalized={normalized:.6f} "
                    f"persistent={persistent:.6f}/{int(expected_persistent)} "
                    f"matching={matching:.6f}/{int(expected_matching)} "
                    f"r4207={r4207_active:.6f} r4206={r4206_active:.6f} "
                    f"learned_norm={learned_norm:.6f} seed_norm={seed_norm:.6f} "
                    f"norm_ratio={norm_ratio:.6f} q0_id={q0_id:.6f} q1_id={q1_id:.6f} "
                    f"full_image={full_image:.6f} hard_support={hard_support:.6f}."
                )

        if bool(_cfg_get(cfg.M1, "V552R4210_ROOTFIX_ENABLED", False)):
            r4210_active = float(means.get("v552r4210_rootfix_enabled", 0.0))
            interior = float(means.get("v552r4210_interior_anchor_enabled", 0.0))
            variable = float(means.get("v552r4210_variable_cardinality_seed_enabled", 0.0))
            independent = float(means.get("v552r4210_independent_overflow_enabled", 0.0))
            alignment = float(means.get("v552r4210_m1_native_alignment_enabled", 0.0))
            expected_interior = bool(_cfg_get(cfg.M1, "V552R4210_INTERIOR_ANCHOR_ENABLED", False))
            expected_variable = bool(_cfg_get(cfg.M1, "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED", False))
            expected_independent = bool(_cfg_get(cfg.M1, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False))
            expected_alignment = bool(_cfg_get(cfg.M1, "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED", False))
            inside = float(means.get("v552r4210_interior_anchor_inside_teacher_rate", 1.0))
            valid_seed_count = float(means.get("v552r4210_valid_seed_count", 0.0))
            num_slots = float(_cfg_get(cfg.M1, "V538_NUM_COMPONENT_SLOTS", 0))
            target_decomp = float(means.get("v552r4205_target_decomposition_error", 0.0))
            pred_decomp = float(means.get("v552r4205_prediction_decomposition_error", 0.0))
            matching = float(means.get("v552r4208_seed_consistent_matching_enabled", 0.0))
            legacy_overflow = float(means.get("v552r4209_balanced_overflow_enabled", 0.0))
            independent_loss = float(means.get("v552r4210_independent_overflow_loss", 0.0))
            native_alignment_loss = float(means.get("v552r4210_m1_native_alignment_loss", 0.0))
            native_alignment_active = float(means.get("v552r4210_m1_native_alignment_active", 0.0))
            contract_values = (
                r4210_active, interior, variable, independent, alignment, inside,
                valid_seed_count, target_decomp, pred_decomp, matching, legacy_overflow,
                independent_loss, native_alignment_loss, native_alignment_active,
            )
            bad = (
                not all(math.isfinite(x) for x in contract_values)
                or r4210_active < 0.5
                or (interior >= 0.5) != expected_interior
                or (variable >= 0.5) != expected_variable
                or (independent >= 0.5) != expected_independent
                or (alignment >= 0.5) != expected_alignment
                or (expected_interior and inside < 1.0 - 1.0e-8)
                or valid_seed_count < -1.0e-8
                or valid_seed_count > num_slots + 1.0e-8
                or ((not r4212_independent_epoch) and abs(target_decomp) > 1.0e-8)
                or ((not r4212_independent_epoch) and abs(pred_decomp) > 1.0e-6)
                or abs(matching) > 1.0e-8
                or abs(legacy_overflow) > 1.0e-8
                or (expected_independent and independent_loss <= 0.0)
                or ((native_alignment_active >= 0.5) != expected_alignment)
                or (expected_alignment and native_alignment_loss <= 0.0)
            )
            if bad:
                raise RuntimeError(
                    "[V552R4210_EPOCH_INSTANCE_VALID_FACTORIZATION_CONTRACT] contract violation: "
                    f"epoch={epoch + 1} active={r4210_active:.3f} "
                    f"interior={interior:.3f}/{int(expected_interior)} inside={inside:.6f} "
                    f"variable={variable:.3f}/{int(expected_variable)} validSeeds={valid_seed_count:.3f}/{num_slots:.0f} "
                    f"independentOverflow={independent:.3f}/{int(expected_independent)} loss={independent_loss:.6f} "
                    f"nativeAlign={alignment:.3f}/{int(expected_alignment)} active={native_alignment_active:.3f} loss={native_alignment_loss:.6f} "
                    f"targetDecomp={target_decomp:.3e} predDecomp={pred_decomp:.3e} "
                    f"hardMatching={matching:.3f} legacyOverflow={legacy_overflow:.3f}."
                )

        if bool(_cfg_get(cfg.M1, "V552R4212_ROOTFIX_ENABLED", False)):
            stage4212 = int(_cfg_get(cfg.M1, "V552R4212_STAGE", 0))
            independent4212 = float(means.get("v552r4212_independent_candidate_set_enabled", 0.0))
            existence4212 = float(means.get("v552r4212_existence_no_object_enabled", 0.0))
            seed_disabled4212 = float(means.get("v552r4212_visual_seed_identity_disabled", 0.0))
            align4212 = float(means.get("v552r4212_candidate_alignment_enabled", 0.0))
            direct4212 = float(means.get("v552r4212_direct_delta_utility_enabled", 0.0))
            stop4212 = float(means.get("v552r4212_zero_stop_one_step_enabled", 0.0))
            target_count4212 = float(means.get("v552r4212_presence_target_count", 0.0))
            deploy_count4212 = float(means.get("v552r4212_deployment_candidate_count", 0.0))
            overlap4212 = float(means.get("v552r4212_independent_soft_overlap_mass", 0.0))
            candidate_loss4212 = float(means.get("v552r4212_candidate_alignment_loss", 0.0))
            utility_mae4212 = float(means.get("v552r4212_direct_delta_utility_mae", 0.0))
            expected_flags4212 = (
                1.0, 1.0, float(stage4212 >= 2), float(stage4212 >= 3),
                float(stage4212 >= 4), float(stage4212 >= 5),
            )
            got_flags4212 = (
                independent4212, existence4212, seed_disabled4212, align4212,
                direct4212, stop4212,
            )
            if (
                not all(math.isfinite(x) for x in got_flags4212 + (
                    target_count4212, deploy_count4212, overlap4212,
                    candidate_loss4212, utility_mae4212,
                ))
                or any((g >= 0.5) != (w >= 0.5) for g, w in zip(got_flags4212, expected_flags4212))
                or target_count4212 < -1.0e-8
                or deploy_count4212 < -1.0e-8
                or overlap4212 < -1.0e-8
                or (stage4212 >= 3 and candidate_loss4212 <= 0.0)
                or (stage4212 >= 4 and utility_mae4212 < 0.0)
            ):
                raise RuntimeError(
                    "[V552R4212_EPOCH_SET_UTILITY_CONTRACT] contract violation: "
                    f"epoch={epoch + 1} stage=E{stage4212} flags={got_flags4212} "
                    f"existTarget={target_count4212:.4f} deployN={deploy_count4212:.4f} "
                    f"overlap={overlap4212:.6f} candLoss={candidate_loss4212:.6f} "
                    f"utilityMAE={utility_mae4212:.6f}."
                )

        if hard_case_memory is not None:
            memory_summary = hard_case_memory.summary()
            means["v547_hard_memory_seen"] = memory_summary["seen"]
            means["v547_hard_memory_mean"] = memory_summary["mean"]
            means["v547_hard_memory_max"] = memory_summary["max"]
            logger.info(
                "[V547_HARD_CASE_MEMORY] seen=%d mean=%.6f max=%.6f",
                int(memory_summary["seen"]),
                memory_summary["mean"],
                memory_summary["max"],
            )

        # V544: ratios are computed from epoch-global counts, never by averaging
        # per-batch ratios.  This removes the empty-class batch bias that made
        # Outcome balanced accuracy and Benefit sign recall mathematically
        # inconsistent in V543B.
        def _global_ratio(numerator_key, denominator_key):
            denominator = float(sums.get(denominator_key, 0.0))
            if denominator <= 0.0:
                return 0.0
            return float(sums.get(numerator_key, 0.0)) / denominator

        v544_benefit_sign = _global_ratio(
            "v544_benefit_gain_positive_count",
            "v544_benefit_total_count",
        )
        v544_harm_sign = _global_ratio(
            "v544_harm_gain_negative_count",
            "v544_harm_total_count",
        )
        v544_neutral_recall = _global_ratio(
            "v544_neutral_outcome_correct_count",
            "v544_neutral_total_count",
        )
        v544_benefit_recall = _global_ratio(
            "v544_benefit_outcome_correct_count",
            "v544_benefit_total_count",
        )
        v544_harm_recall = _global_ratio(
            "v544_harm_outcome_correct_count",
            "v544_harm_total_count",
        )
        means["v544_benefit_gain_positive_rate_global"] = v544_benefit_sign
        means["v544_harm_gain_negative_rate_global"] = v544_harm_sign
        means["v544_balanced_sign_accuracy_global"] = 0.5 * (
            v544_benefit_sign + v544_harm_sign
        )
        means["v544_neutral_outcome_recall_global"] = v544_neutral_recall
        means["v544_benefit_outcome_recall_global"] = v544_benefit_recall
        means["v544_harm_outcome_recall_global"] = v544_harm_recall
        means["v544_outcome_balanced_accuracy_global"] = (
            v544_neutral_recall + v544_benefit_recall + v544_harm_recall
        ) / 3.0
        benefit_count = float(sums.get("v544_benefit_total_count", 0.0))
        nonbenefit_count = float(sums.get("v544_nonbenefit_total_count", 0.0))
        means["v544_true_benefit_separation_global"] = (
            float(sums.get("v544_benefit_probability_sum", 0.0))
            / max(benefit_count, 1.0)
            - float(sums.get("v544_nonbenefit_probability_sum", 0.0))
            / max(nonbenefit_count, 1.0)
        )
        means["v544_signed_outcome_mean_on_benefit_global"] = (
            float(sums.get("v544_signed_outcome_benefit_sum", 0.0))
            / max(benefit_count, 1.0)
        )
        harm_count = float(sums.get("v544_harm_total_count", 0.0))
        means["v544_signed_outcome_mean_on_harm_global"] = (
            float(sums.get("v544_signed_outcome_harm_sum", 0.0))
            / max(harm_count, 1.0)
        )
        means["v544_zero_benefit_batch_rate"] = float(
            sums.get("v544_zero_benefit_batch", 0.0)
        ) / max(1, len(epoch_losses))

        # Replace the legacy batch-mean contract fields with their rigorous
        # epoch-global counterparts while retaining the raw count fields.
        means["v543_balanced_sign_accuracy"] = means[
            "v544_balanced_sign_accuracy_global"
        ]
        means["v543_outcome_balanced_accuracy"] = means[
            "v544_outcome_balanced_accuracy_global"
        ]
        means["v543_true_benefit_separation"] = means[
            "v544_true_benefit_separation_global"
        ]
        means["v542_benefit_gain_positive_rate"] = means[
            "v544_benefit_gain_positive_rate_global"
        ]
        means["v542_harm_gain_negative_rate"] = means[
            "v544_harm_gain_negative_rate_global"
        ]
        means["v549_current_epoch"] = float(epoch + 1)
        means["v549_shadow_execute_count_epoch"] = float(
            sums.get("v549_shadow_execute_count", 0.0)
        )
        means["v549_shadow_improved_count_epoch"] = float(
            sums.get("v549_shadow_improved_count", 0.0)
        )
        means["v549_shadow_harmful_count_epoch"] = float(
            sums.get("v549_shadow_harmful_count", 0.0)
        )
        means["v549_shadow_selected_gain_sum_epoch"] = float(
            sums.get("v549_shadow_selected_gain_sum", 0.0)
        )
        means["v549_shadow_case_count_epoch"] = float(
            sums.get("v549_shadow_case_count", 0.0)
        )
        means["v552r44_audit_execute_count_epoch"] = float(
            sums.get("v552r44_audit_execute_count", 0.0)
        )
        means["v552r44_audit_improved_count_epoch"] = float(
            sums.get("v552r44_audit_improved_count", 0.0)
        )
        means["v552r44_audit_harmful_count_epoch"] = float(
            sums.get("v552r44_audit_harmful_count", 0.0)
        )
        means["v552r44_audit_selected_gain_sum_epoch"] = float(
            sums.get("v552r44_audit_selected_gain_sum", 0.0)
        )
        means["v552r44_audit_case_count_epoch"] = float(
            sums.get("v552r44_audit_case_count", 0.0)
        )
        means["v552r44_policy_audit_execute_count_epoch"] = float(
            sums.get("v552r44_policy_audit_execute_count", 0.0)
        )
        means["v552r44_policy_audit_improved_count_epoch"] = float(
            sums.get("v552r44_policy_audit_improved_count", 0.0)
        )
        means["v552r44_policy_audit_harmful_count_epoch"] = float(
            sums.get("v552r44_policy_audit_harmful_count", 0.0)
        )
        means["v552r44_policy_audit_selected_gain_sum_epoch"] = float(
            sums.get("v552r44_policy_audit_selected_gain_sum", 0.0)
        )
        means["v552r44_policy_audit_case_count_epoch"] = float(
            sums.get("v552r44_policy_audit_case_count", 0.0)
        )
        means["v552r46_formal_policy_audit_execute_count_epoch"] = float(
            sums.get("v552r46_formal_policy_audit_execute_count", 0.0)
        )
        means["v552r46_formal_policy_audit_improved_count_epoch"] = float(
            sums.get("v552r46_formal_policy_audit_improved_count", 0.0)
        )
        means["v552r46_formal_policy_audit_harmful_count_epoch"] = float(
            sums.get("v552r46_formal_policy_audit_harmful_count", 0.0)
        )
        means["v552r46_formal_policy_audit_selected_gain_sum_epoch"] = float(
            sums.get("v552r46_formal_policy_audit_selected_gain_sum", 0.0)
        )
        means["v552r46_formal_policy_audit_case_count_epoch"] = float(
            sums.get("v552r46_formal_policy_audit_case_count", 0.0)
        )
        _v538_update_quality_gate(model, cfg, means, logger=logger)

        lrs = {
            group.get("name", str(i)): group["lr"]
            for i, group in enumerate(optimizer.param_groups)
        }

        logger.info(
            "EPOCH: %03d | TRAIN_LOSS=%.5f BASE_LOSS=%.5f "
            "ANCHOR_LOSS=%.5f PROPOSAL_LOSS=%.5f "
            "anchor_ratio=%.3f cand_nominal=%.4f cand_effective=%.4f "
            "aux/base=%.4f | protocol=%s | lr=%s",
            epoch + 1,
            mean(epoch_losses),
            means.get("base_loss", 0.0),
            means.get("anchor_loss", 0.0),
            means.get("proposal_loss", 0.0),
            anchor_ratio,
            candidate_ratio_nominal,
            means.get("v471_candidate_ratio_effective", 0.0),
            means.get("v471_aux_to_base_loss_ratio", 0.0),
            (
                "JOINT_E2E_VAL_SELECTION"
                if use_validation_selection else "JOINT_E2E_LAST_EPOCH"
            ),
            lrs,
        )

        tracked_keys = [
            "v463_residual_loss",
            "v463_fp_target_area",
            "v463_fn_target_area",
            "v463_candidate_purity_loss",
            "v463_candidate_harm_loss",
            "v463_ccv_loss",
            "v463_ccv_reg_loss",
            "v463_ccv_harm_loss",
            "v463_ccv_pareto_loss",
            "v463_ccv_rank_loss",
            "v463_oracle_delta_dsc",
            "v463_positive_candidate_rate",
            "v463_harmful_candidate_rate",
            "v463_ccv_changed_rate",
            "v467_deploy_margin_loss",
            "v467_deploy_pos_loss",
            "v467_deploy_harm_loss",
            "v467_deploy_case_loss",
            "v467_utility_mean",
            "v467_utility_max",
            "v468_residual_ce_loss",
            "v468_residual_sparse_dice_loss",
            "v468_candidate_loss",
            "v468_candidate_oracle_loss",
            "v468_candidate_direction_loss",
            "v468_candidate_downside_loss",
            "v468_soft_oracle_utility",
            "v468_soft_oracle_delta_dsc",
            "v468_soft_oracle_delta_boundary",
            "v468_candidate_benefit_fraction",
            "v468_candidate_harm_fraction",
            "v468_utility_target_mean",
            "v469_fp_focal_loss",
            "v469_fn_focal_loss",
            "v469_fp_tversky_loss",
            "v469_fn_tversky_loss",
            "v469_fp_precision",
            "v469_fp_recall",
            "v469_fp_f1",
            "v469_fn_precision",
            "v469_fn_recall",
            "v469_fn_f1",
            "v469_candidate_coverage_loss",
            "v469_gate_nonempty_loss",
            "v469_valid_action_rate",
            "v469_ccv_weight",
            "v469_deploy_weight",
            "v469_direct_utility_reg_loss",
            "v469_ccv_inputs_detached_rate",
            "v470_soft_fp_target_area",
            "v470_soft_fn_target_area",
            "v470_ccv_to_base_grad_scale",
            "v470_ccv_to_m1_grad_scale",
            "v470_aux_to_base_grad_scale",
            "v471_candidate_ratio_nominal",
            "v471_candidate_ratio_effective",
            "v471_aux_to_base_loss_ratio",
            "v426_family_oracle_loss",
            "v426_action_quality_loss",
            "v426_action_no_harm_loss",
            "v426_action_advantage_loss",
            "v426_delete_action_best_dice",
            "v426_fill_action_best_dice",
                    "v426_base_dice",
                    "v426_global_action_best_dice",
                    "v426_delete_best_gain_vs_c0",
                    "v426_fill_best_gain_vs_c0",
                    "v426_global_best_gain_vs_c0",
            "v426_delete_residual_case_rate",
            "v426_fill_residual_case_rate",
            "v396_mean_visual_gain",
            "v396_mean_q",
            "v396_positive_fraction",
            "v396_harmful_fraction",
            "v396_oracle_gain",
            "v422_delete_fp_removed",
            "v422_delete_tp_removed",
            "v422_boundary_fill_fn_added",
            "v422_boundary_fill_bg_added",
            "v422_fill_budget_loss",
            "v428_adaptive_loss",
            "v428_c6_mean_delta",
            "unified_m1_loss",
            "unified_m1_choice_loss",
            "unified_m1_no_harm_loss",
            "unified_m1_benefit_loss",
            "unified_m1_area_loss",
            "unified_m1_positive_action_rate",
            "unified_m1_harmful_action_rate",
            "unified_m1_positive_case_rate",
            "unified_m1_hard_preserve_rate",
            "unified_m1_selected_positive_rate",
            "unified_m1_selected_harmful_rate",
            "unified_m1_hard_delta_dice",
            "unified_m1_hard_delta_nsd",
            "unified_m1_hard_fusion_dice",
            "unified_m1_hard_fusion_nsd",
            "unified_m1_expected_area",
            "tpmhg_loss",
            "tpmhg_oracle_loss",
            "tpmhg_bce_loss",
            "tpmhg_role_loss",
            "tpmhg_diversity_loss",
            "tpmhg_area_loss",
            "tpmhg_base_dice",
            "tpmhg_best_dice",
            "tpmhg_oracle_gain",
            "tpmhg_pairwise_l1",
            "tpmhg_positive_candidate_rate",
            "tpmhg_harmful_candidate_rate",
            "tpmhg_positive_case_rate",
            "tpmhg_best_slot_mean",
            "tpmhg_hyp_area",
            "cem_loss",
            "cem_component_count",
            "cem_error_coverage_loss",
            "cem_error_coverage_precision",
            "cem_error_coverage_recall",
            "cem_support_precision",
            "cem_support_precision_loss",
            "cem_correction_outside_ratio",
            "cem_correction_precision_loss",
            "cem_redundancy_loss",
            "cem_soft_oracle_gain",
            "cem_hard_oracle_gain",
            "cem_single_best_gain",
            "cem_composed_best_gain",
            "cem_candidate_harm_fraction",
            "cem_harm_fraction_penalty",
            "cem_candidate_fp_edit_mass",
            "cem_candidate_tp_removal_mass",
            "cem_candidate_edit_mass",
            "cem_candidate_beneficial_mass",
            "cem_utility_reg_loss",
            "cem_utility_mean_loss",
            "cem_sigma_calibration_loss",
            "cem_lcb_calibration_loss",
            "cem_predicted_sigma_mean",
            "cem_predicted_lcb_mean",
            "cem_positive_lcb_pass_rate",
            "cem_harmful_lcb_reject_rate",
            "cem_benefit_loss",
            "cem_benefit_target_rate",
            "cem_benefit_predicted_rate",
            "cem_benefit_precision",
            "cem_benefit_recall",
            "cem_benefit_accuracy",
            "cem_deployable_candidate_rate",
            "cem_best_action_score",
            "cem_best_action_harm",
            "cem_best_action_benefit",
            "cem_harm_reg_loss",
            "cem_candidate_accept_loss",
            "cem_expected_negative_gain_loss",
            "cem_rank_loss",
            "cem_v479_compact_local_loss",
            "cem_v479_gate_loss",
            "cem_v479_leakage_loss",
            "cem_cf_mean_loss",
            "cem_cf_quantile_loss",
            "cem_cf_q10_coverage",
            "v479_xbm_bank_size",
            "cem_cf_pairwise_loss",
            "cem_cf_pairwise_accuracy",
            "cem_cf_null_loss",
            "cem_m3_policy_regret_loss",
            "cem_m3_expected_utility",
            "cem_m3_oracle_utility",
            "cem_final_tail_loss",
            "cem_discovery_coherence_loss",
            "cem_discovery_diversity_loss",
            "cem_discovery_oracle_loss",
            "cem_discovery_oracle_quality",
            "cem_discovery_best_gain",
            "cem_failure_loss",
            "cem_selection_loss",
            "cem_quality_correlation",
            "cem_failure_target_rate",
            "cem_selector_scale",
            "cem_changed_rate",
            "cem_selected_dice",
            "cem_selected_delta_dice",
            "v484_error_state_loss",
            "v484_existence_loss",
            "v484_purity_loss",
            "v484_fp_delete_loss",
            "v484_fn_fill_loss",
            "v484_boundary_loss",
            "v484_global_loss",
            "v484_m2_local_loss",
            "v484_m2_global_loss",
            "v484_m3_regret_loss",
            "v484_base_dice",
            "v484_local_oracle_gain",
            "v484_local_harmful_rate",
            "v484_local_active_rate",
            "v484_global_active_rate",
            "v484_support_precision",
            "v484_correction_outside_ratio",
            "v488_total_loss",
            "v488_m2_policy_loss",
            "v488_m2_effect_loss",
            "v488_m2_evidence_loss",
            "v488_m2_uncertainty_loss",
            "v488_m2_pwo_seg_loss",
            "v488_m2_gt_seg_loss",
            "v488_m2_boundary_loss",
            "v488_m3_gate_loss",
            "v488_m3_seg_loss",
            "v488_m3_boundary_loss",
            "v488_m3_no_harm_loss",
            "v488_m3_rollback_loss",
            "v488_base_dice",
            "v488_global_oracle_dice",
            "v488_pwo_dice",
            "v488_pwo_gap_over_global",
            "v488_m2_dice",
            "v488_m3_final_dice",
            "v488_m2_gain_vs_base",
            "v488_m3_gain_vs_m2",
            "v488_nonbase_weight",
            "v488_m3_gate_rate",
            "v488_correctable_pixel_rate",
            "v485_error_state_loss",
            "v485_support_supervision_loss",
            "v485_delete_support_loss",
            "v485_fill_support_loss",
            "v485_boundary_support_loss",
            "v485_direction_loss",
            "v485_existence_loss",
            "v485_purity_loss",
            "v485_edit_budget_loss",
            "v485_m2_local_loss",
            "v485_base_dice",
            "v485_local_oracle_gain",
            "v485_positive_candidate_rate",
            "v485_harmful_candidate_rate",
            "v485_candidate_mean_abs_change",
            "v485_candidate_max_abs_change",
            "v485_candidate_non_noop_rate",
            "v485_support_precision",
            "v485_correction_outside_ratio",
            "v485_delete_support_iou",
            "v485_fill_support_iou",
            "v485_boundary_support_iou",
            "v485_fp_removed",
            "v485_fn_added",
            "v485_tp_removed",
            "v485_bg_added",
            "v485_local_active_rate",
            "v486_candidate_repair_loss",
            "v486_delete_repair_loss",
            "v486_fill_repair_loss",
            "v486_boundary_repair_loss",
            "v486_outside_preserve_loss",
            "v486_edit_floor_loss",
            "v486_target_exists_rate",
            "v486_mean_target_area",
            "v428_c6_gate_mean",
        ]
        v561_enabled_runtime = bool(_cfg_get(cfg.M1, "V561_BCRS_ENABLED", False))
        if (
            bool(_cfg_get(cfg.M1, "V560_CLEAN_CORE_ENABLED", False))
            and not v561_enabled_runtime
            and epoch == 0
        ):
            factual_residual = float(means.get("v560_factual_residual_fraction", 0.0))
            teacher_count = float(means.get("v538_teacher_component_count", 0.0))
            mask_bce = float(means.get("v560_standard_mask_bce", 0.0))
            mask_dice = float(means.get("v560_standard_mask_dice_loss", 0.0))
            candidate_align = float(means.get("v552r4212_candidate_alignment_loss", 0.0))
            min_teacher_fraction = float(_cfg_get(cfg.M1, "V538_TEACHER_MIN_PIXELS", 4)) / float(224 * 224)
            if factual_residual > min_teacher_fraction and teacher_count <= 0.0:
                raise RuntimeError(
                    "V560 scientific contract failed at epoch 1: factual residuals exist "
                    f"(fraction={factual_residual:.6f}) but TeacherComponentCount=0."
                )
            if teacher_count > 0.0 and (mask_bce <= 0.0 or mask_dice <= 0.0 or candidate_align <= 0.0):
                raise RuntimeError(
                    "V560 scientific contract failed at epoch 1: Teacher components exist but "
                    f"core supervision is inactive (BCE={mask_bce:.6f}, Dice={mask_dice:.6f}, "
                    f"CandidateAlign={candidate_align:.6f})."
                )

        if v561_enabled_runtime and epoch == 0:
            factual_residual = float(means.get("v560_factual_residual_fraction", 0.0))
            teacher_count = float(means.get("v538_teacher_component_count", 0.0))
            mask_bce = float(means.get("v560_standard_mask_bce", 0.0))
            mask_dice = float(means.get("v560_standard_mask_dice_loss", 0.0))
            action_loss = float(means.get("v538_action_loss", 0.0))
            presence_loss = float(means.get("v538_presence_loss", 0.0))
            candidate_align = float(means.get("v552r4212_candidate_alignment_loss", 0.0))
            geometry_owner = float(means.get("v561_geometry_owner", 0.0))
            min_teacher_fraction = float(_cfg_get(cfg.M1, "V538_TEACHER_MIN_PIXELS", 4)) / float(224 * 224)
            if geometry_owner < 0.999:
                raise RuntimeError(
                    "V561 scientific contract failed at epoch 1: BCRS did not own mask geometry "
                    f"(owner={geometry_owner:.6f})."
                )
            if factual_residual > min_teacher_fraction and teacher_count <= 0.0:
                raise RuntimeError(
                    "V561 scientific contract failed at epoch 1: factual residuals exist "
                    f"(fraction={factual_residual:.6f}) but TeacherComponentCount=0."
                )
            if teacher_count > 0.0 and (
                mask_bce <= 0.0 or mask_dice <= 0.0
                or action_loss <= 0.0 or presence_loss <= 0.0
            ):
                raise RuntimeError(
                    "V561 scientific contract failed at epoch 1: Teacher exists but one of the "
                    "three optimizer-owned M1 responsibilities is inactive: "
                    f"maskBCE={mask_bce:.6f}, maskDice={mask_dice:.6f}, "
                    f"action={action_loss:.6f}, presence={presence_loss:.6f}."
                )
            if abs(candidate_align) > 1.0e-10:
                raise RuntimeError(
                    "V561 scientific contract failed: whole-image CandidateAlignment must be diagnostic-only/zero, "
                    f"got {candidate_align:.6f}."
                )

        tracked = {
            key: means[key]
            for key in tracked_keys
            if key in means
        }
        if _semlt(cfg):
            tracked.update({
                key: value for key, value in means.items()
                if str(key).startswith(("geotr_m1_", "semlt_", "geotopo_", "mhcs_", "jbt_"))
            })
        elif _mhcs(cfg):
            tracked.update({
                key: value for key, value in means.items()
                if str(key).startswith(("mhcs_", "geotopo_", "geotr_", "sparc_hr_"))
            })
        elif _clean_dynamic_component_set(cfg):
            clean_whitelist = {
                "v484_base_dice", "v484_local_oracle_gain",
                "v538_m1_objective", "v538_m2_objective",
                "v538_teacher_component_count", "v538_teacher_component_oracle_gain",
                "v538_action_realizable_teacher_oracle_gain",
                "v561_teacher_set_oracle_gain", "v561_student_set_oracle_gain",
                "v561_set_oracle_realization_ratio",
                "v538_presence_loss", "v538_action_loss",
                "v538_m1_grad_norm_preclip", "v538_m1_grad_norm_postclip",
                "v538_m2_grad_norm_preclip", "v538_m2_grad_norm_postclip",
            }
            tracked.update({
                key: value for key, value in means.items()
                if str(key).startswith(("clean_", "tc_")) or key in clean_whitelist
            })
        else:
            tracked.update({
                key: value for key, value in means.items()
                if str(key).startswith(("v489_", "v503_", "v504_", "v505_", "v506_", "v507_", "v518_", "v519_", "v520_", "v521_", "v522_", "v523_", "v524_", "v526_", "v527_", "v528_", "v529_", "v530_", "v531_", "v532_", "v533_", "v534_", "v535_", "v536_", "v537_", "v538_", "v541_", "v542_", "v543_", "v544_", "v545_", "v547_", "v548_", "v549_", "v550_", "v551_", "v552_", "v552r3_", "v552r4_", "v552r41_", "v552r42_", "v552r43_", "v552r44_", "v552r45_", "v552r46_", "v552r47_", "v552r48_", "v552r49_", "v552r410_", "v552r411_", "v552r412_", "v552r413_", "v552r414_", "v552r415_", "v552r416_", "v552r417_", "v552r418_", "v552r419_", "v552r420_", "v552r4201_", "v552r4203_", "v552r4204_", "v552r4205_", "v552r4206_", "v552r4207_", "v552r4208_", "v552r4209_", "v552r4210_", "v552r4211_", "v552r4212_", "v560_", "v561_", "v562_", "v563_", "v564_", "v565_", "clean_"))
            })
        if tracked:
            logger.info(
                "M1_DIAG: %s",
                " | ".join(
                    f"{key}={value:.6f}"
                    for key, value in tracked.items()
                ),
            )

        # AUTO_PARAM_ADAPTER_AFTER_DIAG_BEGIN
        if auto_adapter is not None and auto_adapter.enabled:
            auto_adapter.on_epoch_end(epoch, means, optimizer)
        # AUTO_PARAM_ADAPTER_AFTER_DIAG_END

        validation_metrics = None
        if use_validation_selection and val_dataloader is not None:
            interval = max(1, int(_cfg_get(cfg.TRAIN, "VAL_INTERVAL", 1)))
            val_start = max(0, int(_cfg_get(cfg.TRAIN, "VAL_START_EPOCH", 0)))
            should_validate = (epoch >= val_start) and (
                (epoch + 1) % interval == 0
                or epoch + 1 == cfg.TRAIN.NUM_EPOCHS
            )
            if should_validate:
                selection_source = str(
                    _cfg_get(cfg.TRAIN, "VAL_WEIGHT_SOURCE", "raw")
                ).strip().lower()
                fixed_train_metrics = None
                fixed_diag_interval = max(
                    1, int(_cfg_get(cfg.TRAIN, "MHCS_FIXED_TRAIN_DIAG_INTERVAL", 1))
                )
                should_fixed_diag = (
                    fixed_train_dataloader is not None
                    and (((epoch + 1) % fixed_diag_interval == 0)
                         or (epoch + 1 == cfg.TRAIN.NUM_EPOCHS))
                )
                if selection_source == "ema" and ema is not None:
                    with ema.apply_ema():
                        validation_metrics = evaluate_validation(
                            model, val_dataloader, cfg.MODEL.DEVICE,
                            ce_loss, dice_loss, cfg
                        )
                        if should_fixed_diag:
                            fixed_train_metrics = evaluate_mhcs_fixed_train(
                                model, fixed_train_dataloader, cfg.MODEL.DEVICE, cfg
                            )
                else:
                    selection_source = "raw"
                    validation_metrics = evaluate_validation(
                        model, val_dataloader, cfg.MODEL.DEVICE,
                        ce_loss, dice_loss, cfg
                    )
                    if should_fixed_diag:
                        fixed_train_metrics = evaluate_mhcs_fixed_train(
                            model, fixed_train_dataloader, cfg.MODEL.DEVICE, cfg
                        )

                if fixed_train_metrics is not None:
                    logger.info(
                        "MHCS_FIXED_TRAIN epoch=%03d | Base=%.6f | BestSingle=%.6f | "
                        "JointOracle=%.6f | HardEnvelope=%.6f | HardPWO=%.6f | "
                        "SoftOracleRisk=%.6f | MeanUtility=%.6f | Envelope=%.6f | "
                        "Final=%.6f | NonBaseMass=%.6f | EffectiveRank=%.6f",
                        epoch + 1,
                        fixed_train_metrics["base"],
                        fixed_train_metrics["best_single"],
                        fixed_train_metrics["patch_oracle"],
                        fixed_train_metrics["hard_envelope"],
                        fixed_train_metrics["hard_pwo"],
                        fixed_train_metrics["soft_oracle_risk"],
                        fixed_train_metrics["global"],
                        fixed_train_metrics["local"],
                        fixed_train_metrics["final"],
                        fixed_train_metrics["gate_alpha"],
                        fixed_train_metrics["effective_rank"],
                    )
                    if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False)):
                        logger.info(
                            "PC2R_FIXED_TRAIN epoch=%03d | CanonicalGain=%+.6f FinalGain=%+.6f CandidateOracle=%+.6f",
                            epoch + 1,
                            fixed_train_metrics.get("pc2r_canonical_gain", 0.0),
                            fixed_train_metrics.get("pc2r_final_gain", 0.0),
                            fixed_train_metrics.get("pc2r_candidate_oracle_gain", 0.0),
                        )

                best_dice = max(best_dice, validation_metrics["base_dice"])
                best_fusion = max(
                    best_fusion, validation_metrics["fusion_dice"]
                )
                best_oracle = max(
                    best_oracle, validation_metrics["oracle_dice"]
                )
                metric_name = str(
                    _cfg_get(
                        cfg.TRAIN, "VAL_SELECTION_METRIC", "fusion_dice"
                    )
                )
                if metric_name not in validation_metrics:
                    raise KeyError(
                        f"Unknown VAL_SELECTION_METRIC={metric_name!r}; "
                        f"available={sorted(validation_metrics)}"
                    )
                selection_value = float(validation_metrics[metric_name])
                tiebreak_name = str(
                    _cfg_get(
                        cfg.TRAIN, "VAL_TIEBREAK_METRIC", "fusion_nsd"
                    )
                )
                selection_tiebreak = float(
                    validation_metrics.get(tiebreak_name, 0.0)
                )
                tolerance = float(
                    _cfg_get(cfg.TRAIN, "VAL_SELECTION_TOLERANCE", 1.0e-8)
                )
                use_shadow_selection = bool(
                    _cfg_get(
                        cfg.M1,
                        "V546_VAL_USE_SHADOW_M2_FOR_SELECTION",
                        False,
                    )
                )
                sparc_hr_selection = bool(
                    _cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)
                )
                if _semlt(cfg):
                    selected_native_dice_key = "native_m1_dice"
                    selected_native_nsd_key = "native_m1_nsd"
                    qualification_base_dice_key = "native_base_dice"
                    qualification_base_nsd_key = "native_base_nsd"
                    qualification_m1_dice_key = "native_m1_dice"
                    qualification_m1_nsd_key = "native_m1_nsd"
                elif sparc_hr_selection:
                    selected_native_dice_key = "native_sparc_hr_dice"
                    selected_native_nsd_key = "native_sparc_hr_nsd"
                    qualification_base_dice_key = "native_sparc_base_hr_dice"
                    qualification_base_nsd_key = "native_sparc_base_hr_nsd"
                    qualification_m1_dice_key = "native_sparc_m1_hr_dice"
                    qualification_m1_nsd_key = "native_sparc_m1_hr_nsd"
                else:
                    selected_native_dice_key = (
                        "native_shadow_m2_dice"
                        if use_shadow_selection else "native_fusion_dice"
                    )
                    selected_native_nsd_key = (
                        "native_shadow_m2_nsd"
                        if use_shadow_selection else "native_fusion_nsd"
                    )
                    qualification_base_dice_key = "native_base_dice"
                    qualification_base_nsd_key = "native_base_nsd"
                    qualification_m1_dice_key = "native_m1_dice"
                    qualification_m1_nsd_key = "native_m1_nsd"
                selected_native_cat_key = (
                    "native_shadow_m2_dice_catastrophic_rate"
                    if use_shadow_selection
                    else "native_fusion_dice_catastrophic_rate"
                )

                current_native_base_dice = float(
                    validation_metrics.get("native_base_dice", validation_metrics["base_dice"])
                )
                current_native_base_nsd = float(
                    validation_metrics.get("native_base_nsd", validation_metrics["base_nsd"])
                )
                current_native_base_cat = float(
                    validation_metrics.get(
                        "native_base_dice_catastrophic_rate",
                        validation_metrics.get("base_dice_catastrophic_rate", 0.0),
                    )
                )
                base_checkpoint_improved = (
                    current_native_base_dice > best_native_base_dice + tolerance
                    or (
                        abs(current_native_base_dice - best_native_base_dice) <= tolerance
                        and current_native_base_nsd > best_native_base_nsd
                    )
                )
                if base_checkpoint_improved:
                    best_native_base_dice = current_native_base_dice
                    best_native_base_nsd = current_native_base_nsd
                    best_native_base_catastrophic_rate = current_native_base_cat
                    base_state = _checkpoint_state(
                        model, optimizer, scheduler, epoch, best_dice,
                        best_fusion, best_oracle, run_name, cfg,
                        phase_b_started=False, ema=ema,
                        weight_source=selection_source,
                        hard_case_memory=hard_case_memory,
                    )
                    base_state.update({
                        "validation_metrics": validation_metrics,
                        "best_native_base_dice": best_native_base_dice,
                        "best_native_base_nsd": best_native_base_nsd,
                        "best_native_base_catastrophic_rate": best_native_base_catastrophic_rate,
                        "checkpoint_role": "best_base_validation_fallback",
                        "split_access_during_training": {
                            "train": True, "val": True, "test": False
                        },
                    })
                    torch.save(base_state, best_base_selection_path)
                    logger.info(
                        "[V507_BASE_GUARD] Saved best Base checkpoint: %s "
                        "DSC/NSD=%.6f/%.6f cat=%.6f",
                        best_base_selection_path,
                        best_native_base_dice,
                        best_native_base_nsd,
                        best_native_base_catastrophic_rate,
                    )

                # R4.20.9: separate, fair M1-only validation selection.  The
                # main final-system checkpoint uses the configured deployed
                # resolution (native_sparc_hr_* for a true HR refiner);
                # this checkpoint exists solely for the M1 ablation/Test claim.
                current_native_m1_dice = float(
                    validation_metrics.get("native_m1_dice", current_native_base_dice)
                )
                current_native_m1_nsd = float(
                    validation_metrics.get("native_m1_nsd", current_native_base_nsd)
                )
                m1_checkpoint_improved = (
                    current_native_m1_dice > best_native_m1_dice + tolerance
                    or (
                        abs(current_native_m1_dice - best_native_m1_dice) <= tolerance
                        and current_native_m1_nsd > best_native_m1_nsd
                    )
                )
                if m1_checkpoint_improved:
                    best_native_m1_dice = current_native_m1_dice
                    best_native_m1_nsd = current_native_m1_nsd
                    best_native_m1_epoch = epoch
                    m1_state = _checkpoint_state(
                        model, optimizer, scheduler, epoch, best_dice,
                        best_fusion, best_oracle, run_name, cfg,
                        phase_b_started=False, ema=ema,
                        weight_source=selection_source,
                        hard_case_memory=hard_case_memory,
                    )
                    m1_state.update({
                        "validation_metrics": validation_metrics,
                        "best_native_m1_dice": best_native_m1_dice,
                        "best_native_m1_nsd": best_native_m1_nsd,
                        "best_native_m1_epoch": best_native_m1_epoch,
                        "checkpoint_role": "best_m1_native_validation_ablation",
                        "selection_metric": "native_m1_dice",
                        "selection_tiebreak_metric": "native_m1_nsd",
                        "selection_protocol": (
                            "M1-only checkpoint selected on validation native_m1_dice; "
                            "main-system deployed-output selection remains separate; Test unopened."
                        ),
                        "split_access_during_training": {
                            "train": True, "val": True, "test": False
                        },
                    })
                    torch.save(m1_state, best_m1_selection_path)
                    logger.info(
                        "[V552R4209_M1_SELECTION] Saved best M1Native checkpoint: %s "
                        "DSC/NSD=%.6f/%.6f epoch=%d",
                        best_m1_selection_path, best_native_m1_dice,
                        best_native_m1_nsd, epoch + 1,
                    )

                qualifies = True
                if bool(_cfg_get(cfg.TRAIN, "VAL_REQUIRE_NATIVE_NONDEGRADATION", False)):
                    qualifies = (
                        validation_metrics.get(selected_native_dice_key, float("-inf"))
                        >= validation_metrics.get(qualification_base_dice_key, float("inf"))
                        and validation_metrics.get(selected_native_nsd_key, float("-inf"))
                        >= validation_metrics.get(qualification_base_nsd_key, float("inf"))
                    )
                if qualifies and bool(_cfg_get(cfg.TRAIN, "VAL_REQUIRE_M3_NONDEGRADATION_OVER_M2", False)):
                    qualifies = (
                        validation_metrics.get("native_fusion_dice", float("-inf"))
                        >= validation_metrics.get("native_m2_dice", float("inf"))
                        and validation_metrics.get("native_fusion_nsd", float("-inf"))
                        >= validation_metrics.get("native_m2_nsd", float("inf"))
                    )
                if qualifies and bool(_cfg_get(cfg.TRAIN, "VAL_REQUIRE_M2_NONDEGRADATION_OVER_BASE", False)):
                    m2_dice_key = selected_native_dice_key if sparc_hr_selection else (
                        "native_shadow_m2_dice" if use_shadow_selection else "native_m2_dice"
                    )
                    m2_nsd_key = selected_native_nsd_key if sparc_hr_selection else (
                        "native_shadow_m2_nsd" if use_shadow_selection else "native_m2_nsd"
                    )
                    qualifies = (
                        validation_metrics.get(m2_dice_key, float("-inf"))
                        >= validation_metrics.get(qualification_base_dice_key, float("inf"))
                        and validation_metrics.get(m2_nsd_key, float("-inf"))
                        >= validation_metrics.get(qualification_base_nsd_key, float("inf"))
                    )
                if qualifies and bool(_cfg_get(
                    cfg.TRAIN, "VAL_REQUIRE_M2_NONDEGRADATION_OVER_M1", False
                )):
                    m2_dice_key = selected_native_dice_key if sparc_hr_selection else (
                        "native_shadow_m2_dice" if use_shadow_selection else "native_m2_dice"
                    )
                    m2_nsd_key = selected_native_nsd_key if sparc_hr_selection else (
                        "native_shadow_m2_nsd" if use_shadow_selection else "native_m2_nsd"
                    )
                    min_m2_over_m1 = float(_cfg_get(
                        cfg.TRAIN, "VAL_MIN_NATIVE_M2_OVER_M1_GAIN", 0.0
                    ))
                    qualifies = (
                        validation_metrics.get(m2_dice_key, float("-inf"))
                        - validation_metrics.get(qualification_m1_dice_key, float("inf"))
                        >= min_m2_over_m1 - tolerance
                        and validation_metrics.get(m2_nsd_key, float("-inf"))
                        >= validation_metrics.get(qualification_m1_nsd_key, float("inf")) - tolerance
                    )
                if qualifies and bool(_cfg_get(
                    cfg.TRAIN, "VAL_REQUIRE_M2_TAIL_NONDEGRADATION_OVER_BASE", False
                )):
                    qualifies = (
                        validation_metrics.get("native_m2_dice_tail_score", float("-inf"))
                        >= validation_metrics.get("native_base_dice_tail_score", float("inf"))
                    )
                if qualifies and bool(_cfg_get(
                    cfg.TRAIN, "VAL_REQUIRE_M2_HARM_NOT_EXCEED_BENEFIT", False
                )):
                    benefit_mag = float(validation_metrics.get("native_m2_mean_positive_gain", 0.0))
                    harm_mag = float(validation_metrics.get("native_m2_mean_harm_magnitude", float("inf")))
                    # Natural unit-free safety contract: an average harmful edit
                    # may not be larger than an average beneficial edit.
                    qualifies = benefit_mag > 0.0 and harm_mag <= benefit_mag + tolerance
                if qualifies:
                    min_m1_oracle_gain = float(
                        _cfg_get(cfg.TRAIN, "VAL_MIN_NATIVE_M1_ORACLE_GAIN", 0.0)
                    )
                    if min_m1_oracle_gain > 0.0:
                        oracle_key = (
                            "native_component_oracle_dice"
                            if bool(_cfg_get(
                                cfg.M1,
                                "V538_ONLINE_COMPONENT_REFINER_ENABLED",
                                False,
                            ))
                            else "native_oracle_dice"
                        )
                        current_m1_oracle_gain = (
                            float(validation_metrics.get(oracle_key, float("-inf")))
                            - float(validation_metrics.get("native_base_dice", float("inf")))
                        )
                        qualifies = current_m1_oracle_gain >= min_m1_oracle_gain - tolerance
                if qualifies and bool(_cfg_get(cfg.TRAIN, "VAL_REQUIRE_BASE_WITHIN_BEST", False)):
                    allowed_drop = max(
                        0.0, float(_cfg_get(cfg.TRAIN, "VAL_BASE_DICE_WITHIN_BEST", 0.0))
                    )
                    qualifies = current_native_base_dice >= best_native_base_dice - allowed_drop
                if qualifies:
                    minimum_base = float(_cfg_get(cfg.TRAIN, "VAL_MIN_NATIVE_BASE_DICE", 0.0))
                    qualifies = current_native_base_dice >= minimum_base
                if qualifies and bool(_cfg_get(cfg.TRAIN, "VAL_REQUIRE_FUSION_CATASTROPHIC_NONDEGRADATION", False)):
                    selected_cat = float(validation_metrics.get(
                        selected_native_cat_key,
                        validation_metrics.get("fusion_dice_catastrophic_rate", 1.0),
                    ))
                    qualifies = selected_cat <= current_native_base_cat + tolerance
                if qualifies:
                    max_cat = float(_cfg_get(cfg.TRAIN, "VAL_MAX_CATASTROPHIC_RATE", 1.0))
                    selected_cat = float(validation_metrics.get(
                        selected_native_cat_key,
                        validation_metrics.get("fusion_dice_catastrophic_rate", 1.0),
                    ))
                    qualifies = selected_cat <= max_cat + tolerance
                if qualifies:
                    min_final_gain = float(_cfg_get(cfg.TRAIN, "VAL_MIN_NATIVE_FINAL_GAIN", 0.0))
                    qualifies = (
                        validation_metrics.get(selected_native_dice_key, float("-inf"))
                        - validation_metrics.get(qualification_base_dice_key, float("inf"))
                        >= min_final_gain - tolerance
                    )
                # V515: validation can run early for diagnostics, but a checkpoint
                # may enter formal best-model selection only after every enabled
                # module has reached the configured curriculum stage.  This key
                # is one-based to match the epoch numbers printed in logs.
                selection_start_epoch_1based = max(
                    1, int(_cfg_get(cfg.TRAIN, "VAL_SELECTION_START_EPOCH", 1))
                )
                selection_eligible = (epoch + 1) >= selection_start_epoch_1based
                improved = selection_eligible and qualifies and (
                    selection_value > best_selection_value + tolerance
                    or (
                        abs(selection_value - best_selection_value) <= tolerance
                        and selection_tiebreak > best_selection_tiebreak
                    )
                )
                logger.info(
                    "VAL epoch=%03d | base DSC/NSD=%.6f/%.6f | "
                    "fusion DSC/NSD=%.6f/%.6f | oracle DSC=%.6f | "
                    "tail(q10/cvar/cat)=%.6f/%.6f/%.6f | "
                    "selection(%s)=%.6f | source=%s | improved=%s",
                    epoch + 1,
                    validation_metrics["base_dice"],
                    validation_metrics["base_nsd"],
                    validation_metrics["fusion_dice"],
                    validation_metrics["fusion_nsd"],
                    validation_metrics["oracle_dice"],
                    validation_metrics.get("fusion_dice_q10", 0.0),
                    validation_metrics.get("fusion_dice_cvar", 0.0),
                    validation_metrics.get("fusion_dice_catastrophic_rate", 0.0),
                    metric_name, selection_value, selection_source, improved,
                )
                if _mhcs(cfg):
                    logger.info(
                        "VAL_MHCS epoch=%03d | Base=%.6f | BestSingle=%.6f | "
                        "JointOracle=%.6f | HardEnvelope=%.6f | MeanUtility=%.6f | "
                        "Envelope=%.6f | Final=%.6f | NonBaseMass=%.6f | "
                        "PWO=%.6f | SoftOracleRisk=%.6f | EffectiveRank=%.6f | "
                        "JointGap=%.6f | FinalVsBase=%+.6f | FinalVsBestSingle=%+.6f | "
                        "FinalVsJointOracle=%+.6f | HardVsBase=%+.6f | GapToPWO=%.6f",
                        epoch + 1,
                        validation_metrics.get("base_dice", 0.0),
                        validation_metrics.get("oracle_dice", 0.0),
                        validation_metrics.get("mhcs_patch_oracle_dice", 0.0),
                        validation_metrics.get("mhcs_hard_envelope_dice", 0.0),
                        validation_metrics.get("mhcs_global_selected_dice", 0.0),
                        validation_metrics.get("mhcs_local_dice", 0.0),
                        validation_metrics.get("fusion_dice", 0.0),
                        validation_metrics.get("mhcs_gate_alpha", 0.0),
                        validation_metrics.get("pwo_dice", 0.0),
                        validation_metrics.get("mhcs_soft_oracle_risk", 0.0),
                        validation_metrics.get("mhcs_effective_rank", 0.0),
                        validation_metrics.get("mhcs_patch_oracle_dice", 0.0)
                        - validation_metrics.get("oracle_dice", 0.0),
                        validation_metrics.get("fusion_dice", 0.0)
                        - validation_metrics.get("base_dice", 0.0),
                        validation_metrics.get("fusion_dice", 0.0)
                        - validation_metrics.get("oracle_dice", 0.0),
                        validation_metrics.get("fusion_dice", 0.0)
                        - validation_metrics.get("mhcs_patch_oracle_dice", 0.0),
                        validation_metrics.get("mhcs_hard_envelope_dice", 0.0)
                        - validation_metrics.get("base_dice", 0.0),
                        validation_metrics.get("pwo_dice", 0.0)
                        - validation_metrics.get("fusion_dice", 0.0),
                    )
                    if "native_mhcs_global_selected_dice" in validation_metrics:
                        logger.info(
                            "VAL_MHCS_NATIVE epoch=%03d | Base=%.6f | MeanUtility=%.6f | "
                            "Final=%.6f | MeanUtilityVsBase=%+.6f | FinalVsMeanUtility=%+.6f",
                            epoch + 1,
                            validation_metrics.get("native_base_dice", 0.0),
                            validation_metrics.get("native_mhcs_global_selected_dice", 0.0),
                            validation_metrics.get("native_fusion_dice", 0.0),
                            validation_metrics.get("native_mhcs_global_selected_dice", 0.0)
                            - validation_metrics.get("native_base_dice", 0.0),
                            validation_metrics.get("native_fusion_dice", 0.0)
                            - validation_metrics.get("native_mhcs_global_selected_dice", 0.0),
                        )
                    if "native_geotr_geometry_dice" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR epoch=%03d | Base=%.6f/%.6f | Transport=%.6f/%.6f (%+.6f/%+.6f) | "
                            "Recon@Base=%.6f/%.6f (%+.6f/%+.6f) | Recon@Transport=%.6f/%.6f (%+.6f/%+.6f) | "
                            "Final=%.6f/%.6f (%+.6f/%+.6f)",
                            epoch + 1,
                            validation_metrics.get("native_base_dice", 0.0),
                            validation_metrics.get("native_base_nsd", 0.0),
                            validation_metrics.get("native_geotr_geometry_dice", 0.0),
                            validation_metrics.get("native_geotr_geometry_nsd", 0.0),
                            validation_metrics.get("native_geotr_geometry_gain", 0.0),
                            validation_metrics.get("native_geotr_geometry_nsd_gain", 0.0),
                            validation_metrics.get("native_geotr_recon_base_dice", 0.0),
                            validation_metrics.get("native_geotr_recon_base_nsd", 0.0),
                            validation_metrics.get("native_geotr_recon_base_gain", 0.0),
                            validation_metrics.get("native_geotr_recon_base_nsd_gain", 0.0),
                            validation_metrics.get("native_geotr_recon_after_geometry_dice", 0.0),
                            validation_metrics.get("native_geotr_recon_after_geometry_nsd", 0.0),
                            validation_metrics.get("native_geotr_recon_after_geometry_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_geotr_recon_after_geometry_nsd_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_m2_dice", validation_metrics.get("native_fusion_dice", 0.0)),
                            validation_metrics.get("native_m2_nsd", validation_metrics.get("native_fusion_nsd", 0.0)),
                            validation_metrics.get("native_m2_gain", 0.0),
                            validation_metrics.get("native_m2_nsd_gain", 0.0),
                        )
                    if "val_sparc_hr_quality_gain" in validation_metrics:
                        logger.info(
                            "VAL_SPARC_HR3 epoch=%03d | Canonical M1/M2=%.6f/%.6f -> %.6f/%.6f (%+.6f/%+.6f) | "
                            "HR Base=%.6f/%.6f M1=%.6f/%.6f M2=%.6f/%.6f (%+.6f/%+.6f) | "
                            "PairAudit=%.6f/%.6f | QGain=%+.6f DenseOracle=%+.6f OneShotOracle=%+.6f Attainable=%+.6f | "
                            "Selected=%+.6f Regret=%.6f Source/PreserveAcc=%.4f/%.4f EditPrec/Harm=%.4f/%.4f CandBenefit=%.4f | "
                            "MeanBenefit/HarmMag=%+.6f/%.6f Valid/NoOp/Dup=%.4f/%.4f/%.4f | "
                            "Execute case/cell=%.4f/%.4f Benefit/Harm=%.4f/%.4f | Corr/Intro/Net=%.0f/%.0f/%+.0f",
                            epoch + 1,
                            validation_metrics.get("native_m1_dice", 0.0),
                            validation_metrics.get("native_m1_nsd", 0.0),
                            validation_metrics.get("native_m2_dice", 0.0),
                            validation_metrics.get("native_m2_nsd", 0.0),
                            validation_metrics.get("native_m2_dice", 0.0)
                            - validation_metrics.get("native_m1_dice", 0.0),
                            validation_metrics.get("native_m2_nsd", 0.0)
                            - validation_metrics.get("native_m1_nsd", 0.0),
                            validation_metrics.get("native_sparc_base_hr_dice", 0.0),
                            validation_metrics.get("native_sparc_base_hr_nsd", 0.0),
                            validation_metrics.get("native_sparc_m1_hr_dice", 0.0),
                            validation_metrics.get("native_sparc_m1_hr_nsd", 0.0),
                            validation_metrics.get("native_sparc_hr_dice", 0.0),
                            validation_metrics.get("native_sparc_hr_nsd", 0.0),
                            validation_metrics.get("native_sparc_hr_dice", 0.0)
                            - validation_metrics.get("native_sparc_m1_hr_dice", 0.0),
                            validation_metrics.get("native_sparc_hr_nsd", 0.0)
                            - validation_metrics.get("native_sparc_m1_hr_nsd", 0.0),
                            validation_metrics.get("sparc_lr_hr_gt_dice", 0.0),
                            validation_metrics.get("sparc_lr_hr_gt_nsd", 0.0),
                            validation_metrics.get("val_sparc_hr_quality_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_atomic_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_true_sequential_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_on_policy_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_utility", 0.0),
                            validation_metrics.get("val_sparc_hr_policy_regret", 0.0),
                            validation_metrics.get("val_sparc_hr_policy_top_choice_accuracy", 0.0),
                            validation_metrics.get("val_sparc_hr_policy_stop_accuracy", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_action_precision", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_action_harm_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_candidate_benefit_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_mean_benefit", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_mean_harm_magnitude", 0.0),
                            validation_metrics.get("val_sparc_hr_valid_candidate_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_no_op_candidate_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_duplicate_candidate_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_execution_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_step_execution_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_benefit_case_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_harm_case_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_corrected_count", 0.0),
                            validation_metrics.get("val_sparc_hr_introduced_count", 0.0),
                            validation_metrics.get("val_sparc_hr_net_correction_count", 0.0),
                        )
                        logger.info(
                            "VAL_SPARC_DENSE epoch=%03d | DenseOracle=%+.6f | Reserved2/3=%+.6f/%+.6f | "
                            "SelectedAdv=%+.6f Reserved2/3=%+.6f/%+.6f | TeacherPreserve=%.4f | EditPrecision=%.4f",
                            epoch + 1,
                            validation_metrics.get("val_sparc_hr_true_oracle_step1_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_true_oracle_step2_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_true_oracle_step3_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_step1_utility", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_step2_utility", 0.0),
                            validation_metrics.get("val_sparc_hr_selected_step3_utility", 0.0),
                            validation_metrics.get("val_sparc_hr_teacher_stop_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_lcb_positive_precision", 0.0),
                        )
                        logger.info(
                            "VAL_SPARC_ROUTER epoch=%03d | AttainableEdit P/R=%.4f/%.4f | AliasAudit P/R=%.4f/%.4f | "
                            "Joint P/R=%.4f/%.4f | Edit/candidate-benefit aliases=%.4f/%.4f/%.4f/%.4f | "
                            "SourceOracle post/proposal/add/remove=%+.6f/%+.6f/%+.6f/%+.6f",
                            epoch + 1,
                            validation_metrics.get("val_sparc_hr_benefit_precision", 0.0),
                            validation_metrics.get("val_sparc_hr_benefit_recall", 0.0),
                            validation_metrics.get("val_sparc_hr_quantile_positive_precision", 0.0),
                            validation_metrics.get("val_sparc_hr_quantile_recall", 0.0),
                            validation_metrics.get("val_sparc_hr_joint_precision", 0.0),
                            validation_metrics.get("val_sparc_hr_joint_recall", 0.0),
                            validation_metrics.get("val_sparc_hr_policy_gate_pass_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_benefit_candidate_pass_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_quantile_candidate_pass_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_joint_candidate_pass_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_posterior_only_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_proposal_only_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_add_only_oracle_gain", 0.0),
                            validation_metrics.get("val_sparc_hr_remove_only_oracle_gain", 0.0),
                        )
                        logger.info(
                            "VAL_SPARC_SOURCES epoch=%03d | Actionable=%.4f | RouteMass M1/post/proposal/add/remove="
                            "%.4f/%.4f/%.4f/%.4f/%.4f",
                            epoch + 1,
                            validation_metrics.get("val_sparc_hr_router_actionable_rate", 0.0),
                            validation_metrics.get("val_sparc_hr_router_anchor_mass", 0.0),
                            validation_metrics.get("val_sparc_hr_router_posterior_mass", 0.0),
                            validation_metrics.get("val_sparc_hr_router_proposal_mass", 0.0),
                            validation_metrics.get("val_sparc_hr_router_add_mass", 0.0),
                            validation_metrics.get("val_sparc_hr_router_remove_mass", 0.0),
                        )
                    if "val_geotr_v4c_typed_macro_f1" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4C epoch=%03d | "
                            "val_geotr_v4c_error_rate=%.6f | val_geotr_v4c_fn_rate=%.6f | val_geotr_v4c_fp_rate=%.6f | "
                            "val_geotr_v4c_typed_macro_f1=%.6f | val_geotr_v4c_fn_f1=%.6f | val_geotr_v4c_fp_f1=%.6f | "
                            "val_geotr_v4c_q_true_error_mass=%.6f | val_geotr_v4c_q_false_positive_mass_correct=%.6f | "
                            "val_geotr_v4c_direction_accuracy=%.6f | "
                            "val_geotr_v4c_oracle_where_gain=%+.6f | val_geotr_v4c_oracle_sign_gain=%+.6f | "
                            "val_geotr_v4c_oracle_typed_gain=%+.6f | val_geotr_v4c_oracle_full_gain=%+.6f | "
                            "val_geotr_v4c_base_paired_error_fraction=%.6f | val_geotr_v4c_geo_paired_error_fraction=%.6f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4c_error_rate", 0.0),
                            validation_metrics.get("val_geotr_v4c_fn_rate", 0.0),
                            validation_metrics.get("val_geotr_v4c_fp_rate", 0.0),
                            validation_metrics.get("val_geotr_v4c_typed_macro_f1", 0.0),
                            validation_metrics.get("val_geotr_v4c_fn_f1", 0.0),
                            validation_metrics.get("val_geotr_v4c_fp_f1", 0.0),
                            validation_metrics.get("val_geotr_v4c_q_true_error_mass", 0.0),
                            validation_metrics.get("val_geotr_v4c_q_false_positive_mass_correct", 0.0),
                            validation_metrics.get("val_geotr_v4c_direction_accuracy", 0.0),
                            validation_metrics.get("val_geotr_v4c_oracle_where_gain", 0.0),
                            validation_metrics.get("val_geotr_v4c_oracle_sign_gain", 0.0),
                            validation_metrics.get("val_geotr_v4c_oracle_typed_gain", 0.0),
                            validation_metrics.get("val_geotr_v4c_oracle_full_gain", 0.0),
                            validation_metrics.get("val_geotr_v4c_base_paired_error_fraction", 0.0),
                            validation_metrics.get("val_geotr_v4c_geo_paired_error_fraction", 0.0),
                        )
                    if "val_geotr_v4g_modelres_soft_gain" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4G epoch=%03d | Stage2Soft=%+.6f | Select C/P/R=%.4f/%.4f/%.4f ErrRate=%.4f | "
                            "PointAcc before/after/gain=%.4f/%.4f/%+.4f Change=%.6f | "
                            "BatchAP margin/std/dis/ent=%.4f/%.4f/%.4f/%.4f | "
                            "Recall@5/10 margin=%.4f/%.4f std=%.4f/%.4f dis=%.4f/%.4f | "
                            "Corr/Intro/Net=%.0f/%.0f/%+.0f CorrRecall=%.4f IntroRate=%.4f | "
                            "SelectorOracle=%+.6f FullResidualOracle=%+.6f | "
                            "NativeStage2=%+.6f NativeBenefit/Harm=%.4f/%.4f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4g_modelres_soft_gain", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_coverage", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_precision", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_recall", 0.0),
                            validation_metrics.get("val_geotr_v4g_anchor_error_rate", 0.0),
                            validation_metrics.get("val_geotr_v4g_point_accuracy_before", 0.0),
                            validation_metrics.get("val_geotr_v4g_point_accuracy_after", 0.0),
                            validation_metrics.get("val_geotr_v4g_point_accuracy_gain", 0.0),
                            validation_metrics.get("val_geotr_v4g_selected_abs_change", 0.0),
                            validation_metrics.get("val_geotr_v4g_margin_error_ap", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_std_error_ap", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_disagreement_error_ap", 0.0),
                            validation_metrics.get("val_geotr_v4g_entropy_error_ap", 0.0),
                            validation_metrics.get("val_geotr_v4g_margin_recall_at_05", 0.0),
                            validation_metrics.get("val_geotr_v4g_margin_recall_at_10", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_std_recall_at_05", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_std_recall_at_10", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_disagreement_recall_at_05", 0.0),
                            validation_metrics.get("val_geotr_v4g_mc_disagreement_recall_at_10", 0.0),
                            validation_metrics.get("val_geotr_v4g_corrected_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_introduced_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_correction_recall", 0.0),
                            validation_metrics.get("val_geotr_v4g_introduction_rate", 0.0),
                            validation_metrics.get("val_geotr_v4g_pred_selector_oracle_gain", 0.0),
                            validation_metrics.get("val_geotr_v4g_full_residual_oracle_gain", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_geotr_stage2_beneficial_case_rate", 0.0),
                            validation_metrics.get("native_geotr_stage2_harmful_case_rate", 0.0),
                        )
                    if "val_geotr_c2r_stage2_gain" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_C2R epoch=%03d | Stage2=%+.6f RegionCov=%.4f ErrDensity/Recall=%.4f/%.4f | "
                            "Consensus/Edit=%.4f/%.4f EditP=%.4f Spread=%.4f | Corr/Intro/Net=%.0f/%.0f/%+.0f | "
                            "RegionOracle=%+.6f Harm=%.4f NativeStage2=%+.6f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_c2r_stage2_gain", 0.0),
                            validation_metrics.get("val_geotr_c2r_region_coverage", 0.0),
                            validation_metrics.get("val_geotr_c2r_region_error_density", 0.0),
                            validation_metrics.get("val_geotr_c2r_region_error_recall", 0.0),
                            validation_metrics.get("val_geotr_c2r_consensus_rate_in_region", 0.0),
                            validation_metrics.get("val_geotr_c2r_edit_rate_in_region", 0.0),
                            validation_metrics.get("val_geotr_c2r_edit_precision", 0.0),
                            validation_metrics.get("val_geotr_c2r_view_spread_in_region", 0.0),
                            validation_metrics.get("val_geotr_v4g_corrected_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_introduced_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                            validation_metrics.get("val_geotr_c2r_region_oracle_gain", 0.0),
                            validation_metrics.get("val_geotr_c2r_harm_case_rate", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                        )
                        if bool(_cfg_get(cfg.M1, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False)):
                            logger.info(
                                "VAL_GEOTR_C2R_V2 epoch=%03d | NativeContext/CandidateOracle=%+.6f/%+.6f | "
                                "ROI unique/overlap/minDist=%.0f/%.0f/%.1f | Candidate/CommitPix=%.0f/%.0f | "
                                "Comp cand/commit=%.0f/%.0f area=%.2f/%.2f | "
                                "Agree E/C/Cand=%.3f/%.3f/%.3f Spread E/C/Cand=%.4f/%.4f/%.4f | "
                                "CanonicalGain=%+.6f Benefit/Harm=%.3f/%.3f",
                                epoch + 1,
                                validation_metrics.get("native_c2r_context_oracle_gain", 0.0),
                                validation_metrics.get("native_c2r_candidate_oracle_gain", 0.0),
                                validation_metrics.get("val_geotr_c2r_roi_unique_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_roi_overlap_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_center_min_chebyshev_distance", 0.0),
                                validation_metrics.get("val_geotr_c2r_candidate_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_commit_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_candidate_component_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_committed_component_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_candidate_component_area_mean", 0.0),
                                validation_metrics.get("val_geotr_c2r_committed_component_area_mean", 0.0),
                                validation_metrics.get("val_geotr_c2r_agreement_error", 0.0),
                                validation_metrics.get("val_geotr_c2r_agreement_correct", 0.0),
                                validation_metrics.get("val_geotr_c2r_agreement_candidate", 0.0),
                                validation_metrics.get("val_geotr_c2r_spread_error", 0.0),
                                validation_metrics.get("val_geotr_c2r_spread_correct", 0.0),
                                validation_metrics.get("val_geotr_c2r_spread_candidate", 0.0),
                                validation_metrics.get("val_geotr_c2r_canonical_mean_gain", 0.0),
                                validation_metrics.get("val_geotr_c2r_benefit_case_rate", 0.0),
                                validation_metrics.get("val_geotr_c2r_harm_case_rate", 0.0),
                            )
                    if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False)) and "val_geotr_c2r_stage2_gain" in validation_metrics:
                        if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)):
                            logger.info(
                                "VAL_PC2R_V32 epoch=%03d | ReachableErr=%.4f UnreachFN/FP=%.0f/%.0f ReachOracle=%+.6f | "
                                "ErrNearBnd<=1/2/3/5/>5=%.3f/%.3f/%.3f/%.3f/%.3f Bnd5/InteriorOracle=%+.6f/%+.6f | "
                                "NativeOp can/cand/area/risk/dir/spread/strength=%+.6f/%+.6f/%+.6f/%+.6f/%+.6f/%+.6f/%+.6f | "
                                "CanonicalGain=%+.6f NativeStage2=%+.6f EmptyGain/Harm=%+.6f/%.3f Q1/Q4Gain=%+.6f/%+.6f",
                                epoch + 1,
                                validation_metrics.get("val_pc2r_reachable_region_error_fraction", 0.0),
                                validation_metrics.get("val_pc2r_unreachable_fn_count", 0.0),
                                validation_metrics.get("val_pc2r_unreachable_fp_count", 0.0),
                                validation_metrics.get("val_pc2r_reachable_oracle_gain", 0.0),
                                validation_metrics.get("val_pc2r_error_near_boundary_1_fraction", 0.0),
                                validation_metrics.get("val_pc2r_error_near_boundary_2_fraction", 0.0),
                                validation_metrics.get("val_pc2r_error_near_boundary_3_fraction", 0.0),
                                validation_metrics.get("val_pc2r_error_near_boundary_5_fraction", 0.0),
                                validation_metrics.get("val_pc2r_error_beyond_boundary_5_fraction", 0.0),
                                validation_metrics.get("val_pc2r_boundary5_oracle_gain", 0.0),
                                validation_metrics.get("val_pc2r_interior_oracle_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_canonical_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_candidate_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_area_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_risk_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_direction_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_spread_gain", 0.0),
                                validation_metrics.get("native_pc2r_stage_strength_gain", 0.0),
                                validation_metrics.get("val_geotr_c2r_canonical_mean_gain", 0.0),
                                validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                                validation_metrics.get("native_pc2r_empty_case_mean_gain", 0.0),
                                validation_metrics.get("native_pc2r_empty_case_harm_rate", 0.0),
                                validation_metrics.get("native_pc2r_q1_small_mean_gain", 0.0),
                                validation_metrics.get("native_pc2r_q4_large_mean_gain", 0.0),
                            )
                        if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED", False)):
                            logger.info(
                                "VAL_PC2R_V31 epoch=%03d | PosteriorDiv=%.5f CenterBias abs/signed=%.5f/%+.5f dLogit=%.5f Reliance fact/shuf=%.5f/%.5f | "
                                "Cert raw/area/risk/dir/spread/strength/all=%.0f/%.0f/%.0f/%.0f/%.0f/%.0f/%.0f | "
                                "RawPix=%.0f RiskPix=%.0f | CanonicalGain=%+.6f NativeStage2=%+.6f",
                                epoch + 1,
                                validation_metrics.get("val_pc2r_selected_posterior_diversity", 0.0),
                                validation_metrics.get("val_pc2r_selected_center_bias_abs", 0.0),
                                validation_metrics.get("val_pc2r_selected_center_bias_signed", 0.0),
                                validation_metrics.get("val_pc2r_mean_abs_delta_logit", 0.0),
                                validation_metrics.get("val_pc2r_reliance_factualized", 0.0),
                                validation_metrics.get("val_pc2r_reliance_shuffled", 0.0),
                                validation_metrics.get("val_pc2r_component_raw_count", 0.0),
                                validation_metrics.get("val_pc2r_component_area_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_risk_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_direction_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_spread_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_strength_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_all_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_raw_correction_pixel_count", 0.0),
                                validation_metrics.get("val_pc2r_risk_support_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_canonical_mean_gain", 0.0),
                                validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            )
                        else:
                            logger.info(
                                "VAL_PC2R_V3 epoch=%03d | PosteriorDiv=%.5f dLogit=%.5f Reliance fact/shuf=%.5f/%.5f | "
                                "Cert raw/area/risk/dir/spread/strength/all=%.0f/%.0f/%.0f/%.0f/%.0f/%.0f/%.0f | "
                                "RawPix=%.0f RiskPix=%.0f | CanonicalGain=%+.6f NativeStage2=%+.6f",
                                epoch + 1,
                                validation_metrics.get("val_pc2r_selected_posterior_diversity", 0.0),
                                validation_metrics.get("val_pc2r_mean_abs_delta_logit", 0.0),
                                validation_metrics.get("val_pc2r_reliance_factualized", 0.0),
                                validation_metrics.get("val_pc2r_reliance_shuffled", 0.0),
                                validation_metrics.get("val_pc2r_component_raw_count", 0.0),
                                validation_metrics.get("val_pc2r_component_area_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_risk_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_direction_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_spread_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_strength_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_component_all_pass_count", 0.0),
                                validation_metrics.get("val_pc2r_raw_correction_pixel_count", 0.0),
                                validation_metrics.get("val_pc2r_risk_support_pixel_count", 0.0),
                                validation_metrics.get("val_geotr_c2r_canonical_mean_gain", 0.0),
                                validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            )
                    if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False)) and not bool(_cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)):
                        logger.info(
                            "VAL_PC2R_ROOTPP epoch=%03d | AllCenterBias logit abs/signed=%.5f/%+.5f probAbs=%.5f | "
                            "Selected probAbs=%.5f SelVsAllLogitAbs=%.5f | BranchDevRMS=%.5f DirUnanimity=%.4f",
                            epoch + 1,
                            validation_metrics.get("val_pc2r_all_center_bias_logit_abs", 0.0),
                            validation_metrics.get("val_pc2r_all_center_bias_logit_signed", 0.0),
                            validation_metrics.get("val_pc2r_all_center_bias_prob_abs", 0.0),
                            validation_metrics.get("val_pc2r_selected_center_bias_prob_abs", 0.0),
                            validation_metrics.get("val_pc2r_selected_vs_all_logit_bias_abs", 0.0),
                            validation_metrics.get("val_pc2r_branch_deviation_rms", 0.0),
                            validation_metrics.get("val_pc2r_branch_direction_unanimity", 0.0),
                        )
                    if bool(_cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)) and "val_geotr_c2r_stage2_gain" in validation_metrics:
                        logger.info(
                            "VAL_AEFR epoch=%03d | stage=%s jointGeo=%s | ROI=%.4f ErrRecall=%.4f Context/CandidateOracle=%+.6f/%+.6f | "
                            "Stage2=%+.6f Harm=%.4f EditP=%.4f Net=%+.0f | "
                            "PostStability=%.4f dStability=%+.4f DisPre/Post=%.4f/%.4f Support=%.4f | "
                            "Reach=%.4f ReachOracle=%+.6f Bnd<=3/5=%.4f/%.4f Bnd5/IntOracle=%+.6f/%+.6f | "
                            "BndFrac=%.4f |Disp|=%.4f |InteriorDz|=%.4f",
                            epoch + 1,
                            str(_cfg_get(cfg.M1, "GEOTR_AEFR_STAGE", "single_atomic")),
                            bool(_cfg_get(cfg.M1, "GEOTR_AEFR_JOINT_GEOMETRY_GRAD_ENABLED", False)),
                            validation_metrics.get("val_geotr_c2r_region_coverage", 0.0),
                            validation_metrics.get("val_geotr_c2r_region_error_recall", 0.0),
                            validation_metrics.get("native_c2r_context_oracle_gain", 0.0),
                            validation_metrics.get("native_c2r_candidate_oracle_gain", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_geotr_stage2_harmful_case_rate", 0.0),
                            validation_metrics.get("val_geotr_c2r_edit_precision", 0.0),
                            validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                            validation_metrics.get("val_geotr_aefr_posterior_stability_support", 0.0),
                            validation_metrics.get("val_geotr_aefr_posterior_stability_improvement", 0.0),
                            validation_metrics.get("val_geotr_aefr_posterior_disagreement_pre", 0.0),
                            validation_metrics.get("val_geotr_aefr_posterior_disagreement_post", 0.0),
                            validation_metrics.get("val_geotr_aefr_action_support_fraction", 0.0),
                            validation_metrics.get("val_pc2r_reachable_region_error_fraction", 0.0),
                            validation_metrics.get("val_pc2r_reachable_oracle_gain", 0.0),
                            validation_metrics.get("val_pc2r_error_near_boundary_3_fraction", 0.0),
                            validation_metrics.get("val_pc2r_error_near_boundary_5_fraction", 0.0),
                            validation_metrics.get("val_pc2r_boundary5_oracle_gain", 0.0),
                            validation_metrics.get("val_pc2r_interior_oracle_gain", 0.0),
                            validation_metrics.get("val_geotr_aefr_boundary_fraction_in_region", 0.0),
                            validation_metrics.get("val_geotr_aefr_mean_abs_boundary_displacement_px", 0.0),
                            validation_metrics.get("val_geotr_aefr_mean_abs_interior_delta_logit", 0.0),
                        )
                        if str(_cfg_get(cfg.M1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower() in {"typed_state_taylor", "typed_state_exact"}:
                            logger.info(
                                "VAL_AEFR_STATE epoch=%03d | MacroF1=%.4f PredEdit=%.4f | "
                                "KEEP P/R/F1=%.4f/%.4f/%.4f | BADD=%.4f/%.4f/%.4f BREM=%.4f/%.4f/%.4f | "
                                "IADD=%.4f/%.4f/%.4f IREM=%.4f/%.4f/%.4f",
                                epoch + 1,
                                validation_metrics.get("val_geotr_aefr_state_macro_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_pred_edit_fraction", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_keep_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_keep_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_keep_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_add_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_add_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_add_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_remove_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_remove_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_boundary_remove_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_add_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_add_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_add_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_remove_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_remove_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_state_interior_remove_f1", 0.0),
                            )
                        if str(_cfg_get(cfg.M1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower() == "soft_ownership_exact":
                            logger.info(
                                "VAL_AEFR_OWN epoch=%03d | MacroF1=%.4f PredEdit=%.4f | EditMass pred/target/absErr=%.4f/%.4f/%.4f | "
                                "SignedMAE=%.4f TargetAbs=%.4f | KEEP P/R/F1=%.4f/%.4f/%.4f | "
                                "ADD P/R/F1=%.4f/%.4f/%.4f REMOVE P/R/F1=%.4f/%.4f/%.4f",
                                epoch + 1,
                                validation_metrics.get("val_geotr_aefr_ownership_macro_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_pred_edit_fraction", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_pred_edit_mass", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_target_edit_mass", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_edit_mass_abs_error", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_signed_action_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_signed_target_abs_mean", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_keep_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_keep_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_keep_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_add_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_add_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_add_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_remove_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_remove_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_ownership_remove_f1", 0.0),
                            )
                        _aefr_stage_for_log = str(_cfg_get(cfg.M1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower()
                        if _aefr_stage_for_log == "intervention_factorized":
                            logger.info(
                                "VAL_AEFR_IFR epoch=%03d | ErrorAP=%.4f R@5/10=%.4f/%.4f | "
                                "Localizer P/R/F1=%.4f/%.4f/%.4f | Edit P/R/F1=%.4f/%.4f/%.4f DirAcc=%.4f | "
                                "ADD F1=%.4f REMOVE F1=%.4f | SignedMAE/ZeroMAE/Adv=%.4f/%.4f/%+.4f | "
                                "Perr/Pok=%.4f/%.4f EditErr/EditOk=%.4f/%.4f",
                                epoch + 1,
                                validation_metrics.get("val_geotr_aefr_intervention_error_ap", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_recall_at_05", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_recall_at_10", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_direction_accuracy", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_add_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_remove_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_signed_action_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_zero_action_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_signed_advantage", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_prob_mean_error", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_prob_mean_correct", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_prob_mean_error_roi", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_prob_mean_correct_roi", 0.0),
                            )
                        elif _aefr_stage_for_log == "selective_minimal_intervention":
                            logger.info(
                                "VAL_AEFR_SMI epoch=%03d | ErrorAP=%.4f R@5/10=%.4f/%.4f | "
                                "Localizer P/R/F1=%.4f/%.4f/%.4f | Commit P/R/F1=%.4f/%.4f/%.4f DirAcc=%.4f | "
                                "ADD F1=%.4f REMOVE F1=%.4f | SignedMAE/ZeroMAE/Adv=%.4f/%.4f/%+.4f | "
                                "Perr/Pok=%.4f/%.4f CommitErr/CommitOk=%.4f/%.4f | BMagMAE=%.4f Pred/Target=%.4f/%.4f",
                                epoch + 1,
                                validation_metrics.get("val_geotr_aefr_intervention_error_ap", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_recall_at_05", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_recall_at_10", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_localizer_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_precision", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_recall", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_direction_accuracy", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_add_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_remove_f1", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_signed_action_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_zero_action_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_signed_advantage", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_prob_mean_error", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_error_prob_mean_correct", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_prob_mean_error_roi", 0.0),
                                validation_metrics.get("val_geotr_aefr_intervention_edit_prob_mean_correct_roi", 0.0),
                                validation_metrics.get("val_geotr_aefr_smi_boundary_mag_mae", 0.0),
                                validation_metrics.get("val_geotr_aefr_smi_boundary_mag_pred_mean", 0.0),
                                validation_metrics.get("val_geotr_aefr_smi_boundary_mag_target_mean", 0.0),
                            )
                        elif _aefr_stage_for_log == "sparse_local_rerendering":
                            _slr_real_num = float(validation_metrics.get("val_geotr_slr_context_realization_num", 0.0))
                            _slr_real_den = abs(float(validation_metrics.get("val_geotr_slr_context_realization_den", 0.0)))
                            _slr_real = _slr_real_num / max(_slr_real_den, 1.0e-8)
                            logger.info(
                                "VAL_AEFR_SLR epoch=%03d | SupportAP=%.4f ValueMAE=%.4f ValueCorr=%.4f | "
                                "RegionRecall=%.4f ChangedP/R=%.4f/%.4f | ContextOracle=%+.6f Realization=%+.4f | "
                                "SoftStage2=%+.6f NativeStage2=%+.6f Harm=%.4f Net=%+.0f | BlendOverlap=%.4f BlendW=%.4f SDFBandMAE=%.4f | RawPatchGain=%+.5f DeployPatchGain=%+.5f OverlapDis=%.5f | ActionRecall=%.4f UnreachSup=%.5f PosFromSel=%.4f | SDFSignAcc=%.4f SDFZeroDice=%.4f | AnchorSDFMAE=%.4f SDFGain=%+.4f BndChangedP/R=%.4f/%.4f InteriorChange=%.5f | ActorDir=%.4f DoseReach=%.4f DoseRatio=%.4f ActionAbs=%.5f",
                                epoch + 1,
                                validation_metrics.get("val_geotr_slr_selector_error_ap", 0.0),
                                validation_metrics.get("val_geotr_slr_selector_value_mae", 0.0),
                                validation_metrics.get("val_geotr_slr_selector_value_corr", 0.0),
                                validation_metrics.get("val_geotr_v4g_selection_recall", 0.0),
                                validation_metrics.get("val_geotr_slr_changed_precision", 0.0),
                                validation_metrics.get("val_geotr_slr_changed_recall", 0.0),
                                validation_metrics.get("val_geotr_pc2r_native_context_oracle_gain", validation_metrics.get("val_geotr_v4g_pred_selector_oracle_gain", 0.0)),
                                _slr_real,
                                validation_metrics.get("val_geotr_v4g_modelres_soft_gain", 0.0),
                                validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                                validation_metrics.get("native_m2_harmful_case_rate", validation_metrics.get("val_geotr_v4g_harm_case_rate", 0.0)),
                                validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                                validation_metrics.get("val_geotr_slr_blend_overlap_fraction", 0.0),
                                validation_metrics.get("val_geotr_slr_blend_weight_mean", 0.0),
                                validation_metrics.get("val_geotr_slr_sdf_band_mae", 0.0),
                                validation_metrics.get("val_geotr_slr_raw_patch_dice_gain", 0.0),
                                validation_metrics.get("val_geotr_slr_deploy_patch_dice_gain", validation_metrics.get("val_geotr_slr_pred_patch_dice_gain", 0.0)),
                                validation_metrics.get("val_geotr_slr_overlap_disagreement", 0.0),
                                validation_metrics.get("val_geotr_slr_actionable_selector_recall", 0.0),
                                validation_metrics.get("val_geotr_slr_unreachable_supervision_fraction", 0.0),
                                validation_metrics.get("val_geotr_slr_positive_from_selector_fraction", 0.0),
                                validation_metrics.get("val_geotr_slr_sdf_sign_acc", 0.0),
                                validation_metrics.get("val_geotr_slr_sdf_zero_cross_dice", 0.0),
                                validation_metrics.get("val_geotr_slr_anchor_sdf_band_mae", 0.0),
                                validation_metrics.get("val_geotr_slr_sdf_gain", 0.0),
                                validation_metrics.get("val_geotr_slr_boundary_changed_precision", 0.0),
                                validation_metrics.get("val_geotr_slr_boundary_changed_recall", 0.0),
                                validation_metrics.get("val_geotr_slr_interior_change_rate", 0.0),
                                validation_metrics.get("val_geotr_slr_actor_direction_accuracy", 0.0),
                                validation_metrics.get("val_geotr_slr_actor_dose_reachability", 0.0),
                                validation_metrics.get("val_geotr_slr_actor_dose_ratio", 0.0),
                                validation_metrics.get("val_geotr_slr_actor_action_abs_mean", 0.0),
                            )
                            if bool(_cfg_get(cfg.M1, "GEOTR_SLR_UCDRT_ENABLED", False)) or bool(_cfg_get(cfg.M1, "GEOTR_SLR_UCDRT_R2_ENABLED", False)):
                                logger.info(
                                    "VAL_UCDRT epoch=%03d | StateF1=%.4f KEEP/MOVE/ADD/REMOVE=%.4f/%.4f/%.4f/%.4f | "
                                    "Utility MAE/Corr=%.5f/%+.4f CommitRate/Precision=%.4f/%.4f | "
                                    "MoveMAE=%.4f InteriorMAE=%.4f | PatchOutsideAction=%.5f TrainSupOutsideAction=%.5f | "
                                    "ReachFrac=%.4f ReachOracle=%+.6f | NativeStage2=%+.6f Harm=%.4f",
                                    epoch + 1,
                                    validation_metrics.get("val_geotr_slr_ucdrt_state_macro_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_keep_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_move_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_add_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_remove_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_utility_mae", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_utility_corr", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_commit_rate", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_commit_precision", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_move_mae", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_interior_mae", 0.0),
                                    validation_metrics.get("val_geotr_slr_patch_outside_action_fraction", 0.0),
                                    validation_metrics.get("val_geotr_slr_unreachable_supervision_fraction", 0.0),
                                    validation_metrics.get("val_pc2r_reachable_region_error_fraction", 0.0),
                                    validation_metrics.get("val_pc2r_reachable_oracle_gain", 0.0),
                                    validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                                    validation_metrics.get("native_m2_harmful_case_rate", validation_metrics.get("val_geotr_v4g_harm_case_rate", 0.0)),
                                )

                            if bool(_cfg_get(cfg.M1, "GEOTR_SLR_UCDRT_R2_ENABLED", False)):
                                logger.info(
                                    "VAL_UCDRT_R2 epoch=%03d | Edit P/R/F1=%.4f/%.4f/%.4f Target/PredEdit=%.4f/%.4f | "
                                    "TypeF1 M/A/R=%.4f %.4f/%.4f/%.4f | HardGain mean/pos/oracle=%+.5f/%.4f/%+.5f "
                                    "SoftGain mean/pos/oracle=%+.5f/%.4f/%+.5f HSGap=%+.5f | "
                                    "Critic P/R/Pos=%.4f/%.4f/%.4f MAE/Corr=%.5f/%+.4f | "
                                    "SetGain pred/oracle/real=%+.5f/%+.5f/%+.4f | NativeStage2=%+.6f Harm=%.4f",
                                    epoch + 1,
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_edit_precision", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_edit_recall", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_edit_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_target_edit_fraction", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_pred_edit_fraction", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_type_macro_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_move_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_add_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_remove_f1", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_hard_mean_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_hard_positive_rate", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_hard_oracle_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_soft_mean_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_soft_positive_rate", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_soft_oracle_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_hard_soft_gap", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_critic_precision", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_critic_recall", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_critic_positive_rate", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_utility_mae", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_utility_corr", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_set_predicted_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_set_oracle_gain", 0.0),
                                    validation_metrics.get("val_geotr_slr_ucdrt_r2_set_realization", 0.0),
                                    validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                                    validation_metrics.get("native_m2_harmful_case_rate", validation_metrics.get("val_geotr_v4g_harm_case_rate", 0.0)),
                                )
                        logger.info(
                            "VAL_AEFR_TRANSITION epoch=%03d | aware=%s rawFlow=%s | dGAbs=%.5f Active=%.4f Flip=%.4f | "
                            "Gfix/Gharm/Gmiss/Gkeep=%.0f/%.0f/%.0f/%.0f | FixPreserve/Break=%.4f/%.4f HarmRepair=%.4f MissRepair=%.4f KeepBreak=%.4f | "
                            "Edit fix/harm/miss/keep=%.4f/%.4f/%.4f/%.4f | AbsDz fix/harm/miss/keep=%.4f/%.4f/%.4f/%.4f",
                            epoch + 1,
                            bool(_cfg_get(cfg.M1, "GEOTR_AEFR_TRANSITION_AWARE_ENABLED", False)),
                            bool(_cfg_get(cfg.M1, "GEOTR_AEFR_USE_RAW_FLOW_EVIDENCE", True)),
                            validation_metrics.get("val_geotr_aefr_transition_abs_mean", 0.0),
                            validation_metrics.get("val_geotr_aefr_transition_active_fraction", 0.0),
                            validation_metrics.get("val_geotr_aefr_transition_flip_fraction", 0.0),
                            validation_metrics.get("val_geotr_aefr_gfix_count", 0.0),
                            validation_metrics.get("val_geotr_aefr_gharm_count", 0.0),
                            validation_metrics.get("val_geotr_aefr_gmiss_count", 0.0),
                            validation_metrics.get("val_geotr_aefr_gkeep_count", 0.0),
                            validation_metrics.get("val_geotr_aefr_gfix_preserve_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gfix_break_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gharm_repair_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gmiss_repair_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gkeep_break_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gfix_edit_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gharm_edit_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gmiss_edit_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gkeep_edit_rate", 0.0),
                            validation_metrics.get("val_geotr_aefr_gfix_action_abs_dz", 0.0),
                            validation_metrics.get("val_geotr_aefr_gharm_action_abs_dz", 0.0),
                            validation_metrics.get("val_geotr_aefr_gmiss_action_abs_dz", 0.0),
                            validation_metrics.get("val_geotr_aefr_gkeep_action_abs_dz", 0.0),
                        )
                    if "val_geotr_v4g_r3_pred_abs_delta" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4G_R3 epoch=%03d | Delta abs all/error/keep=%.6f/%.6f/%.6f | "
                            "ErrorSignAcc=%.4f EmpErrFracSelected=%.4f | NativeStage2=%+.6f | Corr/Intro/Net=%.0f/%.0f/%+.0f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4g_r3_pred_abs_delta", 0.0),
                            validation_metrics.get("val_geotr_v4g_r3_error_pred_abs_delta", 0.0),
                            validation_metrics.get("val_geotr_v4g_r3_keep_pred_abs_delta", 0.0),
                            validation_metrics.get("val_geotr_v4g_r3_error_sign_accuracy", 0.0),
                            validation_metrics.get("val_geotr_v4g_r3_empirical_error_fraction_selected", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("val_geotr_v4g_corrected_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_introduced_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                        )
                    if "val_geotr_v4g_r4_flip_probability_mean" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4G_R4 epoch=%03d | FlipProb all/error/correct=%.4f/%.4f/%.4f | "
                            "Flip P/Rsel/FalseFlip=%.4f/%.4f/%.4f EditRate=%.4f | "
                            "Sel micro P/R=%.4f/%.4f macroR=%.4f | NativeStage2=%+.6f Soft=%+.6f | "
                            "Corr/Intro/Net=%.0f/%.0f/%+.0f Harm=%.4f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4g_r4_flip_probability_mean", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_flip_probability_error", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_flip_probability_correct", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_flip_precision", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_flip_recall_selected", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_false_flip_rate", 0.0),
                            validation_metrics.get("val_geotr_v4g_r4_effective_edit_rate", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_precision", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_recall", 0.0),
                            validation_metrics.get("val_geotr_v4g_selection_recall_macro_batch", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("val_geotr_v4g_modelres_soft_gain", 0.0),
                            validation_metrics.get("val_geotr_v4g_corrected_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_introduced_error_count", 0.0),
                            validation_metrics.get("val_geotr_v4g_net_correction_count", 0.0),
                            validation_metrics.get("native_geotr_stage2_harmful_case_rate", 0.0),
                        )
                    if "val_geotr_v4f_modelres_hard_gain" in validation_metrics:
                        _tau_keys = [30, 40, 50, 60, 65, 70, 80, 90]
                        _curve = " ".join(
                            f"t{t:02d}:{validation_metrics.get(f'val_geotr_v4f_tau{t:02d}_gain', 0.0):+.4f}/P{validation_metrics.get(f'val_geotr_v4f_tau{t:02d}_precision', 0.0):.2f}/R{validation_metrics.get(f'val_geotr_v4f_tau{t:02d}_recall', 0.0):.2f}"
                            for t in _tau_keys
                        )
                        logger.info(
                            "VAL_GEOTR_V4F epoch=%03d | Soft/Hard=%+.6f/%+.6f TeacherDose=%+.6f | "
                            "Proposal rate/P/R=%.4f/%.4f/%.4f | Exec rate/RepairP/DirP/R=%.4f/%.4f/%.4f/%.4f | "
                            "Dose true/false=%.4f/%.4f | FalseSuppress=%+.6f GTSupportDose=%+.6f | NativeStage2=%+.6f NativeBenefit/Harm=%.4f/%.4f | Curve %s",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4f_modelres_soft_gain", 0.0),
                            validation_metrics.get("val_geotr_v4f_modelres_hard_gain", 0.0),
                            validation_metrics.get("val_geotr_v4f_teacher_dose_gain", 0.0),
                            validation_metrics.get("val_geotr_v4f_candidate_rate", 0.0),
                            validation_metrics.get("val_geotr_v4f_proposal_precision", 0.0),
                            validation_metrics.get("val_geotr_v4f_proposal_recall", 0.0),
                            validation_metrics.get("val_geotr_v4f_execution_rate", 0.0),
                            validation_metrics.get("val_geotr_v4f_execution_precision_repair", 0.0),
                            validation_metrics.get("val_geotr_v4f_execution_precision_direction", 0.0),
                            validation_metrics.get("val_geotr_v4f_execution_recall", 0.0),
                            validation_metrics.get("val_geotr_v4f_true_action_dose", 0.0),
                            validation_metrics.get("val_geotr_v4f_false_action_dose", 0.0),
                            validation_metrics.get("val_geotr_v4f_false_action_suppression_gain", 0.0),
                            validation_metrics.get("val_geotr_v4f_gt_support_pred_dose_gain", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_geotr_stage2_beneficial_case_rate", 0.0),
                            validation_metrics.get("native_geotr_stage2_harmful_case_rate", 0.0),
                            _curve,
                        )
                    if "val_geotr_v4e_stage2_soft_gain" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4E epoch=%03d | SoftStage2=%+.6f TeacherHOW=%+.6f | "
                            "SoftBenefit/Harm=%.4f/%.4f | TypedF1=%.4f (FN=%.4f FP=%.4f) | "
                            "Support true/pred=%.4f/%.4f P/R=%.4f/%.4f | "
                            "Edit hard/margin/conf=%.6f/%.6f/%.6f | MagMAE FN/FP=%.4f/%.4f | "
                            "OffMag add/rem=%.4f/%.4f | Cap pred/targetFN/targetFP=%.4f/%.4f/%.4f | "
                            "OracleSupport/Mag/Full=%+.6f/%+.6f/%+.6f | NativeStage2=%+.6f NativeBenefit/Harm=%.4f/%.4f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4e_stage2_soft_gain", 0.0),
                            validation_metrics.get("val_geotr_v4e_teacher_how_gain", 0.0),
                            validation_metrics.get("val_geotr_v4e_stage2_soft_benefit_rate", 0.0),
                            validation_metrics.get("val_geotr_v4e_stage2_soft_harm_rate", 0.0),
                            validation_metrics.get("val_geotr_v4e_global_typed_macro_f1", 0.0),
                            validation_metrics.get("val_geotr_v4e_fn_f1", 0.0),
                            validation_metrics.get("val_geotr_v4e_fp_f1", 0.0),
                            validation_metrics.get("val_geotr_v4e_true_support_rate", 0.0),
                            validation_metrics.get("val_geotr_v4e_predicted_support_rate", 0.0),
                            validation_metrics.get("val_geotr_v4e_support_precision", 0.0),
                            validation_metrics.get("val_geotr_v4e_support_recall", 0.0),
                            validation_metrics.get("val_geotr_v4e_edit_hard_error_mean", 0.0),
                            validation_metrics.get("val_geotr_v4e_edit_margin_correct_mean", 0.0),
                            validation_metrics.get("val_geotr_v4e_edit_confident_correct_mean", 0.0),
                            validation_metrics.get("val_geotr_v4e_magnitude_fn_mae", 0.0),
                            validation_metrics.get("val_geotr_v4e_magnitude_fp_mae", 0.0),
                            validation_metrics.get("val_geotr_v4e_offsupport_add_mean", 0.0),
                            validation_metrics.get("val_geotr_v4e_offsupport_remove_mean", 0.0),
                            validation_metrics.get("val_geotr_v4e_predicted_magnitude_cap_fraction", 0.0),
                            validation_metrics.get("val_geotr_v4e_target_fn_cap_fraction", 0.0),
                            validation_metrics.get("val_geotr_v4e_target_fp_cap_fraction", 0.0),
                            validation_metrics.get("val_geotr_v4e_oracle_support_gain", 0.0),
                            validation_metrics.get("val_geotr_v4e_oracle_magnitude_gain", 0.0),
                            validation_metrics.get("val_geotr_v4e_oracle_full_gain", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            validation_metrics.get("native_geotr_stage2_beneficial_case_rate", 0.0),
                            validation_metrics.get("native_geotr_stage2_harmful_case_rate", 0.0),
                        )
                    if "val_geotr_v4d_stage2_gain" in validation_metrics:
                        logger.info(
                            "VAL_GEOTR_V4D epoch=%03d | Stage2Gain=%+.6f | TeacherHOW=%+.6f | "
                            "Benefit/Harm=%.4f/%.4f | MeanBenefit/Harm=%+.6f/%+.6f | "
                            "GlobalTypedF1=%.4f (FN=%.4f FP=%.4f) | "
                            "SeverityMass(error/correct)=%.4f/%.4f softP=%.4f | "
                            "SeverityMAE(FN/FP)=%.4f/%.4f | MagMAE(FN/FP)=%.4f/%.4f | "
                            "EditPrecision=%.4f EditErr/Correct=%.6f/%.6f ratio=%.2f | "
                            "OracleWhere/Typed/TargetMag=%+.6f/%+.6f/%+.6f",
                            epoch + 1,
                            validation_metrics.get("val_geotr_v4d_stage2_gain", 0.0),
                            validation_metrics.get("val_geotr_v4d_teacher_how_gain", 0.0),
                            validation_metrics.get("val_geotr_v4d_stage2_benefit_rate", 0.0),
                            validation_metrics.get("val_geotr_v4d_stage2_harm_rate", 0.0),
                            validation_metrics.get("val_geotr_v4d_stage2_mean_benefit", 0.0),
                            validation_metrics.get("val_geotr_v4d_stage2_mean_harm", 0.0),
                            validation_metrics.get("val_geotr_v4d_global_typed_macro_f1", 0.0),
                            validation_metrics.get("val_geotr_v4d_global_fn_f1", 0.0),
                            validation_metrics.get("val_geotr_v4d_global_fp_f1", 0.0),
                            validation_metrics.get("val_geotr_v4d_severity_true_error_mass", 0.0),
                            validation_metrics.get("val_geotr_v4d_severity_correct_mass", 0.0),
                            validation_metrics.get("val_geotr_v4d_severity_soft_precision", 0.0),
                            validation_metrics.get("val_geotr_v4d_severity_fn_mae", 0.0),
                            validation_metrics.get("val_geotr_v4d_severity_fp_mae", 0.0),
                            validation_metrics.get("val_geotr_v4d_magnitude_fn_mae", 0.0),
                            validation_metrics.get("val_geotr_v4d_magnitude_fp_mae", 0.0),
                            validation_metrics.get("val_geotr_v4d_edit_precision", 0.0),
                            validation_metrics.get("val_geotr_v4d_edit_error_mean", 0.0),
                            validation_metrics.get("val_geotr_v4d_edit_correct_mean", 0.0),
                            validation_metrics.get("val_geotr_v4d_edit_error_correct_ratio", 0.0),
                            validation_metrics.get("val_geotr_v4d_oracle_where_gain", 0.0),
                            validation_metrics.get("val_geotr_v4d_oracle_typed_gain", 0.0),
                            validation_metrics.get("val_geotr_v4d_oracle_target_magnitude_gain", 0.0),
                        )
                    if bool(_cfg_get(cfg.M1, "GEOTR_PC2R_ROOT_AUTOPSY_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False) or _cfg_get(cfg.M1, "GEOTR_AEFR_ENABLED", False)):
                        logger.info(
                            "VAL_PC2R_STAGE2_SAFETY epoch=%03d | BenefitRate=%.6f | HarmRate=%.6f | "
                            "MeanBenefit=%+.6f | MeanHarm=%+.6f | Harm/Benefit=%.6f | "
                            "NativeStage2=%+.6f qualifies=%s",
                            epoch + 1,
                            validation_metrics.get("native_m2_beneficial_case_rate", 0.0),
                            validation_metrics.get("native_m2_harmful_case_rate", 0.0),
                            validation_metrics.get("native_m2_mean_positive_gain", 0.0),
                            validation_metrics.get("native_m2_mean_harmful_change", 0.0),
                            validation_metrics.get("native_m2_harm_to_benefit_ratio", 0.0),
                            validation_metrics.get("native_geotr_stage2_gain_vs_geometry", 0.0),
                            qualifies,
                        )
                    else:
                        logger.info(
                            "VAL_MHCS_R48_SAFETY epoch=%03d | BaseTail=%.6f | M2Tail=%.6f | "
                            "BenefitRate=%.6f | HarmRate=%.6f | MeanBenefit=%+.6f | "
                            "MeanHarm=%+.6f | Harm/Benefit=%.6f | qualifies=%s",
                            epoch + 1,
                            validation_metrics.get("native_base_dice_tail_score", 0.0),
                            validation_metrics.get("native_m2_dice_tail_score", 0.0),
                            validation_metrics.get("native_m2_beneficial_case_rate", 0.0),
                            validation_metrics.get("native_m2_harmful_case_rate", 0.0),
                            validation_metrics.get("native_m2_mean_positive_gain", 0.0),
                            validation_metrics.get("native_m2_mean_harmful_change", 0.0),
                            validation_metrics.get("native_m2_harm_to_benefit_ratio", 0.0),
                            qualifies,
                        )

                if _v488_is_m2m3_only(cfg) or _v489_or_v490_end_to_end(cfg):
                    logger.info(
                        "V488_V489_VAL epoch=%03d | PWO DSC/NSD=%.6f/%.6f | "
                        "M2 DSC/NSD=%.6f/%.6f (gain=%.6f) | "
                        "M3 DSC/NSD=%.6f/%.6f (gain_vs_M2=%.6f) | "
                        "gap_to_PWO M2/M3=%.6f/%.6f",
                        epoch + 1,
                        validation_metrics.get("pwo_dice", 0.0),
                        validation_metrics.get("pwo_nsd", 0.0),
                        validation_metrics.get("m2_dice", 0.0),
                        validation_metrics.get("m2_nsd", 0.0),
                        validation_metrics.get("m2_gain", 0.0),
                        validation_metrics.get("fusion_dice", 0.0),
                        validation_metrics.get("fusion_nsd", 0.0),
                        validation_metrics.get("m3_gain_vs_m2", 0.0),
                        validation_metrics.get("m2_gap_to_pwo", 0.0),
                        validation_metrics.get("final_gap_to_pwo", 0.0),
                    )
                if _semlt(cfg) and "native_fusion_score" in validation_metrics:
                    exact_m1_log = str(_cfg_get(cfg.M1, "PROTOCOL", "")).strip().lower() == "geotr_m1"
                    logger.info(
                        "%s epoch=%03d | Base DSC/NSD=%.6f/%.6f | "
                        "%s DSC/NSD=%.6f/%.6f | gain=%+.6f/%+.6f | "
                        "selected_metric(%s)=%.6f | qualifies=%s | M2/M3=absent",
                        "VAL_GEOTR_M1" if exact_m1_log else "VAL_SEMLT",
                        epoch + 1,
                        validation_metrics["native_base_dice"],
                        validation_metrics["native_base_nsd"],
                        "M1 Transport" if exact_m1_log else "SemLT",
                        validation_metrics.get("native_m1_dice", validation_metrics["native_base_dice"]),
                        validation_metrics.get("native_m1_nsd", validation_metrics["native_base_nsd"]),
                        validation_metrics.get("native_m1_dice", validation_metrics["native_base_dice"])
                        - validation_metrics["native_base_dice"],
                        validation_metrics.get("native_m1_nsd", validation_metrics["native_base_nsd"])
                        - validation_metrics["native_base_nsd"],
                        metric_name,
                        selection_value,
                        qualifies,
                    )
                if (not _semlt(cfg)) and "native_fusion_score" in validation_metrics:
                    logger.info(
                        "VAL_NATIVE epoch=%03d | base DSC/NSD=%.6f/%.6f | "
                        "M1Native DSC/NSD=%.6f/%.6f | "
                        "M2 DSC/NSD=%.6f/%.6f | fusion(M3) DSC/NSD=%.6f/%.6f | "
                        "actionOracle DSC/NSD=%.6f/%.6f | "
                        "componentOracle DSC/NSD=%.6f/%.6f | "
                        "native_score=%.6f | selected_metric(%s)=%.6f | qualifies=%s",
                        epoch + 1,
                        validation_metrics["native_base_dice"],
                        validation_metrics["native_base_nsd"],
                        validation_metrics.get("native_m1_dice", validation_metrics["native_base_dice"]),
                        validation_metrics.get("native_m1_nsd", validation_metrics["native_base_nsd"]),
                        validation_metrics.get("native_m2_dice", validation_metrics["native_fusion_dice"]),
                        validation_metrics.get("native_m2_nsd", validation_metrics["native_fusion_nsd"]),
                        validation_metrics["native_fusion_dice"],
                        validation_metrics["native_fusion_nsd"],
                        validation_metrics["native_oracle_dice"],
                        validation_metrics["native_oracle_nsd"],
                        validation_metrics.get(
                            "native_component_oracle_dice",
                            validation_metrics["native_oracle_dice"],
                        ),
                        validation_metrics.get(
                            "native_component_oracle_nsd",
                            validation_metrics["native_oracle_nsd"],
                        ),
                        validation_metrics["native_fusion_score"],
                        metric_name,
                        selection_value,
                        qualifies,
                    )
                    m1_target_dsc = float(_cfg_get(cfg.M1, "V552R4209_M1_TARGET_DSC", 0.86))
                    m1_base_dsc = validation_metrics["native_base_dice"]
                    m1_native_dsc = validation_metrics.get("native_m1_dice", m1_base_dsc)
                    m1_oracle_dsc = validation_metrics.get(
                        "native_component_oracle_dice",
                        validation_metrics["native_oracle_dice"],
                    )
                    required_gain = max(m1_target_dsc - m1_base_dsc, 0.0)
                    available_student_gain = max(m1_oracle_dsc - m1_base_dsc, 0.0)
                    realization = (
                        max(m1_native_dsc - m1_base_dsc, 0.0) / max(available_student_gain, 1.0e-12)
                        if available_student_gain > 0.0 else 0.0
                    )
                    logger.info(
                        "V552R4209_M1_GOAL epoch=%03d | targetDSC=%.6f | "
                        "Base=%.6f requiredGain=%.6f | M1Native=%.6f gap=%+.6f | "
                        "ComponentOracle=%.6f oracleGap=%+.6f | realization=%.6f | goalReached=%s",
                        epoch + 1, m1_target_dsc, m1_base_dsc, required_gain,
                        m1_native_dsc, m1_native_dsc - m1_target_dsc,
                        m1_oracle_dsc, m1_oracle_dsc - m1_target_dsc,
                        realization, m1_native_dsc >= m1_target_dsc,
                    )
                    if bool(_cfg_get(cfg.M1, "V552R4210_ROOTFIX_ENABLED", False)):
                        m1_target_4210 = float(_cfg_get(cfg.M1, "V552R4210_M1_TARGET_DSC", m1_target_dsc))
                        native_gain_4210 = m1_native_dsc - m1_base_dsc
                        oracle_gain_4210 = m1_oracle_dsc - m1_base_dsc
                        logger.info(
                            "V552R4210_M1_GOAL epoch=%03d | targetDSC=%.6f | "
                            "Base=%.6f | M1Native=%.6f nativeGain=%+.6f improvesBase=%s | "
                            "ComponentOracle=%.6f oracleGain=%+.6f | realization=%.6f | targetReached=%s",
                            epoch + 1, m1_target_4210, m1_base_dsc, m1_native_dsc,
                            native_gain_4210, native_gain_4210 > 0.0,
                            m1_oracle_dsc, oracle_gain_4210, realization,
                            m1_native_dsc >= m1_target_4210,
                        )
                    if bool(_cfg_get(cfg.M1, "V552R4211_ROOTFIX_ENABLED", False)):
                        m1_target_4211 = float(_cfg_get(cfg.M1, "V552R4211_M1_TARGET_DSC", m1_target_dsc))
                        native_gain_4211 = m1_native_dsc - m1_base_dsc
                        oracle_gain_4211 = m1_oracle_dsc - m1_base_dsc
                        logger.info(
                            "V552R4211_M1_GOAL epoch=%03d | targetDSC=%.6f | "
                            "Base=%.6f | M1Native=%.6f nativeGain=%+.6f improvesBase=%s | "
                            "ComponentOracle=%.6f oracleGain=%+.6f | realization=%.6f | targetReached=%s",
                            epoch + 1, m1_target_4211, m1_base_dsc, m1_native_dsc,
                            native_gain_4211, native_gain_4211 > 0.0,
                            m1_oracle_dsc, oracle_gain_4211, realization,
                            m1_native_dsc >= m1_target_4211,
                        )
                    if bool(_cfg_get(cfg.M1, "V552R4212_ROOTFIX_ENABLED", False)):
                        m1_target_4212 = float(_cfg_get(cfg.M1, "V552R4212_M1_TARGET_DSC", m1_target_dsc))
                        native_gain_4212 = m1_native_dsc - m1_base_dsc
                        oracle_gain_4212 = m1_oracle_dsc - m1_base_dsc
                        logger.info(
                            "V552R4212_M1_GOAL epoch=%03d | stage=%d | targetDSC=%.6f | "
                            "Base=%.6f | M1Native=%.6f nativeGain=%+.6f improvesBase=%s | "
                            "ComponentOracle=%.6f oracleGain=%+.6f | realization=%.6f | targetReached=%s",
                            epoch + 1, int(_cfg_get(cfg.M1, "V552R4212_STAGE", 0)),
                            m1_target_4212, m1_base_dsc, m1_native_dsc, native_gain_4212,
                            native_gain_4212 > 0.0, m1_oracle_dsc, oracle_gain_4212,
                            realization, m1_native_dsc >= m1_target_4212,
                        )
                    logger.info(
                        "V546_SHADOW_VAL epoch=%03d | shadowM2 DSC/NSD=%.6f/%.6f "
                        "gain=%.6f/%.6f | cat=%.6f | selection_source=%s",
                        epoch + 1,
                        validation_metrics.get("native_shadow_m2_dice", 0.0),
                        validation_metrics.get("native_shadow_m2_nsd", 0.0),
                        validation_metrics.get("native_shadow_m2_gain", 0.0),
                        validation_metrics.get("native_shadow_m2_nsd_gain", 0.0),
                        validation_metrics.get(
                            "native_shadow_m2_dice_catastrophic_rate", 0.0
                        ),
                        "shadow" if use_shadow_selection else "deployed",
                    )
                    logger.info(
                        "V515_SELECTION_GATE epoch=%03d | eligible=%s | start_epoch=%03d",
                        epoch + 1, selection_eligible, selection_start_epoch_1based,
                    )
                if improved:
                    best_selection_value = selection_value
                    best_selection_tiebreak = selection_tiebreak
                    best_selection_epoch = epoch
                    best_state = _checkpoint_state(
                        model, optimizer, scheduler, epoch, best_dice,
                        best_fusion, best_oracle, run_name, cfg,
                        phase_b_started=False, ema=ema,
                        weight_source=selection_source,
                        hard_case_memory=hard_case_memory,
                    )
                    best_state.update({
                        "selection_protocol": (
                            "Best checkpoint selected on validation only; "
                            "Test was never opened during training."
                        ),
                        "validation_metrics": validation_metrics,
                        "best_selection_value": best_selection_value,
                        "best_selection_tiebreak": best_selection_tiebreak,
                        "best_selection_epoch": best_selection_epoch,
                        "best_native_base_dice": best_native_base_dice,
                        "best_native_base_nsd": best_native_base_nsd,
                        "best_native_base_catastrophic_rate": best_native_base_catastrophic_rate,
                        "selection_metric": metric_name,
                        "selection_tiebreak_metric": tiebreak_name,
                        "split_access_during_training": {
                            "train": True, "val": True, "test": False
                        },
                    })
                    torch.save(best_state, best_selection_path)
                    logger.info(
                        "Saved formal best-Val checkpoint: %s",
                        best_selection_path,
                    )

        # A strict no-harm selector can legitimately reject every deployed M2
        # checkpoint. For SPARC, create an explicit Preserve/M1 fallback from
        # the independently validation-selected M1 checkpoint. Falling all the
        # way back to Base would discard the validated Transport gain. Other
        # historical protocols retain their existing Base fallback semantics.
        if (
            use_validation_selection
            and epoch + 1 == cfg.TRAIN.NUM_EPOCHS
            and best_selection_epoch < 0
            and (
                os.path.isfile(best_m1_selection_path)
                or os.path.isfile(best_base_selection_path)
            )
        ):
            sparc_preserve_m1 = bool(
                _cfg_get(cfg.M1, "GEOTR_SPARC_HR_ENABLED", False)
            ) and os.path.isfile(best_m1_selection_path)
            fallback_source = (
                best_m1_selection_path if sparc_preserve_m1
                else best_base_selection_path
            )
            fallback_state = torch.load(fallback_source, map_location="cpu")
            fallback_epoch = int(fallback_state.get("epoch", -1))
            fallback_metrics = fallback_state.get("validation_metrics", {})
            fallback_dice_key = (
                "native_sparc_m1_hr_dice" if sparc_preserve_m1 else "native_base_dice"
            )
            fallback_nsd_key = (
                "native_sparc_m1_hr_nsd" if sparc_preserve_m1 else "native_base_nsd"
            )
            fallback_value = float(fallback_metrics.get(
                fallback_dice_key,
                fallback_metrics.get("native_base_dice", best_native_base_dice),
            ))
            fallback_state.update({
                "checkpoint_role": (
                    "best_m1_preserve_fallback_no_safe_m2" if sparc_preserve_m1
                    else "best_base_preserve_fallback_no_safe_m2"
                ),
                "deployment_epoch_override": -1,
                "best_selection_value": fallback_value,
                "best_selection_tiebreak": float(
                    fallback_metrics.get(
                        fallback_nsd_key,
                        fallback_metrics.get("native_base_nsd", best_native_base_nsd),
                    )
                ),
                "best_selection_epoch": fallback_epoch,
                "selection_protocol": (
                    "No post-deploy M2 checkpoint satisfied strict validation "
                    "no-harm constraints; deploy physical Preserve fallback."
                ),
                "split_access_during_training": {
                    "train": True, "val": True, "test": False
                },
            })
            torch.save(fallback_state, best_selection_path)
            best_selection_epoch = fallback_epoch
            best_selection_value = fallback_value
            best_selection_tiebreak = float(fallback_state["best_selection_tiebreak"])
            logger.warning(
                "[SPARC_SAFE_FALLBACK] No safe M2 checkpoint qualified; saved "
                "Preserve fallback to %s from %s (source epoch=%d).",
                best_selection_path, fallback_source,
                fallback_epoch + 1 if fallback_epoch >= 0 else -1,
            )

        state = _checkpoint_state(
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            best_fusion,
            best_oracle,
            run_name,
            cfg,
            phase_b_started=False,
            ema=ema,
            weight_source="raw",
            hard_case_memory=hard_case_memory,
        )
        state["best_selection_value"] = best_selection_value
        state["best_selection_tiebreak"] = best_selection_tiebreak
        state["best_selection_epoch"] = best_selection_epoch
        state["best_native_base_dice"] = best_native_base_dice
        state["best_native_base_nsd"] = best_native_base_nsd
        state["best_native_base_catastrophic_rate"] = best_native_base_catastrophic_rate
        state["best_native_m1_dice"] = best_native_m1_dice
        state["best_native_m1_nsd"] = best_native_m1_nsd
        state["best_native_m1_epoch"] = best_native_m1_epoch
        if validation_metrics is not None:
            state["validation_metrics"] = validation_metrics

        # AUTO_PARAM_ADAPTER_CHECKPOINT_BEGIN
        if auto_adapter is not None and auto_adapter.enabled:
            state["auto_param_adapter"] = auto_adapter.state_dict()
        # AUTO_PARAM_ADAPTER_CHECKPOINT_END

        if use_validation_selection:
            state["selection_protocol"] = (
                "Validation-only checkpoint selection using the pre-declared "
                "metric; Test is never opened during training."
            )
            state["split_access_during_training"] = {
                "train": True,
                "val": True,
                "test": False,
            }
        else:
            state["selection_protocol"] = (
                "Train-only. Val and Test are never opened; last epoch is used."
            )
            state["split_access_during_training"] = {
                "train": True,
                "val": False,
                "test": False,
            }

        torch.save(state, resume_path)

        early_stopping_patience = max(
            0, int(_cfg_get(cfg.TRAIN, "EARLY_STOPPING_PATIENCE_EPOCHS", 0))
        )
        early_stopping_min_epoch = max(
            1, int(_cfg_get(cfg.TRAIN, "EARLY_STOPPING_MIN_EPOCH", 1))
        )
        early_stop_triggered = bool(
            use_validation_selection
            and validation_metrics is not None
            and early_stopping_patience > 0
            and best_native_m1_epoch >= 0
            and epoch + 1 >= early_stopping_min_epoch
            and epoch - best_native_m1_epoch >= early_stopping_patience
        )

        if early_stop_triggered:
            logger.info(
                "EARLY STOP | epoch=%d | best_m1_epoch=%d | patience=%d",
                epoch + 1,
                best_native_m1_epoch + 1,
                early_stopping_patience,
            )

        if epoch + 1 == cfg.TRAIN.NUM_EPOCHS or early_stop_triggered:
            final_path = os.path.join(
                checkpoint_dir,
                f"{run_name}_last_epoch.pth" if not early_stop_triggered else
                f"{run_name}_early_stopped_epoch_{epoch + 1:03d}.pth",
            )

            state["is_last_epoch_checkpoint"] = True
            state["early_stopped"] = early_stop_triggered
            state["configured_num_epochs"] = int(cfg.TRAIN.NUM_EPOCHS)

            torch.save(state, final_path)

            if use_validation_selection:
                logger.info(
                    "TRAIN COMPLETE | last_epoch_checkpoint=%s | "
                    "best_val_checkpoint=%s | best_base_checkpoint=%s | "
                    "best_m1_native_checkpoint=%s | "
                    "best_val_epoch=%d | best_selection_value=%.6f | "
                    "best_m1_epoch=%d best_m1_native_dice=%.6f | "
                    "Test was never opened.",
                    final_path,
                    best_selection_path,
                    best_base_selection_path,
                    best_m1_selection_path,
                    best_selection_epoch + 1 if best_selection_epoch >= 0 else -1,
                    best_selection_value,
                    best_native_m1_epoch + 1 if best_native_m1_epoch >= 0 else -1,
                    best_native_m1_dice,
                )
            else:
                logger.info(
                    "TRAIN COMPLETE | last_epoch_checkpoint=%s | "
                    "checkpoint_policy=physical_last_epoch | "
                    "Val and Test were never opened during training.",
                    final_path,
                )
            if early_stop_triggered:
                break


if __name__ == "__main__":
    main()
