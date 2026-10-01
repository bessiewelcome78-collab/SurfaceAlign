#/home/tsz-25/MedCLIPSeg-pristine/utils/auto_param_adapter.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_set(node: Any, key: str, value: Any) -> None:
    if node is None:
        return
    if isinstance(node, dict):
        node[key] = value
    else:
        setattr(node, key, value)


def _section(cfg: Any, key: str) -> Any:
    return _cfg_get(cfg, key, None)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _optimizer_groups(optimizer):
    base = getattr(optimizer, "base_optimizer", optimizer)
    return getattr(base, "param_groups", [])


class AutoParamAdapter:
    """
    Dataset/result adaptive controller for unified M1 training.

    It does NOT use Val/Test labels.
    It only consumes train-time diagnostics already produced by train.py:
      - v422_delete_fp_removed
      - v422_delete_tp_removed
      - v422_boundary_fill_fn_added
      - v422_boundary_fill_bg_added
      - v426_* losses
      - v428_adaptive_loss
      - proposal_loss/base_loss

    The goal is not to pretend these values are globally optimal.
    The goal is to avoid hard-coding BUSI-specific constants and make
    candidate supervision self-adjusting under a fixed safety envelope.
    """

    def __init__(self, cfg: Any, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.auto = _section(cfg, "AUTO_ADAPTER")
        self.m1 = _section(cfg, "M1")

        self.enabled = bool(_cfg_get(self.auto, "ENABLED", False))
        self.history: list[dict[str, float]] = []

        self.candidate_weight = _as_float(
            _cfg_get(
                self.auto,
                "CANDIDATE_WEIGHT_INIT",
                1.0,
            ),
            1.0,
        )
        self.candidate_min = _as_float(_cfg_get(self.auto, "CANDIDATE_WEIGHT_MIN", 0.25), 0.25)
        self.candidate_max = _as_float(_cfg_get(self.auto, "CANDIDATE_WEIGHT_MAX", 4.00), 4.00)
        self.effective_candidate_min = _as_float(_cfg_get(self.auto, "EFFECTIVE_CANDIDATE_RATIO_MIN", 0.05), 0.05)
        self.effective_candidate_max = _as_float(_cfg_get(self.auto, "EFFECTIVE_CANDIDATE_RATIO_MAX", 0.80), 0.80)

        self.oracle_weight = _as_float(
            _cfg_get(self.auto, "ORACLE_WEIGHT_INIT", _cfg_get(self.m1, "V426_FAMILY_ORACLE_WEIGHT", 1.0)),
            1.0,
        )
        self.oracle_min = _as_float(_cfg_get(self.auto, "ORACLE_WEIGHT_MIN", 0.50), 0.50)
        self.oracle_max = _as_float(_cfg_get(self.auto, "ORACLE_WEIGHT_MAX", 3.00), 3.00)

        self.quality_weight = _as_float(
            _cfg_get(self.auto, "QUALITY_WEIGHT_INIT", _cfg_get(self.m1, "V426_ACTION_QUALITY_WEIGHT", 1.0)),
            1.0,
        )
        self.quality_min = _as_float(_cfg_get(self.auto, "QUALITY_WEIGHT_MIN", 0.50), 0.50)
        self.quality_max = _as_float(_cfg_get(self.auto, "QUALITY_WEIGHT_MAX", 2.00), 2.00)

        self.safe_bank_weight = _as_float(
            _cfg_get(self.auto, "SAFE_BANK_WEIGHT_INIT", _cfg_get(self.m1, "M1_SAFE_BANK_WEIGHT", 1.0)),
            1.0,
        )
        self.safe_bank_min = _as_float(_cfg_get(self.auto, "SAFE_BANK_WEIGHT_MIN", 0.50), 0.50)
        self.safe_bank_max = _as_float(_cfg_get(self.auto, "SAFE_BANK_WEIGHT_MAX", 3.00), 3.00)

        self.base_lr_mult = _as_float(_cfg_get(self.auto, "BASE_LR_MULT_INIT", 1.0), 1.0)
        self.m1_lr_mult = _as_float(_cfg_get(self.auto, "M1_LR_MULT_INIT", 1.0), 1.0)
        self.lr_mult_min = _as_float(_cfg_get(self.auto, "M1_LR_MULT_MIN", _cfg_get(self.auto, "LR_MULT_MIN", 0.30)), 0.30)
        self.lr_mult_max = _as_float(_cfg_get(self.auto, "LR_MULT_MAX", 2.00), 2.00)

        self.target_fill_fn = _as_float(_cfg_get(self.auto, "TARGET_BOUNDARY_FN_ADDED", 0.020), 0.020)
        self.max_fill_bg = _as_float(_cfg_get(self.auto, "MAX_BOUNDARY_BG_ADDED", 0.00025), 0.00025)
        self.min_delete_fp = _as_float(_cfg_get(self.auto, "MIN_DELETE_FP_REMOVED", 0.0010), 0.0010)
        self.max_delete_tp = _as_float(_cfg_get(self.auto, "MAX_DELETE_TP_REMOVED", 0.00020), 0.00020)

        self.adaptive_loss_patience = int(_cfg_get(self.auto, "ADAPTIVE_LOSS_PATIENCE", 2))
        self.warmup_epochs = int(_cfg_get(self.auto, "WARMUP_EPOCHS", 2))

        if self.enabled:
            self._apply_to_cfg()

        if self.enabled and self.logger is not None:
            self.logger.info(
                "[AUTO_ADAPTER] enabled | candidate=%.4f oracle=%.4f quality=%.4f safe_bank=%.4f "
                "| target_fill_fn=%.6f max_fill_bg=%.6f min_del_fp=%.6f max_del_tp=%.6f",
                self.candidate_weight,
                self.oracle_weight,
                self.quality_weight,
                self.safe_bank_weight,
                self.target_fill_fn,
                self.max_fill_bg,
                self.min_delete_fp,
                self.max_delete_tp,
            )

    def _apply_to_cfg(self) -> None:
        if not self.enabled:
            return
        if self.m1 is None:
            return

        # Keep inner candidate loss normalized.
        # AutoAdapter controls candidate strength only through
        # the effective candidate_ratio returned by on_epoch_begin().
        _cfg_set(self.m1, "CANDIDATE_LOSS_WEIGHT", 1.0)

        # These are real adaptive loss weights.
        _cfg_set(self.m1, "V426_FAMILY_ORACLE_WEIGHT", float(self.oracle_weight))
        _cfg_set(self.m1, "V426_ACTION_QUALITY_WEIGHT", float(self.quality_weight))
        _cfg_set(self.m1, "M1_SAFE_BANK_WEIGHT", float(self.safe_bank_weight))

    def _apply_lr(self, optimizer) -> None:
        for group in _optimizer_groups(optimizer):
            name = str(group.get("name", "")).lower()
            current = float(group.get("lr", 0.0))
            if current <= 0:
                continue

            if "base" == name:
                group["lr"] = current * self.base_lr_mult
            elif "m1" in name or "pse" in name:
                group["lr"] = current * self.m1_lr_mult

    def on_epoch_begin(self, epoch: int, model, optimizer, candidate_ratio: float) -> float:
        if not self.enabled:
            return float(candidate_ratio)

        self._apply_to_cfg()
        self._apply_lr(optimizer)

        # train.py has its own warmup/ramp. We keep that shape but multiply
        # by the adaptive candidate weight.
        adapted = float(candidate_ratio) * float(self.candidate_weight)
        if float(candidate_ratio) > 0.0:
            adapted = _clamp(adapted, self.effective_candidate_min, self.effective_candidate_max)
        else:
            adapted = 0.0

        if self.logger is not None:
            self.logger.info(
                "[AUTO_ADAPTER] epoch_begin=%03d | cand_ratio %.4f -> %.4f | "
                "oracle=%.4f quality=%.4f safe_bank=%.4f lr_mult(base=%.3f,m1=%.3f)",
                epoch + 1,
                float(candidate_ratio),
                adapted,
                self.oracle_weight,
                self.quality_weight,
                self.safe_bank_weight,
                self.base_lr_mult,
                self.m1_lr_mult,
            )
        return adapted

    def on_epoch_end(self, epoch: int, means: Dict[str, float], optimizer=None) -> None:
        if not self.enabled:
            return

        m = {str(k): _as_float(v, 0.0) for k, v in means.items()}
        self.history.append(m)

        # V484 diagnostics support. In the error-state causal pipeline the old
        # v422 spatial keys are intentionally absent, so do not treat them as
        # missing/failure. Adapt only from V484 candidate-quality diagnostics.
        v484_keys = (
            "v484_base_dice",
            "v484_local_oracle_gain",
            "v484_local_harmful_rate",
            "v484_support_precision",
            "v484_correction_outside_ratio",
            "v484_local_active_rate",
            "v485_candidate_non_noop_rate",
            "v485_candidate_mean_abs_change",
        )
        if any(key in m for key in v484_keys):
            oracle_gain = m.get("v485_local_oracle_gain", m.get("v484_local_oracle_gain", 0.0))
            harmful_rate = m.get("v485_harmful_candidate_rate", m.get("v484_local_harmful_rate", 1.0))
            support_precision = m.get("v485_support_precision", m.get("v484_support_precision", 0.0))
            outside_ratio = m.get("v485_correction_outside_ratio", m.get("v484_correction_outside_ratio", 1.0))
            active_rate = m.get("v485_local_active_rate", m.get("v484_local_active_rate", 0.0))
            non_noop = m.get("v485_candidate_non_noop_rate", 0.0)
            mean_change = m.get("v485_candidate_mean_abs_change", 0.0)

            # V485 safety logic: never increase candidate pressure when the
            # candidate bank has no oracle gain, no non-noop edits, or edits
            # mostly outside the error target. This prevents the old failure
            # mode where oracle_gain=0 but cand_ratio kept rising.
            invalid_bank = (
                oracle_gain <= 1.0e-5
                or non_noop < 0.10
                or mean_change < 1.0e-5
                or outside_ratio > 0.85
            )
            unsafe = harmful_rate > 0.35 or outside_ratio > 0.75
            useful_and_safe = (
                oracle_gain > 0.003
                and non_noop > 0.30
                and support_precision > 0.10
                and outside_ratio < 0.80
                and harmful_rate < 0.30
            )

            if invalid_bank or unsafe:
                self.candidate_weight = _clamp(
                    self.candidate_weight * 0.60,
                    self.candidate_min,
                    self.candidate_max,
                )
                self.quality_weight = _clamp(
                    self.quality_weight * 1.03,
                    self.quality_min,
                    self.quality_max,
                )
            elif epoch + 1 > self.warmup_epochs and useful_and_safe:
                self.candidate_weight = _clamp(
                    self.candidate_weight * 1.05,
                    self.candidate_min,
                    self.candidate_max,
                )

            self._apply_to_cfg()
            if self.logger is not None:
                self.logger.info(
                    "[AUTO_ADAPTER_V485] epoch_end=%03d | gain=%.5f harmful=%.4f "
                    "support=%.4f outside=%.4f active=%.4f nonnoop=%.4f change=%.6f | candidate_weight=%.4f",
                    epoch + 1,
                    oracle_gain,
                    harmful_rate,
                    support_precision,
                    outside_ratio,
                    active_rate,
                    non_noop,
                    mean_change,
                    self.candidate_weight,
                )
            return

        # Only adapt when spatial M1 diagnostics are actually present.
        # Otherwise missing metrics become zeros and can falsely trigger
        # "too conservative but safe" updates.
        spatial_keys = (
            "v422_boundary_fill_fn_added",
            "v422_boundary_fill_bg_added",
            "v422_delete_fp_removed",
            "v422_delete_tp_removed",
        )
        if not all(key in m for key in spatial_keys):
            if self.logger is not None:
                self.logger.info(
                    "[AUTO_ADAPTER] epoch_end=%03d | skipped update: missing spatial diagnostics.",
                    epoch + 1,
                )
            return

        fill_fn = m.get("v422_boundary_fill_fn_added", 0.0)
        fill_bg = m.get("v422_boundary_fill_bg_added", 0.0)
        del_fp = m.get("v422_delete_fp_removed", 0.0)
        del_tp = m.get("v422_delete_tp_removed", 0.0)
        proposal = m.get("proposal_loss", 0.0)
        base = m.get("base_loss", 0.0)
        adaptive_loss = m.get("v428_adaptive_loss", 0.0)

        too_conservative = (
            fill_fn < self.target_fill_fn * 0.85
            and fill_bg <= self.max_fill_bg
            and del_tp <= self.max_delete_tp
        )
        unsafe_fill = fill_bg > self.max_fill_bg
        unsafe_delete = del_tp > self.max_delete_tp
        delete_too_weak = del_fp < self.min_delete_fp and del_tp <= self.max_delete_tp

        adaptive_worse = False
        if len(self.history) >= self.adaptive_loss_patience + 1:
            recent = [
                h.get("v428_adaptive_loss", 0.0)
                for h in self.history[-(self.adaptive_loss_patience + 1):]
            ]
            # Do not treat a constant zero/near-zero auxiliary loss as worse.
            # Only trigger when the loss is meaningfully positive and increasing.
            if max(recent) > 1.0e-6:
                adaptive_worse = all(
                    recent[i + 1] > recent[i] + 1.0e-6
                    for i in range(len(recent) - 1)
                )

        # 1) Candidate too weak and still safe: increase candidate/oracle pressure.
        if epoch + 1 > self.warmup_epochs and too_conservative:
            self.candidate_weight = _clamp(self.candidate_weight * 1.15 + 0.02, self.candidate_min, self.candidate_max)
            self.oracle_weight = _clamp(self.oracle_weight * 1.10, self.oracle_min, self.oracle_max)
            self.safe_bank_weight = _clamp(self.safe_bank_weight * 1.05, self.safe_bank_min, self.safe_bank_max)

        # 2) Delete does not remove enough FP: strengthen quality/oracle without relaxing harm.
        if epoch + 1 > self.warmup_epochs and delete_too_weak:
            self.oracle_weight = _clamp(self.oracle_weight * 1.08, self.oracle_min, self.oracle_max)
            self.quality_weight = _clamp(self.quality_weight * 1.05, self.quality_min, self.quality_max)

        # 3) Unsafe edits: strengthen the safe-bank first.
        # Do not globally kill the whole M1 candidate branch for mild Delete
        # risk, otherwise useful Fill candidates are suppressed together with
        # Delete. Candidate pressure is reduced only for unsafe Fill or for
        # severe Delete TP removal.
        if unsafe_fill:
            self.candidate_weight = _clamp(
                self.candidate_weight * 0.85,
                self.candidate_min,
                self.candidate_max,
            )
            self.safe_bank_weight = _clamp(
                self.safe_bank_weight * 1.20,
                self.safe_bank_min,
                self.safe_bank_max,
            )

        if unsafe_delete:
            self.safe_bank_weight = _clamp(
                self.safe_bank_weight * 1.15,
                self.safe_bank_min,
                self.safe_bank_max,
            )
            self.quality_weight = _clamp(
                self.quality_weight * 1.05,
                self.quality_min,
                self.quality_max,
            )
            if del_tp > 2.0 * max(self.max_delete_tp, 1.0e-12):
                self.candidate_weight = _clamp(
                    self.candidate_weight * 0.90,
                    self.candidate_min,
                    self.candidate_max,
                )

        # 4) Adaptive C6 instability: do not blindly push C6 harder.
        if adaptive_worse:
            self.candidate_weight = _clamp(self.candidate_weight * 0.92, self.candidate_min, self.candidate_max)
            self.m1_lr_mult = _clamp(self.m1_lr_mult * 0.90, self.lr_mult_min, self.lr_mult_max)

        # 5) Loss-scale controller: proposal is too dominant or too weak relative to base.
        ratio = proposal / max(base, 1e-6)
        if ratio > 15.0:
            self.candidate_weight = _clamp(self.candidate_weight * 0.90, self.candidate_min, self.candidate_max)
        elif ratio < 5.0 and not (unsafe_fill or unsafe_delete):
            self.candidate_weight = _clamp(self.candidate_weight * 1.05, self.candidate_min, self.candidate_max)

        self._apply_to_cfg()

        if self.logger is not None:
            self.logger.info(
                "[AUTO_ADAPTER] epoch_end=%03d | fill_fn=%.6f fill_bg=%.6f del_fp=%.6f del_tp=%.6f "
                "| too_cons=%s unsafe_fill=%s unsafe_delete=%s adaptive_worse=%s "
                "| candidate=%.4f oracle=%.4f quality=%.4f safe_bank=%.4f m1_lr_mult=%.3f",
                epoch + 1,
                fill_fn,
                fill_bg,
                del_fp,
                del_tp,
                str(bool(too_conservative)),
                str(bool(unsafe_fill)),
                str(bool(unsafe_delete)),
                str(bool(adaptive_worse)),
                self.candidate_weight,
                self.oracle_weight,
                self.quality_weight,
                self.safe_bank_weight,
                self.m1_lr_mult,
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "candidate_weight": self.candidate_weight,
            "oracle_weight": self.oracle_weight,
            "quality_weight": self.quality_weight,
            "safe_bank_weight": self.safe_bank_weight,
            "base_lr_mult": self.base_lr_mult,
            "m1_lr_mult": self.m1_lr_mult,
            "target_fill_fn": self.target_fill_fn,
            "max_fill_bg": self.max_fill_bg,
            "min_delete_fp": self.min_delete_fp,
            "max_delete_tp": self.max_delete_tp,
            "history": self.history[-20:],
        }
