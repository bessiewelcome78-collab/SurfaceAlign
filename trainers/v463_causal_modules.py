from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x[:, None]
    if x.dim() == 4 and x.shape[1] != 1:
        return x[:, :1]
    return x


class _ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        groups = min(8, cout)
        while groups > 1 and cout % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V463ResidualErrorHead(nn.Module):
    """Residual-error predictor used by the local action bank.

    V469 replaces the unstable whole-image FP/FN/TP/BG softmax with two
    conditional binary tasks while preserving the historical four-channel
    public interface:

      channel 0: P(FP | Base foreground)
      channel 1: P(FN | Base-background boundary band)
      channel 2: TP safety = 1 - P(FP)
      channel 3: BG safety = 1 - P(FN)

    Only channels 0 and 1 are learned in conditional mode.  Channels 2 and 3
    are deterministic complements, which keeps downstream CCV features and old
    diagnostics compatible without asking a single softmax to model four
    extremely imbalanced states over the whole image.
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: Optional[int] = None,
        multiclass: bool = False,
        conditional_binary: bool = False,
    ) -> None:
        super().__init__()
        h = int(hidden_channels or latent_channels)
        self.conditional_binary = bool(conditional_binary)
        self.multiclass = bool(multiclass and not self.conditional_binary)
        out_channels = 2 if self.conditional_binary else 4
        self.net = nn.Sequential(
            _ConvGNAct(latent_channels + 4, h),
            _ConvGNAct(h, h),
            nn.Conv2d(h, out_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        latent: torch.Tensor,
        base_prob: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        image_edge: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_prob = _to_b1hw(base_prob).float()
        entropy = _to_b1hw(entropy).float()
        boundary = _to_b1hw(boundary).float()
        image_edge = _to_b1hw(image_edge).float()
        if image_edge.shape[-2:] != latent.shape[-2:]:
            image_edge = F.interpolate(
                image_edge,
                size=latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        edge_norm = image_edge / image_edge.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(EPS)
        x = torch.cat(
            [latent.float(), base_prob, entropy, boundary, edge_norm], dim=1
        )
        raw_logits = self.net(x)

        if self.conditional_binary:
            fp_logit = raw_logits[:, 0:1]
            fn_logit = raw_logits[:, 1:2]
            fp_prob = torch.sigmoid(fp_logit).clamp(EPS, 1.0 - EPS)
            fn_prob = torch.sigmoid(fn_logit).clamp(EPS, 1.0 - EPS)
            tp_prob = (1.0 - fp_prob).clamp(EPS, 1.0 - EPS)
            bg_prob = (1.0 - fn_prob).clamp(EPS, 1.0 - EPS)

            # Four-channel compatibility tensors.  The complements are logits
            # of the opposite binary event, so all downstream differences keep
            # a meaningful sign and scale.
            logits = torch.cat(
                [fp_logit, fn_logit, -fp_logit, -fn_logit], dim=1
            )
            probs = torch.cat([fp_prob, fn_prob, tp_prob, bg_prob], dim=1)
            delete_causal_logit = fp_logit
            fill_causal_logit = fn_logit
        else:
            logits = raw_logits
            probs = (
                torch.softmax(logits, dim=1)
                if self.multiclass
                else torch.sigmoid(logits)
            ).clamp(EPS, 1.0 - EPS)

            if self.multiclass:
                delete_other = torch.logsumexp(
                    logits[:, 1:4], dim=1, keepdim=True
                )
                fill_other = torch.logsumexp(
                    torch.cat([logits[:, 0:1], logits[:, 2:4]], dim=1),
                    dim=1,
                    keepdim=True,
                )
                delete_causal_logit = logits[:, 0:1] - delete_other
                fill_causal_logit = logits[:, 1:2] - fill_other
            else:
                delete_causal_logit = logits[:, 0:1] - logits[:, 2:3]
                fill_causal_logit = logits[:, 1:2] - logits[:, 3:4]

        return {
            "v463_residual_logits": logits,
            "v463_residual_probs": probs,
            "v463_fp_logits": logits[:, 0:1],
            "v463_fn_logits": logits[:, 1:2],
            "v463_tp_risk_logits": logits[:, 2:3],
            "v463_bg_risk_logits": logits[:, 3:4],
            "v463_fp_prob": probs[:, 0:1],
            "v463_fn_prob": probs[:, 1:2],
            "v463_tp_risk_prob": probs[:, 2:3],
            "v463_bg_risk_prob": probs[:, 3:4],
            "v468_delete_causal_logit": delete_causal_logit,
            "v468_fill_causal_logit": fill_causal_logit,
            "v469_fp_binary_logit": logits[:, 0:1],
            "v469_fn_binary_logit": logits[:, 1:2],
            "v469_residual_conditional_binary": logits.new_full(
                (logits.shape[0],), float(self.conditional_binary)
            ),
            "v468_residual_multiclass": logits.new_full(
                (logits.shape[0],), float(self.multiclass)
            ),
        }


class V463CausalCounterfactualVerifier(nn.Module):
    """Candidate-vs-Preserve verifier with a direct monotonic utility head.

    The V468 utility mixed two noisy delta regressors, uncertainty penalties and
    class probabilities.  Diagnostics showed that Pareto/Harm classifiers had
    excellent AUC while the composed utility ranked positives almost in reverse.
    V469 keeps all historical outputs, but adds a seventh direct utility output
    trained against the true weighted gain.  Pareto/Harm probabilities are used
    only as an eligibility gate; they are no longer algebraically mixed into the
    ranking score.
    """

    def __init__(
        self,
        cfg: Any,
        feature_dim: int = 32,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        hidden = int(hidden_dim or _cfg_get(m1, "V463_CCV_HIDDEN_DIM", 96))
        self.eps_dsc = float(_cfg_get(m1, "V463_CCV_DSC_EPS", 5.0e-4))
        self.eps_nsd = float(_cfg_get(m1, "V463_CCV_NSD_EPS", 5.0e-4))
        self.max_harm_prob = float(
            _cfg_get(m1, "V463_CCV_MAX_HARM_PROB", 0.45)
        )
        self.min_pareto_prob = float(
            _cfg_get(m1, "V463_CCV_MIN_PARETO_PROB", 0.55)
        )
        self.sigma_scale = float(_cfg_get(m1, "V463_CCV_SIGMA_SCALE", 0.5))
        self.use_hard_accept = bool(
            _cfg_get(m1, "V463_CCV_HARD_ACCEPT", False)
        )
        self.use_classification_eligibility = bool(
            _cfg_get(m1, "V469_USE_CLASSIFICATION_ELIGIBILITY", True)
        )
        self.utility_threshold = float(
            _cfg_get(m1, "V463_CCV_UTILITY_THRESHOLD", 0.0)
        )
        self.direct_utility_scale = max(
            1.0e-4,
            float(_cfg_get(m1, "V469_DIRECT_UTILITY_SCALE", 0.05)),
        )
        self.utility_dsc_weight = float(
            _cfg_get(m1, "M2_UTILITY_DSC_WEIGHT", 0.60)
        )
        self.utility_nsd_weight = float(
            _cfg_get(m1, "M2_UTILITY_NSD_WEIGHT", 0.40)
        )
        norm = max(self.utility_dsc_weight + self.utility_nsd_weight, EPS)
        self.utility_dsc_weight /= norm
        self.utility_nsd_weight /= norm

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 7),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            # Mild uncertainty prior; direct utility starts at Preserve parity.
            self.net[-1].bias[4:6].fill_(-3.0)
            self.net[-1].bias[6].zero_()

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Stable local mean for binary or straight-through masks.

        Empty masks return exactly zero and receive no denominator-amplified
        gradient.  Pixel-count denominators are clamped to one, never 1e-6.
        """
        numerator = (x * m).sum(dim=(-2, -1))
        denominator = m.sum(dim=(-2, -1))
        value = numerator / denominator.clamp_min(1.0)
        return torch.where(
            denominator > 0.5,
            value,
            torch.zeros_like(value),
        )

    @staticmethod
    def _safe_area(m: torch.Tensor) -> torch.Tensor:
        return m.mean(dim=(-2, -1))

    def _features(
        self,
        base_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: Optional[torch.Tensor],
        action_types: torch.Tensor,
        entropy: Optional[torch.Tensor],
        boundary: Optional[torch.Tensor],
        residual_probs: Optional[torch.Tensor],
        visual_scores: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        base_logits = _to_b1hw(base_logits).float()
        base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
        if candidate_logits.dim() != 4:
            raise ValueError(
                "candidate_logits must be [B,K,H,W], got "
                f"{tuple(candidate_logits.shape)}"
            )
        cand_prob = torch.sigmoid(candidate_logits.float()).clamp(
            EPS, 1.0 - EPS
        )
        b, k, h, w = cand_prob.shape
        factual = factual_masks.float()
        if factual.dim() == 3:
            factual = factual[:, None]
        if factual.shape[-2:] != (h, w):
            factual = F.interpolate(factual, size=(h, w), mode="nearest")
        if control_masks is None:
            control = torch.zeros_like(factual)
        else:
            control = control_masks.float()
            if control.dim() == 3:
                control = control[:, None]
            if control.shape[-2:] != (h, w):
                control = F.interpolate(control, size=(h, w), mode="nearest")
        factual = factual[:, :k]
        control = control[:, :k]
        valid_action = factual.flatten(2).sum(dim=-1) > 0.5

        base_k = base_prob.expand(-1, k, -1, -1)
        delta_prob = cand_prob - base_k
        abs_delta = delta_prob.abs()
        entropy_k = entropy
        if entropy_k is None:
            entropy_k = -(
                base_prob * base_prob.log()
                + (1.0 - base_prob) * (1.0 - base_prob).log()
            )
        entropy_k = _to_b1hw(entropy_k).float().expand(-1, k, -1, -1)
        boundary_k = _to_b1hw(
            boundary if boundary is not None else torch.zeros_like(base_prob)
        ).float().expand(-1, k, -1, -1)

        factual_area = self._safe_area(factual)
        control_area = self._safe_area(control)
        area_ratio = torch.where(
            control_area > 0.0,
            factual_area / control_area.clamp_min(EPS),
            torch.zeros_like(factual_area),
        )
        f_base = self._masked_mean(base_k, factual)
        f_cand = self._masked_mean(cand_prob, factual)
        f_delta = self._masked_mean(delta_prob, factual)
        f_abs_delta = self._masked_mean(abs_delta, factual)
        f_entropy = self._masked_mean(entropy_k, factual)
        f_boundary = self._masked_mean(boundary_k, factual)
        c_base = self._masked_mean(base_k, control)
        c_cand = self._masked_mean(cand_prob, control)
        c_delta = self._masked_mean(delta_prob, control)
        c_abs_delta = self._masked_mean(abs_delta, control)
        c_entropy = self._masked_mean(entropy_k, control)
        c_boundary = self._masked_mean(boundary_k, control)

        action_types = action_types.to(base_logits.device).long().reshape(-1)[:k]
        type_oh = F.one_hot(action_types.clamp(0, 3), num_classes=4).float()
        type_oh = type_oh[None].expand(b, -1, -1)
        delete = action_types[None].expand(b, -1) < 2
        delete_sign = torch.where(delete, -1.0, 1.0)
        relative_delta = f_delta - c_delta

        if residual_probs is None:
            f_benefit = factual_area * 0.0
            f_risk = factual_area * 0.0
            c_benefit = factual_area * 0.0
            c_risk = factual_area * 0.0
        else:
            residual = residual_probs.float()
            if residual.shape[-2:] != (h, w):
                residual = F.interpolate(
                    residual,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            if residual.shape[1] != 4:
                raise ValueError(
                    f"residual_probs must have 4 channels, got {residual.shape[1]}"
                )
            fp = residual[:, 0:1].expand(-1, k, -1, -1)
            fn = residual[:, 1:2].expand(-1, k, -1, -1)
            tp = residual[:, 2:3].expand(-1, k, -1, -1)
            bg = residual[:, 3:4].expand(-1, k, -1, -1)
            benefit_map = torch.where(delete[:, :, None, None], fp, fn)
            risk_map = torch.where(delete[:, :, None, None], tp, bg)
            f_benefit = self._masked_mean(benefit_map, factual)
            f_risk = self._masked_mean(risk_map, factual)
            c_benefit = self._masked_mean(benefit_map, control)
            c_risk = self._masked_mean(risk_map, control)

        if visual_scores is None:
            visual = factual_area * 0.0
        else:
            visual = visual_scores.float()[:, :k]

        x = torch.cat(
            [
                factual_area[..., None],
                control_area[..., None],
                area_ratio[..., None].clamp(max=10.0),
                f_base[..., None],
                f_cand[..., None],
                f_delta[..., None],
                f_abs_delta[..., None],
                f_entropy[..., None],
                f_boundary[..., None],
                c_base[..., None],
                c_cand[..., None],
                c_delta[..., None],
                c_abs_delta[..., None],
                c_entropy[..., None],
                c_boundary[..., None],
                relative_delta[..., None],
                delete_sign[..., None],
                type_oh,
                (factual_area - control_area)[..., None],
                (f_boundary - c_boundary)[..., None],
                (f_entropy - c_entropy)[..., None],
                (f_abs_delta - c_abs_delta)[..., None],
                (f_cand - c_cand)[..., None],
                (f_base - c_base)[..., None],
                f_benefit[..., None],
                f_risk[..., None],
                (f_benefit - f_risk)[..., None],
                (c_benefit - c_risk)[..., None],
                visual[..., None],
            ],
            dim=-1,
        )
        if x.shape[-1] != 32:
            raise RuntimeError(
                f"V469 CCV feature dimension mismatch: {x.shape[-1]} != 32"
            )
        return {
            "features": x,
            "valid_action": valid_action,
        }

    def forward(
        self,
        *,
        base_logits: torch.Tensor,
        candidate_logits_all: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: Optional[torch.Tensor],
        action_types: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        boundary: Optional[torch.Tensor] = None,
        residual_probs: Optional[torch.Tensor] = None,
        visual_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_logits_all.dim() != 4 or candidate_logits_all.shape[1] < 2:
            raise ValueError("CCV requires candidate_logits_all [B,K+1,H,W].")
        candidate_logits = candidate_logits_all[:, 1:]
        feature_pack = self._features(
            base_logits,
            candidate_logits,
            factual_masks,
            control_masks,
            action_types,
            entropy,
            boundary,
            residual_probs,
            visual_scores,
        )
        x = feature_pack["features"]
        valid_action = feature_pack["valid_action"]
        raw = self.net(x)
        tau_dsc = 0.05 * torch.tanh(raw[..., 0])
        tau_nsd = 0.05 * torch.tanh(raw[..., 1])
        harm_logit = raw[..., 2]
        pareto_logit = raw[..., 3]
        sigma_dsc = 0.02 * F.softplus(raw[..., 4]) + 1.0e-4
        sigma_nsd = 0.02 * F.softplus(raw[..., 5]) + 1.0e-4
        direct_utility = self.direct_utility_scale * torch.tanh(raw[..., 6])
        p_harm = torch.sigmoid(harm_logit)
        p_pareto = torch.sigmoid(pareto_logit)
        lcb_dsc = tau_dsc - self.sigma_scale * sigma_dsc
        lcb_nsd = tau_nsd - self.sigma_scale * sigma_nsd

        class_eligible = (
            (p_harm < self.max_harm_prob)
            & (p_pareto > self.min_pareto_prob)
        )
        strict_accept = (
            valid_action
            & class_eligible
            & (lcb_dsc > self.eps_dsc)
            & (lcb_nsd > -self.eps_nsd)
            & (direct_utility > self.utility_threshold)
        )
        if self.use_hard_accept:
            accept = strict_accept
        else:
            accept = valid_action & (direct_utility > self.utility_threshold)
            if self.use_classification_eligibility:
                accept = accept & class_eligible

        masked_utility = direct_utility.masked_fill(~accept, -1.0e4)
        best_action = masked_utility.argmax(dim=1)
        changed = accept.any(dim=1)
        cands = candidate_logits_all[:, 1:]
        chosen_logits = cands.gather(
            1,
            best_action[:, None, None, None].expand(
                -1, 1, cands.shape[-2], cands.shape[-1]
            ),
        )[:, 0]
        base_2d = _to_b1hw(base_logits)[:, 0]
        final_logits = torch.where(changed[:, None, None], chosen_logits, base_2d)
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
        selected_hard = torch.zeros_like(direct_utility)
        if selected_hard.numel() > 0:
            selected_hard.scatter_(
                1, best_action[:, None], changed[:, None].float()
            )
        soft_logits = direct_utility.masked_fill(~valid_action, -1.0e4)
        select_soft = torch.softmax(
            torch.cat(
                [direct_utility.new_zeros(direct_utility.shape[0], 1), soft_logits],
                dim=1,
            ),
            dim=1,
        )
        return {
            "v463_ccv_tau_dsc": tau_dsc,
            "v463_ccv_tau_nsd": tau_nsd,
            "v463_ccv_sigma_dsc": sigma_dsc,
            "v463_ccv_sigma_nsd": sigma_nsd,
            "v463_ccv_lcb_dsc": lcb_dsc,
            "v463_ccv_lcb_nsd": lcb_nsd,
            "v463_ccv_harm_logit": harm_logit,
            "v463_ccv_p_harm": p_harm,
            "v463_ccv_pareto_logit": pareto_logit,
            "v463_ccv_p_pareto": p_pareto,
            "v469_ccv_direct_utility": direct_utility,
            "v463_ccv_utility": direct_utility,
            "v469_ccv_valid_action": valid_action.float(),
            "v469_ccv_class_eligible": class_eligible.float(),
            "v463_ccv_strict_accept_mask": strict_accept.float(),
            "v463_ccv_accept_mask": accept.float(),
            "v463_ccv_selected_hard": selected_hard,
            "v463_ccv_selected_index": torch.where(
                changed, best_action + 1, torch.zeros_like(best_action)
            ),
            "v463_ccv_changed": changed.float(),
            "v463_ccv_select_soft": select_soft,
            "v463_ccv_final_logits": final_logits,
            "v463_ccv_final_probs": final_probs,
        }
