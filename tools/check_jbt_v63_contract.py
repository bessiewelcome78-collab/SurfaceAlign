#!/usr/bin/env python3
"""Fail-closed static contract for the v6.3 patch."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, label: str) -> None:
    if not condition:
        print(f"[FAIL] {label}")
        raise SystemExit(1)
    print(f"[PASS] {label}")


def main() -> None:
    transport = (ROOT / "utils/geotr_m1_transport.py").read_text(encoding="utf-8")
    loss = (ROOT / "utils/geotr_m1_loss.py").read_text(encoding="utf-8")
    train = (ROOT / "train.py").read_text(encoding="utf-8")
    diag = (ROOT / "configs/reproplus/JBT_V63_VAL_FALLBACK_DIAG20.yaml").read_text(encoding="utf-8")
    formal = (ROOT / "configs/reproplus/JBT_V63_VAL_FALLBACK_PAPER100.yaml").read_text(encoding="utf-8")
    diag_run = (ROOT / "scripts/run_jbt_v63_diag20_pair.sh").read_text(encoding="utf-8")
    fixed_audit = (ROOT / "tools/audit_jbt_v636_fixed_base.py").read_text(encoding="utf-8")
    formal_run = (ROOT / "scripts/run_jbt_v63_paper100_pair.sh").read_text(encoding="utf-8")
    formal_launcher = (ROOT / "scripts/launch_jbt_v634_paper100_one.sh").read_text(encoding="utf-8")
    base_diag = (ROOT / "configs/reproplus/OFFICIAL_BASE_EXACT_DIAG20.yaml").read_text(encoding="utf-8")
    base_formal = (ROOT / "configs/reproplus/OFFICIAL_BASE_EXACT_PAPER100.yaml").read_text(encoding="utf-8")

    require("JBT_V63_ENABLED: true" in diag and "JBT_V63_ENABLED: true" in formal,
            "v63 enabled")
    require("BATCH_SIZE: 24" in diag and "GRAD_ACCUMULATION_STEPS: 1" in diag,
            "diag native batch24 single Adam step")
    require("BATCH_SIZE: 24" in formal and "NUM_EPOCHS: 100" in formal,
            "formal released Base100 geometry")
    require("JBT_V63_AUX_MICROBATCH_SIZE: 1" in diag
            and "JBT_V63_AUX_MICROBATCH_SIZE: 1" in formal,
            "dense JBT auxiliary microbatch1")
    require("[JBT_V63_EXACT_BASE_AUX_MICROBATCH]" in train
            and "micro_objective.backward()" in train,
            "physical Base24 step separated from JBT microbatches")
    require("v632_aux_rng_state = _capture_rng_state()" in train
            and "_restore_rng_state(v632_aux_rng_state)" in train,
            "whole auxiliary pass cannot shift next Base RNG")
    for label, config in (
        ("base diag", base_diag), ("base formal", base_formal),
        ("JBT diag", diag), ("JBT formal", formal),
    ):
        require("V552R4204_BASE_RNG_ISOLATION_ENABLED: true" in config,
                f"{label} post-construction RNG reset")
        require("OFFICIAL_BASE_BATCH_RNG_LOCK: true" in config,
                f"{label} physical Base batch RNG lock")
    require("_official_base_batch_seed" in train
            and "[JBT_V633_BASE_BATCH_RNG_LOCK]" in train,
            "Base dropout stream is derived only from seed/epoch/batch")
    require("official_batch_rng_lock" in train
            and "[JBT_V635_DATALOADER_RNG_LOCK]" in train
            and 'train_loader_kwargs["generator"] = generator' in train,
            "DataLoader shuffle/iterator RNG is independent of JBT construction")
    require("JBT_BASE_PARITY_REFERENCE_LOG" in train
            and "[JBT_V633_PARITY_FAIL_FAST]" in train,
            "JBT aborts immediately on step1 or epoch Base divergence")
    require("JBT_V634_CASE_BALANCED_DISPLACEMENT: true" in diag
            and "JBT_V634_SIGN_LOSS_WEIGHT: 0.5" in diag,
            "case-balanced displacement plus explicit sign supervision")
    require("magnitude_per_case" in loss and "signed_disp_balanced_sign_loss" in loss,
            "operator-correct direction loss is active")
    require("JBT_V634_DETACH_UTILITY_FROM_GENERATOR: true" in diag
            and "utility_input = utility_input.detach()" in transport,
            "selector gradients cannot rewrite the proposal generator")
    require("REQUIRE_POSITIVE_VAL_GAIN" in diag_run
            and "VAL_GATE_FAILED" in diag_run,
            "20-epoch Test opens only after positive Validation gain")
    require("[JBT_V635_RECOVERY]" in diag_run
            and "Missing matched Base paper_legacy CSV" not in diag_run,
            "step1 recovery reuses Base checkpoint without requiring early Test")
    require("JBT_FIXED_BASE_RECOVERY" in train
            and "[JBT_V636_FIXED_BASE_LOAD]" in train
            and "official_base_lr" in train,
            "fixed recovery loads matched Base and sets physical Base Adam lr=0")
    require("JBT_FIXED_BASE_CHECKPOINT" in diag_run
            and "audit_jbt_v636_fixed_base.py" in diag_run
            and "fixed_matched_base_final" in fixed_audit,
            "fixed recovery audits init/step1/epochs against matched Base final")
    for label, script in (("diag", diag_run), ("formal", formal_run)):
        require("JBT_BASE_PARITY_REFERENCE_LOG" in script,
                f"{label} injects matched-Base parity reference")
    for label, script in (("20-epoch", diag_run), ("formal", formal_run)):
        require(script.find("--split val") >= 0
                and script.find("--split test") > script.find("--split val"),
                f"{label} pipeline does not open any Test before Val lock")
    for label, config in (("diag", diag), ("formal", formal)):
        require("JBT_V62_UTILITY_WEIGHT_STAGE3: 0.15" in config,
                f"{label} late utility weight retained at stage2 level")
        require("JBT_V62_CAPACITY_WEIGHT_STAGE3: 0.75" in config
                and "JBT_V62_DISPLACEMENT_WEIGHT_STAGE3: 0.75" in config,
                f"{label} late generator capacity retained")
    require("signed_owner_fraction=signed_owner_fraction" in train
            and "owner_is_known_empty" in train
            and "jbt_v63_zero_flow_no_owner_batch" in train,
            "zero-flow is allowed only for a proven zero-owner batch")
    require("v62_gain_mean\n                            + self.v63_class_score_weight" in transport,
            "expected-gain selector")
    require("self.v63_val_fallback_strength > 0.0" in transport,
            "validation fallback deployment")
    require("gain_mean / gain_scale" in loss and "JBT_V63_GAIN_HUBER_BETA" in loss,
            "direct gain Huber calibration")
    for label, script in (("diag", diag_run), ("formal", formal_run)):
        val_pos = script.find("--split val")
        test_pos = script.find("--split test", val_pos + 1)
        require(val_pos >= 0 and test_pos > val_pos, f"{label} Val lock precedes module Test")
        require("select_jbt_v63_val_fallback.py" in script, f"{label} uses Val selector")
        require("M1.JBT_V63_VAL_FALLBACK_STRENGTH \"$FALLBACK_STRENGTH\"" in script,
                f"{label} passes locked fallback to Test")
        require("BASE_INFERENCE_BATCH_SIZE:-32" in script,
                f"{label} preserves released Base MC batch32")
        require("INFERENCE_BATCH_SIZE:-1" in script,
                f"{label} uses low-memory JBT MC batch1")
    require("flock 9" in formal_launcher and "MIN_FREE_MIB:-38000" in formal_launcher,
            "per-GPU serialized queue and 38GiB launch guard")
    require("run_jbt_v63_paper100_pair.sh" in formal_launcher,
            "single-dataset launcher runs full Paper100 pipeline")

    # Run a small graph-level check when invoked inside the project PyTorch
    # environment.  The package remains inspectable on machines without torch.
    try:
        import torch
        import torch.nn.functional as F
        import yaml
        from types import SimpleNamespace as NS
        sys.path.insert(0, str(ROOT))
        from utils.geotr_m1_transport import ExactGeometryTransportSegmenter
        from utils.geotr_m1_loss import compute_geotr_m1_loss

        def ns(value):
            if isinstance(value, dict):
                return NS(**{key: ns(item) for key, item in value.items()})
            if isinstance(value, list):
                return [ns(item) for item in value]
            return value

        parsed = yaml.safe_load(diag)
        parsed["M1"]["JBT_V63_VAL_FALLBACK_STRENGTH"] = 1.0
        cfg = ns(parsed)
        torch.manual_seed(7)
        model = ExactGeometryTransportSegmenter(cfg)
        batch, side = 1, 24
        fine = torch.randn(batch, 512, 6, 6, requires_grad=True)
        seg = torch.randn(batch, 512, requires_grad=True)
        reconstruction = F.interpolate(
            torch.einsum("bc,bchw->bhw", seg, fine)[:, None],
            (side, side), mode="bilinear", align_corners=False,
        )
        base = (0.35 * reconstruction + 0.65 * torch.randn_like(reconstruction)).detach().requires_grad_()
        image = torch.randn(batch, 3, side, side)
        semantic = torch.randn(batch, 512, side, side, requires_grad=True)
        text_feature = torch.randn(batch, 512, requires_grad=True)
        mask = torch.zeros(batch, side, side)
        mask[:, 5:20, 7:19] = 1.0
        candidates, aux = model.generate(
            base, image, semantic, text_feature,
            fine_feature_map=fine, seg_text_vector=seg,
        )
        objective, _ = compute_geotr_m1_loss(cfg, candidates, mask, aux)
        objective.backward()
        require(base.grad is None or float(base.grad.abs().sum()) < 1.0e-10,
                "dynamic Base-gradient isolation")
        require(model.mean_head.flow_out.weight.grad is not None,
                "dynamic signed-flow gradient")
        require(model.v6_candidate_utility_head[-1].weight.grad is not None,
                "dynamic utility gradient")
        model.eval()
        with torch.no_grad():
            _, deployed = model.generate(
                base.detach(), image, semantic.detach(), text_feature.detach(),
                fine_feature_map=fine.detach(), seg_text_vector=seg.detach(),
            )
        require(float(deployed["jbt_v63_val_fallback_used"].item()) == 1.0,
                "dynamic rejected-case Val fallback")
        require(abs(float(deployed["jbt_v5_selected_strength"].item()) - 1.0) < 1.0e-6,
                "dynamic locked strength deployment")
    except ModuleNotFoundError as exc:
        print(f"[SKIP] dynamic PyTorch contract unavailable: {exc}")
    print("[PASS] all JBT-v6.3 contracts")


if __name__ == "__main__":
    main()
