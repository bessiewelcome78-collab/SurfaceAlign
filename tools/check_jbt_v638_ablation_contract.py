#!/usr/bin/env python3
"""Static fail-fast checks for the v6.3.8 deletion-ablation supplement."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
run = (ROOT / "scripts/run_jbt_v638_ablation_one.sh").read_text(encoding="utf-8")
launch = (ROOT / "scripts/launch_jbt_v638_ablation_dataset.sh").read_text(encoding="utf-8")
summary = (ROOT / "tools/summarize_jbt_v638_ablation.py").read_text(encoding="utf-8")
extended = (ROOT / "tools/eval_jbt_extended_metrics.py").read_text(encoding="utf-8")
extended_summary = (ROOT / "tools/collect_jbt_v638_extended_ablation.py").read_text(encoding="utf-8")
relocate = (ROOT / "scripts/relocate_jbt_v638_one_when_safe.sh").read_text(encoding="utf-8")
config = (ROOT / "configs/reproplus/JBT_V63_VAL_FALLBACK_PAPER100.yaml").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)
    print("[PASS] " + message)


require("NUM_EPOCHS: 100" in config, "all ablations inherit 100 training epochs")
require("--num-samples 30" in run, "Validation/Test use MC30")
require(run.index("--split val") < run.index("--split test"), "Validation locks deployment before Test")
require("JBT_FIXED_BASE_CHECKPOINT" in run, "every arm uses the same frozen fresh Base100 host")
require("assert_base_trajectory_parity.py" in run, "every arm audits immutable Base/PVL hashes")
require(
    "M1.GEOTR_M1_USE_SEMANTIC_CONDITIONING false" in run,
    "NO_SEMANTIC removes only semantic-map conditioning",
)
require(
    all(token in run for token in (
        "M1.JBT_V6_SIGNED_DISPLACEMENT_SUPERVISION false",
        "M1.JBT_V62_DISPLACEMENT_WEIGHT_STAGE1 0.0",
        "M1.JBT_V62_DISPLACEMENT_WEIGHT_STAGE2 0.0",
        "M1.JBT_V62_DISPLACEMENT_WEIGHT_STAGE3 0.0",
    )),
    "NO_DISPLACEMENT removes signed-displacement supervision in all curriculum stages",
)
require(
    all(token in run for token in (
        "M1.JBT_CASE_UTILITY_GATE_ENABLED false",
        "M1.JBT_V6_CANDIDATE_UTILITY_ENABLED false",
        "M1.JBT_V62_UTILITY_WEIGHT_STAGE1 0.0",
        "M1.JBT_V62_UTILITY_WEIGHT_STAGE2 0.0",
        "M1.JBT_V62_UTILITY_WEIGHT_STAGE3 0.0",
    )),
    "NO_UTILITY removes the learned candidate/case selector",
)
require(
    "for ARM in NO_SEMANTIC NO_DISPLACEMENT NO_UTILITY" in launch,
    "each dataset executes the three arms sequentially",
)
require(
    "/tmp/jbt_v638_ablation_gpu_${GPU}.lock" in launch,
    "one dataset worker exclusively owns its assigned physical GPU",
)
require("fcntl.flock" in summary and "os.replace" in summary, "shared table updates are locked and atomic")
require("delta_dsc_vs_full_pp" in summary, "summary reports each deletion relative to FULL")
require(
    all(metric in extended for metric in (
        "IoU", "Precision", "Recall", "Specificity", "Accuracy",
        "HD95_ModelPx", "ASSD_ModelPx", "AreaAbsError",
    )),
    "extended evaluator covers region, classification, boundary, and area metrics",
)
require("paired_sign_flip_p_two_sided" in extended, "extended metrics use paired sign-flip tests")
require("--reference-csv" in run, "each ablation is paired case-by-case against FULL")
require("fcntl.flock" in extended_summary and "os.replace" in extended_summary,
        "extended shared table updates are locked and atomic")
require("TARGET_MAX_UTIL" in relocate and "TARGET_MIN_FREE_MIB" in relocate,
        "migration waits for both compute and memory availability")
require(relocate.index("TARGET_MAX_UTIL") < relocate.index("[STOP EXACT SOURCE]"),
        "migration keeps the source alive until the target is safe")
print("[PASS] all JBT-v6.3.8 ablation supplement contracts")
