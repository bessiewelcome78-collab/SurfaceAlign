#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(value: bool, label: str) -> None:
    if not value:
        raise SystemExit(f"[FAIL] {label}")
    print(f"[PASS] {label}")


def main() -> None:
    run = (ROOT / "scripts/run_jbt_v637_full_only_paper100.sh").read_text(encoding="utf-8")
    launch = (ROOT / "scripts/launch_jbt_v637_full_only_one.sh").read_text(encoding="utf-8")
    summary = (ROOT / "tools/summarize_jbt_v637_formal.py").read_text(encoding="utf-8")
    diagnosis = (ROOT / "tools/jbt_v63_diagnose.py").read_text(encoding="utf-8")
    base = (ROOT / "configs/reproplus/OFFICIAL_BASE_EXACT_PAPER100.yaml").read_text(encoding="utf-8")
    jbt = (ROOT / "configs/reproplus/JBT_V63_VAL_FALLBACK_PAPER100.yaml").read_text(encoding="utf-8")
    require("NUM_EPOCHS: 100" in base and "BATCH_SIZE: 24" in base, "official host Paper100/B24")
    require("LEARNING_RATE: 0.0003" in base and "OPTIMIZER: adam" in base, "paper Adam 3e-4")
    require("NUM_EPOCHS: 100" in jbt and "TEST:\n  NUM_SAMPLES: 30" in jbt, "JBT100 and MC30")
    require("GEOTR_M1_MAIN_E2E100: false" in jbt,
            "legacy free_2d/no-gate main contract disabled")
    require("JBT_OVR=" in run and "M1.GEOTR_M1_MAIN_E2E100 false" in run,
            "fixed-host signed-flow override is explicit on every JBT phase")
    require("JBT_FIXED_BASE_RECOVERY=1" in run and "JBT_FIXED_BASE_CHECKPOINT" in run, "validated fixed-host structure")
    require("audit_jbt_v636_fixed_base.py" in run and "assert_base_trajectory_parity.py" in run, "immutable Base/PVL audit")
    require(run.find("--split val") < run.find("--split test"), "Validation lock before one-shot Test")
    require("matched_base_" not in run and "joint_base_" not in run, "no Base Test performance comparison")
    require("jbt_v637_final_paper_legacy.csv" in run and "jbt_v637_final_true2d.csv" in run, "final DSC/NSD protocols")
    require("PAPER100_ALL4_SUMMARY.csv" in summary and "fcntl.LOCK_EX" in summary, "concurrency-safe shared table")
    require(all(name in summary for name in ("BUSI", "Kvasir", "ISIC", "BTMRI")), "all four paper benchmarks")
    require("flock 9" in launch and "flock -n 8" in launch and "MIN_FREE_MIB" in launch,
            "per-dataset duplicate guard, per-GPU queue and memory guard")
    require('>>"$LOG"' in launch, "duplicate launch cannot truncate the master log")
    require("--resume" in run, "safe relaunch/resume")
    require("fixed_host_single_posterior_expected" in diagnosis
            and "'hard_contract':False" in diagnosis,
            "fixed-host mechanics are mode-aware; amplitude ratio is diagnostic")
    print("[PASS] all JBT-v6.3.7 formal contracts")


if __name__ == "__main__":
    main()
