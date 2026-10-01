#!/usr/bin/env python3
from pathlib import Path
import math, sys, yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "jbtlite" / "formal100_bestval"
expected = {
    ("BUSI","BASE"):(0.0,0.0),
    ("BUSI","EDGE"):(0.25,0.0),
    ("BUSI","NORMAL"):(0.0,0.05),
    ("BUSI","FULL"):(0.25,0.05),
    ("Kvasir","FULL"):(0.25,0.05),
    ("ISIC","FULL"):(0.25,0.05),
    ("BTMRI","FULL"):(0.25,0.05),
}
errors=[]
for (ds,arm),(ew,nw) in expected.items():
    p = CFG / f"JBTL4_{ds}_{arm}_PAPER100.yaml"
    if not p.exists():
        errors.append(f"missing {p}")
        continue
    c=yaml.safe_load(p.read_text())
    tr=c.get("TRAIN",{}); m1=c.get("M1",{}); te=c.get("TEST",{}); data=c.get("DATASET",{})
    checks = [
        (data.get("NAME")==ds, "DATASET.NAME"),
        (m1.get("ENABLED") is False, "M1.ENABLED must be false"),
        (int(tr.get("NUM_EPOCHS",-1))==100, "NUM_EPOCHS=100"),
        (int(tr.get("SCHEDULER_TOTAL_EPOCHS",-1))==100, "SCHEDULER_TOTAL_EPOCHS=100"),
        (int(tr.get("BATCH_SIZE",-1))==24, "BATCH_SIZE=24"),
        (abs(float(tr.get("LEARNING_RATE",-1))-3e-4)<1e-12, "LEARNING_RATE=3e-4"),
        (str(tr.get("OPTIMIZER","")).lower()=="adam", "OPTIMIZER=adam"),
        (bool(tr.get("USE_VALIDATION_SELECTION",False)) is True, "best-Val selection must be enabled"),
        (str(tr.get("VAL_SELECTION_METRIC","")).lower()=="native_fusion_dice", "VAL_SELECTION_METRIC=native_fusion_dice"),
        (str(tr.get("VAL_TIEBREAK_METRIC","")).lower()=="native_fusion_nsd", "VAL_TIEBREAK_METRIC=native_fusion_nsd"),
        (int(tr.get("VAL_INTERVAL",-1))==1, "VAL_INTERVAL=1"),
        (int(tr.get("VAL_NUM_SAMPLES",-1))==1, "VAL_NUM_SAMPLES=1"),
        (bool(tr.get("VAL_NATIVE_METRICS",False)) is True, "VAL_NATIVE_METRICS=true"),
        (bool(tr.get("REPLAY_PUBLIC_REPO_VALIDATION_RNG_AFTER_SELECTION",False)) is True,
         "public validation RNG replay after selection must be true"),
        (bool(tr.get("RBAL_SCHEDULE_ENABLED",False)) is True, "RBAL schedule enabled"),
        (str(tr.get("RBAL_SCHEDULE_TYPE","")).lower()=="cosine", "RBAL schedule cosine"),
        (int(tr.get("RBAL_FULL_WEIGHT_EPOCHS",-1))==20, "RBAL full strength through epoch20"),
        (int(tr.get("RBAL_DECAY_END_EPOCH",-1))==80, "RBAL decays to zero at epoch80"),
        (int(te.get("NUM_SAMPLES",-1))==30, "TEST.NUM_SAMPLES=30"),
        (abs(float(tr.get("RBAL_EDGE_WEIGHT",-99))-ew)<1e-12, f"EDGE={ew}"),
        (abs(float(tr.get("RBAL_NORMAL_WEIGHT",-99))-nw)<1e-12, f"NORMAL={nw}"),
    ]
    for ok,msg in checks:
        if not ok: errors.append(f"{p.name}: {msg}")
    for k in ("BASE_BOUNDARY_DICE_WEIGHT","BASE_HARD_NEGATIVE_WEIGHT","BASE_SPILL_RATIO_WEIGHT",
              "BASE_UNDERFILL_RATIO_WEIGHT","BASE_SYMMETRIC_TVERSKY_WEIGHT","BASE_EMPTY_MASK_WEIGHT"):
        if abs(float(tr.get(k,0.0)))>1e-12:
            errors.append(f"{p.name}: hidden base auxiliary {k} nonzero")

def scale(epoch):
    if epoch <= 20: return 1.0
    if epoch >= 80: return 0.0
    progress=(epoch-20)/60.0
    return 0.5*(1+math.cos(math.pi*progress))

if errors:
    print("[JBTL4_BESTVAL_AUDIT_FAIL]")
    for e in errors: print(" -",e)
    sys.exit(20)
print("[JBTL4_BESTVAL_AUDIT_PASS]")
print("Protocol: 224 / 100ep / B24 / Adam3e-4 / Val-best checkpoint / Test MC30 once.")
print("Selection: highest native Val DSC; exact tie broken by corrected true2d Val NSD; Test never selects checkpoint.")
print("Aux schedule: full epochs1-20 -> cosine decay -> zero from epoch80 onward.")
for e in (1,20,40,50,60,80,100):
    print(f"schedule epoch={e:03d} scale={scale(e):.6f} EDGEfull={0.25*scale(e):.6f} NORMALfull={0.05*scale(e):.6f}")
