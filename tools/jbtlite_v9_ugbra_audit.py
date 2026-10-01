#!/usr/bin/env python3
"""Fail-closed structural/protocol audit for JBT-Lite v9 UGBRA.

This audit intentionally does NOT require dataset/checkpoint access. It verifies:
  1) UGBRA is a feature module, not a newly added loss;
  2) zero-init gives exact identity at the first forward;
  3) the residual path becomes trainable after gamma moves away from zero;
  4) BUSI uses a strict 2x2 factorial ablation;
  5) all non-treatment training settings are identical across four arms;
  6) the original JBT-Lite train.py and Surface Alignment loss implementation
     remain unchanged in this v9 package.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import math
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "jbtlite" / "ugbra_v9"

EXPECTED_UNCHANGED_HASHES = {
    "train.py": "5e20fb47014cae045b5eb3f9dc8427dfd4a458a8e00dff14f25590ebee3bd007",
    "utils/jbtl_rbal_loss.py": "ca262dd74d8f43e63d10917229bd01200c0c11ad9de0fb1ca3a83249a420c2d9",
}

ARMS = {
    "BASE":   {"edge": 0.00, "ugbra": False},
    "EDGE20": {"edge": 0.25, "ugbra": False},
    "UGBRA":  {"edge": 0.00, "ugbra": True},
    "FULL":   {"edge": 0.25, "ugbra": True},
}

COMMON_TRAIN = {
    "NUM_EPOCHS": 100,
    "SCHEDULER_TOTAL_EPOCHS": 100,
    "BATCH_SIZE": 24,
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 0.0,
    "OPTIMIZER": "adam",
    "VAL_NUM_SAMPLES": 10,
    "VAL_MC_SEED": 20260910,
    "USE_VALIDATION_SELECTION": True,
    "VAL_SELECTION_METRIC": "native_fusion_dice",
    "VAL_TIEBREAK_METRIC": "native_fusion_nsd",
    "RBAL_NORMAL_WEIGHT": 0.0,
    "RBAL_SCHEDULE_ENABLED": True,
    "RBAL_SCHEDULE_TYPE": "hard_cutoff",
    "RBAL_FULL_WEIGHT_EPOCHS": 20,
    "RBAL_DECAY_END_EPOCH": 21,
    "RBAL_EDGE_GRAD_SCOPE": "all",
    "STRICT_REPRODUCIBILITY": True,
    "ALLOW_TF32": False,
    "DETERMINISTIC": True,
}

EXPECTED_UGBRA = {
    "REDUCTION": 8,
    "GAMMA_INIT": 0.0,
    "DETACH_UNCERTAINTY": True,
    "GATE_FLOOR": 0.05,
    "USE_IMAGE_EDGE": True,
}


def die(msg: str) -> None:
    raise SystemExit("[JBTL9_UGBRA_AUDIT_FAIL] " + msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# 1) Explicitly prove that this v9 did not add/modify a loss implementation.
for rel, expected in EXPECTED_UNCHANGED_HASHES.items():
    path = ROOT / rel
    if not path.is_file():
        die(f"missing unchanged core file: {rel}")
    actual = sha256(path)
    if actual != expected:
        die(f"{rel} changed unexpectedly: {actual} != {expected}")

# 2) Static integration contract.
module_path = ROOT / "trainers" / "ugbra.py"
model_path = ROOT / "trainers" / "medclipseg_unimedclip.py"
if not module_path.is_file():
    die("trainers/ugbra.py missing")
model_text = model_path.read_text(errors="replace")
required_snippets = [
    "from .ugbra import UncertaintyGatedBoundaryResonanceAdapter",
    "self.ugbra_enabled",
    "self.ugbra(decoder_features, coarse_logits, image=raw_image)",
    'base_trainable = {"pvl_adapters", "mask_head", "upscale", "ugbra"}',
]
for snippet in required_snippets:
    if snippet not in model_text:
        die(f"model integration snippet missing: {snippet}")

# 3) Standalone module numerical contract (avoid importing trainer package/deps).
spec = importlib.util.spec_from_file_location("jbtl9_ugbra_standalone", module_path)
if spec is None or spec.loader is None:
    die("cannot load UGBRA module")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Adapter = mod.UncertaintyGatedBoundaryResonanceAdapter

torch.manual_seed(20260911)
adapter = Adapter(
    channels=64,
    reduction=8,
    gamma_init=0.0,
    detach_uncertainty=True,
    gate_floor=0.05,
    use_image_edge=True,
)
x = torch.randn(2, 64, 20, 20, requires_grad=True)
z = torch.randn(2, 1, 20, 20)
image = torch.randn(2, 3, 80, 80)
y, diag = adapter(x, z, image=image, return_diagnostics=True)
if not torch.equal(x, y):
    die(f"zero-init is not exact identity; max_abs={(y-x).abs().max().item():.3e}")
if abs(float(diag["ugbra_source3_mean"]) - 1.0 / 3.0) > 1e-6:
    die("initial source selector is not uniform")
if abs(float(diag["ugbra_source5_mean"]) - 1.0 / 3.0) > 1e-6:
    die("initial source selector is not uniform")
if abs(float(diag["ugbra_source_image_mean"]) - 1.0 / 3.0) > 1e-6:
    die("initial source selector is not uniform")

y.mean().backward()
if adapter.gamma.grad is None or not torch.isfinite(adapter.gamma.grad):
    die("gamma has no finite gradient at zero-init")
non_gamma_grad = sum(
    0.0 if p.grad is None else float(p.grad.detach().abs().sum())
    for n, p in adapter.named_parameters() if n != "gamma"
)
if non_gamma_grad != 0.0:
    die("non-gamma UGBRA tensors should receive zero gradient at exact ReZero init")

with torch.no_grad():
    adapter.gamma.fill_(0.1)
adapter.zero_grad(set_to_none=True)
x2 = torch.randn(2, 64, 16, 16, requires_grad=True)
z2 = torch.randn(2, 1, 16, 16)
image2 = torch.randn(2, 3, 64, 64)
adapter(x2, z2, image=image2).mean().backward()
non_gamma_grad = sum(
    0.0 if p.grad is None else float(p.grad.detach().abs().sum())
    for n, p in adapter.named_parameters() if n != "gamma"
)
if not math.isfinite(non_gamma_grad) or non_gamma_grad <= 0.0:
    die("UGBRA residual branches do not become trainable after gamma opens")

# Full 512-channel module must remain small.
full_adapter = Adapter(512, 8, 0.0, True, 0.05, True)
full_params = sum(p.numel() for p in full_adapter.parameters())
if not (100_000 <= full_params <= 250_000):
    die(f"unexpected UGBRA parameter count: {full_params}")

# 4) Factorial BUSI protocol.
loaded = {}
for arm, treatment in ARMS.items():
    p = CFG_DIR / f"JBTL9_BUSI_{arm}_PAPER100.yaml"
    if not p.is_file():
        die(f"missing BUSI config: {p.name}")
    cfg = yaml.safe_load(p.read_text())
    loaded[arm] = cfg
    if cfg.get("DATASET", {}).get("NAME") != "BUSI":
        die(f"{arm}: DATASET.NAME != BUSI")
    if bool(cfg.get("M1", {}).get("ENABLED", True)):
        die(f"{arm}: old M1/semantic/candidate path must remain disabled")
    tr = cfg.get("TRAIN", {})
    for key, expected in COMMON_TRAIN.items():
        actual = tr.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                die(f"{arm}: TRAIN.{key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            die(f"{arm}: TRAIN.{key}={actual!r}, expected {expected!r}")
    if abs(float(tr.get("RBAL_EDGE_WEIGHT", -1.0)) - treatment["edge"]) > 1e-12:
        die(f"{arm}: wrong existing Surface loss treatment")
    uc = cfg.get("MODEL", {}).get("UGBRA", {})
    if bool(uc.get("ENABLED", False)) != treatment["ugbra"]:
        die(f"{arm}: wrong UGBRA treatment")
    for key, expected in EXPECTED_UGBRA.items():
        if uc.get(key) != expected:
            die(f"{arm}: MODEL.UGBRA.{key}={uc.get(key)!r}, expected {expected!r}")

# Compare entire configs after removing exactly the declared treatments/tags.
def sanitized(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c.get("TRAIN", {}).pop("RUN_TAG", None)
    c.get("M1", {}).pop("RUN_TAG", None)
    c.get("TRAIN", {}).pop("RBAL_EDGE_WEIGHT", None)
    c.get("MODEL", {}).pop("UGBRA", None)
    return c

base_s = sanitized(loaded["BASE"])
for arm in ("EDGE20", "UGBRA", "FULL"):
    if sanitized(loaded[arm]) != base_s:
        die(f"{arm}: contains undeclared config differences vs BASE")

# 5) Other datasets must be paired BASE/FULL with same treatment contract.
for ds in ("Kvasir", "ISIC", "BTMRI"):
    for arm in ("BASE", "FULL"):
        p = CFG_DIR / f"JBTL9_{ds}_{arm}_PAPER100.yaml"
        if not p.is_file():
            die(f"missing {ds} {arm} config")
        cfg = yaml.safe_load(p.read_text())
        want_on = arm == "FULL"
        if bool(cfg["MODEL"]["UGBRA"]["ENABLED"]) != want_on:
            die(f"{ds}/{arm}: wrong UGBRA state")
        expected_edge = 0.25 if want_on else 0.0
        if abs(float(cfg["TRAIN"]["RBAL_EDGE_WEIGHT"]) - expected_edge) > 1e-12:
            die(f"{ds}/{arm}: wrong edge weight")
        if float(cfg["TRAIN"]["RBAL_NORMAL_WEIGHT"]) != 0.0:
            die(f"{ds}/{arm}: no new normal loss is allowed")

print("[JBTL9_UGBRA_AUDIT_PASS]")
print(f"UGBRA full-channel trainable parameters: {full_params:,} (~{full_params/1e6:.3f}M)")
print("Zero-init identity: exact; source mix at init: 1/3, 1/3, 1/3")
print("Original train.py + Surface Alignment implementation: byte-identical to v8 source")
print("BUSI factorial: BASE / EDGE20 / UGBRA / FULL")
print("No semantic branch, no candidate selector, no new auxiliary loss")
