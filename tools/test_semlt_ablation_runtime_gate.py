#!/usr/bin/env python3
"""Dependency-light regression test for the SemLT checkpoint routing bug.

The full train.py imports the dataset/medical stack, so this test extracts only
``_validate_semlt_protocol`` from its AST and executes it with tiny protocol
stubs.  The test therefore exercises the *actual validator source* while still
being runnable before CUDA/data initialization.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import yaml


def ns(x):
    if isinstance(x, dict):
        return NS(**{k: ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [ns(v) for v in x]
    return x


def cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def load_validator():
    tree = ast.parse(Path("train.py").read_text(encoding="utf-8"))
    fn = find_function(tree, "_validate_semlt_protocol")
    module = ast.Module(body=[copy.deepcopy(fn)], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {
        "_cfg_get": cfg_get,
        "_semlt": lambda cfg: bool(cfg_get(cfg_get(cfg, "M1", None), "ENABLED", False)),
        "validate_geotr_m1_checkpoint_protocol": lambda m1, train: [],
    }
    exec(compile(module, "<train.py:_validate_semlt_protocol>", "exec"), scope)
    return scope["_validate_semlt_protocol"]


def expect_ok(fn, cfg, label):
    try:
        fn(cfg)
    except Exception as exc:  # pragma: no cover - failure output is the point
        raise AssertionError(f"{label} unexpectedly failed: {exc}") from exc


def expect_fail(fn, cfg, needle, label):
    try:
        fn(cfg)
    except ValueError as exc:
        text = str(exc)
        assert needle in text, f"{label}: missing expected diagnostic {needle!r}: {text}"
        return
    raise AssertionError(f"{label} unexpectedly passed")


def main() -> None:
    raw = yaml.safe_load(Path("configs/SEMLT_ABLATION_FROZEN100.yaml").read_text(encoding="utf-8"))
    base = ns(raw)
    fn = load_validator()

    # Exact configuration used by launch_semlt_ablation_variant_one.sh after it
    # supplies --init-checkpoint.  This is the path that failed before R1.
    causal = copy.deepcopy(base)
    causal.init_checkpoint = "/tmp/locked_common_base.pth"
    expect_ok(fn, causal, "valid causal common-Base checkpoint")

    # A causal ablation without its required Base must still fail closed.
    missing_base = copy.deepcopy(base)
    missing_base.init_checkpoint = ""
    expect_fail(
        fn,
        missing_base,
        "GEOTR_M1_CAUSAL_ABLATION requires a common Base checkpoint",
        "causal without Base",
    )

    # An undeclared from-scratch M1 run is still forbidden from loading a task
    # checkpoint.  The fix must not weaken this safety rule.
    generic = copy.deepcopy(base)
    generic.M1.GEOTR_M1_CAUSAL_ABLATION = False
    generic.M1.GEOTR_M1_MAIN_E2E100 = False
    generic.M1.V469_FREEZE_BASE = False
    generic.init_checkpoint = "/tmp/task_checkpoint.pth"
    expect_fail(
        fn,
        generic,
        "from-scratch M1-only training forbids task checkpoints",
        "generic from-scratch checkpoint",
    )

    # Main E2E100 and causal ablation are explicitly mutually exclusive.
    both = copy.deepcopy(base)
    both.M1.GEOTR_M1_MAIN_E2E100 = True
    both.init_checkpoint = "/tmp/locked_common_base.pth"
    expect_fail(
        fn,
        both,
        "GEOTR_M1_MAIN_E2E100 and GEOTR_M1_CAUSAL_ABLATION are mutually exclusive",
        "causal+main conflict",
    )

    print(
        "[PASS] SemLT runtime checkpoint gate: valid causal common-Base checkpoint accepted; "
        "missing Base, generic task-checkpoint loading, and causal/main conflicts remain fail-closed."
    )


if __name__ == "__main__":
    main()
