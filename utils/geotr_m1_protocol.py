"""Shared fail-closed checkpoint policy for physical GEOTR-M1 runs.

The training entry point and the standalone FORMAL54 preflight import this
module.  Keeping the policy in one dependency-free location prevents the YAML
checker and the runtime validator from silently accepting different protocols.
"""


def _get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def validate_geotr_m1_checkpoint_protocol(m1, train):
    """Return protocol violations without mutating the configuration.

    ``paper100`` is the explicitly declared reference-paper track: no Val set is
    opened and the physical epoch-100 checkpoint is tested. ``restore150`` is
    the project's historical track: Val is opened every epoch and selects M1 by
    native Dice/NSD.  Missing protocol names retain the historical validation-
    selection contract so older M1-only configs cannot accidentally become
    train-only experiments.
    """

    errors = []
    protocol = str(
        _get(m1, "GEOTR_M1_FORMAL_PROTOCOL", "") or ""
    ).strip().lower()
    if protocol not in {"", "paper100", "restore150", "diag20"}:
        return [
            "M1.GEOTR_M1_FORMAL_PROTOCOL must be paper100, restore150 or diag20 "
            f"(got {protocol!r})"
        ]

    use_selection = bool(_get(train, "USE_VALIDATION_SELECTION", False))

    if protocol == "diag20":
        if use_selection:
            errors.append("diag20 requires TRAIN.USE_VALIDATION_SELECTION=false")
        if int(_get(train, "NUM_EPOCHS", 0)) != 20:
            errors.append("diag20 requires TRAIN.NUM_EPOCHS=20")
        # Keep the 100-epoch cosine schedule so epochs 1..20 are directly
        # comparable with the intended PAPER100 run.
        if int(_get(train, "SCHEDULER_TOTAL_EPOCHS", 0)) != 100:
            errors.append("diag20 requires TRAIN.SCHEDULER_TOTAL_EPOCHS=100")
        if str(_get(train, "OPTIMIZER", "")).strip().lower() != "adam":
            errors.append("diag20 requires TRAIN.OPTIMIZER=adam")
        if bool(_get(train, "USE_EMA", False)):
            errors.append("diag20 requires TRAIN.USE_EMA=false")
        if int(_get(train, "EARLY_STOPPING_PATIENCE_EPOCHS", -1)) != 0:
            errors.append("diag20 requires TRAIN.EARLY_STOPPING_PATIENCE_EPOCHS=0")
        return errors

    if protocol == "paper100":
        if use_selection:
            errors.append(
                "paper100 requires TRAIN.USE_VALIDATION_SELECTION=false; "
                "the physical epoch-100 checkpoint is the only Test checkpoint"
            )
        if int(_get(train, "NUM_EPOCHS", 0)) != 100:
            errors.append("paper100 requires TRAIN.NUM_EPOCHS=100")
        if int(_get(train, "SCHEDULER_TOTAL_EPOCHS", 0)) != 100:
            errors.append("paper100 requires TRAIN.SCHEDULER_TOTAL_EPOCHS=100")
        if str(_get(train, "OPTIMIZER", "")).strip().lower() != "adam":
            errors.append("paper100 requires TRAIN.OPTIMIZER=adam")
        if bool(_get(train, "USE_EMA", False)):
            errors.append("paper100 requires TRAIN.USE_EMA=false")
        if int(_get(train, "EARLY_STOPPING_PATIENCE_EPOCHS", -1)) != 0:
            errors.append(
                "paper100 requires TRAIN.EARLY_STOPPING_PATIENCE_EPOCHS=0"
            )
        if str(_get(train, "VAL_WEIGHT_SOURCE", "raw")).strip().lower() != "raw":
            errors.append("paper100 requires TRAIN.VAL_WEIGHT_SOURCE=raw")
        return errors

    # restore150 and all older unnamed physical M1-only configs remain fail-
    # closed around the established validation-only checkpoint selection path.
    if not use_selection:
        errors.append("TRAIN.USE_VALIDATION_SELECTION must be true")
    if str(_get(train, "VAL_SELECTION_METRIC", "")) != "native_m1_dice":
        errors.append("TRAIN.VAL_SELECTION_METRIC must be native_m1_dice")
    if str(_get(train, "VAL_TIEBREAK_METRIC", "")) != "native_m1_nsd":
        errors.append("TRAIN.VAL_TIEBREAK_METRIC must be native_m1_nsd")
    if int(_get(train, "VAL_INTERVAL", 1)) != 1:
        errors.append("TRAIN.VAL_INTERVAL must be 1 for declared validation selection")

    if protocol == "restore150":
        if int(_get(train, "NUM_EPOCHS", 0)) != 150:
            errors.append("restore150 requires TRAIN.NUM_EPOCHS=150")
        if int(_get(train, "SCHEDULER_TOTAL_EPOCHS", 0)) != 150:
            errors.append("restore150 requires TRAIN.SCHEDULER_TOTAL_EPOCHS=150")
        if str(_get(train, "OPTIMIZER", "")).strip().lower() != "adamw":
            errors.append("restore150 requires TRAIN.OPTIMIZER=adamw")
        if not bool(_get(train, "USE_EMA", False)):
            errors.append("restore150 requires TRAIN.USE_EMA=true")
        if int(_get(train, "EARLY_STOPPING_PATIENCE_EPOCHS", -1)) != 0:
            errors.append(
                "restore150 requires TRAIN.EARLY_STOPPING_PATIENCE_EPOCHS=0"
            )
    return errors
