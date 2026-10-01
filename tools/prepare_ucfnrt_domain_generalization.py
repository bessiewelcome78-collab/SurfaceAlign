#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

EXPECTED_TARGET_CASES = {
    "BUSBRA": 282,
    "BUSUC": 122,
    "BUID": 35,
    "UDIAT": 25,
    "ColonDB": 360,
    "ClinicDB": 61,
    "CVC300": 60,
    "BKAI": 100,
    "BRISC": 1000,
    "UWaterlooSkinCancer": 41,
}

SOURCE_SPECS = {
    "BUSI": {
        "config": "configs/BUSI_SEMLT_UC_FNRT_FORMAL100.yaml",
        "checkpoint": "runs/UCFNRT_FORMAL100_BUSI_S42_20260831_032834/BUSI/trained_models/seed42/MedCLIPSeg_unimedclip_ViT-B-16_SEMLT_UC_FNRT_FORMAL100_BUSI_last_epoch.pth",
        "targets": ["BUSBRA", "BUSUC", "BUID", "UDIAT"],
    },
    "Kvasir": {
        "config": "configs/Kvasir_SEMLT_UC_FNRT_FORMAL100.yaml",
        "checkpoint": "runs/UCFNRT_FORMAL100_Kvasir_S42_20260831_061201/Kvasir/trained_models/seed42/MedCLIPSeg_unimedclip_ViT-B-16_SEMLT_UC_FNRT_FORMAL100_Kvasir_last_epoch.pth",
        "targets": ["ColonDB", "ClinicDB", "CVC300", "BKAI"],
    },
    "BTMRI": {
        "config": "configs/BTMRI_SEMLT_UC_FNRT_FORMAL100.yaml",
        "checkpoint": "runs/UCFNRT_FORMAL100_BTMRI_S42_20260831_032834/BTMRI/trained_models/seed42/MedCLIPSeg_unimedclip_ViT-B-16_SEMLT_UC_FNRT_FORMAL100_BTMRI_last_epoch.pth",
        "targets": ["BRISC"],
    },
    "ISIC": {
        "config": "configs/ISIC_SEMLT_UC_FNRT_FORMAL100.yaml",
        "checkpoint": "runs/UCFNRT_FORMAL100_ISIC_S42_20260831_032834/ISIC/trained_models/seed42/MedCLIPSeg_unimedclip_ViT-B-16_SEMLT_UC_FNRT_FORMAL100_ISIC_last_epoch.pth",
        "targets": ["UWaterlooSkinCancer"],
    },
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_get(d: Any, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def assert_source_protocol(cfg: dict, source: str) -> None:
    errors = []
    checks = [
        (cfg_get(cfg, "DATASET", "NAME"), source, "DATASET.NAME"),
        (bool(cfg_get(cfg, "M1", "SEMLT_UC_FNRT", default=False)), True, "M1.SEMLT_UC_FNRT"),
        (str(cfg_get(cfg, "M1", "GEOTR_M1_FORMAL_PROTOCOL", default="")).lower(), "paper100", "M1.GEOTR_M1_FORMAL_PROTOCOL"),
        (int(cfg_get(cfg, "M1", "GEOTR_TRAIN_POSTERIOR_SAMPLES", default=-1)), 10, "M1.GEOTR_TRAIN_POSTERIOR_SAMPLES"),
        (int(cfg_get(cfg, "M1", "SEMLT_LOCAL_RADIUS_PX", default=-1)), 8, "M1.SEMLT_LOCAL_RADIUS_PX"),
        (int(cfg_get(cfg, "TRAIN", "BATCH_SIZE", default=-1)), 24, "TRAIN.BATCH_SIZE"),
        (int(cfg_get(cfg, "TRAIN", "NUM_EPOCHS", default=-1)), 100, "TRAIN.NUM_EPOCHS"),
        (float(cfg_get(cfg, "TRAIN", "LEARNING_RATE", default=-1)), 3e-4, "TRAIN.LEARNING_RATE"),
        (bool(cfg_get(cfg, "TRAIN", "USE_VALIDATION_SELECTION", default=True)), False, "TRAIN.USE_VALIDATION_SELECTION"),
        (int(cfg_get(cfg, "TEST", "NUM_SAMPLES", default=-1)), 30, "TEST.NUM_SAMPLES"),
        (bool(cfg_get(cfg, "TEST", "USE_LATEST", default=False)), True, "TEST.USE_LATEST"),
    ]
    for got, want, name in checks:
        if isinstance(want, float):
            ok = abs(float(got) - want) < 1e-12
        else:
            ok = got == want
        if not ok:
            errors.append(f"{name}: got={got!r}, expected={want!r}")
    if errors:
        raise RuntimeError(
            f"Source config {source} is not the locked UC-FNRT PAPER100 protocol:\n  - "
            + "\n  - ".join(errors)
        )


def find_test_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "Test_Folder",
        dataset_root / "test",
        dataset_root / "Test",
        dataset_root,
    ]
    for c in candidates:
        if (c / "img").is_dir() and (c / "label").is_dir():
            return c
    raise FileNotFoundError(
        f"Cannot locate test root with img/ and label/ under {dataset_root}. "
        "Expected e.g. data/<target>/Test_Folder/{img,label}."
    )


def count_images(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _norm_stem(stem: str) -> str:
    """Normalize common mask-view suffixes without changing case identity."""
    x = stem.lower().strip()
    # Iteratively remove only well-known representation suffixes.
    pat = re.compile(
        r"(?:[_\-.](?:mask|masks|gt|label|labels|seg|segmentation|binary|bin|multiclass|multi))+$",
        flags=re.IGNORECASE,
    )
    prev = None
    while prev != x:
        prev = x
        x = pat.sub("", x)
    return x


def _binary_occupancy(path: Path):
    """Read a segmentation label as foreground occupancy for equivalence audit.

    BKAI masks can exist in both binary and color-coded multiclass forms. For
    domain-generalization binary polyp segmentation, any non-zero class pixel is
    foreground. This routine is used ONLY to prove that duplicate label views
    encode the same foreground support; it does not alter source data.
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "OpenCV/numpy are required for duplicate-mask equivalence audit"
        ) from exc
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise RuntimeError(f"Cannot read label for equivalence audit: {path}")
    if x.ndim == 2:
        fg = x > 0
    elif x.ndim == 3:
        fg = np.any(x > 0, axis=2)
    else:
        raise RuntimeError(f"Unsupported label ndim={x.ndim}: {path}")
    return fg


def _choose_equivalent_label(cands: list[Path], image_name: str) -> tuple[Path, str]:
    if len(cands) == 1:
        return cands[0], "unique-normalized-match"

    occ = [_binary_occupancy(p) for p in cands]
    shape0 = occ[0].shape
    if any(x.shape != shape0 for x in occ[1:]):
        raise RuntimeError(
            f"Multiple labels matched {image_name}, but shapes differ: "
            + ", ".join(f"{p.name}:{x.shape}" for p, x in zip(cands, occ))
        )

    import numpy as np
    if not all(np.array_equal(occ[0], x) for x in occ[1:]):
        raise RuntimeError(
            f"Multiple non-equivalent labels matched {image_name}; refusing to choose: "
            + ", ".join(p.name for p in cands)
        )

    # Prefer an explicitly binary representation when both binary and color-coded
    # masks are present. Otherwise use a deterministic filename order.
    def rank(p: Path):
        n = p.stem.lower()
        binary = 0 if ("binary" in n or re.search(r"(?:^|[_\-.])bin(?:$|[_\-.])", n)) else 1
        png = 0 if p.suffix.lower() == ".png" else 1
        return (binary, png, len(p.name), p.name.lower())

    chosen = sorted(cands, key=rank)[0]
    return chosen, "equivalent-duplicate-view"


def build_canonical_test_view(raw_test_root: Path, expected: int, view_root: Path, target: str) -> tuple[Path, dict]:
    """Create a read-only symlink view with exactly one proven label per image.

    This is used only when the raw target directory has the correct number of
    images but extra label representations. It never drops an unmatched case and
    never resolves genuinely conflicting annotations silently.
    """
    imgs = image_files(raw_test_root / "img")
    labs = image_files(raw_test_root / "label")
    if len(imgs) != expected:
        raise RuntimeError(
            f"{target}: cannot canonicalize because img={len(imgs)} != expected={expected}"
        )

    by_norm: dict[str, list[Path]] = {}
    for p in labs:
        by_norm.setdefault(_norm_stem(p.stem), []).append(p)

    selections = []
    missing = []
    for img in imgs:
        key = _norm_stem(img.stem)
        cands = by_norm.get(key, [])
        if not cands:
            # Exact basename/stem fallback before failing.
            cands = [p for p in labs if p.name == img.name or p.stem == img.stem]
        if not cands:
            missing.append(img.name)
            continue
        chosen, reason = _choose_equivalent_label(cands, img.name)
        selections.append((img, chosen, reason, [p.name for p in cands]))

    if missing:
        raise RuntimeError(
            f"{target}: {len(missing)} images have no uniquely auditable label; examples={missing[:10]}"
        )
    if len(selections) != expected:
        raise RuntimeError(
            f"{target}: canonical selections={len(selections)} != expected={expected}"
        )

    img_view = view_root / "img"
    lab_view = view_root / "label"
    img_view.mkdir(parents=True, exist_ok=True)
    lab_view.mkdir(parents=True, exist_ok=True)
    # Clean only our generated view, never user source data.
    for d in (img_view, lab_view):
        for q in d.iterdir():
            if q.is_symlink() or q.is_file():
                q.unlink()

    audit_rows = []
    for img, lab, reason, cand_names in selections:
        di = img_view / img.name
        di.symlink_to(img.resolve())
        # Keep the image stem so eval's normalized case matching is unambiguous.
        dl = lab_view / f"{img.stem}{lab.suffix.lower()}"
        if dl.exists() or dl.is_symlink():
            raise RuntimeError(f"{target}: duplicate canonical output name: {dl.name}")
        dl.symlink_to(lab.resolve())
        audit_rows.append({
            "image": img.name,
            "chosen_label": lab.name,
            "reason": reason,
            "candidate_labels": cand_names,
        })

    audit = {
        "target": target,
        "raw_test_root": str(raw_test_root),
        "raw_images": len(imgs),
        "raw_labels": len(labs),
        "canonical_cases": len(audit_rows),
        "selection_rows": audit_rows,
    }
    (view_root / "CANONICAL_LABEL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return view_root, audit


def find_prompt_file(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "Prompts_Folder" / "Test_text_original.xlsx",
        dataset_root / "Prompts_Folder" / "Test_text.xlsx",
        dataset_root / "Prompts_Folder" / "Test_text_original.xls",
        dataset_root / "Prompts_Folder" / "Test_text.xls",
        dataset_root / "Test_text_original.xlsx",
        dataset_root / "Test_text.xlsx",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No existing OOD Test prompt file found under {dataset_root}. "
        "For a fair DG run, do not generate prompts after inspecting target GT. "
        "Provide the paper/protocol prompt file first."
    )


def ensure_prompt_view(prompt_file: Path, view_dir: Path) -> Path:
    """test.py expects Test_text_original.xlsx for --prompt_design original.

    We never modify the prompt contents. If the existing file is named Test_text.xlsx,
    create a symlink view with the expected filename.
    """
    view_dir.mkdir(parents=True, exist_ok=True)
    dst = view_dir / "Test_text_original.xlsx"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(prompt_file.resolve())
    return view_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root-id", required=True)
    ap.add_argument("--output-config-dir", default="configs/ucfnrt_domain_generalization")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    run_root = project / f"runs/UCFNRT_DOMAIN_GENERALIZATION_{args.root_id}" / f"seed{args.seed}"
    config_dir = project / args.output_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "UC-FNRT cross-dataset domain generalization; train source once, direct OOD test, no adaptation",
        "seed": args.seed,
        "root_id": args.root_id,
        "run_root": str(run_root),
        "targets": [],
    }

    for source, spec in SOURCE_SPECS.items():
        source_cfg_path = project / spec["config"]
        ckpt = project / spec["checkpoint"]
        if not source_cfg_path.is_file():
            raise FileNotFoundError(f"Missing source config: {source_cfg_path}")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing already-trained source checkpoint: {ckpt}")

        with source_cfg_path.open("r", encoding="utf-8") as f:
            source_cfg = yaml.safe_load(f)
        assert_source_protocol(source_cfg, source)
        ckpt_sha = sha256(ckpt)

        for target in spec["targets"]:
            dataset_root = project / "data" / target
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"Missing OOD target dataset directory: {dataset_root}")
            raw_test_root = find_test_root(dataset_root)
            n_img = count_images(raw_test_root / "img")
            n_lab = count_images(raw_test_root / "label")
            expected = EXPECTED_TARGET_CASES[target]
            canonical_audit = None
            if n_img != expected:
                raise RuntimeError(
                    f"{target} case-count mismatch versus the MedCLIPSeg Table-S1 protocol: "
                    f"img={n_img}, label={n_lab}, expected={expected}. "
                    "Image split itself is wrong; stop instead of changing the split."
                )
            if n_lab == expected:
                test_root = raw_test_root
            else:
                print(
                    f"[AUDIT] {target}: img={n_img}, raw_label={n_lab}, expected={expected}; "
                    "attempting label-view canonicalization without modifying source data"
                )
                test_root, canonical_audit = build_canonical_test_view(
                    raw_test_root,
                    expected,
                    run_root / "dataset_views" / target / "Test_Folder",
                    target,
                )
                n_view_img = count_images(test_root / "img")
                n_view_lab = count_images(test_root / "label")
                if n_view_img != expected or n_view_lab != expected:
                    raise RuntimeError(
                        f"{target}: canonical view invalid: img={n_view_img}, "
                        f"label={n_view_lab}, expected={expected}"
                    )
                print(
                    f"[PASS] {target}: canonical label view has exactly {expected} cases; "
                    f"audit={test_root / 'CANONICAL_LABEL_AUDIT.json'}"
                )

            prompt_file = find_prompt_file(dataset_root)
            prompt_view = ensure_prompt_view(
                prompt_file,
                run_root / "prompt_views" / target,
            )

            cfg = copy.deepcopy(source_cfg)
            cfg.setdefault("DATASET", {})
            # Critical DG rule: TRAIN/VAL paths remain the source-domain paths.
            # Only the evaluation dataset identity, Test path, and prompt path change.
            cfg["DATASET"]["NAME"] = target
            cfg["DATASET"]["TEST_PATH"] = str(test_root) + "/"
            cfg["DATASET"]["TEXT_PROMPT_PATH"] = str(prompt_view) + "/"
            cfg.setdefault("TEST", {})["NUM_SAMPLES"] = 30
            cfg["TEST"]["USE_LATEST"] = True

            out_cfg = config_dir / f"{source}_to_{target}_UCFNRT_DG.yaml"
            with out_cfg.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            entry = {
                "source": source,
                "target": target,
                "display_source": "Kvasir-SEG" if source == "Kvasir" else source,
                "display_target": "UWaterloo" if target == "UWaterlooSkinCancer" else target,
                "source_config": str(source_cfg_path),
                "target_config": str(out_cfg),
                "checkpoint": str(ckpt),
                "checkpoint_sha256": ckpt_sha,
                "test_root": str(test_root),
                "raw_test_root": str(raw_test_root),
                "canonical_label_audit": (
                    str(test_root / "CANONICAL_LABEL_AUDIT.json")
                    if canonical_audit is not None else None
                ),
                "prompt_file": str(prompt_file.resolve()),
                "prompt_view": str(prompt_view),
                "expected_cases": expected,
                "output_dir": str(run_root / source / target / "formal_test"),
                "state_dir": str(run_root / source / target / "formal_state"),
            }
            manifest["targets"].append(entry)
            print(
                f"[PREPARED] {source:7s} -> {target:24s} "
                f"cases={expected:4d} config={out_cfg.relative_to(project)}"
            )

    manifest_path = run_root / "DG_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[PASS] DG manifest: {manifest_path}")


if __name__ == "__main__":
    main()
