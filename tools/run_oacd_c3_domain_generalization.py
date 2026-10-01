#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

PAIRS = [
    ("BUSI",   "BUSBRA",     282, "Breast Ultrasound"),
    ("BUSI",   "BUSUC",      122, "Breast Ultrasound"),
    ("BUSI",   "BUID",        35, "Breast Ultrasound"),
    ("BUSI",   "UDIAT",       25, "Breast Ultrasound"),
    ("Kvasir", "ColonDB",    360, "Polyp Endoscopy"),
    ("Kvasir", "ClinicDB",    61, "Polyp Endoscopy"),
    ("Kvasir", "CVC300",      60, "Polyp Endoscopy"),
    ("Kvasir", "BKAI",       100, "Polyp Endoscopy"),
    ("BTMRI",  "BRISC",     1000, "Brain MRI"),
    ("ISIC",   "UWaterloo",   41, "Skin Dermatoscopy"),
]

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_free_mib(gpu: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.splitlines():
        idx, free = [x.strip() for x in line.split(",")[:2]]
        if int(idx) == gpu:
            return int(float(free))
    raise RuntimeError(f"GPU {gpu} not found")


def wait_gpu(gpu: int, min_free_mib: int, poll_seconds: int) -> None:
    while True:
        free = get_free_mib(gpu)
        if free >= min_free_mib:
            print(
                f"[GPU READY] physical_gpu={gpu} free={free}MiB required={min_free_mib}MiB",
                flush=True,
            )
            return
        print(
            f"[GPU WAIT] physical_gpu={gpu} free={free}MiB required={min_free_mib}MiB",
            flush=True,
        )
        time.sleep(poll_seconds)


def run_logged(cmd: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD] " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[CMD] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}; see {log_path}")


def count_files(root: Path) -> int:
    if not root.is_dir():
        raise RuntimeError(f"Missing directory: {root}")
    return sum(
        1 for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def resolve_target_folder(project: Path, target: str) -> str:
    if target != "UWaterloo":
        return target
    for name in ("UWaterloo", "UWaterlooSkinCancer"):
        if (project / "data" / name).is_dir():
            return name
    raise RuntimeError(
        "Neither data/UWaterloo nor data/UWaterlooSkinCancer exists"
    )


def source_tag(source: str) -> str:
    return f"SEMLT_UCFNRT_OACD_C3_FORMAL100_{source}"


def discover_source_checkpoint(
    source_root: Path,
    source: str,
    seed: int,
) -> Path:
    model_dir = source_root / source / source / "trained_models" / f"seed{seed}"
    tag = source_tag(source)
    candidates = sorted(model_dir.glob(f"*{tag}_last_epoch.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Formal DG requires exactly one physical last-epoch checkpoint for {source}. "
            f"Found {len(candidates)} under {model_dir}: {[p.name for p in candidates]}"
        )
    return candidates[0].resolve()


def build_target_config(
    *,
    project: Path,
    base_cfg: dict,
    source: str,
    target_display: str,
    seed: int,
) -> tuple[Path, str]:
    target_folder = resolve_target_folder(project, target_display)
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("DATASET", {})
    cfg["DATASET"]["NAME"] = target_folder
    cfg["DATASET"]["TRAIN_PATH"] = f"./data/{source}/Train_Folder/"
    cfg["DATASET"]["VAL_PATH"] = f"./data/{source}/Val_Folder/"
    cfg["DATASET"]["TEST_PATH"] = f"./data/{target_folder}/Test_Folder/"
    cfg["DATASET"]["TEXT_PROMPT_PATH"] = f"./data/{target_folder}/Prompts_Folder/"

    tag = source_tag(source)
    cfg.setdefault("M1", {})["RUN_TAG"] = tag
    cfg.setdefault("TRAIN", {})["RUN_TAG"] = tag
    cfg.setdefault("TEST", {})["NUM_SAMPLES"] = 30

    out_dir = project / "configs" / "oacd_c3_domain_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{source}_to_{target_display}_OACD_C3_DG.yaml"
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out, target_folder


def validate_target(
    *,
    project: Path,
    target_folder: str,
    expected_cases: int,
) -> Path:
    test_root = project / "data" / target_folder / "Test_Folder"
    n_img = count_files(test_root / "img")
    n_gt = count_files(test_root / "label")
    if n_img != expected_cases or n_gt != expected_cases:
        raise RuntimeError(
            f"{target_folder}: expected {expected_cases} Test pairs, "
            f"found images={n_img}, labels={n_gt}"
        )

    prompt = project / "data" / target_folder / "Prompts_Folder" / "Test_text_original.xlsx"
    if not prompt.is_file():
        raise RuntimeError(f"Missing paper-matched target prompt file: {prompt}")
    df = pd.read_excel(prompt)
    if len(df) != expected_cases:
        raise RuntimeError(
            f"{target_folder}: prompt rows={len(df)} != expected={expected_cases}"
        )
    required = {"Image", "Description"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{target_folder}: prompt file missing columns {sorted(missing)}"
        )
    print(
        f"[TARGET PASS] {target_folder} cases={expected_cases} prompt={prompt}",
        flush=True,
    )
    return prompt


def run_name(cfg_path: Path) -> str:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    clip = cfg["MODEL"]["CLIP_MODEL"]
    backbone = str(cfg["MODEL"]["BACKBONE"]).replace("/", "-")
    tag = cfg["M1"]["RUN_TAG"]
    return f"MedCLIPSeg_{clip}_{backbone}_{tag}"


def summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if "DSC" not in df.columns or "NSD" not in df.columns:
        raise RuntimeError(f"Metric columns missing in {path}")
    return {
        "cases": int(len(df)),
        "dsc_percent": float(df["DSC"].mean() * 100.0),
        "nsd_percent": float(df["NSD"].mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-mib", type=int, default=20000)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument(
        "--sources",
        default="BUSI,Kvasir,BTMRI,ISIC",
        help="Comma-separated source subset; formal full table should use all four.",
    )
    args = ap.parse_args()

    project = Path(args.project).resolve()
    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    py = str(Path(args.python).resolve())
    seed = int(args.seed)
    selected_sources = {x.strip() for x in args.sources.split(",") if x.strip()}

    allowed = {"BUSI", "Kvasir", "BTMRI", "ISIC"}
    unknown = selected_sources - allowed
    if unknown:
        raise RuntimeError(f"Unknown sources: {sorted(unknown)}")

    base_cfg_path = project / "configs" / "ucfnrt_oacd_formal100" / "OACD_C3_LOCKED_FORMAL100.yaml"
    if not base_cfg_path.is_file():
        raise RuntimeError(f"Missing locked C3 base config: {base_cfg_path}")
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "state").mkdir(parents=True, exist_ok=True)

    # Fail closed before any new OOD target is opened: every selected source must
    # already have its physical last-epoch checkpoint.
    source_ckpts: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for source in sorted(selected_sources):
        ckpt = discover_source_checkpoint(source_root, source, seed)
        source_ckpts[source] = ckpt
        source_hashes[source] = sha256(ckpt)
        print(
            f"[SOURCE LOCKED] {source} checkpoint={ckpt} sha256={source_hashes[source]}",
            flush=True,
        )

    plan = [p for p in PAIRS if p[0] in selected_sources]

    # Build + validate all target configs before running inference.
    prepared = []
    for source, target_display, expected_cases, domain in plan:
        cfg_path, target_folder = build_target_config(
            project=project,
            base_cfg=base_cfg,
            source=source,
            target_display=target_display,
            seed=seed,
        )
        prompt = validate_target(
            project=project,
            target_folder=target_folder,
            expected_cases=expected_cases,
        )
        prepared.append(
            (source, target_display, target_folder, expected_cases, domain, cfg_path, prompt)
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = f"{project}:{project / 'utils'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["HF_HOME"] = str(project / "checkpoints")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLBACKEND"] = "Agg"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    results = []
    total = len(prepared)

    for i, item in enumerate(prepared, 1):
        source, target_display, target_folder, expected_cases, domain, cfg_path, prompt = item
        state_dir = run_root / "state" / f"{source}_to_{target_display}"
        state_dir.mkdir(parents=True, exist_ok=True)
        success = state_dir / "SUCCESS.lock"
        result_json = state_dir / "RESULT.json"

        if success.is_file() and result_json.is_file():
            obj = json.loads(result_json.read_text(encoding="utf-8"))
            results.append(obj)
            print(f"[SKIP {i:02d}/{total}] {source}->{target_display} already complete", flush=True)
            continue

        ckpt = source_ckpts[source]
        log_path = run_root / "logs" / f"{i:02d}_{source}_to_{target_display}.log"
        out = run_root / f"seed{seed}" / source / target_display / "formal_test"
        out.mkdir(parents=True, exist_ok=True)

        print("=" * 100, flush=True)
        print(
            f"[DG {i:02d}/{total}] {source} -> {target_display} "
            f"| source-trained / target-tested / NO adaptation | seed={seed}",
            flush=True,
        )
        print(f"checkpoint={ckpt}", flush=True)
        print(f"config={cfg_path}", flush=True)
        print(f"target_prompt={prompt}", flush=True)
        print("=" * 100, flush=True)

        wait_gpu(args.gpu, args.min_free_mib, args.poll_seconds)

        # Repository runtime protocol validation, if available.
        runtime_preflight = project / "tools" / "preflight_oacd_runtime.py"
        if runtime_preflight.is_file():
            run_logged(
                [
                    py,
                    str(runtime_preflight),
                    "--project", str(project),
                    "--config", str(cfg_path),
                    "--seed", str(seed),
                ],
                log_path,
                env,
            )

        run_logged(
            [
                py, "-u", str(project / "test.py"),
                "--config-file", str(cfg_path),
                "--seed", str(seed),
                "--split", "test",
                "--prompt_design", "original",
                "--num-samples", "30",
                "--checkpoint", str(ckpt),
                "--output-dir", str(out),
            ],
            log_path,
            env,
        )

        rn = run_name(cfg_path)
        result_root = out / target_folder / "seg_results" / f"seed{seed}"

        specs = [
            ("BaseNative", "true2d", "test_BaseNative_true2d.csv"),
            ("M1Native", "true2d", "test_M1Native_true2d.csv"),
            ("BaseNative", "paper_legacy", "test_BaseNative_paper_legacy.csv"),
            ("M1Native", "paper_legacy", "test_M1Native_paper_legacy.csv"),
        ]
        for suffix, nsd_mode, csv_name in specs:
            run_logged(
                [
                    py, "-u", str(project / "utils" / "eval.py"),
                    "--config-file", str(cfg_path),
                    "--seed", str(seed),
                    "--split", "test",
                    "--output-dir", str(out),
                    "--result-name", f"{rn}_{suffix}",
                    "--csv-name", csv_name,
                    "--nsd-mode", nsd_mode,
                ],
                log_path,
                env,
            )

        # Paired Base-vs-M1 stats use full image Name key in the project's
        # DG comparator, avoiding BUSUC numeric-ID collisions.
        comparator = project / "tools" / "compare_ucfnrt_dg_paired.py"
        if comparator.is_file():
            for mode in ("true2d", "paper_legacy"):
                run_logged(
                    [
                        py, "-u", str(comparator),
                        "--base-csv", str(result_root / f"test_BaseNative_{mode}.csv"),
                        "--m1-csv", str(result_root / f"test_M1Native_{mode}.csv"),
                        "--output-prefix", str(result_root / f"paired_{mode}"),
                        "--protocol-label", f"DG_{source}_to_{target_display}_OACD_C3_{mode}_seed{seed}",
                    ],
                    log_path,
                    env,
                )

        b_true = summarize_csv(result_root / "test_BaseNative_true2d.csv")
        m_true = summarize_csv(result_root / "test_M1Native_true2d.csv")
        b_leg = summarize_csv(result_root / "test_BaseNative_paper_legacy.csv")
        m_leg = summarize_csv(result_root / "test_M1Native_paper_legacy.csv")

        for label, summary in (
            ("Base true2d", b_true),
            ("M1 true2d", m_true),
            ("Base legacy", b_leg),
            ("M1 legacy", m_leg),
        ):
            if summary["cases"] != expected_cases:
                raise RuntimeError(
                    f"{source}->{target_display} {label}: cases={summary['cases']} != {expected_cases}"
                )

        obj = {
            "source": source,
            "target": target_display,
            "target_folder": target_folder,
            "domain": domain,
            "seed": seed,
            "protocol": "source-trained / target-tested / no-adaptation",
            "selection": "physical source last epoch; no target validation",
            "mc_samples": 30,
            "prompt_protocol": "target Test_text_original.xlsx (paper-matched)",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": source_hashes[source],
            "cases": expected_cases,
            "base_dsc_percent": b_true["dsc_percent"],
            "m1_dsc_percent": m_true["dsc_percent"],
            "delta_dsc_percent": m_true["dsc_percent"] - b_true["dsc_percent"],
            "base_true2d_nsd_percent": b_true["nsd_percent"],
            "m1_true2d_nsd_percent": m_true["nsd_percent"],
            "delta_true2d_nsd_percent": m_true["nsd_percent"] - b_true["nsd_percent"],
            "base_paper_legacy_nsd_percent": b_leg["nsd_percent"],
            "m1_paper_legacy_nsd_percent": m_leg["nsd_percent"],
            "result_root": str(result_root),
        }
        result_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        success.write_text("PASS\n", encoding="utf-8")
        results.append(obj)

        print(
            f"[PASS] {source}->{target_display} "
            f"M1 DSC={obj['m1_dsc_percent']:.2f}% "
            f"true2d NSD={obj['m1_true2d_nsd_percent']:.2f}% "
            f"legacy NSD={obj['m1_paper_legacy_nsd_percent']:.2f}%",
            flush=True,
        )

    df = pd.DataFrame(results)
    if not df.empty:
        order = {(s, t): i for i, (s, t, _, _) in enumerate(PAIRS)}
        df["_order"] = [order[(s, t)] for s, t in zip(df["source"], df["target"])]
        df = df.sort_values("_order").drop(columns="_order")
    summary_csv = run_root / "DG_SUMMARY.csv"
    df.to_csv(summary_csv, index=False)

    macro = {
        "seed": seed,
        "n_targets": int(len(df)),
        "m1_ood_macro_dsc_percent": float(df["m1_dsc_percent"].mean()) if len(df) else None,
        "m1_ood_macro_true2d_nsd_percent": float(df["m1_true2d_nsd_percent"].mean()) if len(df) else None,
        "base_ood_macro_dsc_percent": float(df["base_dsc_percent"].mean()) if len(df) else None,
        "delta_ood_macro_dsc_percent": float(df["delta_dsc_percent"].mean()) if len(df) else None,
        "source_checkpoint_sha256": source_hashes,
    }
    (run_root / "DG_MACRO.json").write_text(
        json.dumps(macro, indent=2), encoding="utf-8"
    )

    print("=" * 100, flush=True)
    print("[PASS] OACD-C3 PAPER-MATCHED DOMAIN GENERALIZATION COMPLETE", flush=True)
    print(f"summary={summary_csv}", flush=True)
    print(f"macro={run_root / 'DG_MACRO.json'}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
