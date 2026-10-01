#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
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
        idx, free = [
            x.strip()
            for x in line.split(",")[:2]
        ]

        if int(idx) == gpu:
            return int(float(free))

    raise RuntimeError(
        f"GPU {gpu} not found in nvidia-smi"
    )


def wait_gpu(
    gpu: int,
    min_free_mib: int,
    poll_seconds: int,
) -> None:
    while True:
        free = get_free_mib(gpu)

        if free >= min_free_mib:
            print(
                f"[GPU READY] "
                f"physical_gpu={gpu} "
                f"free={free}MiB "
                f"required={min_free_mib}MiB",
                flush=True,
            )
            return

        print(
            f"[GPU WAIT] "
            f"physical_gpu={gpu} "
            f"free={free}MiB "
            f"required={min_free_mib}MiB",
            flush=True,
        )

        time.sleep(
            poll_seconds
        )


def run_logged(
    cmd: list[str],
    log_path: Path,
    env: dict,
) -> None:
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[CMD] "
        + " ".join(cmd),
        flush=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log:
        log.write(
            "\n[CMD] "
            + " ".join(cmd)
            + "\n"
        )

        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"Command failed rc={rc}; "
            f"see {log_path}"
        )


def resolve_project_path(
    project: Path,
    value: str | Path,
) -> Path:
    p = Path(value)

    if not p.is_absolute():
        p = project / p

    return p.resolve()


def run_name_from_cfg(
    cfg_path: Path,
) -> str:
    with cfg_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    clip_model = (
        cfg
        .get("MODEL", {})
        .get(
            "CLIP_MODEL",
            "unimedclip",
        )
    )

    backbone = str(
        cfg
        .get("MODEL", {})
        .get(
            "BACKBONE",
            "ViT-B/16",
        )
    ).replace("/", "-")

    run_tag = (
        cfg
        .get("M1", {})
        .get("RUN_TAG")
        or
        cfg
        .get("TRAIN", {})
        .get("RUN_TAG")
    )

    if not run_tag:
        raise RuntimeError(
            f"RUN_TAG missing in {cfg_path}"
        )

    return (
        f"MedCLIPSeg_"
        f"{clip_model}_"
        f"{backbone}_"
        f"{run_tag}"
    )


def validate_direct_original_prompt(
    *,
    project: Path,
    cfg_path: Path,
    target: str,
    expected_cases: int,
) -> None:
    with cfg_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    dataset_name = str(
        cfg["DATASET"]["NAME"]
    ).strip()

    # Fail closed: OOD target name in manifest and config must match.
    if dataset_name != target:
        raise RuntimeError(
            "DG target/config mismatch: "
            f"manifest={target} "
            f"config_DATASET.NAME={dataset_name}"
        )

    expected_prompt_dir = (
        project /
        "data" /
        dataset_name /
        "Prompts_Folder"
    ).resolve()

    configured_prompt_dir = resolve_project_path(
        project,
        cfg["DATASET"]["TEXT_PROMPT_PATH"],
    )

    if (
        configured_prompt_dir
        != expected_prompt_dir
    ):
        raise RuntimeError(
            "DG DIRECT-PROMPT CONTRACT FAILED:\n"
            f"target={target}\n"
            f"configured={configured_prompt_dir}\n"
            f"required={expected_prompt_dir}\n"
            "Domain-generalization evaluation "
            "must use the original dataset "
            "Prompts_Folder directly."
        )

    prompt_file = (
        expected_prompt_dir /
        "Test_text_original.xlsx"
    )

    if not prompt_file.is_file():
        raise RuntimeError(
            f"Original prompt file missing: "
            f"{prompt_file}"
        )

    prompt_df = pd.read_excel(
        prompt_file
    )

    if (
        len(prompt_df)
        != int(expected_cases)
    ):
        raise RuntimeError(
            f"{target} prompt row count "
            f"{len(prompt_df)} "
            f"!= expected cases "
            f"{expected_cases}"
        )

    required_columns = {
        "Image",
        "Description",
    }

    missing = (
        required_columns
        - set(prompt_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{target} original prompt "
            f"missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"[PROMPT PASS] "
        f"{target} "
        f"direct_original={prompt_file} "
        f"rows={len(prompt_df)}",
        flush=True,
    )


def summarize_csv(
    path: Path,
) -> dict:
    df = pd.read_csv(path)

    if (
        "DSC" not in df.columns
        or
        "NSD" not in df.columns
    ):
        raise RuntimeError(
            f"Metric columns missing in {path}"
        )

    return {
        "cases":
            int(len(df)),
        "dsc_percent":
            float(
                df["DSC"].mean()
                * 100.0
            ),
        "nsd_percent":
            float(
                df["NSD"].mean()
                * 100.0
            ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project",
        required=True,
    )

    ap.add_argument(
        "--python",
        required=True,
    )

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--gpu",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--min-free-mib",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
    )

    args = ap.parse_args()

    project = Path(
        args.project
    ).resolve()

    py = str(
        Path(
            args.python
        ).resolve()
    )

    manifest_path = Path(
        args.manifest
    ).resolve()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    seed = int(
        manifest["seed"]
    )

    run_root = Path(
        manifest["run_root"]
    ).resolve()

    base_env = os.environ.copy()

    base_env[
        "PYTORCH_ALLOC_CONF"
    ] = "expandable_segments:True"

    base_env.pop(
        "PYTORCH_CUDA_ALLOC_CONF",
        None,
    )

    base_env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(args.gpu)

    base_env["PYTHONPATH"] = (
        f"{project}:"
        f"{project / 'utils'}"
        + (
            ":"
            + base_env["PYTHONPATH"]
            if base_env.get(
                "PYTHONPATH"
            )
            else ""
        )
    )

    targets = manifest["targets"]

    total = len(targets)

    for i, e in enumerate(
        targets,
        1,
    ):
        source = str(
            e["source"]
        )

        target = str(
            e["target"]
        )

        expected_cases = int(
            e["expected_cases"]
        )

        cfg = resolve_project_path(
            project,
            e["target_config"],
        )

        ckpt = resolve_project_path(
            project,
            e["checkpoint"],
        )

        out = resolve_project_path(
            project,
            e["output_dir"],
        )

        state = resolve_project_path(
            project,
            e["state_dir"],
        )

        state.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = (
            state /
            "SUCCESS.lock"
        )

        result_json = (
            state /
            "RESULT.json"
        )

        detail_log = (
            run_root /
            "logs" /
            f"{i:02d}_"
            f"{source}_to_{target}.log"
        )

        if (
            success.is_file()
            and
            result_json.is_file()
        ):
            print(
                f"[SKIP {i:02d}/{total}] "
                f"{source}->{target}: "
                f"already complete",
                flush=True,
            )
            continue

        print(
            "=" * 88,
            flush=True,
        )

        print(
            f"[DG {i:02d}/{total}] "
            f"{source} -> {target} "
            f"| no adaptation "
            f"| seed={seed}",
            flush=True,
        )

        print(
            f"checkpoint={ckpt}",
            flush=True,
        )

        print(
            f"config={cfg}",
            flush=True,
        )

        print(
            "=" * 88,
            flush=True,
        )

        if not cfg.is_file():
            raise RuntimeError(
                f"Target config missing: "
                f"{cfg}"
            )

        if not ckpt.is_file():
            raise RuntimeError(
                f"Source checkpoint missing: "
                f"{ckpt}"
            )

        # ----------------------------------------------------
        # 1. Frozen source checkpoint contract
        # ----------------------------------------------------
        if (
            sha256(ckpt)
            !=
            e["checkpoint_sha256"]
        ):
            raise RuntimeError(
                f"Source checkpoint SHA changed: "
                f"{ckpt}"
            )

        # ----------------------------------------------------
        # 2. Direct original target-prompt contract
        # ----------------------------------------------------
        validate_direct_original_prompt(
            project=project,
            cfg_path=cfg,
            target=target,
            expected_cases=expected_cases,
        )

        # ----------------------------------------------------
        # 2b. Exact OOD Test-set audit
        # ----------------------------------------------------
        with cfg.open("r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f)

        test_root = resolve_project_path(
            project,
            cfg_obj["DATASET"]["TEST_PATH"],
        )

        image_root = test_root / "img"
        label_root = test_root / "label"

        valid_ext = {
            ".png", ".jpg", ".jpeg",
            ".bmp", ".tif", ".tiff",
        }

        test_images = sorted(
            x for x in image_root.iterdir()
            if x.is_file()
            and x.suffix.lower() in valid_ext
        )

        test_labels = sorted(
            x for x in label_root.iterdir()
            if x.is_file()
            and x.suffix.lower() in valid_ext
        )

        if len(test_images) != expected_cases:
            raise RuntimeError(
                f"{target}: Test image count="
                f"{len(test_images)} != expected="
                f"{expected_cases}; root={image_root}"
            )

        if len(test_labels) != expected_cases:
            raise RuntimeError(
                f"{target}: Test GT count="
                f"{len(test_labels)} != expected="
                f"{expected_cases}; root={label_root}"
            )

        print(
            f"[TESTSET PASS] {target} "
            f"images={len(test_images)} "
            f"labels={len(test_labels)} "
            f"root={test_root}",
            flush=True,
        )

        # ----------------------------------------------------
        # 3. GPU wait
        # ----------------------------------------------------
        wait_gpu(
            args.gpu,
            args.min_free_mib,
            args.poll_seconds,
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_name = run_name_from_cfg(
            cfg
        )

        # ----------------------------------------------------
        # 4. OOD test:
        #    frozen source checkpoint,
        #    target data only,
        #    original target text,
        #    no adaptation.
        # ----------------------------------------------------
        test_cmd = [
            py,
            "-u",
            str(
                project /
                "test.py"
            ),
            "--config-file",
            str(cfg),
            "--seed",
            str(seed),
            "--split",
            "test",
            "--prompt_design",
            "original",
            "--num-samples",
            "30",
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out),
        ]

        run_logged(
            test_cmd,
            detail_log,
            base_env,
        )

        result_root = (
            out /
            target /
            "seg_results" /
            f"seed{seed}"
        )

        # ----------------------------------------------------
        # 5. Evaluation
        # ----------------------------------------------------
        eval_specs = [
            (
                "BaseNative",
                "true2d",
                "test_BaseNative_true2d.csv",
            ),
            (
                "M1Native",
                "true2d",
                "test_M1Native_true2d.csv",
            ),
            (
                "BaseNative",
                "paper_legacy",
                "test_BaseNative_paper_legacy.csv",
            ),
            (
                "M1Native",
                "paper_legacy",
                "test_M1Native_paper_legacy.csv",
            ),
        ]

        for (
            suffix,
            nsd_mode,
            csv_name,
        ) in eval_specs:
            cmd = [
                py,
                "-u",
                str(
                    project /
                    "utils/eval.py"
                ),
                "--config-file",
                str(cfg),
                "--seed",
                str(seed),
                "--split",
                "test",
                "--output-dir",
                str(out),
                "--result-name",
                f"{run_name}_{suffix}",
                "--csv-name",
                csv_name,
                "--nsd-mode",
                nsd_mode,
            ]

            run_logged(
                cmd,
                detail_log,
                base_env,
            )

        # ----------------------------------------------------
        # 6. DG-specific paired statistics.
        #
        # IMPORTANT:
        # Use full image filename Name as pair key.
        #
        # Numeric Case_ID is NOT unique in BUSUC:
        # 00142.png and 0142.png both collapse to 142.
        # ----------------------------------------------------
        for mode in (
            "true2d",
            "paper_legacy",
        ):
            cmd = [
                py,
                "-u",
                str(
                    project /
                    "tools/"
                    "compare_ucfnrt_dg_paired.py"
                ),
                "--base-csv",
                str(
                    result_root /
                    f"test_BaseNative_{mode}.csv"
                ),
                "--m1-csv",
                str(
                    result_root /
                    f"test_M1Native_{mode}.csv"
                ),
                "--output-prefix",
                str(
                    result_root /
                    f"paired_{mode}"
                ),
                "--protocol-label",
                (
                    f"DG_"
                    f"{source}_to_{target}_"
                    f"UC_FNRT_{mode}_"
                    f"seed{seed}"
                ),
            ]

            run_logged(
                cmd,
                detail_log,
                base_env,
            )

        # ----------------------------------------------------
        # 7. Summaries
        # ----------------------------------------------------
        b_true = summarize_csv(
            result_root /
            "test_BaseNative_true2d.csv"
        )

        m_true = summarize_csv(
            result_root /
            "test_M1Native_true2d.csv"
        )

        b_legacy = summarize_csv(
            result_root /
            "test_BaseNative_paper_legacy.csv"
        )

        m_legacy = summarize_csv(
            result_root /
            "test_M1Native_paper_legacy.csv"
        )

        for label, summary in (
            ("Base true2d", b_true),
            ("M1 true2d", m_true),
            ("Base legacy", b_legacy),
            ("M1 legacy", m_legacy),
        ):
            if (
                summary["cases"]
                !=
                expected_cases
            ):
                raise RuntimeError(
                    f"{target} {label}: "
                    f"evaluated cases="
                    f"{summary['cases']} "
                    f"!= expected="
                    f"{expected_cases}"
                )

        result = {
            "source":
                source,
            "target":
                target,
            "display_source":
                e["display_source"],
            "display_target":
                e["display_target"],
            "seed":
                seed,
            "protocol":
                "source-trained / target-tested / no-adaptation",
            "prompt_protocol":
                "direct original dataset Test_text_original.xlsx",
            "prompt_path":
                str(
                    (
                        project /
                        "data" /
                        target /
                        "Prompts_Folder" /
                        "Test_text_original.xlsx"
                    ).resolve()
                ),
            "paired_key":
                "Name",
            "checkpoint":
                str(ckpt),
            "checkpoint_sha256":
                e[
                    "checkpoint_sha256"
                ],
            "cases":
                m_true["cases"],
            "base_dsc_percent":
                b_true[
                    "dsc_percent"
                ],
            "m1_dsc_percent":
                m_true[
                    "dsc_percent"
                ],
            "delta_dsc_percent":
                (
                    m_true[
                        "dsc_percent"
                    ]
                    -
                    b_true[
                        "dsc_percent"
                    ]
                ),
            "base_nsd_true2d_percent":
                b_true[
                    "nsd_percent"
                ],
            "m1_nsd_true2d_percent":
                m_true[
                    "nsd_percent"
                ],
            "delta_nsd_true2d_percent":
                (
                    m_true[
                        "nsd_percent"
                    ]
                    -
                    b_true[
                        "nsd_percent"
                    ]
                ),
            "base_nsd_paper_legacy_percent":
                b_legacy[
                    "nsd_percent"
                ],
            "m1_nsd_paper_legacy_percent":
                m_legacy[
                    "nsd_percent"
                ],
            "delta_nsd_paper_legacy_percent":
                (
                    m_legacy[
                        "nsd_percent"
                    ]
                    -
                    b_legacy[
                        "nsd_percent"
                    ]
                ),
            "result_root":
                str(result_root),
            "log":
                str(detail_log),
        }

        result_json.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        success.write_text(
            "PASS\n",
            encoding="utf-8",
        )

        print(
            f"[PASS] "
            f"{source}->{target} "
            f"DSC="
            f"{result['m1_dsc_percent']:.2f}% "
            f"legacyNSD="
            f"{result['m1_nsd_paper_legacy_percent']:.2f}% "
            f"true2dNSD="
            f"{result['m1_nsd_true2d_percent']:.2f}%",
            flush=True,
        )

    # --------------------------------------------------------
    # 8. Final Table-2 collector
    # --------------------------------------------------------
    cmd = [
        py,
        "-u",
        str(
            project /
            "tools/"
            "collect_ucfnrt_domain_generalization.py"
        ),
        "--project",
        str(project),
        "--manifest",
        str(manifest_path),
    ]

    run_logged(
        cmd,
        run_root /
        "logs/99_collect.log",
        base_env,
    )

    print(
        "[PASS] ALL DOMAIN-GENERALIZATION "
        f"TARGETS COMPLETE: {run_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
