"""Paper-grade experiment orchestrator for cross-domain FFB maturity detection.

This script does not replace ``full_ssod_ffb_pipeline.py``. It builds a
reproducible command matrix for the SCI/Q2-oriented experiments and can execute
them sequentially when requested.

Recommended workflow:
    python pythonProject1/run_paper_experiments.py --dry-run
    python pythonProject1/run_paper_experiments.py --preset smoke --execute
    python pythonProject1/run_paper_experiments.py --preset formal --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass
class Experiment:
    name: str
    role: str
    command: List[str]
    expected_summary: Path
    tags: List[str] = field(default_factory=list)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def quote_command(parts: Iterable[str]) -> str:
    rendered = []
    for part in parts:
        if any(ch in str(part) for ch in (" ", "\t", "&", "(", ")")):
            rendered.append(f'"{part}"')
        else:
            rendered.append(str(part))
    return " ".join(rendered)


def model2_command(
    cfg: dict,
    output_root: Path,
    epochs: int,
    seed: int,
    *,
    base_model: str | None = None,
    class_balanced: bool = False,
    ssod: bool = False,
    hard_class_relaxation: bool = False,
    hard_class_relabel: bool = False,
    imgsz: int | None = None,
    batch: int | None = None,
    scene_quota: int | None = None,
    balanced_target: int | None = None,
    skip_protocol: bool = False,
    train_cache: str = "none",
    extra_flags: List[str] | None = None,
) -> List[str]:
    model_cfg = cfg["model"]
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_model2_adaptation.py")),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    if base_model:
        command.extend(["--base-model", base_model])
    else:
        command.extend(["--base-model", model_cfg.get("baseline_model_path", model_cfg.get("base_model", "yolov8n.pt"))])
    if class_balanced:
        command.append("--class-balanced-external")
    if ssod:
        command.append("--run-ssod-after-adaptation")
    if hard_class_relaxation:
        # Target the classes that were previously rejected by SSOD most often.
        command.extend(["--class-threshold-overrides", "abnormal=0.45,under_ripe=0.45"])
        command.extend(["--quality-threshold-by-class", "abnormal=0.62,under_ripe=0.62"])
        command.extend(["--base-conf-threshold", "0.50"])
    if hard_class_relabel:
        command.append("--enable-folder-guided-hard-class-relabel")
        command.extend(["--hard-classes", "abnormal,under_ripe"])
        command.extend(["--hard-class-source-map", "under_ripe:unripe|ripe,abnormal:empty|unripe|overripe"])
        command.extend(["--hard-class-min-quality", "0.80"])
        command.extend(["--hard-class-min-conf", "0.55"])
        command.extend(["--hard-class-min-support", "1"])
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])
    if batch is not None:
        command.extend(["--batch", str(batch)])
    if scene_quota is not None:
        command.extend(["--scene-quota-per-iteration", str(scene_quota)])
    if balanced_target is not None:
        command.extend(["--balanced-target-images-per-class", str(balanced_target)])
    if skip_protocol:
        command.append("--skip-protocol-reports")
        command.append("--reuse-prepared-dataset")
    if train_cache and train_cache != "none":
        command.extend(["--train-cache", train_cache])
    if extra_flags:
        command.extend(extra_flags)
    return command


def baseline_command(cfg: dict, output_root: Path, epochs: int, seed: int, base_model: str, *, imgsz: int | None = None, skip_protocol: bool = False, train_cache: str = "none") -> List[str]:
    datasets = cfg["datasets"]
    model_cfg = cfg["model"]
    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--output-root",
        str(output_root),
        "--original-dataset",
        datasets["in_domain_sciencedb"],
        "--external-test-dataset",
        datasets["external_preprocessed_ffb"],
        "--new-images",
        datasets.get("unlabeled_new_images", ""),
        "--base-model",
        base_model,
        "--max-epochs",
        str(epochs),
        "--self-training-iterations",
        "0",
        "--imgsz",
        str(imgsz or model_cfg.get("imgsz", 640)),
        "--batch",
        str(model_cfg.get("batch", 8)),
        "--device",
        str(model_cfg.get("device", "0")),
        "--seed",
        str(seed),
    ]
    if skip_protocol:
        command.append("--skip-protocol-reports")
        command.append("--reuse-prepared-dataset")
    if train_cache and train_cache != "none":
        command.extend(["--train-cache", train_cache])
    return command


def build_experiments(cfg: dict, preset: str, output_base: Path, seeds: List[int], epochs: int) -> List[Experiment]:
    experiments: List[Experiment] = []
    model_zoo = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt", "yolov10n.pt"]
    if preset in {"smoke", "stage1", "hard_relabel"}:
        model_zoo = ["yolov8n.pt"]

    for seed in seeds:
        seed_root = output_base / f"seed_{seed}"
        if preset == "hard_relabel":
            seed_prepared = output_base / f"seed_{seed}" / "stage1_model2_balanced" / "datasets" / "prepared_supervised_merged"
            shared_prepared = output_base / "seed_42" / "stage1_model2_balanced" / "datasets" / "prepared_supervised_merged"
            source_prepared = seed_prepared if (seed_prepared / "data.yaml").exists() else shared_prepared
            root = seed_root / "stage1_model2_balanced_ssod_hard_relabel"
            experiments.append(
                Experiment(
                    name=f"stage1_model2_balanced_ssod_hard_relabel_seed{seed}",
                    role="stage1_ssod_folder_guided_hard_class_relabel",
                    command=model2_command(
                        cfg,
                        root,
                        epochs,
                        seed,
                        class_balanced=True,
                        ssod=True,
                        hard_class_relaxation=True,
                        hard_class_relabel=True,
                        imgsz=416,
                        batch=4,
                        scene_quota=100,
                        balanced_target=800,
                        skip_protocol=True,
                        train_cache="disk",
                        extra_flags=["--prepared-supervised-root", str(source_prepared)],
                    ),
                    expected_summary=root / "reports" / "summary_metrics.csv",
                    tags=["stage1", "model2", "ssod", "hard_class_relabel"],
                )
            )
            continue
        if preset == "stage1":
            model2_root = seed_root / "stage1_model2_balanced"
            experiments.append(
                Experiment(
                    name=f"stage1_model2_balanced_seed{seed}",
                    role="stage1_main_model_fast_10epoch",
                    command=model2_command(
                        cfg,
                        model2_root,
                        epochs,
                        seed,
                        class_balanced=True,
                        imgsz=416,
                        batch=4,
                        scene_quota=100,
                        balanced_target=800,
                        skip_protocol=True,
                        train_cache="disk",
                    ),
                    expected_summary=model2_root / "reports" / "summary_metrics.csv",
                    tags=["stage1", "model2", "balanced"],
                )
            )
            ssod_root = seed_root / "stage1_model2_balanced_ssod_relaxed"
            experiments.append(
                Experiment(
                    name=f"stage1_model2_balanced_ssod_relaxed_seed{seed}",
                    role="stage1_ssod_relaxed_fast_10epoch",
                    command=model2_command(
                        cfg,
                        ssod_root,
                        epochs,
                        seed,
                        class_balanced=True,
                        ssod=True,
                        hard_class_relaxation=True,
                        imgsz=416,
                        batch=4,
                        scene_quota=100,
                        balanced_target=800,
                        skip_protocol=True,
                        train_cache="disk",
                        extra_flags=[
                            "--prepared-supervised-root",
                            str(model2_root / "datasets" / "prepared_supervised_merged"),
                        ],
                    ),
                    expected_summary=ssod_root / "reports" / "summary_metrics.csv",
                    tags=["stage1", "model2", "ssod", "hard_class_relaxation"],
                )
            )
            continue

        for base_model in model_zoo:
            safe_model = base_model.replace(".pt", "").replace("/", "_").replace("\\", "_")
            root = seed_root / f"baseline_{safe_model}"
            experiments.append(
                Experiment(
                    name=f"baseline_{safe_model}_seed{seed}",
                    role="modern_detector_baseline",
                    command=baseline_command(cfg, root, epochs, seed, base_model, skip_protocol=(preset == "smoke"), imgsz=(416 if preset == "smoke" else None)),
                    expected_summary=root / "reports" / "summary_metrics.csv",
                    tags=["baseline", safe_model],
                )
            )

            adapted_root = seed_root / f"model2_balanced_{safe_model}"
            experiments.append(
                Experiment(
                    name=f"model2_balanced_{safe_model}_seed{seed}",
                    role="modern_detector_class_balanced_external_adaptation",
                    command=model2_command(
                        cfg,
                        adapted_root,
                        epochs,
                        seed,
                        base_model=base_model,
                        class_balanced=True,
                        imgsz=(416 if preset == "smoke" else None),
                        scene_quota=(100 if preset == "smoke" else None),
                        balanced_target=800,
                        skip_protocol=(preset == "smoke"),
                    ),
                    expected_summary=adapted_root / "reports" / "summary_metrics.csv",
                    tags=["model2", "balanced", safe_model],
                )
            )

        model2_root = seed_root / "model2_balanced_from_sciencedb_baseline"
        experiments.append(
            Experiment(
                name=f"model2_balanced_seed{seed}",
                role="main_model_class_balanced_external_adaptation",
                command=model2_command(cfg, model2_root, epochs, seed, class_balanced=True, balanced_target=800),
                expected_summary=model2_root / "reports" / "summary_metrics.csv",
                tags=["main", "model2", "balanced"],
            )
        )

        ssod_root = seed_root / "model2_balanced_ssod_relaxed_hard_classes"
        experiments.append(
            Experiment(
                name=f"model2_balanced_ssod_relaxed_seed{seed}",
                role="ssod_extension_with_hard_class_relaxation",
                command=model2_command(
                    cfg,
                    ssod_root,
                    epochs,
                    seed,
                    class_balanced=True,
                    ssod=True,
                    hard_class_relaxation=True,
                    balanced_target=800,
                ),
                expected_summary=ssod_root / "reports" / "summary_metrics.csv",
                tags=["model2", "ssod", "hard_class_relaxation"],
            )
        )

        for ablation, flags in {
            "ssod_no_consistency": ["--disable-consistency"],
            "ssod_no_quality_score": ["--disable-quality-scoring"],
            "ssod_fixed_threshold": ["--disable-dynamic-thresholds"],
            "ssod_allow_folder_mismatch": ["--allow-folder-mismatch"],
        }.items():
            root = seed_root / ablation
            experiments.append(
                Experiment(
                    name=f"{ablation}_seed{seed}",
                    role="ssod_ablation",
                    command=model2_command(
                        cfg,
                        root,
                        epochs,
                        seed,
                        class_balanced=True,
                        ssod=True,
                        hard_class_relaxation=True,
                        extra_flags=flags,
                        balanced_target=800,
                    ),
                    expected_summary=root / "reports" / "summary_metrics.csv",
                    tags=["ablation", ablation],
                )
            )

    return experiments


def write_manifest(experiments: List[Experiment], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": exp.name,
            "role": exp.role,
            "tags": ";".join(exp.tags),
            "expected_summary": str(exp.expected_summary),
            "command": quote_command(exp.command),
        }
        for exp in experiments
    ]
    with (output_dir / "paper_experiment_manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "role", "tags", "expected_summary", "command"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "paper_experiment_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiments(experiments: List[Experiment], resume: bool) -> None:
    for index, exp in enumerate(experiments, start=1):
        if resume and exp.expected_summary.exists():
            print(f"[{index}/{len(experiments)}] SKIP existing: {exp.name}")
            continue
        print(f"[{index}/{len(experiments)}] RUN {exp.name}")
        print(quote_command(exp.command))
        subprocess.run(exp.command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or execute SCI/Q2 paper experiment matrix.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-base", type=Path, default=Path("LOCAL_DATA_ROOT/paper_formal_experiments"))
    parser.add_argument("--preset", choices=["stage1", "hard_relabel", "smoke", "formal"], default="stage1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Defaults: smoke=42, formal=42,3407,2026.")
    parser.add_argument("--execute", action="store_true", help="Actually run experiments. Without this flag only manifests are written.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs if args.epochs is not None else (10 if args.preset in {"stage1", "hard_relabel"} else (1 if args.preset == "smoke" else 50))
    default_seeds = "42" if args.preset in {"stage1", "hard_relabel", "smoke"} else "42,3407,2026"
    seeds = [int(item.strip()) for item in (args.seeds or default_seeds).split(",") if item.strip()]
    experiments = build_experiments(cfg, args.preset, args.output_base, seeds, epochs)

    manifest_dir = Path(cfg["project"]["output_root"]) / "reports" / "paper_experiment_plan"
    write_manifest(experiments, manifest_dir)
    print(f"Wrote experiment manifest: {manifest_dir}")
    print(f"Experiment count: {len(experiments)}; epochs={epochs}; seeds={seeds}; preset={args.preset}")
    if args.execute:
        run_experiments(experiments, resume=args.resume)
    else:
        print("Dry run only. Add --execute to launch training.")


if __name__ == "__main__":
    main()

