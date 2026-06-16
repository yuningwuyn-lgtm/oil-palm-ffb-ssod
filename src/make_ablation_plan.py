"""Write the paper ablation command matrix without launching training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ablation command table.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("LOCAL_PROJECT_ROOT/paper_framework/reports/ablation_commands.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["project"]["output_root"])
    model2_root = Path("LOCAL_PROJECT_ROOT/paper_framework_model2_adapt_3e")

    rows = [
        {
            "experiment": "model0_sciencedb_baseline",
            "purpose": "ScienceDB/OILPALM-only baseline; proves external domain shift.",
            "command": "python pythonProject1/train_baseline.py",
        },
        {
            "experiment": "model1_full_ssod_new_images",
            "purpose": "Quality-controlled pseudo-label self-training on new_images.",
            "command": (
                "python pythonProject1/full_ssod_ffb_pipeline.py "
                f"--output-root {root.as_posix()} "
                f"--original-dataset \"{cfg['datasets']['in_domain_sciencedb']}\" "
                f"--external-test-dataset \"{cfg['datasets']['external_preprocessed_ffb']}\" "
                f"--new-images \"{cfg['datasets']['unlabeled_new_images']}\" "
                f"--skip-baseline-training --baseline-model-path {cfg['model']['baseline_model_path']} "
                f"--teacher-model {cfg['model']['baseline_model_path']} "
                "--max-epochs 10 --self-training-iterations 1"
            ),
        },
        {
            "experiment": "model2_external_domain_adaptation_3e",
            "purpose": "Cross-domain adaptation with labeled preprocessed-ffb train/valid.",
            "command": f"python pythonProject1/train_model2_adaptation.py --epochs {cfg['model'].get('model2_adaptation_epochs', 3)} --output-root {model2_root.as_posix()}",
        },
        {
            "experiment": "ablation_no_consistency",
            "purpose": "Measure multi-view consistency contribution.",
            "command": "append --disable-consistency to the Model 1 command",
        },
        {
            "experiment": "ablation_no_quality_score",
            "purpose": "Measure composite pseudo-label quality score contribution.",
            "command": "append --disable-quality-scoring to the Model 1 command",
        },
        {
            "experiment": "ablation_fixed_threshold",
            "purpose": "Measure dynamic class-wise threshold contribution.",
            "command": "append --disable-dynamic-thresholds to the Model 1 command",
        },
        {
            "experiment": "ablation_unbalanced_selection",
            "purpose": "Measure balanced per-class pseudo-label selection contribution.",
            "command": "append --max-pseudo-images-per-class 0 to the Model 1 command",
        },
        {
            "experiment": "model2_plus_ssod",
            "purpose": "Test whether SSOD adds value after external-domain adaptation.",
            "command": f"python pythonProject1/train_model2_adaptation.py --run-ssod-after-adaptation --epochs {cfg['model'].get('model2_adaptation_epochs', 3)}",
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["experiment", "purpose", "command"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote ablation command table: {args.output}")


if __name__ == "__main__":
    main()

