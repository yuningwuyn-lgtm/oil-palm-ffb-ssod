"""Run the evaluation-only Outdoor Zenodo third-domain comparison."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 2026, 3407)
DATASET_ROOT = Path(
    r"LOCAL_DATA_ROOT\public_datasets\zenodo_outdoor_tenera_ffb"
    r"\MunirahRosbi_Outdoor-Tenera-Oil-Palm-Fruit-Image-v1"
)
OUTPUT_ROOT = Path(
    r"LOCAL_PROJECT_ROOT\paper_framework\reports"
    r"\third_domain_outdoor_zenodo_formal"
)
PROTOCOL_ROOT = Path(r"LOCAL_DATA_ROOT\paper_protocol_v2_formal")
STRICT_ROOT = Path(
    r"LOCAL_DATA_ROOT\paper_precision_first_ssod_formal"
)


def checkpoint(stage: str, seed: int) -> Path:
    if stage == "source_only":
        return (
            PROTOCOL_ROOT
            / f"seed_{seed}"
            / stage
            / "runs"
            / "formal_train"
            / "weights"
            / "best.pt"
        )
    if stage == "model2_balanced":
        return (
            PROTOCOL_ROOT
            / f"seed_{seed}"
            / stage
            / "runs"
            / "formal_train"
            / "weights"
            / "best.pt"
        )
    if stage == "strict_ssod":
        return (
            STRICT_ROOT
            / f"seed_{seed}"
            / "strict_ssod_no_relabel"
            / "runs"
            / "self_train_iter_1"
            / "weights"
            / "best.pt"
        )
    raise ValueError(stage)


def evaluate(stage: str, seed: int) -> dict:
    output_root = OUTPUT_ROOT / stage / f"seed_{seed}"
    summary = output_root / "outdoor_zenodo_zero_shot_summary.json"
    model = checkpoint(stage, seed)
    if not model.exists():
        raise FileNotFoundError(model)
    if not summary.exists():
        command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_outdoor_zenodo_third_domain.py")),
            "--dataset-root",
            str(DATASET_ROOT),
            "--model",
            str(model),
            "--output-root",
            str(output_root),
            "--split",
            "test",
            "--imgsz",
            "640",
            "--device",
            "0",
        ]
        print("\n>>>", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "stage": stage,
        "seed": seed,
        "images": payload["images"],
        "coverage": payload["coverage"],
        "top1_accuracy": payload["top1_accuracy"],
        "macro_precision": payload["macro_precision"],
        "macro_recall": payload["macro_recall"],
        "macro_f1": payload["macro_f1"],
        "checkpoint": str(model),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> None:
    summary_rows = []
    metric_names = (
        "coverage",
        "top1_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    for stage in ("source_only", "model2_balanced", "strict_ssod"):
        subset = [row for row in rows if row["stage"] == stage]
        summary = {"stage": stage, "seeds": len(subset), "images_per_seed": subset[0]["images"]}
        for metric in metric_names:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summary_rows.append(summary)
    write_csv(OUTPUT_ROOT / "third_domain_zero_shot_seed_results.csv", rows)
    write_csv(OUTPUT_ROOT / "third_domain_zero_shot_summary.csv", summary_rows)


def main() -> None:
    rows = []
    for stage in ("source_only", "model2_balanced", "strict_ssod"):
        for seed in SEEDS:
            print(f"\n=== third-domain zero-shot: stage={stage}, seed={seed} ===", flush=True)
            rows.append(evaluate(stage, seed))
    aggregate(rows)
    print(f"Saved summary: {OUTPUT_ROOT / 'third_domain_zero_shot_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

