"""Build the paper-facing retain/reject table for the priority experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional


REPORT_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports")
RETAINED_ROOT = Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_ssod_fp250_urpos640_yolov8n")
PRIORITY_A_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_a_boundary250_seed42_run2")
PRIORITY_B_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_b_ordinal015_seed42")
PRIORITY_C_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_c_offline_ema_teacher_seed42")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def external_metrics(path: Path) -> dict:
    macro = next(row for row in read_csv(path) if row["class_name"] == "macro")
    under_ripe = next(row for row in read_csv(path) if row["class_name"] == "under_ripe")
    return {
        "external_map50": macro["map50"],
        "external_map5095": macro["map5095"],
        "calibrated_threshold": macro["best_calibrated_threshold"],
        "calibrated_precision": macro["best_calibrated_precision"],
        "calibrated_recall": macro["best_calibrated_recall"],
        "calibrated_f1": macro["best_calibrated_f1"],
        "classwise_threshold_precision": macro["classwise_threshold_precision"],
        "classwise_threshold_recall": macro["classwise_threshold_recall"],
        "classwise_threshold_f1": macro["classwise_threshold_f1"],
        "under_ripe_ap50": under_ripe["ap50"],
        "under_ripe_ap5095": under_ripe["ap50_95"],
        "checkpoint": macro["model_path"],
    }


def row(
    model: str,
    role: str,
    external_csv: Path,
    outdoor_json: Optional[Path],
    decision: str,
    reason: str,
) -> dict:
    metrics = external_metrics(external_csv)
    outdoor = read_json(outdoor_json)
    return {
        "model": model,
        "role": role,
        **metrics,
        "outdoor_zero_shot_top1": outdoor.get("top1_accuracy", ""),
        "outdoor_zero_shot_macro_f1": outdoor.get("macro_f1", ""),
        "decision": decision,
        "reason": reason,
    }


def main() -> None:
    rows = [
        row(
            "Retained Stage2 refinement from Stage1 SSOD teacher + FPHard + URPos",
            "development-best refined model",
            RETAINED_ROOT / "reports" / "external_preprocessed_ffb_metrics.csv",
            REPORT_ROOT / "third_domain_outdoor_zenodo" / "outdoor_zenodo_zero_shot_summary.json",
            "retain",
            "Best decisive external masked mAP and calibrated F1; Stage2 uses supervised refinement initialized from the Stage1 SSOD teacher.",
        ),
        row(
            "Priority A compact boundary sampling",
            "under_ripe boundary sampling ablation",
            PRIORITY_A_ROOT / "reports" / "external_preprocessed_ffb_metrics.csv",
            REPORT_ROOT / "priority_a_boundary250_seed42_third_domain" / "outdoor_zenodo_zero_shot_summary.json",
            "reject",
            "Reduced external masked mAP, under_ripe AP, and Outdoor zero-shot accuracy.",
        ),
        row(
            "Priority B ordinal auxiliary loss gain=0.15",
            "true ordinal-loss ablation",
            PRIORITY_B_ROOT / "reports" / "external_preprocessed_ffb_metrics.csv",
            REPORT_ROOT / "priority_b_ordinal015_seed42_third_domain" / "outdoor_zenodo_zero_shot_summary.json",
            "reject",
            "Did not improve decisive external masked metrics; under_ripe AP decreased.",
        ),
        row(
            "Priority C offline EMA teacher iteration 1",
            "iterative teacher refresh ablation",
            PRIORITY_C_ROOT / "reports" / "iter_1_external_preprocessed_ffb_metrics.csv",
            None,
            "reject",
            "Teacher refresh reduced external masked mAP relative to retained final.",
        ),
        row(
            "Priority C offline EMA teacher iteration 2",
            "iterative teacher refresh ablation",
            PRIORITY_C_ROOT / "reports" / "iter_2_external_preprocessed_ffb_metrics.csv",
            REPORT_ROOT / "priority_c_offline_ema_teacher_seed42_third_domain" / "outdoor_zenodo_zero_shot_summary.json",
            "reject",
            "Second refresh worsened iteration 1, consistent with pseudo-label drift.",
        ),
    ]
    mendeley = read_json(REPORT_ROOT / "third_domain_mendeley_ordinal" / "mendeley_ordinal_zero_shot_summary.json")
    for result in rows:
        result["mendeley_zero_shot_top1"] = ""
        result["mendeley_zero_shot_macro_f1"] = ""
        result["mendeley_verified_duplicate_matches"] = ""
    rows[0]["mendeley_zero_shot_top1"] = mendeley["top1_accuracy"]
    rows[0]["mendeley_zero_shot_macro_f1"] = mendeley["macro_f1"]
    rows[0]["mendeley_verified_duplicate_matches"] = mendeley["duplicate_check"]["verified_duplicate_matches"]

    output_csv = REPORT_ROOT / "priority_optimization_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_ROOT / "priority_optimization_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_csv}")
    for result in rows:
        print(
            f"{result['decision']:>6} | {result['model']:<48} | "
            f"external mAP50={float(result['external_map50']):.6f} | "
            f"calibrated F1={float(result['calibrated_f1']):.6f}"
        )


if __name__ == "__main__":
    main()

