"""Generate paper-ready comparison tables and plots from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = {
    "Model 0": Path("LOCAL_PROJECT_ROOT/paper_framework"),
    "Model 1": Path("LOCAL_PROJECT_ROOT/paper_framework"),
    "Model 2": Path("LOCAL_PROJECT_ROOT/paper_framework_model2_adapt_3e_clean"),
    "Model 2 Balanced": Path("LOCAL_DATA_ROOT/paper_framework_model2_balanced_3e"),
    "Model 2+SSOD": Path("LOCAL_DATA_ROOT/paper_framework_model2_balanced_ssod_3e"),
}


def runs_from_manifest(manifest_path: Path) -> Dict[str, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".json":
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig", newline="")))
    runs: Dict[str, Path] = {}
    for row in rows:
        summary = Path(row["expected_summary"])
        runs[row["name"]] = summary.parent.parent
    return runs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def find_stage(df: pd.DataFrame, contains: str) -> Optional[pd.Series]:
    if df is None or "stage" not in df:
        return None
    rows = df[df["stage"].astype(str).str.contains(contains, regex=False, na=False)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def find_stage_exact(df: pd.DataFrame, stage: str) -> Optional[pd.Series]:
    if df is None or "stage" not in df:
        return None
    rows = df[df["stage"].astype(str).eq(stage)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def external_metrics_path(run_root: Path, iteration: Optional[int] = None) -> Path:
    if iteration is not None:
        candidate = run_root / "reports" / f"iter_{iteration}_external_preprocessed_ffb_metrics.csv"
        if candidate.exists():
            return candidate
    return run_root / "reports" / "external_preprocessed_ffb_metrics.csv"


def collect_model_rows(runs: Dict[str, Path]) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict]]:
    model_rows: List[dict] = []
    class_rows: List[dict] = []
    calibration_rows: List[dict] = []
    fixed_threshold_rows: List[dict] = []
    pr_curve_rows: List[dict] = []
    classwise_threshold_rows: List[dict] = []

    for model_name, root in runs.items():
        summary = read_csv_if_exists(root / "reports" / "summary_metrics.csv")
        if summary is None:
            model_rows.append({"model": model_name, "status": "missing", "run_root": str(root)})
            continue

        model_name_lower = model_name.lower()
        is_ssod_run = "ssod" in model_name_lower or model_name in {"Model 1", "Model 2+SSOD"}

        if model_name == "Model 0":
            ext_stage = find_stage_exact(summary, "baseline_external_preprocessed_ffb_masked_4class")
            ext_path = None
        elif is_ssod_run:
            ext_stage = find_stage_exact(summary, "iter_1_heldout_external_preprocessed_ffb_masked_4class")
            ext_path = external_metrics_path(root, iteration=1)
        else:
            ext_stage = find_stage_exact(summary, "baseline_external_preprocessed_ffb_masked_4class")
            ext_path = external_metrics_path(root)

        in_domain_stage = find_stage(summary, "baseline_test")
        if is_ssod_run:
            iter_stage = find_stage(summary, "iter_1_val")
            in_domain_stage = iter_stage if iter_stage is not None else in_domain_stage

        row = {
            "model": model_name,
            "status": "complete" if ext_stage is not None else "missing_external",
            "run_root": str(root),
            "in_domain_map50": float(in_domain_stage["map50"]) if in_domain_stage is not None else "",
            "in_domain_map5095": float(in_domain_stage["map5095"]) if in_domain_stage is not None else "",
            "external_map50": float(ext_stage["map50"]) if ext_stage is not None else "",
            "external_map5095": float(ext_stage["map5095"]) if ext_stage is not None else "",
        }

        ext_df = read_csv_if_exists(ext_path) if ext_path is not None else None
        if ext_df is not None and not ext_df.empty:
            macro = ext_df[ext_df["class_name"].astype(str).eq("macro")]
            if not macro.empty:
                macro_row = macro.iloc[0]
                for key in ("best_calibrated_threshold", "best_calibrated_precision", "best_calibrated_recall", "best_calibrated_f1"):
                    if key in macro_row and pd.notna(macro_row[key]):
                        row[key] = float(macro_row[key])
            for _, class_row in ext_df[~ext_df["class_name"].astype(str).eq("macro")].iterrows():
                class_rows.append(
                    {
                        "model": model_name,
                        "class_name": class_row.get("class_name", ""),
                        "gt_boxes": class_row.get("gt_boxes", ""),
                        "ap50": class_row.get("ap50", ""),
                        "ap50_95": class_row.get("ap50_95", ""),
                    }
                )

        cal_path = ext_path.with_name(f"{ext_path.stem}_calibration.csv") if ext_path is not None else None
        cal_df = read_csv_if_exists(cal_path)
        if cal_df is not None:
            for _, cal_row in cal_df.iterrows():
                calibration_rows.append(
                    {
                        "model": model_name,
                        "threshold": cal_row.get("threshold", ""),
                        "class_name": cal_row.get("class_name", ""),
                        "precision": cal_row.get("precision", ""),
                        "recall": cal_row.get("recall", ""),
                        "f1": cal_row.get("f1", ""),
                    }
                )
        fixed_path = ext_path.with_name(f"{ext_path.stem}_fixed_thresholds.csv") if ext_path is not None else None
        fixed_df = read_csv_if_exists(fixed_path)
        if fixed_df is not None:
            for _, fixed_row in fixed_df.iterrows():
                fixed_threshold_rows.append(
                    {
                        "model": model_name,
                        "threshold": fixed_row.get("threshold", ""),
                        "class_name": fixed_row.get("class_name", ""),
                        "precision": fixed_row.get("precision", ""),
                        "recall": fixed_row.get("recall", ""),
                        "f1": fixed_row.get("f1", ""),
                        "tp": fixed_row.get("tp", ""),
                        "fp": fixed_row.get("fp", ""),
                        "fn": fixed_row.get("fn", ""),
                    }
                )
        pr_path = ext_path.with_name(f"{ext_path.stem}_pr_curve.csv") if ext_path is not None else None
        pr_df = read_csv_if_exists(pr_path)
        if pr_df is not None:
            for _, pr_row in pr_df.iterrows():
                pr_curve_rows.append(
                    {
                        "model": model_name,
                        "class_name": pr_row.get("class_name", ""),
                        "rank": pr_row.get("rank", ""),
                        "confidence": pr_row.get("confidence", ""),
                        "precision": pr_row.get("precision", ""),
                        "recall": pr_row.get("recall", ""),
                        "tp": pr_row.get("tp", ""),
                        "fp": pr_row.get("fp", ""),
                        "is_tp": pr_row.get("is_tp", ""),
                    }
                )
        classwise_path = ext_path.with_name(f"{ext_path.stem}_classwise_threshold_search.csv") if ext_path is not None else None
        classwise_df = read_csv_if_exists(classwise_path)
        if classwise_df is not None:
            for _, threshold_row in classwise_df.iterrows():
                classwise_threshold_rows.append(
                    {
                        "model": model_name,
                        "threshold": threshold_row.get("threshold", ""),
                        "class_name": threshold_row.get("class_name", ""),
                        "precision": threshold_row.get("precision", ""),
                        "recall": threshold_row.get("recall", ""),
                        "f1": threshold_row.get("f1", ""),
                        "tp": threshold_row.get("tp", ""),
                        "fp": threshold_row.get("fp", ""),
                        "fn": threshold_row.get("fn", ""),
                        "target_precision": threshold_row.get("target_precision", ""),
                        "meets_target_precision": threshold_row.get("meets_target_precision", ""),
                        "selected": threshold_row.get("selected", ""),
                        "selection_rule": threshold_row.get("selection_rule", ""),
                    }
                )
        model_rows.append(row)

    return model_rows, class_rows, calibration_rows, fixed_threshold_rows, pr_curve_rows, classwise_threshold_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_high_precision_table(classwise_threshold_rows: List[dict], output_path: Path) -> None:
    rows = [
        row
        for row in classwise_threshold_rows
        if str(row.get("selected", "")) == "1" and row.get("class_name", "") in {"macro_selected", "abnormal", "ripe", "under_ripe", "unripe"}
    ]
    write_csv(output_path, rows)


def plot_model_comparison(rows: List[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df["status"].eq("complete")].copy()
    if df.empty:
        return
    df["external_map50"] = pd.to_numeric(df["external_map50"], errors="coerce")
    df["external_map5095"] = pd.to_numeric(df["external_map5095"], errors="coerce")
    x = range(len(df))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], df["external_map50"], width=width, label="mAP@0.5")
    plt.bar([i + width / 2 for i in x], df["external_map5095"], width=width, label="mAP@0.5:0.95")
    plt.xticks(list(x), df["model"], rotation=15, ha="right")
    plt.ylabel("External preprocessed-ffb score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_classwise_ap(class_rows: List[dict], output_path: Path, model_name: str = "Model 2") -> None:
    df = pd.DataFrame(class_rows)
    if df.empty or "model" not in df:
        return
    df = df[df["model"].eq(model_name)].copy()
    if df.empty:
        df = pd.DataFrame(class_rows)
        first_model = str(df["model"].dropna().iloc[0]) if not df["model"].dropna().empty else ""
        df = df[df["model"].eq(first_model)].copy()
    if df.empty:
        return
    df["ap50"] = pd.to_numeric(df["ap50"], errors="coerce")
    df["ap50_95"] = pd.to_numeric(df["ap50_95"], errors="coerce")
    x = range(len(df))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], df["ap50"], width=width, label="AP@0.5")
    plt.bar([i + width / 2 for i in x], df["ap50_95"], width=width, label="AP@0.5:0.95")
    plt.xticks(list(x), df["class_name"], rotation=15, ha="right")
    plt.ylabel("AP")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_calibration(calibration_rows: List[dict], output_path: Path, model_name: str = "Model 2") -> None:
    df = pd.DataFrame(calibration_rows)
    if df.empty or "model" not in df or "class_name" not in df:
        return
    df = df[(df["model"].eq(model_name)) & (df["class_name"].eq("macro"))].copy()
    if df.empty:
        df = pd.DataFrame(calibration_rows)
        first_model = str(df["model"].dropna().iloc[0]) if not df["model"].dropna().empty else ""
        df = df[(df["model"].eq(first_model)) & (df["class_name"].eq("macro"))].copy()
    if df.empty:
        return
    for col in ("threshold", "precision", "recall", "f1"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    plt.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    plt.plot(df["threshold"], df["f1"], marker="o", label="F1")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_pr_curve(pr_rows: List[dict], output_path: Path, model_name: str = "Model 2") -> None:
    df = pd.DataFrame(pr_rows)
    if df.empty or "model" not in df or "class_name" not in df:
        return
    df = df[df["model"].eq(model_name)].copy()
    if df.empty:
        df = pd.DataFrame(pr_rows)
        first_model = str(df["model"].dropna().iloc[0]) if not df["model"].dropna().empty else ""
        df = df[df["model"].eq(first_model)].copy()
    if df.empty:
        return
    for col in ("precision", "recall"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    plt.figure(figsize=(7, 5))
    for class_name, group in df.groupby("class_name", sort=False):
        group = group.sort_values("recall")
        plt.plot(group["recall"], group["precision"], label=class_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_aggregate_table(rows: List[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty or "model" not in df:
        write_csv(output_path, [])
        return
    df = df[df["status"].eq("complete")].copy()
    if df.empty:
        write_csv(output_path, [])
        return
    # Strip seed suffix so formal repeats are summarized as mean +/- std.
    df["experiment_family"] = df["model"].astype(str).str.replace(r"_seed\d+$", "", regex=True)
    metric_cols = ["in_domain_map50", "in_domain_map5095", "external_map50", "external_map5095", "best_calibrated_f1"]
    for col in metric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out_rows: List[dict] = []
    for family, group in df.groupby("experiment_family", sort=False):
        row = {"experiment_family": family, "n_runs": int(len(group))}
        for col in metric_cols:
            if col in group:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"] = group[col].std(ddof=0)
        out_rows.append(row)
    write_csv(output_path, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper summary tables and plots.")
    parser.add_argument("--output-dir", type=Path, default=Path("LOCAL_PROJECT_ROOT/paper_framework/reports/paper_summary"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional paper_experiment_manifest.csv/json generated by run_paper_experiments.py.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    runs = runs_from_manifest(args.manifest) if args.manifest else DEFAULT_RUNS
    rows, class_rows, calibration_rows, fixed_threshold_rows, pr_curve_rows, classwise_threshold_rows = collect_model_rows(runs)
    write_csv(args.output_dir / "model_comparison.csv", rows)
    write_aggregate_table(rows, args.output_dir / "model_comparison_mean_std.csv")
    write_csv(args.output_dir / "classwise_external_ap.csv", class_rows)
    write_csv(args.output_dir / "external_calibration_summary.csv", calibration_rows)
    write_csv(args.output_dir / "external_fixed_threshold_summary.csv", fixed_threshold_rows)
    write_csv(args.output_dir / "external_pr_curve_points.csv", pr_curve_rows)
    write_csv(args.output_dir / "external_classwise_threshold_search.csv", classwise_threshold_rows)
    write_high_precision_table(classwise_threshold_rows, args.output_dir / "external_high_precision_operating_points.csv")
    plot_model_comparison(rows, args.output_dir / "model_comparison_external_map.png")
    plot_classwise_ap(class_rows, args.output_dir / "model2_classwise_external_ap.png")
    plot_calibration(calibration_rows, args.output_dir / "model2_calibration_curve.png")
    plot_pr_curve(pr_curve_rows, args.output_dir / "model2_pr_curve.png")
    print(f"Wrote paper summary artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()

