"""Mine all-class false positives and build a precision-focused YOLO subset.

This utility supports the high-precision operating-point target used in the
paper pipeline. It mines train/valid splits only, never the external test split,
and writes a small YOLO dataset that can be merged as an extra supervised source.

The generated labels are the original ground-truth labels, not pseudo-labels.
For images with no usable target-class boxes, the label is intentionally empty.
That gives YOLO hard background/negative examples and directly suppresses high
confidence false positives during the next Model2+SSOD+FPHard training run.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    BoxPrediction,
    box_iou,
    box_to_pixels,
    convert_to_yolo_format,
    ensure_dir,
    link_or_copy_file,
    list_images,
    load_label_boxes,
    predict_one,
    set_safe_delete_root,
    write_data_yaml,
    write_rows_csv,
)


@dataclass
class FalsePositiveCase:
    image_path: Path
    label_path: Path
    split: str
    predicted_class_id: int
    confidence: float
    best_same_class_iou: float
    best_any_class_iou: float
    matched_gt_class_id: int
    category: str


def best_iou_stats(pred_box: BoxPrediction, gt_boxes: Sequence[BoxPrediction]) -> Tuple[float, float, int]:
    same_class = 0.0
    any_class = 0.0
    any_class_id = -1
    for gt_box in gt_boxes:
        iou = box_iou(pred_box, gt_box)
        if gt_box.class_id == pred_box.class_id:
            same_class = max(same_class, iou)
        if iou > any_class:
            any_class = iou
            any_class_id = gt_box.class_id
    return same_class, any_class, any_class_id


def classify_false_positive(pred_box: BoxPrediction, gt_boxes: Sequence[BoxPrediction], any_iou: float, any_class_id: int) -> str:
    if not gt_boxes:
        return "background_no_target_gt"
    if any_iou >= 0.50 and any_class_id != pred_box.class_id:
        return f"class_confusion_{CANONICAL_CLASSES[any_class_id]}_as_{CANONICAL_CLASSES[pred_box.class_id]}"
    if any_iou >= 0.10:
        return "localization_or_duplicate_fp"
    return "background_fp_near_scene"


def mine_split(
    model: YOLO,
    dataset_root: Path,
    split: str,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    allowed_class_ids: Sequence[int],
) -> List[FalsePositiveCase]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    cases: List[FalsePositiveCase] = []
    for image_path in list_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        gt_boxes = load_label_boxes(label_path, allowed_class_ids=set(allowed_class_ids))
        predictions = [
            box
            for box in predict_one(model, image_path, imgsz=imgsz, device=device, conf=conf_threshold)
            if box.class_id in allowed_class_ids
        ]
        for pred_box in predictions:
            same_iou, any_iou, any_class_id = best_iou_stats(pred_box, gt_boxes)
            if same_iou >= iou_threshold:
                continue
            cases.append(
                FalsePositiveCase(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    predicted_class_id=pred_box.class_id,
                    confidence=pred_box.confidence,
                    best_same_class_iou=same_iou,
                    best_any_class_iou=any_iou,
                    matched_gt_class_id=any_class_id,
                    category=classify_false_positive(pred_box, gt_boxes, any_iou, any_class_id),
                )
            )
    return cases


def select_cases(
    cases: Sequence[FalsePositiveCase],
    max_cases_per_class: int,
    max_cases_per_category: int,
    seed: int,
) -> List[FalsePositiveCase]:
    rng = random.Random(seed)
    by_class: Dict[int, List[FalsePositiveCase]] = {}
    for case in cases:
        by_class.setdefault(case.predicted_class_id, []).append(case)

    selected: List[FalsePositiveCase] = []
    for class_id, class_cases in sorted(by_class.items()):
        by_category: Dict[str, List[FalsePositiveCase]] = {}
        for case in class_cases:
            by_category.setdefault(case.category, []).append(case)
        class_selected: List[FalsePositiveCase] = []
        for category_cases in by_category.values():
            ordered = sorted(category_cases, key=lambda item: item.confidence, reverse=True)
            cap = max_cases_per_category if max_cases_per_category > 0 else len(ordered)
            class_selected.extend(ordered[:cap])
        class_selected = sorted(class_selected, key=lambda item: item.confidence, reverse=True)
        if max_cases_per_class > 0 and len(class_selected) > max_cases_per_class:
            high_conf = class_selected[: max_cases_per_class // 2]
            remainder = class_selected[max_cases_per_class // 2 :]
            rng.shuffle(remainder)
            class_selected = high_conf + remainder[: max_cases_per_class - len(high_conf)]
        selected.extend(class_selected)
    return selected


def write_filtered_label(src_label: Path, dst_label: Path, allowed_class_ids: Sequence[int]) -> None:
    ensure_dir(dst_label.parent)
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return
    allowed = {int(class_id) for class_id in allowed_class_ids}
    lines: List[str] = []
    for line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id in allowed:
            lines.append(line)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_dataset(
    selected_cases: Sequence[FalsePositiveCase],
    output_root: Path,
    allowed_class_ids: Sequence[int],
    repeat: int,
) -> Path:
    dataset_root = output_root / "false_positive_hard_dataset"
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"
    ensure_dir(image_dir)
    ensure_dir(label_dir)
    ensure_dir(dataset_root / "valid" / "images")
    ensure_dir(dataset_root / "valid" / "labels")
    ensure_dir(dataset_root / "test" / "images")
    ensure_dir(dataset_root / "test" / "labels")

    for case_idx, case in enumerate(selected_cases):
        for repeat_idx in range(max(repeat, 1)):
            stem = f"fp_{case_idx:05d}_r{repeat_idx}"
            dst_image = image_dir / f"{stem}{case.image_path.suffix.lower()}"
            dst_label = label_dir / f"{stem}.txt"
            link_or_copy_file(case.image_path, dst_image)
            write_filtered_label(case.label_path, dst_label, allowed_class_ids)

    write_data_yaml(dataset_root)
    return dataset_root


def save_visuals(
    cases: Sequence[FalsePositiveCase],
    output_dir: Path,
    max_visuals: int,
    model: YOLO,
    imgsz: int,
    device: str,
) -> None:
    if max_visuals <= 0:
        return
    ensure_dir(output_dir)
    for idx, case in enumerate(sorted(cases, key=lambda item: item.confidence, reverse=True)[:max_visuals], start=1):
        with Image.open(case.image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for gt_box in load_label_boxes(case.label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS)):
                draw.rectangle(box_to_pixels(gt_box, image.width, image.height), outline="lime", width=2)
            for pred_box in predict_one(model, case.image_path, imgsz=imgsz, device=device, conf=case.confidence):
                if pred_box.class_id == case.predicted_class_id and abs(pred_box.confidence - case.confidence) < 1e-6:
                    draw.rectangle(box_to_pixels(pred_box, image.width, image.height), outline="red", width=3)
                    break
            out_name = f"fp_{idx:03d}_{CANONICAL_CLASSES[case.predicted_class_id]}_{case.image_path.name}"
            image.save(output_dir / out_name)


def write_case_tables(all_cases: Sequence[FalsePositiveCase], selected_cases: Sequence[FalsePositiveCase], output_root: Path) -> None:
    def to_row(case: FalsePositiveCase) -> dict:
        matched = CANONICAL_CLASSES.get(case.matched_gt_class_id, "") if case.matched_gt_class_id >= 0 else ""
        return {
            "image": str(case.image_path),
            "label": str(case.label_path),
            "split": case.split,
            "predicted_class": CANONICAL_CLASSES[case.predicted_class_id],
            "confidence": f"{case.confidence:.6f}",
            "best_same_class_iou": f"{case.best_same_class_iou:.6f}",
            "best_any_class_iou": f"{case.best_any_class_iou:.6f}",
            "matched_gt_class": matched,
            "category": case.category,
        }

    write_rows_csv(output_root / "all_false_positive_cases.csv", [to_row(case) for case in all_cases])
    write_rows_csv(output_root / "selected_false_positive_cases.csv", [to_row(case) for case in selected_cases])


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine all-class false positives for precision-focused training.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-threshold", type=float, default=0.40)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-cases-per-class", type=int, default=450)
    parser.add_argument("--max-cases-per-category", type=int, default=250)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-visuals", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args()

    set_safe_delete_root(parsed.output_root)
    ensure_dir(parsed.output_root)
    canonical_yaml = convert_to_yolo_format(parsed.external_dataset, parsed.output_root / "canonical_external")
    dataset_root = canonical_yaml.parent
    model = YOLO(str(parsed.model))

    all_cases: List[FalsePositiveCase] = []
    for split in ["train", "valid"]:
        split_dir = dataset_root / split / "images"
        if split_dir.exists():
            all_cases.extend(
                mine_split(
                    model=model,
                    dataset_root=dataset_root,
                    split=split,
                    imgsz=parsed.imgsz,
                    device=parsed.device,
                    conf_threshold=parsed.conf_threshold,
                    iou_threshold=parsed.iou_threshold,
                    allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
                )
            )

    selected_cases = select_cases(
        all_cases,
        max_cases_per_class=parsed.max_cases_per_class,
        max_cases_per_category=parsed.max_cases_per_category,
        seed=parsed.seed,
    )
    hard_dataset = build_dataset(
        selected_cases,
        parsed.output_root,
        allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
        repeat=parsed.repeat,
    )
    write_case_tables(all_cases, selected_cases, parsed.output_root)
    save_visuals(
        selected_cases,
        parsed.output_root / "false_positive_visuals",
        parsed.max_visuals,
        model=model,
        imgsz=parsed.imgsz,
        device=parsed.device,
    )

    summary = {
        "model": str(parsed.model),
        "external_dataset": str(parsed.external_dataset),
        "hard_dataset": str(hard_dataset),
        "data_yaml": str(hard_dataset / "data.yaml"),
        "test_split_used_for_training": False,
        "conf_threshold": parsed.conf_threshold,
        "iou_threshold": parsed.iou_threshold,
        "all_cases": len(all_cases),
        "selected_cases": len(selected_cases),
        "repeat": parsed.repeat,
        "selected_by_predicted_class": {},
        "selected_by_category": {},
    }
    for case in selected_cases:
        class_name = CANONICAL_CLASSES[case.predicted_class_id]
        summary["selected_by_predicted_class"][class_name] = summary["selected_by_predicted_class"].get(class_name, 0) + 1
        summary["selected_by_category"][case.category] = summary["selected_by_category"].get(case.category, 0) + 1
    (parsed.output_root / "false_positive_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

