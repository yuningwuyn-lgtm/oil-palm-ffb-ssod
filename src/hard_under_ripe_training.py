"""Mine under_ripe hard cases and build an oversampled YOLO training subset.

This utility is intentionally separate from the main SSOD pipeline. It uses an
already trained detector to find difficult under_ripe cases from train/valid
splits only, then writes a small YOLO dataset that can be added as an
extra-supervised source. The external test split is never mined or copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    CLASS_NAME_TO_ID,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    box_iou,
    box_to_pixels,
    convert_to_yolo_format,
    copy_file,
    ensure_dir,
    link_or_copy_file,
    list_images,
    load_label_boxes,
    predict_one,
    set_safe_delete_root,
    write_data_yaml,
    write_rows_csv,
)


UNDER_RIPE_ID = CLASS_NAME_TO_ID["under_ripe"]
CONFUSING_CLASS_IDS = {CLASS_NAME_TO_ID["ripe"], CLASS_NAME_TO_ID["unripe"]}


@dataclass
class HardCase:
    image_path: Path
    label_path: Path
    split: str
    category: str
    confidence: float
    best_iou: float
    predicted_class: str
    note: str


def best_same_class_iou(pred_box, gt_boxes, class_id: int) -> float:
    best = 0.0
    for gt_box in gt_boxes:
        if gt_box.class_id == class_id:
            best = max(best, box_iou(pred_box, gt_box))
    return best


def best_overlap_with_gt(pred_box, gt_boxes) -> Tuple[float, int]:
    best = 0.0
    best_class = -1
    for gt_box in gt_boxes:
        iou = box_iou(pred_box, gt_box)
        if iou > best:
            best = iou
            best_class = gt_box.class_id
    return best, best_class


def has_under_ripe_gt(gt_boxes) -> bool:
    return any(box.class_id == UNDER_RIPE_ID for box in gt_boxes)


def mine_split(
    model: YOLO,
    dataset_root: Path,
    split: str,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    max_images: int,
) -> List[HardCase]:
    images = list_images(dataset_root / split / "images")
    if max_images > 0:
        images = images[:max_images]
    labels_dir = dataset_root / split / "labels"
    cases: List[HardCase] = []
    for index, image_path in enumerate(images, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = load_label_boxes(label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS))
        if not gt_boxes:
            continue
        predictions = [
            box for box in predict_one(model, image_path, imgsz=imgsz, device=device, conf=conf_threshold)
            if box.class_id in set(PREPROCESSED_FFB_EVAL_CLASS_IDS)
        ]

        under_gt = [box for box in gt_boxes if box.class_id == UNDER_RIPE_ID]
        for gt_box in under_gt:
            same_class_match = any(
                pred.class_id == UNDER_RIPE_ID and box_iou(pred, gt_box) >= iou_threshold
                for pred in predictions
            )
            if same_class_match:
                continue
            confusing_preds = [
                pred for pred in predictions
                if pred.class_id in CONFUSING_CLASS_IDS and box_iou(pred, gt_box) >= iou_threshold
            ]
            if confusing_preds:
                best_pred = max(confusing_preds, key=lambda pred: (box_iou(pred, gt_box), pred.confidence))
                cases.append(
                    HardCase(
                        image_path=image_path,
                        label_path=label_path,
                        split=split,
                        category=f"under_ripe_confused_as_{CANONICAL_CLASSES[best_pred.class_id]}",
                        confidence=best_pred.confidence,
                        best_iou=box_iou(best_pred, gt_box),
                        predicted_class=CANONICAL_CLASSES[best_pred.class_id],
                        note="under_ripe_gt_overlapped_by_adjacent_class_prediction",
                    )
                )
            else:
                cases.append(
                    HardCase(
                        image_path=image_path,
                        label_path=label_path,
                        split=split,
                        category="under_ripe_false_negative",
                        confidence=0.0,
                        best_iou=0.0,
                        predicted_class="none",
                        note="under_ripe_gt_without_matching_prediction",
                    )
                )

        for pred in predictions:
            if pred.class_id != UNDER_RIPE_ID:
                continue
            same_iou = best_same_class_iou(pred, gt_boxes, UNDER_RIPE_ID)
            if same_iou >= iou_threshold:
                continue
            overlap_iou, overlap_class = best_overlap_with_gt(pred, gt_boxes)
            if overlap_class == UNDER_RIPE_ID:
                continue
            category = "under_ripe_false_positive"
            if overlap_class in CONFUSING_CLASS_IDS:
                category = f"{CANONICAL_CLASSES[overlap_class]}_confused_as_under_ripe"
            elif not has_under_ripe_gt(gt_boxes):
                category = "under_ripe_hard_negative_no_under_ripe_gt"
            cases.append(
                HardCase(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    category=category,
                    confidence=pred.confidence,
                    best_iou=overlap_iou,
                    predicted_class="under_ripe",
                    note="high_conf_under_ripe_prediction_not_matching_under_ripe_gt",
                )
            )
        if index % 200 == 0:
            print(f"  mined {split}: {index}/{len(images)} images, cases={len(cases)}")
    return cases


def select_cases(cases: Sequence[HardCase], max_per_category: int, seed: int) -> List[HardCase]:
    grouped: Dict[str, List[HardCase]] = {}
    for case in cases:
        grouped.setdefault(case.category, []).append(case)
    rng = random.Random(seed)
    selected: List[HardCase] = []
    for category, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        items.sort(key=lambda item: (item.confidence, item.best_iou), reverse=True)
        selected.extend(items[:max_per_category] if max_per_category > 0 else items)
    return selected


def write_case_visuals(cases: Sequence[HardCase], output_dir: Path, max_visuals: int) -> None:
    ensure_dir(output_dir)
    for idx, case in enumerate(cases[:max_visuals], start=1):
        with Image.open(case.image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            gt_boxes = load_label_boxes(case.label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS))
            for gt_box in gt_boxes:
                color = (0, 180, 0) if gt_box.class_id == UNDER_RIPE_ID else (80, 120, 255)
                draw.rectangle(box_to_pixels(gt_box, width, height), outline=color, width=3)
            label = f"{case.category} conf={case.confidence:.2f} iou={case.best_iou:.2f}"
            draw.rectangle((0, 0, min(width, 520), 20), fill=(0, 0, 0))
            draw.text((4, 3), label, fill=(255, 255, 255))
            image.save(output_dir / f"hard_{idx:03d}_{case.category}_{case.image_path.name}", quality=92)


def build_hard_dataset(cases: Sequence[HardCase], output_root: Path, repeat: int) -> Path:
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")
    used_names = set()
    for case_idx, case in enumerate(cases):
        for repeat_idx in range(max(1, repeat)):
            # Keep names short. The combined training root is deeply nested on
            # Windows, and long source filenames can exceed MAX_PATH when copied.
            name = f"hu_{case_idx:05d}_r{repeat_idx}{case.image_path.suffix.lower()}"
            while name.lower() in used_names:
                name = f"hu_{case_idx:05d}_r{repeat_idx}_{random.randrange(10**6)}{case.image_path.suffix.lower()}"
            used_names.add(name.lower())
            link_or_copy_file(case.image_path, output_root / "train" / "images" / name)
            link_or_copy_file(case.label_path, output_root / "train" / "labels" / f"{Path(name).stem}.txt")
    return write_data_yaml(output_root)


def write_case_report(cases: Sequence[HardCase], output_csv: Path) -> None:
    rows = [
        {
            "image": str(case.image_path),
            "label": str(case.label_path),
            "split": case.split,
            "category": case.category,
            "confidence": f"{case.confidence:.6f}",
            "best_iou": f"{case.best_iou:.6f}",
            "predicted_class": case.predicted_class,
            "note": case.note,
        }
        for case in cases
    ]
    write_rows_csv(output_csv, rows)


def summarize_cases(cases: Sequence[HardCase]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for case in cases:
        summary[case.category] = summary.get(case.category, 0) + 1
    return dict(sorted(summary.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine under_ripe hard cases and build a hard-negative training subset.")
    parser.add_argument("--external-dataset", type=Path, default=Path("LOCAL_DATA_ROOT/preprocessed-ffb/yolov8"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("LOCAL_PROJECT_ROOT/paper_framework/datasets/under_ripe_hard_mining"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--max-cases-per-category", type=int, default=350)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-visuals", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_root)
    set_safe_delete_root(args.output_root)
    canonical_yaml = convert_to_yolo_format(args.external_dataset, args.output_root / "canonical_external")
    canonical_root = canonical_yaml.parent
    model = YOLO(str(args.model))

    all_cases: List[HardCase] = []
    for split in ("train", "valid"):
        all_cases.extend(
            mine_split(
                model=model,
                dataset_root=canonical_root,
                split=split,
                imgsz=args.imgsz,
                device=args.device,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                max_images=args.max_images_per_split,
            )
        )
    selected = select_cases(all_cases, max_per_category=args.max_cases_per_category, seed=args.seed)
    dataset_root = args.output_root / "hard_under_ripe_dataset"
    if dataset_root.exists():
        import shutil

        shutil.rmtree(dataset_root)
    data_yaml = build_hard_dataset(selected, dataset_root, repeat=args.repeat)
    write_case_report(all_cases, args.output_root / "all_hard_cases_under_ripe.csv")
    write_case_report(selected, args.output_root / "selected_hard_cases_under_ripe.csv")
    write_case_visuals(selected, args.output_root / "visuals", max_visuals=args.max_visuals)
    summary = {
        "model": str(args.model),
        "external_dataset": str(args.external_dataset),
        "canonical_root": str(canonical_root),
        "hard_dataset": str(dataset_root),
        "data_yaml": str(data_yaml),
        "splits_mined": ["train", "valid"],
        "test_split_used_for_training": False,
        "all_cases": summarize_cases(all_cases),
        "selected_cases": summarize_cases(selected),
        "repeat": args.repeat,
    }
    (args.output_root / "hard_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

