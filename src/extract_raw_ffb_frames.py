"""Extract weak-label frames from raw oil-palm FFB videos for SSOD.

The Scientific Data raw-video archive stores class hints in filenames, e.g.
"kurang masak" and "abnormal". These are image-level weak labels, not bounding
box annotations. The extracted frames are intended for pseudo-label generation
with folder-class consistency, not direct supervised YOLO training.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ")).strip()


def classes_from_video_name(path: Path) -> set[str]:
    name = normalize_text(path.stem)
    classes: set[str] = set()
    if "abnormal" in name:
        classes.add("abnormal")
    if "kurang masak" in name:
        classes.add("under_ripe")
    if "mentah" in name:
        classes.add("unripe")
    # Match masak only after removing "kurang masak".
    without_under = name.replace("kurang masak", "")
    if re.search(r"\bmasak\b", without_under):
        classes.add("ripe")
    if "terlalu masak" in name or "overripe" in name:
        classes.add("overripe")
    return classes


def copy_existing_new_images(source: Path, output: Path) -> None:
    if not source.exists():
        return
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS):
            dst = output / class_dir.name / image_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            try:
                dst.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, dst)


def extract_video_frames(video_path: Path, classes: set[str], output: Path, frames_per_video: int, max_width: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [{"video": str(video_path), "class": "", "frame_index": "", "output": "", "status": "open_failed"}]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return [{"video": str(video_path), "class": "", "frame_index": "", "output": "", "status": "no_frames"}]

    if frames_per_video <= 1:
        indices = [frame_count // 2]
    else:
        start = max(frame_count // 10, 0)
        end = max(frame_count - frame_count // 10, start + 1)
        indices = [round(start + (end - start) * i / max(frames_per_video - 1, 1)) for i in range(frames_per_video)]

    rows = []
    safe_stem = re.sub(r"[^a-zA-Z0-9]+", "_", video_path.stem).strip("_")
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            rows.append({"video": str(video_path), "class": "", "frame_index": frame_index, "output": "", "status": "read_failed"})
            continue
        height, width = frame.shape[:2]
        if max_width > 0 and width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        for class_name in sorted(classes):
            dst = output / class_name / f"rawffb_{safe_stem}_f{frame_index:06d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rows.append({"video": str(video_path), "class": class_name, "frame_index": frame_index, "output": str(dst), "status": "ok"})
    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract weak-label frames from raw FFB videos.")
    parser.add_argument("--raw-video-root", type=Path, default=Path("LOCAL_DATA_ROOT/raw-ffb-plantation"))
    parser.add_argument("--base-new-images", type=Path, default=Path("LOCAL_DATA_ROOT/new_images_augmented_for_ssod"))
    parser.add_argument("--output", type=Path, default=Path("LOCAL_DATA_ROOT/new_images_raw_ffb_augmented_for_ssod"))
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--target-classes", default="abnormal,under_ripe", help="Comma-separated classes to extract from weak video names.")
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    copy_existing_new_images(args.base_new_images, args.output)

    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}
    rows = []
    videos = sorted(args.raw_video_root.rglob("*.mp4"))
    for idx, video_path in enumerate(videos, start=1):
        classes = classes_from_video_name(video_path) & target_classes
        if not classes:
            continue
        rows.extend(extract_video_frames(video_path, classes, args.output, args.frames_per_video, args.max_width))
        if idx % 50 == 0:
            print(f"processed {idx}/{len(videos)} videos")

    report = args.output / "raw_video_frame_report.csv"
    with report.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video", "class", "frame_index", "output", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote extracted weak-label frames to: {args.output}")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()

