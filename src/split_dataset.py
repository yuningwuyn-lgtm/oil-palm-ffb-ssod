import os
import random
import shutil
from pathlib import Path

# ====== 配置 ======
BASE_DIR = Path(r"LOCAL_DATA_ROOT\OILPALM.yolov8")

# 原始数据（只读，不会被修改）
IMG_DIR = BASE_DIR / "train" / "images"
LABEL_DIR = BASE_DIR / "train" / "labels"

# ⭐ 输出到新目录（关键，避免 SameFileError）
OUTPUT_DIR = BASE_DIR / "split_dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42  # 固定随机种子，保证可复现

# ====== 检查 ======
assert IMG_DIR.exists(), f"images 不存在: {IMG_DIR}"
assert LABEL_DIR.exists(), f"labels 不存在: {LABEL_DIR}"

# 支持 jpg/png/jpeg
images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpeg"))
assert len(images) > 0, "没有找到图片文件"

# ====== 打乱（可复现） ======
random.seed(SEED)
random.shuffle(images)

# ====== 切分 ======
n = len(images)
train_cut = int(n * TRAIN_RATIO)
val_cut = int(n * (TRAIN_RATIO + VAL_RATIO))

splits = {
    "train": images[:train_cut],
    "valid": images[train_cut:val_cut],
    "test": images[val_cut:]
}

# ====== 创建目录 ======
for split in splits:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# ====== 复制文件 ======
copied = 0
missing_labels = 0

for split, files in splits.items():
    for img_path in files:
        dst_img = OUTPUT_DIR / split / "images" / img_path.name
        dst_label = OUTPUT_DIR / split / "labels" / (img_path.stem + ".txt")

        # 复制图片（如果已存在则跳过，避免重复）
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # 复制标签（可能有缺失）
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        if label_path.exists():
            if not dst_label.exists():
                shutil.copy2(label_path, dst_label)
        else:
            missing_labels += 1

        copied += 1

print("✅ 数据划分完成")
print(f"总图片数: {n}")
print(f"train: {len(splits['train'])}")
print(f"valid: {len(splits['valid'])}")
print(f"test : {len(splits['test'])}")
print(f"已处理文件: {copied}")
print(f"缺失标签数量: {missing_labels}")

# ====== 生成 data.yaml（自动写好，直接可训练） ======
yaml_path = OUTPUT_DIR / "data.yaml"

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: 6
names: ['abnormal', 'empty', 'overripe', 'ripe', 'under_ripe', 'unripe']
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✅ 已生成 data.yaml: {yaml_path}")
