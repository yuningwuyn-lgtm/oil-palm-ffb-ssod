from ultralytics import YOLO
from pathlib import Path
import time
import torch


def main():
    # ====== 路径配置 ======
    DATA_YAML = r"LOCAL_DATA_ROOT\OILPALM.yolov8\split_dataset\data.yaml"
    PROJECT_DIR = r"LOCAL_PROJECT_ROOT"
    TEST_DIR = r"LOCAL_DATA_ROOT\OILPALM.yolov8\split_dataset\test\images"

    # ====== 检查 ======
    assert Path(DATA_YAML).exists(), f"data.yaml 不存在: {DATA_YAML}"
    assert Path(TEST_DIR).exists(), f"测试路径不存在: {TEST_DIR}"

    # ====== 初始化模型 ======
    model = YOLO("yolov8n.pt")

    # =========================
    # 1️⃣ 训练
    # =========================
    print("🚀 开始训练...")
    train_results = model.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        project=PROJECT_DIR,
        name="exp",          # 不再写死 exp1
        exist_ok=False,      # 自动生成 exp, exp2, exp3...
        workers=0,
        cache=False
    )

    # 🔴 自动获取训练输出路径
    save_dir = Path(train_results.save_dir)
    print(f"训练保存路径: {save_dir}")

    # =========================
    # 2️⃣ 验证（test集）
    # =========================
    print("\n📊 开始验证...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        workers=0
    )

    print("==== 验证结果 ====")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # =========================
    # 3️⃣ 推理
    # =========================
    print("\n🧪 开始推理...")

    model.predict(
        source=TEST_DIR,
        save=True,
        imgsz=640,
        conf=0.25,
        project=PROJECT_DIR,
        name="predict",
        workers=0
    )

    print(f"推理结果已保存到: {Path(PROJECT_DIR) / 'predict'}")

    # =========================
    # 4️⃣ Latency 测试
    # =========================
    print("\n⚡ 开始测速度...")

    # 🔴 自动定位 best.pt
    best_model_path = save_dir / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt: {best_model_path}")

    print(f"使用模型: {best_model_path}")

    model = YOLO(str(best_model_path))

    # warm-up
    for _ in range(5):
        model(TEST_DIR, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(10):
        model(TEST_DIR, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()

    latency = (end - start) / 10 * 1000
    fps = 1000 / latency

    print("==== 性能 ====")
    print(f"Latency: {latency:.2f} ms")
    print(f"FPS: {fps:.2f}")


# 🔴 Windows 必须
if __name__ == "__main__":
    main()
