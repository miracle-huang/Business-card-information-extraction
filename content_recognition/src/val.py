from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO


def main(
    model: str,
    data: str,
    device: str = "0",
    split: str = "val",
) -> None:
    """
    YOLO validation entry (code-driven, no CLI)

    Args:
        model: path to trained model weights (.pt), e.g. best.pt
        data: path to data.yaml
        device: '0', 'cpu', '0,1', etc.
        split: which split to evaluate ('val' or 'test')
    """
    # 1) 加载模型
    model = YOLO(model)

    # 2) 运行验证 / 测试
    metrics = model.val(
        data=data,
        device=device,
        split=split,   # 默认用 val，也可以改成 test
    )

    # 3) 打印核心指标（detect 任务）
    print("\n[Validation Results]")
    try:
        print(f"mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"Precision    : {metrics.box.mp:.4f}")
        print(f"Recall       : {metrics.box.mr:.4f}")
    except Exception:
        # 防止不同版本 Ultralytics 字段略有差异
        print(metrics)

    # 4) 输出保存目录（如果有）
    if hasattr(metrics, "save_dir"):
        print(f"\n[Done] Save dir: {metrics.save_dir}")


if __name__ == "__main__":
    # ===== 在这里直接传参数 =====
    main(
        model="runs_local/debug_yolo11m_bs2_img640/weights/best.pt",
        data="data/roboflow_v1/data.yaml",
        device="0",     # 或 "cpu"
        split="test",    # 可改为 "test"
    )