from __future__ import annotations
from ultralytics import YOLO


def main(
    model: str,
    source: str,
    device: str = "0",
    save: bool = False,
) -> None:
    """
    YOLO prediction entry (code-driven, no CLI)

    Args:
        model: path to .pt model (e.g., yolo11n.pt or runs/.../best.pt)
        source: image / directory / video / camera id
        device: '0', '1', 'cpu', '0,1', etc.
        save: whether to save prediction results
    """
    model = YOLO(model)

    results = model.predict(
        source=source,
        device=device,
        save=save,
    )

    print(f"pred results: {len(results)}")


if __name__ == "__main__":
    # ===== 在这里直接传参数 =====
    main(
        model="runs_local/debug_yolo11m_bs2_img640/weights/best.pt",
        # source="data/roboflow_v1/test/images",
        source="data/real-business-card",
        device="0",        # 或 "cpu"
        save=True,
    )