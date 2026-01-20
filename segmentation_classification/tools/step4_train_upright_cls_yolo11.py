#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =========================
# CONFIG (edit here)
# =========================
DATA_DIR = Path(r"segmentation_classification/assets/step4_cls_upright_orient")  # step4生成的目录
# 推荐：用本地权重，避免自动下载失败
MODEL = r"yolo11n-cls.pt"  # 或者 "yolo11n-cls.pt"（若你能联网）
EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 4
SEED = 42

ULTRA_PROJECT = str(Path("segmentation_classification") / "runs")
ULTRA_NAME = "step4_upright_cls"
ULTRA_EXIST_OK = True


def main() -> None:
    # hard check if local model path is used
    m = Path(MODEL)
    if (str(m).endswith(".pt") or str(m).endswith(".yaml")) and not m.exists():
        raise FileNotFoundError(
            f"Missing model file: {m}\n"
            "If you cannot download automatically, please put yolo11n-cls.pt in segmentation_classification/weights/ "
            "and set MODEL to that local path."
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_DIR),   # cls 任务直接给根目录
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,

        project=ULTRA_PROJECT,
        name=ULTRA_NAME,
        exist_ok=ULTRA_EXIST_OK,

        # 不额外加增强，保持干净
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        amp=True,
        pretrained=True,
    )

    print("[OK] Step4 cls training finished.")
    out_dir = Path(ULTRA_PROJECT) / ULTRA_NAME / "weights"
    print("  weights:")
    print(f"    best: {out_dir / 'best.pt'}")
    print(f"    last: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
