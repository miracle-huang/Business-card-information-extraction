# script/obb/train_kpt_hybrid.py
from __future__ import annotations

import sys
from pathlib import Path

# ✅ ensure project root on sys.path, fix: ModuleNotFoundError: No module named 'src'
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.synth.kpt_synth import SynthKptConfig
from src.hybrid_train.hybrid_pose_trainer import HybridPoseTrainer


# =========================
# 你只需要改这里的参数（不走命令行）
# =========================
# Step2 输出（静态数据集）
DATA_YAML = r"configs\dataset_card4kpt.yaml"

# 训练输出目录
PROJECT_DIR = r"runs_kpt"
EXP_NAME = r"kpt_hybrid_no_aug"

# 模型（pose）
MODEL_WEIGHTS = r"yolo11m-pose.pt"  # 如果你是 YOLOv8，用 yolov8n-pose.pt

# 训练超参
EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = "0"        # "cpu" / "0" / "0,1"
WORKERS = 0         # ✅ 强烈建议 Windows 下先用 0，避免多进程文件占用

# 动态生成数据（步骤1同源）
BG_DIR = Path(r"data\background")
CARD_DIR = Path(r"data\business_card_raw")
RUNTIME_DIR = Path(r"data\synth_kpt_runtime")

# 动态数据规模：每个 epoch 动态生成 = static_train_len * MULTIPLIER
RUNTIME_MULTIPLIER = 1.0
RUNTIME_SEED = 12345

# synth 生成图像大小（应与你步骤1一致）
OUT_W = 1536
OUT_H = 1536

# 每张图放几张名片（与你步骤1一致）
MIN_CARDS = 2
MAX_CARDS = 4

# 不启用任何数据增强：YOLO 侧超参全部设为 0
NO_AUG_OVERRIDES = dict(
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    bgr=0.0,
    close_mosaic=0,
)


def main():
    # 1) 配置 HybridPoseTrainer 的“动态数据生成参数”
    HybridPoseTrainer.RUNTIME_DIR = RUNTIME_DIR
    HybridPoseTrainer.RUNTIME_SEED = RUNTIME_SEED
    HybridPoseTrainer.RUNTIME_MULTIPLIER = RUNTIME_MULTIPLIER

    HybridPoseTrainer.RUNTIME_SYNTH_CFG = SynthKptConfig(
        bg_dir=BG_DIR,
        card_dir=CARD_DIR,
        out_dir=RUNTIME_DIR,   # 会被 trainer 每个 epoch 替换/清空
        num_images=10,         # 会被 trainer 按 static_len 自动覆盖
        out_w=OUT_W,
        out_h=OUT_H,
        min_cards=MIN_CARDS,
        max_cards=MAX_CARDS,
        # 其余 synth 参数沿用你步骤1的默认值（你需要的话可继续在这里补齐）
    )

    # 2) 启动训练
    model = YOLO(MODEL_WEIGHTS)

    overrides = dict(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXP_NAME,
        plots=False, # 跳过打印标签分布图
        # 关闭增强
        **NO_AUG_OVERRIDES,
    )

    model.train(trainer=HybridPoseTrainer, **overrides)


if __name__ == "__main__":
    main()
