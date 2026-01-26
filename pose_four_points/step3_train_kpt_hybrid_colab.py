"""
[Step 3 Colab] 混合训练脚本 (Colab 优化版)
功能：针对 Google Colab 环境优化的关键点检测模型训练脚本。
改进点：
1. 默认使用 /content/ 目录存储动态合成数据，避免 Google Drive 同步延迟导致的 FileNotFoundError。
2. 修复了 Epoch 0 重复刷新数据集的问题。
3. 自动修正数据集路径为绝对路径，增强稳定性。
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ✅ 强制将当前目录和根目录加入 path
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from synth.kpt_synth import SynthKptConfig
# 使用 Colab 优化的 Trainer
from hybrid_train.hybrid_pose_trainer_colab import HybridPoseTrainerColab

# =========================
# 参数配置
# =========================

# 是否在 Colab 环境（自动检测）
IS_COLAB = os.path.exists('/content')

# 1. 静态数据集配置
DATA_YAML = str(CUR_DIR / "assets" / "step2_train_dataset" / "dataset_card4kpt.yaml")

# 2. 训练输出目录
PROJECT_DIR = str(CUR_DIR / "run")
EXP_NAME = "kpt_hybrid_yolo11x_pose_colab"

# 3. 模型配置
# 建议在 Colab A100/L4 上使用较大的模型，如 x-pose 或 m-pose
MODEL_WEIGHTS = "yolo11x-pose.pt" 

# 4. 训练超参
EPOCHS = 50
IMGSZ = 640
BATCH = 32          # Colab A100 80G 可以开更大，比如 32 或 64
DEVICE = "0"        # GPU
WORKERS = 4         # Colab 上可以开启多线程，通常 4-8 比较合适

# 5. 动态数据生成配置
# ✅ 关键优化：如果是 Colab，直接使用本地 SSD 路径 /content/，彻底解决 FileNotFoundError
if IS_COLAB:
    RUNTIME_DIR = Path("/content/synth_kpt_runtime")
    print(f"检测到 Colab 环境，动态数据集将存储在本地 SSD: {RUNTIME_DIR}")
else:
    RUNTIME_DIR = CUR_DIR / "assets" / "synth_kpt_runtime"

# 背景和名片素材路径（确保是绝对路径）
BG_DIR = REPO_ROOT / "data" / "background"
CARD_DIR = REPO_ROOT / "data" / "business_card_raw"

# 检查路径是否存在
if not BG_DIR.exists():
    print(f"警告: 背景目录不存在: {BG_DIR}")
if not CARD_DIR.exists():
    print(f"警告: 名片素材目录不存在: {CARD_DIR}")

# 动态数据规模
RUNTIME_MULTIPLIER = 1.0
RUNTIME_SEED = 12345

# synth 生成图像大小
OUT_W = 1536
OUT_H = 1536
MIN_CARDS = 2
MAX_CARDS = 4

# 关闭 YOLO 默认增强，依靠混合合成
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
    # 1) 配置 Trainer 参数
    HybridPoseTrainerColab.RUNTIME_DIR = RUNTIME_DIR
    HybridPoseTrainerColab.RUNTIME_SEED = RUNTIME_SEED
    HybridPoseTrainerColab.RUNTIME_MULTIPLIER = RUNTIME_MULTIPLIER

    HybridPoseTrainerColab.RUNTIME_SYNTH_CFG = SynthKptConfig(
        bg_dir=BG_DIR,
        card_dir=CARD_DIR,
        out_dir=RUNTIME_DIR,
        num_images=10, # 占位符，会被自动计算
        out_w=OUT_W,
        out_h=OUT_H,
        min_cards=MIN_CARDS,
        max_cards=MAX_CARDS,
    )

    # 2) 加载模型
    model = YOLO(MODEL_WEIGHTS)

    # 3) 启动训练
    overrides = dict(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXP_NAME,
        plots=False,
        **NO_AUG_OVERRIDES,
    )

    # 使用 Colab 优化的 Trainer
    model.train(trainer=HybridPoseTrainerColab, **overrides)

if __name__ == "__main__":
    main()
