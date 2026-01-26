"""
[Step 3 Colab] 混合训练脚本 (Colab 优化版 v3)
功能：针对 Google Colab 环境优化的关键点检测模型训练脚本。
改进点：
1. 采用更鲁棒的路径探测机制，兼容 Google Drive 可能存在的同步延迟或大小写问题。
2. 自动生成临时 YAML 并修正 'path' 为 Linux 绝对路径。
3. 将动态数据存储在 /content/ 以避开 Drive 延迟。
"""
from __future__ import annotations

import sys
import os
import yaml
import time
from pathlib import Path

# ✅ 配置路径
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"--- Colab Environment Setup ---")
print(f"Working Dir: {Path.cwd()}")
print(f"Script Location: {__file__}")

from ultralytics import YOLO
from synth.kpt_synth import SynthKptConfig
from hybrid_train.hybrid_pose_trainer_colab import HybridPoseTrainerColab

# =========================
# 路径探测与 YAML 修复
# =========================

def get_robust_yaml_path():
    # 预期路径
    expected = CUR_DIR / "assets" / "step2_train_dataset" / "dataset_card4kpt.yaml"
    
    if expected.exists():
        return expected
    
    print(f"警告: 在预期位置未找到 YAML: {expected}")
    print("正在尝试自动探测数据集配置文件...")
    
    # 尝试在 assets 下搜索任何 .yaml 文件
    assets_dir = CUR_DIR / "assets"
    if assets_dir.exists():
        yaml_files = list(assets_dir.rglob("*.yaml"))
        # 排除掉我们自己生成的 _colab.yaml
        yaml_files = [f for f in yaml_files if "_colab" not in f.name]
        if yaml_files:
            print(f"探测到候选 YAML: {yaml_files[0]}")
            return yaml_files[0]
            
    return expected

ORIGINAL_YAML = get_robust_yaml_path()
COLAB_YAML_PATH = CUR_DIR / "assets" / "step2_train_dataset" / "dataset_card4kpt_colab.yaml"

# 确保目标目录存在
COLAB_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

if ORIGINAL_YAML.exists():
    try:
        with open(ORIGINAL_YAML, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 修正 path 为绝对路径（指向 dataset 所在目录）
        data_config['path'] = str(ORIGINAL_YAML.parent.resolve())
        print(f"成功加载并修复 YAML 内容，path -> {data_config['path']}")
        
        with open(COLAB_YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)
        
        DATA_YAML = str(COLAB_YAML_PATH)
    except Exception as e:
        print(f"处理 YAML 时发生错误: {e}")
        DATA_YAML = str(ORIGINAL_YAML)
else:
    print(f"严重错误: 无法找到数据集 YAML 文件。")
    DATA_YAML = str(ORIGINAL_YAML)

# =========================
# 训练参数
# =========================

IS_COLAB = os.path.exists('/content')
PROJECT_DIR = str(CUR_DIR / "run")
EXP_NAME = "kpt_hybrid_yolo11x_pose_colab"
MODEL_WEIGHTS = "yolo11x-pose.pt" 

EPOCHS = 50
IMGSZ = 640
BATCH = 32
DEVICE = "0"
WORKERS = 4

# 动态数据路径优化
if IS_COLAB:
    RUNTIME_DIR = Path("/content/synth_kpt_runtime")
    # 修正 REPO_ROOT 下的数据路径，确保是绝对路径
    BG_DIR = REPO_ROOT / "data" / "background"
    CARD_DIR = REPO_ROOT / "data" / "business_card_raw"
else:
    RUNTIME_DIR = CUR_DIR / "assets" / "synth_kpt_runtime"
    BG_DIR = Path("data/background")
    CARD_DIR = Path("data/business_card_raw")

# YOLO 增强关闭
NO_AUG_OVERRIDES = dict(
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0, close_mosaic=0,
)

def main():
    # 检测素材路径是否存在
    if not BG_DIR.exists():
        print(f"错误: 背景素材路径不存在: {BG_DIR}")
    if not CARD_DIR.exists():
        print(f"错误: 名片素材路径不存在: {CARD_DIR}")

    HybridPoseTrainerColab.RUNTIME_DIR = RUNTIME_DIR
    HybridPoseTrainerColab.RUNTIME_SEED = 12345
    HybridPoseTrainerColab.RUNTIME_MULTIPLIER = 1.0

    HybridPoseTrainerColab.RUNTIME_SYNTH_CFG = SynthKptConfig(
        bg_dir=BG_DIR,
        card_dir=CARD_DIR,
        out_dir=RUNTIME_DIR,
        num_images=10, 
        out_w=1536,
        out_h=1536,
        min_cards=2,
        max_cards=4,
    )

    print(f"正在加载模型: {MODEL_WEIGHTS}")
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
        plots=False,
        **NO_AUG_OVERRIDES,
    )

    print("开始训练...")
    model.train(trainer=HybridPoseTrainerColab, **overrides)

if __name__ == "__main__":
    main()
