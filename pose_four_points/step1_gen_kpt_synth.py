"""
[Step 1] 静态合成数据生成脚本
功能：从背景库和名片模板库中随机采样，批量生成带 YOLO-Pose 标签（4个关键点）的训练集。
用于生成“静态 Pool”，作为模型训练的基础数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

# ✅ ensure current dir on sys.path for local imports
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from synth.kpt_synth import SynthKptConfig, generate_dataset

# =========================
# ✅ 只改这里的参数即可
# =========================
CFG = SynthKptConfig(
    # 输入路径（按你要求）
    bg_dir=Path(r"data/background"),
    card_dir=Path(r"data/business_card_raw"),

    # 输出路径
    out_dir=Path(r"pose_four_points/assets/step1_test"),

    # 生成多少张
    num_images=20,

    # 输出图片尺寸
    out_w=1536,
    out_h=1536,

    # 每张图放几张名片（2~4）
    min_cards=2,
    max_cards=4,

    # 旋转范围
    angle_min=0.0,
    angle_max=360.0,

    # 可视化检查（强烈建议先开）
    save_viz=True,

    # 不重叠采样的尝试次数（放不下时可以调大）
    max_tries_per_card=800,
    max_restarts_per_image=80,
)
# 随机种子（可复现）
SEED = 42
# 并行进程数（根据你的 CPU 核心数调整）
NUM_WORKERS = 6


def main():
    generate_dataset(CFG, seed=SEED, num_workers=NUM_WORKERS)


if __name__ == "__main__":
    main()
