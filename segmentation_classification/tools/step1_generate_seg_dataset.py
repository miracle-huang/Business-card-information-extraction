#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# ✅ ensure segmentation_classification root on sys.path
PROJ = Path(__file__).resolve().parents[1]  # .../segmentation_classification
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.synth.rigid_seg_synth import SynthConfig, generate_dataset  # noqa: E402


# =========================
# CONFIG: edit here (NO CLI args)
# =========================
CONFIG = SynthConfig(
    # 你自己改成真实路径（也可用相对路径）
    bg_dir=Path(r"data/background"),
    card_dir=Path(r"data/business_card_raw"),
    out_dir=Path(r"segmentation_classification/assets/seg_step1_output"),

    num_images=500,

    min_cards=2,
    max_cards=4,

    margin_to_img=120,
    min_gap_between_cards=50,

    weight_2=3.0,
    weight_3=3.0,
    weight_4=5.0,

    dynamic_bg_enlarge=True,
    dynamic_bg_only_for_3plus=True,
    max_bg_scale=4.0,

    # ✅ 名片大小基本一致：固定宽度，只做旋转（无透视、无颜色增强）
    fixed_card_w=720,

    angle_min=0.0,
    angle_max=360.0,

    # 如果你想强约束更严格，就把 trials / retries 提高一点
    max_place_trials_per_card=160,
    max_image_retries=60,

    # 输出尺寸：None 表示保持背景原始尺寸；如果你想统一训练尺寸就指定
    out_w=None,
    out_h=None,

    save_debug=True,
    num_workers=6,
    seed=42,
)


if __name__ == "__main__":
    import src.synth.rigid_seg_synth as rs  # noqa: E402

    print("[DBG] using rigid_seg_synth:", rs.__file__)
    print("[DBG] seed:", CONFIG.seed)
    print("[DBG] out_dir:", CONFIG.out_dir)

    generate_dataset(CONFIG, overwrite=True)
    print("[OK] Step1 done:", CONFIG.out_dir.resolve())
