#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# ✅ ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]  # .../Business-card-information-extraction
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from four_angles.src.synth.rigid_synth import SynthConfig, generate_dataset  # noqa: E402


# =========================
# CONFIG: edit here (NO CLI args)
# =========================
CONFIG = SynthConfig(
    bg_dir=Path(r"data/background"),
    card_dir=Path(r"data/business_card_raw"),
    out_dir=Path(r"four_angles/assets/step1_out_test_model"),

    num_images=20,

    # out_w=1920, out_h=1080,  # 可选：统一背景尺寸；否则保持原尺寸

    min_cards=2,
    max_cards=4,

    margin_to_img=90,
    min_gap_between_cards=40,   # 名片之间距离更大就调大
    fixed_card_w=700,

    angle_min=0.0,
    angle_max=360.0,

    save_debug=True,            # 批量生成想快一点可改 False
    num_workers=6,              # 你也可以改成 1（单进程）
    seed=42,
)


if __name__ == "__main__":
    generate_dataset(CONFIG, overwrite=True)
    print("[OK] Step1 done:", CONFIG.out_dir.resolve())
