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
    out_dir=Path(r"four_angles/assets/step1_test"),

    num_images=20,            # ✅ 建议：至少几千张（corners 小目标更吃数据）

    min_cards=2,
    max_cards=4,

    # ✅ 角点框变大后，建议把 margin_to_img 也略增（避免 strict_center 导致摆放更难）
    margin_to_img=150,
    min_gap_between_cards=60,
    fixed_card_w=700,

    angle_min=0.0,
    angle_max=360.0,

    # ✅ 关键：角点框变大（提升 recall）
    corner_box_ratio=0.40,
    corner_box_min=120,
    corner_box_max=220,
    strict_corner_center=True,  # ✅ 保证“顶点=框中心”，出界就放弃

    save_debug=True,            # 批量生成想快一点可改 False
    num_workers=6,
    seed=12345,
)


if __name__ == "__main__":
    import four_angles.src.synth.rigid_synth as rs
    print("[DBG] using rigid_synth:", rs.__file__)
    print("[DBG] seed:", CONFIG.seed)
    print("[DBG] out_dir:", CONFIG.out_dir)

    generate_dataset(CONFIG, overwrite=True)
    print("[OK] Step1 done:", CONFIG.out_dir.resolve())
