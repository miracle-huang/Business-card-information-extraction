#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-line summary of this module.

Purpose
-------
Generate the dataset of photos of a business card placed on a background.

Usage
-----
python -m ...

Meta
----
File: gen_seg_dataset.py
Time: 2026-01-05 18:16:59
Author: HuangZhiying
Email: 
"""

import sys
from pathlib import Path

# ✅ 强制把项目根目录加入 sys.path，确保能 import src.*
ROOT = Path(__file__).resolve().parents[1]  # .../Business-card-information-extraction
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synth.seg_synth import SynthConfig, generate_dataset  # noqa: E402


def main():
    cfg = SynthConfig(
        bg_dir=Path("data/background"),
        card_dir=Path("data/business_card_v2"),
        out_dir=Path("data/try_business_card_dectection"),

        num_images=10,

        # 输出画布大小（清晰度相关）
        out_w=1536,
        out_h=1536,

        # 每张图 2~4 张名片
        min_cards=2,
        max_cards=4,

        # ✅ 同一张图内名片“长边一致”，先固定保证一致性与清晰
        min_card_long=700,
        max_card_long=700,
        same_size_in_image=True,

        # 下面这些是新增配置
        target_count_weights=(1.0, 2.0, 10.0),
        prefer_full_target=True,
        retry_prob_if_underfilled={3: 0.25, 4: 0.85},
        min_accept_when_target4=3,

        must_place_all=False,

        # 不做透视，仅旋转
        angle_min=0.0,
        angle_max=360.0,

        # 名片之间留间隙
        # min_gap_px=12,
        min_gap_px=6,
        margin=20,
        max_tries_per_card=250,
        max_image_tries=30,

        seed=42,
        save_debug=True,
    )

    generate_dataset(cfg)
    print(f"Done. Output: {cfg.out_dir.resolve()}")


if __name__ == "__main__":
    main()
