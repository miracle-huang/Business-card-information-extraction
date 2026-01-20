#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_jpg(path: Path, img_bgr: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def rotate_exact(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate by exact multiples of 90 degrees WITHOUT introducing blank margins."""
    a = angle % 360
    if a == 0:
        return img
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # CCW
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # CW
    return img


def resize_keep_aspect(img: np.ndarray, long_side: int) -> np.ndarray:
    """
    Optional: make sizes more consistent WITHOUT padding.
    - long_side: target size of max(H,W)
    """
    if long_side <= 0:
        return img
    h, w = img.shape[:2]
    mx = max(h, w)
    if mx == long_side:
        return img
    scale = long_side / float(mx)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


@dataclass
class GenCfg:
    # 输入：几乎无空白的“单张名片原图”（默认正向）
    card_dir: Path = Path(r"data/business_card_raw")

    # 输出：Ultralytics cls 数据集结构（train/val/rot0...）
    out_dir: Path = Path(r"segmentation_classification/data/cls_upright_orient")

    # 切分比例
    train_ratio: float = 0.8
    seed: int = 42

    # 每张原图生成多少套（1套=四个方向各1张）
    sets_per_card: int = 1

    # ✅ 不要 jitter（否则任意角度旋转必然产生四角空白）
    jitter_deg: float = 0.0  # 强制建议保持 0.0

    # 可选：只做“等比例缩放”让尺寸更一致（不会加空白）
    # None/0 表示不缩放；例如设成 960 表示把最长边缩放到 960
    long_side: int = 0

    # 类别名字
    classes: Tuple[str, ...] = ("rot0", "rot90", "rot180", "rot270")

    # 保存质量
    jpg_quality: int = 95


def main(cfg: GenCfg) -> None:
    if cfg.jitter_deg != 0.0:
        raise ValueError(
            "This generator is 'rotation only without blank margins'. "
            "So jitter_deg must be 0.0. Any non-90-degree rotation will introduce blank corners."
        )

    cards = list_images(cfg.card_dir)
    if not cards:
        raise FileNotFoundError(f"No card images found in: {cfg.card_dir}")

    rng = random.Random(cfg.seed)
    rng.shuffle(cards)

    n_total = len(cards)
    n_train = max(1, int(round(n_total * cfg.train_ratio)))
    if n_total >= 2:
        n_train = min(n_total - 1, n_train)

    train_cards = cards[:n_train]
    val_cards = cards[n_train:]

    clear_dir(cfg.out_dir)

    # Prepare dirs
    for split in ("train", "val"):
        for cls in cfg.classes:
            (cfg.out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    base_angles = [0, 90, 180, 270]

    def process_one(card_path: Path, split: str, set_idx: int) -> None:
        img = imread_bgr(card_path)

        # optional resize (no padding)
        if cfg.long_side and cfg.long_side > 0:
            img = resize_keep_aspect(img, cfg.long_side)

        for cls_name, ang in zip(cfg.classes, base_angles):
            out = rotate_exact(img, ang)
            out_name = f"{card_path.stem}__set{set_idx:02d}__{cls_name}.jpg"
            out_path = cfg.out_dir / split / cls_name / out_name
            imwrite_jpg(out_path, out, quality=cfg.jpg_quality)

    # train
    for c in train_cards:
        for s in range(cfg.sets_per_card):
            process_one(c, "train", s)

    # val
    for c in val_cards:
        for s in range(cfg.sets_per_card):
            process_one(c, "val", s)

    train_count = sum(len(list((cfg.out_dir / "train" / cls).glob("*.jpg"))) for cls in cfg.classes)
    val_count = sum(len(list((cfg.out_dir / "val" / cls).glob("*.jpg"))) for cls in cfg.classes)

    print("[OK] Step4 dataset generated (rotation-only, no crop, no padding)")
    print(f"  cards total: {n_total}, train cards: {len(train_cards)}, val cards: {len(val_cards)}")
    print(f"  images: train={train_count}, val={val_count}")
    print(f"  out_dir: {cfg.out_dir.resolve()}")


if __name__ == "__main__":
    CONFIG = GenCfg(
        card_dir=Path(r"data/business_card_raw"),
        out_dir=Path(r"segmentation_classification/data/cls_upright_orient"),
        train_ratio=0.8,
        sets_per_card=1,
        jitter_deg=0.0,   # ✅ 必须 0
        long_side=0,      # 可选：比如 960，让尺寸更一致（不加空白）
        seed=42,
        jpg_quality=95,
    )
    main(CONFIG)
