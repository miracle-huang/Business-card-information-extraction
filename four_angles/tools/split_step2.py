#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SplitConfig:
    src_dir: Path
    out_dir: Path

    split_mode: str = "count"  # "ratio" or "count"
    train_ratio: float = 0.8

    train_count: Optional[int] = 40
    val_count: Optional[int] = 20
    drop_excess: bool = True

    seed: int = 42
    mode: str = "copy"  # "copy" or "move"

    class_names: Tuple[str, ...] = (
        "card",
        "corner_tl",
        "corner_tr",
        "corner_br",
        "corner_bl",
    )


CONFIG = SplitConfig(
    src_dir=Path(r"four_angles/assets/step1_out"),
    out_dir=Path(r"four_angles/assets/step2_out_dataset_yolo"),
    split_mode="count",
    train_count=40,
    val_count=20,
    drop_excess=True,
    seed=42,
    mode="copy",
)


def list_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def ensure_pair(img_path: Path, lbl_dir: Path) -> Path:
    lbl = lbl_dir / f"{img_path.stem}.txt"
    if not lbl.exists():
        raise FileNotFoundError(f"Missing label for image: {img_path.name} -> {lbl}")
    return lbl


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode} (use 'copy' or 'move')")


def write_yaml(cfg: SplitConfig, yaml_path: Path) -> None:
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(cfg.class_names)])
    text = (
        f"path: {cfg.out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_lines}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def resolve_split_counts(cfg: SplitConfig, total: int) -> Tuple[int, int]:
    if total < 2:
        raise ValueError(f"Need at least 2 samples to split, but got total={total}")

    mode = cfg.split_mode.lower().strip()
    if mode == "ratio":
        if not (0.0 < cfg.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {cfg.train_ratio}")
        n_train = int(round(total * cfg.train_ratio))
        n_train = max(1, min(total - 1, n_train))
        n_val = total - n_train
        return n_train, n_val

    if mode == "count":
        tc, vc = cfg.train_count, cfg.val_count
        if tc is None and vc is None:
            raise ValueError("count mode requires train_count and/or val_count")
        if tc is None:
            n_val = int(vc)  # type: ignore
            n_train = total - n_val
        elif vc is None:
            n_train = int(tc)
            n_val = total - n_train
        else:
            n_train, n_val = int(tc), int(vc)

        if n_train < 1 or n_val < 1:
            raise ValueError(f"Invalid split: train={n_train}, val={n_val} (need both >=1)")
        if n_train + n_val > total:
            raise ValueError(f"train+val exceeds total: {n_train}+{n_val}>{total}")
        return n_train, n_val

    raise ValueError(f"Unknown split_mode: {cfg.split_mode} (use 'ratio' or 'count')")


def main(cfg: SplitConfig) -> None:
    src_img_dir = cfg.src_dir / "images"
    src_lbl_dir = cfg.src_dir / "labels"
    if not src_img_dir.exists():
        raise FileNotFoundError(f"Missing: {src_img_dir}")
    if not src_lbl_dir.exists():
        raise FileNotFoundError(f"Missing: {src_lbl_dir}")

    imgs = list_images(src_img_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found in: {src_img_dir}")

    pairs = []
    for img in imgs:
        lbl = ensure_pair(img, src_lbl_dir)
        pairs.append((img, lbl))

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train, n_val = resolve_split_counts(cfg, n_total)

    selected = pairs[: (n_train + n_val)] if (n_train + n_val) < n_total else pairs
    train_pairs = selected[:n_train]
    val_pairs = selected[n_train : n_train + n_val]

    out_images_train = cfg.out_dir / "images" / "train"
    out_images_val = cfg.out_dir / "images" / "val"
    out_labels_train = cfg.out_dir / "labels" / "train"
    out_labels_val = cfg.out_dir / "labels" / "val"

    clear_dir(cfg.out_dir)
    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_val.mkdir(parents=True, exist_ok=True)
    out_labels_train.mkdir(parents=True, exist_ok=True)
    out_labels_val.mkdir(parents=True, exist_ok=True)

    for img, lbl in train_pairs:
        copy_or_move(img, out_images_train / img.name, cfg.mode)
        copy_or_move(lbl, out_labels_train / lbl.name, cfg.mode)

    for img, lbl in val_pairs:
        copy_or_move(img, out_images_val / img.name, cfg.mode)
        copy_or_move(lbl, out_labels_val / lbl.name, cfg.mode)

    yaml_path = cfg.out_dir / "dataset.yaml"
    write_yaml(cfg, yaml_path)

    print("[OK] Step2 split done")
    print(f"  total: {n_total}, train: {len(train_pairs)}, val: {len(val_pairs)}")
    print(f"  out  : {cfg.out_dir.resolve()}")
    print(f"  yaml : {yaml_path.resolve()}")


if __name__ == "__main__":
    main(CONFIG)
