#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# ✅ CONFIG: edit here
# =========================
@dataclass
class SplitConfig:
    # Step1 output dir (contains images/ and labels/)
    src_dir: Path

    # Step2 output dataset dir (will create YOLO structure inside)
    out_dir: Path

    # split mode: "ratio" or "count"
    split_mode: str = "ratio"  # "ratio" or "count"

    # ---- mode = "ratio" ----
    train_ratio: float = 0.8

    # ---- mode = "count" ----
    # You may set:
    # 1) train_count only  -> val_count = total - train_count
    # 2) val_count only    -> train_count = total - val_count
    # 3) both provided     -> must satisfy train+val <= total (remaining are dropped)
    # If both None -> error in count mode
    train_count: Optional[int] = None
    val_count: Optional[int] = None

    # If train+val < total, do we drop leftovers or force use all?
    # - True : drop extras (common when you want fixed-size sets)
    # - False: auto-assign leftover to val (if val_count None) or train (if train_count None),
    #          else error if both specified but not covering all.
    drop_excess: bool = True

    seed: int = 42

    # "copy" is safer; "move" saves disk but destructive
    mode: str = "copy"  # "copy" or "move"

    # YOLO class names (must match Step1 class IDs)
    class_names: Tuple[str, ...] = (
        "card",
        "corner_tl",
        "corner_tr",
        "corner_br",
        "corner_bl",
    )


CONFIG = SplitConfig(
    src_dir=Path(r"data/four_angles/synth_step1_rigid_vertexcorner"),
    out_dir=Path(r"data/four_angles/dataset_yolo"),
    split_mode="count",     # "ratio" or "count"
    # ---- ratio mode ----
    train_ratio=0.8,

    # ---- count mode ----
    train_count=80,        # 只给 train_count 也行
    val_count=40,          # 可选；如果不写，会自动用剩余
    drop_excess=True,       # train+val < total 时，是否丢弃多余样本

    seed=42,
    mode="copy",
)


# =========================
# helpers
# =========================
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
    """
    Return (n_train, n_val) based on cfg.split_mode.
    """
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
        tc = cfg.train_count
        vc = cfg.val_count

        if tc is None and vc is None:
            raise ValueError("count mode requires train_count and/or val_count")

        if tc is not None and tc <= 0:
            raise ValueError(f"train_count must be >0, got {tc}")
        if vc is not None and vc <= 0:
            raise ValueError(f"val_count must be >0, got {vc}")

        if tc is None:
            # only val given
            n_val = vc  # type: ignore
            n_train = total - n_val
        elif vc is None:
            # only train given
            n_train = tc
            n_val = total - n_train
        else:
            # both given
            n_train = tc
            n_val = vc

        if n_train < 1 or n_val < 1:
            raise ValueError(f"Invalid split: train={n_train}, val={n_val} (need both >=1)")

        if n_train + n_val > total:
            raise ValueError(f"train+val exceeds total: {n_train}+{n_val}>{total}")

        if n_train + n_val < total and (not cfg.drop_excess):
            # if not dropping excess, we force using all samples
            # - if one side was None, we'd already allocate leftover there
            # - here implies both specified but not covering all -> error
            if tc is not None and vc is not None:
                raise ValueError(
                    f"drop_excess=False but train+val < total ({n_train+n_val}<{total}). "
                    f"Either set drop_excess=True or adjust counts."
                )

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

    # verify pairs
    pairs = []
    for img in imgs:
        lbl = ensure_pair(img, src_lbl_dir)
        pairs.append((img, lbl))

    # shuffle
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train, n_val = resolve_split_counts(cfg, n_total)

    # choose subset if dropping excess
    selected = pairs[: (n_train + n_val)] if (n_train + n_val) < n_total else pairs
    train_pairs = selected[:n_train]
    val_pairs = selected[n_train : n_train + n_val]

    # prepare output structure (clean rebuild)
    out_images_train = cfg.out_dir / "images" / "train"
    out_images_val = cfg.out_dir / "images" / "val"
    out_labels_train = cfg.out_dir / "labels" / "train"
    out_labels_val = cfg.out_dir / "labels" / "val"

    # WARNING: we clear out_dir to keep it clean
    clear_dir(cfg.out_dir)
    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_val.mkdir(parents=True, exist_ok=True)
    out_labels_train.mkdir(parents=True, exist_ok=True)
    out_labels_val.mkdir(parents=True, exist_ok=True)

    # copy/move
    for img, lbl in train_pairs:
        copy_or_move(img, out_images_train / img.name, cfg.mode)
        copy_or_move(lbl, out_labels_train / lbl.name, cfg.mode)

    for img, lbl in val_pairs:
        copy_or_move(img, out_images_val / img.name, cfg.mode)
        copy_or_move(lbl, out_labels_val / lbl.name, cfg.mode)

    # write dataset.yaml
    yaml_path = cfg.out_dir / "dataset.yaml"
    write_yaml(cfg, yaml_path)

    used = len(train_pairs) + len(val_pairs)
    dropped = n_total - used

    print("[OK] Split done")
    print(f"  mode : {cfg.split_mode}")
    print(f"  total: {n_total}")
    print(f"  train: {len(train_pairs)}")
    print(f"  val  : {len(val_pairs)}")
    print(f"  used : {used}")
    print(f"  drop : {dropped} (drop_excess={cfg.drop_excess})")
    print(f"  out  : {cfg.out_dir}")
    print(f"  yaml : {yaml_path}")


if __name__ == "__main__":
    main(CONFIG)
