from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml


def _read_data_yaml(data_yaml: str | Path) -> Dict[str, Any]:
    p = Path(data_yaml)
    if not p.exists():
        raise FileNotFoundError(f"data.yaml not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(base: Path, split_rel: str) -> Path:
    # data.yaml 里 train/val/test 通常是相对 path 的相对路径
    return (base / split_rel).resolve()


def sanity_check_yolo_dataset(data_yaml: str | Path) -> None:
    cfg = _read_data_yaml(data_yaml)
    base = Path(cfg["path"]).resolve()

    for k in ["train", "val"]:
        if k not in cfg:
            raise ValueError(f"data.yaml missing key: {k}")

    splits = {"train": cfg["train"], "val": cfg["val"], "test": cfg.get("test")}
    print(f"[Sanity] base path: {base}")

    for split, rel_img in splits.items():
        if rel_img is None:
            continue
        img_dir = resolve_split_dir(base, rel_img)
        if not img_dir.exists():
            raise FileNotFoundError(f"[Sanity] {split} images dir not found: {img_dir}")

        # images -> labels
        # 约定：xxx/images 对应 xxx/labels
        if img_dir.name != "images":
            # 也支持用户自定义 images 路径，尽力推断 labels
            labels_dir = img_dir.parent / "labels"
        else:
            labels_dir = img_dir.parent / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"[Sanity] {split} labels dir not found: {labels_dir}")

        img_files = list(img_dir.glob("*.*"))
        if len(img_files) == 0:
            raise ValueError(f"[Sanity] {split} images empty: {img_dir}")

        # 抽样检查 label 是否存在
        sample = img_files[:20]
        missing = 0
        for img in sample:
            label = labels_dir / (img.stem + ".txt")
            if not label.exists():
                missing += 1
        print(f"[Sanity] {split}: images={len(img_files)}, labels_dir={labels_dir}, "
              f"missing_labels_in_first_{len(sample)}={missing}")

    nc = cfg.get("nc")
    names = cfg.get("names")
    print(f"[Sanity] nc={nc}, names={names}")
