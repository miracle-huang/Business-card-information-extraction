#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
segmentation_classification/tools/step3_train_seg_routeA.py

Step3 (Route A) - YOLO11 Segmentation Training:
- Per-batch 50% static + 50% runtime synthetic data (MixedBatchSampler).
- ✅ Runtime synthetic images count per epoch == static train images count (N).
- ✅ Avoid WinError 32: write runtime into epoch subfolders: _runtime/rt_e000, rt_e001, ...
- ✅ Can use multi workers because each epoch rebuilds a NEW InfiniteDataLoader.

Fix for Ultralytics 8.3.240 progress > 100%:
- Ultralytics may compute progress total from len(train_loader) or a cached nb.
- InfiniteDataLoader.__len__ may not reflect our batch_sampler length.
- We wrap train_loader with a LenProxyLoader so that len(loader) == num_batches (true batches).
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

# =========================
# Ensure repo root on sys.path
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentation_classification.src.synth.rigid_seg_synth import (  # noqa: E402
    SynthConfig,
    generate_dataset,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# CONFIG (edit here)
# =========================
DATASET_ROOT = Path(r"segmentation_classification/assets/seg_step2_yolo_dataset")
DATA_YAML = DATASET_ROOT / "dataset.yaml"
STATIC_TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"
STATIC_VAL_IMG_DIR = DATASET_ROOT / "images" / "val"

BG_DIR = Path(r"data/background")
CARD_DIR = Path(r"data/business_card_raw")

RUNTIME_ROOT = DATASET_ROOT / "_runtime"
RUNTIME_KEEP_LAST_N_EPOCHS = 3  # keep last N runtime dirs (best-effort delete older)

RUNTIME_CHECK_DIR = DATASET_ROOT / "_runtime_check"
RUNTIME_CHECK_SAMPLES_PER_EPOCH = 1

MODEL_WEIGHTS = "yolo11s-seg.pt" # yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
EPOCHS = 100
IMGSZ = 960
BATCH = 4          # even recommended
WORKERS = 4
DEVICE = 0
SEED = 42

SYNTH_TEMPLATE = dict(
    out_w=None,
    out_h=None,
    min_cards=2,
    max_cards=4,
    margin_to_img=120,
    min_gap_between_cards=50,
    fixed_card_w=700,
    angle_min=0.0,
    angle_max=360.0,
    max_place_trials_per_card=160,
    max_image_retries=60,
    weight_2=3.0,
    weight_3=3.0,
    weight_4=5.0,
    dynamic_bg_enlarge=True,
    dynamic_bg_only_for_3plus=True,
    max_bg_scale=4.0,
)

RUNTIME_SYNTH_WORKERS = 4

ULTRA_PROJECT = str(Path("segmentation_classification") / "runs")
ULTRA_NAME = "step3_seg"
ULTRA_EXIST_OK = True


# =========================
# Small FS helpers (Windows-safe)
# =========================
def safe_rmtree(p: Path) -> bool:
    if not p.exists():
        return True
    try:
        shutil.rmtree(p)
        return True
    except Exception:
        return False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


# =========================
# Loader length proxy (critical fix)
# =========================
class LenProxyLoader:
    """
    Wrap a DataLoader-like object but force __len__ to a fixed value.
    This makes Ultralytics progress bar total consistent with our MixedBatchSampler length.
    """
    def __init__(self, loader, fixed_len: int):
        self._loader = loader
        self._fixed_len = int(fixed_len)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return self._fixed_len

    def __getattr__(self, name):
        # forward everything else
        return getattr(self._loader, name)


# =========================
# Mixed loader: 50% static + 50% runtime per batch
# =========================
class MixedProxyDataset:
    """Route (src, idx) to static or runtime dataset."""
    def __init__(self, ds_static, ds_dyn):
        self.ds_static = ds_static
        self.ds_dyn = ds_dyn
        self.collate_fn = getattr(ds_static, "collate_fn", None) or getattr(ds_dyn, "collate_fn", None)

    def __len__(self) -> int:
        return max(len(self.ds_static), 1)

    def __getitem__(self, item):
        src, idx = item
        return self.ds_static[int(idx)] if src == 0 else self.ds_dyn[int(idx)]


class MixedBatchSampler:
    """Each batch: half static, half runtime."""
    def __init__(self, n_static: int, n_dyn: int, batch_size: int, num_batches: int, seed: int):
        self.n_static = int(n_static)
        self.n_dyn = int(n_dyn)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.bs_static = max(1, self.batch_size // 2)
        self.bs_dyn = self.batch_size - self.bs_static

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        s_idx = list(range(self.n_static))
        d_idx = list(range(self.n_dyn))
        rng.shuffle(s_idx)
        rng.shuffle(d_idx)
        sp, dp = 0, 0

        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.bs_static):
                if sp >= len(s_idx):
                    rng.shuffle(s_idx)
                    sp = 0
                batch.append((0, s_idx[sp]))
                sp += 1
            for _ in range(self.bs_dyn):
                if dp >= len(d_idx):
                    rng.shuffle(d_idx)
                    dp = 0
                batch.append((1, d_idx[dp]))
                dp += 1
            rng.shuffle(batch)
            yield batch


# =========================
# Runtime sanity check (draw polygons from YOLO-seg labels)
# =========================
def _imread_bgr(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _read_yolo_seg_polys(lbl_path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Synth labels: cls x1 y1 x2 y2 ... (normalized pairs).
    """
    if not lbl_path.exists():
        return []
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 1 + 6:
            continue
        cid = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
        if len(nums) % 2 != 0:
            continue
        pts = []
        for i in range(0, len(nums), 2):
            pts.append((nums[i], nums[i + 1]))
        out.append((cid, pts))
    return out


def save_runtime_check_images(runtime_dir: Path, out_dir: Path, epoch: int, k: int, seed: int):
    img_dir = runtime_dir / "images"
    lbl_dir = runtime_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return

    imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        return

    rng = random.Random(seed)
    picks = rng.sample(imgs, k=min(k, len(imgs)))
    ensure_dir(out_dir)

    for i, img_path in enumerate(picks):
        img = _imread_bgr(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        polys = _read_yolo_seg_polys(lbl_path)

        for cid, pts_norm in polys:
            pts = np.array([[int(x * W), int(y * H)] for (x, y) in pts_norm], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            if len(pts) > 0:
                cv2.putText(img, str(cid), (pts[0][0], max(0, pts[0][1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_path = out_dir / f"epoch_{epoch:03d}_{i:02d}_{img_path.stem}.jpg"
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(str(out_path))


# =========================
# Ultralytics callback wiring
# =========================
def register_callback(model, event_name: str, fn):
    if hasattr(model, "add_callback"):
        model.add_callback(event_name, fn)
        return
    cbs = getattr(model, "callbacks", None)
    if isinstance(cbs, dict):
        cbs.setdefault(event_name, []).append(fn)
        return
    raise RuntimeError("Cannot register callback (no add_callback/callbacks).")


def _ddp_barrier_if_available():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _cleanup_old_runtime_dirs(current_epoch: int) -> None:
    if RUNTIME_KEEP_LAST_N_EPOCHS <= 0:
        return
    threshold = current_epoch - RUNTIME_KEEP_LAST_N_EPOCHS
    if threshold < 0:
        return
    for d in sorted(RUNTIME_ROOT.glob("rt_e*")):
        try:
            name = d.name
            if not name.startswith("rt_e"):
                continue
            ep = int(name.replace("rt_e", ""))
            if ep <= threshold:
                safe_rmtree(d)
        except Exception:
            continue


def build_mixed_train_loader(trainer) -> None:
    """
    Core:
    - Generate runtime dataset into epoch folder (_runtime/rt_e###), dyn_images == static_train_count (N).
    - Build ds_static & ds_dyn via build_yolo_dataset.
    - Build a NEW InfiniteDataLoader with num_workers>0 and assign to trainer.train_loader.
    - ✅ Force len(train_loader) == num_batches (true), to fix progress bar total in ultralytics 8.3.240.
    """
    from ultralytics.data.build import InfiniteDataLoader, build_yolo_dataset, seed_worker

    epoch = int(getattr(trainer, "epoch", 0))
    epoch_seed = SEED + 1000 * epoch

    static_imgs = list_images(STATIC_TRAIN_IMG_DIR)
    static_count = len(static_imgs)
    if static_count <= 0:
        raise RuntimeError(f"No static train images found: {STATIC_TRAIN_IMG_DIR}")

    batch = int(getattr(trainer.args, "batch", BATCH))
    bs_static = max(1, batch // 2)
    bs_dyn = batch - bs_static

    # requirement: runtime images count == static train images count
    dyn_images = static_count

    # true epoch length: cover all static once (approx)
    # because each batch only has bs_static static samples
    num_batches = int(math.ceil(static_count / float(bs_static)))

    rank = int(getattr(trainer, "rank", 0))
    runtime_epoch_dir = RUNTIME_ROOT / f"rt_e{epoch:03d}"

    # 1) Generate runtime dataset (rank0 only)
    if rank == 0:
        ensure_dir(RUNTIME_ROOT)
        safe_rmtree(runtime_epoch_dir)
        ensure_dir(runtime_epoch_dir)

        cfg = SynthConfig(
            bg_dir=BG_DIR,
            card_dir=CARD_DIR,
            out_dir=runtime_epoch_dir,
            num_images=dyn_images,
            save_debug=False,
            num_workers=RUNTIME_SYNTH_WORKERS,
            seed=epoch_seed,
            **SYNTH_TEMPLATE,
        )
        generate_dataset(cfg, overwrite=False)

        if RUNTIME_CHECK_SAMPLES_PER_EPOCH > 0:
            save_runtime_check_images(
                runtime_dir=runtime_epoch_dir,
                out_dir=RUNTIME_CHECK_DIR,
                epoch=epoch,
                k=RUNTIME_CHECK_SAMPLES_PER_EPOCH,
                seed=epoch_seed + 7,
            )

        _cleanup_old_runtime_dirs(epoch)

    _ddp_barrier_if_available()

    # 2) Build Ultralytics datasets
    stride = 32
    try:
        s = trainer.model.stride
        stride = int(max(s)) if hasattr(s, "__iter__") else int(s)
    except Exception:
        stride = 32

    ds_static = build_yolo_dataset(
        cfg=trainer.args,
        img_path=str(STATIC_TRAIN_IMG_DIR),
        batch=batch,
        data=trainer.data,
        mode="train",
        rect=False,
        stride=stride,
    )
    ds_dyn = build_yolo_dataset(
        cfg=trainer.args,
        img_path=str(runtime_epoch_dir / "images"),
        batch=batch,
        data=trainer.data,
        mode="train",
        rect=False,
        stride=stride,
    )

    mixed_ds = MixedProxyDataset(ds_static, ds_dyn)
    sampler = MixedBatchSampler(
        n_static=len(ds_static),
        n_dyn=len(ds_dyn),
        batch_size=batch,
        num_batches=num_batches,
        seed=epoch_seed,
    )

    nw = min(os.cpu_count() or 8, int(getattr(trainer.args, "workers", WORKERS)))
    g = torch.Generator()
    g.manual_seed(epoch_seed)

    base_loader = InfiniteDataLoader(
        dataset=mixed_ds,
        batch_sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=mixed_ds.collate_fn,
    )

    # ✅ critical: force len(loader) == num_batches
    loader = LenProxyLoader(base_loader, fixed_len=num_batches)

    trainer.train_loader = loader
    trainer.trainset = mixed_ds

    # also set trainer.nb (some parts use it)
    try:
        trainer.nb = num_batches
    except Exception:
        pass

    if rank == 0:
        print(
            f"[Step3 RouteA-SEG] epoch={epoch} "
            f"static_imgs={static_count} dyn_imgs={dyn_images} "
            f"ds_static={len(ds_static)} ds_dyn={len(ds_dyn)} "
            f"batches={num_batches} batch={batch} (static={bs_static}, runtime={bs_dyn})"
        )
        print(f"[Runtime] using: {runtime_epoch_dir.resolve()}")
        if RUNTIME_CHECK_SAMPLES_PER_EPOCH > 0:
            print(f"[Check] runtime preview saved to: {RUNTIME_CHECK_DIR.resolve()}")


def main():
    from ultralytics import YOLO

    # basic checks
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {DATA_YAML}")
    if not STATIC_TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"static train dir not found: {STATIC_TRAIN_IMG_DIR}")
    if not STATIC_VAL_IMG_DIR.exists():
        raise FileNotFoundError(f"val dir not found: {STATIC_VAL_IMG_DIR}")
    if not BG_DIR.exists():
        raise FileNotFoundError(f"BG_DIR not found: {BG_DIR}")
    if not CARD_DIR.exists():
        raise FileNotFoundError(f"CARD_DIR not found: {CARD_DIR}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = YOLO(MODEL_WEIGHTS)

    # epoch-start callback: rebuild loader
    register_callback(model, "on_train_epoch_start", lambda trainer: build_mixed_train_loader(trainer))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        cache=False,
        rect=False,

        project=ULTRA_PROJECT,
        name=ULTRA_NAME,
        exist_ok=ULTRA_EXIST_OK,

        # ---- Disable augmentations ----
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        close_mosaic=0,

        amp=True,
        pretrained=True,
    )

    print("[OK] Training finished. Check console: 'Logging results to ...'.")


if __name__ == "__main__":
    main()
