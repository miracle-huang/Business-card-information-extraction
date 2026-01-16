#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rigid-only segmentation synth engine (Step1)
- NO augmentation
- NO perspective / distortion (warpAffine only)
- card size kept consistent (fixed_card_w), rotation only
- NO overlap (with optional min gap)
- background size NOT fixed: keep original unless target is 3/4, then enlarge background dynamically
- Windows-friendly multiprocessing (spawn)

YOLO-seg label format (polygon):
  cls x1 y1 x2 y2 x3 y3 x4 y4   (normalized, 4-corner polygon)
"""

from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import multiprocessing as mp

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# Config
# =========================
@dataclass
class SynthConfig:
    bg_dir: Path
    card_dir: Path
    out_dir: Path

    num_images: int = 500

    # output size: if None -> keep background original size
    out_w: Optional[int] = None
    out_h: Optional[int] = None

    # cards per image
    min_cards: int = 2
    max_cards: int = 4

    # distance to image border (pixels)
    margin_to_img: int = 90

    # strict no overlap. If you want a visible gap, set >0 (pixels).
    min_gap_between_cards: int = 40

    # fixed card width in pixels AFTER resizing (keeps size consistent)
    fixed_card_w: int = 700

    # full rotation range
    angle_min: float = 0.0
    angle_max: float = 360.0

    # placement attempts
    max_place_trials_per_card: int = 160
    max_image_retries: int = 60

    # debug visualization
    save_debug: bool = True

    # multiprocessing
    num_workers: int = max(1, (os.cpu_count() or 8) - 1)

    seed: int = 42

    # ---------- New: target count weighting (increase 3/4 ratio) ----------
    # for {2,3,4} when min_cards=2 max_cards=4
    weight_2: float = 1.0
    weight_3: float = 3.0
    weight_4: float = 4.0

    # ---------- New: dynamic background enlarge for 3/4 cards ----------
    dynamic_bg_enlarge: bool = True
    dynamic_bg_only_for_3plus: bool = True
    max_bg_scale: float = 3.0  # cap to avoid too huge images; set larger if you want


# =========================
# IO helpers (Chinese path OK)
# =========================
def imread_any(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def ensure_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Geometry helpers
# =========================
def bbox_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def compute_roi_from_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad: int,
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_xyxy
    rx0 = int(math.floor(x0 - pad))
    ry0 = int(math.floor(y0 - pad))
    rx1 = int(math.ceil(x1 + pad))
    ry1 = int(math.ceil(y1 + pad))

    # clamp for slicing [ry0:ry1, rx0:rx1], so upper bound can be img_w/img_h
    rx0 = max(0, min(img_w, rx0))
    ry0 = max(0, min(img_h, ry0))
    rx1 = max(0, min(img_w, rx1))
    ry1 = max(0, min(img_h, ry1))

    if rx1 - rx0 <= 1 or ry1 - ry0 <= 1:
        return None
    return rx0, ry0, rx1, ry1


def aabb_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return False
    return True


def affine_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (M @ pts_h.T).T
    return out.astype(np.float32)


def card_corners(Wc: int, Hc: int) -> np.ndarray:
    """
    TL, TR, BR, BL corners in pixel coordinates.
    Use W-1/H-1 to avoid off-by-one that makes bounds tighter than necessary.
    """
    w1 = float(max(0, Wc - 1))
    h1 = float(max(0, Hc - 1))
    return np.array([[0.0, 0.0], [w1, 0.0], [w1, h1], [0.0, h1]], dtype=np.float32)


def corners_within_margin(quad: np.ndarray, img_w: int, img_h: int, margin: int) -> bool:
    if np.any(quad[:, 0] < margin) or np.any(quad[:, 0] > img_w - 1 - margin):
        return False
    if np.any(quad[:, 1] < margin) or np.any(quad[:, 1] > img_h - 1 - margin):
        return False
    return True


def build_affine_rotation_translation(Wc: int, Hc: int, angle_deg: float, center_bg: Tuple[float, float]) -> np.ndarray:
    """
    Rotation around card center (scale=1.0), then translate to background center (cx_bg, cy_bg).
    """
    cx_bg, cy_bg = center_bg
    center_card = ((Wc - 1) / 2.0, (Hc - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center_card, angle_deg, 1.0)  # scale=1.0 (no resize aug)
    M[0, 2] += cx_bg - center_card[0]
    M[1, 2] += cy_bg - center_card[1]
    return M.astype(np.float32)


def yolo_seg_line(class_id: int, quad: np.ndarray, img_w: int, img_h: int) -> Optional[str]:
    """
    YOLO-seg label line:
      cls x1 y1 x2 y2 x3 y3 x4 y4  (normalized)
    """
    q = quad.astype(np.float32).copy()
    q[:, 0] = np.clip(q[:, 0], 0, img_w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, img_h - 1)

    xmin, ymin, xmax, ymax = bbox_from_points(q)
    if (xmax - xmin) <= 2.0 or (ymax - ymin) <= 2.0:
        return None

    coords = []
    for x, y in q:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(coords)


# =========================
# Card cache
# =========================
@dataclass
class CardCacheItem:
    bgr: np.ndarray
    mask_u8: np.ndarray  # 0~255
    H: int
    W: int
    diag: float


def resize_card_to_fixed_width_keep_alpha(card: np.ndarray, fixed_w: int) -> np.ndarray:
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    h, w = card.shape[:2]
    if w <= 0 or w == fixed_w:
        return card

    scale = fixed_w / float(w)
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(card, (fixed_w, new_h), interpolation=interp)
    return resized


def get_card_cached(cache: Dict[str, CardCacheItem], card_path: str, cfg: SynthConfig) -> CardCacheItem:
    """
    Cache stores fixed-size cards (fixed_card_w). We do NOT downscale cards anymore
    because we will enlarge background dynamically for 3/4 cards.
    """
    if card_path in cache:
        return cache[card_path]

    raw = imread_any(Path(card_path))
    resized = resize_card_to_fixed_width_keep_alpha(raw, cfg.fixed_card_w)

    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    if resized.ndim == 3 and resized.shape[2] == 4:
        bgr = resized[:, :, :3].copy()
        alpha = resized[:, :, 3].copy()
        mask_u8 = alpha
    else:
        bgr = resized[:, :, :3].copy()
        mask_u8 = np.full((bgr.shape[0], bgr.shape[1]), 255, dtype=np.uint8)

    Hc, Wc = bgr.shape[:2]
    diag = float(math.sqrt(Wc * Wc + Hc * Hc))
    item = CardCacheItem(bgr=bgr, mask_u8=mask_u8, H=Hc, W=Wc, diag=diag)
    cache[card_path] = item
    return item


# =========================
# ROI warp/composite
# =========================
def warp_affine_roi(card_bgr: np.ndarray, card_mask_u8: np.ndarray, M_roi: np.ndarray, roi_w: int, roi_h: int):
    warp_bgr = cv2.warpAffine(
        card_bgr,
        M_roi,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warp_mask = cv2.warpAffine(
        card_mask_u8,
        M_roi,
        (roi_w, roi_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warp_bgr, warp_mask


def composite_roi_inplace(bg_roi: np.ndarray, fg_roi: np.ndarray, mask_u8: np.ndarray) -> None:
    alpha = (mask_u8.astype(np.float32) / 255.0)[:, :, None]
    out = bg_roi.astype(np.float32) * (1.0 - alpha) + fg_roi.astype(np.float32) * alpha
    bg_roi[:] = np.clip(out, 0, 255).astype(np.uint8)


# =========================
# Debug
# =========================
def draw_debug(img_bgr: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    vis = img_bgr.copy()
    for quad in quads:
        q = quad.astype(np.int32)
        cv2.polylines(vis, [q], isClosed=True, color=(0, 255, 0), thickness=2)
        for (x, y) in q:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
    return vis


# =========================
# New: choose target number with weights
# =========================
def sample_num_cards(rng: random.Random, cfg: SynthConfig) -> int:
    choices = list(range(cfg.min_cards, cfg.max_cards + 1))
    weights = []
    for c in choices:
        if c == 2:
            weights.append(float(cfg.weight_2))
        elif c == 3:
            weights.append(float(cfg.weight_3))
        elif c == 4:
            weights.append(float(cfg.weight_4))
        else:
            weights.append(1.0)
    return rng.choices(choices, weights=weights, k=1)[0]


# =========================
# New: enlarge background if needed for 3/4 cards
# =========================
def required_canvas_for_target(
    max_diag: float,
    target_n: int,
    margin: int,
    gap: int,
) -> Tuple[float, float]:
    """
    Conservative feasibility estimate using "circle packing" approximation.
    Let each card have radius r = max_diag/2 (covers any rotation).

    centers spacing >= 2r + gap
    boundary constraint: center at least (margin + r) away from borders

    For target 2: layout 1x2 (rows=1, cols=2)
    For target 3/4: layout 2x2 (rows=2, cols=2)  (conservative)
    """
    r = max_diag / 2.0
    if target_n <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2

    req_w = 2.0 * (margin + r) + (cols - 1) * (2.0 * r + gap)
    req_h = 2.0 * (margin + r) + (rows - 1) * (2.0 * r + gap)
    return req_w, req_h


def maybe_enlarge_background(
    bg_bgr: np.ndarray,
    cfg: SynthConfig,
    target_n: int,
    chosen_cards: List[CardCacheItem],
) -> np.ndarray:
    if not cfg.dynamic_bg_enlarge:
        return bg_bgr
    if cfg.dynamic_bg_only_for_3plus and target_n < 3:
        return bg_bgr
    if not chosen_cards:
        return bg_bgr

    h, w = bg_bgr.shape[:2]
    max_diag = max(ci.diag for ci in chosen_cards)

    req_w, req_h = required_canvas_for_target(
        max_diag=max_diag,
        target_n=target_n,
        margin=int(cfg.margin_to_img),
        gap=int(cfg.min_gap_between_cards),
    )

    s = max(req_w / float(w), req_h / float(h), 1.0)
    s = min(s, float(cfg.max_bg_scale))
    if s <= 1.0 + 1e-6:
        return bg_bgr

    new_w = int(round(w * s))
    new_h = int(round(h * s))
    # upscale => INTER_LINEAR
    bg_big = cv2.resize(bg_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return bg_big


# =========================
# Single image synthesis
# =========================
def synth_one_image(
    cfg: SynthConfig,
    bg_paths: List[str],
    card_paths: List[str],
    idx: int,
    out_img_path: Path,
    out_lbl_path: Path,
    out_dbg_path: Optional[Path],
    card_cache: Dict[str, CardCacheItem],
) -> bool:
    """
    For each image, we:
      1) sample target number (weighted, prefer 3/4)
      2) pre-select card templates
      3) load background (keep original size unless cfg.out_w/out_h set)
      4) if target is 3/4, enlarge background dynamically if needed
      5) place cards sequentially (each card has its own placement trials)
    """
    for retry in range(cfg.max_image_retries):
        rng = random.Random(cfg.seed + idx * 1000003 + retry * 99991)

        # ---- choose target & cards for this image (preselect) ----
        target_n = sample_num_cards(rng, cfg)
        if len(card_paths) >= target_n:
            chosen_paths = [rng.choice(card_paths) for _ in range(target_n)]
        else:
            chosen_paths = [rng.choice(card_paths) for _ in range(target_n)]

        chosen_items: List[CardCacheItem] = [get_card_cached(card_cache, p, cfg) for p in chosen_paths]

        # ---- background ----
        bg_path = Path(rng.choice(bg_paths))
        bg_raw = imread_any(bg_path)

        if bg_raw.ndim == 2:
            bg_raw = cv2.cvtColor(bg_raw, cv2.COLOR_GRAY2BGR)
        bg_bgr = bg_raw[:, :, :3] if (bg_raw.ndim == 3 and bg_raw.shape[2] == 4) else bg_raw

        # optional fixed output size (you can set out_w/out_h=None to NOT fix)
        if cfg.out_w is not None and cfg.out_h is not None:
            bg_bgr = cv2.resize(bg_bgr, (int(cfg.out_w), int(cfg.out_h)), interpolation=cv2.INTER_AREA)

        # dynamic enlarge for 3/4 cards (when out_w/out_h is None, this makes size adaptive)
        bg_bgr = maybe_enlarge_background(bg_bgr, cfg, target_n=target_n, chosen_cards=chosen_items)

        img_h, img_w = bg_bgr.shape[:2]
        occ = np.zeros((img_h, img_w), dtype=np.uint8)

        pad = int(cfg.min_gap_between_cards)
        kernel = None
        if pad > 0:
            k = pad * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        placed_boxes_expanded: List[Tuple[float, float, float, float]] = []
        labels: List[str] = []
        quads_dbg: List[np.ndarray] = []

        # ---- place sequentially, so target distribution is controlled ----
        placed = 0
        for card_item in chosen_items:
            Wc, Hc = card_item.W, card_item.H
            if Wc < 10 or Hc < 10:
                continue

            # quick feasibility for single card under this canvas
            radius = 0.5 * math.sqrt(float(Wc * Wc + Hc * Hc))
            x_min = cfg.margin_to_img + radius
            x_max = img_w - cfg.margin_to_img - radius
            y_min = cfg.margin_to_img + radius
            y_max = img_h - cfg.margin_to_img - radius
            if x_max <= x_min or y_max <= y_min:
                # this background too small even after enlarge
                placed = -999
                break

            success_this = False
            for _try in range(cfg.max_place_trials_per_card):
                angle = rng.uniform(cfg.angle_min, cfg.angle_max)
                cx = rng.uniform(x_min, x_max)
                cy = rng.uniform(y_min, y_max)

                M = build_affine_rotation_translation(Wc, Hc, angle, (cx, cy))
                quad = affine_transform_points(M, card_corners(Wc, Hc))  # TL,TR,BR,BL

                if not corners_within_margin(quad, img_w, img_h, cfg.margin_to_img):
                    continue

                xmin, ymin, xmax, ymax = bbox_from_points(quad)
                exp_bbox = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

                hit = False
                for b in placed_boxes_expanded:
                    if aabb_intersects(exp_bbox, b):
                        hit = True
                        break
                if hit:
                    continue

                roi = compute_roi_from_bbox((xmin, ymin, xmax, ymax), img_w, img_h, pad=pad)
                if roi is None:
                    continue
                rx0, ry0, rx1, ry1 = roi
                roi_w = rx1 - rx0
                roi_h = ry1 - ry0

                M_roi = M.copy()
                M_roi[0, 2] -= rx0
                M_roi[1, 2] -= ry0

                warp_bgr_roi, warp_mask_roi_u8 = warp_affine_roi(
                    card_item.bgr, card_item.mask_u8, M_roi, roi_w=roi_w, roi_h=roi_h
                )

                # enforce min gap using dilated new mask vs occupancy
                if kernel is not None:
                    dilated_roi = cv2.dilate(warp_mask_roi_u8, kernel, iterations=1)
                else:
                    dilated_roi = warp_mask_roi_u8

                occ_roi = occ[ry0:ry1, rx0:rx1]
                inter = cv2.bitwise_and(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)
                if cv2.countNonZero(inter) > 0:
                    continue

                # composite (NO color aug / NO blur / NO noise)
                bg_roi = bg_bgr[ry0:ry1, rx0:rx1]
                composite_roi_inplace(bg_roi, warp_bgr_roi, warp_mask_roi_u8)
                occ_roi[:] = cv2.bitwise_or(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)

                placed_boxes_expanded.append(exp_bbox)

                l = yolo_seg_line(0, quad, img_w, img_h)
                if l:
                    labels.append(l)

                quads_dbg.append(quad)
                placed += 1
                success_this = True
                break

            if not success_this:
                # couldn't place this card -> retry whole image
                placed = -999
                break

        if placed >= cfg.min_cards:
            imwrite(out_img_path, bg_bgr)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

            if cfg.save_debug and out_dbg_path is not None:
                dbg = draw_debug(bg_bgr, quads_dbg)
                imwrite(out_dbg_path, dbg)

            return True

    return False


def worker_run(worker_id: int, indices: List[int], cfg: SynthConfig, bg_paths: List[str], card_paths: List[str]) -> None:
    out_img_dir = cfg.out_dir / "images"
    out_lbl_dir = cfg.out_dir / "labels"
    out_dbg_dir = cfg.out_dir / "debug_vis"

    card_cache: Dict[str, CardCacheItem] = {}

    for k, idx in enumerate(indices):
        name = f"synth_{idx:06d}"
        out_img_path = out_img_dir / f"{name}.jpg"
        out_lbl_path = out_lbl_dir / f"{name}.txt"
        out_dbg_path = (out_dbg_dir / f"{name}.jpg") if cfg.save_debug else None

        ok = synth_one_image(
            cfg=cfg,
            bg_paths=bg_paths,
            card_paths=card_paths,
            idx=idx,
            out_img_path=out_img_path,
            out_lbl_path=out_lbl_path,
            out_dbg_path=out_dbg_path,
            card_cache=card_cache,
        )
        if not ok:
            raise RuntimeError(
                f"[Worker {worker_id}] Failed idx={idx} after {cfg.max_image_retries} retries.\n"
                f"cfg: out={cfg.out_w}x{cfg.out_h}, fixed_w={cfg.fixed_card_w}, "
                f"margin={cfg.margin_to_img}, gap={cfg.min_gap_between_cards}, "
                f"cards/img={cfg.min_cards}~{cfg.max_cards}, weights(2,3,4)=({cfg.weight_2},{cfg.weight_3},{cfg.weight_4}), "
                f"dyn_bg={cfg.dynamic_bg_enlarge}, max_bg_scale={cfg.max_bg_scale}\n"
                "Tip: if you insist on large cards AND strict gap/margin, increase max_bg_scale."
            )

        if (k + 1) % 20 == 0:
            print(f"[Worker {worker_id}] done {k+1}/{len(indices)}", flush=True)


def split_indices(n: int, num_workers: int) -> List[List[int]]:
    buckets = [[] for _ in range(num_workers)]
    for i in range(n):
        buckets[i % num_workers].append(i)
    return [b for b in buckets if b]


def generate_dataset(cfg: SynthConfig, overwrite: bool = True) -> None:
    """
    Create:
      cfg.out_dir/images/*.jpg
      cfg.out_dir/labels/*.txt   (YOLO-seg polygon labels)
      cfg.out_dir/debug_vis/*.jpg  (optional)
    """
    bg_paths = [str(p) for p in list_images(cfg.bg_dir)]
    card_paths = [str(p) for p in list_images(cfg.card_dir)]
    if not bg_paths:
        raise FileNotFoundError(f"No background images found in: {cfg.bg_dir}")
    if not card_paths:
        raise FileNotFoundError(f"No card images found in: {cfg.card_dir}")

    if overwrite:
        ensure_clean_dir(cfg.out_dir)
    (cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "labels").mkdir(parents=True, exist_ok=True)
    if cfg.save_debug:
        (cfg.out_dir / "debug_vis").mkdir(parents=True, exist_ok=True)

    print("[INFO] Synth Seg Dataset (Rigid / Rotation-only)")
    print(f"  out_dir  : {cfg.out_dir}")
    print(f"  num      : {cfg.num_images}")
    print(f"  cards/img: {cfg.min_cards}~{cfg.max_cards}")
    print(f"  weights  : 2->{cfg.weight_2}, 3->{cfg.weight_3}, 4->{cfg.weight_4}")
    print(f"  fixed_w  : {cfg.fixed_card_w}")
    print(f"  margin   : {cfg.margin_to_img}")
    print(f"  min_gap  : {cfg.min_gap_between_cards}")
    print(f"  angle    : {cfg.angle_min}~{cfg.angle_max}")
    print(f"  out_size : {cfg.out_w} x {cfg.out_h} (None means keep bg size)")
    print(f"  dyn_bg   : {cfg.dynamic_bg_enlarge}, only_3plus={cfg.dynamic_bg_only_for_3plus}, max_scale={cfg.max_bg_scale}")
    print(f"  workers  : {cfg.num_workers}")
    print(f"  debug    : {cfg.save_debug}")

    if cfg.num_workers <= 1:
        worker_run(
            worker_id=0,
            indices=list(range(cfg.num_images)),
            cfg=cfg,
            bg_paths=bg_paths,
            card_paths=card_paths,
        )
        print("[OK] Done (single process).")
        return

    ctx = mp.get_context("spawn")
    buckets = split_indices(cfg.num_images, cfg.num_workers)
    args_list = [(wid, indices, cfg, bg_paths, card_paths) for wid, indices in enumerate(buckets)]

    with ctx.Pool(processes=len(buckets)) as pool:
        pool.starmap(worker_run, args_list)

    print("[OK] Done (multiprocessing).")
