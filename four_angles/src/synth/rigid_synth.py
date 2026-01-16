#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rigid-only synth engine (shared by Step1 and Step3)
- ROI-only warpAffine
- Fast expanded-AABB reject
- uint8 masks + INTER_NEAREST + countNonZero overlap
- Windows-friendly multiprocessing (spawn)

YOLO classes (fixed):
0: card
1: corner_tl
2: corner_tr
3: corner_br
4: corner_bl
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

    # ========= Corner box =========
    # corner bbox side length = corner_box_ratio * min(card_w, card_h), clipped to [min,max]
    # (建议：为了提高 corner recall，把角点框做大一些，让框内包含更多名片内部纹理/文字)
    corner_box_ratio: float = 0.30
    corner_box_min: int = 96
    corner_box_max: int = 180

    # 严格保证“顶点在角点框正中心”
    # True: 若角点框会出界，则直接放弃该次摆放（不 clamp），保证语义一致性
    strict_corner_center: bool = True

    # placement attempts
    max_place_trials_per_card: int = 140

    # max retries for one output image (if too strict constraints)
    max_image_retries: int = 40

    # debug visualization
    save_debug: bool = True

    # multiprocessing
    num_workers: int = max(1, (os.cpu_count() or 8) - 1)

    seed: int = 42


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


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Math / geometry
# =========================
def clamp_bbox_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int
) -> Tuple[float, float, float, float]:
    xmin = max(0.0, min(float(w - 1), xmin))
    ymin = max(0.0, min(float(h - 1), ymin))
    xmax = max(0.0, min(float(w - 1), xmax))
    ymax = max(0.0, min(float(h - 1), ymax))
    return xmin, ymin, xmax, ymax


def bbox_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def yolo_line(
    class_id: int, xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Optional[str]:
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"


def corner_square_bbox_centered(
    pt_xy: np.ndarray, side: int, img_w: int, img_h: int, strict: bool
) -> Optional[Tuple[float, float, float, float]]:
    """
    以 pt 为中心的正方形 bbox。
    strict=True：若 bbox 会出界，返回 None（保证 pt 永远是 bbox 的中心，避免 clamp 导致中心偏移）
    strict=False：出界则 clamp（不建议用于“顶点=中心”的语义任务）
    """
    x, y = float(pt_xy[0]), float(pt_xy[1])
    half = side / 2.0
    xmin, ymin, xmax, ymax = x - half, y - half, x + half, y + half

    if strict:
        if xmin < 0 or ymin < 0 or xmax > (img_w - 1) or ymax > (img_h - 1):
            return None
        return xmin, ymin, xmax, ymax

    return clamp_bbox_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)


def affine_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (M @ pts_h.T).T
    return out.astype(np.float32)


def card_corners(Wc: int, Hc: int) -> np.ndarray:
    return np.array([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]], dtype=np.float32)


def corners_within_margin(quad: np.ndarray, img_w: int, img_h: int, margin: int) -> bool:
    if np.any(quad[:, 0] < margin) or np.any(quad[:, 0] > img_w - 1 - margin):
        return False
    if np.any(quad[:, 1] < margin) or np.any(quad[:, 1] > img_h - 1 - margin):
        return False
    return True


def build_affine_rotation_translation(Wc: int, Hc: int, angle_deg: float, center_bg: Tuple[float, float]) -> np.ndarray:
    cx_bg, cy_bg = center_bg
    center_card = (Wc / 2.0, Hc / 2.0)
    M = cv2.getRotationMatrix2D(center_card, angle_deg, 1.0)
    M[0, 2] += cx_bg - center_card[0]
    M[1, 2] += cy_bg - center_card[1]
    return M.astype(np.float32)


def aabb_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return False
    return True


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

    rx0 = max(0, min(img_w, rx0))
    ry0 = max(0, min(img_h, ry0))
    rx1 = max(0, min(img_w, rx1))
    ry1 = max(0, min(img_h, ry1))

    if rx1 - rx0 <= 1 or ry1 - ry0 <= 1:
        return None
    return rx0, ry0, rx1, ry1


# =========================
# Card cache
# =========================
@dataclass
class CardCacheItem:
    bgr: np.ndarray
    mask_u8: np.ndarray
    H: int
    W: int
    corner_side: int


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
    if card_path in cache:
        return cache[card_path]

    raw = imread_any(Path(card_path))
    resized = resize_card_to_fixed_width_keep_alpha(raw, cfg.fixed_card_w)

    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    if resized.shape[2] == 4:
        bgr = resized[:, :, :3].copy()
        alpha = resized[:, :, 3].copy()
        mask_u8 = alpha
    else:
        bgr = resized[:, :, :3].copy()
        mask_u8 = np.full((bgr.shape[0], bgr.shape[1]), 255, dtype=np.uint8)

    Hc, Wc = bgr.shape[:2]

    side = int(round(cfg.corner_box_ratio * min(Wc, Hc)))
    side = max(cfg.corner_box_min, min(cfg.corner_box_max, side))

    item = CardCacheItem(bgr=bgr, mask_u8=mask_u8, H=Hc, W=Wc, corner_side=side)
    cache[card_path] = item
    return item


# =========================
# ROI warp/composite
# =========================
def warp_affine_roi(card_bgr: np.ndarray, card_mask_u8: np.ndarray, M_roi: np.ndarray, roi_w: int, roi_h: int):
    warp_bgr = cv2.warpAffine(
        card_bgr, M_roi, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warp_mask = cv2.warpAffine(
        card_mask_u8, M_roi, (roi_w, roi_h),
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
def draw_debug(
    img_bgr: np.ndarray,
    quads: List[np.ndarray],
    card_bboxes: List[Tuple[float, float, float, float]],
    corner_bboxes: List[List[Tuple[int, Tuple[float, float, float, float]]]],
) -> np.ndarray:
    vis = img_bgr.copy()
    for i, quad in enumerate(quads):
        q = quad.astype(np.int32)
        cv2.polylines(vis, [q], isClosed=True, color=(0, 255, 0), thickness=2)

        xmin, ymin, xmax, ymax = card_bboxes[i]
        cv2.rectangle(vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

        labels = ["TL", "TR", "BR", "BL"]
        for j, (x, y) in enumerate(quad):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(vis, labels[j], (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for class_id, (cxmin, cymin, cxmax, cymax) in corner_bboxes[i]:
            cv2.rectangle(vis, (int(cxmin), int(cymin)), (int(cxmax), int(cymax)), (255, 0, 255), 2)
            cv2.putText(vis, str(class_id), (int(cxmin), int(cymin) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return vis


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
    for retry in range(cfg.max_image_retries):
        rng = random.Random(cfg.seed + idx * 1000003 + retry * 99991)

        bg_path = Path(rng.choice(bg_paths))
        bg_raw = imread_any(bg_path)

        if bg_raw.ndim == 2:
            bg_raw = cv2.cvtColor(bg_raw, cv2.COLOR_GRAY2BGR)
        bg_bgr = bg_raw[:, :, :3] if (bg_raw.ndim == 3 and bg_raw.shape[2] == 4) else bg_raw

        if cfg.out_w is not None and cfg.out_h is not None:
            bg_bgr = cv2.resize(bg_bgr, (cfg.out_w, cfg.out_h), interpolation=cv2.INTER_AREA)

        img_h, img_w = bg_bgr.shape[:2]
        occ = np.zeros((img_h, img_w), dtype=np.uint8)

        num_cards_target = rng.randint(cfg.min_cards, cfg.max_cards)

        placed_boxes_expanded: List[Tuple[float, float, float, float]] = []
        labels: List[str] = []
        quads_dbg: List[np.ndarray] = []
        card_bboxes_dbg: List[Tuple[float, float, float, float]] = []
        corner_bboxes_dbg: List[List[Tuple[int, Tuple[float, float, float, float]]]] = []

        placed = 0
        max_trials = num_cards_target * cfg.max_place_trials_per_card

        pad = int(cfg.min_gap_between_cards)
        kernel = None
        if pad > 0:
            k = pad * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        for _ in range(max_trials):
            if placed >= num_cards_target:
                break

            card_path = rng.choice(card_paths)
            card_item = get_card_cached(card_cache, card_path, cfg)
            Wc, Hc = card_item.W, card_item.H
            side = card_item.corner_side

            # 为了严格保证“顶点=角点框中心”，角点框必须完全落在图内
            # 所以要把“边界安全距离”提升到 max(margin_to_img, side/2)
            safe_margin = cfg.margin_to_img
            if cfg.strict_corner_center:
                safe_margin = max(safe_margin, int(math.ceil(side / 2.0)) + 2)

            radius = 0.5 * math.sqrt(float(Wc * Wc + Hc * Hc))
            x_min = safe_margin + radius
            x_max = img_w - safe_margin - radius
            y_min = safe_margin + radius
            y_max = img_h - safe_margin - radius
            if x_max <= x_min or y_max <= y_min:
                continue

            angle = rng.uniform(cfg.angle_min, cfg.angle_max)
            cx = rng.uniform(x_min, x_max)
            cy = rng.uniform(y_min, y_max)

            M = build_affine_rotation_translation(Wc, Hc, angle, (cx, cy))
            quad = affine_transform_points(M, card_corners(Wc, Hc))  # TL,TR,BR,BL

            if not corners_within_margin(quad, img_w, img_h, safe_margin):
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

            if kernel is not None:
                dilated_roi = cv2.dilate(warp_mask_roi_u8, kernel, iterations=1)
            else:
                dilated_roi = warp_mask_roi_u8

            occ_roi = occ[ry0:ry1, rx0:rx1]
            inter = cv2.bitwise_and(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)
            if cv2.countNonZero(inter) > 0:
                continue

            bg_roi = bg_bgr[ry0:ry1, rx0:rx1]
            composite_roi_inplace(bg_roi, warp_bgr_roi, warp_mask_roi_u8)
            occ_roi[:] = cv2.bitwise_or(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)

            placed_boxes_expanded.append(exp_bbox)

            # card bbox label
            xmin_c, ymin_c, xmax_c, ymax_c = clamp_bbox_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)
            l0 = yolo_line(0, xmin_c, ymin_c, xmax_c, ymax_c, img_w, img_h)
            if l0:
                labels.append(l0)

            # corner labels (strict centered)
            corner_ids = [1, 2, 3, 4]  # TL TR BR BL
            corners_for_debug: List[Tuple[int, Tuple[float, float, float, float]]] = []

            ok_corners = True
            for pt, cid in zip(quad, corner_ids):
                bbox = corner_square_bbox_centered(pt, side, img_w, img_h, strict=cfg.strict_corner_center)
                if bbox is None:
                    ok_corners = False
                    break
                cxmin, cymin, cxmax, cymax = bbox
                l = yolo_line(cid, cxmin, cymin, cxmax, cymax, img_w, img_h)
                if l:
                    labels.append(l)
                    corners_for_debug.append((cid, (cxmin, cymin, cxmax, cymax)))

            # 只要出现任何一个角点框会出界（strict=True），就放弃这次摆放（保证语义一致）
            if cfg.strict_corner_center and (not ok_corners):
                continue

            quads_dbg.append(quad)
            card_bboxes_dbg.append((xmin_c, ymin_c, xmax_c, ymax_c))
            corner_bboxes_dbg.append(corners_for_debug)

            placed += 1

        if placed >= cfg.min_cards:
            imwrite(out_img_path, bg_bgr)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

            if cfg.save_debug and out_dbg_path is not None:
                dbg = draw_debug(bg_bgr, quads_dbg, card_bboxes_dbg, corner_bboxes_dbg)
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
                "Try: reduce fixed_card_w / reduce margin_to_img / reduce min_gap_between_cards / increase out_w,out_h / reduce max_cards."
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
      cfg.out_dir/labels/*.txt
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

    print("[INFO] Synth Dataset")
    print(f"  out_dir  : {cfg.out_dir}")
    print(f"  num      : {cfg.num_images}")
    print(f"  cards/img: {cfg.min_cards}~{cfg.max_cards}")
    print(f"  fixed_w  : {cfg.fixed_card_w}")
    print(f"  margin   : {cfg.margin_to_img}")
    print(f"  min_gap  : {cfg.min_gap_between_cards}")
    print(f"  corner   : ratio={cfg.corner_box_ratio} min={cfg.corner_box_min} max={cfg.corner_box_max} strict_center={cfg.strict_corner_center}")
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
