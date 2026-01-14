#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step1 (Rigid-only, vertex-centered corner boxes):
- Each synthesized image places 2~4 cards.
- Card size is kept consistent by resizing all cards to a fixed width.
- NO perspective / NO shear / NO random scaling variations.
- Only rotation + translation when compositing onto background (warpAffine).
- No overlap (strict), optional min gap via mask dilation.
- Corner detection boxes are SQUARE boxes centered at the 4 corner vertices,
  so that each vertex is exactly at the center of its corner bbox.

Classes (fixed order):
0: card
1: corner_tl
2: corner_tr
3: corner_br
4: corner_bl
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# ✅ CONFIG: edit here
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
    margin_to_img: int = 80

    # strict no overlap. If you want a visible gap, set >0 (pixels).
    min_gap_between_cards: int = 0

    # fixed card width in pixels AFTER resizing (keeps size consistent)
    fixed_card_w: int = 700

    # full rotation (no restriction)
    angle_min: float = 0.0
    angle_max: float = 360.0

    # corner bbox side length = corner_box_ratio * min(card_w, card_h), clipped to [min,max]
    corner_box_ratio: float = 0.18
    corner_box_min: int = 40
    corner_box_max: int = 120

    # placement attempts
    max_place_trials_per_card: int = 140

    # debug visualization
    save_debug: bool = True

    seed: int = 42


CONFIG = SynthConfig(
    bg_dir=Path(r"data/background"),
    card_dir=Path(r"data/business_card_raw"),
    out_dir=Path(r"data/four_angles/synth_step1_rigid_vertexcorner"),
    num_images=10,
    # out_w=1920, out_h=1080,  # 可选：统一背景尺寸；否则保持原尺寸
    margin_to_img=90,          # 建议 >= corner_box_max//2 + 10
    min_gap_between_cards=40,   # 严格不重叠（0即可）
    fixed_card_w=700,          # 所有名片统一宽度（像素）
    save_debug=True,
)


# =========================
# IO helpers (Chinese path OK)
# =========================
def imread_any(path: Path) -> np.ndarray:
    """Read image with cv2 (supports Chinese path). Keep alpha if present."""
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
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


# =========================
# BBox / labels
# =========================
def clamp_bbox_xyxy(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> Tuple[float, float, float, float]:
    xmin = max(0.0, min(float(w - 1), xmin))
    ymin = max(0.0, min(float(h - 1), ymin))
    xmax = max(0.0, min(float(w - 1), xmax))
    ymax = max(0.0, min(float(h - 1), ymax))
    return xmin, ymin, xmax, ymax


def bbox_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def yolo_line(class_id: int, xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Optional[str]:
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"


def corner_square_bbox(pt_xy: np.ndarray, side: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Axis-aligned square bbox centered at pt."""
    x, y = float(pt_xy[0]), float(pt_xy[1])
    half = side / 2.0
    xmin, ymin, xmax, ymax = x - half, y - half, x + half, y + half
    return clamp_bbox_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)


# =========================
# Affine helpers
# =========================
def affine_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine matrix to points (N,2)."""
    pts = pts.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,3)
    out = (M @ pts_h.T).T  # (N,2)
    return out.astype(np.float32)


def card_corners(Wc: int, Hc: int) -> np.ndarray:
    """Return TL,TR,BR,BL in card coordinate system."""
    return np.array([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]], dtype=np.float32)


def corners_within_margin(quad: np.ndarray, img_w: int, img_h: int, margin: int) -> bool:
    if np.any(quad[:, 0] < margin) or np.any(quad[:, 0] > img_w - 1 - margin):
        return False
    if np.any(quad[:, 1] < margin) or np.any(quad[:, 1] > img_h - 1 - margin):
        return False
    return True


def build_affine_rotation_translation(Wc: int, Hc: int, angle_deg: float, center_bg: Tuple[float, float]) -> np.ndarray:
    """
    Build affine matrix M that:
    - rotates card around its own center (Wc/2, Hc/2) by angle_deg
    - translates so that card center maps to center_bg (cx,cy)
    """
    cx_bg, cy_bg = center_bg
    center_card = (Wc / 2.0, Hc / 2.0)
    M = cv2.getRotationMatrix2D(center_card, angle_deg, 1.0)  # 2x3
    M[0, 2] += cx_bg - center_card[0]
    M[1, 2] += cy_bg - center_card[1]
    return M.astype(np.float32)


# =========================
# Card preprocessing
# =========================
def resize_card_to_fixed_width(card: np.ndarray, fixed_w: int) -> np.ndarray:
    """Resize card to fixed width (keep aspect ratio). Keep alpha if present."""
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    h, w = card.shape[:2]
    if w <= 0:
        return card
    if w == fixed_w:
        return card

    scale = fixed_w / float(w)
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(card, (fixed_w, new_h), interpolation=interp)
    return resized


# =========================
# Warp/composite (Rigid only)
# =========================
def warp_card_affine_to_bg(bg_bgr: np.ndarray, card_img: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp card onto background with affine M (2x3).
    Returns (warp_bgr, warp_mask01).
    mask uses alpha if present else full 1.
    """
    h_bg, w_bg = bg_bgr.shape[:2]

    if card_img.shape[2] == 4:
        bgr = card_img[:, :, :3]
        alpha = card_img[:, :, 3].astype(np.float32) / 255.0
        mask = alpha
    else:
        bgr = card_img
        mask = np.ones((card_img.shape[0], card_img.shape[1]), dtype=np.float32)

    warp_bgr = cv2.warpAffine(
        bgr, M, (w_bg, h_bg),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warp_mask = cv2.warpAffine(
        mask, M, (w_bg, h_bg),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warp_mask = np.clip(warp_mask, 0.0, 1.0)
    return warp_bgr, warp_mask


def composite(bg_bgr: np.ndarray, fg_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    mask3 = mask01[:, :, None]
    out = bg_bgr.astype(np.float32) * (1.0 - mask3) + fg_bgr.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# Debug draw
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
# Synthesis
# =========================
def synth_one(
    cfg: SynthConfig,
    bg_path: Path,
    card_paths: List[Path],
    card_cache: Dict[Path, np.ndarray],
    out_img_path: Path,
    out_lbl_path: Path,
    out_dbg_path: Optional[Path],
) -> bool:
    rng = random.Random(cfg.seed + hash(out_img_path.stem) % (10**9))

    bg_raw = imread_any(bg_path)
    if bg_raw.ndim == 2:
        bg_raw = cv2.cvtColor(bg_raw, cv2.COLOR_GRAY2BGR)
    bg_bgr = bg_raw[:, :, :3] if (bg_raw.ndim == 3 and bg_raw.shape[2] == 4) else bg_raw

    if cfg.out_w is not None and cfg.out_h is not None:
        bg_bgr = cv2.resize(bg_bgr, (cfg.out_w, cfg.out_h), interpolation=cv2.INTER_AREA)

    img_h, img_w = bg_bgr.shape[:2]

    # occupancy mask for overlap control
    occ = np.zeros((img_h, img_w), dtype=np.uint8)

    num_cards = rng.randint(cfg.min_cards, cfg.max_cards)

    labels: List[str] = []
    all_quads: List[np.ndarray] = []
    all_card_bboxes: List[Tuple[float, float, float, float]] = []
    all_corner_bboxes: List[List[Tuple[int, Tuple[float, float, float, float]]]] = []

    placed = 0
    max_global_trials = num_cards * cfg.max_place_trials_per_card

    for _ in range(max_global_trials):
        if placed >= num_cards:
            break

        card_path = rng.choice(card_paths)

        # cache resized to fixed width
        if card_path not in card_cache:
            raw = imread_any(card_path)
            card_cache[card_path] = resize_card_to_fixed_width(raw, cfg.fixed_card_w)
        card_img = card_cache[card_path]

        if card_img.ndim != 3:
            continue

        Hc, Wc = card_img.shape[:2]
        if Wc < 10 or Hc < 10:
            continue

        # Corner box side length based on card size
        side = int(round(cfg.corner_box_ratio * min(Wc, Hc)))
        side = max(cfg.corner_box_min, min(cfg.corner_box_max, side))

        # Safety: ensure margin is enough for vertex-centered corner boxes
        if cfg.margin_to_img < (side // 2 + 5):
            # not fatal, but would be frequently clamped; better to raise
            pass

        angle = rng.uniform(cfg.angle_min, cfg.angle_max)

        # sample a center; we then validate rotated corners inside margin
        cx = rng.uniform(cfg.margin_to_img + Wc / 2.0, img_w - cfg.margin_to_img - Wc / 2.0)
        cy = rng.uniform(cfg.margin_to_img + Hc / 2.0, img_h - cfg.margin_to_img - Hc / 2.0)

        M = build_affine_rotation_translation(Wc, Hc, angle, (cx, cy))

        # quad in bg coords (TL,TR,BR,BL)
        quad = affine_transform_points(M, card_corners(Wc, Hc))

        # enforce border margin on rotated corners
        if not corners_within_margin(quad, img_w, img_h, cfg.margin_to_img):
            continue

        # warp mask for collision test
        warp_bgr, warp_mask01 = warp_card_affine_to_bg(bg_bgr, card_img, M)
        warp_mask_u8 = (warp_mask01 > 0.5).astype(np.uint8) * 255

        # strict no overlap (+ optional gap)
        if cfg.min_gap_between_cards > 0:
            k = cfg.min_gap_between_cards * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(warp_mask_u8, kernel, iterations=1)
        else:
            dilated = warp_mask_u8

        overlap = cv2.bitwise_and(occ, (dilated > 0).astype(np.uint8))
        if int(overlap.sum()) > 0:
            continue

        # composite
        bg_bgr = composite(bg_bgr, warp_bgr, warp_mask01)
        occ = cv2.bitwise_or(occ, (dilated > 0).astype(np.uint8))

        # 1) card bbox (axis-aligned)
        xmin, ymin, xmax, ymax = bbox_from_points(quad)
        xmin, ymin, xmax, ymax = clamp_bbox_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)
        line = yolo_line(0, xmin, ymin, xmax, ymax, img_w, img_h)
        if line is None:
            continue
        labels.append(line)

        # 2) corner bboxes centered at vertices
        corner_boxes_for_debug: List[Tuple[int, Tuple[float, float, float, float]]] = []
        corner_class_ids = [1, 2, 3, 4]  # TL,TR,BR,BL

        for pt, cid in zip(quad, corner_class_ids):
            cxmin, cymin, cxmax, cymax = corner_square_bbox(pt, side, img_w, img_h)
            c_line = yolo_line(cid, cxmin, cymin, cxmax, cymax, img_w, img_h)
            if c_line is not None:
                labels.append(c_line)
                corner_boxes_for_debug.append((cid, (cxmin, cymin, cxmax, cymax)))

        all_quads.append(quad)
        all_card_bboxes.append((xmin, ymin, xmax, ymax))
        all_corner_bboxes.append(corner_boxes_for_debug)

        placed += 1

    if placed < cfg.min_cards:
        return False

    # write outputs
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_img_path, bg_bgr)
    out_lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    if cfg.save_debug and out_dbg_path is not None:
        out_dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg = draw_debug(bg_bgr, all_quads, all_card_bboxes, all_corner_bboxes)
        imwrite(out_dbg_path, dbg)

    return True


def main(cfg: SynthConfig) -> None:
    bg_paths = list_images(cfg.bg_dir)
    card_paths = list_images(cfg.card_dir)
    if not bg_paths:
        raise FileNotFoundError(f"No background images found in: {cfg.bg_dir}")
    if not card_paths:
        raise FileNotFoundError(f"No card images found in: {cfg.card_dir}")

    out_img_dir = cfg.out_dir / "images"
    out_lbl_dir = cfg.out_dir / "labels"
    out_dbg_dir = cfg.out_dir / "debug_vis"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_debug:
        out_dbg_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    card_cache: Dict[Path, np.ndarray] = {}

    made = 0
    tries = 0
    while made < cfg.num_images:
        bg_path = random.choice(bg_paths)
        name = f"synth_{made:06d}"
        out_img_path = out_img_dir / f"{name}.jpg"
        out_lbl_path = out_lbl_dir / f"{name}.txt"
        out_dbg_path = out_dbg_dir / f"{name}.jpg" if cfg.save_debug else None

        ok = synth_one(cfg, bg_path, card_paths, card_cache, out_img_path, out_lbl_path, out_dbg_path)
        tries += 1
        if ok:
            made += 1

        # safety if constraints make placement impossible
        if tries > cfg.num_images * 80 and made == 0:
            raise RuntimeError(
                "Failed too many times. Likely constraints are too strict for your backgrounds.\n"
                "Try: reduce fixed_card_w / reduce margin_to_img / increase background size / reduce min_cards."
            )

    print(f"[OK] Generated {made} images into: {cfg.out_dir}")
    print(f"  images: {out_img_dir}")
    print(f"  labels: {out_lbl_dir}")
    if cfg.save_debug:
        print(f"  debug : {out_dbg_dir}")
    print(f"[INFO] fixed_card_w={cfg.fixed_card_w}, margin_to_img={cfg.margin_to_img}, min_gap={cfg.min_gap_between_cards}")
    print(f"[INFO] corner_box_ratio={cfg.corner_box_ratio}, corner_box_min={cfg.corner_box_min}, corner_box_max={cfg.corner_box_max}")


if __name__ == "__main__":
    main(CONFIG)