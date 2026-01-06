from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# IO helpers (Chinese path OK)
# =========================
def imread_any(path: Path) -> np.ndarray:
    """
    Read image with cv2.imdecode to support Chinese paths.
    Keep alpha channel if present (PNG).
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
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
        raise FileNotFoundError(folder)
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


# =========================
# Geometry helpers
# =========================
def poly_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    Compute intersection area of two convex polygons (4-point quads expected).
    Return 0 if no intersection.

    Note: cv2.intersectConvexConvex expects convex polygons.
    Our quads are rectangles transformed by affine, always convex.
    """
    p1 = poly1.astype(np.float32)
    p2 = poly2.astype(np.float32)
    area, _ = cv2.intersectConvexConvex(p1, p2)
    if area is None:
        return 0.0
    return float(area)


def points_inside_image(pts: np.ndarray, W: int, H: int, margin: int = 0) -> bool:
    x = pts[:, 0]
    y = pts[:, 1]
    return (x.min() >= margin) and (y.min() >= margin) and (x.max() <= W - 1 - margin) and (y.max() <= H - 1 - margin)


def affine_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    M: (2,3)
    pts: (N,2)
    return: (N,2)
    """
    pts_h = np.hstack([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)])
    out = (M.astype(np.float32) @ pts_h.T).T
    return out


def compute_bbox_from_pts(pts: np.ndarray, W: int, H: int) -> Tuple[float, float, float, float]:
    """
    pts in pixel coords, return YOLO normalized bbox (cx,cy,w,h).
    """
    xmin = float(np.min(pts[:, 0]))
    xmax = float(np.max(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    ymax = float(np.max(pts[:, 1]))

    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    bw = (xmax - xmin) / W
    bh = (ymax - ymin) / H
    return cx, cy, bw, bh


def normalize_points(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    out = pts.astype(np.float32).copy()
    out[:, 0] /= float(W)
    out[:, 1] /= float(H)
    return out


# =========================
# Config
# =========================
@dataclass
class SynthKptConfig:
    bg_dir: Path = Path(r"data\background")
    card_dir: Path = Path(r"data\business_card_raw")
    out_dir: Path = Path(r"data\synth_kpt_pool")

    num_images: int = 200
    out_w: int = 1536
    out_h: int = 1536

    min_cards: int = 2
    max_cards: int = 4

    # rotation range (degrees)
    angle_min: float = 0.0
    angle_max: float = 360.0

    # 名片大小范围（相对 min(out_w,out_h) 的比例）
    # 为了“同图更一致 + 更可控”，这里默认范围比之前更收敛一些
    w_frac_2: Tuple[float, float] = (0.50, 0.58)
    w_frac_3: Tuple[float, float] = (0.42, 0.50)
    w_frac_4: Tuple[float, float] = (0.34, 0.42)

    # ✅ 同一张图内名片大小一致性：围绕 base_target_w 做轻微抖动（±8%）
    same_image_size_jitter: float = 0.03

    # Placement constraints
    border_margin: int = 8           # keep cards fully inside image with this margin
    max_tries_per_card: int = 800    # rejection sampling tries per card
    max_restarts_per_image: int = 80 # if can't place all cards, restart composition

    # Save debug visualization (optional)
    save_viz: bool = False


# =========================
# Core generator
# =========================
def _choose_target_width(cfg: SynthKptConfig, n_cards: int, min_dim: int, rng: random.Random) -> int:
    if n_cards <= 2:
        lo, hi = cfg.w_frac_2
    elif n_cards == 3:
        lo, hi = cfg.w_frac_3
    else:
        lo, hi = cfg.w_frac_4
    return int(rng.uniform(lo, hi) * float(min_dim))


def _resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0:
        return img
    scale = target_w / float(w)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _prepare_card_and_alpha(card: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return card_bgr, alpha (0..255) or None.
    """
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    if card.shape[2] == 4:
        bgr = card[:, :, :3].copy()
        a = card[:, :, 3].copy()
        return bgr, a
    else:
        return card[:, :, :3].copy(), None


def _fit_background_to_size(bg_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Center-crop after resize to cover, keeping aspect.
    """
    h, w = bg_bgr.shape[:2]
    if (w, h) == (out_w, out_h):
        return bg_bgr

    scale = max(out_w / float(w), out_h / float(h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(bg_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)

    x0 = max(0, (nw - out_w) // 2)
    y0 = max(0, (nh - out_h) // 2)
    return resized[y0:y0 + out_h, x0:x0 + out_w].copy()


def _draw_viz(img: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    """
    Draw quad polygon and keypoint index:
      0: TL, 1: TR, 2: BR, 3: BL  (semantic)
    """
    out = img.copy()
    for q in quads:
        q_int = q.round().astype(int)
        cv2.polylines(out, [q_int], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, (x, y) in enumerate(q_int):
            cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(out, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return out


def _compose_one(
    bg_bgr: np.ndarray,
    card_paths: List[Path],
    cfg: SynthKptConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Returns:
      composed_bgr,
      label_lines (one per card),
      quads_px (for viz)
    """
    H, W = bg_bgr.shape[:2]
    min_dim = min(W, H)

    n_cards = rng.randint(cfg.min_cards, cfg.max_cards)

    # sample card images
    if len(card_paths) >= n_cards:
        chosen = rng.sample(card_paths, k=n_cards)
    else:
        chosen = [rng.choice(card_paths) for _ in range(n_cards)]

    # ✅ 同一张图只采样一次基准宽度
    base_target_w = _choose_target_width(cfg, n_cards, min_dim, rng)
    jitter = max(0.0, float(cfg.same_image_size_jitter))

    placed_quads: List[np.ndarray] = []
    label_lines: List[str] = []
    canvas = bg_bgr.copy()

    for card_path in chosen:
        raw = imread_any(card_path)
        card_bgr, alpha = _prepare_card_and_alpha(raw)

        # ✅ 同图同尺度：围绕 base_target_w 做小幅抖动
        target_w = int(round(base_target_w * rng.uniform(1.0 - jitter, 1.0 + jitter)))
        target_w = max(40, min(target_w, min_dim))  # safety clamp

        card_bgr = _resize_keep_aspect(card_bgr, target_w)
        if alpha is not None:
            alpha = _resize_keep_aspect(alpha, target_w)

        ch, cw = card_bgr.shape[:2]
        if cw < 2 or ch < 2:
            continue

        # semantic corners in card coordinate (TL, TR, BR, BL)
        corners = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
        center = (cw / 2.0, ch / 2.0)

        placed = False
        for _ in range(cfg.max_tries_per_card):
            angle = rng.uniform(cfg.angle_min, cfg.angle_max)

            # build affine transform: rotate around card center, then translate to random center on background
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # choose destination center
            dst_cx = rng.uniform(cfg.border_margin, W - 1 - cfg.border_margin)
            dst_cy = rng.uniform(cfg.border_margin, H - 1 - cfg.border_margin)

            # translate so that card center maps to (dst_cx, dst_cy)
            M[0, 2] += (dst_cx - center[0])
            M[1, 2] += (dst_cy - center[1])

            quad = affine_transform_points(M, corners)  # (4,2) in px

            # must be fully inside image
            if not points_inside_image(quad, W, H, margin=cfg.border_margin):
                continue

            # check overlap with already placed cards
            overlap = False
            for q2 in placed_quads:
                inter = poly_intersection_area(quad, q2)
                if inter > 1.0:  # epsilon area
                    overlap = True
                    break
            if overlap:
                continue

            # warp card to canvas size
            warped = cv2.warpAffine(
                card_bgr, M, (W, H),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            if alpha is not None:
                warped_a = cv2.warpAffine(
                    alpha, M, (W, H),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                a = (warped_a.astype(np.float32) / 255.0)[:, :, None]
                canvas = (canvas.astype(np.float32) * (1.0 - a) + warped.astype(np.float32) * a).astype(np.uint8)
            else:
                # full-rectangle mask, warped to rotated rect
                mask = np.full((ch, cw), 255, dtype=np.uint8)
                warped_m = cv2.warpAffine(
                    mask, M, (W, H),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                canvas[warped_m > 0] = warped[warped_m > 0]

            # build label line (YOLO pose)
            quad_n = normalize_points(quad, W, H)
            cx, cy, bw, bh = compute_bbox_from_pts(quad, W, H)

            # clamp safety
            def _clamp01(x: float) -> float:
                return max(0.0, min(1.0, float(x)))

            cx, cy, bw, bh = map(_clamp01, (cx, cy, bw, bh))
            quad_n = np.clip(quad_n, 0.0, 1.0)

            # visibility fixed to 2 (visible) since we ensure fully inside & no overlap
            v = 2

            parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for i in range(4):
                parts += [f"{quad_n[i, 0]:.6f}", f"{quad_n[i, 1]:.6f}", str(v)]
            label_lines.append(" ".join(parts))
            placed_quads.append(quad)

            placed = True
            break

        if not placed:
            # If one card cannot be placed, signal failure by raising; outer loop will restart image
            raise RuntimeError("Failed to place a card without overlap (try adjusting size ranges or margins).")

    return canvas, label_lines, placed_quads


def generate_dataset(cfg: SynthKptConfig, seed: int = 42) -> None:
    rng = random.Random(seed)

    bg_paths = list_images(cfg.bg_dir)
    card_paths = list_images(cfg.card_dir)
    if not bg_paths:
        raise RuntimeError(f"No backgrounds found in: {cfg.bg_dir}")
    if not card_paths:
        raise RuntimeError(f"No business cards found in: {cfg.card_dir}")

    out_img_dir = cfg.out_dir / "images"
    out_lbl_dir = cfg.out_dir / "labels"
    out_viz_dir = cfg.out_dir / "viz"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_viz:
        out_viz_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(cfg.num_images):
        # choose background and fit to fixed size
        bg_path = rng.choice(bg_paths)
        bg = imread_any(bg_path)
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        if bg.ndim == 3 and bg.shape[2] == 4:
            bg = bg[:, :, :3]

        bg = _fit_background_to_size(bg, cfg.out_w, cfg.out_h)

        # create one sample with retries (because non-overlap with 2~4 cards is rejection-sampling)
        ok = False
        last_err = None
        for _ in range(cfg.max_restarts_per_image):
            try:
                composed, label_lines, quads = _compose_one(bg, card_paths, cfg, rng)
                # must have 2~4 objects
                if not (cfg.min_cards <= len(label_lines) <= cfg.max_cards):
                    raise RuntimeError(f"Generated object count out of range: {len(label_lines)}")
                ok = True
                break
            except Exception as e:
                last_err = e
                continue

        if not ok:
            raise RuntimeError(f"Failed to generate image {idx} after retries. Last error: {last_err}")

        stem = f"img_{idx+1:06d}"
        img_out = out_img_dir / f"{stem}.jpg"
        lbl_out = out_lbl_dir / f"{stem}.txt"

        imwrite_jpg(img_out, composed, quality=95)
        lbl_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        if cfg.save_viz:
            viz = _draw_viz(composed, quads)
            imwrite_jpg(out_viz_dir / f"{stem}.jpg", viz, quality=95)

        if (idx + 1) % 50 == 0:
            print(f"[kpt_synth] generated {idx+1}/{cfg.num_images} -> {cfg.out_dir}")
