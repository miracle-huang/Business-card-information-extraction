#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
segmentation_classification/tools/step5_predict_warp_upright_v5.py

Step5 - Predict segmentation -> warp each card -> (optional) refine warp -> rotate to upright by classifier.

Fixes vs v4:
1) Corner refine uses contour edge-line fitting (cv2.fitLine) + intersections, more stable than ShiTomasi-only.
2) Optional second-stage warp refinement in warped space to remove residual skew.
3) Tighter crop using warped mask with controllable erosion to suppress "background ring".

v5.1 hotfix:
- Ensure output subfolders (debug/warped/upright/meta) exist before writing.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# =========================
# Ensure repo root on sys.path (Windows friendly)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# IO helpers (Chinese path OK)
# =========================
def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def iter_images(source: Union[str, Path]) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    for x in sorted(p.rglob("*")):
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            out.append(x)
    return out


# =========================
# Geometry helpers
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points to tl, tr, br, bl."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def inset_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    """Move quad points slightly toward center to suppress background."""
    if ratio <= 0:
        return quad.astype(np.float32)
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return c + (q - c) * (1.0 - float(ratio))


def line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """Intersection of 2 param lines: p1+t*d1 and p2+u*d2. Return point or None if parallel."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float32)
    b = (p2 - p1).astype(np.float32)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-6:
        return None
    t_u = np.linalg.solve(A, b)
    t = float(t_u[0])
    return (p1 + t * d1).astype(np.float32)


def point_line_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from pt to infinite line through a-b."""
    v = b - a
    w = pt - a
    denom = float(np.hypot(v[0], v[1]) + 1e-9)
    return float(abs(np.cross(v, w)) / denom)


# =========================
# Mask / contour helpers
# =========================
def polygon_to_mask(poly_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """poly_xy: (N,2) float -> binary mask uint8 {0,255}."""
    H, W = hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = poly_xy.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def clean_mask(mask: np.ndarray, close_ksize: int = 7) -> np.ndarray:
    """Remove holes/noise; keep largest connected component."""
    m = mask.copy()
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.fillPoly(out, [cnt], 255)
    return out


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# =========================
# Quad estimation (core fix)
# =========================
def approx_quad_from_contour(cnt: np.ndarray, eps_ratio: float) -> Optional[np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    eps = float(eps_ratio) * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    return None


def box_from_min_area_rect(cnt: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return order_points(box)


def fit_edge_lines_and_intersections(cnt: np.ndarray, init_box: np.ndarray, dist_thresh: float) -> Optional[np.ndarray]:
    """
    Use init_box edges as guidance:
    - For each of 4 edges, collect contour points close to that edge
    - Fit line with cv2.fitLine
    - Intersect adjacent lines -> refined corners
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    box = init_box.astype(np.float32)  # tl,tr,br,bl
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]

    fitted = []
    for (a, b) in edges:
        dists = np.array([point_line_distance(p, a, b) for p in pts], dtype=np.float32)
        sel = pts[dists < dist_thresh]
        if sel.shape[0] < 40:
            k = min(200, pts.shape[0])
            idx = np.argsort(dists)[:k]
            sel = pts[idx]
        if sel.shape[0] < 10:
            return None

        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        d = np.array([vx, vy], dtype=np.float32)
        p0 = np.array([x0, y0], dtype=np.float32)
        n = float(np.hypot(d[0], d[1]) + 1e-9)
        d = d / n
        fitted.append((p0, d))

    corners = []
    for i in range(4):
        p1, d1 = fitted[i]
        p2, d2 = fitted[(i + 1) % 4]
        inter = line_intersection(p1, d1, p2, d2)
        if inter is None:
            return None
        corners.append(inter)

    quad = np.stack(corners, axis=0).astype(np.float32)
    return order_points(quad)


def estimate_quad_from_mask(mask: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns (quad, method). quad is ordered tl,tr,br,bl in image coords.
    """
    cnt = largest_contour(mask)
    if cnt is None:
        return None, "no_contour"

    q = approx_quad_from_contour(cnt, eps_ratio=0.012)
    if q is not None:
        return order_points(q), "approx4_eps0.012"

    box = box_from_min_area_rect(cnt)
    w = float(np.linalg.norm(box[1] - box[0]))
    h = float(np.linalg.norm(box[3] - box[0]))
    dist_thresh = max(3.0, 0.01 * min(w, h))
    q2 = fit_edge_lines_and_intersections(cnt, box, dist_thresh=dist_thresh)
    if q2 is not None:
        return q2, f"fitLine_edges_dist{dist_thresh:.1f}"

    return box, "minAreaRect"


# =========================
# Warp + crop
# =========================
def warp_by_quad(img_bgr: np.ndarray, mask_u8: np.ndarray, quad: np.ndarray, pad: int, border_val: int = 255):
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    out_w = int(round(max(wA, wB)))
    out_h = int(round(max(hA, hB)))

    out_w = max(out_w, 64)
    out_h = max(out_h, 64)

    dst = np.array(
        [[pad, pad], [out_w - 1 + pad, pad], [out_w - 1 + pad, out_h - 1 + pad], [pad, out_h - 1 + pad]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

    warped_img = cv2.warpPerspective(
        img_bgr,
        M,
        (out_w + 2 * pad, out_h + 2 * pad),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_val, border_val, border_val),
    )
    warped_mask = cv2.warpPerspective(
        mask_u8,
        M,
        (out_w + 2 * pad, out_h + 2 * pad),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_img, warped_mask


def tight_crop_by_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, erode_px: int, margin_px: int) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    m = mask_u8.copy()
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k, iterations=1)

    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0]), True

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(img_bgr.shape[1] - 1, x2 + margin_px)
    y2 = min(img_bgr.shape[0] - 1, y2 + margin_px)

    cropped = img_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    return cropped, (x1, y1, x2, y2), False


# =========================
# Upright by classification (try 4 rotations, pick best "rot0" prob)
# =========================
def rot90_ccw(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def get_rot0_class_id(cls_model: YOLO) -> int:
    names = getattr(cls_model, "names", None)
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "rot0":
                return int(k)
    if isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if str(v).lower() == "rot0":
                return int(i)
    return 0


def choose_upright_by_cls(img_bgr: np.ndarray, cls_model: YOLO, imgsz: int, device) -> Tuple[np.ndarray, int, float, Dict[str, float]]:
    rot0_id = get_rot0_class_id(cls_model)

    probs_map = {}
    best_k = 0
    best_p = -1.0
    best_img = img_bgr

    for k in range(4):
        cand = rot90_ccw(img_bgr, k)
        r = cls_model.predict(cand, imgsz=imgsz, device=device, verbose=False)[0]
        p = float(r.probs.data[rot0_id].item())
        probs_map[f"k{k}"] = p
        if p > best_p:
            best_p = p
            best_k = k
            best_img = cand

    named = {}
    try:
        names = cls_model.names
        vec = cls_model.predict(img_bgr, imgsz=imgsz, device=device, verbose=False)[0].probs.data.detach().cpu().numpy()
        if isinstance(names, dict):
            for kk, vv in names.items():
                named[str(vv)] = float(vec[int(kk)])
        elif isinstance(names, (list, tuple)):
            for i, vv in enumerate(names):
                named[str(vv)] = float(vec[int(i)])
    except Exception:
        pass

    return best_img, best_k, best_p, named


# =========================
# Config
# =========================
@dataclass
class Step5Config:
    source: str = r"segmentation_classification/assets/seg_step1_test/images"

    seg_weights: str = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cls_weights: str = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"

    imgsz_seg: int = 960
    conf_seg: float = 0.25
    iou_seg: float = 0.5
    max_det: int = 20
    device: Union[int, str] = 0

    imgsz_cls: int = 384

    only_class_id: Optional[int] = 0

    quad_inset_ratio: float = 0.004
    warp_pad: int = 6
    refine_second_warp: bool = True
    mask_close_ksize: int = 7

    crop_erode_px: int = 3
    crop_margin_px: int = 0
    min_area_px: int = 20000

    out_dir: str = r"segmentation_classification/outputs/step5_v5"
    save_debug: bool = True
    save_warp: bool = True
    save_upright: bool = True
    save_meta: bool = True


# =========================
# Debug draw
# =========================
def draw_quad(img: np.ndarray, quad: np.ndarray, color=(0, 255, 0), thickness=2):
    q = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [q], True, color, thickness)


def main(cfg: Step5Config):
    out_root = Path(cfg.out_dir)
    out_debug = out_root / "debug"
    out_warp = out_root / "warped"
    out_upr = out_root / "upright"
    out_meta = out_root / "meta"

    # ✅ v5.1: ensure all subfolders exist (fix FileNotFoundError)
    out_root.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)
    out_warp.mkdir(parents=True, exist_ok=True)
    out_upr.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    seg_model = YOLO(cfg.seg_weights)
    cls_model = YOLO(cfg.cls_weights)

    paths = iter_images(cfg.source)
    if not paths:
        raise RuntimeError(f"No images found in: {cfg.source}")

    for img_path in paths:
        img = imread_bgr(img_path)
        H, W = img.shape[:2]

        results = seg_model.predict(
            source=img,
            imgsz=cfg.imgsz_seg,
            conf=cfg.conf_seg,
            iou=cfg.iou_seg,
            max_det=cfg.max_det,
            device=cfg.device,
            retina_masks=True,
            verbose=False,
        )[0]

        meta: Dict = {
            "image": str(img_path),
            "orig_hw": [H, W],
            "cards": [],
        }

        dbg = img.copy()

        if results.masks is None or results.boxes is None or len(results.boxes) == 0:
            if cfg.save_debug:
                imwrite(out_debug / f"{img_path.stem}__debug.jpg", dbg)
            if cfg.save_meta:
                (out_meta / f"{img_path.stem}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            continue

        polys = results.masks.xy
        boxes = results.boxes.xyxy.detach().cpu().numpy()
        confs = results.boxes.conf.detach().cpu().numpy()
        clss = results.boxes.cls.detach().cpu().numpy().astype(int)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)

        card_idx = 0
        for j in order:
            cls_id = int(clss[j])
            if cfg.only_class_id is not None and cls_id != int(cfg.only_class_id):
                continue

            x1, y1, x2, y2 = boxes[j].tolist()
            area = float((x2 - x1) * (y2 - y1))
            if area < cfg.min_area_px:
                continue

            poly = np.array(polys[j], dtype=np.float32)
            if poly.shape[0] < 4:
                continue

            mask = polygon_to_mask(poly, (H, W))
            mask = clean_mask(mask, close_ksize=cfg.mask_close_ksize)

            quad, method = estimate_quad_from_mask(mask)
            if quad is None:
                continue

            quad = inset_quad(quad, cfg.quad_inset_ratio)

            if cfg.save_debug:
                cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                draw_quad(dbg, quad, color=(0, 255, 0), thickness=2)

            warp1, wmask1 = warp_by_quad(img, mask, quad, pad=cfg.warp_pad, border_val=255)

            quad2 = None
            quad2_method = None
            warp2, wmask2 = warp1, wmask1
            if cfg.refine_second_warp:
                wmask_clean = clean_mask(wmask1, close_ksize=max(5, cfg.mask_close_ksize))
                quad2, quad2_method = estimate_quad_from_mask(wmask_clean)
                if quad2 is not None:
                    quad2 = inset_quad(quad2, max(0.0, cfg.quad_inset_ratio * 0.5))
                    warp2, wmask2 = warp_by_quad(warp1, wmask_clean, quad2, pad=0, border_val=255)

            crop_img, crop_xyxy, crop_fail = tight_crop_by_mask(
                warp2,
                wmask2,
                erode_px=cfg.crop_erode_px,
                margin_px=cfg.crop_margin_px,
            )

            upright, best_k, best_p, named_probs = choose_upright_by_cls(
                crop_img, cls_model, imgsz=cfg.imgsz_cls, device=cfg.device
            )

            stem = f"{img_path.stem}__card{card_idx:02d}"
            if cfg.save_warp:
                imwrite(out_warp / f"{stem}__warp.jpg", crop_img)
            if cfg.save_upright:
                imwrite(out_upr / f"{stem}__upright.jpg", upright)

            meta["cards"].append(
                {
                    "idx": card_idx,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "seg_conf": float(confs[j]),
                    "seg_cls_id": cls_id,
                    "quad_xy_ordered": quad.astype(float).tolist(),
                    "quad_method": method,
                    "quad_inset_ratio": float(cfg.quad_inset_ratio),
                    "warp_pad": int(cfg.warp_pad),
                    "warp1_wh": [int(warp1.shape[1]), int(warp1.shape[0])],
                    "refine_second_warp": bool(cfg.refine_second_warp),
                    "quad2_xy_ordered": None if quad2 is None else quad2.astype(float).tolist(),
                    "quad2_method": quad2_method,
                    "tight_crop_xyxy": list(map(int, crop_xyxy)),
                    "mask_crop_fail": bool(crop_fail),
                    "upright_best_k90_ccw": int(best_k),
                    "cls_rot0_prob": float(best_p),
                    "cls_probs_named": named_probs,
                    "warp_path": str(out_warp / f"{stem}__warp.jpg"),
                    "upright_path": str(out_upr / f"{stem}__upright.jpg"),
                }
            )

            card_idx += 1

        meta["num_cards"] = card_idx

        if cfg.save_debug:
            imwrite(out_debug / f"{img_path.stem}__debug.jpg", dbg)
        if cfg.save_meta:
            (out_meta / f"{img_path.stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print(f"[OK] Done. Outputs in: {Path(cfg.out_dir).resolve()}")


# =========================
# CONFIG EDIT ZONE
# =========================
def build_config() -> Step5Config:
    cfg = Step5Config()

    cfg.source = r"segmentation_classification/assets/seg_step1_test/images"
    cfg.seg_weights = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cfg.cls_weights = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"
    cfg.out_dir = r"segmentation_classification/outputs/step5_v5"

    # tuning defaults
    cfg.crop_erode_px = 3
    cfg.quad_inset_ratio = 0.004
    cfg.warp_pad = 6
    cfg.refine_second_warp = True

    return cfg


if __name__ == "__main__":
    main(build_config())
