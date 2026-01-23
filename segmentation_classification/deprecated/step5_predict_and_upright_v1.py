#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
segmentation_classification/tools/step5_predict_warp_and_upright.py

Step5:
1) Use trained YOLO11-seg model to predict business card instances on background images.
2) For each instance, compute a quadrilateral by minAreaRect on mask polygon points.
3) Perspective warp (geometric rectify) to obtain a "flat" card.
4) Use trained YOLO11-cls classifier (rot0/rot90/rot180/rot270) to rotate to upright.
5) Save outputs: warped cards, upright cards, debug overlays, and per-image JSON metadata.

- No CLI args. Edit CONFIG at top.
- Chinese path safe IO (cv2.imdecode/tofile).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

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


def iter_images(src: Union[str, Path]) -> List[Path]:
    p = Path(src)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    if p.is_dir():
        out: List[Path] = []
        for ext in IMG_EXTS:
            out.extend(p.rglob(f"*{ext}"))
        return sorted(out)
    raise FileNotFoundError(f"Not a valid file/dir: {src}")


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


def quad_from_polygon_minarea(poly_xy: np.ndarray) -> np.ndarray:
    """
    Given polygon points (N,2) in pixel coords, compute a quadrilateral using minAreaRect.
    Returns 4 points (4,2) float32.
    """
    poly_xy = poly_xy.astype(np.float32)
    rect = cv2.minAreaRect(poly_xy)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)        # (4,2)
    return box.astype(np.float32)


def warp_quad(img_bgr: np.ndarray, quad: np.ndarray, max_side: int = 1400) -> np.ndarray:
    """
    Perspective warp by 4 points.
    - auto size by quad edge lengths
    - clamp max side to avoid huge outputs
    """
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    W = max(int(round(wA)), int(round(wB)), 1)
    H = max(int(round(hA)), int(round(hB)), 1)

    # clamp max_side (keep aspect)
    scale = 1.0
    if max(W, H) > max_side:
        scale = max_side / float(max(W, H))
        W = max(1, int(round(W * scale)))
        H = max(1, int(round(H * scale)))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    out = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def rotate_by_label(img_bgr: np.ndarray, label: str) -> np.ndarray:
    """
    Our Step4 dataset used OpenCV positive angles = CCW:
      rot90 means the image is CCW 90 relative to upright -> to fix rotate CW 90.
    """
    s = label.lower()
    if "rot0" in s or s.endswith("0"):
        return img_bgr
    if "rot90" in s or "90" in s:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if "rot180" in s or "180" in s:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if "rot270" in s or "270" in s:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # fallback: no-op
    return img_bgr


def to_jsonable(x):
    """Convert numpy/path types to JSON-serializable python types."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# =========================
# CONFIG
# =========================
@dataclass
class Step5Config:
    # Inputs
    source: Union[str, Path] = Path(r"segmentation_classification/assets/test_backgrounds")

    # Models (use local paths to avoid download)
    seg_weights: Union[str, Path] = Path(r"segmentation_classification/runs/step3_seg/weights/best.pt")
    cls_weights: Union[str, Path] = Path(r"segmentation_classification/runs/step4_upright_cls/weights/best.pt")

    # Predict params
    imgsz_seg: int = 960
    conf_seg: float = 0.25
    iou_seg: float = 0.5
    max_det: int = 20

    imgsz_cls: int = 384

    device: Union[int, str] = 0  # 0 / "cpu"

    # Output
    out_dir: Union[str, Path] = Path(r"segmentation_classification/outputs/step5")

    # Warp
    warp_max_side: int = 1400

    # Debug
    save_debug_overlay: bool = True
    save_warped: bool = True
    save_upright: bool = True

    # Filtering
    only_class_id: Optional[int] = 0  # set None to keep all classes
    sort_by_area_desc: bool = True


# =========================
# CONFIG EDIT ZONE (edit here)
# =========================
def build_config() -> Step5Config:
    cfg = Step5Config()

    # Inputs
    cfg.source = r"segmentation_classification/assets/seg_step1_test/images"  # 背景图目录或单张图

    # Models (local paths)
    cfg.seg_weights = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cfg.cls_weights = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"

    # Predict params
    cfg.imgsz_seg = 960
    cfg.conf_seg = 0.25
    cfg.iou_seg = 0.5
    cfg.max_det = 20

    cfg.imgsz_cls = 384
    cfg.device = 0  # 0 / "cpu"

    # Output
    cfg.out_dir = r"segmentation_classification/outputs/step5"

    # Warp
    cfg.warp_max_side = 1400

    # Debug
    cfg.save_debug_overlay = True
    cfg.save_warped = True
    cfg.save_upright = True

    # Filtering
    cfg.only_class_id = 0     # 只保留名片这一类；如果你只有1类也可以保持0；想不过滤就设 None
    cfg.sort_by_area_desc = True

    return cfg


# =========================
# Main
# =========================
def classify_orientation(model_cls: YOLO, img_bgr: np.ndarray, imgsz: int, device) -> Tuple[str, float, int]:
    """
    Return (label_name, prob, class_id) from YOLO classification model.
    """
    pred = model_cls.predict(
        source=img_bgr,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r = pred[0]
    probs = getattr(r, "probs", None)
    if probs is None:
        return ("rot0", 0.0, 0)
    top1 = int(probs.top1)
    conf = float(probs.top1conf) if hasattr(probs, "top1conf") else float(probs.data[top1])
    names = r.names if hasattr(r, "names") and isinstance(r.names, dict) else {}
    label = str(names.get(top1, str(top1)))
    return (label, conf, top1)


def draw_overlay(img_bgr: np.ndarray, cards: List[Dict]) -> np.ndarray:
    out = img_bgr.copy()
    for c in cards:
        quad = np.array(c["quad_xy"], dtype=np.int32)
        cv2.polylines(out, [quad], True, (0, 255, 0), 2)
        x1, y1, x2, y2 = c["bbox_xyxy"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        txt = f"{c.get('cls_fix_label','?')} {c.get('cls_fix_prob',0.0):.2f}"
        cv2.putText(out, txt, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def main(cfg: Step5Config) -> None:
    src_list = iter_images(cfg.source)
    if not src_list:
        raise FileNotFoundError(f"No images found in: {cfg.source}")

    seg_w = Path(cfg.seg_weights)
    cls_w = Path(cfg.cls_weights)
    if not seg_w.exists():
        raise FileNotFoundError(f"Seg weights not found: {seg_w}")
    if not cls_w.exists():
        raise FileNotFoundError(f"Cls weights not found: {cls_w}")

    out_dir = Path(cfg.out_dir)
    out_warp = out_dir / "warped"
    out_upright = out_dir / "upright"
    out_debug = out_dir / "debug"
    out_meta = out_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    if cfg.save_warped:
        out_warp.mkdir(parents=True, exist_ok=True)
    if cfg.save_upright:
        out_upright.mkdir(parents=True, exist_ok=True)
    if cfg.save_debug_overlay:
        out_debug.mkdir(parents=True, exist_ok=True)

    model_seg = YOLO(str(seg_w))
    model_cls = YOLO(str(cls_w))

    for img_path in src_list:
        img_bgr = imread_bgr(img_path)
        H, W = img_bgr.shape[:2]

        # --- segmentation predict ---
        results = model_seg.predict(
            source=img_bgr,
            imgsz=cfg.imgsz_seg,
            conf=cfg.conf_seg,
            iou=cfg.iou_seg,
            max_det=cfg.max_det,
            device=cfg.device,
            verbose=False,
        )
        r = results[0]

        cards: List[Dict] = []

        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is None or masks is None or masks.xy is None:
            # no detections
            meta = {
                "image": str(img_path),
                "orig_hw": [H, W],
                "cards": [],
            }
            (out_meta / f"{img_path.stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=to_jsonable),
                encoding="utf-8",
            )
            if cfg.save_debug_overlay:
                imwrite(out_debug / f"{img_path.stem}__debug.jpg", img_bgr)
            continue

        # Gather detections
        polys = masks.xy  # list of (Ni,2) in pixel coords
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
        confs = boxes.conf.cpu().numpy().astype(float) if hasattr(boxes, "conf") else None

        n = min(len(polys), xyxy.shape[0] if xyxy is not None else len(polys))
        idxs = list(range(n))

        # optional filter by class id
        if cfg.only_class_id is not None and cls_ids is not None:
            idxs = [i for i in idxs if int(cls_ids[i]) == int(cfg.only_class_id)]

        # sort by area desc
        if cfg.sort_by_area_desc:
            areas = []
            for i in idxs:
                x1, y1, x2, y2 = xyxy[i]
                areas.append(((x2 - x1) * (y2 - y1), i))
            idxs = [i for _, i in sorted(areas, key=lambda t: t[0], reverse=True)]

        for k, i in enumerate(idxs):
            poly = np.asarray(polys[i], dtype=np.float32)
            if poly.shape[0] < 4:
                continue

            quad = quad_from_polygon_minarea(poly)  # (4,2)
            warped = warp_quad(img_bgr, quad, max_side=cfg.warp_max_side)

            # classify + rotate to upright
            label, p, cid = classify_orientation(model_cls, warped, cfg.imgsz_cls, cfg.device)
            upright = rotate_by_label(warped, label)

            warp_path = None
            upright_path = None
            if cfg.save_warped:
                warp_path = out_warp / f"{img_path.stem}__card{k:02d}__warp.jpg"
                imwrite(warp_path, warped)
            if cfg.save_upright:
                upright_path = out_upright / f"{img_path.stem}__card{k:02d}__upright.jpg"
                imwrite(upright_path, upright)

            x1, y1, x2, y2 = xyxy[i].tolist()
            det = {
                "idx": k,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "seg_conf": float(confs[i]) if confs is not None else None,
                "seg_cls_id": int(cls_ids[i]) if cls_ids is not None else None,
                "quad_xy": quad.astype(float).tolist(),
                "quad_xy_ordered": order_points(quad).astype(float).tolist(),
                "cls_fix_label": label,
                "cls_fix_prob": float(p),
                "cls_fix_id": int(cid),
                "warp_path": str(warp_path) if warp_path else None,
                "upright_path": str(upright_path) if upright_path else None,
            }
            cards.append(det)

        # debug overlay
        if cfg.save_debug_overlay:
            # quad in overlay expects int32
            overlay_cards = []
            for c in cards:
                quad_xy = np.array(c["quad_xy_ordered"], dtype=np.int32)
                overlay_cards.append({
                    "quad_xy": quad_xy,
                    "bbox_xyxy": c["bbox_xyxy"],
                    "cls_fix_label": c["cls_fix_label"],
                    "cls_fix_prob": c["cls_fix_prob"],
                })
            dbg = draw_overlay(img_bgr, overlay_cards)
            imwrite(out_debug / f"{img_path.stem}__debug.jpg", dbg)

        # meta
        meta = {
            "image": str(img_path),
            "orig_hw": [H, W],
            "num_cards": len(cards),
            "cards": cards,
        }
        (out_meta / f"{img_path.stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=to_jsonable),
            encoding="utf-8",
        )

        print(f"[OK] {img_path.name}: cards={len(cards)}")

    print(f"\n[Done] outputs -> {Path(cfg.out_dir).resolve()}")


if __name__ == "__main__":
    CONFIG = build_config()
    main(CONFIG)
