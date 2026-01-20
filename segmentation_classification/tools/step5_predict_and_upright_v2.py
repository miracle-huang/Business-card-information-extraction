#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
segmentation_classification/tools/step5_predict_and_upright.py

Step5:
1) YOLO11-seg predict cards on background
2) Build quad from mask contour (approxPolyDP -> 4pt) else fallback minAreaRect
3) Warp perspective (image + mask), then crop tightly by warped mask (with small erode)
4) Upright by cls model: evaluate 4 rotated candidates, pick one with max P(rot0)

This greatly reduces background leakage and stabilizes upright results.
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


def imwrite(path: Path, img_bgr: np.ndarray, jpg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def iter_images(source: Union[str, Path]) -> List[Path]:
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    if p.is_dir():
        out = []
        for ext in IMG_EXTS:
            out += list(p.rglob(f"*{ext}"))
        return sorted(out)
    raise FileNotFoundError(source)


def json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # make Path JSON serializable
    def _default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


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


def quad_inset(quad: np.ndarray, inset_ratio: float = 0.015) -> np.ndarray:
    """Move quad points slightly towards center to reduce background leakage."""
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return (q * (1.0 - inset_ratio) + c * inset_ratio).astype(np.float32)


def warp_quad(
    img_bgr: np.ndarray,
    quad_xy: np.ndarray,
    max_side: int = 1400,
    border_value: int = 255,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Warp by quad to a fronto-parallel rectangle.
    Returns: warped_img, H, (out_w, out_h)
    """
    rect = order_points(quad_xy)
    (tl, tr, br, bl) = rect

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    w = max(2, int(round(max(wA, wB))))
    h = max(2, int(round(max(hA, hB))))

    # limit size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w = max(2, int(round(w * scale)))
        h = max(2, int(round(h * scale)))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        img_bgr,
        Hm,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
    return warped, Hm, (w, h)


def rotate_exact_90(img: np.ndarray, k90: int) -> np.ndarray:
    """k90 in {0,1,2,3} => rotate CCW by 0/90/180/270."""
    k = int(k90) % 4
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


# =========================
# Mask/Quad extraction from Ultralytics seg output
# =========================
def poly_to_mask(polys_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Fill polygon into a binary mask (uint8 0/255)."""
    h, w = hw
    m = np.zeros((h, w), dtype=np.uint8)
    if polys_xy is None or len(polys_xy) < 3:
        return m
    pts = np.round(polys_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(m, [pts], 255)
    return m


def quad_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Try get 4 corners from mask contour.
    Returns (quad(4,2) float32, method)
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("no contour from mask")
    cnt = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(cnt, True)
    # try multiple eps to get 4-point polygon
    for eps_ratio in (0.008, 0.012, 0.016, 0.02, 0.028, 0.04):
        approx = cv2.approxPolyDP(cnt, epsilon=eps_ratio * peri, closed=True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            return order_points(quad), f"approx4_eps{eps_ratio:.3f}"

    # fallback: minAreaRect
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)  # (4,2)
    return order_points(box), "minAreaRect"


def warp_and_tight_crop_by_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    quad_xy: np.ndarray,
    warp_max_side: int,
    inset_ratio: float,
    erode_px: int,
    crop_margin: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Warp image + warp mask, then tight crop by (eroded) warped mask bbox.
    Returns cropped_warp, debug_info
    """
    quad = quad_inset(quad_xy, inset_ratio=inset_ratio) if inset_ratio > 0 else quad_xy.astype(np.float32)

    warped, Hm, (w, h) = warp_quad(img_bgr, quad, max_side=warp_max_side, border_value=255)

    warped_mask = cv2.warpPerspective(
        mask_u8,
        Hm,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        warped_mask = cv2.erode(warped_mask, kernel, iterations=1)

    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: no mask after warp/erode -> return warped
        info = {
            "warp_size": [h, w],
            "tight_crop_xyxy": None,
            "mask_crop_fail": True,
        }
        return warped, info

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # margin and clip
    x1 = max(0, x1 - crop_margin)
    y1 = max(0, y1 - crop_margin)
    x2 = min(w - 1, x2 + crop_margin)
    y2 = min(h - 1, y2 + crop_margin)

    cropped = warped[y1 : y2 + 1, x1 : x2 + 1].copy()

    info = {
        "warp_size": [h, w],
        "tight_crop_xyxy": [x1, y1, x2, y2],
        "mask_crop_fail": False,
    }
    return cropped, info


# =========================
# Classification: choose rotation that maximizes P(rot0)
# =========================
def cls_prob_rot0(cls_model: YOLO, img_bgr: np.ndarray, imgsz: int, device: Union[str, int]) -> Tuple[float, Dict]:
    """
    Return P(rot0) and full probs dict.
    """
    r = cls_model.predict(source=img_bgr, imgsz=imgsz, device=device, verbose=False)[0]
    probs = r.probs  # ultralytics.engine.results.Probs
    if probs is None:
        return 0.0, {}
    p = probs.data.detach().cpu().numpy().astype(float)  # shape (nc,)
    names = getattr(cls_model, "names", None) or getattr(r, "names", None) or {}
    # build name->prob
    d = {}
    for i, pi in enumerate(p):
        d[str(names.get(i, str(i)))] = float(pi)

    rot0_prob = d.get("rot0", None)
    if rot0_prob is None:
        # if class name differs, fallback to top1 mapping not possible -> use max as proxy
        rot0_prob = float(np.max(p))
    return float(rot0_prob), d


def upright_by_best_rot0(
    cls_model: YOLO,
    warp_bgr: np.ndarray,
    imgsz: int,
    device: Union[str, int],
) -> Tuple[np.ndarray, Dict]:
    """
    Generate 4 candidates (0/90/180/270), choose one with max P(rot0).
    """
    best_k = 0
    best_p = -1.0
    best_probs = None

    for k in (0, 1, 2, 3):
        cand = rotate_exact_90(warp_bgr, k)
        p_rot0, prob_dict = cls_prob_rot0(cls_model, cand, imgsz=imgsz, device=device)
        if p_rot0 > best_p:
            best_p = p_rot0
            best_k = k
            best_probs = prob_dict

    out = rotate_exact_90(warp_bgr, best_k)
    return out, {"best_k90_ccw": int(best_k), "best_rot0_prob": float(best_p), "probs": best_probs or {}}


# =========================
# Config
# =========================
@dataclass
class Step5Config:
    # input
    source: Union[str, Path] = r"segmentation_classification/assets/seg_step1_test/images"

    # models
    seg_weights: str = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cls_weights: str = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"

    # seg predict
    imgsz_seg: int = 960
    conf_seg: float = 0.25
    iou_seg: float = 0.5
    max_det: int = 20
    only_class_id: Optional[int] = 0  # card class id; set None to disable filtering
    sort_by_area_desc: bool = True

    # warp/crop tuning (关键)
    warp_max_side: int = 1400
    quad_inset_ratio: float = 0.015   # 0.0~0.03，越大越“收紧”
    mask_erode_px: int = 2            # 0~4，越大越去边缘背景
    crop_margin: int = 4              # tight crop 后再留一点边

    # cls predict (upright selection)
    imgsz_cls: int = 384
    device: Union[int, str] = 0

    # outputs
    out_dir: Union[str, Path] = r"segmentation_classification/outputs/step5"
    save_debug_overlay: bool = True
    save_warped: bool = True
    save_upright: bool = True


def build_config() -> Step5Config:
    cfg = Step5Config()

    # ===== edit here =====
    cfg.source = r"segmentation_classification/assets/seg_step1_test/images"
    cfg.seg_weights = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cfg.cls_weights = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"

    cfg.imgsz_seg = 960
    cfg.conf_seg = 0.25
    cfg.iou_seg = 0.5
    cfg.max_det = 20

    # 优化这几个通常收益最大
    cfg.quad_inset_ratio = 0.02   # 更紧一点（减少背景）
    cfg.mask_erode_px = 2         # 先 2，背景仍多就调 3
    cfg.crop_margin = 4

    cfg.imgsz_cls = 384
    cfg.device = 0

    cfg.out_dir = r"segmentation_classification/outputs/step5_v2"
    cfg.save_debug_overlay = True
    cfg.save_warped = True
    cfg.save_upright = True
    # =====================

    return cfg


# =========================
# Debug draw
# =========================
def draw_debug(img_bgr: np.ndarray, cards_meta: List[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for c in cards_meta:
        x1, y1, x2, y2 = [int(round(v)) for v in c["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

        quad = np.array(c["quad_xy_ordered"], dtype=np.float32)
        pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)

        label = c.get("cls_fix_label", "na")
        prob = float(c.get("cls_fix_prob", 0.0))
        cv2.putText(
            out,
            f"{label} {prob:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return out


def main(cfg: Step5Config) -> None:
    out_dir = Path(cfg.out_dir)
    meta_dir = out_dir / "meta"
    dbg_dir = out_dir / "debug"
    warped_dir = out_dir / "warped"
    upright_dir = out_dir / "upright"
    for d in (meta_dir, dbg_dir, warped_dir, upright_dir):
        d.mkdir(parents=True, exist_ok=True)

    seg_model = YOLO(cfg.seg_weights)
    cls_model = YOLO(cfg.cls_weights)

    paths = iter_images(cfg.source)
    if not paths:
        raise FileNotFoundError(cfg.source)

    for img_path in paths:
        img = imread_bgr(img_path)
        H, W = img.shape[:2]

        seg_res = seg_model.predict(
            source=img,
            imgsz=cfg.imgsz_seg,
            conf=cfg.conf_seg,
            iou=cfg.iou_seg,
            max_det=cfg.max_det,
            device=cfg.device,
            verbose=False,
        )[0]

        boxes = seg_res.boxes
        masks = seg_res.masks

        cards_meta: List[dict] = []

        if boxes is None or len(boxes) == 0 or masks is None or masks.xy is None:
            meta = {
                "image": str(img_path),
                "orig_hw": [H, W],
                "num_cards": 0,
                "cards": [],
            }
            json_dump(meta_dir / f"{img_path.stem}.json", meta)
            continue

        # gather instances
        insts = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float).tolist()

            if cfg.only_class_id is not None and cls_id != int(cfg.only_class_id):
                continue

            poly = masks.xy[i]  # (n,2) float in original pixels
            poly_np = np.array(poly, dtype=np.float32)

            # area for sorting
            area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))

            insts.append((area, i, cls_id, conf, xyxy, poly_np))

        if cfg.sort_by_area_desc:
            insts.sort(key=lambda x: x[0], reverse=True)

        for new_idx, (_, i, cls_id, conf, xyxy, poly_np) in enumerate(insts):
            mask_u8 = poly_to_mask(poly_np, (H, W))

            # get quad from mask
            try:
                quad, quad_method = quad_from_mask(mask_u8)
            except Exception:
                # fallback: use bbox as axis-aligned quad
                x1, y1, x2, y2 = xyxy
                quad = order_points(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
                quad_method = "bbox_fallback"

            quad_ordered = quad.astype(float).tolist()

            # warp + mask-tight crop
            warp_crop, crop_info = warp_and_tight_crop_by_mask(
                img_bgr=img,
                mask_u8=mask_u8,
                quad_xy=quad,
                warp_max_side=cfg.warp_max_side,
                inset_ratio=cfg.quad_inset_ratio,
                erode_px=cfg.mask_erode_px,
                crop_margin=cfg.crop_margin,
            )

            # upright (4-candidate pick rot0 max)
            upright_img, upright_info = upright_by_best_rot0(
                cls_model=cls_model,
                warp_bgr=warp_crop,
                imgsz=cfg.imgsz_cls,
                device=cfg.device,
            )

            # save
            warp_path = warped_dir / f"{img_path.stem}__card{new_idx:02d}__warp.jpg"
            upright_path = upright_dir / f"{img_path.stem}__card{new_idx:02d}__upright.jpg"
            if cfg.save_warped:
                imwrite(warp_path, warp_crop)
            if cfg.save_upright:
                imwrite(upright_path, upright_img)

            # keep your original meta keys, plus extra diagnostics
            cards_meta.append(
                {
                    "idx": int(new_idx),
                    "bbox_xyxy": xyxy,
                    "seg_conf": conf,
                    "seg_cls_id": int(cls_id),
                    "quad_xy": poly_np[:4].astype(float).tolist() if len(poly_np) >= 4 else poly_np.astype(float).tolist(),
                    "quad_xy_ordered": quad_ordered,
                    "quad_method": quad_method,
                    "warp_path": str(warp_path).replace("/", "\\"),
                    "upright_path": str(upright_path).replace("/", "\\"),
                    # new info
                    "warp_size": crop_info.get("warp_size"),
                    "tight_crop_xyxy": crop_info.get("tight_crop_xyxy"),
                    "mask_crop_fail": crop_info.get("mask_crop_fail", False),
                    "upright_best_k90_ccw": upright_info.get("best_k90_ccw"),
                    "cls_rot0_prob": upright_info.get("best_rot0_prob"),
                    "cls_probs": upright_info.get("probs", {}),
                    # keep compatibility fields
                    "cls_fix_label": "rot0",
                    "cls_fix_prob": upright_info.get("best_rot0_prob", 0.0),
                    "cls_fix_id": 0,
                }
            )

        meta = {
            "image": str(img_path).replace("/", "\\"),
            "orig_hw": [H, W],
            "num_cards": len(cards_meta),
            "cards": cards_meta,
        }
        json_dump(meta_dir / f"{img_path.stem}.json", meta)

        if cfg.save_debug_overlay and cards_meta:
            dbg = draw_debug(img, cards_meta)
            imwrite(dbg_dir / f"{img_path.stem}__debug.jpg", dbg)

        print(f"[OK] {img_path.name} -> cards={len(cards_meta)}")


if __name__ == "__main__":
    CONFIG = build_config()
    main(CONFIG)
