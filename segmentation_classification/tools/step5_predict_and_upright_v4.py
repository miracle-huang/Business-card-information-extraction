#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step5 - Predict seg -> warp -> (tight crop) -> (micro-deskew) -> upright(0/90/180/270 by cls)

Fixes in this version:
1) ✅ Clip mask by quad BEFORE warping -> prevents warped_mask covering full frame -> tight crop works
2) ✅ More robust small-angle deskew from minAreaRect (OpenCV angle semantics handled)
3) ✅ Safer padding/crop defaults so bbox won't hit image border
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
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def quad_inset(quad: np.ndarray, inset_ratio: float) -> np.ndarray:
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return (q * (1.0 - inset_ratio) + c * inset_ratio).astype(np.float32)


def rotate_exact_90(img: np.ndarray, k90: int) -> np.ndarray:
    k = int(k90) % 4
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


# =========================
# Mask helpers
# =========================
def poly_to_mask(poly_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    m = np.zeros((h, w), dtype=np.uint8)
    if poly_xy is None or len(poly_xy) < 3:
        return m
    pts = np.round(poly_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(m, [pts], 255)
    return m


def quad_to_mask(quad_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    m = np.zeros((h, w), dtype=np.uint8)
    q = np.round(quad_xy).astype(np.int32)
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    cv2.fillPoly(m, [q.reshape(-1, 2)], 255)
    return m


def refine_quad_with_shitomasi(mask_u8: np.ndarray, quad: np.ndarray, max_dist: float = 25.0) -> Tuple[np.ndarray, bool]:
    q = quad.astype(np.float32).copy()
    edges = cv2.Canny(mask_u8, 50, 150)
    corners = cv2.goodFeaturesToTrack(
        edges,
        maxCorners=120,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None:
        return order_points(q), False

    pts = corners.reshape(-1, 2).astype(np.float32)

    changed = False
    used = set()
    for i in range(4):
        d = np.linalg.norm(pts - q[i], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= float(max_dist):
            key = (int(round(pts[j, 0])), int(round(pts[j, 1])))
            if key in used:
                continue
            used.add(key)
            q[i] = pts[j]
            changed = True

    return order_points(q), changed


def quad_from_mask(mask_u8: np.ndarray, refine_max_dist: float) -> Tuple[np.ndarray, str, bool]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("no contour from mask")
    cnt = max(cnts, key=cv2.contourArea)

    # use hull for stability
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)

    quad = None
    method = None

    for eps_ratio in (0.006, 0.008, 0.010, 0.012, 0.016, 0.02, 0.028, 0.04):
        approx = cv2.approxPolyDP(hull, epsilon=eps_ratio * peri, closed=True)
        if len(approx) == 4:
            quad = order_points(approx.reshape(4, 2).astype(np.float32))
            method = f"approx4_hull_eps{eps_ratio:.3f}"
            break

    if quad is None:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.float32)
        quad = order_points(box)
        method = "minAreaRect_hull"

    quad2, refined = refine_quad_with_shitomasi(mask_u8, quad, max_dist=refine_max_dist)
    if refined:
        method += "+shiTomasi"
    return quad2, method, refined


# =========================
# Warp + Crop + Deskew
# =========================
def warp_quad_with_pad(
    img: np.ndarray,
    quad_xy: np.ndarray,
    max_side: int,
    pad: int,
    border_value: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    rect = order_points(quad_xy)
    (tl, tr, br, bl) = rect

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    w = max(2, int(round(max(wA, wB))))
    h = max(2, int(round(max(hA, hB))))

    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w = max(2, int(round(w * scale)))
        h = max(2, int(round(h * scale)))

    pad = int(max(0, pad))
    out_w = w + 2 * pad
    out_h = h + 2 * pad

    dst = np.array(
        [[pad, pad], [pad + w - 1, pad], [pad + w - 1, pad + h - 1], [pad, pad + h - 1]],
        dtype=np.float32,
    )
    Hm = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        img,
        Hm,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
    return warped, Hm, (out_w, out_h)


def tight_crop_from_mask(img: np.ndarray, mask: np.ndarray, crop_margin: int) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], bool]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img, mask, None, True

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    m = int(max(0, crop_margin))
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w - 1, x2 + m)
    y2 = min(h - 1, y2 + m)

    return img[y1 : y2 + 1, x1 : x2 + 1].copy(), mask[y1 : y2 + 1, x1 : x2 + 1].copy(), [x1, y1, x2, y2], False


def estimate_deskew_angle_from_mask(mask: np.ndarray) -> float:
    """Return deskew rotation degrees (positive CCW). 0 means no rotation."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        return 0.0

    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h), angle)
    (w, h) = rect[1]
    angle = float(rect[2])  # [-90, 0)
    # OpenCV trick: make it represent rotation of the LONG side
    if w < h:
        angle = angle + 90.0
    # if angle is +5 => need rotate -5 to deskew
    rot = -angle
    # normalize to [-90,90]
    while rot > 90:
        rot -= 180
    while rot < -90:
        rot += 180
    return float(rot)


def rotate_affine_keep(mask_or_img: np.ndarray, rot_deg: float, border_value, interp) -> np.ndarray:
    h, w = mask_or_img.shape[:2]
    cx, cy = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        mask_or_img,
        M,
        (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def warp_and_crop(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    quad_xy: np.ndarray,
    warp_max_side: int,
    warp_pad: int,
    inset_ratio_for_warp: float,
    inset_ratio_for_clip: float,
    erode_px: int,
    crop_margin: int,
    final_deskew: bool,
    max_deskew_deg: float,
) -> Tuple[np.ndarray, Dict]:
    H, W = mask_u8.shape[:2]

    # ---- 1) inset quad (two levels) ----
    quad = order_points(quad_xy)
    quad_clip = quad_inset(quad, inset_ratio_for_clip) if inset_ratio_for_clip > 0 else quad
    quad_warp = quad_inset(quad, inset_ratio_for_warp) if inset_ratio_for_warp > 0 else quad

    # ---- 2) ✅ CLIP mask by quad BEFORE warping (core fix) ----
    qm = quad_to_mask(quad_clip, (H, W))
    mask_clip = cv2.bitwise_and(mask_u8, qm)

    # fill tiny holes (optional but helps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clip = cv2.morphologyEx(mask_clip, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---- 3) warp image + warped mask ----
    warped, Hm, (out_w, out_h) = warp_quad_with_pad(
        img_bgr, quad_warp, max_side=warp_max_side, pad=warp_pad, border_value=255
    )
    warped_mask = cv2.warpPerspective(
        mask_clip,
        Hm,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # erode to remove thin rim of background
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        warped_mask = cv2.erode(warped_mask, ker, iterations=1)

    # ---- 4) tight crop #1 ----
    crop1_img, crop1_mask, xyxy1, fail1 = tight_crop_from_mask(warped, warped_mask, crop_margin=crop_margin)

    deskew_deg = 0.0
    deskew_applied = False
    xyxy2 = None
    fail2 = None

    # ---- 5) final micro deskew ----
    if final_deskew and (not fail1):
        rot = estimate_deskew_angle_from_mask(crop1_mask)
        if abs(rot) >= 0.25 and abs(rot) <= float(max_deskew_deg):
            crop2_img = rotate_affine_keep(crop1_img, rot, border_value=(255, 255, 255), interp=cv2.INTER_LINEAR)
            crop2_mask = rotate_affine_keep(crop1_mask, rot, border_value=0, interp=cv2.INTER_NEAREST)
            deskew_deg = float(rot)
            deskew_applied = True

            crop3_img, crop3_mask, xyxy2, fail2 = tight_crop_from_mask(crop2_img, crop2_mask, crop_margin=crop_margin)
            info = {
                "warp_out_wh": [out_w, out_h],
                "tight_crop_xyxy_1": xyxy1,
                "mask_crop_fail_1": fail1,
                "deskew_deg": deskew_deg,
                "deskew_applied": deskew_applied,
                "tight_crop_xyxy_2": xyxy2,
                "mask_crop_fail_2": fail2,
            }
            return crop3_img, info

    info = {
        "warp_out_wh": [out_w, out_h],
        "tight_crop_xyxy_1": xyxy1,
        "mask_crop_fail_1": fail1,
        "deskew_deg": deskew_deg,
        "deskew_applied": deskew_applied,
        "tight_crop_xyxy_2": xyxy2,
        "mask_crop_fail_2": fail2,
    }
    return crop1_img, info


# =========================
# Classification: choose rotation that maximizes P(rot0)
# =========================
def cls_prob_rot0(cls_model: YOLO, img_bgr: np.ndarray, imgsz: int, device: Union[str, int]) -> Tuple[float, Dict]:
    r = cls_model.predict(source=img_bgr, imgsz=imgsz, device=device, verbose=False)[0]
    probs = r.probs
    if probs is None:
        return 0.0, {}
    p = probs.data.detach().cpu().numpy().astype(float)
    names = getattr(cls_model, "names", None) or getattr(r, "names", None) or {}
    d = {}
    for i, pi in enumerate(p):
        d[str(names.get(i, str(i)))] = float(pi)
    rot0_prob = d.get("rot0", None)
    if rot0_prob is None:
        rot0_prob = float(np.max(p))
    return float(rot0_prob), d


def upright_by_best_rot0(cls_model: YOLO, img_bgr: np.ndarray, imgsz: int, device: Union[str, int]) -> Tuple[np.ndarray, Dict]:
    best_k = 0
    best_p = -1.0
    best_probs = None
    for k in (0, 1, 2, 3):
        cand = rotate_exact_90(img_bgr, k)
        p_rot0, prob_dict = cls_prob_rot0(cls_model, cand, imgsz=imgsz, device=device)
        if p_rot0 > best_p:
            best_p = p_rot0
            best_k = k
            best_probs = prob_dict
    out = rotate_exact_90(img_bgr, best_k)
    return out, {"best_k90_ccw": int(best_k), "best_rot0_prob": float(best_p), "probs": best_probs or {}}


# =========================
# Config
# =========================
@dataclass
class Step5Config:
    source: Union[str, Path] = r"segmentation_classification/assets/seg_step1_test/images"

    seg_weights: str = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cls_weights: str = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"

    imgsz_seg: int = 960
    conf_seg: float = 0.25
    iou_seg: float = 0.5
    max_det: int = 20
    only_class_id: Optional[int] = 0
    sort_by_area_desc: bool = True

    # warp/crop tuning
    warp_max_side: int = 1600
    warp_pad: int = 18  # ✅ pad 一定要 > crop_margin，避免 bbox 被扩展到 0
    quad_inset_ratio_for_warp: float = 0.020
    quad_inset_ratio_for_clip: float = 0.010  # ✅ clip 用更小 inset，避免误切掉边缘
    mask_erode_px: int = 2
    crop_margin: int = 1

    refine_corner_max_dist: float = 28.0

    final_deskew: bool = True
    max_deskew_deg: float = 6.0

    imgsz_cls: int = 384
    device: Union[int, str] = 0

    out_dir: Union[str, Path] = r"segmentation_classification/outputs/step5_v4"
    save_debug_overlay: bool = True
    save_warped: bool = True
    save_upright: bool = True


def build_config() -> Step5Config:
    cfg = Step5Config()
    # ===== edit here =====
    cfg.source = r"segmentation_classification/assets/seg_step1_test/images"
    cfg.seg_weights = r"segmentation_classification/runs/step3_seg/weights/best.pt"
    cfg.cls_weights = r"segmentation_classification/runs/step4_upright_cls/weights/best.pt"
    cfg.out_dir = r"segmentation_classification/outputs/step5_v4"

    # border 太多：优先调这两个
    cfg.mask_erode_px = 3
    cfg.crop_margin = 0

    # 仍然轻微歪：优先调这个
    cfg.refine_corner_max_dist = 32.0
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
        extra = ""
        if c.get("deskew_applied", False):
            extra = f" deskew={c.get('deskew_deg', 0.0):.2f}"
        cv2.putText(
            out,
            f"{label} {prob:.2f}{extra}",
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
            meta = {"image": str(img_path), "orig_hw": [H, W], "num_cards": 0, "cards": []}
            json_dump(meta_dir / f"{img_path.stem}.json", meta)
            continue

        insts = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float).tolist()

            if cfg.only_class_id is not None and cls_id != int(cfg.only_class_id):
                continue

            poly = masks.xy[i]
            poly_np = np.array(poly, dtype=np.float32)
            area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            insts.append((area, i, cls_id, conf, xyxy, poly_np))

        if cfg.sort_by_area_desc:
            insts.sort(key=lambda x: x[0], reverse=True)

        for new_idx, (_, i, cls_id, conf, xyxy, poly_np) in enumerate(insts):
            mask_u8 = poly_to_mask(poly_np, (H, W))

            # quad from mask + refine
            try:
                quad, quad_method, refined = quad_from_mask(mask_u8, refine_max_dist=cfg.refine_corner_max_dist)
            except Exception:
                x1, y1, x2, y2 = xyxy
                quad = order_points(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
                quad_method = "bbox_fallback"
                refined = False

            warp_crop, crop_info = warp_and_crop(
                img_bgr=img,
                mask_u8=mask_u8,
                quad_xy=quad,
                warp_max_side=cfg.warp_max_side,
                warp_pad=cfg.warp_pad,
                inset_ratio_for_warp=cfg.quad_inset_ratio_for_warp,
                inset_ratio_for_clip=cfg.quad_inset_ratio_for_clip,
                erode_px=cfg.mask_erode_px,
                crop_margin=cfg.crop_margin,
                final_deskew=cfg.final_deskew,
                max_deskew_deg=cfg.max_deskew_deg,
            )

            upright_img, upright_info = upright_by_best_rot0(
                cls_model=cls_model,
                img_bgr=warp_crop,
                imgsz=cfg.imgsz_cls,
                device=cfg.device,
            )

            warp_path = warped_dir / f"{img_path.stem}__card{new_idx:02d}__warp.jpg"
            upright_path = upright_dir / f"{img_path.stem}__card{new_idx:02d}__upright.jpg"
            if cfg.save_warped:
                imwrite(warp_path, warp_crop)
            if cfg.save_upright:
                imwrite(upright_path, upright_img)

            cards_meta.append(
                {
                    "idx": int(new_idx),
                    "bbox_xyxy": xyxy,
                    "seg_conf": conf,
                    "seg_cls_id": int(cls_id),
                    "quad_xy_ordered": quad.astype(float).tolist(),
                    "quad_method": quad_method,
                    "quad_refined": bool(refined),
                    "warp_path": str(warp_path).replace("/", "\\"),
                    "upright_path": str(upright_path).replace("/", "\\"),
                    "warp_out_wh": crop_info.get("warp_out_wh"),
                    "tight_crop_xyxy_1": crop_info.get("tight_crop_xyxy_1"),
                    "mask_crop_fail_1": crop_info.get("mask_crop_fail_1"),
                    "deskew_deg": crop_info.get("deskew_deg"),
                    "deskew_applied": crop_info.get("deskew_applied"),
                    "tight_crop_xyxy_2": crop_info.get("tight_crop_xyxy_2"),
                    "mask_crop_fail_2": crop_info.get("mask_crop_fail_2"),
                    "upright_best_k90_ccw": upright_info.get("best_k90_ccw"),
                    "cls_rot0_prob": upright_info.get("best_rot0_prob"),
                    "cls_probs": upright_info.get("probs", {}),
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
