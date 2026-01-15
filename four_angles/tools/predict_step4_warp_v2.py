#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
four_angles/tools/predict_step4_warp_v2.py

Step4 improved (no retrain):
1) Two-pass inference:
   - pass A: detect cards only with higher conf, stricter iou
   - pass B: detect corners only with lower conf, looser iou
2) Corner association is CLASS-AGNOSTIC:
   - ignore predicted corner cls (TL/TR/BR/BL may be confused)
   - choose 4 corner points by geometry (nearest to expected card corners)
3) Extra card de-dup (custom NMS) to reduce repeated/false card boxes.

Assumed classes (dataset.yaml order):
0: card
1: corner_tl
2: corner_tr
3: corner_br
4: corner_bl
"""

from __future__ import annotations

import json
import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# CONFIG (edit here)
# =========================
MODEL_WEIGHTS = Path(r"four_angles/runs/step3_detect_routeA/weights/best.pt")
DEVICE = 0  # 0 / "cpu"

SOURCE = Path(r"four_angles/assets/step1_out_test_model/images")  # folder or file
OUT_DIR = Path(r"four_angles/outputs/step4_out_v2")
SAVE_DEBUG_VIS = True

# model input size
IMGSZ = 1280
MAX_DET = 1000

# ---- Pass A (cards) ----
CARD_CONF = 0.30
CARD_IOU = 0.40
CARD_MIN_AREA_RATIO = 0.01  # filter tiny false cards, area >= ratio * image_area
CARD_NMS_IOU = 0.50         # custom NMS for cards (reduce duplicates)

# ---- Pass B (corners) ----
CORNER_CONF = 0.05
CORNER_IOU = 0.70

# ---- Association ----
CARD_EXPAND_RATIO = 0.18    # larger than before (more tolerant)
CARD_EXPAND_PX = 80

# cost = dist/diag - GAMMA*conf  (lower is better)
GAMMA = 0.6
TOPK_PER_CORNER = 6         # take topK candidates for each expected corner, brute-force 6^4=1296

# ---- Warp sanity ----
MIN_WARP_W = 80
MIN_WARP_H = 50
MAX_WARP_EDGE = 2500
FORCE_LANDSCAPE = False

# class names for debug
NAMES = ["card", "TL", "TR", "BR", "BL"]


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


def iter_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    raise FileNotFoundError(source)


# =========================
# Geometry helpers
# =========================
def xyxy_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def expand_xyxy(xyxy: np.ndarray, ratio: float, px: float, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, xyxy)
    w = x2 - x1
    h = y2 - y1
    ex = max(px, ratio * w)
    ey = max(px, ratio * h)
    x1 -= ex
    y1 -= ey
    x2 += ex
    y2 += ey
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W - 1.0, x2))
    y2 = max(0.0, min(H - 1.0, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def point_in_xyxy(x: float, y: float, xyxy: np.ndarray) -> bool:
    x1, y1, x2, y2 = map(float, xyxy)
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def quad_size(quad: np.ndarray) -> Tuple[int, int]:
    # quad: tl,tr,br,bl
    tl, tr, br, bl = quad.astype(np.float32)
    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))
    W = int(round(max(wA, wB)))
    H = int(round(max(hA, hB)))
    return W, H


def warp_from_quad(img_bgr: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    quad = quad.astype(np.float32)
    W, H = quad_size(quad)
    if W < MIN_WARP_W or H < MIN_WARP_H:
        return None
    W = min(W, MAX_WARP_EDGE)
    H = min(H, MAX_WARP_EDGE)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
    if FORCE_LANDSCAPE and warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


# =========================
# Detection
# =========================
@dataclass
class Det:
    cls: int
    conf: float
    xyxy: np.ndarray  # (4,)


def parse_ultralytics_result(res) -> List[Det]:
    dets: List[Det] = []
    if res.boxes is None:
        return dets
    b = res.boxes
    xyxy = b.xyxy.detach().cpu().numpy()
    cls = b.cls.detach().cpu().numpy().astype(int)
    conf = b.conf.detach().cpu().numpy()
    for i in range(len(xyxy)):
        dets.append(Det(cls=int(cls[i]), conf=float(conf[i]), xyxy=xyxy[i].astype(np.float32)))
    return dets


def predict_dets(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    iou: float,
    classes: Optional[List[int]] = None,
) -> List[Det]:
    """Call model.predict safely across ultralytics versions."""
    try:
        results = model.predict(
            source=img_bgr,
            imgsz=IMGSZ,
            conf=conf,
            iou=iou,
            max_det=MAX_DET,
            device=DEVICE,
            classes=classes,
            verbose=False,
        )
        dets = parse_ultralytics_result(results[0])
        if classes is not None:
            dets = [d for d in dets if d.cls in set(classes)]
        return dets
    except TypeError:
        # fallback if `classes` not supported
        results = model.predict(
            source=img_bgr,
            imgsz=IMGSZ,
            conf=conf,
            iou=iou,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False,
        )
        dets = parse_ultralytics_result(results[0])
        if classes is not None:
            dets = [d for d in dets if d.cls in set(classes)]
        return dets


# =========================
# Card postprocess
# =========================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-9))


def nms_dets(dets: List[Det], thr: float) -> List[Det]:
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Det] = []
    for d in dets:
        ok = True
        for k in keep:
            if iou_xyxy(d.xyxy, k.xyxy) > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def filter_small_cards(cards: List[Det], W: int, H: int, min_ratio: float) -> List[Det]:
    out = []
    img_area = float(W * H)
    for c in cards:
        x1, y1, x2, y2 = map(float, c.xyxy)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area >= min_ratio * img_area:
            out.append(c)
    return out


# =========================
# Corner association (class-agnostic)
# =========================
def assoc_corners_class_agnostic(card: Det, corner_dets: List[Det], W: int, H: int) -> Optional[Dict[str, Det]]:
    """
    Choose 4 corner detections around this card, ignoring their cls.
    Return mapping: {"tl":Det,"tr":Det,"br":Det,"bl":Det}
    """
    card_exp = expand_xyxy(card.xyxy, CARD_EXPAND_RATIO, CARD_EXPAND_PX, W, H)
    x1, y1, x2, y2 = map(float, card.xyxy)
    diag = math.hypot(x2 - x1, y2 - y1) + 1e-6

    expected = {
        "tl": (x1, y1),
        "tr": (x2, y1),
        "br": (x2, y2),
        "bl": (x1, y2),
    }
    keys = ["tl", "tr", "br", "bl"]

    # collect candidates near card (by center-in-expanded-bbox) with conf threshold
    cands: List[Det] = []
    for d in corner_dets:
        if d.conf < CORNER_CONF:
            continue
        cx, cy = xyxy_center(d.xyxy)
        if point_in_xyxy(cx, cy, card_exp):
            cands.append(d)

    if len(cands) < 4:
        return None

    # for each expected corner, take topK candidates by cost
    topk_lists: Dict[str, List[Tuple[float, Det]]] = {}
    for k in keys:
        ex, ey = expected[k]
        scored: List[Tuple[float, Det]] = []
        for d in cands:
            cx, cy = xyxy_center(d.xyxy)
            dist = math.hypot(cx - ex, cy - ey) / diag
            cost = dist - GAMMA * d.conf
            scored.append((cost, d))
        scored.sort(key=lambda t: t[0])
        topk_lists[k] = scored[:TOPK_PER_CORNER]
        if not topk_lists[k]:
            return None

    # brute-force distinct assignment (<= 6^4)
    best_cost = 1e9
    best = None
    for a_cost, a in topk_lists["tl"]:
        for b_cost, b in topk_lists["tr"]:
            if b is a:
                continue
            for c_cost, c in topk_lists["br"]:
                if c is a or c is b:
                    continue
                for d_cost, d in topk_lists["bl"]:
                    if d is a or d is b or d is c:
                        continue
                    total = a_cost + b_cost + c_cost + d_cost
                    if total < best_cost:
                        best_cost = total
                        best = {"tl": a, "tr": b, "br": c, "bl": d}

    return best


# =========================
# Debug drawing
# =========================
def draw_debug(img_bgr: np.ndarray, cards: List[Det], corners: List[Det], assigned: List[Optional[Dict[str, Det]]]) -> np.ndarray:
    vis = img_bgr.copy()

    # draw cards
    for i, card in enumerate(cards):
        x1, y1, x2, y2 = card.xyxy
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"card{i} {card.conf:.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        sel = assigned[i]
        if not sel:
            cv2.putText(vis, "corners: MISSING", (int(x1), int(y2) + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue

        # selected corners (show predicted cls to observe confusion)
        pts = []
        for key, name in [("tl", "TL"), ("tr", "TR"), ("br", "BR"), ("bl", "BL")]:
            d = sel[key]
            cx, cy = xyxy_center(d.xyxy)
            pts.append((cx, cy))
            xx1, yy1, xx2, yy2 = d.xyxy
            cv2.rectangle(vis, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (255, 0, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
            pred_name = NAMES[d.cls] if 0 <= d.cls < len(NAMES) else str(d.cls)
            cv2.putText(vis, f"{name}<-{pred_name} {d.conf:.2f}", (int(xx1), max(0, int(yy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        quad = np.array(pts, dtype=np.float32).reshape(4, 2)
        cv2.polylines(vis, [quad.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    return vis


# =========================
# Main
# =========================
def main():
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"MODEL_WEIGHTS not found: {MODEL_WEIGHTS}")

    imgs = iter_images(SOURCE)
    if not imgs:
        raise FileNotFoundError(f"No images found in: {SOURCE}")

    out_warps = OUT_DIR / "warps"
    out_json = OUT_DIR / "json"
    out_dbg = OUT_DIR / "debug_vis"
    out_warps.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_VIS:
        out_dbg.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_WEIGHTS))

    for img_path in imgs:
        img = imread_bgr(img_path)
        H, W = img.shape[:2]

        # ---- Pass A: cards only ----
        cards = predict_dets(model, img, conf=CARD_CONF, iou=CARD_IOU, classes=[0])
        cards = [c for c in cards if c.cls == 0 and c.conf >= CARD_CONF]
        cards = filter_small_cards(cards, W, H, CARD_MIN_AREA_RATIO)
        cards = nms_dets(cards, CARD_NMS_IOU)
        cards.sort(key=lambda d: d.conf, reverse=True)

        # ---- Pass B: corners only ----
        corner_dets = predict_dets(model, img, conf=CORNER_CONF, iou=CORNER_IOU, classes=[1, 2, 3, 4])
        corner_dets = [d for d in corner_dets if d.cls in (1, 2, 3, 4) and d.conf >= CORNER_CONF]

        assigned_all: List[Optional[Dict[str, Det]]] = []
        meta_out = {
            "image": str(img_path),
            "image_size": [W, H],
            "model": str(MODEL_WEIGHTS),
            "passA": {"card_conf": CARD_CONF, "card_iou": CARD_IOU, "card_nms_iou": CARD_NMS_IOU},
            "passB": {"corner_conf": CORNER_CONF, "corner_iou": CORNER_IOU},
            "cards": [],
        }

        ok_count = 0
        for i, card in enumerate(cards):
            sel = assoc_corners_class_agnostic(card, corner_dets, W, H)
            assigned_all.append(sel)

            if sel is None:
                meta_out["cards"].append({
                    "index": i,
                    "status": "missing_corners",
                    "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                })
                continue

            tl = xyxy_center(sel["tl"].xyxy)
            tr = xyxy_center(sel["tr"].xyxy)
            br = xyxy_center(sel["br"].xyxy)
            bl = xyxy_center(sel["bl"].xyxy)
            quad = np.array([tl, tr, br, bl], dtype=np.float32)

            warped = warp_from_quad(img, quad)
            if warped is None:
                meta_out["cards"].append({
                    "index": i,
                    "status": "warp_failed",
                    "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                    "quad_points": quad.tolist(),
                })
                continue

            out_img = out_warps / f"{img_path.stem}_card{i:02d}.jpg"
            imwrite(out_img, warped)
            ok_count += 1

            meta_out["cards"].append({
                "index": i,
                "status": "ok",
                "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                "selected_corners": {
                    "tl": {"cls": int(sel["tl"].cls), "conf": float(sel["tl"].conf), "xyxy": sel["tl"].xyxy.tolist()},
                    "tr": {"cls": int(sel["tr"].cls), "conf": float(sel["tr"].conf), "xyxy": sel["tr"].xyxy.tolist()},
                    "br": {"cls": int(sel["br"].cls), "conf": float(sel["br"].conf), "xyxy": sel["br"].xyxy.tolist()},
                    "bl": {"cls": int(sel["bl"].cls), "conf": float(sel["bl"].conf), "xyxy": sel["bl"].xyxy.tolist()},
                },
                "quad_points": quad.tolist(),
                "warp_size": [int(warped.shape[1]), int(warped.shape[0])],
                "warp_path": str(out_img),
            })

        # save json
        (out_json / f"{img_path.stem}.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # debug vis
        if SAVE_DEBUG_VIS:
            dbg = draw_debug(img, cards, corner_dets, assigned_all)
            imwrite(out_dbg / f"{img_path.stem}.jpg", dbg)

        print(f"[OK] {img_path.name} cards={len(cards)} warp_ok={ok_count} corners_total={len(corner_dets)}")

    print("\nDone.")
    print(f"  warps    : {out_warps.resolve()}")
    print(f"  json     : {out_json.resolve()}")
    if SAVE_DEBUG_VIS:
        print(f"  debug_vis: {out_dbg.resolve()}")


if __name__ == "__main__":
    main()
