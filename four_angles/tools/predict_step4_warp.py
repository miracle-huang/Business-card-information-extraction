#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
four_angles/tools/predict_step4_warp.py

Step4:
- Use trained YOLO detection model (5 classes: card + 4 corners).
- For each detected card:
    - find 4 semantic corners (tl,tr,br,bl) inside/near the card bbox
    - use the CENTER of each corner bbox as the vertex point
    - compute perspective transform and warp to upright card crop
- Save:
    out_dir/warps/<image_stem>_cardXX.jpg
    out_dir/json/<image_stem>_cardXX.json
    out_dir/debug_vis/<image_stem>.jpg  (optional)

Classes (must match your dataset.yaml):
0: card
1: corner_tl
2: corner_tr
3: corner_br
4: corner_bl
"""

from __future__ import annotations

import json
import math
import sys
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
# ✅ model weights (改成你的 best.pt 路径)
MODEL_WEIGHTS = Path(r"four_angles/runs/step3_detect_routeA/weights/best.pt")
DEVICE = 0  # 0 / "cpu"

# ✅ input images (file or folder)
SOURCE = Path(r"four_angles/assets/step1_out_test_model/images")  # 也可以是单张图片路径

# ✅ output
OUT_DIR = Path(r"four_angles/outputs/step4_out")
SAVE_DEBUG_VIS = True

# ✅ predict params
IMGSZ = 1280
CONF_THRES_CARD = 0.25
CONF_THRES_CORNER = 0.25
IOU_NMS = 0.5
MAX_DET = 300

# ✅ association params
# corners must be inside expanded card bbox
CARD_EXPAND_RATIO = 0.08   # expand by ratio of card size
CARD_EXPAND_PX = 24        # and/or fixed pixels
DIST_ALPHA = 0.8           # score = conf - alpha*(dist/diag)

# ✅ warp sanity
MIN_WARP_W = 80
MIN_WARP_H = 50
MAX_WARP_EDGE = 2500       # clamp if super large

# If True, force output to landscape (width >= height) by rotating 90 if needed
FORCE_LANDSCAPE = False


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
    H_img, W_img = img_bgr.shape[:2]
    quad = quad.astype(np.float32)

    # size
    W, H = quad_size(quad)
    if W < MIN_WARP_W or H < MIN_WARP_H:
        return None

    # clamp insane
    W = min(W, MAX_WARP_EDGE)
    H = min(H, MAX_WARP_EDGE)

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)

    if FORCE_LANDSCAPE and warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# =========================
# Detection parsing + association
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


def select_best_corner(
    candidates: List[Det],
    expected_xy: Tuple[float, float],
    diag: float,
) -> Optional[Det]:
    if not candidates:
        return None
    ex, ey = expected_xy
    best = None
    best_score = -1e9
    for d in candidates:
        cx, cy = xyxy_center(d.xyxy)
        dist = math.hypot(cx - ex, cy - ey)
        score = d.conf - DIST_ALPHA * (dist / max(diag, 1e-6))
        if score > best_score:
            best_score = score
            best = d
    return best


def assign_corners_to_card(card: Det, corners_by_cls: Dict[int, List[Det]], W: int, H: int) -> Optional[Dict[int, Det]]:
    # expand card bbox for corner search
    card_exp = expand_xyxy(card.xyxy, CARD_EXPAND_RATIO, CARD_EXPAND_PX, W, H)
    x1, y1, x2, y2 = map(float, card.xyxy)
    diag = math.hypot(x2 - x1, y2 - y1)

    # expected anchors (based on card bbox)
    expected = {
        1: (x1, y1),  # TL
        2: (x2, y1),  # TR
        3: (x2, y2),  # BR
        4: (x1, y2),  # BL
    }

    chosen: Dict[int, Det] = {}
    for cid in (1, 2, 3, 4):
        pool = []
        for d in corners_by_cls.get(cid, []):
            if d.conf < CONF_THRES_CORNER:
                continue
            cx, cy = xyxy_center(d.xyxy)
            if point_in_xyxy(cx, cy, card_exp):
                pool.append(d)
        best = select_best_corner(pool, expected[cid], diag)
        if best is None:
            return None
        chosen[cid] = best

    return chosen


# =========================
# Debug drawing
# =========================
def draw_debug(img_bgr: np.ndarray, cards: List[Det], assigned: List[Optional[Dict[int, Det]]]) -> np.ndarray:
    vis = img_bgr.copy()

    for i, card in enumerate(cards):
        x1, y1, x2, y2 = card.xyxy
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"card{i} {card.conf:.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        sel = assigned[i]
        if not sel:
            cv2.putText(vis, "corners: MISSING", (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        # draw corner boxes and centers
        pts = []
        for cid, name in zip((1, 2, 3, 4), ("TL", "TR", "BR", "BL")):
            d = sel[cid]
            cx, cy = xyxy_center(d.xyxy)
            pts.append((cx, cy))
            xx1, yy1, xx2, yy2 = d.xyxy
            cv2.rectangle(vis, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (255, 0, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
            cv2.putText(vis, f"{name} {d.conf:.2f}", (int(xx1), max(0, int(yy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # draw quad
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

        # predict on numpy image (unicode path safe)
        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=min(CONF_THRES_CARD, CONF_THRES_CORNER),
            iou=IOU_NMS,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False,
        )
        res = results[0]
        dets = parse_ultralytics_result(res)

        # split by class
        cards = [d for d in dets if d.cls == 0 and d.conf >= CONF_THRES_CARD]
        corners_by_cls: Dict[int, List[Det]] = {1: [], 2: [], 3: [], 4: []}
        for d in dets:
            if d.cls in (1, 2, 3, 4):
                corners_by_cls[d.cls].append(d)

        # sort cards by conf (high first)
        cards.sort(key=lambda d: d.conf, reverse=True)

        assigned_all: List[Optional[Dict[int, Det]]] = []

        saved_any = False
        meta_out = {
            "image": str(img_path),
            "image_size": [W, H],
            "model": str(MODEL_WEIGHTS),
            "cards": [],
        }

        for i, card in enumerate(cards):
            sel = assign_corners_to_card(card, corners_by_cls, W, H)
            assigned_all.append(sel)
            if sel is None:
                meta_out["cards"].append({
                    "index": i,
                    "status": "missing_corners",
                    "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                })
                continue

            # corner points = centers of corner bboxes (semantic order)
            tl = xyxy_center(sel[1].xyxy)
            tr = xyxy_center(sel[2].xyxy)
            br = xyxy_center(sel[3].xyxy)
            bl = xyxy_center(sel[4].xyxy)
            quad = np.array([tl, tr, br, bl], dtype=np.float32)

            warped = warp_from_quad(img, quad)
            if warped is None:
                meta_out["cards"].append({
                    "index": i,
                    "status": "warp_failed",
                    "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                    "quad": quad.tolist(),
                })
                continue

            out_img = out_warps / f"{img_path.stem}_card{i:02d}.jpg"
            imwrite(out_img, warped)
            saved_any = True

            meta_out["cards"].append({
                "index": i,
                "status": "ok",
                "card": {"xyxy": card.xyxy.tolist(), "conf": card.conf},
                "corners": {
                    "tl": {"xyxy": sel[1].xyxy.tolist(), "conf": sel[1].conf},
                    "tr": {"xyxy": sel[2].xyxy.tolist(), "conf": sel[2].conf},
                    "br": {"xyxy": sel[3].xyxy.tolist(), "conf": sel[3].conf},
                    "bl": {"xyxy": sel[4].xyxy.tolist(), "conf": sel[4].conf},
                },
                "quad_points": quad.tolist(),
                "warp_size": [int(warped.shape[1]), int(warped.shape[0])],
                "warp_path": str(out_img),
            })

        # save json even if none saved (for debugging)
        (out_json / f"{img_path.stem}.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if SAVE_DEBUG_VIS:
            dbg = draw_debug(img, cards, assigned_all)
            imwrite(out_dbg / f"{img_path.stem}.jpg", dbg)

        print(f"[OK] {img_path.name} -> cards={len(cards)} saved_warps={sum(1 for c in meta_out['cards'] if c.get('status')=='ok')}")

    print(f"\nDone. Outputs:")
    print(f"  warps    : {out_warps.resolve()}")
    print(f"  json     : {out_json.resolve()}")
    if SAVE_DEBUG_VIS:
        print(f"  debug_vis: {out_dbg.resolve()}")


if __name__ == "__main__":
    main()