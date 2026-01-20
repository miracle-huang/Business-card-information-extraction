from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# ✅ ensure project root on sys.path
# =========================
ROOT = Path(__file__).resolve().parents[2]  # script/obb -> script -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


# =========================
# Geometry / warp
# =========================
def quad_area(pts: np.ndarray) -> float:
    """Polygon area by shoelace (pts: (4,2))."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_self_intersecting_quad(pts: np.ndarray) -> bool:
    """Check if quad edges intersect (simple check for 4 points in given order)."""
    def seg_intersect(a, b, c, d) -> bool:
        # Proper intersection for segments ab and cd
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    p0, p1, p2, p3 = pts
    # edges: (0-1,1-2,2-3,3-0). Non-adjacent pairs: (0-1) with (2-3), (1-2) with (3-0)
    return seg_intersect(p0, p1, p2, p3) or seg_intersect(p1, p2, p3, p0)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Fallback geometric ordering: tl, tr, br, bl."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def warp_by_semantic_kpts(
    img_bgr: np.ndarray,
    kpts_px: np.ndarray,  # (4,2) order: TL,TR,BR,BL
    border_value: Tuple[int, int, int] = (255, 255, 255),
    min_side: int = 64,
    max_side: int = 4096,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Warp perspective using semantic corners TL,TR,BR,BL.
    Return (crop_bgr, meta).
    """
    H, W = img_bgr.shape[:2]
    pts = kpts_px.astype(np.float32)

    # If the predicted order produces a crossing quad, fallback to geometric ordering
    used_fallback = False
    if is_self_intersecting_quad(pts):
        pts = order_points(pts)
        used_fallback = True

    tl, tr, br, bl = pts

    wA = float(np.linalg.norm(br - bl))
    wB = float(np.linalg.norm(tr - tl))
    hA = float(np.linalg.norm(tr - br))
    hB = float(np.linalg.norm(tl - bl))

    out_w = int(round(max(wA, wB)))
    out_h = int(round(max(hA, hB)))

    out_w = int(np.clip(out_w, min_side, max_side))
    out_h = int(np.clip(out_h, min_side, max_side))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(
        img_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    meta = {
        "out_w": out_w,
        "out_h": out_h,
        "used_fallback_order_points": used_fallback,
    }
    return warped, meta


def draw_viz(img_bgr: np.ndarray, quads: List[np.ndarray], confs: List[float]) -> np.ndarray:
    out = img_bgr.copy()
    for i, (q, c) in enumerate(zip(quads, confs)):
        q = q.astype(int)
        cv2.polylines(out, [q], isClosed=True, color=(0, 255, 0), thickness=2)
        for j, (x, y) in enumerate(q):
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(out, str(j), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(out, f"card{i:02d} conf={c:.2f}", (q[0, 0], max(15, q[0, 1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return out


# =========================
# Config (edit here)
# =========================
@dataclass
class PredictWarpConfig:
    # model
    model_path: Path = Path(r"runs_kpt/kpt_hybrid_no_aug/weights/best.pt")  # ←改成你的 best.pt 真实路径
    device: str = "0"  # "cpu" or "0"
    imgsz: int = 640

    # input/output
    input_dir: Path = Path(r"data/test_images_obb/images")  # ←改成你的待处理图片文件夹
    out_dir: Path = Path(r"outputs/kpt_warp")

    # thresholds
    conf: float = 0.25
    iou: float = 0.5
    kpt_conf: float = 0.30          # 每个关键点最低置信度（无 conf 时跳过此过滤）
    min_quad_area: float = 2000.0   # 最小四边形面积（像素^2），防止极小误检

    # output options
    save_viz: bool = True
    save_json: bool = True
    max_cards_per_image: int = 20   # 防止极端误检爆输出


CFG = PredictWarpConfig()


# =========================
# Prediction extraction (version-tolerant)
# =========================
def _get_boxes_xyxy_and_conf(result) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (xyxy Nx4, conf N).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)
    if hasattr(conf, "cpu"):
        conf = conf.cpu().numpy()
    else:
        conf = np.asarray(conf)
    return xyxy.astype(np.float32), conf.astype(np.float32)


def _get_keypoints_xy_and_conf(result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return:
      kpts_xy: (N,K,2) in pixel
      kpts_conf: (N,K) or None if not available
    """
    k = getattr(result, "keypoints", None)
    if k is None:
        return None, None

    # xy
    xy = getattr(k, "xy", None)
    if xy is None:
        # some versions use .data with shape (N,K,3)
        data = getattr(k, "data", None)
        if data is None:
            return None, None
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        else:
            data = np.asarray(data)
        if data.ndim == 3 and data.shape[2] >= 2:
            xy = data[:, :, :2]
        else:
            return None, None
    else:
        if hasattr(xy, "cpu"):
            xy = xy.cpu().numpy()
        else:
            xy = np.asarray(xy)

    # conf (optional)
    cf = getattr(k, "conf", None)
    if cf is not None:
        if hasattr(cf, "cpu"):
            cf = cf.cpu().numpy()
        else:
            cf = np.asarray(cf)

    return xy.astype(np.float32), (cf.astype(np.float32) if cf is not None else None)


# =========================
# Main
# =========================
def main():
    if not CFG.model_path.exists():
        raise FileNotFoundError(f"model not found: {CFG.model_path}")
    if not CFG.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {CFG.input_dir}")

    out_crops = CFG.out_dir / "crops"
    out_viz = CFG.out_dir / "viz"
    out_json = CFG.out_dir / "json"
    out_crops.mkdir(parents=True, exist_ok=True)
    if CFG.save_viz:
        out_viz.mkdir(parents=True, exist_ok=True)
    if CFG.save_json:
        out_json.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(CFG.model_path))

    img_paths = list_images(CFG.input_dir)
    if not img_paths:
        print(f"[predict] no images found in: {CFG.input_dir}")
        return

    print(f"[predict] model={CFG.model_path}")
    print(f"[predict] input={CFG.input_dir}  images={len(img_paths)}")
    print(f"[predict] out={CFG.out_dir}")

    for idx, img_path in enumerate(img_paths, start=1):
        img_bgr = imread_bgr(img_path)
        H, W = img_bgr.shape[:2]

        # predict (pass numpy to avoid Chinese-path issues)
        results = model.predict(
            source=img_bgr,
            imgsz=CFG.imgsz,
            conf=CFG.conf,
            iou=CFG.iou,
            device=CFG.device,
            verbose=False,
        )

        result = results[0]
        boxes_xyxy, boxes_conf = _get_boxes_xyxy_and_conf(result)
        kpts_xy, kpts_conf = _get_keypoints_xy_and_conf(result)

        record: Dict[str, Any] = {
            "image": str(img_path),
            "width": W,
            "height": H,
            "cards": [],
        }

        quads_for_viz: List[np.ndarray] = []
        confs_for_viz: List[float] = []

        if kpts_xy is None or kpts_xy.shape[0] == 0:
            # no detections
            if CFG.save_viz:
                viz = img_bgr
                imwrite(out_viz / f"{img_path.stem}.jpg", viz)
            if CFG.save_json:
                (out_json / f"{img_path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{idx}/{len(img_paths)}] {img_path.name}: no detections")
            continue

        # sort instances by box conf desc (if available)
        order = np.argsort(-boxes_conf) if boxes_conf.size == kpts_xy.shape[0] else np.arange(kpts_xy.shape[0])

        saved = 0
        for inst_id in order:
            if saved >= CFG.max_cards_per_image:
                break

            quad = kpts_xy[inst_id]  # (K,2)
            if quad.shape[0] < 4:
                continue

            quad4 = quad[:4].astype(np.float32)  # TL,TR,BR,BL (as trained)
            area = quad_area(quad4)
            if area < CFG.min_quad_area:
                continue

            # bounds check (allow small tolerance)
            if (quad4[:, 0].min() < -5) or (quad4[:, 1].min() < -5) or (quad4[:, 0].max() > W + 5) or (quad4[:, 1].max() > H + 5):
                continue

            # kpt conf filtering (if available)
            if kpts_conf is not None:
                c4 = kpts_conf[inst_id][:4]
                if float(np.min(c4)) < float(CFG.kpt_conf):
                    continue
                kpt_conf_list = [float(x) for x in c4.tolist()]
            else:
                kpt_conf_list = None

            box_conf = float(boxes_conf[inst_id]) if boxes_conf.size == kpts_xy.shape[0] else None
            box_xyxy = boxes_xyxy[inst_id].tolist() if boxes_xyxy.size == kpts_xy.shape[0] * 4 else None

            crop, warp_meta = warp_by_semantic_kpts(img_bgr, quad4)

            crop_name = f"{img_path.stem}_card{saved:02d}.jpg"
            crop_path = out_crops / crop_name
            imwrite(crop_path, crop)

            record["cards"].append(
                {
                    "id": int(saved),
                    "box_conf": box_conf,
                    "box_xyxy": box_xyxy,
                    "kpts_px_tltrbrbl": [[float(x), float(y)] for x, y in quad4.tolist()],
                    "kpts_conf": kpt_conf_list,
                    "quad_area": float(area),
                    "warp": warp_meta,
                    "crop": str(crop_path),
                }
            )

            quads_for_viz.append(quad4)
            confs_for_viz.append(box_conf if box_conf is not None else 0.0)
            saved += 1

        # save viz/json
        if CFG.save_viz:
            viz = draw_viz(img_bgr, quads_for_viz, confs_for_viz) if quads_for_viz else img_bgr
            imwrite(out_viz / f"{img_path.stem}.jpg", viz)

        if CFG.save_json:
            (out_json / f"{img_path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{idx}/{len(img_paths)}] {img_path.name}: saved {saved} crop(s)")

    print("[predict] done.")


if __name__ == "__main__":
    main()
