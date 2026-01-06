from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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


def iter_sources(source: Union[str, Path]) -> List[Path]:
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        files = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        return sorted(files)
    raise FileNotFoundError(source)


# =========================
# Geometry helpers
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 points to tl, tr, br, bl.
    pts: (4,2) array, the 4 corner points.
    """
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def expand_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    """Scale quad around its center, scale>1 expands."""
    if scale is None or scale <= 0:
        return quad
    c = quad.mean(axis=0, keepdims=True)
    return c + (quad - c) * scale


def warp_quad(img_bgr: np.ndarray, quad: np.ndarray, prefer_landscape: bool = False) -> np.ndarray:
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    W = max(2, int(round(max(wA, wB))))
    H = max(2, int(round(max(hA, hB))))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC)

    # Optional: enforce landscape (width >= height)
    if prefer_landscape and warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def mask_to_quad(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """mask -> largest contour -> minAreaRect -> 4 points."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    rect = cv2.minAreaRect(c)
    quad = cv2.boxPoints(rect).astype(np.float32)
    return quad


# =========================
# Config + Runner
# =========================
@dataclass
class PredictWarpConfig:
    weights: Union[str, Path]
    source: Union[str, Path]
    out_dir: Union[str, Path] = "outputs/step4_cards"

    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.7
    device: str = "0"

    retina_masks: bool = True
    min_area_ratio: float = 0.02
    expand: float = 1.03
    prefer_landscape: bool = False


def run_predict_and_warp(cfg: PredictWarpConfig) -> Dict[str, dict]:
    """
    Run segmentation inference -> quad -> warp -> export.
    Returns:
        dict mapping image_path -> meta dict (same as saved json).
    """
    weights = Path(cfg.weights)
    out_dir = Path(cfg.out_dir)

    out_cards = out_dir / "cards"
    out_debug = out_dir / "debug"
    out_meta = out_dir / "meta"
    out_cards.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    files = iter_sources(cfg.source)
    if not files:
        raise RuntimeError(f"No images found in {cfg.source}")

    all_meta: Dict[str, dict] = {}

    for img_path in files:
        results = model.predict(
            source=str(img_path),
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            verbose=False,
            retina_masks=cfg.retina_masks,
        )
        r = results[0]

        img = r.orig_img.copy() if getattr(r, "orig_img", None) is not None else imread_bgr(img_path)
        H, W = img.shape[:2]
        img_area = float(H * W)
        min_area = cfg.min_area_ratio * img_area

        if r.masks is None or r.boxes is None:
            print(f"[WARN] no masks/boxes: {img_path.name}")
            continue

        masks = r.masks.data
        masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)

        confs = r.boxes.conf
        confs = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)

        mh, mw = masks.shape[1], masks.shape[2]
        need_resize = (mh != H) or (mw != W)

        instances = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if need_resize:
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

            m_u8 = (m > 0.5).astype(np.uint8) * 255
            area = float(cv2.countNonZero(m_u8))
            if area < min_area:
                continue

            quad = mask_to_quad(m_u8)
            if quad is None:
                continue

            quad = expand_quad(quad, cfg.expand)
            quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)

            cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
            instances.append(
                {
                    "det_index": int(i),
                    "conf": float(confs[i]),
                    "quad": quad,
                    "cx": cx,
                    "cy": cy,
                }
            )

        if not instances:
            print(f"[WARN] no valid instances after filtering: {img_path.name}")
            continue

        # stable order: top-to-bottom then left-to-right
        instances.sort(key=lambda x: (x["cy"], x["cx"]))

        dbg = img.copy()
        meta = {"image": str(img_path), "cards": []}

        for k, inst in enumerate(instances):
            quad = inst["quad"]
            warped = warp_quad(img, quad, prefer_landscape=cfg.prefer_landscape)

            out_name = f"{img_path.stem}_card_{k:02d}.jpg"
            out_path = out_cards / out_name
            imwrite(out_path, warped)

            q_ordered = order_points(quad).astype(np.float32)
            qi = q_ordered.astype(np.int32)
            cv2.polylines(dbg, [qi], True, (0, 0, 255), 3)
            cv2.putText(
                dbg,
                f"{k} {inst['conf']:.2f}",
                (int(inst["cx"]), int(inst["cy"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            meta["cards"].append(
                {
                    "id": k,
                    "det_index": inst["det_index"],
                    "conf": inst["conf"],
                    "quad": q_ordered.tolist(),  # tl,tr,br,bl
                    "out_file": str(out_path),
                }
            )

        imwrite(out_debug / f"{img_path.stem}_debug.jpg", dbg)
        (out_meta / f"{img_path.stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        all_meta[str(img_path)] = meta
        print(f"[OK] {img_path.name}: {len(instances)} cards -> {out_cards}")

    print(f"\nDone. Outputs in: {Path(cfg.out_dir).resolve()}")
    return all_meta


# =========================
# Optional CLI wrapper
# =========================
def parse_args() -> PredictWarpConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs_local/seg_train/mix50_callback/weights/last.pt")
    ap.add_argument("--source", type=str, required=True, help="image file or folder")
    ap.add_argument("--out_dir", type=str, default="outputs/step4_cards")

    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="0")

    ap.add_argument("--retina_masks", action="store_true")
    ap.add_argument("--min_area_ratio", type=float, default=0.02)
    ap.add_argument("--expand", type=float, default=1.03)
    ap.add_argument("--prefer_landscape", action="store_true")
    args = ap.parse_args()

    # NOTE: 如果命令行不加 --retina_masks，则默认 False；而我们配置默认 True
    # 这里保持“命令行显式控制”：用户不传就用默认 True（更稳）
    retina = True if args.retina_masks else True

    return PredictWarpConfig(
        weights=args.weights,
        source=args.source,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=retina,
        min_area_ratio=args.min_area_ratio,
        expand=args.expand,
        prefer_landscape=args.prefer_landscape,
    )


if __name__ == "__main__":
    # ✅ 直接在代码里传参调用（不走命令行）
    cfg = PredictWarpConfig(
        weights="runs_local/seg_train/mix50_callback/weights/last.pt",

        # 单张图：
        source="data/try_business_card_dectection/images_all",
        # 或文件夹批量：
        # source="data/seg_synth_step1/images_all",

        out_dir="outputs/step4_cards_v2",

        imgsz=1024,
        conf=0.25,
        iou=0.85,          # 两张卡很近时可提高一点，减少互相抑制
        device="0",        # 没GPU改成 "cpu"

        retina_masks=True,
        min_area_ratio=0.02,
        expand=1.05,       # 裁切太紧就调大到 1.06~1.10
        prefer_landscape=True,
    )

    run_predict_and_warp(cfg)
