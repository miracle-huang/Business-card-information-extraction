from __future__ import annotations

import argparse
import json
import sys
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


def expand_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    """Scale quad around its center, scale>1 expands."""
    if scale is None or scale <= 0:
        return quad
    c = quad.mean(axis=0, keepdims=True)
    return c + (quad - c) * scale


def quad_to_warp_matrix(quad: np.ndarray) -> tuple[np.ndarray, int, int]:
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
    return M, W, H


def mask_to_quad(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """mask -> largest contour -> minAreaRect -> 4 points."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    rect = cv2.minAreaRect(c)
    quad = cv2.boxPoints(rect).astype(np.float32)
    return quad


# =========================
# Mask post-process (fix "eroded content")
# =========================
def solidify_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    让mask变“实心”：
    1) close 连接边缘
    2) floodFill 填洞
    """
    m = (mask_u8 > 0).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    h, w = m.shape[:2]
    ff = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    m_filled = cv2.bitwise_or(m, holes)
    return m_filled


def crop_by_warped_mask(
    warped_img: np.ndarray,
    warped_mask_u8: np.ndarray,
    margin: int = 8,
    fill_bg_white: bool = True,
    fill_dilate_px: int = 2,
) -> np.ndarray:
    """
    Tight crop warped_img using warped_mask_u8 (0/255).
    - solidify mask (close+fill holes)
    - crop to bbox of mask
    - fill outside mask with white, using DILATED mask to avoid wiping card content
    """
    m = solidify_mask(warped_mask_u8)

    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return warped_img

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(warped_img.shape[1] - 1, x2 + margin)
    y2 = min(warped_img.shape[0] - 1, y2 + margin)

    cropped = warped_img[y1 : y2 + 1, x1 : x2 + 1]
    cropped_mask = m[y1 : y2 + 1, x1 : x2 + 1]

    if not fill_bg_white:
        return cropped

    # ✅ 用膨胀 mask 来决定“名片区域”，避免把名片内容误涂白
    if fill_dilate_px > 0:
        k2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (fill_dilate_px * 2 + 1, fill_dilate_px * 2 + 1),
        )
        fill_mask = cv2.dilate(cropped_mask, k2, iterations=1)
    else:
        fill_mask = cropped_mask

    out = cropped.copy()
    out[fill_mask == 0] = 255
    return out


# =========================
# OCR-based upright (fix wrong rotation)
# =========================
def try_build_paddle_ocr(lang: str = "japan", use_angle_cls: bool = False):
    """
    用于“判方向”的OCR：必须 use_angle_cls=False
    否则 PaddleOCR 自己会转正文字，导致 90/180/270 分数差距很小 -> 乱转。
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None

    try:
        ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)
    except TypeError:
        ocr = PaddleOCR(lang=lang)
    return ocr


def ocr_score_paddle(ocr, img_bgr: np.ndarray) -> float:
    """
    OCR 分数：conf * len(text) 累加
    分数越高，越像“正确方向的文字”
    """
    if ocr is None:
        return 0.0

    # 适当缩放加速
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side > 960:
        scale = 960.0 / max_side
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        raw = ocr.ocr(img_bgr)
    except Exception:
        return 0.0

    if not raw:
        return 0.0

    # 有的版本外层多包一层
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]

    score = 0.0
    for item in raw:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                tc = item[1]
                if isinstance(tc, (list, tuple)) and len(tc) >= 2:
                    text = str(tc[0])
                    conf = float(tc[1])
                    if conf < 0.6:
                        continue
                    score += conf * max(1, len(text))
        except Exception:
            continue

    return float(score)


def best_upright_rotation_4way(
    img_bgr: np.ndarray,
    ocr=None,
    min_abs_score: float = 6.0,
    min_ratio: float = 1.15,
) -> tuple[np.ndarray, int]:
    """
    在 0/90/180/270 之间选最佳方向。
    保护机制：
    - best 分数太低：不转（避免瞎转）
    - best/second 不够大：不转（避免横版变竖版）
    """
    if ocr is None:
        return img_bgr, 0

    candidates = [
        (0, img_bgr),
        (90, cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img_bgr, cv2.ROTATE_180)),
        (270, cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    scored = []
    for deg, im in candidates:
        s = ocr_score_paddle(ocr, im)
        scored.append((s, deg, im))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_s, best_deg, best_img = scored[0]
    second_s = scored[1][0]

    if best_s < min_abs_score:
        return img_bgr, 0

    if second_s > 0 and best_s / (second_s + 1e-6) < min_ratio:
        return img_bgr, 0

    return best_img, best_deg


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

    # 背景控制
    expand: float = 1.01  # 1.00~1.02 推荐，太大会带背景
    crop_by_mask: bool = True
    crop_margin: int = 8
    fill_bg_white: bool = True
    fill_dilate_px: int = 2  # 2~3 常用；0 表示不膨胀

    # 方向控制
    auto_upright: bool = True
    ocr_lang: str = "japan"
    upright_min_abs_score: float = 4.0
    upright_min_ratio: float = 1.05

    # 可选：你不建议在 upright 前做这个（容易冲突），默认 False
    prefer_landscape: bool = False


def run_predict_and_warp(cfg: PredictWarpConfig) -> Dict[str, dict]:
    weights = Path(cfg.weights)
    out_dir = Path(cfg.out_dir)

    out_cards = out_dir / "cards"
    out_debug = out_dir / "debug"
    out_meta = out_dir / "meta"
    out_cards.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    # ✅ 判方向 OCR：use_angle_cls=False
    ocr = None
    if cfg.auto_upright:
        ocr = try_build_paddle_ocr(lang=cfg.ocr_lang, use_angle_cls=False)
        if ocr is None:
            print("[WARN] PaddleOCR not available. auto_upright will be skipped.")

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
                    "mask_u8": m_u8,
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
            M, WW, HH = quad_to_warp_matrix(quad)

            warped = cv2.warpPerspective(img, M, (WW, HH), flags=cv2.INTER_CUBIC)
            warped_mask = cv2.warpPerspective(inst["mask_u8"], M, (WW, HH), flags=cv2.INTER_NEAREST)

            if cfg.crop_by_mask:
                warped = crop_by_warped_mask(
                    warped_img=warped,
                    warped_mask_u8=warped_mask,
                    margin=cfg.crop_margin,
                    fill_bg_white=cfg.fill_bg_white,
                    fill_dilate_px=cfg.fill_dilate_px,
                )

            # （可选）统一横向：不建议开，容易和 upright 冲突
            if cfg.prefer_landscape and warped.shape[0] > warped.shape[1]:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            rot_deg = 0
            if cfg.auto_upright and ocr is not None:
                warped, rot_deg = best_upright_rotation_4way(
                    warped,
                    ocr=ocr,
                    min_abs_score=cfg.upright_min_abs_score,
                    min_ratio=cfg.upright_min_ratio,
                )

            out_name = f"{img_path.stem}_card_{k:02d}.jpg"
            out_path = out_cards / out_name
            imwrite(out_path, warped)

            # debug：画四点（在原图上）
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
                    "upright_rot_deg": rot_deg,  # 0/90/180/270
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
# CLI
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

    ap.add_argument("--expand", type=float, default=1.01)
    ap.add_argument("--no_crop_by_mask", action="store_true")
    ap.add_argument("--crop_margin", type=int, default=8)
    ap.add_argument("--no_fill_bg_white", action="store_true")
    ap.add_argument("--fill_dilate_px", type=int, default=2)

    ap.add_argument("--no_auto_upright", action="store_true")
    ap.add_argument("--ocr_lang", type=str, default="japan")
    ap.add_argument("--upright_min_abs_score", type=float, default=4.0)
    ap.add_argument("--upright_min_ratio", type=float, default=1.05)

    ap.add_argument("--prefer_landscape", action="store_true")

    args = ap.parse_args()

    return PredictWarpConfig(
        weights=args.weights,
        source=args.source,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=True if args.retina_masks else True,  # 推荐总开
        min_area_ratio=args.min_area_ratio,
        expand=args.expand,
        crop_by_mask=not args.no_crop_by_mask,
        crop_margin=args.crop_margin,
        fill_bg_white=not args.no_fill_bg_white,
        fill_dilate_px=args.fill_dilate_px,
        auto_upright=not args.no_auto_upright,
        ocr_lang=args.ocr_lang,
        upright_min_abs_score=args.upright_min_abs_score,
        upright_min_ratio=args.upright_min_ratio,
        prefer_landscape=args.prefer_landscape,
    )


def default_cfg() -> PredictWarpConfig:
    # ✅ 你自己改默认路径即可
    return PredictWarpConfig(
        weights="runs_local/seg_train/mix50_callback/weights/last.pt",
        source="data/try_business_card_dectection/images_all",
        out_dir="outputs/step4_cards_final",
        imgsz=1024,
        conf=0.25,
        iou=0.85,
        device="0",
        retina_masks=True,
        min_area_ratio=0.02,
        expand=1.01,
        crop_by_mask=True,
        crop_margin=8,
        fill_bg_white=True,
        fill_dilate_px=2,
        auto_upright=True,
        ocr_lang="japan",
        upright_min_abs_score=6.0,
        upright_min_ratio=1.15,
        prefer_landscape=False,
    )


if __name__ == "__main__":
    # 三态：
    #   USE_CLI = True  -> 强制走命令行
    #   USE_CLI = False -> 强制走默认 cfg（代码写死）
    #   USE_CLI = None  -> 自动：有命令行参数就走 CLI，否则走默认
    USE_CLI = None

    if USE_CLI is True:
        cfg = parse_args()
    elif USE_CLI is False:
        cfg = default_cfg()
    else:
        cfg = parse_args() if len(sys.argv) > 1 else default_cfg()

    run_predict_and_warp(cfg)
