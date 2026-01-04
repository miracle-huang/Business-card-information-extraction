from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class SynthConfig:
    bg_dir: Path
    card_dir: Path
    out_dir: Path

    # 总共生成多少张
    num_images: int = 60

    # 输出画布大小（越大越清晰，但越费存储/训练）
    out_w: int = 1536
    out_h: int = 1536

    # 每张图随机放置名片数量范围
    min_cards: int = 2
    max_cards: int = 4

    # ✅ 名片缩放：按“长边一致”缩放（适配横/竖版），保证信息清晰
    # 建议先固定：min=max，例如 760；想让不同图尺寸略随机可给一个范围
    min_card_long: int = 760
    max_card_long: int = 760

    # ✅ 同一张图内是否所有名片都用同一个尺寸（长边一致）
    same_size_in_image: bool = True

    # ✅ 不做透视，仅旋转
    angle_min: float = 0.0
    angle_max: float = 360.0

    # 名片与名片之间的最小间隙（像素），用 mask 膨胀实现（建议 6~12）
    min_gap_px: int = 12

    # 距离画布边缘留白（防止贴边）
    margin: int = 20

    # 单张名片放置尝试次数（找不到不重叠位置就算这张放置失败）
    max_tries_per_card: int = 250

    # ✅ 整张图重试次数：如果最终放置数量 < min_cards（或 must_place_all 不满足），就整张重生成
    max_image_tries: int = 30

    # ✅ 是否强制放满本图目标张数（例如抽到 4 张就必须放满 4 张）
    # False: 只要最终 >= min_cards 就算成功（推荐）
    must_place_all: bool = False

    # ✅ 目标张数采样偏向：2/3/4 的权重（越大越容易被选中）
    # 默认让 4 明显更多
    target_count_weights: Tuple[float, float, float] = (1.0, 2.0, 8.0)  # for [2,3,4]

    # ✅ 如果“目标是4但没放满”，是否更倾向整张重试
    prefer_full_target: bool = True

    # ✅ 当目标=4 但只放到2/3时，整张重试的概率（越大越容易重试直到放满）
    retry_prob_if_underfilled: dict = field(default_factory=lambda: {3: 0.25, 4: 0.80})

    # ✅ 目标=4 时，最低也要放到多少张才接受（避免目标=4却只生成2张）
    min_accept_when_target4: int = 3

    seed: int | None = 42
    save_debug: bool = False  # 是否输出 debug 可视化


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def imread_bgr(path: Path) -> np.ndarray:
    # 支持中文路径
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imread_rgba(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img


def imwrite(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def resize_and_random_crop(bg: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """把背景缩放到至少覆盖(out_w,out_h)，再随机裁剪到固定大小。"""
    H, W = bg.shape[:2]
    scale = max(out_w / W, out_h / H)
    if scale > 1.0:
        nw, nh = int(math.ceil(W * scale)), int(math.ceil(H * scale))
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_CUBIC)

    H, W = bg.shape[:2]
    if W == out_w and H == out_h:
        return bg

    x0 = random.randint(0, max(0, W - out_w))
    y0 = random.randint(0, max(0, H - out_h))
    return bg[y0:y0 + out_h, x0:x0 + out_w].copy()


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """把4点排序为 tl, tr, br, bl（用于稳定输出 polygon）"""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def quad_to_yolo_seg_line(quad_xy: np.ndarray, W: int, H: int, cls: int = 0) -> str:
    quad = order_points_tl_tr_br_bl(quad_xy.copy())
    quad[:, 0] = np.clip(quad[:, 0], 0, W - 1) / float(W)
    quad[:, 1] = np.clip(quad[:, 1], 0, H - 1) / float(H)
    nums = [str(cls)] + [f"{v:.6f}" for v in quad.reshape(-1).tolist()]
    return " ".join(nums)


def alpha_blend(bg_bgr: np.ndarray, fg_bgra: np.ndarray, x: int, y: int) -> None:
    """把 fg_bgra 贴到 bg_bgr 的 (x,y) 位置（就地修改 bg_bgr）"""
    H, W = bg_bgr.shape[:2]
    fh, fw = fg_bgra.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + fw), min(H, y + fh)
    if x1 <= x0 or y1 <= y0:
        return

    roi = bg_bgr[y0:y1, x0:x1].astype(np.float32)
    fg = fg_bgra[y0 - y:y1 - y, x0 - x:x1 - x, :3].astype(np.float32)
    a = fg_bgra[y0 - y:y1 - y, x0 - x:x1 - x, 3:4].astype(np.float32) / 255.0

    out = fg * a + roi * (1.0 - a)
    bg_bgr[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


def rotated_card_and_local_quad(card_bgra: np.ndarray, target_long: int, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      - 旋转后的 card_bgra（expand=True）
      - 在旋转图坐标系下的四点 quad（4x2）
    说明：
      - 先按“长边一致”缩放，再旋转
      - quad 用 alpha mask 的 minAreaRect 精确拟合
    """
    ch, cw = card_bgra.shape[:2]
    long_side = max(cw, ch)
    scale = target_long / float(long_side)

    nw = max(2, int(round(cw * scale)))
    nh = max(2, int(round(ch * scale)))
    card = cv2.resize(card_bgra, (nw, nh), interpolation=cv2.INTER_CUBIC)

    h, w = card.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # expand
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(card, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    alpha = rotated[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("alpha mask has no contour")
    c = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    quad = cv2.boxPoints(rect).astype(np.float32)  # 4x2 in rotated coords
    return rotated, quad


def make_dilate_kernel(gap_px: int) -> np.ndarray | None:
    if gap_px <= 0:
        return None
    k = gap_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def generate_one_image(
    bg_paths: List[Path],
    card_paths: List[Path],
    cfg: SynthConfig,
    out_img_path: Path,
    out_lbl_path: Path,
    out_dbg_path: Path | None = None,
) -> None:
    """
    生成单张合成图：
    - 如果最终放置数量 < cfg.min_cards（或 must_place_all 不满足），整张图重试
    - 因此不会出现只有1张的情况（除非 min_cards=1）
    """
    for attempt in range(cfg.max_image_tries):
        # 1) 背景 -> 固定尺寸
        bg_path = random.choice(bg_paths)
        bg = imread_bgr(bg_path)
        bg = resize_and_random_crop(bg, cfg.out_w, cfg.out_h)

        H, W = bg.shape[:2]
        occupied = np.zeros((H, W), dtype=np.uint8)
        kernel = make_dilate_kernel(cfg.min_gap_px)

        # 2) 本张图目标数量
        # ✅ 目标数量采样：偏向 4
        candidates = list(range(cfg.min_cards, cfg.max_cards + 1))  # e.g. [2,3,4]
        base = {2: cfg.target_count_weights[0], 3: cfg.target_count_weights[1], 4: cfg.target_count_weights[2]}
        weights = [base.get(k, 1.0) for k in candidates]
        n_cards_target = random.choices(candidates, weights=weights, k=1)[0]
        chosen = random.sample(card_paths, n_cards_target)

        # 3) 同一张图内统一尺寸（每次整张重试会重新采样 base_long）
        base_long = random.randint(cfg.min_card_long, cfg.max_card_long) if cfg.same_size_in_image else None

        quads_global: List[np.ndarray] = []
        label_lines: List[str] = []

        # 4) 逐张放置（放不下的卡先“跳过”，最终由数量判断是否整张重试）
        for cp in chosen:
            card_bgra = imread_rgba(cp)

            placed = False
            for _ in range(cfg.max_tries_per_card):
                target_long = base_long if base_long is not None else random.randint(cfg.min_card_long, cfg.max_card_long)
                angle = random.uniform(cfg.angle_min, cfg.angle_max)

                rot_bgra, quad_local = rotated_card_and_local_quad(card_bgra, target_long=target_long, angle_deg=angle)
                rh, rw = rot_bgra.shape[:2]

                # 位置范围（保证在画布内 + margin）
                x_min = cfg.margin
                y_min = cfg.margin
                x_max = W - cfg.margin - rw
                y_max = H - cfg.margin - rh
                if x_max < x_min or y_max < y_min:
                    continue

                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)

                cand = (rot_bgra[:, :, 3] > 0).astype(np.uint8)

                occ = occupied
                if kernel is not None:
                    occ = cv2.dilate(occupied, kernel)

                roi_occ = occ[y:y + rh, x:x + rw]
                if np.any((roi_occ > 0) & (cand > 0)):
                    continue

                # 接受：贴图 + 更新占用
                alpha_blend(bg, rot_bgra, x, y)
                occupied[y:y + rh, x:x + rw] = (occupied[y:y + rh, x:x + rw] | cand).astype(np.uint8)

                quad_global = quad_local + np.array([x, y], dtype=np.float32)
                quads_global.append(quad_global)
                label_lines.append(quad_to_yolo_seg_line(quad_global, W, H, cls=0))

                placed = True
                break

            if not placed:
                # 这张卡放不下：先跳过，后面由 placed_n 判定是否整张重试
                pass

        # 5) 判断是否满足数量要求
        placed_n = len(label_lines)

        # 基础要求：至少放够 min_cards
        ok = placed_n >= cfg.min_cards

        # 强制放满（你现在不想用这个，所以保持 False 即可）
        if cfg.must_place_all:
            ok = ok and (placed_n == n_cards_target)

        # ✅ 不强制放满，但“更偏向放满目标数”
        if ok and (not cfg.must_place_all) and cfg.prefer_full_target:
            # 目标=4时，至少放到3才接受（否则整张重试）
            if n_cards_target == 4 and placed_n < cfg.min_accept_when_target4:
                ok = False
            # 若没达到目标张数，则按概率整张重试（提高放满概率）
            elif placed_n < n_cards_target:
                p = cfg.retry_prob_if_underfilled.get(n_cards_target, 0.0)
                if random.random() < p:
                    ok = False

        if not ok:
            # 整张重试
            continue

        # 6) 保存输出
        imwrite(out_img_path, bg)
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        if cfg.save_debug and out_dbg_path is not None:
            dbg = bg.copy()
            for i, q in enumerate(quads_global):
                qi = q.astype(np.int32)
                cv2.polylines(dbg, [qi], True, (0, 0, 255), 3)
                cx, cy = int(q[:, 0].mean()), int(q[:, 1].mean())
                cv2.putText(dbg, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            imwrite(out_dbg_path, dbg)

        return  # ✅ 成功生成就退出

    # 如果多次重试仍失败
    raise RuntimeError(
        f"Failed to generate a valid image after {cfg.max_image_tries} attempts. "
        f"Try: smaller min_card_long/max_card_long, larger out_w/out_h, smaller min_gap_px, "
        f"or set must_place_all=False."
    )


def generate_dataset(cfg: SynthConfig) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    bg_paths = list_images(cfg.bg_dir)
    card_paths = list_images(cfg.card_dir)

    if not bg_paths:
        raise FileNotFoundError(f"No backgrounds in: {cfg.bg_dir}")
    if len(card_paths) < cfg.max_cards:
        raise FileNotFoundError(f"Not enough cards in: {cfg.card_dir} (need >= {cfg.max_cards})")

    images_dir = cfg.out_dir / "images_all"
    labels_dir = cfg.out_dir / "labels_all"
    debug_dir = cfg.out_dir / "debug" if cfg.save_debug else None

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i in range(cfg.num_images):
        img_path = images_dir / f"img_{i:05d}.jpg"
        lbl_path = labels_dir / f"img_{i:05d}.txt"
        dbg_path = (debug_dir / f"img_{i:05d}.jpg") if debug_dir else None

        generate_one_image(
            bg_paths=bg_paths,
            card_paths=card_paths,
            cfg=cfg,
            out_img_path=img_path,
            out_lbl_path=lbl_path,
            out_dbg_path=dbg_path,
        )

    (cfg.out_dir / "README_step1.txt").write_text(
        "Step1 synthesis done.\n"
        f"Generated: {cfg.num_images}\n"
        f"Images: {images_dir}\n"
        f"Labels: {labels_dir}\n"
        "Label format: cls + 4-point polygon (normalized, tl-tr-br-bl).\n"
        f"min_cards={cfg.min_cards}, max_cards={cfg.max_cards}, must_place_all={cfg.must_place_all}\n"
        f"same_size_in_image={cfg.same_size_in_image}, min_card_long={cfg.min_card_long}, max_card_long={cfg.max_card_long}\n"
        f"out_w={cfg.out_w}, out_h={cfg.out_h}, min_gap_px={cfg.min_gap_px}\n",
        encoding="utf-8",
    )
