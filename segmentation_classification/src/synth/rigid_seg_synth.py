#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rigid-only segmentation synth engine (Step1)
- NO augmentation
- NO perspective / distortion (warpAffine only)
- card size kept consistent (fixed_card_w), rotation only
- NO overlap (with optional min gap)
- background size NOT fixed: keep original unless target is 3/4, then enlarge background dynamically
- Windows-friendly multiprocessing (spawn)

YOLO-seg label format (polygon):
  cls x1 y1 x2 y2 x3 y3 x4 y4   (normalized, 4-corner polygon)
"""

from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import multiprocessing as mp

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# Config
# =========================
@dataclass
class SynthConfig:
    bg_dir: Path
    card_dir: Path
    out_dir: Path

    num_images: int = 500

    # output size: if None -> keep background original size
    out_w: Optional[int] = None
    out_h: Optional[int] = None

    # cards per image
    min_cards: int = 2
    max_cards: int = 4

    # distance to image border (pixels)
    margin_to_img: int = 90

    # strict no overlap. If you want a visible gap, set >0 (pixels).
    min_gap_between_cards: int = 40

    # fixed card width in pixels AFTER resizing (keeps size consistent)
    fixed_card_w: int = 700

    # full rotation range
    angle_min: float = 0.0
    angle_max: float = 360.0

    # placement attempts
    max_place_trials_per_card: int = 160
    max_image_retries: int = 60

    # debug visualization
    save_debug: bool = True

    # multiprocessing
    num_workers: int = max(1, (os.cpu_count() or 8) - 1)

    seed: int = 42

    # ---------- New: target count weighting (increase 3/4 ratio) ----------
    # for {2,3,4} when min_cards=2 max_cards=4
    weight_2: float = 1.0
    weight_3: float = 3.0
    weight_4: float = 4.0

    # ---------- New: dynamic background enlarge for 3/4 cards ----------
    dynamic_bg_enlarge: bool = True
    dynamic_bg_only_for_3plus: bool = True
    max_bg_scale: float = 3.0  # cap to avoid too huge images; set larger if you want


# =========================
# IO helpers (Chinese path OK)
# =========================
def imread_any(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8) # 用 NumPy 读取图像的二进制字节流
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) # 用 OpenCV 解码成图像矩阵 IMREAD_UNCHANGED保留原始格式
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() # path.suffix 返回文件扩展名
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def list_images(folder: Path) -> List[Path]: # 递归遍历 folder 目录下所有子目录 
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]) # 选出所有图片文件
    # 按路径排序
    # 返回排序后的 Path 列表


def ensure_clean_dir(p: Path) -> None:
    # 如果目录存在，先删除
    if p.exists():
        shutil.rmtree(p)
    # 再创建目录
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Geometry helpers 几何辅助函数
# =========================
def bbox_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    # 从一组二维点坐标中，计算其 轴对齐外接矩形（axis-aligned bounding box, AABB）
    xs = pts[:, 0] # 取所有点的 x 坐标
    ys = pts[:, 1] # 取所有点的 y 坐标
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

# 从一个浮点 bbox（xyxy 格式）出发，加 padding，并裁剪到图像范围内，生成一个可用于 numpy 切片的整数 ROI。
def compute_roi_from_bbox(
    bbox_xyxy: Tuple[float, float, float, float], # (x0, y0, x1, y1)
    img_w: int, # 图像宽度
    img_h: int, # 图像高度
    pad: int, # padding, 扩张像素
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_xyxy
    rx0 = int(math.floor(x0 - pad))
    ry0 = int(math.floor(y0 - pad))
    rx1 = int(math.ceil(x1 + pad))
    ry1 = int(math.ceil(y1 + pad))

    # clamp for slicing [ry0:ry1, rx0:rx1], so upper bound can be img_w/img_h
    # 把坐标值强制限制在图像合法边界之内，避免越界（out-of-bounds）
    rx0 = max(0, min(img_w, rx0))
    ry0 = max(0, min(img_h, ry0))
    rx1 = max(0, min(img_w, rx1))
    ry1 = max(0, min(img_h, ry1))

    if rx1 - rx0 <= 1 or ry1 - ry0 <= 1:
        return None
    return rx0, ry0, rx1, ry1

# 判断两个 AABB（Axis-Aligned Bounding Box，轴对齐矩形）是否相交
def aabb_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return False
    return True

# 使用仿射变换矩阵 M 对一组二维点 pts 进行批量坐标变换。
# 用于计算名片四角在旋转+平移后的新位置
def affine_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # M: 2x3 仿射变换矩阵
    # pts: (N, 2) 坐标点
    pts = pts.astype(np.float32) # OpenCV 默认使用 float32
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (M @ pts_h.T).T
    return out.astype(np.float32) # 返回变换后的坐标点

# 生成一张“卡片图像”的四个角点坐标（按 TL, TR, BR, BL 顺序），用于几何变换或标注同步。
# card_corners → 描述整张卡片
# ROI → 描述卡片在背景中的位置
def card_corners(Wc: int, Hc: int) -> np.ndarray:
    """
    TL, TR, BR, BL corners in pixel coordinates.
    Use W-1/H-1 to avoid off-by-one that makes bounds tighter than necessary.
    """
    w1 = float(max(0, Wc - 1))
    h1 = float(max(0, Hc - 1))
    return np.array([[0.0, 0.0], [w1, 0.0], [w1, h1], [0.0, h1]], dtype=np.float32)

# 判断一个四边形（通常是名片四角）是否完全位于图像内部，并且距离图像边界至少 margin 像素
def corners_within_margin(quad: np.ndarray, img_w: int, img_h: int, margin: int) -> bool:
    #quad: np.ndarray   # shape = (4, 2)，四个角点
    #img_w: int         # 图像宽度
    #img_h: int         # 图像高度
    #margin: int        # 安全边距
    if np.any(quad[:, 0] < margin) or np.any(quad[:, 0] > img_w - 1 - margin):
        return False
    if np.any(quad[:, 1] < margin) or np.any(quad[:, 1] > img_h - 1 - margin):
        return False
    return True

# 构建仿射变换矩阵（旋转+平移）
def build_affine_rotation_translation(Wc: int, Hc: int, angle_deg: float, center_bg: Tuple[float, float]) -> np.ndarray:
    """
    Rotation around card center (scale=1.0), then translate to background center (cx_bg, cy_bg).
    """
    cx_bg, cy_bg = center_bg
    center_card = ((Wc - 1) / 2.0, (Hc - 1) / 2.0)
    # 旋转：以名片中心为圆心，旋转 angle_deg 度
    M = cv2.getRotationMatrix2D(center_card, angle_deg, 1.0)  # scale=1.0 (no resize aug)
    # 平移：将旋转后的图像移动到背景中心
    M[0, 2] += cx_bg - center_card[0]
    M[1, 2] += cy_bg - center_card[1]
    return M.astype(np.float32)

# 把一个四边形（quad）转换为 YOLO segmentation 标注格式的一行字符串（归一化坐标）
# 把几何四角转换为 YOLO 训练可用格式
def yolo_seg_line(class_id: int, quad: np.ndarray, img_w: int, img_h: int) -> Optional[str]:
    """
    YOLO-seg label line:
      cls x1 y1 x2 y2 x3 y3 x4 y4  (normalized)
    """
    q = quad.astype(np.float32).copy()
    q[:, 0] = np.clip(q[:, 0], 0, img_w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, img_h - 1)

    xmin, ymin, xmax, ymax = bbox_from_points(q)
    if (xmax - xmin) <= 2.0 or (ymax - ymin) <= 2.0:
        return None

    coords = []
    for x, y in q:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(coords)


# =========================
# Card cache
# =========================
@dataclass
class CardCacheItem:
    # 缓存加载和预处理后的名片数据，避免重复读取
    bgr: np.ndarray # BGR 图像
    mask_u8: np.ndarray  # 掩膜 0~255
    H: int # 高度
    W: int # 宽度
    diag: float # 对角线长度

# 将名片缩放到固定宽度 fixed_w，保持宽高比
def resize_card_to_fixed_width_keep_alpha(card: np.ndarray, fixed_w: int) -> np.ndarray:
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    h, w = card.shape[:2]
    if w <= 0 or w == fixed_w:
        return card

    scale = fixed_w / float(w)
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(card, (fixed_w, new_h), interpolation=interp)
    return resized


def get_card_cached(cache: Dict[str, CardCacheItem], card_path: str, cfg: SynthConfig) -> CardCacheItem:
    """
    Cache stores fixed-size cards (fixed_card_w). We do NOT downscale cards anymore
    because we will enlarge background dynamically for 3/4 cards.
    读取一张名片图 → 统一到固定宽度 → 拆出 BGR 与 mask → 计算一些几何属性 → 放入缓存并复用。
    """
    # 1) 缓存命中直接返回
    if card_path in cache:
        return cache[card_path]

    # 2) 读取并缩放到固定宽度
    raw = imread_any(Path(card_path))
    resized = resize_card_to_fixed_width_keep_alpha(raw, cfg.fixed_card_w)

    # 3) 如果读出来是灰度图 (H,W)，转成BGR (H,W,3)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    # 4) 拆分 BGR 与 mask（优先用 alpha）
    if resized.ndim == 3 and resized.shape[2] == 4:
        bgr = resized[:, :, :3].copy()
        alpha = resized[:, :, 3].copy()
        mask_u8 = alpha
    else:
        bgr = resized[:, :, :3].copy()
        mask_u8 = np.full((bgr.shape[0], bgr.shape[1]), 255, dtype=np.uint8)

    # 5) 计算尺寸与对角线长度
    Hc, Wc = bgr.shape[:2]
    diag = float(math.sqrt(Wc * Wc + Hc * Hc))

    # 6) 打包成 CardCacheItem，写入缓存并返回
    item = CardCacheItem(bgr=bgr, mask_u8=mask_u8, H=Hc, W=Wc, diag=diag)
    cache[card_path] = item
    return item


# =========================
# ROI warp/composite ROI 仿射变换与合成
# =========================

'''
在 ROI 局部区域内 对名片做仿射变换
用同一个 2×3 仿射矩阵 M_roi，
把 卡片的颜色图（BGR） 和 卡片的 mask（u8） 变换到一个指定大小的 ROI 坐标系中，
得到“变换后的卡片贴图”和“变换后的有效区域”。
'''
def warp_affine_roi(card_bgr: np.ndarray, card_mask_u8: np.ndarray, M_roi: np.ndarray, roi_w: int, roi_h: int):
    warp_bgr = cv2.warpAffine(
        card_bgr,  # 输入图像
        M_roi,  # 2×3 仿射变换矩阵
        (roi_w, roi_h),  # 输出图像尺寸
        flags=cv2.INTER_LINEAR,  # 插值方式
        borderMode=cv2.BORDER_CONSTANT,  # 边界模式
        borderValue=(0, 0, 0),  # 边界填充值
    )
    warp_mask = cv2.warpAffine(
        card_mask_u8,  # 输入掩膜
        M_roi,  # 同一个仿射矩阵
        (roi_w, roi_h),  # 同一个输出尺寸
        flags=cv2.INTER_NEAREST,  # mask 用最近邻插值（保持 0/255）
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warp_bgr, warp_mask

'''
Alpha blending（α 混合）: 按照透明度（alpha）把前景图像叠加到背景图像上
'''
def composite_roi_inplace(bg_roi: np.ndarray, fg_roi: np.ndarray, mask_u8: np.ndarray) -> None:
    alpha = (mask_u8.astype(np.float32) / 255.0)[:, :, None] # 归一化到 0~1，并扩展为 3 通道
    out = bg_roi.astype(np.float32) * (1.0 - alpha) + fg_roi.astype(np.float32) * alpha # alpha 混合
    bg_roi[:] = np.clip(out, 0, 255).astype(np.uint8) # 把数组中的所有元素限制在指定区间内


# =========================
# Debug
# =========================
# 在图像副本上画出每张名片的四边形轮廓（绿色）和角点（红色圆点）
def draw_debug(img_bgr: np.ndarray, quads: List[np.ndarray]) -> np.ndarray:
    vis = img_bgr.copy()
    for quad in quads:
        q = quad.astype(np.int32)
        cv2.polylines(vis, [q], isClosed=True, color=(0, 255, 0), thickness=2)
        for (x, y) in q:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
    return vis


# =========================
# New: choose target number with weights
# =========================
# 根据配置中的权重，从 [min_cards, max_cards] 范围内随机采样本张图要放多少张卡片。
def sample_num_cards(rng: random.Random, cfg: SynthConfig) -> int:
    choices = list(range(cfg.min_cards, cfg.max_cards + 1))
    weights = []
    for c in choices:
        if c == 2:
            weights.append(float(cfg.weight_2))
        elif c == 3:
            weights.append(float(cfg.weight_3))
        elif c == 4:
            weights.append(float(cfg.weight_4))
        else:
            weights.append(1.0)
    return rng.choices(choices, weights=weights, k=1)[0]


# =========================
# New: enlarge background if needed for 3/4 cards
# =========================
# 动态背景放大
def required_canvas_for_target(
    max_diag: float,
    target_n: int,
    margin: int,
    gap: int,
) -> Tuple[float, float]:
    """
    Conservative feasibility estimate using "circle packing" approximation.
    Let each card have radius r = max_diag/2 (covers any rotation).

    centers spacing >= 2r + gap
    boundary constraint: center at least (margin + r) away from borders

    For target 2: layout 1x2 (rows=1, cols=2)
    For target 3/4: layout 2x2 (rows=2, cols=2)  (conservative)
    """
    r = max_diag / 2.0 # 用圆来近似卡片，半径为 max_diag（对角线）/2

    # 根据目标卡片数量选择布局
    if target_n <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2

    # 计算需要的背景尺寸
    req_w = 2.0 * (margin + r) + (cols - 1) * (2.0 * r + gap)
    req_h = 2.0 * (margin + r) + (rows - 1) * (2.0 * r + gap)
    return req_w, req_h

# 根据当前要放的卡片数量和尺寸，必要时动态放大背景图，以确保卡片能安全排布（不重叠、不贴边）
def maybe_enlarge_background(
    bg_bgr: np.ndarray,
    cfg: SynthConfig,
    target_n: int,
    chosen_cards: List[CardCacheItem],
) -> np.ndarray:

    # 配置开关控制
    if not cfg.dynamic_bg_enlarge:
        return bg_bgr
    if cfg.dynamic_bg_only_for_3plus and target_n < 3:
        return bg_bgr
    if not chosen_cards:
        return bg_bgr

    h, w = bg_bgr.shape[:2]
    max_diag = max(ci.diag for ci in chosen_cards)

    req_w, req_h = required_canvas_for_target(
        max_diag=max_diag,
        target_n=target_n,
        margin=int(cfg.margin_to_img),
        gap=int(cfg.min_gap_between_cards),
    )

    s = max(req_w / float(w), req_h / float(h), 1.0) # 计算理论所需放大比例
    s = min(s, float(cfg.max_bg_scale)) # 限制最大放大比例
    if s <= 1.0 + 1e-6:
        return bg_bgr

    # round：四舍五入
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    # upscale => INTER_LINEAR
    bg_big = cv2.resize(bg_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return bg_big


# =========================
# Single image synthesis
# =========================
def synth_one_image(
    cfg: SynthConfig,
    bg_paths: List[str],
    card_paths: List[str],
    idx: int,
    out_img_path: Path,
    out_lbl_path: Path,
    out_dbg_path: Optional[Path],
    card_cache: Dict[str, CardCacheItem],
) -> bool:
    """
    For each image, we:
      1) sample target number (weighted, prefer 3/4)
      2) pre-select card templates
      3) load background (keep original size unless cfg.out_w/out_h set)
      4) if target is 3/4, enlarge background dynamically if needed
      5) place cards sequentially (each card has its own placement trials)
    """
    for retry in range(cfg.max_image_retries):
        rng = random.Random(cfg.seed + idx * 1000003 + retry * 99991)

        # ---- choose target & cards for this image (preselect) ----
        target_n = sample_num_cards(rng, cfg)
        # 1. 从名片库中随机选出 target_n 张名片路径
        if len(card_paths) >= target_n:
            chosen_paths = [rng.choice(card_paths) for _ in range(target_n)]
        else:
            chosen_paths = [rng.choice(card_paths) for _ in range(target_n)]

        # 通过前面的缓存机制，获取真实的名片数据（图像、Mask、长宽对角线等）
        chosen_items: List[CardCacheItem] = [get_card_cached(card_cache, p, cfg) for p in chosen_paths]

        # ---- background ----
        # 2. 随机选一张背景图
        bg_path = Path(rng.choice(bg_paths))
        bg_raw = imread_any(bg_path)

        # 如果背景图是灰度的，转成 BGR
        if bg_raw.ndim == 2:
            bg_raw = cv2.cvtColor(bg_raw, cv2.COLOR_GRAY2BGR)
        bg_bgr = bg_raw[:, :, :3] if (bg_raw.ndim == 3 and bg_raw.shape[2] == 4) else bg_raw

        # optional fixed output size (you can set out_w/out_h=None to NOT fix)
        # （可选）如果配置文件写死了背景输出尺寸，先强行 Resize
        if cfg.out_w is not None and cfg.out_h is not None:
            bg_bgr = cv2.resize(bg_bgr, (int(cfg.out_w), int(cfg.out_h)), interpolation=cv2.INTER_AREA)

        # dynamic enlarge for 3/4 cards (when out_w/out_h is None, this makes size adaptive)
        # 【核心功能】：如果抽到了 3、4 张名片，原始背景可能放不下，触发智能动态拉伸
        bg_bgr = maybe_enlarge_background(bg_bgr, cfg, target_n=target_n, chosen_cards=chosen_items)

        img_h, img_w = bg_bgr.shape[:2]
        # 3. 初始化全黑的 Mask（用于记录名片占用的区域）
        occ = np.zeros((img_h, img_w), dtype=np.uint8)

        pad = int(cfg.min_gap_between_cards)
        kernel = None
        if pad > 0:
            # 构造一个膨胀卷积核，后面对名片掩膜做膨胀，变相制造 gap 的安全区
            k = pad * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        placed_boxes_expanded: List[Tuple[float, float, float, float]] = []
        labels: List[str] = [] # 记录 YOLO 字符串标注
        quads_dbg: List[np.ndarray] = [] # 记录四个角（用于画 debug 可视化）

        # ---- place sequentially, so target distribution is controlled ----
        # 逐张摆放名片
        placed = 0
        for card_item in chosen_items:
            Wc, Hc = card_item.W, card_item.H
            if Wc < 10 or Hc < 10:
                continue

            # quick feasibility for single card under this canvas
            # 极速可行性检测：算一下这张名片的半径，如果只放这一张名片，加上 margin 都塞不进背景图像，直接判为死局
            radius = 0.5 * math.sqrt(float(Wc * Wc + Hc * Hc))
            x_min = cfg.margin_to_img + radius
            x_max = img_w - cfg.margin_to_img - radius
            y_min = cfg.margin_to_img + radius
            y_max = img_h - cfg.margin_to_img - radius
            if x_max <= x_min or y_max <= y_min:
                # this background too small even after enlarge
                placed = -999
                break

            success_this = False
            # 每张名片有最多 max_place_trials_per_card 次（默认160次）机会去天上随机掉落找位置
            for _try in range(cfg.max_place_trials_per_card):
                angle = rng.uniform(cfg.angle_min, cfg.angle_max)
                cx = rng.uniform(x_min, x_max)
                cy = rng.uniform(y_min, y_max)

                M = build_affine_rotation_translation(Wc, Hc, angle, (cx, cy))
                quad = affine_transform_points(M, card_corners(Wc, Hc))  # TL,TR,BR,BL

                # 1. 检查所有的四个角有没有越过安全边距（margin）
                if not corners_within_margin(quad, img_w, img_h, cfg.margin_to_img):
                    continue

                # 2. 算出一个能包住这张旋转名片并且加上排斥 pad 边距的正矩形框（AABB大框）
                xmin, ymin, xmax, ymax = bbox_from_points(quad)
                exp_bbox = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

                # 3. 粗筛碰撞检测：用这个极其简单的正矩形框，去和以前贴好名片的矩形框们检测撞没撞
                hit = False
                for b in placed_boxes_expanded:
                    if aabb_intersects(exp_bbox, b):
                        hit = True
                        break
                if hit:
                    continue
                
                # 能走到这里，说明外框通过了。接下来计算它在背景上占据的局部 ROI（Region of Interest）坐标
                roi = compute_roi_from_bbox((xmin, ymin, xmax, ymax), img_w, img_h, pad=pad)
                if roi is None:
                    continue
                rx0, ry0, rx1, ry1 = roi
                roi_w = rx1 - rx0
                roi_h = ry1 - ry0

                # 将世界坐标用的仿射矩阵 M，平移成局部 ROI 适用的坐标系 M_roi
                M_roi = M.copy()
                M_roi[0, 2] -= rx0
                M_roi[1, 2] -= ry0

                # ⭐性能核心所在：不旋转整张名图层，只对ROI尺寸这个巴掌大的区域做 cv2.warpAffine 旋转变换
                warp_bgr_roi, warp_mask_roi_u8 = warp_affine_roi(
                    card_item.bgr, card_item.mask_u8, M_roi, roi_w=roi_w, roi_h=roi_h
                )

                # enforce min gap using dilated new mask vs occupancy
                # 如果要求了间隙(pad>0)，就利用刚才生成的内核对这张旋转过后的 Mask 做一次形态学【膨胀】（变胖）
                if kernel is not None:
                    dilated_roi = cv2.dilate(warp_mask_roi_u8, kernel, iterations=1)
                else:
                    dilated_roi = warp_mask_roi_u8

                # 像素级别终极检测：从占据图 occ 中切出当前所处的对应区域，将其与膨胀后的名片 Mask 做与运算。
                # 如果非零（countNonZero > 0），说明有任意像素重合 -> 说明实际边缘撞击了，退回重试！
                occ_roi = occ[ry0:ry1, rx0:rx1]
                inter = cv2.bitwise_and(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)
                if cv2.countNonZero(inter) > 0:
                    continue

                # composite (NO color aug / NO blur / NO noise) 
                # 当前位置既不出界、也没有任何干涉，是个非常完美的地方

                # 1. 物理叠加：将旋转后的彩色名片利用原 Mask 合成到真实大背景中
                bg_roi = bg_bgr[ry0:ry1, rx0:rx1]
                composite_roi_inplace(bg_roi, warp_bgr_roi, warp_mask_roi_u8)
                
                # 2. 更新占据图：将膨胀后的 Mask 叠加到全局占据图 occ 上，标记这块区域已被占用
                occ_roi[:] = cv2.bitwise_or(occ_roi, (dilated_roi > 0).astype(np.uint8) * 255)

                placed_boxes_expanded.append(exp_bbox)

                # 3. 生成标签：根据四边形坐标生成 YOLO 格式的分割标签
                l = yolo_seg_line(0, quad, img_w, img_h)
                if l:
                    labels.append(l)

                quads_dbg.append(quad)
                placed += 1
                success_this = True
                break
            
            # 如果这张名片在天上试了所有次数（max_place_trials_per_card 次）都没成功（撞墙/出界）
            if not success_this:
                # couldn't place this card -> retry whole image
                placed = -999
                break

        # 名片的 for 循环结束。如果成功放完了预期数量的名片（或者 >= 最小要求的数目）：
        if placed >= cfg.min_cards:
            imwrite(out_img_path, bg_bgr)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

            if cfg.save_debug and out_dbg_path is not None:
                dbg = draw_debug(bg_bgr, quads_dbg)
                imwrite(out_dbg_path, dbg)

            return True

    return False

# 在批量生成几万甚至几十万张合成图片时，单核 CPU 会非常慢，因此程序把大任务拆分成多个小任务，交给不同的核心去同时进行。
def worker_run(worker_id: int, indices: List[int], cfg: SynthConfig, bg_paths: List[str], card_paths: List[str]) -> None:
    out_img_dir = cfg.out_dir / "images"
    out_lbl_dir = cfg.out_dir / "labels"
    out_dbg_dir = cfg.out_dir / "debug_vis"

    card_cache: Dict[str, CardCacheItem] = {}

    for k, idx in enumerate(indices):
        name = f"synth_{idx:06d}"
        out_img_path = out_img_dir / f"{name}.jpg"
        out_lbl_path = out_lbl_dir / f"{name}.txt"
        out_dbg_path = (out_dbg_dir / f"{name}.jpg") if cfg.save_debug else None

        ok = synth_one_image(
            cfg=cfg,
            bg_paths=bg_paths,
            card_paths=card_paths,
            idx=idx,
            out_img_path=out_img_path,
            out_lbl_path=out_lbl_path,
            out_dbg_path=out_dbg_path,
            card_cache=card_cache,
        )
        if not ok:
            raise RuntimeError(
                f"[Worker {worker_id}] Failed idx={idx} after {cfg.max_image_retries} retries.\n"
                f"cfg: out={cfg.out_w}x{cfg.out_h}, fixed_w={cfg.fixed_card_w}, "
                f"margin={cfg.margin_to_img}, gap={cfg.min_gap_between_cards}, "
                f"cards/img={cfg.min_cards}~{cfg.max_cards}, weights(2,3,4)=({cfg.weight_2},{cfg.weight_3},{cfg.weight_4}), "
                f"dyn_bg={cfg.dynamic_bg_enlarge}, max_bg_scale={cfg.max_bg_scale}\n"
                "Tip: if you insist on large cards AND strict gap/margin, increase max_bg_scale."
            )

        if (k + 1) % 20 == 0:
            print(f"[Worker {worker_id}] done {k+1}/{len(indices)}", flush=True)

# 简单的平均分配：把从 0 到 n-1 这 n 个任务号，轮流扔给每个核心
def split_indices(n: int, num_workers: int) -> List[List[int]]:
    buckets = [[] for _ in range(num_workers)]
    for i in range(n):
        buckets[i % num_workers].append(i)
    return [b for b in buckets if b]


def generate_dataset(cfg: SynthConfig, overwrite: bool = True) -> None:
    """
    Create:
      cfg.out_dir/images/*.jpg
      cfg.out_dir/labels/*.txt   (YOLO-seg polygon labels)
      cfg.out_dir/debug_vis/*.jpg  (optional)
    """
    bg_paths = [str(p) for p in list_images(cfg.bg_dir)]
    card_paths = [str(p) for p in list_images(cfg.card_dir)]
    if not bg_paths:
        raise FileNotFoundError(f"No background images found in: {cfg.bg_dir}")
    if not card_paths:
        raise FileNotFoundError(f"No card images found in: {cfg.card_dir}")

    if overwrite:
        ensure_clean_dir(cfg.out_dir)
    (cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "labels").mkdir(parents=True, exist_ok=True)
    if cfg.save_debug:
        (cfg.out_dir / "debug_vis").mkdir(parents=True, exist_ok=True)

    print("[INFO] Synth Seg Dataset (Rigid / Rotation-only)")
    print(f"  out_dir  : {cfg.out_dir}")
    print(f"  num      : {cfg.num_images}")
    print(f"  cards/img: {cfg.min_cards}~{cfg.max_cards}")
    print(f"  weights  : 2->{cfg.weight_2}, 3->{cfg.weight_3}, 4->{cfg.weight_4}")
    print(f"  fixed_w  : {cfg.fixed_card_w}")
    print(f"  margin   : {cfg.margin_to_img}")
    print(f"  min_gap  : {cfg.min_gap_between_cards}")
    print(f"  angle    : {cfg.angle_min}~{cfg.angle_max}")
    print(f"  out_size : {cfg.out_w} x {cfg.out_h} (None means keep bg size)")
    print(f"  dyn_bg   : {cfg.dynamic_bg_enlarge}, only_3plus={cfg.dynamic_bg_only_for_3plus}, max_scale={cfg.max_bg_scale}")
    print(f"  workers  : {cfg.num_workers}")
    print(f"  debug    : {cfg.save_debug}")

    if cfg.num_workers <= 1:
        worker_run(
            worker_id=0,
            indices=list(range(cfg.num_images)),
            cfg=cfg,
            bg_paths=bg_paths,
            card_paths=card_paths,
        )
        print("[OK] Done (single process).")
        return

    ctx = mp.get_context("spawn")
    buckets = split_indices(cfg.num_images, cfg.num_workers)
    args_list = [(wid, indices, cfg, bg_paths, card_paths) for wid, indices in enumerate(buckets)]

    with ctx.Pool(processes=len(buckets)) as pool:
        pool.starmap(worker_run, args_list)

    print("[OK] Done (multiprocessing).")
