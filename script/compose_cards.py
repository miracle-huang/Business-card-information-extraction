import argparse
import random
import math
from pathlib import Path
from datetime import datetime

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def transform_card(img: Image.Image, angle_deg: float, scale: float) -> Image.Image:
    if scale != 1.0:
        w, h = img.size
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.LANCZOS)
    return img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)


def ensure_bg_big_enough(bg: Image.Image, req_w: int, req_h: int) -> Image.Image:
    bw, bh = bg.size
    if bw >= req_w and bh >= req_h:
        return bg
    s = max(req_w / bw, req_h / bh, 1.0)
    nw = int(math.ceil(bw * s))
    nh = int(math.ceil(bh * s))
    return bg.resize((nw, nh), resample=Image.LANCZOS)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def randint_safe(a: int, b: int) -> int:
    return a if b < a else random.randint(a, b)


def tight_crop(img_rgba: Image.Image, boxes_xywh, pad: int) -> Image.Image:
    """
    boxes_xywh: [(x, y, w, h), ...] on img coordinates
    """
    W, H = img_rgba.size
    x0 = min(x for x, y, w, h in boxes_xywh)
    y0 = min(y for x, y, w, h in boxes_xywh)
    x1 = max(x + w for x, y, w, h in boxes_xywh)
    y1 = max(y + h for x, y, w, h in boxes_xywh)

    x0 = clamp(x0 - pad, 0, W)
    y0 = clamp(y0 - pad, 0, H)
    x1 = clamp(x1 + pad, 0, W)
    y1 = clamp(y1 + pad, 0, H)

    # PIL crop 的右下是开区间，但这里不影响使用
    return img_rgba.crop((x0, y0, x1, y1))


def generate_one(
    bg_path: Path,
    card_paths,
    out_path: Path,
    angle_min: float,
    angle_max: float,
    scale_min: float,
    scale_max: float,
    margin: int,
    gap: int,
    pad: int,
    crop_pad: int,
    resize_long_side: int | None,
):
    bg = load_rgba(bg_path)

    # 4 张名片：随机角度+随机缩放
    cards = []
    for cp in card_paths:
        card = load_rgba(cp)
        angle = random.uniform(angle_min, angle_max)
        scale = random.uniform(scale_min, scale_max)
        card_t = transform_card(card, angle, scale)
        cards.append(card_t)

    # 用最大名片尺寸确定每个格子的最小容纳尺寸（保证不重叠）
    max_w = max(c.size[0] for c in cards)
    max_h = max(c.size[1] for c in cards)
    region_w = max_w + 2 * pad
    region_h = max_h + 2 * pad

    # 背景最小需求
    req_w = margin * 2 + region_w * 2 + gap
    req_h = margin * 2 + region_h * 2 + gap
    bg = ensure_bg_big_enough(bg, req_w, req_h)

    bw, bh = bg.size
    block_w = region_w * 2 + gap
    block_h = region_h * 2 + gap

    # 关键：不要固定左上角，随机把 2×2 的“块”放到背景任意位置
    bx0 = randint_safe(margin, bw - margin - block_w)
    by0 = randint_safe(margin, bh - margin - block_h)

    regions = [
        (bx0, by0, bx0 + region_w, by0 + region_h),  # TL
        (bx0 + region_w + gap, by0, bx0 + block_w, by0 + region_h),  # TR
        (bx0, by0 + region_h + gap, bx0 + region_w, by0 + block_h),  # BL
        (bx0 + region_w + gap, by0 + region_h + gap, bx0 + block_w, by0 + block_h),  # BR
    ]

    random.shuffle(regions)
    random.shuffle(cards)

    placed_boxes = []  # for tight crop

    for card_img, (rx0, ry0, rx1, ry1) in zip(cards, regions):
        cw, ch = card_img.size

        px_min = rx0 + pad
        py_min = ry0 + pad
        px_max = rx1 - pad - cw
        py_max = ry1 - pad - ch

        x = randint_safe(px_min, px_max)
        y = randint_safe(py_min, py_max)

        bg.alpha_composite(card_img, dest=(x, y))
        placed_boxes.append((x, y, cw, ch))

    # ✅ 合成后紧致裁剪：减少空白
    bg = tight_crop(bg, placed_boxes, pad=crop_pad)

    # ✅ 可选：把最终图缩放到固定“长边”，方便训练/可视化一致
    if resize_long_side is not None and resize_long_side > 0:
        w, h = bg.size
        if w >= h:
            nw = resize_long_side
            nh = int(round(h * (resize_long_side / w)))
        else:
            nh = resize_long_side
            nw = int(round(w * (resize_long_side / h)))
        bg = bg.resize((nw, nh), resample=Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_dir", type=str, default=r"data\background")
    parser.add_argument("--card_dir", type=str, default=r"data\business_card_v2")
    parser.add_argument("--out_dir", type=str, default=r"data\synth_output")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--angle_min", type=float, default=0.0)
    parser.add_argument("--angle_max", type=float, default=360.0)
    parser.add_argument("--scale_min", type=float, default=0.85)
    parser.add_argument("--scale_max", type=float, default=1.05)

    parser.add_argument("--margin", type=int, default=40)
    parser.add_argument("--gap", type=int, default=60)
    parser.add_argument("--pad", type=int, default=20)

    # ✅ 新增：紧致裁剪留白（越小空白越少）
    parser.add_argument("--crop_pad", type=int, default=80)

    # ✅ 新增：把最终图缩放到固定长边（比如 1024/1280/1600），不想缩放就设为 0
    parser.add_argument("--resize_long_side", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    bg_dir = Path(args.bg_dir)
    card_dir = Path(args.card_dir)
    out_dir = Path(args.out_dir)

    bgs = list_images(bg_dir)
    cards_all = list_images(card_dir)
    if not bgs:
        raise FileNotFoundError(f"背景目录没有图片：{bg_dir}")
    if len(cards_all) < 4:
        raise FileNotFoundError(f"名片图片不足 4 张：{card_dir}（当前 {len(cards_all)} 张）")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    resize_long_side = args.resize_long_side if args.resize_long_side > 0 else None

    for i in range(args.n):
        bg_path = random.choice(bgs)
        card_paths = random.sample(cards_all, 4)
        out_path = out_dir / f"synth_{ts}_{i:05d}.jpg"

        generate_one(
            bg_path=bg_path,
            card_paths=card_paths,
            out_path=out_path,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            margin=args.margin,
            gap=args.gap,
            pad=args.pad,
            crop_pad=args.crop_pad,
            resize_long_side=resize_long_side,
        )

    print(f"Done. 输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()