import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/seg_synth_step1", help="Folder that contains images_all/labels_all")
    ap.add_argument("--out_dir", type=str, default="data/seg_synth_step2_yolo", help="Output YOLO dataset folder")
    ap.add_argument("--train", type=int, default=40, help="Number of train images")
    ap.add_argument("--val", type=int, default=20, help="Number of val images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move files instead of copy (default: copy)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    images_all = in_dir / "images_all"
    labels_all = in_dir / "labels_all"

    if not images_all.exists():
        raise FileNotFoundError(f"Missing: {images_all}")
    if not labels_all.exists():
        raise FileNotFoundError(f"Missing: {labels_all}")

    imgs = list_images(images_all)
    if len(imgs) < args.train + args.val:
        raise ValueError(f"Not enough images: {len(imgs)} < train+val={args.train + args.val}")

    # 检查 label 是否齐全
    missing = []
    for im in imgs:
        lb = labels_all / f"{im.stem}.txt"
        if not lb.exists():
            missing.append(lb.name)
    if missing:
        raise FileNotFoundError(f"Missing label files: {missing[:10]} ... total={len(missing)}")

    # 随机打乱并切分
    random.seed(args.seed)
    random.shuffle(imgs)

    train_imgs = imgs[: args.train]
    val_imgs = imgs[args.train : args.train + args.val]

    # 输出目录
    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    lbl_train_dir = out_dir / "labels" / "train"
    lbl_val_dir = out_dir / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2

    def transfer_one(im_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
        lb_path = labels_all / f"{im_path.stem}.txt"
        op(str(im_path), str(dst_img_dir / im_path.name))
        op(str(lb_path), str(dst_lbl_dir / lb_path.name))

    for im in train_imgs:
        transfer_one(im, img_train_dir, lbl_train_dir)
    for im in val_imgs:
        transfer_one(im, img_val_dir, lbl_val_dir)

    # data.yaml（Ultralytics YOLO）
    # 注意：path 写成相对项目根目录更方便
    yaml_text = f"""# YOLOv8/YOLO11 Segmentation dataset
path: {out_dir.as_posix()}
train: images/train
val: images/val

names:
  0: business_card
"""
    write_text(out_dir / "data.yaml", yaml_text)

    info = (
        f"in_dir: {in_dir.resolve()}\n"
        f"out_dir: {out_dir.resolve()}\n"
        f"train: {len(train_imgs)}\n"
        f"val: {len(val_imgs)}\n"
        f"seed: {args.seed}\n"
        f"mode: {'MOVE' if args.move else 'COPY'}\n"
    )
    write_text(out_dir / "split_info.txt", info)

    print("✅ Split done.")
    print(info)


if __name__ == "__main__":
    main()
