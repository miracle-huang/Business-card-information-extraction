import sys
import shutil
from pathlib import Path

# ✅ 强制把项目根目录加入 sys.path，确保能 import src.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.build import build_yolo_dataset, build_dataloader  # noqa: E402
from src.synth.seg_synth import SynthConfig, generate_dataset  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def copy_pair(img_path: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    lb = src_lbl_dir / f"{img_path.stem}.txt"
    if not lb.exists():
        raise FileNotFoundError(f"Missing label: {lb}")
    shutil.copy2(img_path, dst_img_dir / img_path.name)
    shutil.copy2(lb, dst_lbl_dir / lb.name)


def write_yaml(out_dir: Path):
    text = f"""# Mixed dataset (static + dynamic)
path: {out_dir.as_posix()}
train: images/train
val: images/val

names:
  0: business_card
"""
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def delete_dynamic_files(train_img_dir: Path, train_lbl_dir: Path, prefix: str = "dyn_"):
    # 删除旧动态样本（图片+标签）
    for p in train_img_dir.glob(f"{prefix}*"):
        if p.is_file():
            p.unlink()
    for p in train_lbl_dir.glob(f"{prefix}*"):
        if p.is_file():
            p.unlink()


def generate_dynamic_to_train(
    dynamic_root: Path,
    mix_train_img: Path,
    mix_train_lbl: Path,
    n_dynamic: int,
    seed: int,
):
    """
    生成 n_dynamic 张动态合成图到 dynamic_root，然后拷贝进 mix_train（命名为 dyn_00000.jpg 等）。
    """
    clear_dir(dynamic_root)

    cfg = SynthConfig(
        bg_dir=Path("data/background"),
        card_dir=Path("data/business_card_v2"),
        out_dir=dynamic_root,
        num_images=n_dynamic,

        out_w=1536,
        out_h=1536,

        min_cards=2,
        max_cards=4,

        # 同图名片尺寸一致，保证清晰
        min_card_long=700,
        max_card_long=700,
        same_size_in_image=True,

        # 让 4 张概率更高（沿用你之前的权重逻辑）
        target_count_weights=(1.0, 2.0, 10.0),
        prefer_full_target=True,
        retry_prob_if_underfilled={3: 0.25, 4: 0.85},
        min_accept_when_target4=3,
        must_place_all=False,

        angle_min=0.0,
        angle_max=360.0,

        min_gap_px=6,
        margin=20,
        max_tries_per_card=250,
        max_image_tries=60,

        seed=seed,
        save_debug=False,
    )

    generate_dataset(cfg)

    dyn_imgs = list_images(dynamic_root / "images_all")
    dyn_lbl_dir = dynamic_root / "labels_all"

    if len(dyn_imgs) != n_dynamic:
        raise RuntimeError(f"Dynamic images mismatch: {len(dyn_imgs)} != {n_dynamic}")

    # 拷贝到 mix/train，固定命名，方便每轮覆盖/删除
    for i, im in enumerate(dyn_imgs):
        dst_im = mix_train_img / f"dyn_{i:05d}{im.suffix.lower()}"
        dst_lb = mix_train_lbl / f"dyn_{i:05d}.txt"

        src_lb = dyn_lbl_dir / f"{im.stem}.txt"
        if not src_lb.exists():
            raise FileNotFoundError(f"Missing dynamic label: {src_lb}")

        shutil.copy2(im, dst_im)
        shutil.copy2(src_lb, dst_lb)


def rebuild_train_loader(trainer, train_img_dir: Path):
    """
    用官方 build_yolo_dataset + build_dataloader 重建 train dataloader，
    确保每个 epoch 的动态 labels/segments 真正生效（不吃旧 cache）。
    """
    # stride：不同任务/模型可能是 tensor/list，这里尽量稳健
    stride = 32
    try:
        s = getattr(trainer.model, "stride", None)
        if s is not None:
            if hasattr(s, "max"):
                stride = int(s.max())
            elif isinstance(s, (list, tuple)) and len(s) > 0:
                stride = int(max(s))
            else:
                stride = int(s)
    except Exception:
        stride = 32

    # 重新建 dataset
    dataset = build_yolo_dataset(
        cfg=trainer.args,
        img_path=str(train_img_dir),
        batch=trainer.args.batch,
        data=trainer.data,
        mode="train",
        rect=False,
        stride=stride,
    )

    # 重新建 dataloader（会重启 worker，确保 dataset 更新）
    rank = getattr(trainer, "rank", -1)
    loader = build_dataloader(
        dataset=dataset,
        batch=trainer.args.batch,
        workers=trainer.args.workers,
        shuffle=True,
        rank=rank,
        drop_last=False,
        pin_memory=True,
    )

    # 替换 trainer 内部引用
    # 尽量清理旧 loader（避免 Windows 多进程残留）
    try:
        old = getattr(trainer, "train_loader", None)
        if old is not None and hasattr(old, "__del__"):
            old.__del__()
    except Exception:
        pass

    trainer.train_loader = loader
    trainer.trainset = dataset


def main():
    # ---------- 固定静态数据（你的 step2 输出） ----------
    static_dir = Path("data/seg_synth_step2_yolo")
    st_train_img = static_dir / "images/train"
    st_train_lbl = static_dir / "labels/train"
    st_val_img = static_dir / "images/val"
    st_val_lbl = static_dir / "labels/val"

    st_train_imgs = list_images(st_train_img)
    st_val_imgs = list_images(st_val_img)

    if len(st_train_imgs) == 0 or len(st_val_imgs) == 0:
        raise RuntimeError("Static train/val is empty. Check data/seg_synth_step2_yolo split.")

    # 50/50：动态数量 = 静态 train 数量（例如40）
    n_dynamic = len(st_train_imgs)

    # ---------- 混合数据目录（训练时用它） ----------
    dynamic_root = Path("data/seg_synth_dynamic_runtime")  # 每个epoch重新生成
    mix_root = Path("data/seg_synth_mix_runtime")

    mix_train_img = mix_root / "images/train"
    mix_train_lbl = mix_root / "labels/train"
    mix_val_img = mix_root / "images/val"
    mix_val_lbl = mix_root / "labels/val"

    # ---------- 训练超参 ----------
    total_epochs = 4
    seed_base = 42

    imgsz = 1024
    batch = 4

    # ✅ Windows 动态重建 loader 时，多 worker 容易踩坑；先用 0 最稳
    # 后续你数据大、流程稳定后再改 2/4/8
    workers = 0
    device = 0  # 没GPU改成 "cpu"

    # ---------- 准备 mix 目录：复制静态 train/val ----------
    clear_dir(mix_root)
    mix_train_img.mkdir(parents=True, exist_ok=True)
    mix_train_lbl.mkdir(parents=True, exist_ok=True)
    mix_val_img.mkdir(parents=True, exist_ok=True)
    mix_val_lbl.mkdir(parents=True, exist_ok=True)

    # 复制静态 train
    for im in st_train_imgs:
        copy_pair(im, st_train_lbl, mix_train_img, mix_train_lbl)

    # 复制静态 val（固定不变）
    for im in st_val_imgs:
        copy_pair(im, st_val_lbl, mix_val_img, mix_val_lbl)

    write_yaml(mix_root)

    # 先生成 epoch0 的动态样本，保证第一轮就是 50/50
    delete_dynamic_files(mix_train_img, mix_train_lbl, prefix="dyn_")
    generate_dynamic_to_train(
        dynamic_root=dynamic_root,
        mix_train_img=mix_train_img,
        mix_train_lbl=mix_train_lbl,
        n_dynamic=n_dynamic,
        seed=seed_base + 0,
    )
    print(f"[Init] static={len(st_train_imgs)} dynamic={n_dynamic} -> mix train={len(list_images(mix_train_img))}")

    # ---------- 定义 callback：每个 epoch 开始时刷新动态数据 + 重建 train_loader ----------
    def on_train_epoch_start(trainer):
        # trainer.epoch 从 0 开始；epoch0 已经在 main 里生成过了
        if trainer.epoch <= 0:
            return

        ep = int(trainer.epoch)
        print(f"\n[Callback] Refresh dynamic data for epoch={ep} ...")

        # 1) 删除旧动态
        delete_dynamic_files(mix_train_img, mix_train_lbl, prefix="dyn_")

        # 2) 生成新动态（seed 随 epoch 变）
        generate_dynamic_to_train(
            dynamic_root=dynamic_root,
            mix_train_img=mix_train_img,
            mix_train_lbl=mix_train_lbl,
            n_dynamic=n_dynamic,
            seed=seed_base + ep,
        )

        # 3) 重建 train_loader，确保 labels/segments 生效
        rebuild_train_loader(trainer, train_img_dir=mix_train_img)

        print(f"[Callback] done. train images now={len(list_images(mix_train_img))} (static+dynamic)")

    # ---------- 开始训练（只调用一次 train，优化器状态自然连续） ----------
    model = YOLO("yolo11n-seg.pt")
    model.add_callback("on_train_epoch_start", on_train_epoch_start)  # 官方 callback 接口 :contentReference[oaicite:1]{index=1}

    model.train(
        data=str(mix_root / "data.yaml"),
        task="segment",
        epochs=total_epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed_base,
        cache=False,  # ✅ 避免 dataset/labels cache 干扰动态更新
        project="runs_local/seg_train",
        name="mix50_callback",
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ Training finished.")
    print("Weights: runs_local/seg_train/mix50_callback/weights/last.pt")


if __name__ == "__main__":
    main()
