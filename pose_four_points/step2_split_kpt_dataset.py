"""
[Step 2] 数据校验与切分脚本
功能：
1. 扫描 Pool 文件夹中的所有合成图像和标签。
2. 校验标签格式是否符合 YOLO-Pose 要求（例如：17个字段，坐标在0-1之间）。
3. 随机打乱并按比例切分为 train 和 val 集合，生成标准的 YOLO 训练目录结构。
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# ✅ ensure current dir on sys.path
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))


# =========================
# ✅ 配置区：只改这里
# =========================
SEED = 42
VAL_RATIO = 0.2        # 例如 0.2 => 80% train / 20% val
VAL_NUM = 100          # 直接指定验证集数量, 优先级高于 VAL_RATIO, None 则不指定
CLEAR_OUT_DIR = True   # True 会清空输出目录再重新生成

# 路径配置
SRC_POOL_DIR = CUR_DIR / "assets" / "step1_outputs"        # 步骤1输出目录
DST_STATIC_DIR = CUR_DIR / "assets" / "step2_train_dataset"  # 步骤2输出目录 (用于训练)
YAML_OUT_PATH = DST_STATIC_DIR / "dataset_card4kpt.yaml"     # 自动生成的配置文件路径 (存放在数据集目录下)


# =========================
# Internal helpers
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def validate_label_file(lbl_path: Path, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate YOLO pose label format for 4 keypoints with visibility:
    expected fields per line = 1(class) + 4(bbox) + 4*(x,y,v) = 17
    Return: (ok, errors)
    """
    errors: List[str] = []
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        errors.append("empty label file")
        return (not strict), errors

    for ln, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 17:
            errors.append(f"line {ln}: expected 17 fields, got {len(parts)}")
            continue

        cls = parts[0]
        if not cls.isdigit():
            errors.append(f"line {ln}: class is not int: {cls}")
            continue

        # bbox floats
        bbox = parts[1:5]
        if not all(_is_float(x) for x in bbox):
            errors.append(f"line {ln}: bbox has non-float")
            continue

        # keypoints: (x,y,v)*4
        kpts = parts[5:]
        for i in range(4):
            x = kpts[i * 3 + 0]
            y = kpts[i * 3 + 1]
            v = kpts[i * 3 + 2]
            if (not _is_float(x)) or (not _is_float(y)):
                errors.append(f"line {ln}: kpt{i} x/y not float")
                break
            if not v.isdigit():
                errors.append(f"line {ln}: kpt{i} v not int")
                break
            vf = int(v)
            if vf not in (0, 1, 2):
                errors.append(f"line {ln}: kpt{i} v not in (0,1,2): {vf}")
                break

            xf = float(x)
            yf = float(y)
            # allow tiny numeric drift
            if xf < -1e-4 or xf > 1.0001 or yf < -1e-4 or yf > 1.0001:
                errors.append(f"line {ln}: kpt{i} out of [0,1]: ({xf:.4f},{yf:.4f})")
                break

    ok = len(errors) == 0
    if strict:
        return ok, errors
    else:
        # non-strict: allow but report
        return True, errors


def main() -> None:
    src_img_dir = SRC_POOL_DIR / "images"
    src_lbl_dir = SRC_POOL_DIR / "labels"

    if not src_img_dir.exists():
        raise FileNotFoundError(src_img_dir)
    if not src_lbl_dir.exists():
        raise FileNotFoundError(src_lbl_dir)

    # collect pairs
    img_paths = list_images(src_img_dir)

    pairs: List[Tuple[Path, Path]] = []
    missing_label = 0
    bad_label_files = 0
    bad_label_details: List[str] = []

    for img_path in img_paths:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_label += 1
            continue

        ok, errs = validate_label_file(lbl_path, strict=True)
        if not ok:
            bad_label_files += 1
            bad_label_details.append(f"[{lbl_path.name}] " + "; ".join(errs[:3]))
            # 严格模式：直接跳过坏标签（也可以改成 raise）
            continue

        pairs.append((img_path, lbl_path))

    if not pairs:
        raise RuntimeError("No valid (image,label) pairs found. Check your pool directory.")

    # shuffle & split
    rng = random.Random(SEED)
    rng.shuffle(pairs)

    n = len(pairs)

    if VAL_NUM is not None and VAL_NUM > 0:
        n_val = VAL_NUM
    else:
        n_val = int(round(n * VAL_RATIO))

    n_val = max(1, n_val) if n >= 5 else max(0, n_val)  # very small datasets: allow 0 val
    n_train = n - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # output dirs
    dst_img_train = DST_STATIC_DIR / "images" / "train"
    dst_img_val = DST_STATIC_DIR / "images" / "val"
    dst_lbl_train = DST_STATIC_DIR / "labels" / "train"
    dst_lbl_val = DST_STATIC_DIR / "labels" / "val"

    if CLEAR_OUT_DIR:
        clear_dir(DST_STATIC_DIR)
    dst_img_train.mkdir(parents=True, exist_ok=True)
    dst_img_val.mkdir(parents=True, exist_ok=True)
    dst_lbl_train.mkdir(parents=True, exist_ok=True)
    dst_lbl_val.mkdir(parents=True, exist_ok=True)

    # copy
    def copy_pairs(pairs_: List[Tuple[Path, Path]], img_out: Path, lbl_out: Path) -> None:
        for img_p, lbl_p in pairs_:
            shutil.copy2(img_p, img_out / img_p.name)
            shutil.copy2(lbl_p, lbl_out / lbl_p.name)

    copy_pairs(train_pairs, dst_img_train, dst_lbl_train)
    if val_pairs:
        copy_pairs(val_pairs, dst_img_val, dst_lbl_val)

    # --- 新增：自动生成 YOLO 训练所需的 YAML 配置文件 ---
    yaml_content = f"""# 自动生成的关键点训练配置文件
path: {DST_STATIC_DIR.absolute().as_posix()} # 数据集根目录（绝对路径）
train: images/train
val: images/val

# 类别信息
nc: 1
names: ['card']

# 关键点信息：4个角点，维度为3 (x, y, visible)
kpt_shape: [4, 3]
"""
    YAML_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT_PATH.write_text(yaml_content, encoding="utf-8")
    
    # report
    print("\n[split_kpt_dataset] DONE")
    print(f"  SRC pool images: {len(img_paths)}")
    print(f"  Missing labels: {missing_label}")
    print(f"  Bad label files skipped: {bad_label_files}")
    if bad_label_details:
        print("  Examples of bad labels:")
        for s in bad_label_details[:5]:
            print("   -", s)

    print(f"  Valid pairs used: {n}")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")
    print(f"  Output dir: {DST_STATIC_DIR}")
    print(f"  Config saved to: {YAML_OUT_PATH}\n")


if __name__ == "__main__":
    main()