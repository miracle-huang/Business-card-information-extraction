from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List

from ultralytics import YOLO

from pipeline.run_pipeline import PipelineConfig, run_on_image, save_json
from ocr.paddle_backend import PaddleOCRBackend


# ========== 你的 data.yaml 类别顺序 ==========
NAMES: List[str] = ["address", "company", "email", "name", "phone"]


# ========== 配置对象：既可 CLI，也可代码直接传 ==========
@dataclass
class PredictOCRConfig:
    # required
    weights: str
    input_path: str

    # optional
    out_dir: str = "runs_local/ocr"

    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    pad: float = 0.03
    save_crops: bool = False

    ocr_lang: str = "japan"  # PaddleOCR language: japan/en/ch/...


# ========== CLI ==========
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO + OCR pipeline for business card fields")

    p.add_argument("--weights", type=str, required=True, help="fine-tuned YOLO .pt path")
    p.add_argument("--input", dest="input_path", type=str, required=True, help="image file or folder")
    p.add_argument("--out", dest="out_dir", type=str, default="runs_local/ocr", help="output directory")

    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pad", type=float, default=0.03)
    p.add_argument("--save_crops", action="store_true", help="save cropped field images for debugging")

    p.add_argument("--ocr_lang", type=str, default="japan", help="PaddleOCR language, e.g. japan/en/ch")
    return p


def parse_cli_args() -> PredictOCRConfig:
    args = build_argparser().parse_args()
    return PredictOCRConfig(
        weights=args.weights,
        input_path=args.input_path,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        pad=args.pad,
        save_crops=bool(args.save_crops),
        ocr_lang=args.ocr_lang,
    )


# ========== Utils ==========
# 生成器函数，遍历目录下所有图片文件
def iter_images(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for fp in sorted(path.rglob("*")):
        if fp.suffix.lower() in exts:
            yield fp


# ========== Core Runner ==========
def run_predict_ocr(cfg: PredictOCRConfig) -> Path:
    input_path = Path(cfg.input_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load YOLO model
    model = YOLO(cfg.weights)

    # 2) Load OCR backend
    ocr = PaddleOCRBackend(lang=cfg.ocr_lang, use_angle_cls=True)

    # 3) Pipeline config
    pipe_cfg = PipelineConfig(
        weights=cfg.weights,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        pad_ratio=cfg.pad,
        save_crops=cfg.save_crops,
    )

    # 4) Output directory
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # 5) Process images
    any_image = False
    for img_path in iter_images(input_path):
        any_image = True
        result = run_on_image(
            image_path=img_path,
            model=model,
            names=NAMES,
            ocr=ocr,
            cfg=pipe_cfg,
            out_dir=out_dir,
        )
        save_json(result, json_dir / f"{img_path.stem}.json")

    if not any_image:
        raise FileNotFoundError(f"No images found under: {input_path}")

    print(f"[OK] Saved results to: {json_dir}")
    return json_dir


# ========== Entry ==========
def main(cfg: Optional[PredictOCRConfig] = None) -> None:
    """
    - cfg is None  -> parse CLI args (add_argument workflow)
    - cfg provided -> run with provided config (code workflow)
    """
    if cfg is None:
        cfg = parse_cli_args()
    run_predict_ocr(cfg)


if __name__ == "__main__":
    # ✅ 方式 1：直接写死参数（你要的：从 main 里直接调用）
    # 改成你自己的路径即可，然后直接：python src/predict_ocr.py
    cfg = PredictOCRConfig(
        weights="runs_local/debug_yolo11m_bs2_img640/weights/best.pt",
        input_path="data/roboflow_v1/test/images",
        out_dir="runs_local/ocr_test",

        imgsz=960,
        conf=0.25,
        iou=0.45,
        device="0",

        pad=0.03,
        save_crops=True,

        ocr_lang="japan",
    )
    main(cfg)

    # ✅ 方式 2：命令行参数（保留 add_argument）
    # 如果你想改回命令行运行，把上面三行（cfg=... 和 main(cfg)）注释掉，
    # 然后取消注释下面这一行即可：
    # main()
