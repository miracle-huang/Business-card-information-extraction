from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 用户配置区域 (USER CONFIGURATION)
# 你可以在这里直接修改默认值，也可以通过命令行覆盖
# ==========================================
CFG_MODEL = "content_recognition/runs/train_yolo11s_bs8_img960/weights/best.pt"
CFG_SOURCE = "data/business_card_v2/test/images"  # 替换为你想要识别的图片/文件夹路径
CFG_PROJECT = "content_recognition/runs/detect"
CFG_NAME = "predict_result"
CFG_DEVICE = "0"
# ==========================================


def main(
    model: str,
    source: str,
    device: str = "0",
    project: str = "runs/predict",
    name: str = "exp",
    save: bool = True,
) -> None:
    """
    YOLO prediction entry
    """
    if not source:
        print("[Error] No source provided. Please set CFG_SOURCE in script or use --source")
        return

    # 1. Load model
    print(f"[Predict] Loading model: {model}")
    yolo_model = YOLO(model)

    # 2. Predict
    # project/name determines where results are saved -> project/name
    print(f"[Predict] Processing source: {source}")
    results = yolo_model.predict(
        source=source,
        device=device,
        save=save,
        project=project,
        name=name,
        exist_ok=True,  # Allow overwriting if same name
    )

    print(f"[Predict] Processed {len(results)} images.")
    # Output path hint
    # Ultralytics automatic naming might result in project/name, project/name2 etc unless exist_ok=True is used
    save_dir = Path(project) / name
    print(f"[Predict] Results saved to: {save_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Prediction Script")
    
    parser.add_argument("--model", type=str, default=CFG_MODEL, help=f"Path to .pt model file (default: {CFG_MODEL})")
    parser.add_argument("--source", type=str, default=CFG_SOURCE, help=f"Image path, dir path, or video path (default: {CFG_SOURCE})")
    parser.add_argument("--device", type=str, default=CFG_DEVICE, help="Device (0, 1, cpu)")
    
    # 输出配置
    parser.add_argument("--project", type=str, default=CFG_PROJECT, help="Project directory for outputs")
    parser.add_argument("--name", type=str, default=CFG_NAME, help="Experiment name")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Do not save output images")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 简单的路径存在性检查 (如果提供了路径)
    if args.model and not Path(args.model).exists():
        print(f"[Warning] Model file not found: {args.model}")
        print("Please check the path or train a model first.")
    
    main(
        model=args.model,
        source=args.source,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
    )