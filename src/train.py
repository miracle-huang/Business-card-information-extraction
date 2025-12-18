from __future__ import annotations
import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.io import load_yaml, TrainConfig
from utils.sanity import sanity_check_yolo_dataset


def train(cfg: TrainConfig) -> None:
    # 1) 检查数据集结构
    sanity_check_yolo_dataset(cfg.data_yaml)

    # 2) 初始化模型（预训练权重）
    model = YOLO(cfg.model)

    # 3) 开始训练
    # Ultralytics 的 train 参数基本与命令行一致，但这里全用代码传入
    train_kwargs = dict(
        data=cfg.data_yaml,
        task=cfg.task,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        seed=cfg.seed,
        project=cfg.project,
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        optimizer=cfg.optimizer,
        freeze=cfg.freeze,
    )

    # 可选超参：只在配置里提供时再传（避免 None 引发覆盖）
    if cfg.lr0 is not None:
        train_kwargs["lr0"] = cfg.lr0

    results = model.train(**train_kwargs)

    # 4) 训练结果位置提示
    # results.save_dir 通常是 runs/.../trainX
    try:
        print(f"[Done] Save dir: {results.save_dir}")
        print(f"[Done] Best: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    except Exception:
        print("[Done] Training finished.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/default.yaml")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_dict = load_yaml(args.config)
    cfg = TrainConfig.from_dict(cfg_dict)
    train(cfg)
