from __future__ import annotations

import shutil
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils import LOGGER

# ✅ RANK 在不同 ultralytics 版本位置不同：做兼容处理
try:
    from ultralytics.utils import RANK  # 常见位置
except Exception:
    RANK = -1  # 单进程/单卡等价 -1

# ✅ 默认 cfg 在不同版本的位置/名字可能不同：做兼容处理
try:
    from ultralytics.cfg import DEFAULT_CFG_DICT  # 常见：dict
except Exception:
    DEFAULT_CFG_DICT = None

from src.hybrid_train.hybrid_yolo_dataset import HybridYoloDataset
from src.synth.kpt_synth import SynthKptConfig, generate_dataset


def _safe_rmtree(p: Path, retry: int = 5, sleep: float = 0.2) -> None:
    if not p.exists():
        return
    for i in range(retry):
        try:
            shutil.rmtree(p)
            return
        except Exception as e:
            if i == retry - 1:
                raise e
            time.sleep(sleep)


class HybridPoseTrainer(PoseTrainer):
    """
    自定义 Trainer（Pose）：
    - train dataset = HybridYoloDataset(static_train, runtime_train)
    - 每个 epoch start：重新生成 runtime 数据，并 rebuild runtime YOLODataset
    """

    RUNTIME_SYNTH_CFG: Optional[SynthKptConfig] = None
    RUNTIME_DIR: Path = Path(r"data\synth_kpt_runtime")
    RUNTIME_SEED: int = 12345
    RUNTIME_MULTIPLIER: float = 1.0

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # ✅ 关键修复：ultralytics 不能吃 cfg=None
        if cfg is None:
            if DEFAULT_CFG_DICT is not None:
                cfg = deepcopy(DEFAULT_CFG_DICT)
            else:
                # 最差兜底：给空 dict（至少不为 None）
                cfg = {}

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        # 注册 epoch-start 回调（如果你的版本没有 add_callback，下一步我再按你版本改写）
        if hasattr(self, "add_callback"):
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start_refresh_runtime)
        else:
            raise RuntimeError(
                "你的 ultralytics 版本没有 add_callback(). "
                "请把 `pip show ultralytics` 的版本号贴我，我会改成你版本支持的回调注册方式。"
            )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Ultralytics 构建 dataloader 时会调用：
        - mode=train：返回 HybridYoloDataset（严格 50/50）
        - mode=val：返回普通静态 dataset
        """
        static_ds = super().build_dataset(img_path, mode=mode, batch=batch)

        if mode != "train":
            return static_ds

        runtime_ds = self._build_runtime_dataset(batch=batch, static_len=len(static_ds), epoch=0)

        hybrid = HybridYoloDataset(static_dataset=static_ds, runtime_dataset=runtime_ds)
        self._hybrid_dataset = hybrid
        return hybrid

    def _build_runtime_dataset(self, batch: int | None, static_len: int, epoch: int):
        cfg = self.RUNTIME_SYNTH_CFG
        if cfg is None:
            raise RuntimeError("HybridPoseTrainer.RUNTIME_SYNTH_CFG 未设置，请在训练脚本中赋值。")

        runtime_n = max(1, int(static_len * float(self.RUNTIME_MULTIPLIER)))

        # ✅ 只在主进程生成动态数据
        if RANK in (-1, 0):
            out_dir = self.RUNTIME_DIR
            _safe_rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            cfg_epoch = replace(cfg, out_dir=out_dir, num_images=runtime_n)
            seed = int(self.RUNTIME_SEED + epoch)

            LOGGER.info(f"[HybridPoseTrainer] Regen runtime dataset: n={runtime_n}, seed={seed}, dir={out_dir}")
            generate_dataset(cfg_epoch, seed=seed)

        runtime_img_dir = self.RUNTIME_DIR / "images"
        if not runtime_img_dir.exists():
            raise FileNotFoundError(f"runtime images dir not found: {runtime_img_dir}")

        # ✅ 用 PoseTrainer 原生 build_dataset 构建 runtime_ds（更兼容版本）
        runtime_ds = PoseTrainer.build_dataset(self, str(runtime_img_dir), mode="train", batch=batch)
        return runtime_ds

    @staticmethod
    def _on_train_epoch_start_refresh_runtime(trainer: "HybridPoseTrainer"):
        """每个 epoch 开始刷新 runtime_dataset"""
        epoch = int(getattr(trainer, "epoch", 0))

        ds = getattr(trainer, "_hybrid_dataset", None)
        if ds is None or not isinstance(ds, HybridYoloDataset):
            return

        static_len = len(ds.static_dataset)
        runtime_ds = trainer._build_runtime_dataset(batch=trainer.args.batch, static_len=static_len, epoch=epoch)
        ds.runtime_dataset = runtime_ds

        LOGGER.info(f"[HybridPoseTrainer] runtime_dataset refreshed for epoch={epoch}")
