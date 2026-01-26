from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils import LOGGER

# ✅ RANK compatibility
try:
    from ultralytics.utils import RANK
except Exception:
    RANK = -1

try:
    from .hybrid_pose_trainer import HybridPoseTrainer, _safe_rmtree
    from .hybrid_yolo_dataset import HybridYoloDataset
except (ImportError, ValueError):
    from hybrid_train.hybrid_pose_trainer import HybridPoseTrainer, _safe_rmtree
    from hybrid_train.hybrid_yolo_dataset import HybridYoloDataset

class HybridPoseTrainerColab(HybridPoseTrainer):
    """
    Colab 优化的 HybridPoseTrainer:
    1. 避免在 Epoch 0 时进行重复的数据刷新（解决与 build_dataset 的冲突）。
    2. 增强文件操作的稳定性。
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._last_refresh_epoch = -1

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        覆盖父类方法，并标记 epoch 0 已刷新。
        """
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == "train":
            self._last_refresh_epoch = 0  # 标记为已在构建阶段生成了 epoch 0 数据
        return dataset

    @staticmethod
    def _on_train_epoch_start_refresh_runtime(trainer: "HybridPoseTrainerColab"):
        """每个 epoch 开始刷新 runtime_dataset，增加重复检查"""
        epoch = int(getattr(trainer, "epoch", 0))

        # ✅ 核心修复：如果是 epoch 0，且在 build_dataset 中已经刷新过了，则跳过
        if epoch == 0 and getattr(trainer, "_last_refresh_epoch", -1) == 0:
            LOGGER.info("[HybridPoseTrainerColab] Skipping refresh for epoch 0 as it was already initialized.")
            return

        # 如果当前 epoch 已经刷新过，也跳过（防止某些版本回调触发多次）
        if getattr(trainer, "_last_refresh_epoch", -1) == epoch:
            return

        ds = getattr(trainer, "_hybrid_dataset", None)
        if ds is None or not isinstance(ds, HybridYoloDataset):
            return

        static_len = len(ds.static_dataset)
        runtime_ds = trainer._build_runtime_dataset(batch=trainer.args.batch, static_len=static_len, epoch=epoch)
        ds.runtime_dataset = runtime_ds
        
        trainer._last_refresh_epoch = epoch
        LOGGER.info(f"[HybridPoseTrainerColab] runtime_dataset refreshed for epoch={epoch}")
