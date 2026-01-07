from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from torch.utils.data import Dataset


@dataclass
class HybridYoloDataset(Dataset):
    """
    混合数据集：严格 50/50
    - 偶数 index -> static_dataset
    - 奇数 index -> runtime_dataset

    同时做 Ultralytics 兼容：
    - 提供 .labels 给 plot_training_labels() 使用
    - 通过 __getattr__ 代理常用属性（names, nc, imgsz, etc.）
    """

    static_dataset: Dataset
    runtime_dataset: Dataset

    def __len__(self) -> int:
        # epoch_size 取两者最大值，然后 *2，保证偶/奇数量一样
        epoch_size = max(len(self.static_dataset), len(self.runtime_dataset))
        return epoch_size * 2

    def __getitem__(self, index: int) -> Any:
        half = index // 2
        if index % 2 == 0:
            j = half % len(self.static_dataset)
            return self.static_dataset[j]
        else:
            j = half % len(self.runtime_dataset)
            return self.runtime_dataset[j]

    @property
    def collate_fn(self):
        # 复用 YOLODataset 的 collate_fn
        return getattr(self.static_dataset, "collate_fn", None)

    @property
    def labels(self) -> List[Any]:
        """
        Ultralytics 在 _setup_train -> plot_training_labels() 会访问 train_loader.dataset.labels
        这里把 static + runtime 的 labels 拼在一起。
        注意：runtime_dataset 会在每个 epoch start 被替换，这个属性会自动反映最新 runtime。
        """
        out: List[Any] = []
        if hasattr(self.static_dataset, "labels"):
            out.extend(getattr(self.static_dataset, "labels"))
        if hasattr(self.runtime_dataset, "labels"):
            out.extend(getattr(self.runtime_dataset, "labels"))
        return out

    def __getattr__(self, name: str):
        """
        代理静态/动态 dataset 的属性，提升兼容性（例如 names/nc/stride 等）
        """
        if hasattr(self.static_dataset, name):
            return getattr(self.static_dataset, name)
        if hasattr(self.runtime_dataset, name):
            return getattr(self.runtime_dataset, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
