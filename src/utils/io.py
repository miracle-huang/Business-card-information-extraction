from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TrainConfig:
    data_yaml: str
    task: str = "detect"
    model: str = "yolo11n.pt"
    imgsz: int = 320
    epochs: int = 1
    batch: int = 1
    device: str | int = 0
    workers: int = 0
    seed: int = 42

    project: str = "runs_local"
    name: str = "debug"
    exist_ok: bool = True

    optimizer: str = "auto"
    lr0: Optional[float] = None
    freeze: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        return TrainConfig(**d)