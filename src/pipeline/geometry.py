from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


@dataclass
class DetBox:
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def pad_box(x1: int, y1: int, x2: int, y2: int, pad_ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def crop_bgr(image_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return image_bgr[y1:y2, x1:x2].copy()


def reading_order_sort(boxes: List[DetBox]) -> List[DetBox]:
    # 先按 y_center 再按 x_center
    def key_fn(b: DetBox):
        x1, y1, x2, y2 = b.xyxy
        return ((y1 + y2) / 2.0, (x1 + x2) / 2.0)
    return sorted(boxes, key=key_fn)
