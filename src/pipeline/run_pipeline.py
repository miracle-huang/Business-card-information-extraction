from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import cv2
import numpy as np

from ultralytics import YOLO

from .geometry import DetBox, pad_box, crop_bgr, reading_order_sort
from .postprocess import postprocess_fields
from ocr.base import OCRBackend


@dataclass
class PipelineConfig:
    weights: str
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    pad_ratio: float = 0.03
    save_crops: bool = False


def yolo_detect(model: YOLO, image_bgr: np.ndarray, names: List[str], cfg: PipelineConfig) -> List[DetBox]:
    results = model.predict(
        source=image_bgr,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        verbose=False,
    )
    r0 = results[0]
    boxes = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return boxes

    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)

    h, w = image_bgr.shape[:2]
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        cls_name = names[k] if 0 <= k < len(names) else str(k)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        x1i, y1i, x2i, y2i = pad_box(x1i, y1i, x2i, y2i, cfg.pad_ratio, w, h)
        boxes.append(DetBox(cls_name=cls_name, conf=float(c), xyxy=(x1i, y1i, x2i, y2i)))
    return boxes


def run_on_image(
    image_path: Path,
    model: YOLO,
    names: List[str],
    ocr: OCRBackend,
    cfg: PipelineConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    dets = yolo_detect(model, image_bgr, names, cfg)

    # 按类别收集文本（同类可能多个框）
    raw_texts: Dict[str, List[str]] = {n: [] for n in names}
    raw_details: List[Dict[str, Any]] = []

    # 同类框阅读顺序拼接更合理：先按类分组再排序
    by_cls: Dict[str, List[DetBox]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_name, []).append(d)

    crops_dir = out_dir / "crops" / image_path.stem
    if cfg.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for cls_name, boxes in by_cls.items():
        boxes_sorted = reading_order_sort(boxes)
        for idx, b in enumerate(boxes_sorted):
            crop = crop_bgr(image_bgr, b.xyxy)
            lines = ocr.recognize(crop)
            text = " ".join([ln.text for ln in lines]).strip()

            if text:
                raw_texts[cls_name].append(text)

            detail = {
                "cls": cls_name,
                "det_conf": b.conf,
                "xyxy": list(b.xyxy),
                "ocr_text": text,
                "ocr_lines": [{"text": ln.text, "conf": ln.conf} for ln in lines],
            }
            raw_details.append(detail)

            if cfg.save_crops:
                cv2.imwrite(str(crops_dir / f"{cls_name}_{idx}.jpg"), crop)

    fields = postprocess_fields(raw_texts)

    out = {
        "image": image_path.name,
        "fields": fields,
        "raw_texts": raw_texts,
        "details": raw_details,
    }
    return out


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")