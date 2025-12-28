from __future__ import annotations
from typing import Any, List

from .base import OCRBackend, OCRLine


from typing import List, Tuple, Any
import numpy as np

class PaddleOCRBackend(OCRBackend):
    def __init__(self, lang: str = "japan", use_angle_cls: bool = True) -> None:
        from paddleocr import PaddleOCR  # 延迟导入
        self.ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)

    def recognize(self, image_bgr) -> List[OCRLine]:
        raw = self.ocr.ocr(image_bgr)
        lines: List[OCRLine] = []

        if not raw:
            return lines

        # ===== Case A: PaddleX / 新版：list[dict] with rec_texts/rec_scores =====
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            for page in raw:
                if not isinstance(page, dict):
                    continue
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []

                if not isinstance(scores, list):
                    scores = [1.0] * len(texts)
                if len(scores) < len(texts):
                    scores = scores + [scores[-1] if scores else 1.0] * (len(texts) - len(scores))

                for t, s in zip(texts, scores):
                    if t is None:
                        continue
                    text = str(t).strip()
                    if not text:
                        continue
                    try:
                        conf = float(s)
                    except Exception:
                        conf = 1.0
                    # lines.append(OCRLine(text=text, conf=conf, box=None))
                    lines.append(OCRLine(text=text, conf=conf))

            if lines:
                return lines

        # ===== Case B: 旧版：[[[box,(text,score)],...]] 或 [[...]] =====
        candidates: Any = raw
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            candidates = raw[0]

        for item in candidates:
            text = None
            conf = None
            box = None

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0]
                second = item[1]

                if isinstance(second, (list, tuple)):
                    if len(second) >= 1:
                        text = second[0]
                    if len(second) >= 2:
                        conf = second[1]
                elif isinstance(second, dict):
                    text = second.get("text") or second.get("label")
                    conf = second.get("score") or second.get("confidence")
                elif isinstance(second, str):
                    text = second

            elif isinstance(item, dict):
                text = item.get("text") or item.get("label")
                conf = item.get("score") or item.get("confidence")

            if text is None:
                continue

            text = str(text).strip()
            if not text:
                continue

            if conf is None:
                conf = 1.0
            try:
                conf = float(conf)
            except Exception:
                conf = 1.0

            lines.append(OCRLine(text=text, conf=conf, box=box))

        return lines
