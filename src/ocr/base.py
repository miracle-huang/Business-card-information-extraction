from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Any


@dataclass
class OCRLine:
    text: str
    conf: float


class OCRBackend(Protocol):
    def recognize(self, image_bgr: Any) -> List[OCRLine]:
        """Input: BGR image (numpy array). Output: list of OCR lines."""
        ...