from __future__ import annotations
import json
import cv2
import easyocr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

# ==========================================
# Main exports
# ==========================================

def load_content_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def load_ocr_reader(languages: List[str] = ['ja', 'en'], gpu: bool = True) -> easyocr.Reader:
    return easyocr.Reader(languages, gpu=gpu)

def recognize_content(
    image_np: np.ndarray,
    model: YOLO,
    reader: easyocr.Reader,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "0"
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Returns (annotated_image, detections_list)
    """
    # YOLO Inference
    results = model.predict(
        source=image_np,
        device=device,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )
    
    result = results[0]
    
    # Draw annotations on a copy
    annotated_img = result.plot()
    
    detections = []
    
    # Process detections
    boxes = result.boxes
    if boxes is None:
        return annotated_img, []

    for idx, box in enumerate(boxes):
        # Coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Clip to image bounds
        h, w = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Class and conf
        cls_id = int(box.cls[0].item())
        if result.names:
            cls_name = result.names[cls_id]
        else:
            cls_name = str(cls_id)
            
        conf = float(box.conf[0].item())
        
        # Crop for OCR
        crop_img = image_np[y1:y2, x1:x2]
        
        ocr_text = ""
        if crop_img.size > 0:
            # OCR
            try:
                # detail=0 returns simple list of strings
                ocr_result = reader.readtext(crop_img, detail=0)
                ocr_text = " ".join(ocr_result)
            except Exception as e:
                print(f"OCR Error: {e}")
                ocr_text = ""
        
        detections.append({
            "class": cls_name,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 4),
            "ocr_text": ocr_text,
            # We could return the crop image too if needed for display
            "crop_img": crop_img 
        })
        
    return annotated_img, detections
