from __future__ import annotations
import argparse
import json
import cv2
import easyocr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# 用户配置区域 (USER CONFIGURATION)
# 你可以在这里直接修改默认值，这样就不需要每次都在命令行输入参数
# ==========================================
CFG_MODEL = "content_recognition/runs/train_yolo11s_bs8_img960/weights/best.pt"   # YOLO 模型路径
CFG_SOURCE = "data/business_card_v2/test/images"                                # 输入图片路径 (文件或文件夹)
CFG_OUTPUT = "content_recognition/runs/detect/ocr_result"                       # 结果保存路径
CFG_DEVICE = "0"                                                                # 设备: "0", "1", "cpu"
CFG_CONF = 0.25                                                                 # 置信度阈值
CFG_IOU = 0.45                                                                  # NMS IOU 阈值
# ==========================================


def run_prediction(
    model_path: str,
    source: str,
    output_dir: str,
    device: str = "0",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
):
    # 1. 初始化路径
    source_path = Path(source)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    crops_dir = out_path / "crops"
    crops_dir.mkdir(exist_ok=True)

    # 2. 加载模型
    print(f"[Init] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # 3. 初始化 EasyOCR
    print("[Init] Loading EasyOCR reader...")
    # 识别中文简繁体和英文
    reader = easyocr.Reader(['ja', 'en'], gpu=(device != 'cpu'))



    # 4. 获取图片列表
    if source_path.is_file():
        image_files = [source_path]
    else:
        # 支持常见图片格式
        image_files = sorted([p for p in source_path.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]])

    if not image_files:
        print(f"[Error] No images found in {source}")
        return

    print(f"[Process] Found {len(image_files)} images. Starting inference...")

    # 5. 循环处理
    for img_file in tqdm(image_files):
        # 读取图片
        # cv2.imread 读取可能会有中文路径问题，建议用 numpy + cv2.imdecode
        img_np = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            print(f"[Warning] Failed to read image: {img_file}")
            continue
            
        # YOLO 推理
        results = model.predict(
            source=img_np,
            device=device,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )
        
        result = results[0]  # 单张图片
        image_name = img_file.name
        image_stem = img_file.stem
        
        # 保存带框的原图
        annotated_img = result.plot()
        annotated_save_path = out_path / f"{image_stem}_detected.jpg"
        # cv2.imwrite 同样可能有中文路径问题
        is_success, im_buf = cv2.imencode(".jpg", annotated_img)
        if is_success:
            im_buf.tofile(str(annotated_save_path))
        
        # 准备 JSON 数据
        json_data = {
            "image_name": image_name,
            "detections": []
        }
        
        # 遍历检测到的物体
        boxes = result.boxes
        for idx, box in enumerate(boxes):
            # 获取坐标 (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 边界保护
            h, w = img_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 类别和置信度
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())
            
            # 裁剪图片
            crop_img = img_np[y1:y2, x1:x2]
            
            if crop_img.size == 0:
                continue
                
            # 保存裁剪图片
            crop_filename = f"{image_stem}_{idx}_{cls_name}.jpg"
            crop_path = crops_dir / crop_filename
            is_success, crop_buf = cv2.imencode(".jpg", crop_img)
            if is_success:
                crop_buf.tofile(str(crop_path))
            
            # OCR 识别
            # detail=0 只返回文本列表
            ocr_result = reader.readtext(crop_img, detail=0)
            ocr_text = " ".join(ocr_result)
            
            # 添加到结果
            detection_info = {
                "class": cls_name,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4),
                "crop_path": str(crop_path.relative_to(out_path)), # 相对路径，方便移植
                "ocr_text": ocr_text
            }
            json_data["detections"].append(detection_info)

        # 保存 JSON
        json_save_path = out_path / f"{image_stem}.json"
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            
    print(f"[Done] Results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + EasyOCR Inference")
    parser.add_argument("--model", type=str, default=CFG_MODEL, help="Path to .pt model")
    parser.add_argument("--source", type=str, default=CFG_SOURCE, help="Input images directory or file")
    parser.add_argument("--save_dir", type=str, default=CFG_OUTPUT, help="Output directory")
    parser.add_argument("--device", type=str, default=CFG_DEVICE, help="Device (0, cpu)")
    parser.add_argument("--conf", type=float, default=CFG_CONF, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=CFG_IOU, help="IoU threshold")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"[Error] Model file not found: {args.model}")
        exit(1)
        
    run_prediction(
        model_path=args.model,
        source=args.source,
        output_dir=args.save_dir,
        device=args.device,
        conf_thres=args.conf,
        iou_thres=args.iou
    )
