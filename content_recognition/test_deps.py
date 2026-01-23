print("Start test")
import easyocr
print("EasyOCR imported")
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
print("EasyOCR initialized")
from ultralytics import YOLO
print("YOLO imported")
model = YOLO("content_recognition/runs/train_yolo11s_bs8_img960/weights/best.pt")
print("YOLO model loaded")
