import torch
import easyocr

def check_device():
    print(f"--- Environment Check ---")
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n--- EasyOCR Init Test ---")
    try:
        # 尝试初始化并开启 GPU
        reader = easyocr.Reader(['en'], gpu=True)
        
        # 检查内部模型的设备。如果使用了 DataParallel，需要从 module 中获取
        if hasattr(reader.detector, 'device'):
            model_device = reader.detector.device
        else:
            # 常见于 DataParallel 包装的情况
            model_device = next(reader.detector.parameters()).device
            
        print(f"EasyOCR is using: {model_device}")
        
        if "cuda" in str(model_device):
            print("✅ EasyOCR IS using GPU.")
        else:
            print("⚠️ EasyOCR is NOT using GPU (falling back to CPU).")
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")

if __name__ == "__main__":
    check_device()
