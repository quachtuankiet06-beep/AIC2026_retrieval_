from pathlib import Path
from paddleocr import PaddleOCR

print("[INFO] Đang tải mô hình PaddleOCR...")
# Khởi tạo mô hình
ocr = PaddleOCR(lang='vi', use_gpu=False, enable_mkldnn=False)
# Đường dẫn ảnh mẫu của bạn
img_path = Path(r"D:\tai_lieu_hoc_tap\AIC_2026\data\keyframes\Keyframes_L22\keyframes\L22_V001\166.jpg")

if not img_path.exists():
    print(f"[ERROR] Không tìm thấy file ảnh tại: {img_path}")
else:
    print(f"[SUCCESS] Đã tìm thấy ảnh: {img_path}")
    print("[INFO] Đang chạy OCR bằng ocr.predict()...")
    
    try:
        # Sử dụng hàm predict() mới của phiên bản này (không truyền cls)
        result = ocr.ocr(str(img_path), cls=False)
        
        print("\n--- KẾT QUẢ THÔ TỪ PADDLEOCR ---")
        print(result)
        print("--------------------------------\n")
        
        extracted_texts = []
        if result and isinstance(result, list) and len(result) > 0:
            res_dict = result[0]
            if isinstance(res_dict, dict):
                texts = res_dict.get("rec_texts", [])
                for text in texts:
                    if text:
                        extracted_texts.append(str(text).strip().lower())
                            
        print(f"-> Mảng `ocr_texts` thu được: {extracted_texts}")
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi chạy OCR: {e}")