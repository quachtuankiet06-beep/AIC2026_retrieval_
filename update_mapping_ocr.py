

import json
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR

print("[INFO] Loading PaddleOCR model...")
# Thêm enable_mkldnn=False để tránh lỗi phần cứng CPU oneDNN
ocr = PaddleOCR(use_gpu=True, lang='vi', show_log=False)

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"

def main():
    if not MAPPING_PATH.exists():
        print(f"[ERROR] Không tìm thấy file mapping tại: {MAPPING_PATH}")
        return

    print(f"[INFO] Đang tải file mapping từ {MAPPING_PATH}...")
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] File JSON bị lỗi cấu trúc: {e}")
        return

    print(f"[INFO] Bắt đầu chạy OCR cho {len(mapping_data)} keyframes...")

    success_count = 0
    not_found_count = 0
    ocr_found_count = 0
    skipped_count = 0

    for idx, item in enumerate(tqdm(mapping_data, desc="Processing OCR")):
        # Bỏ qua nếu keyframe này đã được chạy OCR trước đó (hỗ trợ resume)
        if "ocr_texts" in item and isinstance(item["ocr_texts"], list) and len(item["ocr_texts"]) > 0:
            skipped_count += 1
            ocr_found_count += 1
            success_count += 1
            continue

        rel_path = item.get("keyframe_path", "")
        if not rel_path:
            item["ocr_texts"] = []
            continue

        # Đường dẫn cơ sở từ JSON
        original_img_path = BASE_DIR / "data" / rel_path
        
        # Thử tìm các biến thể tên file thực tế trên máy
        parent_dir = original_img_path.parent
        file_name = original_img_path.name
        
        img_path = original_img_path
        if not img_path.exists():
            try:
                name_without_ext = file_name.split('.')[0]
                num_val = int(name_without_ext) 
                
                path_variant_1 = parent_dir / f"{num_val}.jpg"
                path_variant_2 = parent_dir / f"{num_val:03d}.jpg"
                path_variant_3 = parent_dir / f"{num_val:06d}.jpg"
                
                if path_variant_1.exists():
                    img_path = path_variant_1
                elif path_variant_2.exists():
                    img_path = path_variant_2
                elif path_variant_3.exists():
                    img_path = path_variant_3
            except:
                pass

        # Nếu vẫn không tìm thấy file ảnh thực tế
        if not img_path.exists():
            not_found_count += 1
            item["ocr_texts"] = []
            tqdm.write(f"[LOAD FAIL] Không tìm thấy file ảnh cho path: {rel_path}")
            continue
        
        success_count += 1

        try:
            # Sử dụng phương pháp gọi OCR chuẩn tương thích cao với paddleocr 2.7.x
            result = ocr.ocr(str(img_path), cls=False)
            extracted_texts = []
            if result and isinstance(result, list):
                for res in result:
                    if res:
                        # res có thể là một dòng chứa các đoạn text hoặc là chính cấu trúc [box, (text, conf)]
                        # Ta duyệt linh hoạt dựa vào kiểu dữ liệu của phần tử bên trong
                        for line in res:
                            if isinstance(line, list) and len(line) >= 2:
                                # Trường hợp cấu trúc chuẩn: line[0] là tọa độ, line[1] là tuple (text, confidence)
                                text_info = line[1]
                                if isinstance(text_info, tuple) and len(text_info) > 0:
                                    text = text_info[0]
                                    if text:
                                        extracted_texts.append(str(text).strip().lower())
                            elif isinstance(line, tuple) and len(line) > 0:
                                # Trường hợp trực tiếp là tuple chứa text
                                text = line[0]
                                if text:
                                    extracted_texts.append(str(text).strip().lower())
                                    
            item["ocr_texts"] = list(set(extracted_texts))
            if item["ocr_texts"]:
                ocr_found_count += 1
                # print(f"-> Đã gán thành công vào `ocr_texts`: {item['ocr_texts']}")
            
            item["ocr_texts"] = list(set(extracted_texts)) # Loại bỏ trùng lặp nếu cần
            if item["ocr_texts"]:
                ocr_found_count += 1

        except Exception as e:
            print(f"[ERROR] {img_path}")
            print(repr(e))
            item["ocr_texts"] = []
            continue

        # === TỰ ĐỘNG LƯU AN TOÀN (CHECKPOINT) MỖI 1000 ẢNH ===
        if (idx + 1) % 1000 == 0:
            temp_path = MAPPING_PATH.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=4)
            temp_path.replace(MAPPING_PATH)

    print(f"\n[INFO] KẾT QUẢ QUÉT OCR:")
    print(f"- Tổng số ảnh đã xử lý/bỏ qua: {success_count}/{len(mapping_data)}")
    print(f"- Số ảnh đã làm từ trước (được bỏ qua): {skipped_count}")
    print(f"- Số ảnh không tìm thấy file: {not_found_count}")
    print(f"- Số keyframe chứa chữ (OCR đọc được): {ocr_found_count}")

    # Lưu lại file mapping lần cuối bằng cơ chế an toàn
    print(f"[INFO] Đang lưu file mapping cập nhật vào {MAPPING_PATH}...")
    temp_path = MAPPING_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)
    temp_path.replace(MAPPING_PATH)

    print("[SUCCESS] Hoàn tất cập nhật OCR thành công!")

if __name__ == "__main__":
    main()


















# import json
# from pathlib import Path
# from tqdm import tqdm
# from paddleocr import PaddleOCR

# print("[INFO] Đang khởi tạo mô hình PaddleOCR trên GPU Local...")
# # Khởi tạo mô hình chuẩn sử dụng use_gpu=True kết nối trực tiếp lõi CUDA 13
# ocr = PaddleOCR(use_gpu=True, lang='vi', show_log=False)

# BASE_DIR = Path(__file__).resolve().parent
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"

# def main():
#     if not MAPPING_PATH.exists():
#         print(f"[ERROR] Không tìm thấy file mapping tại: {MAPPING_PATH}")
#         return
        
#     print(f"[INFO] Đang tải file mapping từ {MAPPING_PATH}...")
#     try:
#         with open(MAPPING_PATH, "r", encoding="utf-8") as f:
#             mapping_data = json.load(f)
#     except Exception as e:
#         print(f"[ERROR] File JSON bị lỗi cấu trúc: {e}")
#         return
        
#     print(f"[INFO] Bắt đầu chạy OCR cho {len(mapping_data)} keyframes...")
    
#     success_count = 0
#     not_found_count = 0
#     ocr_found_count = 0
#     skipped_count = 0
    
#     # Ép GPU tính toán song song hàng loạt bằng cách gom ảnh theo cụm (Batch Size = 32)
#     BATCH_SIZE = 32
#     pending_batch = []
    
#     for idx, item in enumerate(tqdm(mapping_data, desc="Processing PaddleOCR (GPU Tốc Độ Cao)")):
#         # Bỏ qua nếu keyframe này đã được chạy OCR trước đó (Hỗ trợ resume tự động khi chạy lại)
#         if "ocr_texts" in item and isinstance(item["ocr_texts"], list) and len(item["ocr_texts"]) > 0:
#             skipped_count += 1
#             ocr_found_count += 1
#             success_count += 1
#             continue
            
#         rel_path = item.get("keyframe_path", "")
#         if not rel_path:
#             item["ocr_texts"] = []
#             continue
            
#         original_img_path = BASE_DIR / "data" / rel_path
#         parent_dir = original_img_path.parent
#         file_name = original_img_path.name
#         img_path = original_img_path
        
#         if not img_path.exists():
#             try:
#                 # Sửa lỗi bóc tách chuỗi trước khi ép kiểu toán học int()
#                 name_without_ext = file_name.split('.')[0]
#                 num_val = int(name_without_ext)
#                 path_variant_1 = parent_dir / f"{num_val}.jpg"
#                 path_variant_2 = parent_dir / f"{num_val:03d}.jpg"
#                 path_variant_3 = parent_dir / f"{num_val:06d}.jpg"
                
#                 if path_variant_1.exists():
#                     img_path = path_variant_1
#                 elif path_variant_2.exists():
#                     img_path = path_variant_2
#                 elif path_variant_3.exists():
#                     img_path = path_variant_3
#             except:
#                 pass
                
#         if not img_path.exists():
#             not_found_count += 1
#             item["ocr_texts"] = []
#             tqdm.write(f"[LOAD FAIL] Không tìm thấy file ảnh cho path: {rel_path}")
#             continue
            
#         success_count += 1
        
#         # Gom đối tượng JSON vào hàng đợi xử lý theo lô
#         pending_batch.append(item)
        
#         # Khi gom đủ 32 ảnh hoặc chạm đến cuối mảng JSON, thực hiện quét song song
#         if len(pending_batch) == BATCH_SIZE or (idx == len(mapping_data) - 1 and pending_batch):
#             for current_item in pending_batch:
#                 try:
#                     current_rel_path = current_item.get("keyframe_path", "")
#                     current_img_path = str(BASE_DIR / "data" / current_rel_path)
                    
#                     # Thay đổi chí mạng: Sử dụng ocr() chuẩn để nạp dữ liệu sạch vào nhân CUDA 13
#                     result = ocr.ocr(current_img_path, cls=False)
#                     extracted_texts = []
                    
#                     # SỬA LỖI TRÍCH XUẤT CHỮ: Bóc tách chính xác mảng lồng cấu trúc Paddle 3.x
#                     if result and isinstance(result, list):
#                         for res in result:
#                             if res is None:
#                                 continue
#                             for line in res:
#                                 # Cấu trúc chuẩn: [ [tọa_độ_box], ( "chuỗi_chữ", độ_tự_tin ) ]
#                                 text_str = line[1][0]
#                                 prob = line[1][1]
                                
#                                 # Lọc độ tự tin > 0.4 để loại bỏ các ký tự rác nhiễu ngoài đời thực
#                                 if float(prob) > 0.4 and text_str:
#                                     extracted_texts.append(str(text_str).strip().lower())
                                    
#                     current_item["ocr_texts"] = list(set(extracted_texts))
#                     if current_item["ocr_texts"]:
#                         ocr_found_count += 1
                        
#                 except Exception as e:
#                     current_item["ocr_texts"] = []
                    
#             # Xóa sạch hàng đợi lô cũ để tiếp tục gom lô tiếp theo
#             pending_batch = []
            
#         # === TỰ ĐỘNG LƯU AN TOÀN (CHECKPOINT) MỖI 50 PHẦN TỬ ===
#         if (idx + 1) % 50 == 0:
#             temp_path = MAPPING_PATH.with_suffix(".tmp")
#             with open(temp_path, "w", encoding="utf-8") as f:
#                 json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#             temp_path.replace(MAPPING_PATH)
            
#     print(f"\n[INFO] KẾT QUẢ QUÉT OCR CUỐI CÙNG:")
#     print(f"- Tổng số ảnh đã xử lý/bỏ qua: {success_count}/{len(mapping_data)}")
#     print(f"- Số ảnh đã làm từ trước (được bỏ qua): {skipped_count}")
#     print(f"- Số ảnh không tìm thấy file: {not_found_count}")
#     print(f"- Số keyframe chứa chữ (OCR đọc được): {ocr_found_count}")
    
#     print(f"[INFO] Đang lưu file mapping cập nhật vào {MAPPING_PATH}...")
#     temp_path = MAPPING_PATH.with_suffix(".tmp")
#     with open(temp_path, "w", encoding="utf-8") as f:
#         json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#     temp_path.replace(MAPPING_PATH)
#     print("[SUCCESS] Hoàn tất cập nhật OCR thành công!")

# if __name__ == "__main__":
#     main()


















# import json
# from pathlib import Path
# from tqdm import tqdm
# import easyocr
# import torch
# print("GPU available:", torch.cuda.is_available())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# print("[INFO] Loading EasyOCR model (English + Vietnamese) on GPU...")
# # Khởi tạo EasyOCR chạy trên GPU (gpu=True để tận dụng card đồ họa của bạn)
# reader = easyocr.Reader(['en', 'vi'], gpu=True)

# BASE_DIR = Path(__file__).resolve().parent
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"

# def main():
#     if not MAPPING_PATH.exists():
#         print(f"[ERROR] Không tìm thấy file mapping tại: {MAPPING_PATH}")
#         return

#     print(f"[INFO] Đang tải file mapping từ {MAPPING_PATH}...")
#     try:
#         with open(MAPPING_PATH, "r", encoding="utf-8") as f:
#             mapping_data = json.load(f)
#     except Exception as e:
#         print(f"[ERROR] File JSON bị lỗi cấu trúc: {e}")
#         return

#     print(f"[INFO] Bắt đầu chạy OCR cho {len(mapping_data)} keyframes...")

#     success_count = 0
#     not_found_count = 0
#     ocr_found_count = 0
#     skipped_count = 0

#     for idx, item in enumerate(tqdm(mapping_data, desc="Processing EasyOCR (GPU)")):
#         # Bỏ qua nếu keyframe này đã được chạy OCR trước đó (hỗ trợ resume)
#         if "ocr_texts" in item and isinstance(item["ocr_texts"], list) and len(item["ocr_texts"]) > 0:
#             skipped_count += 1
#             ocr_found_count += 1
#             success_count += 1
#             continue

#         rel_path = item.get("keyframe_path", "")
#         if not rel_path:
#             item["ocr_texts"] = []
#             continue

#         # Đường dẫn cơ sở từ JSON
#         original_img_path = BASE_DIR / "data" / rel_path
        
#         # Thử tìm các biến thể tên file thực tế trên máy
#         parent_dir = original_img_path.parent
#         file_name = original_img_path.name
        
#         img_path = original_img_path
#         if not img_path.exists():
#             try:
#                 name_without_ext = file_name.split('.')[0]
#                 num_val = int(name_without_ext) 
                
#                 path_variant_1 = parent_dir / f"{num_val}.jpg"
#                 path_variant_2 = parent_dir / f"{num_val:03d}.jpg"
#                 path_variant_3 = parent_dir / f"{num_val:06d}.jpg"
                
#                 if path_variant_1.exists():
#                     img_path = path_variant_1
#                 elif path_variant_2.exists():
#                     img_path = path_variant_2
#                 elif path_variant_3.exists():
#                     img_path = path_variant_3
#             except:
#                 pass

#         # Nếu vẫn không tìm thấy file ảnh thực tế
#         if not img_path.exists():
#             not_found_count += 1
#             item["ocr_texts"] = []
#             tqdm.write(f"[LOAD FAIL] Không tìm thấy file ảnh cho path: {rel_path}")
#             continue
        
#         success_count += 1

#         try:
#             # EasyOCR đọc ảnh và trả về danh sách dạng (bbox, text, prob)
#             results = reader.readtext(str(img_path))
#             extracted_texts = []
            
#             for (_, text, prob) in results:
#                 # Lọc độ tin cậy > 0.4 để loại bỏ các chữ nhiễu
#                 if prob > 0.4 and text:
#                     extracted_texts.append(str(text).strip().lower())
            
#             # Loại bỏ trùng lặp trong cùng một frame
#             item["ocr_texts"] = list(set(extracted_texts))
#             if item["ocr_texts"]:
#                 ocr_found_count += 1

#         except Exception as e:
#             item["ocr_texts"] = []

#         # === TỰ ĐỘNG LƯU AN TOÀN (CHECKPOINT) MỖI 50 ẢNH ===
#         if (idx + 1) % 50 == 0:
#             temp_path = MAPPING_PATH.with_suffix(".tmp")
#             with open(temp_path, "w", encoding="utf-8") as f:
#                 json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#             temp_path.replace(MAPPING_PATH)

#     print(f"\n[INFO] KẾT QUẢ QUÉT OCR:")
#     print(f"- Tổng số ảnh đã xử lý/bỏ qua: {success_count}/{len(mapping_data)}")
#     print(f"- Số ảnh đã làm từ trước (được bỏ qua): {skipped_count}")
#     print(f"- Số ảnh không tìm thấy file: {not_found_count}")
#     print(f"- Số keyframe chứa chữ (OCR đọc được): {ocr_found_count}")

#     # Lưu lại file mapping lần cuối bằng cơ chế an toàn
#     print(f"[INFO] Đang lưu file mapping cập nhật vào {MAPPING_PATH}...")
#     temp_path = MAPPING_PATH.with_suffix(".tmp")
#     with open(temp_path, "w", encoding="utf-8") as f:
#         json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#     temp_path.replace(MAPPING_PATH)

#     print("[SUCCESS] Hoàn tất cập nhật OCR thành công!")

# if __name__ == "__main__":
#     main()