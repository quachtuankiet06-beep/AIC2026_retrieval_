import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tqdm import tqdm
from paddleocr import PaddleOCR

print("[INFO] Loading PaddleOCR model...")
# Thêm enable_mkldnn=False để tránh lỗi phần cứng CPU oneDNN
ocr = PaddleOCR(use_gpu=True, lang='vi', show_log=False)

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping_new.json"

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
        # Bỏ qua nếu keyframe này đã được chạy OCR và cấu trúc mới đã tồn tại (hỗ trợ resume)
        # if "ocr_texts" in item and isinstance(item["ocr_texts"], list) and len(item["ocr_texts"]) > 0:
        #     # Kiểm tra nếu phần tử đầu tiên là dict (đã đúng định dạng mới) thì mới skip
        #     if isinstance(item["ocr_texts"][0], dict):
        #         skipped_count += 1
        #         ocr_found_count += 1
        #         success_count += 1
        #         continue

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
            # Gọi PaddleOCR
            result = ocr.ocr(str(img_path), cls=False)
            extracted_list = []
            
            if result and isinstance(result, list):
                for res in result:
                    if res:
                        for line in res:
                            if isinstance(line, list) and len(line) >= 2:
                                text_info = line[1]
                                if isinstance(text_info, tuple) and len(text_info) >= 2:
                                    text = text_info[0]
                                    conf = text_info[1]
                                    if text:
                                        extracted_list.append({
                                            "text": str(text).strip().lower(),
                                            "score": round(float(conf), 4)
                                        })
                            elif isinstance(line, tuple) and len(line) >= 2:
                                text = line[0]
                                conf = line[1]
                                if text:
                                    extracted_list.append({
                                        "text": str(text).strip().lower(),
                                        "score": round(float(conf), 4)
                                    })
                                    
            # Lưu thẳng vào item mà không qua bước lọc trùng cứng nhắc nào cả
            item["ocr_texts"] = extracted_list
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

# print("[INFO] Loading PaddleOCR PP-OCRv6 model...")

# ocr = PaddleOCR(
#     lang="vi",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )

# BASE_DIR = Path(__file__).resolve().parent
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping_new copy.json"


# def extract_ocr_texts_from_result(result_obj):
#     extracted_list = []

#     # PaddleOCR 3.x: Result object -> json dict
#     data = result_obj.json if hasattr(result_obj, "json") else {}
#     res = data.get("res", {}) if isinstance(data, dict) else {}

#     texts = res.get("rec_texts", []) or []
#     scores = res.get("rec_scores", []) or []

#     for i, text in enumerate(texts):
#         if not text:
#             continue

#         score = scores[i] if i < len(scores) else 0.0
#         extracted_list.append({
#             "text": str(text).strip().lower(),
#             "score": round(float(score), 4),
#         })

#     return extracted_list


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

#     print(f"[INFO] Bắt đầu chạy OCR (PP-OCRv6) cho {len(mapping_data)} keyframes...")

#     success_count = 0
#     not_found_count = 0
#     ocr_found_count = 0
#     skipped_count = 0

#     for idx, item in enumerate(tqdm(mapping_data, desc="Processing OCR")):
#         # CHECKPOINT RESUME: Chỉ bỏ qua khi đã có "ocr_texts" và mảng không bị rỗng
#         if "ocr_texts" in item and item.get("ocr_texts"):
#             skipped_count += 1
#             ocr_found_count += 1
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

#         try:
#             # PaddleOCR 3.x: dùng predict(), không truyền cls=False nữa
#             result = ocr.predict(str(img_path))

#             extracted_list = []
#             for res in result:
#                 extracted_list.extend(extract_ocr_texts_from_result(res))

#             item["ocr_texts"] = extracted_list
#             if item["ocr_texts"]:
#                 ocr_found_count += 1

#         except Exception as e:
#             print(f"[ERROR] {img_path}")
#             print(repr(e))
#             item["ocr_texts"] = []
#             continue

#         if (idx + 1) % 1000 == 0:
#             temp_path = MAPPING_PATH.with_suffix(".tmp")
#             with open(temp_path, "w", encoding="utf-8") as f:
#                 json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#             temp_path.replace(MAPPING_PATH)

#     print(f"\n[INFO] KẾT QUẢ QUÉT OCR:")
#     print(f"- Tổng số ảnh đã xử lý: {success_count}/{len(mapping_data)}")
#     print(f"- Số ảnh không tìm thấy file: {not_found_count}")
#     print(f"- Số keyframe chứa chữ (OCR đọc được): {ocr_found_count}")

#     print(f"[INFO] Đang lưu file mapping cập nhật vào {MAPPING_PATH}...")
#     temp_path = MAPPING_PATH.with_suffix(".tmp")
#     with open(temp_path, "w", encoding="utf-8") as f:
#         json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#     temp_path.replace(MAPPING_PATH)

#     print("[SUCCESS] Hoàn tất cập nhật OCR PP-OCRv6 thành công!")


# if __name__ == "__main__":
#     main()





# import json
# from pathlib import Path
# from tqdm import tqdm
# from paddleocr import PaddleOCR

# print("[INFO] Loading PaddleOCR PP-OCRv6 model...")

# # Khởi tạo PaddleOCR (Đảm bảo môi trường paddlepaddle-gpu đã được cài đặt)
# ocr = PaddleOCR(
#     lang="vi",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     # use_gpu=True # Bật nếu phiên bản paddle của bạn yêu cầu tường minh
# )

# BASE_DIR = Path(__file__).resolve().parent
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping_new copy.json"
# BATCH_SIZE = 16  # Tăng/giảm tùy thuộc vào VRAM thực tế (8GB có thể thử từ 16 đến 32)

# def extract_ocr_texts_from_result(result_obj):
#     extracted_list = []
#     data = result_obj.json if hasattr(result_obj, "json") else {}
#     res = data.get("res", {}) if isinstance(data, dict) else {}

#     texts = res.get("rec_texts", []) or []
#     scores = res.get("rec_scores", []) or []

#     for i, text in enumerate(texts):
#         if not text:
#             continue
#         score = scores[i] if i < len(scores) else 0.0
#         extracted_list.append({
#             "text": str(text).strip().lower(),
#             "score": round(float(score), 4),
#         })
#     return extracted_list

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

#     print(f"[INFO] Bắt đầu quét OCR theo lô (Batch Size = {BATCH_SIZE}) cho {len(mapping_data)} keyframes...")

#     success_count = 0
#     not_found_count = 0
#     ocr_found_count = 0
#     skipped_count = 0

#     # Gom các item cần xử lý vào danh sách batch
#     pending_items = []
    
#     def flush_batch():
#         nonlocal success_count, ocr_found_count
#         if not pending_items:
#             return

#         batch_paths = [str(item["_resolved_path"]) for item, _ in pending_items]
        
#         try:
#             # PaddleOCR hỗ trợ truyền danh sách đường dẫn để chạy batch
#             results = ocr.predict(batch_paths)
            
#             for (item, _), result in zip(pending_items, results):
#                 extracted_list = []
#                 # Đảm bảo xử lý đúng cấu trúc trả về dạng list hoặc single result theo batch
#                 res_iter = result if isinstance(result, list) else [result]
#                 for res in res_iter:
#                     extracted_list.extend(extract_ocr_texts_from_result(res))

#                 item["ocr_texts"] = extracted_list
#                 success_count += 1
#                 if item["ocr_texts"]:
#                     ocr_found_count += 1
#         except Exception as e:
#             print(f"\n[ERROR] Lỗi khi xử lý batch: {repr(e)}")
#             # Fallback chạy từng cái nếu batch lỗi đột xuất
#             for item, img_path in pending_items:
#                 try:
#                     res = ocr.predict(str(img_path))
#                     extracted_list = []
#                     res_iter = res if isinstance(res, list) else [res]
#                     for r in res_iter:
#                         extracted_list.extend(extract_ocr_texts_from_result(r))
#                     item["ocr_texts"] = extracted_list
#                     success_count += 1
#                     if item["ocr_texts"]:
#                         ocr_found_count += 1
#                 except Exception as sub_e:
#                     print(f"[ERROR] Fallback lỗi với {img_path}: {repr(sub_e)}")
#                     item["ocr_texts"] = []

#         pending_items.clear()

#     for idx, item in enumerate(tqdm(mapping_data, desc="Processing OCR Batch")):
#         # CHECKPOINT RESUME: Chỉ bỏ qua khi đã có "ocr_texts" và mảng không bị rỗng
#         if "ocr_texts" in item and item.get("ocr_texts"):
#             skipped_count += 1
#             ocr_found_count += 1
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
#             continue

#         # Lưu đường dẫn đã resolve tạm vào item để dùng cho batch
#         item["_resolved_path"] = img_path
#         pending_items.append((item, img_path))

#         if len(pending_items) >= BATCH_SIZE:
#             flush_batch()

#         # Lưu định kỳ mỗi 1000 items
#         if (idx + 1) % 1000 == 0:
#             flush_batch()
#             # Dọn dẹp key tạm trước khi lưu json
#             for it in mapping_data:
#                 it.pop("_resolved_path", None)
                
#             temp_path = MAPPING_PATH.with_suffix(".tmp")
#             with open(temp_path, "w", encoding="utf-8") as f:
#                 json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#             temp_path.replace(MAPPING_PATH)

#     # Xốt nốt phần dư trong batch cuối
#     flush_batch()

#     # Dọn dẹp key tạm cuối cùng
#     for it in mapping_data:
#         it.pop("_resolved_path", None)

#     print(f"\n[INFO] KẾT QUẢ QUÉT OCR:")
#     print(f"- Tổng số ảnh đã xử lý: {success_count}/{len(mapping_data)}")
#     print(f"- Số ảnh không tìm thấy file: {not_found_count}")
#     print(f"- Số keyframe chứa chữ (OCR đọc được): {ocr_found_count}")

#     print(f"[INFO] Đang lưu file mapping cập nhật vào {MAPPING_PATH}...")
#     temp_path = MAPPING_PATH.with_suffix(".tmp")
#     with open(temp_path, "w", encoding="utf-8") as f:
#         json.dump(mapping_data, f, ensure_ascii=False, indent=4)
#     temp_path.replace(MAPPING_PATH)

#     print("[SUCCESS] Hoàn tất cập nhật OCR PP-OCRv6 thành công!")

# if __name__ == "__main__":
#     main()