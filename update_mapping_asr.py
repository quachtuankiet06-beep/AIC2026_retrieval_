import json
from pathlib import Path
from tqdm import tqdm

# Cấu hình đường dẫn trên local
KEYFRAMES_MAPPING_PATH = Path(r"D:\tai_lieu_hoc_tap\AIC_2026\data\indexes\keyframes_mapping.json")
ASR_MAPPING_PATH = Path(r"D:\tai_lieu_hoc_tap\AIC_2026\data\indexes\asr_mapping.json")

def main():
    print("[INFO] Đang tải file ASR mapping và Keyframes mapping...")
    
    if not ASR_MAPPING_PATH.exists():
        print(f"[ERROR] Không tìm thấy file ASR: {ASR_MAPPING_PATH}")
        return
        
    if not KEYFRAMES_MAPPING_PATH.exists():
        print(f"[ERROR] Không tìm thấy file Keyframes mapping: {KEYFRAMES_MAPPING_PATH}")
        return

    # Load dữ liệu ASR
    with open(ASR_MAPPING_PATH, 'r', encoding='utf-8') as f:
        asr_data = json.load(f)

    # Load dữ liệu Keyframes Mapping
    with open(KEYFRAMES_MAPPING_PATH, 'r', encoding='utf-8') as f:
        keyframes_mapping = json.load(f)

    print(f"[INFO] Bắt đầu gán asr_text cho {len(keyframes_mapping)} keyframes...")

    updated_count = 0
    for item in tqdm(keyframes_mapping, desc="Processing ASR Enrichment"):
        video_id = item.get("video_id", "")
        pts_time = item.get("pts_time", 0.0)
        
        # Xử lý phần đuôi key (.mp4) để khớp với key trong asr_mapping
        asr_key = video_id if video_id in asr_data else f"{video_id}.mp4"
        
        asr_text = ""  # Mặc định nếu không tìm thấy lời thoại trong khoảng thời gian này
        
        if asr_key in asr_data:
            segments = asr_data[asr_key]
            # Quét tìm segment chứa mốc thời gian pts_time của keyframe
            for seg in segments:
                if seg["start"] <= pts_time <= seg["end"]:
                    asr_text = seg["text"]
                    break
        
        # Thêm hoặc cập nhật trực tiếp trường asr_text vào object
        item["asr_text"] = asr_text
        updated_count += 1

    # Ghi đè trực tiếp vào file keyframes_mapping.json gốc
    print(f"[INFO] Đang ghi đè dữ liệu cập nhật trực tiếp vào: {KEYFRAMES_MAPPING_PATH}")
    with open(KEYFRAMES_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(keyframes_mapping, f, ensure_ascii=False, indent=4)

    print(f"[SUCCESS] Hoàn tất! Đã cập nhật thành công asr_text cho {updated_count} keyframes trực tiếp vào file.")

if __name__ == "__main__":
    main()