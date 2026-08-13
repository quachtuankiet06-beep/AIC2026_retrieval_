import json
from pathlib import Path
from tqdm import tqdm

def filter_and_normalize_mapping(input_json_path, output_json_path, root_dir):
    """
    Đọc file mapping gốc, chuẩn hóa lại tên file ảnh thành xxx.jpg dựa trên keyframe_index,
    kiểm tra sự tồn tại thực tế trên ổ cứng, và re-index lại vector_index liên tục từ 0.
    """
    input_path = Path(input_json_path)
    output_path = Path(output_json_path)
    root_dir = Path(root_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Không tìm thấy file gốc: {input_json_path}")
        return

    print("Đang đọc file JSON gốc...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    current_vector_index = 0
    missing_count = 0

    print("Đang kiểm tra, chuẩn hóa định dạng ảnh và re-index...")
    for entry in tqdm(data):
        # Lấy các thông tin cần thiết từ entry cũ
        video_id = entry.get("video_id")
        keyframe_index = entry.get("keyframe_index") # hoặc dùng frame_idx tùy ý bạn
        
        if not video_id or keyframe_index is None:
            continue

        # Lấy batch_name từ video_id (ví dụ: L21 từ L21_V001)
        batch_name = video_id.split("_")[0] if "_" in video_id else "Keyframes_L21"

        # --- CHUẨN HÓA TÊN ẢNH THÀNH xxx.jpg (VD: 1 -> 001.jpg, 27 -> 027.jpg) ---
        normalized_img_name = f"{int(keyframe_index):03d}.jpg"

        # Xây dựng lại keyframe_path theo đúng chuẩn đường dẫn tương đối mới
        # Ví dụ: keyframes/Keyframes_L21/keyframes/L21_V001/001.jpg
        new_keyframe_path = (
            f"keyframes/"
            f"Keyframes_{batch_name}/"
            f"keyframes/"
            f"{video_id}/"
            f"{normalized_img_name}"
        )

        # Đường dẫn tuyệt đối để kiểm tra sự tồn tại trên ổ cứng
        abs_path = root_dir / new_keyframe_path

        if abs_path.exists():
            # Cập nhật lại đường dẫn ảnh đã chuẩn hóa vào entry
            entry["keyframe_path"] = new_keyframe_path
            # Gán lại vector_index tuần tự mới
            entry["vector_index"] = current_vector_index
            
            new_data.append(entry)
            current_vector_index += 1
        else:
            # Nếu file ảnh không tồn tại trên ổ cứng, bỏ qua
            missing_count += 1

    # Lưu kết quả ra file JSON mới
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 60)
    print("[SUCCESS] Hoàn tất quá trình lọc và chuẩn hóa mapping!")
    print("=" * 60)
    print(f"Tổng số record cũ        : {len(data)}")
    print(f"Số lượng ảnh bị thiếu/bỏ : {missing_count}")
    print(f"Tổng số record hợp lệ mới: {len(new_data)}")
    print(f"Đã lưu file mới tại      : {output_path}")

# --- CẤU HÌNH ĐƯỜNG DẪN THỰC TẾ ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent # Điều chỉnh theo thư mục project của bạn

    INPUT_FILE = BASE_DIR / "data" / "indexes" / "keyframes_mapping_vitb32.json"  # File cũ (vit_b32 hoặc tương tự)
    OUTPUT_FILE = BASE_DIR / "data" / "indexes" / "keyframes_mapping_siglib2.json" # File mới xuất ra cho SigLIP2
    ROOT_DIR = BASE_DIR / "data"                                          # Thư mục cha chứa folder 'keyframes'

    filter_and_normalize_mapping(INPUT_FILE, OUTPUT_FILE, ROOT_DIR)