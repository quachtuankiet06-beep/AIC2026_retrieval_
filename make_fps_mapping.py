import os
import pandas as pd
from pathlib import Path
import json
output_path = r"D:\tai_lieu_hoc_tap\AIC_2026\data\mapping\video_fps_mapping.json"
def extract_video_fps_mapping(mapping_dir):
    """
    Duyệt qua tất cả các file video.csv trong thư mục chỉ định,
    trích xuất video_id và fps tương ứng.
    """
    mapping_path = Path(mapping_dir)
    video_objects = []

    # Kiểm tra xem đường dẫn có tồn tại không
    if not mapping_path.exists():
        print(f"[ERROR] Thư mục không tồn tại: {mapping_dir}")
        return video_objects

    # Lọc tất cả các file có đuôi .csv (có thể tùy chỉnh pattern tên file nếu cần)
    csv_files = list(mapping_path.glob("*.csv"))
    
    print(f"[INFO] Tìm thấy {len(csv_files)} file CSV trong thư mục.")

    for csv_file in csv_files:
        try:
            # Đọc file CSV (chỉ cần đọc một vài dòng đầu để lấy fps và video_id nhanh chóng)
            # Sử dụng pandas để đọc
            df = pd.read_csv(csv_file)
            
            if df.empty:
                continue

            # Giả sử trong file csv có cột 'video_id' (hoặc lấy tên file làm video_id) và cột 'fps'
            # Bạn có thể điều chỉnh lại tên cột cho khớp với dữ liệu thực tế của bạn
            video_id = str(df.iloc[0].get("video_id", csv_file.stem))
            fps = float(df.iloc[0].get("fps", 25.0)) # Mặc định 25.0 nếu không tìm thấy cột fps

            # Tạo object (dictionary) lưu trữ
            video_obj = {
                "video_id": video_id,
                "fps": fps,
                "csv_filename": csv_file.name
            }
            
            video_objects.append(video_obj)

        except Exception as e:
            print(f"[WARNING] Lỗi khi đọc file {csv_file.name}: {e}")

    print(f"[INFO] Đã trích xuất thành công {len(video_objects)} object video.")
    return video_objects

# --- Ví dụ cách gọi hàm trong dự án của bạn ---
if __name__ == "__main__":
    dir_path = r"D:\tai_lieu_hoc_tap\AIC_2026\data\mapping\map-keyframes-aic25-b1\map-keyframes"
    
    list_of_videos = extract_video_fps_mapping(dir_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list_of_videos, f, ensure_ascii=False, indent=4)
    # In thử 5 object đầu tiên để kiểm tra kết quả
    for obj in list_of_videos[:5]:
        print(obj)