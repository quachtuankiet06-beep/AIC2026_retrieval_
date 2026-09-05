import csv
import json
from pathlib import Path

# Xác định đường dẫn gốc dự án từ vị trí file script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Đường dẫn dữ liệu đầu vào & đầu ra
KEYFRAMES_DIR = PROJECT_ROOT / "data" / "Custom_Keyframes" 
FPS_MAPPING_FILE = PROJECT_ROOT / "data" / "mapping" / "video_fps_mapping.json"
OUTPUT_CSV_DIR = PROJECT_ROOT / "data" / "map_csv"


def create_csv_mappings():
    # Tạo thư mục chứa file CSV xuất ra nếu chưa tồn tại
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

    if not FPS_MAPPING_FILE.exists():
        print(f"[ERROR] Không tìm thấy file: {FPS_MAPPING_FILE}")
        return

    # 1. Load danh sách FPS từ video_fps_mapping.json
    with open(FPS_MAPPING_FILE, "r", encoding="utf-8") as f:
        fps_data = json.load(f)

    # Chuyển dạng danh sách thành Dictionary dạng lookup {video_id: fps}
    fps_lookup = {}
    for item in fps_data:
        fps_lookup[item["video_id"]] = float(item["fps"])

    print(f"[INFO] Đã đọc FPS cho {len(fps_lookup)} video.")

    # 2. Duyệt qua tất cả các thư mục video trong Custom_Keyframes
    video_folders = sorted([p for p in KEYFRAMES_DIR.glob("L*") if p.is_dir()])

    if not video_folders:
        print(f"[ERROR] Không tìm thấy thư mục video tại: {KEYFRAMES_DIR}")
        return

    success_count = 0

    for video_folder in video_folders:
        video_id = video_folder.name

        if video_id not in fps_lookup:
            print(
                f"[WARNING] Bỏ qua {video_id} do không tìm thấy trong video_fps_mapping.json"
            )
            continue

        fps = fps_lookup[video_id]

        # Lấy danh sách frame_idx từ tên file ảnh
        frame_indices = []
        for img_path in video_folder.glob("*.jpg"):
            try:
                frame_idx = int(img_path.stem)
                frame_indices.append(frame_idx)
            except ValueError:
                continue

        # Sắp xếp frame_idx tăng dần
        frame_indices.sort()

        if not frame_indices:
            continue

        # 3. Ghi ra file CSV tương ứng (ví dụ: data/map_csv/L21_V001.csv)
        output_csv_path = OUTPUT_CSV_DIR / f"{video_id}.csv"

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Ghi Header chuẩn theo mẫu
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])

            for idx, frame_idx in enumerate(frame_indices, start=1):
                # Calculate pts_time
                pts_time = round(frame_idx / fps, 4)
                writer.writerow([idx, pts_time, fps, frame_idx])

        success_count += 1

    print(
        f"[SUCCESS] Đã tạo thành công {success_count} file CSV tại: {OUTPUT_CSV_DIR}"
    )


if __name__ == "__main__":
    create_csv_mappings()