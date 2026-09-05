import json
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ==========================================================
# Build Mapping Database cho Custom Keyframes (Đã tối ưu tự động bắt path)
# ==========================================================

def build_custom_mapping_database(
    custom_keyframes_dir,
    map_keyframes_dir,
    metadata_root,
    output_json_path,
):
    custom_keyframes_dir = Path(custom_keyframes_dir)
    map_keyframes_dir = Path(map_keyframes_dir)
    metadata_root = Path(metadata_root)
    output_json_path = Path(output_json_path)

    # Tự động xử lý trường hợp bị lồng thư mục Custom_Keyframes/Custom_Keyframes
    actual_custom_dir = custom_keyframes_dir
    nested_dir = custom_keyframes_dir / "Custom_Keyframes"
    if nested_dir.exists() and nested_dir.is_dir():
        actual_custom_dir = nested_dir

    # Lấy danh sách tất cả các thư mục video có mặt trong Custom Keyframes (bắt đầu bằng L)
    video_folders = sorted([p for p in actual_custom_dir.glob("L*") if p.is_dir()])

    print(f"[INFO] Tìm thấy {len(video_folders)} thư mục video custom tại: {actual_custom_dir}")

    mapping_database = []
    vector_index = 0

    for video_folder in tqdm(video_folders, desc="Building Mapping"):
        video_id = video_folder.name

        # 1. Đọc file CSV map tương ứng cho video này
        csv_path = map_keyframes_dir / f"{video_id}.csv"

        if not csv_path.exists():
            print(f"[WARNING] Không tìm thấy file CSV cho : {video_id}, bỏ qua.")
            continue

        df = pd.read_csv(csv_path)

        # 2. Đọc Metadata của video (nếu có)
        metadata_json = metadata_root / f"{video_id}.json"
        metadata = {}

        if metadata_json.exists():
            try:
                with open(metadata_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                metadata = {
                    "title": meta.get("title", ""),
                    "description": meta.get("description", ""),
                    "keywords": meta.get("keywords", []),
                    "publish_date": meta.get("publish_date", "")
                }
            except Exception:
                pass

        # 3. Duyệt qua từng hàng trong file CSV tương ứng với từng keyframe thực tế
        for _, row in df.iterrows():
            n = int(row["n"])
            keyframe_index = n 
            frame_idx = int(row["frame_idx"])
            pts_time = float(row["pts_time"])
            fps = float(row["fps"])
            
            # Kiểm tra file ảnh có thực sự tồn tại trong thư mục video không
            img_abs_path = video_folder / f"{frame_idx}.jpg"
            if not img_abs_path.exists():
                img_abs_path = video_folder / f"{frame_idx:06d}.jpg"

            # Tạo đường dẫn tương đối chuẩn cho keyframe_path
            try:
                rel_img_path = img_abs_path.relative_to(custom_keyframes_dir.parent)
            except ValueError:
                rel_img_path = img_abs_path.relative_to(custom_keyframes_dir)

            # 4. Gom record cho từng frame
            mapping_database.append({
                "vector_index": vector_index,
                "video_id": video_id,
                "keyframe_index": keyframe_index,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "fps": fps,
                "keyframe_path": str(rel_img_path).replace("\\", "/"), 
                "object_path": "",
                "metadata_path": f"data/metadata/media-info/{video_id}.json",
                "object_entities": [],
                "metadata": metadata,
            })

            vector_index += 1

    # =======================================================
    # Lưu kết quả ra file JSON
    # =======================================================
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_database, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("[SUCCESS] Custom Keyframes Mapping Database Created")
    print("=" * 70)
    print(f"Total Records : {len(mapping_database)}")
    print(f"Output Path   : {output_json_path}")


# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    build_custom_mapping_database(
        custom_keyframes_dir=
            BASE_DIR
            / "data"
            / "Custom_Keyframes",

        map_keyframes_dir=
            BASE_DIR
            / "data"
            / "map_csv",

        metadata_root=
            BASE_DIR
            / "data"
            / "metadata"
            / "media-info",

        output_json_path=
            BASE_DIR
            / "data"
            / "indexes"
            / "keyframes_mapping_new.json",
    )