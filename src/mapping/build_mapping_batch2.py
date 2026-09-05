import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def append_batch2_mapping(
    existing_json_path,
    map_keyframes_dir_b2,
    objects_root_b2,
    metadata_root,
    output_json_path,
    objects_rel_prefix="data/object/objects-aic25-b2/objects",
):
    """
    Append Batch 2 keyframes metadata to existing keyframes_mapping.json.
    Dựa trực tiếp vào các file CSV mapping keyframes trên ổ cứng thay vì clip_features.

    Args:
        existing_json_path: Đường dẫn tới keyframes_mapping.json hiện tại (Batch 1).
        map_keyframes_dir_b2: Đường dẫn tới thư mục chứa file .csv keyframes của Batch 2.
        objects_root_b2: Đường dẫn thư mục chứa object detection JSON của Batch 2.
        metadata_root: Đường dẫn thư mục chứa metadata (media-info).
        output_json_path: Đường dẫn lưu file keyframes_mapping_new.json mới.
        objects_rel_prefix: Đường dẫn tương đối dùng để lưu trong JSON cho object_path của Batch 2.
    """

    existing_json_path = Path(existing_json_path)
    map_keyframes_dir_b2 = Path(map_keyframes_dir_b2)
    objects_root_b2 = Path(objects_root_b2)
    metadata_root = Path(metadata_root)
    output_json_path = Path(output_json_path)

    # ----------------------------------------------------------
    # 1. Load database cũ (Batch 1) và lấy vector_index tiếp theo
    # ----------------------------------------------------------
    mapping_database = []
    existing_video_ids = set()

    if existing_json_path.exists():
        print(f"[INFO] Loading existing mapping from: {existing_json_path}")
        with open(existing_json_path, "r", encoding="utf-8") as f:
            mapping_database = json.load(f)

        existing_video_ids = {item["video_id"] for item in mapping_database}
        print(
            f"[INFO] Found {len(mapping_database)} existing records from {len(existing_video_ids)} videos."
        )
    else:
        print(
            f"[WARNING] Existing mapping file not found at {existing_json_path}. Starting fresh!"
        )

    # vector_index sẽ bắt đầu tiếp nối từ độ dài của batch 1 (ví dụ: N, N+1, N+2,...)
    vector_index = len(mapping_database)

    # ----------------------------------------------------------
    # 2. Quét trực tiếp danh sách CSV keyframes của Batch 2
    # ----------------------------------------------------------
    csv_files = sorted(list(map_keyframes_dir_b2.rglob("*.csv")))
    print(f"[INFO] Found {len(csv_files)} video CSV files in Batch 2 keyframes dir.")

    added_videos_count = 0

    for csv_path in tqdm(csv_files, desc="Building Batch 2 Mapping"):
        video_id = csv_path.stem

        # Bỏ qua nếu video này đã tồn tại trong mapping cũ (tránh trùng lặp)
        if video_id in existing_video_ids:
            continue

        batch_name = video_id.split("_")[0]
        df = pd.read_csv(csv_path)

        # --------------------------------------------
        # Load Metadata JSON (nếu có)
        # --------------------------------------------
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
                    "publish_date": meta.get("publish_date", ""),
                }
            except Exception as e:
                print(f"[WARNING] Failed to load metadata for {video_id}: {e}")

        # --------------------------------------------
        # Duyệt qua từng Keyframe trong CSV
        # --------------------------------------------
        for _, row in df.iterrows():
            n = int(row["n"])
            keyframe_index = n
            frame_idx = int(row["frame_idx"])
            pts_time = float(row["pts_time"])
            fps = int(row["fps"])

            # Path Keyframe
            image_name = f"{keyframe_index:06d}.jpg"
            keyframe_path = (
                f"keyframes/"
                f"Keyframes_{batch_name}/"
                f"keyframes/"
                f"{video_id}/"
                f"{image_name}"
            )

            # Path Object (Relative & Absolute)
            object_filename = f"{n:03d}.json"
            object_path = f"{objects_rel_prefix}/{video_id}/{object_filename}"

            abs_object = objects_root_b2 / video_id / object_filename

            object_entities = []
            if abs_object.exists():
                try:
                    with open(abs_object, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    object_entities = list(
                        dict.fromkeys(
                            obj.get("detection_class_entities", [])
                        )
                    )
                except Exception:
                    pass

            # Path Metadata
            metadata_path = f"data/metadata/media-info/{video_id}.json"

            # Record mới
            mapping_database.append(
                {
                    "vector_index": vector_index,
                    "video_id": video_id,
                    "keyframe_index": keyframe_index,
                    "frame_idx": frame_idx,
                    "pts_time": pts_time,
                    "fps": fps,
                    "keyframe_path": keyframe_path,
                    "object_path": object_path,
                    "metadata_path": metadata_path,
                    "object_entities": object_entities,
                    "metadata": metadata,
                }
            )

            vector_index += 1

        added_videos_count += 1

    # ----------------------------------------------------------
    # 3. Xuất file keyframes_mapping_new.json
    # ----------------------------------------------------------
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_database, f, ensure_ascii=False, indent=4)

    print()
    print("=" * 70)
    print("[SUCCESS] Keyframes Mapping Appended Successfully")
    print("=" * 70)
    print(f"New Videos Added : {added_videos_count}")
    print(f"Total Records    : {len(mapping_database)}")
    print(f"Output File      : {output_json_path}")


# ==========================================================
# Run Execution
# ==========================================================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    append_batch2_mapping(
        # File JSON Batch 1 gốc
        existing_json_path=BASE_DIR
        / "data"
        / "indexes"
        / "keyframes_mapping.json",
        # Thư mục chứa các file .csv keyframe của Batch 2
        map_keyframes_dir_b2=BASE_DIR
        / "data"
        / "mapping"
        / "map-keyframes-aic25-b2"
        / "map-keyframes",  # <-- Đổi tên folder b2 tương ứng nếu khác
        # Thư mục chứa các file object json của Batch 2
        objects_root_b2=BASE_DIR
        / "data"
        / "object"
        / "objects-aic25-b2"
        / "objects",  # <-- Đổi tên folder b2 tương ứng nếu khác
        # Metadata chung (nếu gộp chung 1 folder media-info)
        metadata_root=BASE_DIR / "data" / "metadata" / "media-info",
        # Output file mapping hợp nhất mới
        output_json_path=BASE_DIR
        / "data"
        / "indexes"
        / "keyframes_mapping_new.json",
        # Đường dẫn tương đối lưu vào chuỗi JSON (tùy chỉnh nếu folder object b2 đặt tên khác)
        objects_rel_prefix="data/object/objects-aic25-b2/objects",
    )