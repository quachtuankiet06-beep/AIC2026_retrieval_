
# import os
# import glob
# import json
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# def build_mapping_json(
#     clip_features_dir="data/CLIP_ft/clip-features-32-aic25-b1",
#     objects_root="data/object/objects-aic25-b1/objects",
#     metadata_root="data/metadata",
#     map_keyframes_dir="data/mapping/map-keyframes-aic25-b1/map-keyframes",
#     output_json_path="data/indexes/keyframes_mapping.json"
# ):
#     print("=" * 50)
#     print("Bắt đầu xây dựng file Mapping JSON chuẩn từ file CSV...")
#     print("=" * 50)
    
#     BASE_DIR = Path(__file__).resolve().parent.parent.parent 
    
#     clip_dir_abs = (BASE_DIR / clip_features_dir) if not Path(clip_features_dir).is_absolute() else Path(clip_features_dir)
#     obj_root_abs = (BASE_DIR / objects_root) if not Path(objects_root).is_absolute() else Path(objects_root)
#     meta_root_abs = (BASE_DIR / metadata_root) if not Path(metadata_root).is_absolute() else Path(metadata_root)
#     map_dir_abs = (BASE_DIR / map_keyframes_dir) if not Path(map_keyframes_dir).is_absolute() else Path(map_keyframes_dir)

#     search_path = os.path.join(clip_dir_abs, "**", "*.npy")
#     npy_files = glob.glob(search_path, recursive=True)
    
#     if not npy_files:
#         print(f"[Error] Không tìm thấy file .npy nào trong: {clip_dir_abs}")
#         return

#     mapping_data = []
#     global_vector_index = 0

#     for file_path in tqdm(sorted(npy_files), desc="Processing Videos", unit="file"):
#         file_path_obj = Path(file_path)
#         video_id = file_path_obj.stem  
        
#         try:
#             # Đọc file CSV mapping tương ứng với video_id (Ví dụ: L21_V001.csv)
#             csv_file_path = map_dir_abs / f"{video_id}.csv"
#             if not csv_file_path.exists():
#                 print(f"\n[Warning] Không tìm thấy file CSV mapping cho {video_id}: {csv_file_path}")
#                 continue

#             try:
#                 csv_mapping_df = pd.read_csv(csv_file_path)
#             except Exception as e:
#                 print(f"\n[Error] Không thể đọc file CSV {csv_file_path}: {e}")
#                 continue

#             # Xử lý metadata của video
#             metadata_file = meta_root_abs / f"{video_id}.json"
#             metadata_content = {}
#             if metadata_file.exists():
#                 try:
#                     with open(metadata_file, "r", encoding="utf-8") as mf:
#                         metadata_content = json.load(mf)
#                 except Exception:
#                     pass

#             parsed_metadata = {
#                 "title": metadata_content.get("title", ""),
#                 "description": metadata_content.get("description", ""),
#                 "keywords": metadata_content.get("keywords", []),
#                 "publish_date": metadata_content.get("publish_date", "")
#             }

#             # Vòng lặp duyệt trực tiếp qua từng dòng trong file CSV mapping
#             # Vòng lặp duyệt trực tiếp qua từng dòng trong file CSV mapping
#             for idx, row in csv_mapping_df.iterrows():
#                 # Lấy chính xác giá trị cột 'n' từ file CSV làm chuẩn
#                 n_val = int(row["n"]) if "n" in csv_mapping_df.columns and not pd.isna(row["n"]) else (idx + 1)
                
#                 # keyframe_index dùng cho tên file ảnh (.jpg) bắt đầu từ 0 nên cần n_val - 1
#                 keyframe_index = n_val - 1 

#                 pts_time = float(row["pts_time"]) if "pts_time" in csv_mapping_df.columns and not pd.isna(row["pts_time"]) else 0.0
#                 fps = int(row["fps"]) if "fps" in csv_mapping_df.columns and not pd.isna(row["fps"]) else 30
#                 frame_idx = int(row["frame_idx"]) if "frame_idx" in csv_mapping_df.columns and not pd.isna(row["frame_idx"]) else (keyframe_index * fps)

#                 # Đường dẫn file ảnh dùng 0-index (000000.jpg, 000001.jpg, ...)
#                 kf_filename_stem = f"{keyframe_index:06d}"
#                 batch_name = video_id.split('_')[0] if '_' in video_id else "L21"
#                 keyframe_path = f"keyframes/{batch_name}/{video_id}/{kf_filename_stem}.jpg"
                
#                 # Tên file object dùng giá trị chuẩn n từ CSV (001.json, 002.json, ...)
#                 obj_file_name = f"{n_val:03d}.json"
#                 object_path = f"data/object/objects-aic25-b1/objects/{video_id}/{obj_file_name}"
                
#                 object_entities = []
#                 abs_obj_file = obj_root_abs / video_id / obj_file_name
                
#                 if abs_obj_file.exists():
#                     try:
#                         with open(abs_obj_file, "r", encoding="utf-8") as of:
#                             obj_data = json.load(of)
#                             if "detection_class_entities" in obj_data:
#                                 raw_entities = obj_data["detection_class_entities"]
#                                 object_entities = list(dict.fromkeys(raw_entities))
#                     except Exception:
#                         pass

#                 item = {
#                     "vector_index": global_vector_index,
#                     "video_id": video_id,
#                     "n": n_val,                     # Giữ nguyên giá trị cột n từ file CSV để kiểm tra
#                     "keyframe_index": keyframe_index, # Index dùng để load đúng file ảnh .jpg (0-based)
#                     "frame_idx": frame_idx,         # Lấy chính xác từ cột frame_idx trong CSV
#                     "pts_time": pts_time,           # Lấy chính xác từ cột pts_time trong CSV
#                     "fps": fps,                     # Lấy chính xác từ cột fps trong CSV
#                     "keyframe_path": keyframe_path,
#                     "object_path": object_path,
#                     "metadata_path": f"data/metadata/media-info/{video_id}.json",
#                     "object_entities": object_entities,
#                     "metadata": parsed_metadata
#                 }
                
#                 mapping_data.append(item)
#                 global_vector_index += 1
                
#         except Exception as e:
#             print(f"\n[Error] Lỗi khi xử lý video {video_id}: {e}")
#             pass

#     output_abs_path = (BASE_DIR / output_json_path) if not Path(output_json_path).is_absolute() else Path(output_json_path)
#     os.makedirs(output_abs_path.parent, exist_ok=True)
    
#     with open(output_abs_path, "w", encoding="utf-8") as f:
#         json.dump(mapping_data, f, ensure_ascii=False, indent=4)

#     print("\n" + "=" * 50)
#     print(f"[Success] Đã tạo thành công file mapping JSON chuẩn từ file CSV!")
#     print(f" - Tổng số vector/keyframes được map: {len(mapping_data)}")
#     print(f" - Lưu tại: {output_abs_path}")
#     print("=" * 50)

# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).resolve().parent.parent.parent 
    
#     CLIP_DIR = BASE_DIR / "data" / "CLIP_ft" / "clip-features-32-aic25-b1" / "clip-features-32"
#     OBJECT_DIR = BASE_DIR / "data" / "object" / "objects-aic25-b1" / "objects"
#     METADATA_DIR = BASE_DIR / "data" / "metadata" / "media-info"
#     MAP_DIR = BASE_DIR / "data" / "mapping" / "map-keyframes-aic25-b1" / "map-keyframes"
#     OUTPUT_JSON = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"
    
#     build_mapping_json(
#         clip_features_dir=str(CLIP_DIR),
#         objects_root=str(OBJECT_DIR),
#         metadata_root=str(METADATA_DIR),
#         map_keyframes_dir=str(MAP_DIR),
#         output_json_path=str(OUTPUT_JSON)
#     )


import json
import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ==========================================================
# Build Mapping Database
# ==========================================================

def build_mapping_database(
    clip_features_dir,
    map_keyframes_dir,
    objects_root,
    metadata_root,
    output_json_path,
):
    """
    Build keyframes_mapping.json

    Mapping này dùng cho toàn bộ Retrieval Pipeline.

    vector_index
            ↓
    Mapping Database
            ↓
    video_id
    frame_idx
    keyframe_path
    object_entities
    metadata
    """

    clip_features_dir = Path(clip_features_dir)
    map_keyframes_dir = Path(map_keyframes_dir)
    objects_root = Path(objects_root)
    metadata_root = Path(metadata_root)
    output_json_path = Path(output_json_path)

    npy_files = sorted(
        glob.glob(
            str(clip_features_dir / "**" / "*.npy"),
            recursive=True,
        )
    )

    print(f"[INFO] Found {len(npy_files)} videos.")

    mapping_database = []

    vector_index = 0

    for npy_file in tqdm(npy_files):

        npy_file = Path(npy_file)

        video_id = npy_file.stem

        # ----------------------------
        # Batch Name
        # ----------------------------

        batch_name = video_id.split("_")[0]

        csv_path = (
            map_keyframes_dir
            / f"{video_id}.csv"
        )

        if not csv_path.exists():
            print(f"[WARNING] Missing CSV : {video_id}")
            continue

        df = pd.read_csv(csv_path)

        # ----------------------------
        # Metadata
        # ----------------------------

        metadata_json = (
            metadata_root
            / f"{video_id}.json"
        )

        metadata = {}

        if metadata_json.exists():

            with open(metadata_json, "r", encoding="utf-8") as f:
                meta = json.load(f)

            metadata = {

                "title":
                    meta.get("title", ""),

                "description":
                    meta.get("description", ""),

                "keywords":
                    meta.get("keywords", []),

                "publish_date":
                    meta.get("publish_date", "")
            }

        # ----------------------------
        # Every Keyframe
        # ----------------------------

        for _, row in df.iterrows():

            n = int(row["n"])

            keyframe_index = n 

            frame_idx = int(row["frame_idx"])

            pts_time = float(row["pts_time"])

            fps = int(row["fps"])

            # ----------------------------------
            # keyframe path
            # ----------------------------------

            image_name = f"{keyframe_index:06d}.jpg"

            keyframe_path = (
                f"keyframes/"
                f"Keyframes_{batch_name}/"
                f"keyframes/"
                f"{video_id}/"
                f"{image_name}"
            )

            # ----------------------------------
            # object path
            # ----------------------------------

            object_filename = f"{n:03d}.json"

            object_path = (
                f"data/object/"
                f"objects-aic25-b1/"
                f"objects/"
                f"{video_id}/"
                f"{object_filename}"
            )

            abs_object = (
                objects_root
                / video_id
                / object_filename
            )

            object_entities = []

            if abs_object.exists():

                try:

                    with open(abs_object, "r", encoding="utf-8") as f:

                        obj = json.load(f)

                    object_entities = list(
                        dict.fromkeys(
                            obj.get(
                                "detection_class_entities",
                                []
                            )
                        )
                    )

                except Exception:
                    pass

            # ----------------------------------
            # metadata path
            # ----------------------------------

            metadata_path = (
                f"data/metadata/"
                f"media-info/"
                f"{video_id}.json"
            )

            # ----------------------------------
            # mapping record
            # ----------------------------------

            mapping_database.append({

                "vector_index":
                    vector_index,

                "video_id":
                    video_id,

                "keyframe_index":
                    keyframe_index,

                "frame_idx":
                    frame_idx,

                "pts_time":
                    pts_time,

                "fps":
                    fps,

                "keyframe_path":
                    keyframe_path,

                "object_path":
                    object_path,

                "metadata_path":
                    metadata_path,

                "object_entities":
                    object_entities,

                "metadata":
                    metadata,
            })

            vector_index += 1

    # =======================================================
    # Save
    # =======================================================

    output_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            mapping_database,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print()

    print("=" * 70)
    print("[SUCCESS] Mapping Database Created")
    print("=" * 70)

    print(f"Total Records : {len(mapping_database)}")
    print(f"Output        : {output_json_path}")


# ==========================================================
# Debug
# ==========================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    build_mapping_database(

        clip_features_dir=
        BASE_DIR
        / "data"
        / "CLIP_ft"
        / "clip-features-32-aic25-b1"
        / "clip-features-32",

        map_keyframes_dir=
        BASE_DIR
        / "data"
        / "mapping"
        / "map-keyframes-aic25-b1"
        / "map-keyframes",

        objects_root=
        BASE_DIR
        / "data"
        / "object"
        / "objects-aic25-b1"
        / "objects",

        metadata_root=
        BASE_DIR
        / "data"
        / "metadata"
        / "media-info",

        output_json_path=
        BASE_DIR
        / "data"
        / "indexes"
        / "keyframes_mapping.json",
    )