# import os
# import sys
# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional

# # Đảm bảo thư mục gốc dự án có trong sys.path để import các module trong src
# ROOT_DIR = Path(__file__).resolve().parent.parent
# if str(ROOT_DIR) not in sys.path:
#     sys.path.append(str(ROOT_DIR))

# from src.mapping.mapping_pipeline import mapping_pipeline
# from src.reranking.reranking_pipeline import reranking_pipeline
# from src.reranking.temporal_reranking_pipeline import temporal_sequence_reranking_optimized
# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline,
#     dfn5b_vit_h14_retrieval_pipeline
# )
# from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
# from src.retrieval.temporal_retrieval_multi_model import temporal_retrieval_multi_model_pipeline
# from src.retrieval.temporal_nms import temporal_nms

# # Cấu hình đường dẫn mặc định
# BASE_DIR = ROOT_DIR
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_new_kf.db"
# RETRIEVAL_CONFIG = BASE_DIR / "configs" / "retrieval.yaml"
# RERANK_CONFIG = BASE_DIR / "configs" / "reranking.yaml"
# VIDEO_FPS_MAPPING_PATH = BASE_DIR / "data" / "mapping" / "video_fps_mapping.json"
# # Đọc từ biến môi trường - mặc định là Windows path khi chạy local
# # Trong Docker container, set VIDEO_DIR=/videos qua docker-compose.yml hoặc -e flag
# VIDEO_DIR = os.environ.get("VIDEO_DIR", r"C:\Users\Public\Documents")

# # Cache FPS mapping
# _video_fps_cache: Dict[str, float] = {}

# def load_video_fps_mapping() -> Dict[str, float]:
#     """Tải và cache thông tin mapping video_id và fps từ file video_fps_mapping.json"""
#     global _video_fps_cache
#     if _video_fps_cache:
#         return _video_fps_cache

#     if VIDEO_FPS_MAPPING_PATH.exists():
#         try:
#             with open(VIDEO_FPS_MAPPING_PATH, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 _video_fps_cache = {item["video_id"]: float(item.get("fps", 25.0)) for item in data}
#                 print(f"[INFO] Loaded {len(_video_fps_cache)} video FPS mappings.")
#         except Exception as e:
#             print(f"[WARNING] Không thể load video_fps_mapping.json: {e}")
#     return _video_fps_cache

# def get_video_fps(video_id: str) -> float:
#     """Lấy FPS của một video cụ thể, mặc định là 25.0 nếu không có trong dữ liệu"""
#     fps_dict = load_video_fps_mapping()
#     return fps_dict.get(str(video_id), 25.0)

# def format_pts_to_hms(pts_time: float) -> str:
#     """Chuyển đổi pts_time (giây) sang định dạng Giờ:Phút:Giây (HH:MM:SS hoặc MM:SS)"""
#     try:
#         total_seconds = float(pts_time)
#         hours = int(total_seconds // 3600)
#         minutes = int((total_seconds % 3600) // 60)
#         seconds = int(total_seconds % 60)
#         if hours > 0:
#             return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
#         else:
#             return f"{minutes:02d}:{seconds:02d}"
#     except Exception:
#         return "00:00"

# def get_submission_frame_info(video_id: str, pts_time: float) -> int:
#     """Tính toán frame_idx từ pts_time và fps của video"""
#     fps = get_video_fps(video_id)
#     return int(round(float(pts_time) * fps))

# def get_video_path(video_id: str) -> Optional[Path]:
#     """Tìm file video trong thư mục Public/Documents hỗ trợ nhiều định dạng"""
#     extensions = [".mp4", ".avi", ".mkv", ".mov", ".MP4"]
#     for ext in extensions:
#         video_path = Path(VIDEO_DIR) / f"{video_id}{ext}"
#         if video_path.exists():
#             return video_path
#     return None

# def get_keyframe_path(video_id: str, img_filename: str) -> Optional[Path]:
#     """Tìm đường dẫn ảnh keyframe trong data/Custom_Keyframes/ hoặc data/keyframes/"""
#     custom_dir = BASE_DIR / "data" / "Custom_Keyframes" / video_id
    
#     # 1. Thử trực tiếp tên file ảnh trong Custom_Keyframes
#     direct_path = custom_dir / img_filename
#     if direct_path.exists():
#         return direct_path

#     # 2. Thử format số frame thành 6 chữ số (VD: 000000.jpg, 000057.jpg)
#     pure_name = Path(img_filename).stem
#     if pure_name.isdigit():
#         idx_val = int(pure_name)
#         six_digit_path = custom_dir / f"{idx_val:06d}.jpg"
#         if six_digit_path.exists():
#             return six_digit_path
        
#         raw_idx_path = custom_dir / f"{idx_val}.jpg"
#         if raw_idx_path.exists():
#             return raw_idx_path

#     # 3. Fallback thư mục keyframes cũ nếu có
#     batch_prefix = video_id.split('_')[0] if '_' in video_id else "Keyframes_L21"
#     folder_batch = f"Keyframes_{batch_prefix}"
#     old_keyframe_path = BASE_DIR / "data" / "keyframes" / folder_batch / "keyframes" / video_id / img_filename
#     if old_keyframe_path.exists():
#         return old_keyframe_path

#     # 4. Fallback glob tìm kiếm
#     found_files = list(BASE_DIR.glob(f"**/Custom_Keyframes/{video_id}/{img_filename}"))
#     if not found_files:
#         found_files = list(BASE_DIR.glob(f"**/keyframes/{video_id}/{img_filename}"))
#     if found_files:
#         return Path(found_files[0])

#     return None

# def convert_time_to_frame(input_str: str) -> Dict[str, Any]:
#     """
#     Công cụ chuyển đổi nhanh chuỗi Video/Time -> Frame Index
#     Ví dụ input_str: "L22_V001 / 2:16" hoặc "L22_V001 / 136.5"
#     """
#     if "/" not in input_str:
#         raise ValueError("Vui lòng nhập đúng định dạng chứa dấu '/' (VD: L22_V001 / 2:16)")

#     parts = input_str.split("/")
#     target_vid = parts[0].strip()
#     time_str = parts[1].strip()

#     if ":" in time_str:
#         time_parts = time_str.split(":")
#         if len(time_parts) == 2:
#             minutes = float(time_parts[0])
#             seconds = float(time_parts[1])
#             total_seconds = minutes * 60 + seconds
#         elif len(time_parts) == 3:
#             hours = float(time_parts[0])
#             minutes = float(time_parts[1])
#             seconds = float(time_parts[2])
#             total_seconds = hours * 3600 + minutes * 60 + seconds
#         else:
#             total_seconds = 0.0
#     else:
#         total_seconds = float(time_str)

#     fps = get_video_fps(target_vid)
#     calc_frame = int(round(total_seconds * fps))

#     return {
#         "video_id": target_vid,
#         "total_seconds": total_seconds,
#         "formatted_time": format_pts_to_hms(total_seconds),
#         "fps": fps,
#         "frame_idx": calc_frame,
#         "submission_string": f"{target_vid} / {calc_frame}"
#     }

# def execute_search(
#     query: str,
#     search_type: str = "standard",  # "standard" hoặc "temporal"
#     retrieval_mode: str = "multi",   # "single" hoặc "multi"
#     single_model_choice: str = "siglip2",
#     use_siglip2: bool = True,
#     use_dfn5b: bool = True,
#     top_k: int = 100,
#     max_kf_gap: int = 150,
#     min_kf_gap: int = 0,
#     beam_width: int = 5
# ) -> Dict[str, Any]:
#     """
#     Thực thi toàn bộ luồng tìm kiếm (Standard hoặc Temporal Sequence),
#     Mapping và Reranking y hệt như logic trong app.py.
#     """
#     query = query.strip()
#     if not query:
#         raise ValueError("Vui lòng nhập nội dung truy vấn!")

#     mapping_p = str(MAPPING_PATH)
#     ret_cfg = str(RETRIEVAL_CONFIG)
#     rerank_cfg = str(RERANK_CONFIG)

#     # -------------------------------------------------------------
#     # LUỒNG 1: TEMPORAL SEQUENCE RETRIEVAL
#     # -------------------------------------------------------------
#     if search_type == "temporal":
#         if retrieval_mode == "single":
#             if single_model_choice == "siglip2":
#                 idx_path = str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_new.index")
#             else:
#                 idx_path = str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_new.index")
#             current_temporal_models = [{"name": single_model_choice, "index_path": idx_path}]
#         else:
#             current_temporal_models = []
#             if use_siglip2:
#                 current_temporal_models.append({
#                     "name": "siglip2",
#                     "index_path": str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_new.index")
#                 })
#             if use_dfn5b:
#                 current_temporal_models.append({
#                     "name": "dfn5b_vit_h14",
#                     "index_path": str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_new.index")
#                 })
#             if not current_temporal_models:
#                 raise ValueError("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")

#         retrieval_results = temporal_retrieval_multi_model_pipeline(
#             raw_query_string=query,
#             model_configs=current_temporal_models,
#             config_path=ret_cfg,
#             max_kf_gap=max_kf_gap,
#             min_kf_gap=min_kf_gap,
#             beam_width=beam_width
#         )

#         if not retrieval_results:
#             return {"results": [], "total": 0, "search_type": "temporal"}

#         # Mapping từng step trong chuỗi
#         for seq_item in retrieval_results:
#             seq_path = seq_item.get("sequence_path", [])
#             mapped_seq_path = []
#             for step_cand in seq_path:
#                 mapped_step_list = mapping_pipeline(
#                     retrieval_results=[step_cand],
#                     mapping_db_path=mapping_p,
#                 )
#                 if mapped_step_list:
#                     mapped_seq_path.append(mapped_step_list[0])
#                 else:
#                     mapped_seq_path.append(step_cand)
#             seq_item["sequence_path"] = mapped_seq_path

#         # Reranking cho Temporal
#         final_results = temporal_sequence_reranking_optimized(
#             raw_query_string=query,
#             sequence_candidates=retrieval_results,
#             config_path=rerank_cfg
#         )

#         final_results = sorted(final_results, key=lambda x: x.get("final_score", 0), reverse=True)
#         for r_idx, c in enumerate(final_results, start=1):
#             c["rank"] = r_idx

#     # -------------------------------------------------------------
#     # LUỒNG 2: STANDARD RETRIEVAL
#     # -------------------------------------------------------------
#     else:
#         if retrieval_mode == "single":
#             if single_model_choice == "siglip2":
#                 idx_path = BASE_DIR / "data" / "indexes" / "siglip2_keyframes_new.index"
#                 retrieval_results = siglip2_retrieval_pipeline(query, str(idx_path), ret_cfg)
#             else:
#                 idx_path = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_new.index"
#                 retrieval_results = dfn5b_vit_h14_retrieval_pipeline(query, str(idx_path), ret_cfg)
#         else:
#             selected_models = []
#             if use_siglip2:
#                 selected_models.append({
#                     "name": "siglip2",
#                     "index_path": str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_new.index")
#                 })
#             if use_dfn5b:
#                 selected_models.append({
#                     "name": "dfn5b_vit_h14",
#                     "index_path": str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_new.index")
#                 })
#             if not selected_models:
#                 raise ValueError("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")

#             retrieval_results = retrieval_multi_model_pipeline(
#                 query_text=query,
#                 model_configs=selected_models,
#                 config_path=ret_cfg
#             )

#         if not retrieval_results:
#             return {"results": [], "total": 0, "search_type": "standard"}

#         candidate_list = mapping_pipeline(
#             retrieval_results=retrieval_results,
#             mapping_db_path=mapping_p,
#         )

#         # Lọc bớt các keyframe quá gần nhau trong cùng 1 video để giảm nhiễu trước khi rerank
#         candidate_list = temporal_nms(
#             candidate_list,
#             min_gap_sec=0.3,     # Tăng khoảng cách lên 1.5 giây để chống nhiễu tốt hơn trong DB 300k keyframes
#             max_per_video=5       # Giới hạn tối đa 5 keyframes mỗi video để tăng tính đa dạng của kết quả
#         )

#         final_results = reranking_pipeline(
#             query_text=query,
#             candidate_list=candidate_list,
#             config_path=rerank_cfg,
#         )

#     # -------------------------------------------------------------
#     # LÀM ĐẦY THÔNG TIN KẾT QUẢ ĐỂ TRẢ VỀ CHO FRONTEND
#     # -------------------------------------------------------------
#     display_results = final_results[:top_k]
#     formatted_results = []

#     for res in display_results:
#         video_id = res.get("video_id", "")
#         fps = get_video_fps(video_id)
#         has_video = get_video_path(video_id) is not None

#         if search_type == "temporal" and "sequence_path" in res:
#             seq_path = res["sequence_path"]
#             formatted_seq_path = []
#             for step_i, step_cand in enumerate(seq_path):
#                 s_vid = step_cand.get("video_id", video_id)
#                 s_fps = get_video_fps(s_vid)
#                 s_frame_val = int(step_cand.get("keyframe_index", step_cand.get("frame_idx", 1)))
                
#                 # Ưu tiên lấy tên file từ keyframe_path
#                 kp = step_cand.get("keyframe_path", "")
#                 if kp:
#                     s_img_filename = Path(kp).name
#                 else:
#                     s_img_filename = f"{int(step_cand.get('frame_idx', s_frame_val)):06d}.jpg"

#                 s_ptime = float(step_cand.get("pts_time", 0.0))
#                 s_calc_frame = int(round(s_ptime * s_fps))

#                 formatted_seq_path.append({
#                     "step_idx": step_i + 1,
#                     "video_id": s_vid,
#                     "keyframe_index": s_frame_val,
#                     "img_filename": s_img_filename,
#                     "keyframe_url": f"/api/keyframe/{s_vid}/{s_img_filename}",
#                     "score": float(step_cand.get("score", 0.0)),
#                     "pts_time": s_ptime,
#                     "formatted_time": format_pts_to_hms(s_ptime),
#                     "fps": s_fps,
#                     "calc_frame": s_calc_frame,
#                     "submission_string": f"{s_vid} / {s_calc_frame}",
#                     "has_video": get_video_path(s_vid) is not None,
#                     "video_url": f"/api/video/{s_vid}" if get_video_path(s_vid) else None
#                 })

#             formatted_results.append({
#                 "rank": res.get("rank", 1),
#                 "video_id": video_id,
#                 "is_temporal": True,
#                 "num_steps": len(seq_path),
#                 "final_score": float(res.get("final_score", 0.0)),
#                 "sequence_path": formatted_seq_path,
#                 "fps": fps,
#                 "has_video": has_video,
#                 "video_url": f"/api/video/{video_id}" if has_video else None
#             })
#         else:
#             frame_val = int(res.get("keyframe_index", res.get("frame_idx", 1)))
            
#             # Ưu tiên lấy tên file từ keyframe_path (Custom_Keyframes lưu 000000.jpg, 000057.jpg)
#             kp = res.get("keyframe_path", "")
#             if kp:
#                 img_filename = Path(kp).name
#             else:
#                 img_filename = f"{int(res.get('frame_idx', frame_val)):06d}.jpg"

#             p_time = float(res.get("pts_time", 0.0))
#             calc_frame = int(round(p_time * fps))

#             formatted_results.append({
#                 "rank": res.get("rank", 1),
#                 "video_id": video_id,
#                 "is_temporal": False,
#                 "keyframe_index": frame_val,
#                 "img_filename": img_filename,
#                 "keyframe_url": f"/api/keyframe/{video_id}/{img_filename}",
#                 "pts_time": p_time,
#                 "formatted_time": format_pts_to_hms(p_time),
#                 "fps": fps,
#                 "calc_frame": calc_frame,
#                 "submission_string": f"{video_id} / {calc_frame}",
#                 "sources": res.get("sources", []),
#                 "retrieval_score": float(res.get("score", res.get("retrieval_score", 0.0))),
#                 "object_score": float(res.get("object_score", 0.0)),
#                 "metadata_score": float(res.get("metadata_score", 0.0)),
#                 "ocr_score": float(res.get("ocr_score", 0.0)),
#                 "asr_score": float(res.get("asr_score", 0.0)),
#                 "final_score": float(res.get("final_score", 0.0)),
#                 "metadata": res.get("metadata", {}),
#                 "object_entities": res.get("object_entities", []),
#                 "has_video": has_video,
#                 "video_url": f"/api/video/{video_id}" if has_video else None
#             })

#     return {
#         "results": formatted_results,
#         "total": len(final_results),
#         "displayed": len(formatted_results),
#         "search_type": search_type
#     }
import os
import sys
import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional

# Đảm bảo thư mục gốc dự án có trong sys.path để import các module trong src
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.mapping.mapping_pipeline import mapping_pipeline
from src.reranking.reranking_pipeline import reranking_pipeline
from src.reranking.temporal_reranking_pipeline import temporal_sequence_reranking_optimized
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline,
    dfn5b_vit_h14_retrieval_pipeline,
    clear_translation_cache,
)
from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
from src.retrieval.temporal_retrieval_multi_model import temporal_retrieval_multi_model_pipeline
from src.retrieval.temporal_nms import temporal_nms

# Cấu hình đường dẫn mặc định
BASE_DIR = ROOT_DIR
MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_b1_b2.db"
RETRIEVAL_CONFIG = BASE_DIR / "configs" / "retrieval.yaml"
RERANK_CONFIG = BASE_DIR / "configs" / "reranking.yaml"
VIDEO_FPS_MAPPING_PATH = BASE_DIR / "data" / "mapping" / "video_fps_mapping.json"

# Đọc từ biến môi trường - mặc định là Windows path khi chạy local
# Trong Docker container, set VIDEO_DIR=/videos qua docker-compose.yml hoặc -e flag
VIDEO_DIR = os.environ.get("VIDEO_DIR", r"C:\Users\Public\Documents")

# Cache FPS mapping
_video_fps_cache: Dict[str, float] = {}


def load_video_fps_mapping() -> Dict[str, float]:
    """Tải và cache thông tin mapping video_id và fps từ file video_fps_mapping.json."""
    global _video_fps_cache
    if _video_fps_cache:
        return _video_fps_cache

    if VIDEO_FPS_MAPPING_PATH.exists():
        try:
            with open(VIDEO_FPS_MAPPING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _video_fps_cache = {
                    str(item["video_id"]): float(item.get("fps", 25.0))
                    for item in data
                    if "video_id" in item
                }
                print(f"[INFO] Loaded {len(_video_fps_cache)} video FPS mappings.")
        except Exception as e:
            print(f"[WARNING] Không thể load video_fps_mapping.json: {e}")

    return _video_fps_cache


def get_video_fps(video_id: str) -> float:
    """Lấy FPS của một video cụ thể, mặc định là 25.0 nếu không có trong dữ liệu."""
    fps_dict = load_video_fps_mapping()
    return fps_dict.get(str(video_id), 25.0)


def format_pts_to_hms(pts_time: float) -> str:
    """Chuyển đổi pts_time (giây) sang định dạng Giờ:Phút:Giây (HH:MM:SS hoặc MM:SS)."""
    try:
        total_seconds = float(pts_time)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    except Exception:
        return "00:00"


def get_submission_frame_info(video_id: str, pts_time: float) -> int:
    """Tính toán frame_idx từ pts_time và fps của video."""
    fps = get_video_fps(video_id)
    return int(round(float(pts_time) * fps))


@lru_cache(maxsize=8192)
def get_video_path(video_id: str) -> Optional[Path]:
    """Tìm file video trong thư mục Public/Documents hỗ trợ nhiều định dạng."""
    extensions = [".mp4", ".avi", ".mkv", ".mov", ".MP4"]
    for ext in extensions:
        video_path = Path(VIDEO_DIR) / f"{video_id}{ext}"
        if video_path.exists():
            return video_path
    return None


_VIDEO_KEYFRAME_DIR_CACHE: Optional[Dict[str, Path]] = None


def get_video_keyframe_dir_registry() -> Dict[str, Path]:
    """Lập chỉ mục nhanh thư mục chứa keyframe của từng video_id (chạy 1 lần, O(1) tra cứu)."""
    global _VIDEO_KEYFRAME_DIR_CACHE
    if _VIDEO_KEYFRAME_DIR_CACHE is not None:
        return _VIDEO_KEYFRAME_DIR_CACHE

    registry: Dict[str, Path] = {}
    data_dir = BASE_DIR / "data"

    # 1. Custom_Keyframes (Batch 1 & một phần Batch 2)
    ck_dir = data_dir / "Custom_Keyframes"
    if ck_dir.exists():
        for p in ck_dir.iterdir():
            if p.is_dir():
                registry[p.name.lower()] = p

    # 2. keyframes_batch2_n
    k2_n = data_dir / "keyframes_batch2_n"
    if k2_n.exists():
        for p in k2_n.iterdir():
            if p.is_dir():
                registry[p.name.lower()] = p
                registry[p.name.replace("-", "_").lower()] = p

    # 3. keyframes_batch2_m và keyframes_batch2_s
    for b in ["keyframes_batch2_m", "keyframes_batch2_s"]:
        bp = data_dir / b
        if bp.exists():
            for kf_folder in bp.glob("**/keyframes"):
                if kf_folder.is_dir():
                    for p in kf_folder.iterdir():
                        if p.is_dir():
                            registry[p.name.lower()] = p
                            registry[p.name.replace("-", "_").lower()] = p

    _VIDEO_KEYFRAME_DIR_CACHE = registry
    print(f"[INFO] Đã lập chỉ mục {len(registry)} thư mục keyframe video phục vụ tức thì!")
    return _VIDEO_KEYFRAME_DIR_CACHE


@lru_cache(maxsize=16384)
def get_keyframe_path(video_id: str, img_filename: str) -> Optional[Path]:
    """Tìm đường dẫn ảnh keyframe siêu tốc (O(1)) không dùng glob đệ quy chậm chạp."""
    registry = get_video_keyframe_dir_registry()

    vid_clean = video_id.strip().lower()
    folder = registry.get(vid_clean)
    if not folder:
        folder = registry.get(vid_clean.replace("-", "_")) or registry.get(vid_clean.replace("_", "-"))

    # Fallback nhanh từ SQLite DB nếu video mới chưa có trong registry
    if not folder:
        try:
            import sqlite3
            db_file = BASE_DIR / "data" / "indexes" / "keyframes_b1_b2.db"
            if db_file.exists():
                with sqlite3.connect(str(db_file), timeout=3) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT keyframe_path FROM keyframes WHERE video_id = ? LIMIT 1", (video_id,))
                    row = cur.fetchone()
                    if row and row[0]:
                        kp_path = BASE_DIR / "data" / row[0]
                        if kp_path.parent.is_dir():
                            folder = kp_path.parent
                            registry[vid_clean] = folder
        except Exception:
            pass

    if not folder or not folder.is_dir():
        return None

    # 1. Thử trực tiếp tên file ảnh trong folder
    direct_path = folder / img_filename
    if direct_path.exists():
        return direct_path

    # 2. Thử format số frame thành 6 chữ số (VD: 000000.jpg, 000057.jpg)
    pure_name = Path(img_filename).stem
    if pure_name.isdigit():
        idx_val = int(pure_name)
        six_digit_path = folder / f"{idx_val:06d}.jpg"
        if six_digit_path.exists():
            return six_digit_path

        raw_idx_path = folder / f"{idx_val}.jpg"
        if raw_idx_path.exists():
            return raw_idx_path

    # 3. Thử các đuôi mở rộng phổ biến
    for ext in [".jpg", ".png", ".webp", ".jpeg"]:
        p = folder / f"{pure_name}{ext}"
        if p.exists():
            return p

    return None


def convert_time_to_frame(input_str: str) -> Dict[str, Any]:
    """
    Công cụ chuyển đổi nhanh chuỗi Video/Time -> Frame Index.
    Ví dụ input_str: "L22_V001 / 2:16" hoặc "L22_V001 / 136.5"
    """
    if "/" not in input_str:
        raise ValueError("Vui lòng nhập đúng định dạng chứa dấu '/' (VD: L22_V001 / 2:16)")

    parts = input_str.split("/")
    target_vid = parts[0].strip()
    time_str = parts[1].strip()

    if ":" in time_str:
        time_parts = time_str.split(":")
        if len(time_parts) == 2:
            minutes = float(time_parts[0])
            seconds = float(time_parts[1])
            total_seconds = minutes * 60 + seconds
        elif len(time_parts) == 3:
            hours = float(time_parts[0])
            minutes = float(time_parts[1])
            seconds = float(time_parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
        else:
            total_seconds = 0.0
    else:
        total_seconds = float(time_str)

    fps = get_video_fps(target_vid)
    calc_frame = int(round(total_seconds * fps))

    return {
        "video_id": target_vid,
        "total_seconds": total_seconds,
        "formatted_time": format_pts_to_hms(total_seconds),
        "fps": fps,
        "frame_idx": calc_frame,
        "submission_string": f"{target_vid} / {calc_frame}",
    }


def execute_search(
    query: str,
    search_type: str = "standard",  # "standard" hoặc "temporal"
    retrieval_mode: str = "multi",   # "single" hoặc "multi"
    single_model_choice: str = "siglip2",
    use_siglip2: bool = True,
    use_dfn5b: bool = True,
    top_k: int = 100,
    max_kf_gap: int = 150,
    min_kf_gap: int = 0,
    beam_width: int = 5,
    topic_filter: str = "all",
) -> Dict[str, Any]:
    """
    Thực thi toàn bộ luồng tìm kiếm (Standard hoặc Temporal Sequence),
    Mapping và Reranking y hệt như logic trong app.py.

    Tối ưu hóa:
    - Cache đường dẫn video / keyframe
    - Giảm gọi I/O lặp lại
    - Đồng bộ temporal NMS với benchmark ở mức min_gap_sec=0.3
    - Bộ lọc chủ đề video (topic_filter) lọc trực tiếp ở tầng FAISS qua IDSelector
    """
    query = query.strip()
    if not query:
        raise ValueError("Vui lòng nhập nội dung truy vấn!")

    # Ép buộc làm mới Translation Cache ở đầu mỗi lượt tìm kiếm để luôn gọi lại Gemini
    clear_translation_cache()

    mapping_p = str(MAPPING_PATH)
    ret_cfg = str(RETRIEVAL_CONFIG)
    rerank_cfg = str(RERANK_CONFIG)

    # -------------------------------------------------------------
    # LUỒNG 1: TEMPORAL SEQUENCE RETRIEVAL
    # -------------------------------------------------------------
    if search_type == "temporal":
        if retrieval_mode == "single":
            if single_model_choice == "siglip2":
                idx_path = str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index")
            else:
                idx_path = str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_b1_b2.index")
            current_temporal_models = [{"name": single_model_choice, "index_path": idx_path}]
        else:
            current_temporal_models = []
            if use_siglip2:
                current_temporal_models.append(
                    {
                        "name": "siglip2",
                        "index_path": str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index"),
                    }
                )
            if use_dfn5b:
                current_temporal_models.append(
                    {
                        "name": "dfn5b_vit_h14",
                        "index_path": str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_b1_b2.index"),
                    }
                )
            if not current_temporal_models:
                raise ValueError("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")

        retrieval_results = temporal_retrieval_multi_model_pipeline(
            raw_query_string=query,
            model_configs=current_temporal_models,
            config_path=ret_cfg,
            max_kf_gap=max_kf_gap,
            min_kf_gap=min_kf_gap,
            beam_width=beam_width,
            topic_filter=topic_filter,
        )

        if not retrieval_results:
            return {"results": [], "total": 0, "search_type": "temporal"}

        # Mapping từng step trong chuỗi
        for seq_item in retrieval_results:
            seq_path = seq_item.get("sequence_path", [])
            mapped_seq_path = []
            for step_cand in seq_path:
                mapped_step_list = mapping_pipeline(
                    retrieval_results=[step_cand],
                    mapping_db_path=mapping_p,
                )
                if mapped_step_list:
                    mapped_seq_path.append(mapped_step_list[0])
                else:
                    mapped_seq_path.append(step_cand)
            seq_item["sequence_path"] = mapped_seq_path

        # Reranking cho Temporal
        final_results = temporal_sequence_reranking_optimized(
            raw_query_string=query,
            sequence_candidates=retrieval_results,
            config_path=rerank_cfg,
        )

        final_results = sorted(final_results, key=lambda x: x.get("final_score", 0.0), reverse=True)
        for r_idx, c in enumerate(final_results, start=1):
            c["rank"] = r_idx

    # -------------------------------------------------------------
    # LUỒNG 2: STANDARD RETRIEVAL
    # -------------------------------------------------------------
    else:
        if retrieval_mode == "single":
            if single_model_choice == "siglip2":
                idx_path = BASE_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index"
                retrieval_results = siglip2_retrieval_pipeline(query, str(idx_path), ret_cfg, topic_filter=topic_filter)
            else:
                idx_path = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_b1_b2.index"
                retrieval_results = dfn5b_vit_h14_retrieval_pipeline(query, str(idx_path), ret_cfg, topic_filter=topic_filter)
        else:
            selected_models = []
            if use_siglip2:
                selected_models.append(
                    {
                        "name": "siglip2",
                        "index_path": str(BASE_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index"),
                    }
                )
            if use_dfn5b:
                selected_models.append(
                    {
                        "name": "dfn5b_vit_h14",
                        "index_path": str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_b1_b2.index"),
                    }
                )
            if not selected_models:
                raise ValueError("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")

            retrieval_results = retrieval_multi_model_pipeline(
                query_text=query,
                model_configs=selected_models,
                config_path=ret_cfg,
                topic_filter=topic_filter,
            )

        if not retrieval_results:
            return {"results": [], "total": 0, "search_type": "standard"}

        candidate_list = mapping_pipeline(
            retrieval_results=retrieval_results,
            mapping_db_path=mapping_p,
        )

        # Đồng bộ với benchmark: nới nhẹ NMS để không làm mất candidate tốt quá sớm
        candidate_list = temporal_nms(
            candidate_list,
            min_gap_sec=0.3,
            max_per_video=5,
        )

        final_results = reranking_pipeline(
            query_text=query,
            candidate_list=candidate_list,
            config_path=rerank_cfg,
        )

    # -------------------------------------------------------------
    # LÀM ĐẦY THÔNG TIN KẾT QUẢ ĐỂ TRẢ VỀ CHO FRONTEND
    # -------------------------------------------------------------
    display_results = final_results[:top_k]
    formatted_results = []

    for res in display_results:
        video_id = res.get("video_id", "")
        fps = get_video_fps(video_id)
        video_path = get_video_path(video_id)
        has_video = video_path is not None

        if search_type == "temporal" and "sequence_path" in res:
            seq_path = res["sequence_path"]
            formatted_seq_path = []
            for step_i, step_cand in enumerate(seq_path):
                s_vid = step_cand.get("video_id", video_id)
                s_fps = get_video_fps(s_vid)
                s_frame_val = int(step_cand.get("keyframe_index", step_cand.get("frame_idx", 1)))

                # Ưu tiên lấy tên file từ keyframe_path
                kp = step_cand.get("keyframe_path", "")
                if kp:
                    s_img_filename = Path(kp).name
                else:
                    s_img_filename = f"{int(step_cand.get('frame_idx', s_frame_val)):06d}.jpg"

                s_ptime = float(step_cand.get("pts_time", 0.0))
                s_calc_frame = int(round(s_ptime * s_fps))
                s_video_path = get_video_path(s_vid)
                s_has_video = s_video_path is not None

                formatted_seq_path.append(
                    {
                        "step_idx": step_i + 1,
                        "video_id": s_vid,
                        "keyframe_index": s_frame_val,
                        "img_filename": s_img_filename,
                        "keyframe_url": f"/api/keyframe/{s_vid}/{s_img_filename}",
                        "score": float(step_cand.get("score", 0.0)),
                        "pts_time": s_ptime,
                        "formatted_time": format_pts_to_hms(s_ptime),
                        "fps": s_fps,
                        "calc_frame": s_calc_frame,
                        "submission_string": f"{s_vid} / {s_calc_frame}",
                        "has_video": s_has_video,
                        "video_url": f"/api/video/{s_vid}" if s_has_video else None,
                    }
                )

            formatted_results.append(
                {
                    "rank": res.get("rank", 1),
                    "video_id": video_id,
                    "is_temporal": True,
                    "num_steps": len(seq_path),
                    "final_score": float(res.get("final_score", 0.0)),
                    "sequence_path": formatted_seq_path,
                    "fps": fps,
                    "has_video": has_video,
                    "video_url": f"/api/video/{video_id}" if has_video else None,
                }
            )
        else:
            frame_val = int(res.get("keyframe_index", res.get("frame_idx", 1)))

            # Ưu tiên lấy tên file từ keyframe_path (Custom_Keyframes lưu 000000.jpg, 000057.jpg)
            kp = res.get("keyframe_path", "")
            if kp:
                img_filename = Path(kp).name
            else:
                img_filename = f"{int(res.get('frame_idx', frame_val)):06d}.jpg"

            p_time = float(res.get("pts_time", 0.0))
            calc_frame = int(round(p_time * fps))

            formatted_results.append(
                {
                    "rank": res.get("rank", 1),
                    "video_id": video_id,
                    "is_temporal": False,
                    "keyframe_index": frame_val,
                    "img_filename": img_filename,
                    "keyframe_url": f"/api/keyframe/{video_id}/{img_filename}",
                    "pts_time": p_time,
                    "formatted_time": format_pts_to_hms(p_time),
                    "fps": fps,
                    "calc_frame": calc_frame,
                    "submission_string": f"{video_id} / {calc_frame}",
                    "sources": res.get("sources", []),
                    "retrieval_score": float(res.get("score", res.get("retrieval_score", 0.0))),
                    "object_score": float(res.get("object_score", 0.0)),
                    "metadata_score": float(res.get("metadata_score", 0.0)),
                    "ocr_score": float(res.get("ocr_score", 0.0)),
                    "asr_score": float(res.get("asr_score", 0.0)),
                    "final_score": float(res.get("final_score", 0.0)),
                    "metadata": res.get("metadata", {}),
                    "object_entities": res.get("object_entities", []),
                    "has_video": has_video,
                    "video_url": f"/api/video/{video_id}" if has_video else None,
                }
            )

    return {
        "results": formatted_results,
        "total": len(final_results),
        "displayed": len(formatted_results),
        "search_type": search_type,
    }


if __name__ == "__main__":
    GT_JSON_FILE = Path(__file__).parent / "ground_truth.json"
    # Lưu ý: file này là services.py, nếu bạn muốn benchmark riêng thì gọi script benchmark.
    print("This module is intended to be imported by the service layer.")
