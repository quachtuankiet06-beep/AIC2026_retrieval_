
# import streamlit as st
# from pathlib import Path
# from PIL import Image
# import sys
# import json
# from src.reranking.temporal_reranking_pipeline import temporal_sequence_reranking_optimized # (Đường dẫn tùy thuộc vào vị trí file chứa hàm của bạn)
# # Thêm đường dẫn gốc của project vào sys.path để import các module bên trong src
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT))

# from src.mapping.mapping_pipeline import mapping_pipeline
# from src.reranking.reranking_pipeline import reranking_pipeline
# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline,
#     clip_b32_retrieval_pipeline
# )
# from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
# from src.retrieval.temporal_retrieval_multi_model import temporal_retrieval_multi_model_pipeline

# # Cấu hình giao diện Streamlit
# st.set_page_config(
#     page_title="Video Search & Reranking System (Multi-Model & Temporal)",
#     page_icon="🎬",
#     layout="wide"
# )

# # Đường dẫn mặc định đến các Config và Thư mục gốc
# BASE_DIR = ROOT
# MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"
# RETRIEVAL_CONFIG = BASE_DIR / "configs" / "retrieval.yaml"
# RERANK_CONFIG = BASE_DIR / "configs" / "reranking.yaml"
# VIDEO_FPS_MAPPING_PATH = BASE_DIR / "data" / "mapping" / "video_fps_mapping.json"

# @st.cache_resource
# def load_pipeline_configs():
#     print("[INFO] Loading system paths and configs...")
#     return str(MAPPING_PATH), str(RETRIEVAL_CONFIG), str(RERANK_CONFIG)

# @st.cache_data
# def load_video_fps_mapping():
#     """Tải thông tin mapping video_id và fps từ file video_fps_mapping.json"""
#     if VIDEO_FPS_MAPPING_PATH.exists():
#         try:
#             with open(VIDEO_FPS_MAPPING_PATH, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 # Tạo dictionary tra cứu nhanh: {video_id: fps}
#                 return {item["video_id"]: float(item.get("fps", 25.0)) for item in data}
#         except Exception as e:
#             print(f"[WARNING] Không thể load video_fps_mapping.json: {e}")
#     return {}

# mapping_p, ret_cfg, rerank_cfg = load_pipeline_configs()
# video_fps_dict = load_video_fps_mapping()

# def format_pts_to_hms(pts_time):
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

# def get_submission_frame_info(video_id, pts_time):
#     """Tính toán frame_idx từ pts_time và fps của video từ video_fps_mapping.json"""
#     fps = video_fps_dict.get(video_id, 25.0) # Mặc định 25.0 nếu không tìm thấy
#     frame_idx = int(round(float(pts_time) * fps))
#     return frame_idx

# # ==========================================
# # GIAO DIỆN SIDEBAR TÙY CHỈNH NÂNG CAO
# # ==========================================
# st.sidebar.header("⚙️ Tùy chỉnh Hệ thống")

# # 1. Chọn kiểu tìm kiếm: Normal hay Temporal Sequence
# search_type = st.sidebar.selectbox(
#     "Kiểu Truy vấn (Search Type):",
#     ["Standard Retrieval (Truy vấn đơn)", "Temporal Sequence (Truy vấn chuỗi thời gian)"]
# )

# # 2. Chọn chế độ chạy Retrieval
# retrieval_mode = st.sidebar.radio(
#     "Chế độ Retrieval:",
#     ["Multi-Model Ensemble (SigLIP2 + ViT-B/32)", "Single Model (Đơn mô hình)"]
# )

# # Khai báo biến cấu hình model
# selected_models = []
# single_model_choice = "siglip2"

# if retrieval_mode == "Single Model (Đơn mô hình)":
#     single_model_choice = st.sidebar.selectbox(
#         "Chọn Model đơn lẻ:",
#         ["siglip2", "clip_b32"]
#     )
# else:
#     st.sidebar.markdown("**Chọn các Model tham gia Ensemble:**")
#     use_siglip2 = st.sidebar.checkbox("SigLIP 2", value=True)
#     use_clip = st.sidebar.checkbox("CLIP ViT-B/32", value=True)

#     if use_siglip2:
#         selected_models.append({
#             "name": "siglip2", 
#             "index_path": str(BASE_DIR / "data" / "indexes" / "siglib2.index")
#         })
#     if use_clip:
#         selected_models.append({
#             "name": "clip_b32", 
#             "index_path": str(BASE_DIR / "data" / "indexes" / "vit_b32.index")
#         })

# top_k_display = st.sidebar.slider("Số lượng kết quả hiển thị (Top K)", min_value=10, max_value=100, value=20, step=10)

# # Cấu hình phụ nếu bật Temporal Mode (Chuyển sang Keyframe Gap & Beam Width)
# max_kf_gap = 150
# min_kf_gap = 0
# beam_width = 5
# if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("🎞️ **Cấu hình Temporal Beam Search:**")
#     max_kf_gap = st.sidebar.slider("Khoảng cách keyframe tối đa giữa các bước", min_value=1, max_value=500, value=150, step=10)
#     min_kf_gap = st.sidebar.slider("Khoảng cách keyframe tối thiểu giữa các bước", min_value=0, max_value=50, value=0, step=1)
#     beam_width = st.sidebar.slider("Beam Width (Độ rộng nhánh chùm mỗi video)", min_value=1, max_value=100, value=5, step=1)
# # ==========================================
# # CÔNG CỤ CHUYỂN ĐỔI NHANH VIDEO / THỜI GIAN -> FRAME_IDX (DÙNG ĐỂ NỘP BÀI)
# # ==========================================
# st.sidebar.markdown("---")
# st.sidebar.markdown("🛠️ **Converter: Video/Time -> Frame Index**")

# # Ô input nhận vào dạng "L22_V001 / 2:16" hoặc "L22_V001 / 60.2"
# converter_input = st.sidebar.text_input("Nhập chuỗi (VD: L22_V001 / 2:16):", "L22_V001 / 2:16")

# if st.sidebar.button("Tính Frame Index"):
#     try:
#         if "/" in converter_input:
#             parts = converter_input.split("/")
#             target_vid = parts[0].strip()
#             time_str = parts[1].strip()
            
#             # Xử lý thời gian: Hỗ trợ cả dạng giây trực tiếp (60.2) hoặc dạng phút:giây (2:16)
#             if ":" in time_str:
#                 time_parts = time_str.split(":")
#                 if len(time_parts) == 2:
#                     minutes = float(time_parts[0])
#                     seconds = float(time_parts[1])
#                     total_seconds = minutes * 60 + seconds
#                 elif len(time_parts) == 3:
#                     hours = float(time_parts[0])
#                     minutes = float(time_parts[1])
#                     seconds = float(time_parts[2])
#                     total_seconds = hours * 3600 + minutes * 60 + seconds
#                 else:
#                     total_seconds = 0.0
#             else:
#                 total_seconds = float(time_str)
            
#             # Tra cứu FPS từ từ điển đã load sẵn
#             fps = video_fps_dict.get(target_vid, 25.0) # Mặc định 25.0 nếu không tìm thấy video trong mapping
#             calc_frame = int(round(total_seconds * fps))
            
#             # Hiển thị kết quả ngay trên sidebar
#             st.sidebar.success(f"🎯 **Kết quả nộp:** `{target_vid} / {calc_frame}`")
#             st.sidebar.info(f"Chi tiết: FPS của `{target_vid}` là `{fps}`, Tổng thời gian: `{total_seconds}s`")
#         else:
#             st.sidebar.warning("Vui lòng nhập đúng định dạng chứa dấu '/' (VD: L22_V001 / 2:16)")
#     except Exception as ex:
#         st.sidebar.error(f"Lỗi cú pháp: {ex}")# ==========================================
# # GIAO DIỆN CHÍNH
# # ==========================================
# st.title("🎥 Hệ thống Tìm kiếm Video Thông Minh (Multi-Model & Temporal Reranking)")
# if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
#     st.markdown("💡 *Chế độ Temporal đang bật:* Sử dụng dấu `/` để phân tách các bước theo thời gian. Ví dụ: `Cảnh mở cửa / Người đàn ông ngồi vào bàn`")
# else:
#     st.markdown("Nhập câu truy vấn của bạn bằng tiếng Việt hoặc tiếng Anh để tìm kiếm các khung hình (keyframe) chính xác nhất.")

# # Ô nhập query
# default_query = "Cảnh mở cửa / Người đàn ông ngồi vào bàn" if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)" else "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh"
# query = st.text_input("🔍 Nhập nội dung tìm kiếm:", default_query)

# if st.button("🚀 Thực hiện Tìm kiếm", type="primary"):
#     if not query.strip():
#         st.warning("Vui lòng nhập nội dung truy vấn!")
#     else:
#         with st.spinner("Đang thực hiện Retrieval, Mapping và Reranking đa tầng..."):
#             try:
#                 # ==========================================
#                 # LUỒNG 1: TEMPORAL SEQUENCE RETRIEVAL
#                 # ==========================================
#                 if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
#                     if retrieval_mode == "Single Model (Đơn mô hình)":
#                         if single_model_choice == "siglip2":
#                             idx_path = str(BASE_DIR / "data" / "indexes" / "siglib2.index")
#                         else:
#                             idx_path = str(BASE_DIR / "data" / "indexes" / "vit_b32.index")
                        
#                         current_temporal_models = [{"name": single_model_choice, "index_path": idx_path}]
#                     else:
#                         if not selected_models:
#                             st.warning("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")
#                             st.stop()
#                         current_temporal_models = selected_models

#                     # 1. Thực hiện temporal retrieval trả về các chuỗi thô
#                     retrieval_results = temporal_retrieval_multi_model_pipeline(
#                         raw_query_string=query,
#                         model_configs=current_temporal_models,
#                         config_path=ret_cfg,
#                         max_kf_gap=max_kf_gap,
#                         min_kf_gap=min_kf_gap,
#                         beam_width=beam_width
#                     )

#                     if len(retrieval_results) == 0:
#                         st.warning("Không tìm thấy chuỗi sự kiện nào thỏa mãn ràng buộc khoảng cách keyframe.")
#                         st.stop()

#                     # 2. MAPPING ĐÚNG CÁCH CHO TỪNG BƯỚC TRONG SEQUENCE PATH
#                     for seq_item in retrieval_results:
#                         seq_path = seq_item.get("sequence_path", [])
#                         mapped_seq_path = []
#                         for step_cand in seq_path:
#                             mapped_step_list = mapping_pipeline(
#                                 retrieval_results=[step_cand],
#                                 mapping_path=mapping_p,
#                             )
#                             if mapped_step_list:
#                                 mapped_seq_path.append(mapped_step_list[0])
#                             else:
#                                 mapped_seq_path.append(step_cand)
#                         seq_item["sequence_path"] = mapped_seq_path

#                     # # 3. TỐI ƯU RERANKING CHO TEMPORAL
#                     # queries_list = [q.strip() for q in query.split("/") if q.strip()]
#                     # num_queries = len(queries_list)

#                     # for seq_cand in retrieval_results:
#                     #     seq_path = seq_cand.get("sequence_path", [])
#                     #     if not seq_path:
#                     #         seq_cand["final_score"] = float(seq_cand.get("score", 0.0))
#                     #         continue

#                     #     total_final_score = 0.0
#                     #     num_steps = len(seq_path)

#                     #     for step_idx, step_frame in enumerate(seq_path):
#                     #         sub_query = queries_list[step_idx] if step_idx < num_queries else queries_list[-1]
                            
#                     #         single_frame_candidate = [{
#                     #             "vector_index": step_frame.get("vector_index"),
#                     #             "video_id": step_frame.get("video_id"),
#                     #             "keyframe_index": step_frame.get("keyframe_index", 0),
#                     #             "frame_idx": step_frame.get("frame_idx", 0),
#                     #             "pts_time": step_frame.get("pts_time", 0.0),
#                     #             "score": step_frame.get("score", 1.0),
#                     #             "object_entities": step_frame.get("object_entities", []),
#                     #             "ocr_texts": step_frame.get("ocr_texts", []),
#                     #             "asr_text": step_frame.get("asr_text", ""),
#                     #             "metadata": step_frame.get("metadata", {})
#                     #         }]

#                     #         reranked_single = reranking_pipeline(sub_query, single_frame_candidate, rerank_cfg)
#                     #         if reranked_single:
#                     #             total_final_score += reranked_single[0].get("final_score", 0.0)

#                     #     seq_cand["final_score"] = round(total_final_score / num_steps, 6)
#                     # 3. TỐI ƯU RERANKING CHO TEMPORAL
#                     queries_list = [q.strip() for q in query.split("/") if q.strip()]
#                     num_queries = len(queries_list)

#                     for seq_cand in retrieval_results:
#                         seq_path = seq_cand.get("sequence_path", [])
#                         if not seq_path:
#                             seq_cand["final_score"] = float(seq_cand.get("score", 0.0))
#                             continue

#                         total_final_score = 0.0
#                         num_steps = len(seq_path)

#                         for step_idx, step_frame in enumerate(seq_path):
#                             sub_query = queries_list[step_idx] if step_idx < num_queries else queries_list[-1]
                            
#                             single_frame_candidate = [{
#                                 "vector_index": step_frame.get("vector_index"),
#                                 "video_id": step_frame.get("video_id"),
#                                 "keyframe_index": step_frame.get("keyframe_index", 0),
#                                 "frame_idx": step_frame.get("frame_idx", 0),
#                                 "pts_time": step_frame.get("pts_time", 0.0),
#                                 "score": step_frame.get("score", 1.0),
#                                 "object_entities": step_frame.get("object_entities", []),
#                                 "ocr_texts": step_frame.get("ocr_texts", []),
#                                 "asr_text": step_frame.get("asr_text", ""),
#                                 "metadata": step_frame.get("metadata", {})
#                             }]

#                             reranked_single = reranking_pipeline(sub_query, single_frame_candidate, rerank_cfg)
#                             if reranked_single:
#                                 total_final_score += reranked_single[0].get("final_score", 0.0)

#                         seq_cand["final_score"] = round(total_final_score / num_steps, 6)
#                     # 4. Sắp xếp lại theo final_score trung bình và gán rank
#                     final_results = sorted(retrieval_results, key=lambda x: x.get("final_score", 0), reverse=True)
#                     for r_idx, c in enumerate(final_results, start=1):
#                         c["rank"] = r_idx

#                 # ==========================================
#                 # LUỒNG 2: STANDARD RETRIEVAL (Truy vấn đơn thông thường)
#                 # ==========================================
#                 else:
#                     if retrieval_mode == "Single Model (Đơn mô hình)":
#                         if single_model_choice == "siglip2":
#                             idx_path = BASE_DIR / "data" / "indexes" / "siglib2.index"
#                             retrieval_results = siglip2_retrieval_pipeline(query, str(idx_path), ret_cfg)
#                         else:  # clip_b32
#                             idx_path = BASE_DIR / "data" / "indexes" / "vit_b32.index"
#                             retrieval_results = clip_b32_retrieval_pipeline(query, str(idx_path), ret_cfg)
#                     else:
#                         if not selected_models:
#                             st.warning("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")
#                             st.stop()
                        
#                         retrieval_results = retrieval_multi_model_pipeline(
#                             query_text=query,
#                             model_configs=selected_models,
#                             config_path=ret_cfg
#                         )

#                     if len(retrieval_results) == 0:
#                         st.warning("Không tìm thấy kết quả phù hợp từ bước Retrieval.")
#                         st.stop()
#                     else:
#                         candidate_list = mapping_pipeline(
#                             retrieval_results=retrieval_results,
#                             mapping_path=mapping_p,
#                         )

#                         final_results = reranking_pipeline(
#                             query_text=query,
#                             candidate_list=candidate_list,
#                             config_path=rerank_cfg,
#                         )

#                 # ==========================================
#                 # HIỂN THỊ KẾT QUẢ RA GIAO DIỆN (STREAMLIT UI)
#                 # ==========================================
#                 st.success(f"Tìm thấy tổng cộng {len(final_results)} kết quả. Đang hiển thị Top {min(top_k_display, len(final_results))}:")
#                 display_results = final_results[:top_k_display]
                
#                 for res in display_results:
#                     with st.container():
#                         st.markdown(f"### 🏆 Top Rank: **{res['rank']}**")
                        
#                         # --- TRƯỜNG HỢP HIỂN THỊ CHUỖI TEMPORAL SEQUENCE ---
#                         if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)" and "sequence_path" in res:
#                             seq_path = res["sequence_path"]
#                             st.markdown(f"**Video ID (Chung):** `{res['video_id']}` | **Số bước chuỗi:** `{len(seq_path)} bước` | **Final Score trung bình:** `{res.get('final_score', 0):.6f}`")
                            
#                             cols = st.columns(len(seq_path))
#                             for step_i, step_cand in enumerate(seq_path):
#                                 with cols[step_i]:
#                                     st.markdown(f"**Bước {step_i + 1}**")
                                    
#                                     v_id = step_cand.get('video_id', res['video_id'])
#                                     batch_prefix = v_id.split('_')[0] if '_' in v_id else "Keyframes_L21"
#                                     folder_batch = f"Keyframes_{batch_prefix}"
                                    
#                                     frame_val = int(step_cand.get('keyframe_index', step_cand.get('frame_idx', 1)))
#                                     img_filename = f"{frame_val:03d}.jpg"
                                    
#                                     full_img_path = BASE_DIR / "data" / "keyframes" / folder_batch / "keyframes" / v_id / img_filename
#                                     if not full_img_path.exists():
#                                         found_files = list(BASE_DIR.glob(f"**/keyframes/{v_id}/{img_filename}"))
#                                         if found_files:
#                                             full_img_path = found_files[0]

#                                     # Hiển thị ảnh
#                                     if full_img_path and Path(full_img_path).exists():
#                                         image = Image.open(full_img_path)
#                                         st.image(image, caption=f"Bước {step_i+1} - KF: {frame_val}", width="stretch")
#                                     else:
#                                         st.error(f"Không tìm thấy file: {img_filename}")
                                    
#                                     # --- XỬ LÝ THỜI GIAN & ĐỊNH DẠNG NỘP BÀI ---
#                                     p_time = step_cand.get('pts_time', 0.0)
#                                     hms_str = format_pts_to_hms(p_time)
#                                     calc_frame = get_submission_frame_info(v_id, p_time)

#                                     # Hiển thị chi tiết từng bước
#                                     st.markdown(
#                                         f"""
#                                         - **Video ID:** `{v_id}`  
#                                         - **Step Score:** `{step_cand.get('score', 0):.4f}`  
#                                         - **KF Index:** `{frame_val}`  
#                                         - **PTS Time:** `{p_time:.2f}s` (`{hms_str}`)  
#                                         - 🎯 **Nộp bài:** `{v_id} / {calc_frame}`
#                                         """
#                                     )

#                         # --- TRƯỜNG HỢP HIỂN THỊ STANDARD RETRIEVAL ---
#                         else:
#                             col1, col2 = st.columns([1, 2])
#                             with col1:
#                                 video_id = res.get('video_id', '')
#                                 batch_prefix = video_id.split('_')[0] if '_' in video_id else "Keyframes_L21"
#                                 folder_batch = f"Keyframes_{batch_prefix}"
                                
#                                 frame_val = int(res.get('keyframe_index', res.get('frame_idx', 1)))
#                                 img_filename = f"{frame_val:03d}.jpg"
                                
#                                 full_img_path = BASE_DIR / "data" / "keyframes" / folder_batch / "keyframes" / video_id / img_filename
                                
#                                 if not full_img_path.exists():
#                                     found_files = list(BASE_DIR.glob(f"**/keyframes/{video_id}/{img_filename}"))
#                                     if found_files:
#                                         full_img_path = found_files[0]

#                                 if full_img_path and Path(full_img_path).exists():
#                                     image = Image.open(full_img_path)
#                                     st.image(image, caption=f"Rank {res['rank']} - {video_id}/{img_filename}", width="stretch")
#                                 else:
#                                     st.error(f"Không tìm thấy file ảnh tại: {full_img_path}")

#                             with col2:
#                                 p_time = res.get('pts_time', 0.0)
#                                 hms_str = format_pts_to_hms(p_time)
#                                 calc_frame = get_submission_frame_info(video_id, p_time)

#                                 st.markdown(f"**Video ID:** `{res['video_id']}` | **Keyframe Index:** `{res['keyframe_index']}` | **PTS Time:** `{p_time:.2f}s` (`{hms_str}`)")
                                
#                                 if "sources" in res:
#                                     st.markdown(f"**Models Found:** `{', '.join(res['sources'])}`")

#                                 st.markdown(
#                                     f"""
#                                     - **Retrieval Score:** `{res.get('score', res.get('retrieval_score', 0)):.4f}`  
#                                     - **Object Score:** `{res.get('object_score', 0):.4f}`  
#                                     - **Metadata Score:** `{res.get('metadata_score', 0):.4f}`  
#                                     - **OCR Score:** `{res.get('ocr_score', 0):.4f}`  
#                                     - **ASR Score:** `{res.get('asr_score', 0):.4f}`  
#                                     - **RRF Final Score:** **`{res.get('final_score', 0):.6f}`** - 🎯 **Định dạng Nộp bài:** `{video_id} / {calc_frame}`
#                                     """
#                                 )
                                
#                                 metadata = res.get("metadata", {})
#                                 if metadata:
#                                     st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
                                
#                                 objects = res.get("object_entities", [])
#                                 if objects:
#                                     st.markdown(f"**Detected Objects:** {', '.join(objects[:10])}")
                                    
#                         st.divider()

#             except Exception as e:
#                 st.error(f"Đã xảy ra lỗi trong quá trình chạy pipeline: {str(e)}")
#                 import traceback
#                 st.text(traceback.format_exc())

import streamlit as st
from pathlib import Path
from PIL import Image
import sys
import json
from src.reranking.temporal_reranking_pipeline import temporal_sequence_reranking_optimized

# Thêm đường dẫn gốc của project vào sys.path để import các module bên trong src
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.mapping.mapping_pipeline import mapping_pipeline
from src.reranking.reranking_pipeline import reranking_pipeline
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline,
    dfn5b_vit_h14_retrieval_pipeline
)
from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
from src.retrieval.temporal_retrieval_multi_model import temporal_retrieval_multi_model_pipeline

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Video Search & Reranking System (Multi-Model & Temporal)",
    page_icon="🎬",
    layout="wide"
)

# Đường dẫn mặc định đến các Config và Thư mục gốc
BASE_DIR = ROOT
MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"
RETRIEVAL_CONFIG = BASE_DIR / "configs" / "retrieval.yaml"
RERANK_CONFIG = BASE_DIR / "configs" / "reranking.yaml"
VIDEO_FPS_MAPPING_PATH = BASE_DIR / "data" / "mapping" / "video_fps_mapping.json"

@st.cache_resource
def load_pipeline_configs():
    print("[INFO] Loading system paths and configs...")
    return str(MAPPING_PATH), str(RETRIEVAL_CONFIG), str(RERANK_CONFIG)

@st.cache_data
def load_video_fps_mapping():
    """Tải thông tin mapping video_id và fps từ file video_fps_mapping.json"""
    if VIDEO_FPS_MAPPING_PATH.exists():
        try:
            with open(VIDEO_FPS_MAPPING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {item["video_id"]: float(item.get("fps", 25.0)) for item in data}
        except Exception as e:
            print(f"[WARNING] Không thể load video_fps_mapping.json: {e}")
    return {}

mapping_p, ret_cfg, rerank_cfg = load_pipeline_configs()
video_fps_dict = load_video_fps_mapping()

def format_pts_to_hms(pts_time):
    """Chuyển đổi pts_time (giây) sang định dạng Giờ:Phút:Giây (HH:MM:SS hoặc MM:SS)"""
    try:
        total_seconds = float(pts_time)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    except Exception:
        return "00:00"

def get_submission_frame_info(video_id, pts_time):
    """Tính toán frame_idx từ pts_time và fps của video từ video_fps_mapping.json"""
    fps = video_fps_dict.get(video_id, 25.0)
    frame_idx = int(round(float(pts_time) * fps))
    return frame_idx

# ==========================================
# GIAO DIỆN SIDEBAR TÙY CHỈNH NÂNG CAO
# ==========================================
st.sidebar.header("⚙️ Tùy chỉnh Hệ thống")

# 1. Chọn kiểu tìm kiếm: Normal hay Temporal Sequence
search_type = st.sidebar.selectbox(
    "Kiểu Truy vấn (Search Type):",
    ["Standard Retrieval (Truy vấn đơn)", "Temporal Sequence (Truy vấn chuỗi thời gian)"]
)

# 2. Chọn chế độ chạy Retrieval
retrieval_mode = st.sidebar.radio(
    "Chế độ Retrieval:",
    ["Multi-Model Ensemble (SigLIP2 + DFN5B ViT-H/14)", "Single Model (Đơn mô hình)"]
)

# Khai báo biến cấu hình model
selected_models = []
single_model_choice = "siglip2"

if retrieval_mode == "Single Model (Đơn mô hình)":
    single_model_choice = st.sidebar.selectbox(
        "Chọn Model đơn lẻ:",
        ["siglip2", "dfn5b_vit_h14"]
    )
else:
    st.sidebar.markdown("**Chọn các Model tham gia Ensemble:**")
    use_siglip2 = st.sidebar.checkbox("SigLIP 2", value=True)
    use_dfn5b = st.sidebar.checkbox("DFN5B ViT-H/14", value=True)

    if use_siglip2:
        selected_models.append({
            "name": "siglip2", 
            "index_path": str(BASE_DIR / "data" / "indexes" / "siglib2.index")
        })
    if use_dfn5b:
        selected_models.append({
            "name": "dfn5b_vit_h14", 
            "index_path": str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index")
        })

top_k_display = st.sidebar.slider("Số lượng kết quả hiển thị (Top K)", min_value=10, max_value=100, value=20, step=10)

# Cấu hình phụ nếu bật Temporal Mode
max_kf_gap = 150
min_kf_gap = 0
beam_width = 5
if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎞️ **Cấu hình Temporal Beam Search:**")
    max_kf_gap = st.sidebar.slider("Khoảng cách keyframe tối đa giữa các bước", min_value=1, max_value=500, value=150, step=10)
    min_kf_gap = st.sidebar.slider("Khoảng cách keyframe tối thiểu giữa các bước", min_value=0, max_value=50, value=0, step=1)
    beam_width = st.sidebar.slider("Beam Width (Độ rộng nhánh chùm mỗi video)", min_value=1, max_value=100, value=5, step=1)

# ==========================================
# CÔNG CỤ CHUYỂN ĐỔI NHANH VIDEO / THỜI GIAN -> FRAME_IDX
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("🛠️ **Converter: Video/Time -> Frame Index**")

converter_input = st.sidebar.text_input("Nhập chuỗi (VD: L22_V001 / 2:16):", "L22_V001 / 2:16")

if st.sidebar.button("Tính Frame Index"):
    try:
        if "/" in converter_input:
            parts = converter_input.split("/")
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
            
            fps = video_fps_dict.get(target_vid, 25.0)
            calc_frame = int(round(total_seconds * fps))
            
            st.sidebar.success(f"🎯 **Kết quả nộp:** `{target_vid} / {calc_frame}`")
            st.sidebar.info(f"Chi tiết: FPS của `{target_vid}` là `{fps}`, Tổng thời gian: `{total_seconds}s`")
        else:
            st.sidebar.warning("Vui lòng nhập đúng định dạng chứa dấu '/' (VD: L22_V001 / 2:16)")
    except Exception as ex:
        st.sidebar.error(f"Lỗi cú pháp: {ex}")

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🎥 Hệ thống Tìm kiếm Video Thông Minh (Multi-Model & Temporal Reranking)")
if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
    st.markdown("💡 *Chế độ Temporal đang bật:* Sử dụng dấu `/` để phân tách các bước theo thời gian. Ví dụ: `Cảnh mở cửa / Người đàn ông ngồi vào bàn`")
else:
    st.markdown("Nhập câu truy vấn của bạn bằng tiếng Việt hoặc tiếng Anh để tìm kiếm các khung hình (keyframe) chính xác nhất.")

default_query = "Cảnh mở cửa / Người đàn ông ngồi vào bàn" if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)" else "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh"
query = st.text_input("🔍 Nhập nội dung tìm kiếm:", default_query)

if st.button("🚀 Thực hiện Tìm kiếm", type="primary"):
    if not query.strip():
        st.warning("Vui lòng nhập nội dung truy vấn!")
    else:
        with st.spinner("Đang thực hiện Retrieval, Mapping và Reranking đa tầng..."):
            try:
                # ==========================================
                # LUỒNG 1: TEMPORAL SEQUENCE RETRIEVAL
                # ==========================================
                if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)":
                    if retrieval_mode == "Single Model (Đơn mô hình)":
                        if single_model_choice == "siglip2":
                            idx_path = str(BASE_DIR / "data" / "indexes" / "siglib2.index")
                        else:
                            idx_path = str(BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index")
                        
                        current_temporal_models = [{"name": single_model_choice, "index_path": idx_path}]
                    else:
                        if not selected_models:
                            st.warning("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")
                            st.stop()
                        current_temporal_models = selected_models

                    retrieval_results = temporal_retrieval_multi_model_pipeline(
                        raw_query_string=query,
                        model_configs=current_temporal_models,
                        config_path=ret_cfg,
                        max_kf_gap=max_kf_gap,
                        min_kf_gap=min_kf_gap,
                        beam_width=beam_width
                    )

                    if len(retrieval_results) == 0:
                        st.warning("Không tìm thấy chuỗi sự kiện nào thỏa mãn ràng buộc khoảng cách keyframe.")
                        st.stop()

                    for seq_item in retrieval_results:
                        seq_path = seq_item.get("sequence_path", [])
                        mapped_seq_path = []
                        for step_cand in seq_path:
                            mapped_step_list = mapping_pipeline(
                                retrieval_results=[step_cand],
                                mapping_path=mapping_p,
                            )
                            if mapped_step_list:
                                mapped_seq_path.append(mapped_step_list[0])
                            else:
                                mapped_seq_path.append(step_cand)
                        seq_item["sequence_path"] = mapped_seq_path

                    # ==========================================
                    # 3. TỐI ƯU RERANKING CHO TEMPORAL (Dùng temporal_sequence_reranking_optimized)
                    # ==========================================
                    # queries_list = [q.strip() for q in query.split("/") if q.strip()]
                    
                    # Gọi trực tiếp module reranking chuyên dụng cho chuỗi thời gian
                    # Hàm này nhận toàn bộ danh sách chuỗi kết quả (retrieval_results) và xử lý tối ưu
                    final_results = temporal_sequence_reranking_optimized(
                        raw_query_string=query,
                        sequence_candidates=retrieval_results,
                        config_path=rerank_cfg
                    )

                    # Sắp xếp lại theo final_score giảm dần và gán lại rank chuẩn
                    final_results = sorted(final_results, key=lambda x: x.get("final_score", 0), reverse=True)
                    for r_idx, c in enumerate(final_results, start=1):
                        c["rank"] = r_idx

                # ==========================================
                # LUỒNG 2: STANDARD RETRIEVAL
                # ==========================================
                else:
                    if retrieval_mode == "Single Model (Đơn mô hình)":
                        if single_model_choice == "siglip2":
                            idx_path = BASE_DIR / "data" / "indexes" / "siglib2.index"
                            retrieval_results = siglip2_retrieval_pipeline(query, str(idx_path), ret_cfg)
                        else:  # dfn5b_vit_h14
                            idx_path = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index"
                            retrieval_results = dfn5b_vit_h14_retrieval_pipeline(query, str(idx_path), ret_cfg)
                    else:
                        if not selected_models:
                            st.warning("Vui lòng chọn ít nhất một mô hình trong chế độ Multi-Model!")
                            st.stop()
                        
                        retrieval_results = retrieval_multi_model_pipeline(
                            query_text=query,
                            model_configs=selected_models,
                            config_path=ret_cfg
                        )

                    if len(retrieval_results) == 0:
                        st.warning("Không tìm thấy kết quả phù hợp từ bước Retrieval.")
                        st.stop()
                    else:
                        candidate_list = mapping_pipeline(
                            retrieval_results=retrieval_results,
                            mapping_path=mapping_p,
                        )

                        final_results = reranking_pipeline(
                            query_text=query,
                            candidate_list=candidate_list,
                            config_path=rerank_cfg,
                        )

                # ==========================================
                # HIỂN THỊ KẾT QUẢ RA GIAO DIỆN (STREAMLIT UI)
                # ==========================================
                st.success(f"Tìm thấy tổng cộng {len(final_results)} kết quả. Đang hiển thị Top {min(top_k_display, len(final_results))}:")
                display_results = final_results[:top_k_display]
                
                for res in display_results:
                    with st.container():
                        st.markdown(f"### 🏆 Top Rank: **{res['rank']}**")
                        
                        if search_type == "Temporal Sequence (Truy vấn chuỗi thời gian)" and "sequence_path" in res:
                            seq_path = res["sequence_path"]
                            st.markdown(f"**Video ID (Chung):** `{res['video_id']}` | **Số bước chuỗi:** `{len(seq_path)} bước` | **Final Score trung bình:** `{res.get('final_score', 0):.6f}`")
                            
                            cols = st.columns(len(seq_path))
                            for step_i, step_cand in enumerate(seq_path):
                                with cols[step_i]:
                                    st.markdown(f"**Bước {step_i + 1}**")
                                    
                                    v_id = step_cand.get('video_id', res['video_id'])
                                    batch_prefix = v_id.split('_')[0] if '_' in v_id else "Keyframes_L21"
                                    folder_batch = f"Keyframes_{batch_prefix}"
                                    
                                    frame_val = int(step_cand.get('keyframe_index', step_cand.get('frame_idx', 1)))
                                    img_filename = f"{frame_val:03d}.jpg"
                                    
                                    full_img_path = BASE_DIR / "data" / "keyframes" / folder_batch / "keyframes" / v_id / img_filename
                                    if not full_img_path.exists():
                                        found_files = list(BASE_DIR.glob(f"**/keyframes/{v_id}/{img_filename}"))
                                        if found_files:
                                            full_img_path = found_files[0]

                                    if full_img_path and Path(full_img_path).exists():
                                        image = Image.open(full_img_path)
                                        st.image(image, caption=f"Bước {step_i+1} - KF: {frame_val}", width="stretch")
                                    else:
                                        st.error(f"Không tìm thấy file: {img_filename}")
                                    
                                    p_time = step_cand.get('pts_time', 0.0)
                                    hms_str = format_pts_to_hms(p_time)
                                    calc_frame = get_submission_frame_info(v_id, p_time)

                                    st.markdown(
                                        f"""
                                        - **Video ID:** `{v_id}`  
                                        - **Step Score:** `{step_cand.get('score', 0):.4f}`  
                                        - **KF Index:** `{frame_val}`  
                                        - **PTS Time:** `{p_time:.2f}s` (`{hms_str}`)  
                                        - 🎯 **Nộp bài:** `{v_id} / {calc_frame}`
                                        """
                                    )
                        else:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                video_id = res.get('video_id', '')
                                batch_prefix = video_id.split('_')[0] if '_' in video_id else "Keyframes_L21"
                                folder_batch = f"Keyframes_{batch_prefix}"
                                
                                frame_val = int(res.get('keyframe_index', res.get('frame_idx', 1)))
                                img_filename = f"{frame_val:03d}.jpg"
                                
                                full_img_path = BASE_DIR / "data" / "keyframes" / folder_batch / "keyframes" / video_id / img_filename
                                
                                if not full_img_path.exists():
                                    found_files = list(BASE_DIR.glob(f"**/keyframes/{video_id}/{img_filename}"))
                                    if found_files:
                                        full_img_path = found_files[0]

                                if full_img_path and Path(full_img_path).exists():
                                    image = Image.open(full_img_path)
                                    st.image(image, caption=f"Rank {res['rank']} - {video_id}/{img_filename}", width="stretch")
                                else:
                                    st.error(f"Không tìm thấy file ảnh tại: {full_img_path}")

                            with col2:
                                p_time = res.get('pts_time', 0.0)
                                hms_str = format_pts_to_hms(p_time)
                                calc_frame = get_submission_frame_info(video_id, p_time)

                                st.markdown(f"**Video ID:** `{res['video_id']}` | **Keyframe Index:** `{res['keyframe_index']}` | **PTS Time:** `{p_time:.2f}s` (`{hms_str}`)")
                                
                                if "sources" in res:
                                    st.markdown(f"**Models Found:** `{', '.join(res['sources'])}`")

                                st.markdown(
                                    f"""
                                    - **Retrieval Score:** `{res.get('score', res.get('retrieval_score', 0)):.4f}`  
                                    - **Object Score:** `{res.get('object_score', 0):.4f}`  
                                    - **Metadata Score:** `{res.get('metadata_score', 0):.4f}`  
                                    - **OCR Score:** `{res.get('ocr_score', 0):.4f}`  
                                    - **ASR Score:** `{res.get('asr_score', 0):.4f}`  
                                    - **RRF Final Score:** **`{res.get('final_score', 0):.6f}`** - 🎯 **Định dạng Nộp bài:** `{video_id} / {calc_frame}`
                                    """
                                )
                                
                                metadata = res.get("metadata", {})
                                if metadata:
                                    st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
                                
                                objects = res.get("object_entities", [])
                                if objects:
                                    st.markdown(f"**Detected Objects:** {', '.join(objects[:10])}")
                                    
                        st.divider()

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình chạy pipeline: {str(e)}")
                import traceback
                st.text(traceback.format_exc())