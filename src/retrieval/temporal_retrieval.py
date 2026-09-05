# import numpy as np
# from pathlib import Path
# import sys
# from collections import defaultdict
# from src.mapping.mapping_pipeline import mapping_pipeline
# from src.reranking.temporal_reranking import (
#     temporal_sequence_reranking
# )
# ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT))

# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline,
#     clip_b32_retrieval_pipeline,
#     load_config
# )



# def temporal_nms(candidates, window=10):

#     if not candidates:
#         return []

#     video_groups = defaultdict(list)

#     for cand in candidates:
#         video_groups[cand["video_id"]].append(cand)

#     filtered = []

#     for video_id, frames in video_groups.items():

#         # score cao nhất trước
#         frames = sorted(
#             frames,
#             key=lambda x: x["score"],
#             reverse=True
#         )

#         kept = []

#         for frame in frames:

#             keep = True

#             for exist in kept:

#                 if abs(
#                     frame["keyframe_index"]
#                     - exist["keyframe_index"]
#                 ) <= window:

#                     keep = False
#                     break

#             if keep:
#                 kept.append(frame)

#         filtered.extend(kept)

#     return filtered

# # def run_single_query_pipeline(query_text, index_path, config_path, model_type="siglip2"):
# #     """
# #     Chạy pipeline retrieval cho một query đơn lẻ dựa trên loại model được chỉ định.
# #     """
# #     m_lower = model_type.lower()
# #     if m_lower == "siglip2":
# #         return siglip2_retrieval_pipeline(query_text, index_path, config_path)
# #     elif m_lower in ["clip_b32", "vit_b32", "clip"]:
# #         return clip_b32_retrieval_pipeline(query_text, index_path, config_path)
# #     else:
# #         raise ValueError(f"Model type '{model_type}' không được hỗ trợ trong temporal retrieval.")

# def run_single_query_pipeline(query_text, index_path, config_path, model_type="siglip2"):
#     """
#     Chạy pipeline retrieval cho một query đơn lẻ dựa trên loại model được chỉ định.
#     Đã loại bỏ clip_b32 và tích hợp model mới (dfn5b_clip_vit_h14).
#     """
#     m_lower = model_type.lower()
    
#     if m_lower == "siglip2":
#         return siglip2_retrieval_pipeline(query_text, index_path, config_path)
#     elif m_lower in ["dfn5b", "dfn5b_clip_vit_h14", "vit_h14"]:
#         return dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path)
#     else:
#         raise ValueError(f"Model type '{model_type}' không được hỗ trợ trong temporal retrieval.")
        
# def temporal_sequence_retrieval(
#     queries,
#     index_paths,
#     config_path,
#     model_type="siglip2",
#     max_kf_gap=150,     # Khoảng cách tối đa giữa 2 keyframe liên tiếp
#     min_kf_gap=0,       # Khoảng cách tối thiểu
#     mapping_path="data/indexes/keyframes_mapping.json",
#     beam_width=5,        # Độ rộng chùm tìm kiếm (Beam Width) cho mỗi video
#     temporal_nms_window=10
# ):
#     """
#     Thực hiện truy vấn chuỗi tuần tự sử dụng Beam Search theo từng video_id:
#     - Mỗi video_id sẽ được chạy một nhánh Beam Search độc lập để tìm ra chuỗi sự kiện khớp nhất bên trong video đó.
#     - Cuối cùng, tổng hợp các chuỗi tốt nhất từ tất cả các video và sắp xếp theo điểm tích lũy.
#     """
#     if not queries:
#         return []

#     num_steps = len(queries)
#     if isinstance(index_paths, str):
#         index_paths = [index_paths] * num_steps

#     # Bước 1: Lấy top-K kết quả độc lập cho từng query trong chuỗi và ánh xạ qua mapping_pipeline
#     step_results = []
#     for i, q_text in enumerate(queries):
#         idx_p = index_paths[i] if i < len(index_paths) else index_paths[0]
#         print(f"[INFO] Running Temporal Step {i+1}/{num_steps} for query: '{q_text}'")
        
#         res = run_single_query_pipeline(q_text, idx_p, config_path, model_type=model_type)
        
#         # Gắn rank bắt buộc trước khi qua mapping_pipeline
#         for r_idx, item in enumerate(res, start=1):
#             item["rank"] = r_idx
            
#         enriched_res = mapping_pipeline(
#             retrieval_results=res,
#             mapping_path=mapping_path
#         )

#         enriched_res = temporal_nms(
#             enriched_res,
#             window=temporal_nms_window
#         )

#         for r_idx, item in enumerate(enriched_res, start=1):
#             item["rank"] = r_idx
#         step_results.append(enriched_res)

#     if not step_results or not step_results[0]:
#         return []

#     # Bước 2: Nhóm các ứng viên theo từng video_id ở mỗi bước để tối ưu việc tra cứu
#     # step_by_video[step_idx][video_id] = [Danh sách các candidate thuộc video này ở bước đó]
#     steps_by_video = []
#     for step_candidates in step_results:
#         video_bucket = defaultdict(list)
#         for cand in step_candidates:
#             v_id = cand.get("video_id")
#             if v_id:
#                 video_bucket[v_id].append(cand)
#         steps_by_video.append(video_bucket)

#     # Lấy danh sách tất cả các video_id xuất hiện ở bước đầu tiên (Query 1)
#     initial_video_ids = list(steps_by_video[0].keys())
#     best_sequences_per_video = []

#     # Bước 3: Chạy Beam Search độc lập cho TỪNG `video_id`
#     for v_id in initial_video_ids:
#         # Khởi tạo beam ban đầu cho video_id này từ Query 1
#         initial_candidates = steps_by_video[0][v_id]
#         if not initial_candidates:
#             continue

#         beam_sequences = []
#         for cand in initial_candidates:
#             kf_idx = int(cand["keyframe_index"])
#             score = float(cand["score"])

#             beam_sequences.append({
#                 "video_id": v_id,

#                 # Beam
#                 "sequence_path": [cand],
#                 "cumulative_score": score,
#                 "last_kf_idx": kf_idx,
#                 "used_keyframes": {kf_idx},

#                 # Temporal reranking
#                 "candidate_scores": [score],
#                 "candidate_ranks": [cand["rank"]],
#                 "candidate_keyframes": [kf_idx]
#             })

#         # Sắp xếp và giữ lại beam_width nhánh tốt nhất cho video này
#         beam_sequences = sorted(beam_sequences, key=lambda x: x["cumulative_score"], reverse=True)[:beam_width]

#         # Duyệt qua các bước tiếp theo (từ bước 1 đến hết)
#         is_video_sequence_valid = True
#         for step_idx in range(1, num_steps):
#             current_video_candidates = steps_by_video[step_idx].get(v_id, [])
#             if not current_video_candidates:
#                 is_video_sequence_valid = False
#                 break

#             candidate_extensions = []
#             for seq in beam_sequences:
#                 last_kf = seq["last_kf_idx"]

#                 for cand in current_video_candidates:
#                     curr_kf = float(cand.get("keyframe_index", 0.0))
#                     kf_diff = curr_kf - last_kf  # Khoảng cách keyframe

#                     # Kiểm tra ràng buộc Keyframe Gap
#                     if min_kf_gap <= kf_diff <= max_kf_gap:
#                         too_close = any(
#                             abs(curr_kf - prev_kf) <= temporal_nms_window
#                             for prev_kf in seq["used_keyframes"]
#                         )

#                         if too_close:
#                             continue
#                         cand_score = float(cand.get("score", 0.0))
                        
#                         new_seq = {
#                             "video_id": v_id,

#                             "sequence_path":
#                                 seq["sequence_path"] + [cand],

#                             "cumulative_score":
#                                 seq["cumulative_score"] + cand_score,

#                             "last_kf_idx":
#                                 curr_kf,

#                             "used_keyframes":
#                                 seq["used_keyframes"] | {curr_kf},

#                             "candidate_scores":
#                                 seq["candidate_scores"] + [cand_score],

#                             "candidate_ranks":
#                                 seq["candidate_ranks"] + [cand["rank"]],

#                             "candidate_keyframes":
#                                 seq["candidate_keyframes"] + [curr_kf]
#                         }
#                         candidate_extensions.append(new_seq)

#             if not candidate_extensions:
#                 is_video_sequence_valid = False
#                 break

#             # Lọc và giữ lại beam_width nhánh tốt nhất cho video này tại bước hiện tại
#             beam_sequences = sorted(
#                 candidate_extensions,
#                 key=lambda x: x["cumulative_score"],
#                 reverse=True
#             )[:beam_width]

#         if is_video_sequence_valid and beam_sequences:
#             # Lấy chuỗi tốt nhất (có cumulative_score cao nhất) trong video này
#             best_seq_for_video = max(beam_sequences, key=lambda x: x["cumulative_score"])
#             sorted_beams = sorted(
#                 beam_sequences,
#                 key=lambda x: x["cumulative_score"],
#                 reverse=True
#             )

#             if len(sorted_beams) >= 2:
#                 margin = (
#                     sorted_beams[0]["cumulative_score"]
#                     -
#                     sorted_beams[1]["cumulative_score"]
#                 )
#             else:
#                 margin = sorted_beams[0]["cumulative_score"]

#             best_seq_for_video["beam_margin"] = margin
#             best_sequences_per_video.append(best_seq_for_video)

#     if not best_sequences_per_video:
#         print("[INFO] Không tìm thấy chuỗi sự kiện nào thỏa mãn toàn bộ ràng buộc keyframe gap theo từng video.")
#         return []

#     # Bước 4: Sắp xếp lại toàn bộ các chuỗi tốt nhất từ các video dựa trên điểm trung bình cộng tích lũy
#     reranked_sequences = temporal_sequence_reranking(
#         temporal_results=best_sequences_per_video,
#         config_path=config_path
#     )

#     # Bước 5: Định dạng lại đầu ra chuẩn dạng danh sách keyframe đại diện
#     final_temporal_results = []
#     for rank_idx, seq in enumerate(reranked_sequences[:100], start=1):
#         representative_cand = seq["sequence_path"][-1] 
        
#         final_temporal_results.append({
#             "rank": rank_idx,
#             "vector_index": representative_cand.get("vector_index"),
#             "score": seq["rerank_score"],
#             "video_id": seq["video_id"],
#             "frame_idx": representative_cand.get("frame_idx"),
#             "keyframe_index": representative_cand.get("keyframe_index"),
#             "pts_time": representative_cand.get("pts_time"),
#             "fps": representative_cand.get("fps"),
#             "sequence_path": seq["sequence_path"],
#             "candidate_scores":
#                 seq["candidate_scores"],

#             "candidate_ranks":
#                 seq["candidate_ranks"],

#             "candidate_keyframes":
#                 seq["candidate_keyframes"],

#             "beam_margin":
#                 seq["beam_margin"],
#             "match_type": "temporal_sequence_beam_per_video"
#         })

#     return final_temporal_results


# def parse_temporal_query(raw_query_string):
#     """
#     Nhận vào chuỗi truy vấn thô từ người dùng, cắt theo dấu '/' 
#     và làm sạch khoảng trắng ở mỗi phần.
#     Ví dụ: "Cảnh mở cửa / Người đàn ông ngồi vào bàn" 
#     -> ["Cảnh mở cửa", "Người đàn ông ngồi vào bàn"]
#     """
#     if not raw_query_string or not isinstance(raw_query_string, str):
#         return []
    
#     queries = [q.strip() for q in raw_query_string.split("/") if q.strip()]
#     return queries


# def temporal_retrieval_pipeline_from_string(
#     raw_query_string,
#     index_paths,
#     config_path,
#     model_type="siglip2",
#     max_kf_gap=150,     
#     min_kf_gap=0,      
#     mapping_path="data/indexes/keyframes_mapping.json",
#     beam_width=5,
#     temporal_nms_window=10
# ):
#     """
#     Hàm pipeline hoàn chỉnh: Nhận chuỗi thô có chứa dấu '/' -> Tự tách query -> Chạy chuỗi temporal retrieval theo Beam Search trên từng video_id.
#     """
#     queries = parse_temporal_query(raw_query_string)
    
#     if not queries:
#         print("[WARNING] Truy vấn rỗng hoặc không có định dạng hợp lệ.")
#         return []
    
#     print(f"[INFO] Phân rã truy vấn theo thời gian thành {len(queries)} bước (Beam Search per Video Mode):")
#     for idx, q in enumerate(queries, 1):
#         print(f"  - Bước {idx}: {q}")

#     # Gọi hàm xử lý chuỗi tuần tự theo từng video_id
#     return temporal_sequence_retrieval(
#         queries=queries,
#         index_paths=index_paths,
#         config_path=config_path,
#         model_type=model_type,
#         max_kf_gap=max_kf_gap,
#         min_kf_gap=min_kf_gap,
#         mapping_path=mapping_path,
#         beam_width=beam_width,
#         temporal_nms_window=temporal_nms_window
#     )


import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.mapping.mapping_pipeline import mapping_pipeline
from src.reranking.temporal_reranking import temporal_sequence_reranking
from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline


def temporal_nms_time(candidates, min_gap_sec=1.0, score_key="score"):
    """
    Temporal NMS dựa trên thời gian thực (pts_time) thay vì index khung hình.
    """
    if not candidates:
        return []

    video_groups = defaultdict(list)
    for cand in candidates:
        video_groups[cand.get("video_id", "__unknown__")].append(cand)

    filtered = []
    for video_id, frames in video_groups.items():
        frames_sorted = sorted(frames, key=lambda x: x.get(score_key, 0.0), reverse=True)
        kept = []
        for frame in frames_sorted:
            f_time = frame.get("pts_time")
            if f_time is None:
                f_time = frame.get("frame_idx", 0) / 25.0

            keep = True
            for exist in kept:
                e_time = exist.get("pts_time")
                if e_time is None:
                    e_time = exist.get("frame_idx", 0) / 25.0

                if abs(f_time - e_time) < min_gap_sec:
                    keep = False
                    break

            if keep:
                kept.append(frame)

        filtered.extend(kept)

    return filtered


def temporal_sequence_retrieval(
    queries,
    model_configs,
    config_path,
    max_kf_gap=150,     
    min_kf_gap=1,       
    mapping_path="data/indexes/keyframes_new.db",
    beam_width=5,        
    min_gap_sec=1.0,
    top_per_video=30
):
    if not queries:
        return []

    num_steps = len(queries)
    step_results = []

    for i, q_text in enumerate(queries):
        print(f"[INFO] Running Multi-Model Temporal Step {i+1}/{num_steps} for query: '{q_text}'")
        
        res = retrieval_multi_model_pipeline(
            query_text=q_text,
            model_configs=model_configs,
            config_path=config_path
        )
        
        if not res:
            step_results.append([])
            continue
            
        enriched_res = mapping_pipeline(
            retrieval_results=res,
            mapping_db_path=mapping_path
        )
        
        # NMS theo thời gian thực
        enriched_res = temporal_nms_time(enriched_res, min_gap_sec=min_gap_sec)

        # Giới hạn top_per_video
        video_bucket = defaultdict(list)
        for cand in enriched_res:
            video_bucket[cand["video_id"]].append(cand)

        filtered_candidates = []
        for frames in video_bucket.values():
            frames_sorted = sorted(frames, key=lambda x: x.get("score", 0.0), reverse=True)[:top_per_video]
            filtered_candidates.extend(frames_sorted)

        # SỬA LỖI 1: MUST SORT TOÀN CỤC TRƯỚC KHU GÁN RANK GLOBAL
        filtered_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for r_idx, item in enumerate(filtered_candidates, start=1):
            item["rank"] = r_idx
            
        step_results.append(filtered_candidates)

    if not step_results or not step_results[0]:
        return []

    steps_by_video = []
    for step_candidates in step_results:
        video_bucket = defaultdict(list)
        for cand in step_candidates:
            v_id = cand.get("video_id")
            if v_id:
                video_bucket[v_id].append(cand)
        steps_by_video.append(video_bucket)

    # SỬA LỖI 2: Lấy giao của các video xuất hiện ở BẤT KỲ step nào thay vì chỉ duy nhất Step 1
    all_video_ids = set()
    for s_map in steps_by_video:
        all_video_ids.update(s_map.keys())

    best_sequences_per_video = []

    for v_id in all_video_ids:
        initial_candidates = steps_by_video[0].get(v_id, [])
        if not initial_candidates:
            continue

        beam_sequences = []
        for cand in initial_candidates:
            kf_idx = int(cand.get("keyframe_index", 0))
            score = float(cand.get("score", 0.0))

            beam_sequences.append({
                "video_id": v_id,
                "sequence_path": [cand],
                "cumulative_score": score,
                "last_kf_idx": kf_idx,
                "used_keyframes": {kf_idx},
                "candidate_scores": [score],
                "candidate_ranks": [cand["rank"]],
                "candidate_keyframes": [kf_idx]
            })

        beam_sequences = sorted(beam_sequences, key=lambda x: x["cumulative_score"], reverse=True)[:beam_width]

        is_video_sequence_valid = True
        for step_idx in range(1, num_steps):
            current_video_candidates = steps_by_video[step_idx].get(v_id, [])
            if not current_video_candidates:
                is_video_sequence_valid = False
                break

            candidate_extensions = []
            for seq in beam_sequences:
                last_kf = seq["last_kf_idx"]

                for cand in current_video_candidates:
                    curr_kf = int(cand.get("keyframe_index", 0))
                    kf_diff = curr_kf - last_kf  

                    # Kiểm tra hướng thời gian và khoảng cách keyframe
                    if not (min_kf_gap <= kf_diff <= max_kf_gap):
                        continue

                    # Chống trùng lặp khung hình chính xác
                    if curr_kf in seq["used_keyframes"]:
                        continue

                    cand_score = float(cand.get("score", 0.0))
                    
                    new_seq = {
                        "video_id": v_id,
                        "sequence_path": seq["sequence_path"] + [cand],
                        "cumulative_score": seq["cumulative_score"] + cand_score,
                        "last_kf_idx": curr_kf,
                        "used_keyframes": seq["used_keyframes"] | {curr_kf},
                        "candidate_scores": seq["candidate_scores"] + [cand_score],
                        "candidate_ranks": seq["candidate_ranks"] + [cand["rank"]],
                        "candidate_keyframes": seq["candidate_keyframes"] + [curr_kf]
                    }
                    candidate_extensions.append(new_seq)

            if not candidate_extensions:
                is_video_sequence_valid = False
                break

            beam_sequences = sorted(
                candidate_extensions,
                key=lambda x: x["cumulative_score"],
                reverse=True
            )[:beam_width]

        if is_video_sequence_valid and beam_sequences:
            best_seq_for_video = max(beam_sequences, key=lambda x: x["cumulative_score"])

            sorted_beams = sorted(
                beam_sequences,
                key=lambda x: x["cumulative_score"],
                reverse=True
            )

            if len(sorted_beams) >= 2:
                margin = sorted_beams[0]["cumulative_score"] - sorted_beams[1]["cumulative_score"]
            else:
                margin = sorted_beams[0]["cumulative_score"]

            best_seq_for_video["beam_margin"] = margin
            best_sequences_per_video.append(best_seq_for_video)

    if not best_sequences_per_video:
        print("[INFO] Không tìm thấy chuỗi sự kiện nào thỏa mãn toàn bộ ràng buộc keyframe gap theo từng video.")
        return []

    reranked_sequences = temporal_sequence_reranking(
        temporal_results=best_sequences_per_video,
        config_path=config_path,
        config_name="reranking_temporal.yaml"
    )

    final_temporal_results = []
    for rank_idx, seq in enumerate(reranked_sequences[:100], start=1):
        # Lấy candidate có score cao nhất trong chuỗi làm representative thay vì luôn chọn frame cuối
        best_cand_in_seq = max(seq["sequence_path"], key=lambda x: x.get("score", 0.0))
        
        final_temporal_results.append({
            "rank": seq["rerank_rank"],
            "vector_index": best_cand_in_seq.get("vector_index"),
            "score": seq["rerank_score"],
            "video_id": seq["video_id"],
            "frame_idx": best_cand_in_seq.get("frame_idx"),
            "keyframe_index": best_cand_in_seq.get("keyframe_index"),
            "pts_time": best_cand_in_seq.get("pts_time"),
            "fps": best_cand_in_seq.get("fps"),
            "sequence_path": seq["sequence_path"],
            "candidate_scores": seq["candidate_scores"],
            "candidate_ranks": seq["candidate_ranks"],
            "candidate_keyframes": seq["candidate_keyframes"],
            "beam_margin": seq["beam_margin"],
            "rerank_features": seq["rerank_features"],
            "match_type": "temporal_sequence_multi_model_beam"
        })

    return final_temporal_results


def parse_temporal_query(raw_query_string):
    if not raw_query_string or not isinstance(raw_query_string, str):
        return []
    return [q.strip() for q in raw_query_string.split("/") if q.strip()]


def temporal_retrieval_multi_model_pipeline(
    raw_query_string,
    model_configs,
    config_path,
    max_kf_gap=150,     
    min_kf_gap=1,      
    mapping_path="data/indexes/keyframes_new.db",
    beam_width=5
):
    queries = parse_temporal_query(raw_query_string)
    if not queries:
        print("[WARNING] Truy vấn rỗng hoặc không có định dạng hợp lệ.")
        return []
    
    if not model_configs:
        print("[ERROR] model_configs bị rỗng.")
        return []

    return temporal_sequence_retrieval(
        queries=queries,
        model_configs=model_configs,
        config_path=config_path,
        max_kf_gap=max_kf_gap,
        min_kf_gap=min_kf_gap,
        mapping_path=mapping_path,
        beam_width=beam_width
    )