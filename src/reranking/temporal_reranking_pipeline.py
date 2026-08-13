

# from src.reranking.reranking_pipeline import reranking_pipeline, load_rerank_config

# def temporal_sequence_reranking_optimized(raw_query_string, sequence_candidates, config_path):
#     """
#     Rerank chuỗi temporal, tận dụng triệt để reranking_pipeline cùng OCR, ASR, Metadata, Object có sẵn.
#     """
#     if not sequence_candidates:
#         return []

#     queries = [q.strip() for q in raw_query_string.split("/") if q.strip()]
#     num_queries = len(queries)

#     for seq_cand in sequence_candidates:
#         seq_path = seq_cand.get("sequence_path", [])
#         if not seq_path:
#             seq_cand["final_score"] = 0.0
#             continue

#         total_final_score = 0.0
#         num_steps = len(seq_path)

#         for step_idx, step_frame in enumerate(seq_path):
#             # Lấy đúng query con tương ứng với bước thời gian
#             sub_query = queries[step_idx] if step_idx < num_queries else queries[-1]

#             # Đóng gói keyframe hiện tại, tận dụng toàn bộ thông tin OCR, ASR, Metadata, Object đã map từ trước
#             single_frame_candidate = [{
#                 "vector_index": step_frame.get("vector_index"),
#                 "video_id": step_frame.get("video_id"),
#                 "keyframe_index": step_frame.get("keyframe_index", 0),
#                 "frame_idx": step_frame.get("frame_idx", 0),
#                 "pts_time": step_frame.get("pts_time", 0.0),
#                 "score": step_frame.get("score", 1.0),
#                 # Tận dụng các trường đa modal sẵn có trong mapping record:
#                 "object_entities": step_frame.get("object_entities", []),
#                 "ocr_texts": step_frame.get("ocr_texts", []),
#                 "asr_text": step_frame.get("asr_text", ""),
#                 "metadata": step_frame.get("metadata", {})
#             }]

#             # Gọi pipeline rerank chuẩn (hàm này sẽ tự động phân tích OCR, ASR, Object,...)
#             reranked_single = reranking_pipeline(sub_query, single_frame_candidate, config_path)

#             if reranked_single:
#                 step_res = reranked_single[0]
#                 # Lấy final_score được chấm từ rerank pipeline
#                 total_final_score += step_res.get("final_score", step_res.get("score", 0.0))
#             else:
#                 total_final_score += step_frame.get("score", 0.0)

#         # Lấy trung bình cộng điểm final_score qua các bước
#         seq_cand["final_score"] = round(total_final_score / num_steps, 6)

#     # Sắp xếp lại theo điểm final_score giảm dần
#     sequence_candidates = sorted(
#         sequence_candidates,
#         key=lambda x: x["final_score"],
#         reverse=True
#     )

#     # Gắn lại rank
#     for rank_idx, c in enumerate(sequence_candidates, start=1):
#         c["rank"] = rank_idx

#     print(f"[INFO] Multi-modal temporal sequence reranking completed for {len(sequence_candidates)} sequences.")
#     return sequence_candidates

from src.reranking.reranking_pipeline import reranking_pipeline, load_rerank_config

def temporal_sequence_reranking_optimized(raw_query_string, sequence_candidates, config_path):
    """
    Rerank chuỗi temporal tối ưu:
    1. Tách query con theo dấu '/'.
    2. Gom tất cả ứng viên (candidates) độc lập của từng bước thời gian để chạy qua 
       reranking_pipeline chuẩn, giúp RRF có tập dữ liệu lớn cạnh tranh và chấm điểm chuẩn xác.
    3. Cập nhật lại điểm số đa modal cho từng frame trong chuỗi sequence.
    4. Tổng hợp điểm chuỗi, sắp xếp và gán lại rank.
    """
    if not sequence_candidates:
        return []

    # 1. Tách các query con từ chuỗi thô
    queries = [q.strip() for q in raw_query_string.split("/") if q.strip()]
    num_queries = len(queries)

    # 2. Thu thập tất cả các ứng viên độc lập theo từng bước (step_idx) từ mọi sequence
    # Để tránh việc gọi reranking_pipeline lặp đi lặp lại cho cùng một khung hình ở các sequence khác nhau,
    # ta có thể gom nhóm hoặc chấm điểm từng bước một cách độc lập.
    
    # Xác định số bước tối đa dựa trên độ dài của sequence path
    max_steps = max(len(seq_cand.get("sequence_path", [])) for seq_cand in sequence_candidates)
    
    # Dictionary lưu trữ điểm số mới cho từng frame theo (step_idx, frame_identifier)
    # frame_identifier có thể là (video_id, keyframe_index) hoặc vector_index
    step_frame_scored_cache = {}

    for step_idx in range(max_steps):
        sub_query = queries[step_idx] if step_idx < num_queries else queries[-1]
        
        # Gom tất cả các khung hình xuất hiện ở bước `step_idx` từ mọi sequence ứng viên
        step_candidates_pool = []
        identifier_to_frame_map = {}

        for seq_cand in sequence_candidates:
            seq_path = seq_cand.get("sequence_path", [])
            if step_idx < len(seq_path):
                step_frame = seq_path[step_idx]
                
                # Tạo một định danh duy nhất cho khung hình này
                frame_id = (
                    step_frame.get("video_id"),
                    step_frame.get("keyframe_index"),
                    step_frame.get("vector_index")
                )
                
                if frame_id not in identifier_to_frame_map:
                    identifier_to_frame_map[frame_id] = step_frame
                    step_candidates_pool.append({
                        "vector_index": step_frame.get("vector_index"),
                        "video_id": step_frame.get("video_id"),
                        "keyframe_index": step_frame.get("keyframe_index", 0),
                        "frame_idx": step_frame.get("frame_idx", 0),
                        "pts_time": step_frame.get("pts_time", 0.0),
                        "score": step_frame.get("score", 1.0),
                        "object_entities": step_frame.get("object_entities", []),
                        "ocr_texts": step_frame.get("ocr_texts", []),
                        "asr_text": step_frame.get("asr_text", ""),
                        "metadata": step_frame.get("metadata", {})
                    })

        # Nếu có ứng viên cho bước này, tiến hành gọi reranking_pipeline (Nơi RRF phát huy tác dụng thực sự!)
        if step_candidates_pool:
            reranked_pool = reranking_pipeline(sub_query, step_candidates_pool, config_path)
            
            # Lưu lại điểm final_score đã qua rerank vào cache
            for res in reranked_pool:
                f_id = (
                    res.get("video_id"),
                    res.get("keyframe_index"),
                    res.get("vector_index")
                )
                # Lấy final_score do reranking_pipeline (RRF) tính toán
                best_score = res.get("final_score", res.get("score", 0.0))
                step_frame_scored_cache[(step_idx, f_id)] = best_score

    # 3. Tính lại điểm tổng cho từng sequence dựa trên điểm đa modal đã được rerank chuẩn
    for seq_cand in sequence_candidates:
        seq_path = seq_cand.get("sequence_path", [])
        if not seq_path:
            seq_cand["final_score"] = 0.0
            continue

        total_final_score = 0.0
        num_steps = len(seq_path)

        for step_idx, step_frame in enumerate(seq_path):
            f_id = (
                step_frame.get("video_id"),
                step_frame.get("keyframe_index"),
                step_frame.get("vector_index")
            )
            
            # Lấy điểm đã được rerank từ cache, nếu không có thì lấy điểm gốc của frame
            cached_score = step_frame_scored_cache.get(
                (step_idx, f_id), 
                step_frame.get("score", 0.0)
            )
            
            # Cập nhật lại điểm cho chính step_frame đó nếu muốn lưu vết
            step_frame["rerank_score"] = cached_score
            total_final_score += cached_score

        # Lấy trung bình cộng điểm final_score qua các bước trong chuỗi
        seq_cand["final_score"] = round(total_final_score / num_steps, 6)

    # 4. Sắp xếp lại toàn bộ sequence_candidates theo điểm final_score giảm dần
    sequence_candidates = sorted(
        sequence_candidates,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # 5. Gắn lại thứ hạng (rank)
    for rank_idx, c in enumerate(sequence_candidates, start=1):
        c["rank"] = rank_idx

    print(f"[INFO] Multi-modal temporal sequence reranking optimized completed for {len(sequence_candidates)} sequences.")
    return sequence_candidates