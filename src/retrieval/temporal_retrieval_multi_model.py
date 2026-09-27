import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
from src.mapping.mapping_pipeline import mapping_pipeline
from src.reranking.temporal_reranking import (
    temporal_sequence_reranking
)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
from src.retrieval.retrieval_pipeline import load_config


def temporal_nms(candidates, window=10):

    if not candidates:
        return []

    video_groups = defaultdict(list)

    for cand in candidates:
        video_groups[cand["video_id"]].append(cand)

    filtered = []

    for video_id, frames in video_groups.items():

        # score cao nhất trước
        frames = sorted(
            frames,
            key=lambda x: x["score"],
            reverse=True
        )

        kept = []

        for frame in frames:

            keep = True

            for exist in kept:

                if abs(
                    frame["keyframe_index"]
                    - exist["keyframe_index"]
                ) <= window:

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
    max_kf_gap=150,     # Khoảng cách tối đa giữa 2 keyframe liên tiếp
    min_kf_gap=1,       # Khoảng cách tối thiểu
    mapping_path="data/indexes/keyframes_b1_b2.db",
    beam_width=5,        # Độ rộng chùm tìm kiếm (Beam Width) cho mỗi video
    temporal_nms_window=5,
    top_per_video=30,
    topic_filter=None,
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
            config_path=config_path,
            topic_filter=topic_filter,
        )
        
        if not res:
            step_results.append([])
            continue
            
        enriched_res = mapping_pipeline(
            retrieval_results=res,
            mapping_db_path=mapping_path
        )
        enriched_res = temporal_nms(
            enriched_res,
            window=temporal_nms_window
        )

        video_bucket = defaultdict(list)
        for cand in enriched_res:
            video_bucket[cand["video_id"]].append(cand)

        filtered_candidates = []
        for frames in video_bucket.values():
            frames = sorted(
                frames,
                key=lambda x: x["score"],
                reverse=True
            )[:top_per_video]
            filtered_candidates.extend(frames)

        enriched_res = filtered_candidates
        for r_idx, item in enumerate(enriched_res, start=1):
            item["rank"] = r_idx
        step_results.append(enriched_res)

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

    initial_video_ids = list(steps_by_video[0].keys())
    best_sequences_per_video = []

    for v_id in initial_video_ids:
        initial_candidates = steps_by_video[0][v_id]
        if not initial_candidates:
            continue

        beam_sequences = []
        for cand in initial_candidates:
            kf_idx = int(cand.get("keyframe_index", 0))
            score = float(cand.get("score", 0.0))

            beam_sequences.append({
                "video_id": v_id,

                # Beam
                "sequence_path": [cand],
                "cumulative_score": score,
                "last_kf_idx": kf_idx,
                "used_keyframes": {kf_idx},

                # Cho Temporal Re-ranking
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

                    # 1. Kiểm tra khoảng cách tối thiểu/tối đa và hướng thời gian (tránh lùi thời gian hoặc trùng lặp)
                    if not (min_kf_gap <= kf_diff <= max_kf_gap):
                        continue

                    # 2. Kiểm tra chống trùng lặp keyframe với các bước trước đó trong cùng chuỗi
                    too_close = any(
                        abs(curr_kf - prev_kf) <= temporal_nms_window
                        for prev_kf in seq["used_keyframes"]
                    )

                    if too_close:
                        continue

                    cand_score = float(cand.get("score", 0.0))
                    
                    new_seq = {
                        "video_id": v_id,
                        "sequence_path": seq["sequence_path"] + [cand],
                        "cumulative_score": seq["cumulative_score"] + cand_score,
                        "last_kf_idx": curr_kf,
                        "used_keyframes": seq["used_keyframes"] | {curr_kf},
                        "candidate_scores":
                            seq["candidate_scores"] + [cand_score],

                        "candidate_ranks":
                            seq["candidate_ranks"] + [cand["rank"]],

                        "candidate_keyframes":
                            seq["candidate_keyframes"] + [curr_kf]
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
            ################################################
            # Beam Margin
            ################################################

            sorted_beams = sorted(
                beam_sequences,
                key=lambda x:x["cumulative_score"],
                reverse=True
            )

            if len(sorted_beams) >= 2:

                margin = (
                    sorted_beams[0]["cumulative_score"]
                    -
                    sorted_beams[1]["cumulative_score"]
                )

            else:

                margin = sorted_beams[0]["cumulative_score"]

            best_seq_for_video["beam_margin"] = margin

            best_sequences_per_video.append(best_seq_for_video)
    if not best_sequences_per_video:
        print("[INFO] Không tìm thấy chuỗi sự kiện nào thỏa mãn toàn bộ ràng buộc keyframe gap theo từng video.")
        return []

    # sorted_sequences = sorted(
    #     best_sequences_per_video,
    #     key=lambda x: x["cumulative_score"] / num_steps,
    #     reverse=True
    # )

    reranked_sequences = temporal_sequence_reranking(
        temporal_results=best_sequences_per_video,
        config_path=config_path,
        config_name="reranking_temporal.yaml"
    )

    final_temporal_results = []
    for rank_idx, seq in enumerate(reranked_sequences[:100], start=1):
        representative_cand = seq["sequence_path"][-1] 
        
        final_temporal_results.append({
            "rank": seq["rerank_rank"],
            "vector_index": representative_cand.get("vector_index"),
            # "score": float(seq["cumulative_score"] / num_steps),
            "score": seq["rerank_score"],
            "video_id": seq["video_id"],
            "frame_idx": representative_cand.get("frame_idx"),
            "keyframe_index": representative_cand.get("keyframe_index"),
            "pts_time": representative_cand.get("pts_time"),
            "fps": representative_cand.get("fps"),
            "sequence_path": seq["sequence_path"],
            "candidate_scores":
                seq["candidate_scores"],

            "candidate_ranks":
                seq["candidate_ranks"],

            "candidate_keyframes":
                seq["candidate_keyframes"],

            "beam_margin":
                seq["beam_margin"],
            "rerank_features": seq["rerank_features"],
            "match_type": "temporal_sequence_multi_model_beam"
        })

    return final_temporal_results

def parse_temporal_query(raw_query_string):
    """
    Nhận vào chuỗi truy vấn thô từ người dùng, cắt theo dấu '/' 
    và làm sạch khoảng trắng ở mỗi phần.
    """
    if not raw_query_string or not isinstance(raw_query_string, str):
        return []
    
    queries = [q.strip() for q in raw_query_string.split("/") if q.strip()]
    return queries


def temporal_retrieval_multi_model_pipeline(
    raw_query_string,
    model_configs,
    config_path,
    max_kf_gap=150,     
    min_kf_gap=0,      
    mapping_path="data/indexes/keyframes_b1_b2.db",
    beam_width=5,
    topic_filter=None,
):
    """
    Hàm pipeline hoàn chỉnh: Nhận chuỗi thô có chứa dấu '/' -> Phân rã query -> Chạy Multi-Model Ensemble kết hợp Beam Search theo từng video_id.
    """
    queries = parse_temporal_query(raw_query_string)
    
    if not queries:
        print("[WARNING] Truy vấn rỗng hoặc không có định dạng hợp lệ.")
        return []
    
    print(f"[INFO] Phân rã truy vấn theo thời gian thành {len(queries)} bước (Multi-Model Ensemble & Beam Search Mode):")
    for idx, q in enumerate(queries, 1):
        print(f"  - Bước {idx}: {q}")

    if not model_configs:
        print("[ERROR] model_configs bị rỗng.")
        return []

    # Gọi hàm xử lý chuỗi tuần tự với toàn bộ danh sách cấu hình mô hình (Multi-Model)
    return temporal_sequence_retrieval(
        queries=queries,
        model_configs=model_configs,
        config_path=config_path,
        max_kf_gap=max_kf_gap,
        min_kf_gap=min_kf_gap,
        mapping_path=mapping_path,
        beam_width=beam_width,
        topic_filter=topic_filter,
    )



# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline,
#     clip_b32_retrieval_pipeline,
#     load_config,
# )

# def temporal_zscore(results):
#     """
#     Chuẩn hóa score của một model.
#     Không dùng StandardScaler sklearn để tránh dependency.
#     """

#     if len(results) == 0:
#         return results

#     scores = np.array(
#         [x["score"] for x in results],
#         dtype=np.float32
#     )

#     mean = scores.mean()
#     std = scores.std()

#     for r in results:

#         if std > 1e-8:
#             r["score_norm"] = (r["score"] - mean) / std
#         else:
#             r["score_norm"] = 0.0

#     return results

# def run_model_retrieval(
#     query_text,
#     model_cfg,
#     config_path,
# ):
#     """
#     Chạy retrieval cho đúng model.
#     """

#     model_name = model_cfg["name"].lower()

#     index_path = model_cfg["index_path"]

#     if model_name == "siglip2":

#         results = siglip2_retrieval_pipeline(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#         )

#     elif model_name in [
#         "clip",
#         "clip_b32",
#         "vit_b32",
#     ]:

#         results = clip_b32_retrieval_pipeline(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#         )

#     else:

#         raise ValueError(
#             f"Unsupported model : {model_name}"
#         )

#     return temporal_zscore(results)


# def temporal_multi_model_fusion(
#     query_text,
#     model_configs,
#     config_path,
#     fusion_topk=400,
# ):
#     """
#     Fusion dành riêng cho Temporal.

#     Không boost.

#     Không penalty.

#     Không consensus.

#     Chỉ cộng score.
#     """

#     merged = {}

#     for cfg in model_configs:

#         results = run_model_retrieval(
#             query_text,
#             cfg,
#             config_path,
#         )

#         model_name = cfg["name"]

#         for item in results:

#             vid = item["vector_index"]

#             if vid not in merged:

#                 merged[vid] = {

#                     "vector_index": vid,

#                     "score": item["score_norm"],

#                     "sources": [model_name]

#                 }

#             else:

#                 merged[vid]["score"] += item["score_norm"]

#                 merged[vid]["sources"].append(model_name)

#     merged = list(merged.values())

#     merged.sort(

#         key=lambda x: x["score"],

#         reverse=True

#     )

#     merged = merged[:fusion_topk]

#     for rank, item in enumerate(merged, 1):

#         item["rank"] = rank

#     return merged 