
import numpy as np
from pathlib import Path
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline, 
    dfn5b_vit_h14_retrieval_pipeline,  # Thay thế cho clip_b32_retrieval_pipeline
    load_config
)

# def z_score_normalization(results):
#     """
#     Chuẩn hóa điểm số (score) của danh sách kết quả sử dụng Z-score (StandardScaler):
#     score_norm = (score - mean) / std
#     Nếu danh sách có độ lệch chuẩn bằng 0, gán score_norm = 0.
#     """
#     if not results:
#         return results

#     scores = [r["score"] for r in results]
#     mean = np.mean(scores)
#     std = np.std(scores)

#     normalized_results = []
#     for r in results:
#         new_r = r.copy()
#         if std > 1e-8:
#             new_r["normalized_score"] = (r["score"] - mean) / std
#         else:
#             new_r["normalized_score"] = float(r["score"])
#         normalized_results.append(new_r)
        
#     return normalized_results

# def retrieval_multi_model_pipeline(
#     query_text,
#     model_configs, 
#     config_path
# ):
#     """
#     model_configs: Danh sách cấu hình các model muốn chạy song song.
#     Ví dụ trong file yaml hoặc khởi tạo:
#     [
#         {"name": "siglip2", "index_path": "path/to/siglib2.index"},
#         {"name": "dfn5b_vit_h14", "index_path": "path/to/dfn5b_clip_vit_h14.index"}
#     ]
#     """
#     all_raw_results = []

#     # Bước 1: Lấy kết quả Top-K độc lập từ từng Embedding Model
#     for cfg_model in model_configs:
#         m_name = cfg_model["name"]
#         idx_path = cfg_model["index_path"]

#         print(f"[INFO] Running retrieval for model: {m_name}")
        
#         m_lower = m_name.lower()
#         if m_lower == "siglip2":
#             res = siglip2_retrieval_pipeline(query_text, idx_path, config_path)
#         elif m_lower in ["dfn5b", "vit_h14", "dfn5b_vit_h14", "clip_h14"]:
#             res = dfn5b_vit_h14_retrieval_pipeline(query_text, idx_path, config_path)
#         else:
#             print(f"[WARNING] Model '{m_name}' không được hỗ trợ. Bỏ qua.")
#             continue

#         # Chuẩn hóa Z-score cho từng mô hình trước khi gộp
#         norm_res = z_score_normalization(res)
#         for r in norm_res:
#             r["model_source"] = m_lower
#         all_raw_results.extend(norm_res)

#     if not all_raw_results:
#         print("[WARNING] Multi-model retrieval không trả về kết quả nào.")
#         return []

#     cfg = load_config(config_path)
#     final_top_k = cfg.get("top_k", 100)
    
#     # Lấy trọng số cấu hình từ file yaml (mặc định 1.0 nếu không khai báo)
#     model_weights = cfg.get("model_weights", {})
#     total_models_configured = len(model_configs)

#     # Bước 2: Gộp kết quả (Union) và xử lý trùng lặp dựa trên vector_index (Áp dụng trọng số & MEQS Fusion)
#     merged_dict = {}
#     for r in all_raw_results:
#         v_idx = r["vector_index"]
#         score_val = r["normalized_score"]
#         src = r["model_source"]

#         # Lấy trọng số của model từ file yaml
#         w = model_weights.get(src, 1.0)
#         weighted_score = w * score_val

#         if v_idx not in merged_dict:
#             merged_dict[v_idx] = {
#                 "vector_index": v_idx,
#                 "final_score": weighted_score,
#                 "sources": [src],
#                 "raw_scores": {src: r["score"]}
#             }
#         else:
#             # Cộng dồn điểm có trọng số từ các mô hình khác nhau
#             merged_dict[v_idx]["final_score"] += weighted_score
#             if src not in merged_dict[v_idx]["sources"]:
#                 merged_dict[v_idx]["sources"].append(src)
#             merged_dict[v_idx]["raw_scores"][src] = r["score"]

#     # Bước 2.1: Xử lý hệ số đồng thuận (Consensus Boosting)
#     for v_idx, cand in merged_dict.items():
#         found_count = len(cand["sources"])
#         if found_count >= total_models_configured and total_models_configured > 1:
#             # Thưởng nhẹ điểm nếu được tất cả các mô hình đồng thuận tìm thấy
#             cand["final_score"] *= 1.10

#     # Bước 3: Sắp xếp lại theo tổng điểm giảm dần và lấy Top-K
#     sorted_candidates = sorted(
#         merged_dict.values(),
#         key=lambda x: x["final_score"],
#         reverse=True
#     )[:final_top_k]

#     # Bước 4: Co giãn điểm số cuối cùng (Score Re-scaling về khoảng [0, 1]) theo chuẩn MEQS
#     if sorted_candidates:
#         scores_list = [c["final_score"] for c in sorted_candidates]
#         min_s = min(scores_list)
#         max_s = max(scores_list)
#         score_range = max_s - min_s

#         for cand in sorted_candidates:
#             if score_range > 1e-8:
#                 cand["scaled_score"] = (cand["final_score"] - min_s) / score_range
#             else:
#                 cand["scaled_score"] = 1.0
#     else:
#         for cand in sorted_candidates:
#             cand["scaled_score"] = float(cand["final_score"])

#     # Gắn lại rank chuẩn và định dạng kết quả đầu ra
#     final_results = []
#     for rank_idx, cand in enumerate(sorted_candidates, start=1):
#         final_results.append({
#             "rank": rank_idx,
#             "vector_index": cand["vector_index"],
#             "score": float(cand["scaled_score"]),      # Điểm số chuẩn hóa trực quan [0, 1]
#             "raw_fusion_score": float(cand["final_score"]), # Điểm gộp thô để debug nếu cần
#             "sources": cand["sources"],
#             "raw_scores": cand["raw_scores"]
#         })

#     print(f"[SUCCESS] Multi-model retrieval fused into {len(final_results)} top candidates.")
#     return final_results
import numpy as np
from pathlib import Path
from collections import defaultdict
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline, 
    dfn5b_vit_h14_retrieval_pipeline,
    load_config
)

def reciprocal_rank_fusion(
    all_model_results: dict, 
    rrf_k: int = 60, 
    model_weights: dict = None
) -> list:
    """
    Thuật toán Reciprocal Rank Fusion (RRF) giúp gộp kết quả từ nhiều mô hình 
    dựa trên THỨ HẠNG (Rank) thay vì Score thô.
    
    Công thức: RRF_Score(candidate) = SUM( weight_m / (rrf_k + rank_m) )
    """
    if model_weights is None:
        model_weights = {}

    fused_scores = defaultdict(lambda: {
        "rrf_score": 0.0,
        "sources": [],
        "raw_ranks": {},
        "raw_scores": {}
    })

    for m_name, results in all_model_results.items():
        weight = model_weights.get(m_name, 1.0)
        
        for rank_idx, item in enumerate(results, start=1):
            v_idx = item["vector_index"]
            rrf_score = weight * (1.0 / (rrf_k + rank_idx))
            
            fused_scores[v_idx]["rrf_score"] += rrf_score
            fused_scores[v_idx]["sources"].append(m_name)
            fused_scores[v_idx]["raw_ranks"][m_name] = rank_idx
            fused_scores[v_idx]["raw_scores"][m_name] = item.get("score", 0.0)

    fused_list = []
    for v_idx, data in fused_scores.items():
        fused_list.append({
            "vector_index": v_idx,
            "rrf_score": data["rrf_score"],
            "sources": data["sources"],
            "raw_ranks": data["raw_ranks"],
            "raw_scores": data["raw_scores"]
        })

    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_list


def retrieval_multi_model_pipeline(
    query_text: str,
    model_configs: list, 
    config_path: str,
    rrf_k: int = 60
):
    """
    Pipeline Retrieval đa mô hình tối ưu cho MRR sử dụng RRF.
    Sử dụng trực tiếp tập 1000 candidate mặc định từ các pipeline đơn lẻ.
    """
    cfg = load_config(config_path)
    final_top_k = cfg.get("top_k", 100)
    model_weights = cfg.get("model_weights", {})

    all_model_results = {}

    # Bước 1: Truy vấn kết quả gốc từ từng mô hình (giữ nguyên 1000 candidate)
    for cfg_model in model_configs:
        m_name = cfg_model["name"]
        idx_path = cfg_model["index_path"]

        m_lower = m_name.lower()
        print(f"[INFO] Running retrieval for model: {m_lower}")
        
        if m_lower == "siglip2":
            res = siglip2_retrieval_pipeline(query_text, idx_path, config_path)
        elif m_lower in ["dfn5b", "vit_h14", "dfn5b_vit_h14", "clip_h14"]:
            res = dfn5b_vit_h14_retrieval_pipeline(query_text, idx_path, config_path)
        else:
            print(f"[WARNING] Model '{m_name}' không được hỗ trợ. Bỏ qua.")
            continue

        if res:
            all_model_results[m_lower] = res

    if not all_model_results:
        print("[WARNING] Multi-model retrieval không trả về kết quả nào.")
        return []

    # Bước 2: Dung hợp toàn bộ Candidate Pool bằng Reciprocal Rank Fusion (RRF)
    fused_candidates = reciprocal_rank_fusion(
        all_model_results=all_model_results,
        rrf_k=rrf_k,
        model_weights=model_weights
    )

    # Bước 3: Cắt về Top-K cuối cùng (Top 100) cho Stage 2 và chuẩn hóa Score về [0, 1]
    top_candidates = fused_candidates[:final_top_k]
    
    if top_candidates:
        max_rrf = top_candidates[0]["rrf_score"]
        min_rrf = top_candidates[-1]["rrf_score"]
        score_range = max_rrf - min_rrf

        final_results = []
        for rank_idx, cand in enumerate(top_candidates, start=1):
            if score_range > 1e-8:
                scaled_s = (cand["rrf_score"] - min_rrf) / score_range
            else:
                scaled_s = 1.0

            final_results.append({
                "rank": rank_idx,
                "vector_index": cand["vector_index"],
                "score": float(scaled_s),
                "rrf_score": float(cand["rrf_score"]),
                "sources": cand["sources"],
                "raw_ranks": cand["raw_ranks"],
                "raw_scores": cand["raw_scores"]
            })
    else:
        final_results = []

    print(f"[SUCCESS] Multi-model RRF fused {len(all_model_results)} models into {len(final_results)} top candidates.")
    return final_results