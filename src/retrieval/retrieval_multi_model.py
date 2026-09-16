
# import numpy as np
# from pathlib import Path
# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline, 
#     dfn5b_vit_h14_retrieval_pipeline,  # Thay thế cho clip_b32_retrieval_pipeline
#     load_config
# )


# import numpy as np
# from pathlib import Path
# from collections import defaultdict
# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline, 
#     dfn5b_vit_h14_retrieval_pipeline,
#     load_config
# )

# def reciprocal_rank_fusion(
#     all_model_results: dict, 
#     rrf_k: int = 60, 
#     model_weights: dict = None
# ) -> list:
#     """
#     Thuật toán Reciprocal Rank Fusion (RRF) giúp gộp kết quả từ nhiều mô hình 
#     dựa trên THỨ HẠNG (Rank) thay vì Score thô.
    
#     Công thức: RRF_Score(candidate) = SUM( weight_m / (rrf_k + rank_m) )
#     """
#     if model_weights is None:
#         model_weights = {}

#     fused_scores = defaultdict(lambda: {
#         "rrf_score": 0.0,
#         "sources": [],
#         "raw_ranks": {},
#         "raw_scores": {}
#     })

#     for m_name, results in all_model_results.items():
#         weight = model_weights.get(m_name, 1.0)
        
#         for rank_idx, item in enumerate(results, start=1):
#             v_idx = item["vector_index"]
#             rrf_score = weight * (1.0 / (rrf_k + rank_idx))
            
#             fused_scores[v_idx]["rrf_score"] += rrf_score
#             fused_scores[v_idx]["sources"].append(m_name)
#             fused_scores[v_idx]["raw_ranks"][m_name] = rank_idx
#             fused_scores[v_idx]["raw_scores"][m_name] = item.get("score", 0.0)

#     fused_list = []
#     for v_idx, data in fused_scores.items():
#         fused_list.append({
#             "vector_index": v_idx,
#             "rrf_score": data["rrf_score"],
#             "sources": data["sources"],
#             "raw_ranks": data["raw_ranks"],
#             "raw_scores": data["raw_scores"]
#         })

#     fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
#     return fused_list


# def retrieval_multi_model_pipeline(
#     query_text: str,
#     model_configs: list, 
#     config_path: str,
#     rrf_k: int = 60,
# ):
#     """
#     Pipeline Retrieval đa mô hình tối ưu cho MRR sử dụng RRF.
#     Sử dụng trực tiếp tập 1000 candidate mặc định từ các pipeline đơn lẻ.
#     """
#     cfg = load_config(config_path)
#     final_top_k = cfg.get("top_k", 100)
#     model_weights = cfg.get("model_weights", {})

#     all_model_results = {}

#     # Bước 1: Truy vấn kết quả gốc từ từng mô hình (giữ nguyên 1000 candidate)
#     for cfg_model in model_configs:
#         m_name = cfg_model["name"]
#         idx_path = cfg_model["index_path"]

#         m_lower = m_name.lower()
#         print(f"[INFO] Running retrieval for model: {m_lower}")
        
#         if m_lower == "siglip2":
#             res = siglip2_retrieval_pipeline(query_text, idx_path, config_path)
#         elif m_lower in ["dfn5b", "vit_h14", "dfn5b_vit_h14", "clip_h14"]:
#             res = dfn5b_vit_h14_retrieval_pipeline(query_text, idx_path, config_path)
#         else:
#             print(f"[WARNING] Model '{m_name}' không được hỗ trợ. Bỏ qua.")
#             continue

#         if res:
#             all_model_results[m_lower] = res

#     if not all_model_results:
#         print("[WARNING] Multi-model retrieval không trả về kết quả nào.")
#         return []

#     # Bước 2: Dung hợp toàn bộ Candidate Pool bằng Reciprocal Rank Fusion (RRF)
#     fused_candidates = reciprocal_rank_fusion(
#         all_model_results=all_model_results,
#         rrf_k=rrf_k,
#         model_weights=model_weights
#     )

#     # Bước 3: Cắt về Top-K cuối cùng (Top 100) cho Stage 2 và chuẩn hóa Score về [0, 1]
#     top_candidates = fused_candidates[:final_top_k]
    
#     if top_candidates:
#         max_rrf = top_candidates[0]["rrf_score"]
#         min_rrf = top_candidates[-1]["rrf_score"]
#         score_range = max_rrf - min_rrf

#         final_results = []
#         for rank_idx, cand in enumerate(top_candidates, start=1):
#             if score_range > 1e-8:
#                 scaled_s = (cand["rrf_score"] - min_rrf) / score_range
#             else:
#                 scaled_s = 1.0

#             final_results.append({
#                 "rank": rank_idx,
#                 "vector_index": cand["vector_index"],
#                 "score": float(scaled_s),
#                 "rrf_score": float(cand["rrf_score"]),
#                 "sources": cand["sources"],
#                 "raw_ranks": cand["raw_ranks"],
#                 "raw_scores": cand["raw_scores"]
#             })
#     else:
#         final_results = []

#     print(f"[SUCCESS] Multi-model RRF fused {len(all_model_results)} models into {len(final_results)} top candidates.")
#     return final_results
















import numpy as np
from pathlib import Path
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline, 
    dfn5b_vit_h14_retrieval_pipeline,  # Thay thế cho clip_b32_retrieval_pipeline
    load_config
)


import numpy as np
from pathlib import Path
from collections import defaultdict
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline, 
    dfn5b_vit_h14_retrieval_pipeline,
    decompose_standard_narrative_query,
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
        "raw_scores": {},
        "extra_info": {}
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
            if "query_plan" in item:
                current_qp = item["query_plan"]
                # Ưu tiên lưu giữ query_plan dạng dict nếu có, hoặc nhận chuỗi nếu chưa có gì
                existing_qp = fused_scores[v_idx]["extra_info"].get("query_plan")
                if isinstance(current_qp, dict) or not existing_qp:
                    fused_scores[v_idx]["extra_info"]["query_plan"] = current_qp

    fused_list = []
    for v_idx, data in fused_scores.items():
        fused_list.append({
            "vector_index": v_idx,
            "rrf_score": data["rrf_score"],
            "sources": data["sources"],
            "raw_ranks": data["raw_ranks"],
            "raw_scores": data["raw_scores"],
            "query_plan": data["extra_info"].get("query_plan", None)
        })

    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_list


def retrieval_multi_model_pipeline(
    query_text: str,
    model_configs: list, 
    config_path: str,
    rrf_k: int = 60,
    query_plan: dict = None,
):
    """
    Pipeline Retrieval đa mô hình tối ưu cho MRR sử dụng RRF.
    Sử dụng trực tiếp tập 1000 candidate mặc định từ các pipeline đơn lẻ.
    """
    cfg = load_config(config_path)
    final_top_k = cfg.get("top_k", 100)
    model_weights = cfg.get("model_weights", {})

    # Tối ưu hóa: Phân rã query 1 lần duy nhất cho toàn bộ các mô hình con
    if query_plan is None:
        query_plan = decompose_standard_narrative_query(query_text)

    all_model_results = {}

    # Bước 1: Truy vấn kết quả gốc từ từng mô hình (giữ nguyên 1000 candidate)
    for cfg_model in model_configs:
        m_name = cfg_model["name"]
        idx_path = cfg_model["index_path"]

        m_lower = m_name.lower()
        print(f"[INFO] Running retrieval for model: {m_lower}")
        
        if m_lower == "siglip2":
            res = siglip2_retrieval_pipeline(query_text, idx_path, config_path, query_plan=query_plan)
        elif m_lower in ["dfn5b", "vit_h14", "dfn5b_vit_h14", "clip_h14"]:
            res = dfn5b_vit_h14_retrieval_pipeline(query_text, idx_path, config_path, query_plan=query_plan)
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
                "raw_scores": cand["raw_scores"],
                "query_plan": cand.get("query_plan", None)
            })
    else:
        final_results = []

    print(f"[SUCCESS] Multi-model RRF fused {len(all_model_results)} models into {len(final_results)} top candidates.")
    return final_results













