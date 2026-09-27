import sys
import json
import time
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.mapping.mapping_pipeline import mapping_pipeline
from src.reranking.reranking_pipeline import reranking_pipeline
from src.retrieval.retrieval_multi_model import retrieval_multi_model_pipeline
from src.retrieval.retrieval_pipeline import (
    siglip2_retrieval_pipeline,
    clear_translation_cache
)
from src.retrieval.temporal_nms import temporal_nms

MAPPING_PATH = str(ROOT_DIR / "data" / "indexes" / "keyframes_b1_b2.db")
RETRIEVAL_CONFIG = str(ROOT_DIR / "configs" / "retrieval.yaml")
RERANK_CONFIG = str(ROOT_DIR / "configs" / "reranking.yaml")


def normalize_video_id(v_id: str) -> str:
    """
    Chuẩn hóa chuỗi Video ID để tránh lỗi so sánh lệch định dạng (vd: L01_V001.mp4 vs L01_V001).
    """
    if not v_id:
        return ""
    v_str = str(v_id).strip().lower()
    if v_str.endswith(".mp4"):
        v_str = v_str[:-4]
    return v_str


def extract_video_ranks(candidate_list, score_key=None):
    """
    Trích xuất thứ tự Video ID duy nhất. 
    Đảm bảo danh sách được sort theo điểm số giảm dần trước khi lấy rank.
    """
    if not candidate_list:
        return []

    # Nếu có chỉ định score_key, đảm bảo sort lại danh sách candidate
    if score_key:
        sorted_cands = sorted(candidate_list, key=lambda x: x.get(score_key, 0.0), reverse=True)
    else:
        sorted_cands = candidate_list

    seen_videos = []
    for cand in sorted_cands:
        raw_vid = cand.get("video_id")
        norm_vid = normalize_video_id(raw_vid)
        if norm_vid and norm_vid not in seen_videos:
            seen_videos.append(norm_vid)
    return seen_videos


def compute_metrics(ground_truth, stage_results, top_k_eval=100, score_key=None):
    """
    Tính toán các chỉ số MRR, Recall@K và Mean Rank.
    """
    mrr_total = 0.0
    r1, r5, r10, r50 = 0, 0, 0, 0
    found_ranks = []
    query_details = {}

    total_queries = len(ground_truth)

    for item in ground_truth:
        q_id = item["query_id"]
        target_vid = normalize_video_id(item["video_id"])
        
        candidates = stage_results.get(q_id, [])
        ranked_videos = extract_video_ranks(candidates, score_key=score_key)[:top_k_eval]

        try:
            rank = ranked_videos.index(target_vid) + 1
        except ValueError:
            rank = None

        if rank is not None:
            rr = 1.0 / rank
            mrr_total += rr
            found_ranks.append(rank)

            if rank == 1: r1 += 1
            if rank <= 5: r5 += 1
            if rank <= 10: r10 += 1
            if rank <= 50: r50 += 1
        else:
            rr = 0.0

        query_details[q_id] = {
            "target_video_id": item["video_id"],
            "rank": rank if rank is not None else f">{top_k_eval}",
            "reciprocal_rank": round(rr, 4)
        }

    avg_mrr = mrr_total / total_queries if total_queries > 0 else 0.0
    mean_rank = sum(found_ranks) / len(found_ranks) if found_ranks else -1.0

    summary = {
        "Total Queries": total_queries,
        "MRR": round(avg_mrr, 4),
        "Recall@1": f"{r1}/{total_queries} ({r1/total_queries:.1%})",
        "Recall@5": f"{r5}/{total_queries} ({r5/total_queries:.1%})",
        "Recall@10": f"{r10}/{total_queries} ({r10/total_queries:.1%})",
        "Recall@50": f"{r50}/{total_queries} ({r50/total_queries:.1%})",
        "Mean Rank (Found)": round(mean_rank, 2)
    }

    return summary, query_details


def run_benchmark(gt_file_path, retrieval_mode="multi"):
    gt_path = Path(gt_file_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Ground Truth tại: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    selected_models = [
        {"name": "siglip2",       "index_path": str(ROOT_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index")},
        {"name": "dfn5b_vit_h14", "index_path": str(ROOT_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14_keyframes_b1_b2.index")}
    ]

    stage1_results = {}
    stage2_results = {}
    execution_times = []

    print("=" * 80)
    print(f"[INFO] BẮT ĐẦU CHẠY BENCHMARK PIPELINE ({len(ground_truth)} QUERIES)")
    print("=" * 80)

    for item in ground_truth:
        q_id = item["query_id"]
        q_text = item["query"]
        target_vid = item["video_id"]

        # Ép gọi mới lại Gemini Translate cho từng query
        clear_translation_cache()

        print(f"\n[QUERY {q_id}] '{q_text[:60]}...' | Ground Truth: {target_vid}")
        start_t = time.time()

        # STAGE 1: RETRIEVAL & MAPPING
        if retrieval_mode == "multi":
            raw_retrieval = retrieval_multi_model_pipeline(
                query_text=q_text,
                model_configs=selected_models,
                config_path=RETRIEVAL_CONFIG
            )
        else:
            siglip2_idx = str(ROOT_DIR / "data" / "indexes" / "siglip2_keyframes_b1_b2.index")
            raw_retrieval = siglip2_retrieval_pipeline(q_text, siglip2_idx, RETRIEVAL_CONFIG)

        mapped_candidates = mapping_pipeline(
            retrieval_results=raw_retrieval,
            mapping_db_path=MAPPING_PATH
        )
        
        # Áp dụng Temporal NMS (Đảm bảo gọi hàm NMS v2 nếu đã nâng cấp)
        mapped_candidates = temporal_nms(
            mapped_candidates,
            min_gap_sec=0.3,
            max_per_video=5
        )
          
        stage1_results[q_id] = mapped_candidates

        # STAGE 2: RERANKING
        reranked_candidates = reranking_pipeline(
            query_text=q_text,
            candidate_list=mapped_candidates,
            config_path=RERANK_CONFIG
        )
        stage2_results[q_id] = reranked_candidates

        elapsed = time.time() - start_t
        execution_times.append(elapsed)
        print(f"[QUERY {q_id}] Khởi chạy hoàn tất trong {elapsed:.2f}s")

    # TÍNH METRICS: Stage 1 dùng boosted_score/score, Stage 2 dùng final_score/rerank_score
    s1_metrics, s1_details = compute_metrics(ground_truth, stage1_results, score_key="boosted_score")
    s2_metrics, s2_details = compute_metrics(ground_truth, stage2_results, score_key="final_score")

    print("\n" + "=" * 75)
    print(f"{'BẢNG SO SÁNH HIỆU NĂNG PHÂN TẦNG PIPELINE':^75}")
    print("=" * 75)
    print(f"{'METRIC':<20} | {'STAGE 1 (RETRIEVAL)':<22} | {'STAGE 2 (RERANKING)':<22}")
    print("-" * 75)
    for k in s1_metrics.keys():
        print(f"{k:<20} | {str(s1_metrics[k]):<22} | {str(s2_metrics[k]):<22}")
    print("=" * 75)

    print("\nCHI TIẾT BIẾN ĐỘNG THỨ HẠNG (RANK VARIATION PER QUERY):")
    print("-" * 75)
    print(f"{'Q_ID':<5} | {'TARGET':<12} | {'STAGE 1 RANK':<14} | {'STAGE 2 RANK':<14} | {'TRẠNG THÁI':<15}")
    print("-" * 75)

    for item in ground_truth:
        q_id = item["query_id"]
        target_vid = item["video_id"]
        r1_rank = s1_details[q_id]["rank"]
        r2_rank = s2_details[q_id]["rank"]

        status = "➖ UNCHANGED"
        if isinstance(r1_rank, int) and isinstance(r2_rank, int):
            if r2_rank < r1_rank:
                status = "🟢 IMPROVED"
            elif r2_rank > r1_rank:
                status = "🔴 DROPPED"
        elif str(r1_rank).startswith(">") and isinstance(r2_rank, int):
            status = "🟢 GAINED"

        print(f"{str(q_id):<5} | {str(target_vid):<12} | {str(r1_rank):<14} | {str(r2_rank):<14} | {status}")
    print("-" * 75)
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
    print(f"[SUMMARY] Thời gian xử lý trung bình / query: {avg_time:.2f} giây.")

    report_dir = ROOT_DIR / "benchmark" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"benchmark_report_{int(time.time())}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "stage1_metrics": s1_metrics,
            "stage2_metrics": s2_metrics,
            "stage1_details": s1_details,
            "stage2_details": s2_details,
            "avg_latency_sec": round(avg_time, 2)
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Báo cáo chi tiết đã được lưu tại: {report_file}\n")


if __name__ == "__main__":
    GT_JSON_FILE = Path(__file__).parent / "ground_truth_test_v3.json"
    run_benchmark(GT_JSON_FILE, retrieval_mode="multi")