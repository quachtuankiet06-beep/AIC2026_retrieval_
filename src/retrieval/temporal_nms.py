import math
from typing import List, Dict, Any
import numpy as np

# ==========================================================
# Mới: Temporal NMS v4 (active) - Lọc trùng ưu tiên điểm số
# ==========================================================
from typing import List, Dict, Any

def temporal_nms(
    candidate_list: List[Dict[str, Any]],
    min_gap_sec: float = 1.0,
    score_key: str = "score",
    max_per_video: int = 5,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Temporal NMS v4: Lọc trùng ngữ nghĩa tối giản, ưu tiên khung hình có điểm cao nhất.
    """
    if not candidate_list:
        return []

    # 1. Gom candidate theo video_id
    video_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidate_list:
        vid = cand.get("video_id", "__unknown__")
        video_buckets.setdefault(vid, []).append(cand)

    kept: List[Dict[str, Any]] = []

    # 2. Xử lý từng video
    for vid, frames in video_buckets.items():
        frames_sorted = sorted(frames, key=lambda x: x.get(score_key, 0.0), reverse=True)
        
        video_kept = []
        for cand in frames_sorted:
            if len(video_kept) >= max_per_video:
                break
                
            # Safe-check tránh crash khi pts_time bị None
            c_time = cand.get("pts_time")
            if c_time is None:
                c_time = cand.get("frame_idx", 0) / 25.0
            
            is_duplicate = False
            for kept_cand in video_kept:
                k_time = kept_cand.get("pts_time")
                if k_time is None:
                    k_time = kept_cand.get("frame_idx", 0) / 25.0
                
                if abs(c_time - k_time) < min_gap_sec:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Đảm bảo giữ boosted_score bằng score thô để tương thích ngược với benchmark script
                cand["boosted_score"] = cand.get(score_key, 0.0)
                video_kept.append(cand)
        
        kept.extend(video_kept)

    # 3. Sort lại kết quả cuối cùng theo điểm số
    kept.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    
    print(f"[INFO] Temporal NMS v4: {len(candidate_list)} -> {len(kept)} candidates (min_gap={min_gap_sec}s).")
    return kept

