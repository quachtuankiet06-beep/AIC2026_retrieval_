from collections import defaultdict
import json
import sqlite3

import faiss
import numpy as np
from tqdm import tqdm


def generate_video_dup_metadata(
    index_path: str,
    db_path: str,
    output_json_path: str,
    sim_threshold: float = 0.985,
    min_window: int = 2,
    max_window: int = 10,
    adaptive_scale: float = 1.0,
    min_adaptive_window: int = 2,
    max_adaptive_window: int = 10,
):
    """
    Tạo metadata per-video dựa trên keyframes window.
    Không dùng pts_time để quyết định logic.
    Chỉ dựa trên thứ tự keyframes trong video và độ tương đồng cosine.
    """

    print("[INFO] Loading FAISS index...")
    old_index = faiss.read_index(index_path)
    ntotal = old_index.ntotal

    print("[INFO] Reconstructing vectors...")
    vectors = old_index.reconstruct_n(0, ntotal).astype(np.float32, copy=False)
    faiss.normalize_L2(vectors)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Chỉ dùng để lấy video_id và thứ tự keyframes
    # Nếu DB của bạn có frame_idx thì đổi ORDER BY sang frame_idx.
    cur.execute(
        """
        SELECT vector_index, video_id
        FROM keyframes
        ORDER BY video_id, vector_index ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    video_buckets = defaultdict(list)
    for idx, vid in rows:
        if idx < ntotal:
            video_buckets[vid].append(int(idx))

    video_stats = {}

    for vid, frame_indices in tqdm(video_buckets.items(), desc="Analyzing Video Windows"):
        n = len(frame_indices)

        if n <= 1:
            video_stats[vid] = {
                "n_keyframes": n,
                "n_windows": 0,
                "median_dup_window": 1.0,
                "mean_dup_window": 1.0,
                "std_dup_window": 0.0,
                "adaptive_window_frames": 2,
            }
            continue

        dup_window_sizes = []
        anchor_pos = 0
        current_window = 1

        # Duyệt theo keyframe order trong video, tăng dần window
        for i in range(1, n):
            curr_idx = frame_indices[i]
            anchor_idx = frame_indices[anchor_pos]

            sim = float(np.dot(vectors[curr_idx], vectors[anchor_idx]))

            if sim >= sim_threshold:
                # Cùng cụm duplicate -> tăng window
                current_window += 1
            else:
                # Sang cụm khác -> chốt window cũ, reset anchor
                dup_window_sizes.append(current_window)
                current_window = 1
                anchor_pos = i

        dup_window_sizes.append(current_window)

        arr = np.asarray(dup_window_sizes, dtype=np.float32)

        median_window = float(np.median(arr))
        mean_window = float(arr.mean())
        p75_window = float(np.percentile(arr, 75))

        adaptive_window = int(round(p75_window * adaptive_scale))
        adaptive_window = int(
            np.clip(adaptive_window, min_adaptive_window, max_adaptive_window)
        )

        video_stats[vid] = {
            "n_keyframes": n,
            "n_windows": int(len(dup_window_sizes)),
            "median_dup_window": round(median_window, 4),
            "mean_dup_window": round(mean_window, 4),
            "p75_dup_window": round(p75_window, 4),
            "adaptive_window_frames": adaptive_window,
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(video_stats, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved metadata for {len(video_stats)} videos to {output_json_path}")


def main():
    generate_video_dup_metadata(
        index_path="data/indexes/siglip2_keyframes_new.index",
        db_path="data/indexes/keyframes_new_kf.db",
        output_json_path="data/indexes/siglip2_dup_stats.json",
        sim_threshold=0.93,
        min_window=2,
        max_window=10,
        adaptive_scale=1.0,
        min_adaptive_window=2,
        max_adaptive_window=10,
    ) 
    generate_video_dup_metadata(
        index_path="data/indexes/dfn5b_clip_vit_h14_keyframes_new.index",
        db_path="data/indexes/keyframes_new_kf.db",
        output_json_path="data/indexes/dfn5b_clip_vit_h14_dup_stats.json",
        sim_threshold=0.93,
        min_window=2,
        max_window=10,
        adaptive_scale=1.0,
        min_adaptive_window=2,
        max_adaptive_window=10,
    )


if __name__ == "__main__":
    main()