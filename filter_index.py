


# ==========================================================
# Cũ: Hàm lọc đơn giản (đã comment)
# ==========================================================
# def filter_index_simple(
#     old_index_path: str,
#     db_path: str,
#     new_index_path: str,
#     sim_threshold: float = 0.985,  # Ngưỡng giống nhau tuyệt đối
#     time_window_sec: float = 3.0,  # Chỉ so sánh nếu nằm trong khoảng 3s gần nhất
# ):
#   # 1. Load FAISS index & Reconstruct vectors
#   print("Loading Index & Reconstructing vectors...")
#   old_index = faiss.read_index(old_index_path)
#   ntotal = old_index.ntotal
#   vectors = old_index.reconstruct_n(0, ntotal)
#   dim = vectors.shape[1]
#
#   # Đảm bảo vector đã được chuẩn hóa L2 để tính Cosine Similarity bằng Dot Product
#   faiss.normalize_L2(vectors)
#
#   # 2. Lấy metadata (video_id, pts_time) từ SQLite
#   conn = sqlite3.connect(db_path)
#   cur = conn.cursor()
#   cur.execute(
#       "SELECT vector_index, video_id, pts_time FROM keyframes ORDER BY"
#       " video_id, pts_time ASC"
#   )
#   rows = cur.fetchall()
#   conn.close()
#
#   # Group theo video
#   video_buckets = defaultdict(list)
#   for idx, vid, pts in rows:
#     if idx < ntotal:
#       video_buckets[vid].append((idx, float(pts)))
#
#   kept_indices = []
#
#   # 3. Lọc trùng đơn giản theo tuyến tính thời gian
#   for vid, frames in tqdm(
#       video_buckets.items(), desc="Filtering Near-Duplicates"
#   ):
#     if not frames:
#       continue
#
#     # Luôn giữ frame đầu tiên của mỗi video làm Anchor ban đầu
#     last_kept_idx, last_kept_pts = frames[0]
#     kept_indices.append(last_kept_idx)
#
#     for idx, pts in frames[1:]:
#       time_diff = pts - last_kept_pts
#
#       # Nếu frame này quá xa về mặt thời gian (> time_window_sec), giữ lại ngay
#       if time_diff > time_window_sec:
#         kept_indices.append(idx)
#         last_kept_idx, last_kept_pts = idx, pts
#         continue
#
#       # Nếu nằm trong cửa sổ thời gian, kiểm tra Cosine Similarity với Anchor gần nhất
#       sim = float(np.dot(vectors[idx], vectors[last_kept_idx]))
#
#       if sim >= sim_threshold:
#         # Trùng lặp cao (>= 0.985) -> Bỏ qua (Duplicate)
#         continue
#       else:
#         # Khác biệt đủ lớn -> Giữ lại và cập nhật Anchor mới
#         kept_indices.append(idx)
#         last_kept_idx, last_kept_pts = idx, pts
#
#   # 4. Ghi Index mới
#   kept_arr = np.array(kept_indices, dtype=np.int64)
#   print(
#       f"\n[RESULTS] Original: {ntotal} | Kept: {len(kept_arr)} "
#       f"({len(kept_arr)/ntotal*100:.2f}%) | Removed: {ntotal - len(kept_arr)}"
#   )
#
#   base_index = faiss.IndexFlatIP(dim)
#   new_index = faiss.IndexIDMap(base_index)
#   new_index.add_with_ids(vectors[kept_arr], kept_arr)
#
#   faiss.write_index(new_index, new_index_path)
#   print(f"Saved new index to: {new_index_path}")
#
# if __name__ == "__main__":
#   filter_index_simple(
#       old_index_path="data/indexes/dfn5b_clip_vit_h14_keyframes_new.index",
#       db_path="data/indexes/keyframes_new_kf.db",
#       new_index_path=(
#           "data/indexes/dfn5b_clip_vit_h14_keyframes_new_rag.index"
#       ),
#       sim_threshold = 0.985,
#       time_window_sec = 4
#   )
# if __name__ == "__main__":
#   filter_index_simple(
#       old_index_path="data/indexes/siglip2_keyframes_new.index",
#       db_path="data/indexes/keyframes_new_kf.db",
#       new_index_path=(
#           "data/indexes/siglip2_keyframes_new_rag.index"
#       ),
#       sim_threshold = 0.985,
#       time_window_sec = 4
#   )

# ==========================================================
# Hàm lọc trùng dựa trên thống kê video (Kẹp trần Window)
# ==========================================================
import json
import sqlite3
from collections import defaultdict
import faiss
import numpy as np
from tqdm import tqdm


def filter_index_simple_per_video(
    old_index_path: str,
    db_path: str,
    video_stats_path: str,
    new_index_path: str,
    sim_threshold: float = 0.96,  # Tăng nhẹ để giữ frame tốt hơn cho Stage 1
    default_window_frames: int = 2,
    max_window_frames: int = 3,   # Kẹp trần khoảng cách window tối đa
    use_stats_key: str = "adaptive_window_frames",
):
    print("[INFO] Loading video stats...")
    with open(video_stats_path, "r", encoding="utf-8") as f:
        video_stats = json.load(f)

    print("[INFO] Loading Index & Reconstructing vectors...")
    old_index = faiss.read_index(old_index_path)
    ntotal = old_index.ntotal
    vectors = old_index.reconstruct_n(0, ntotal).astype(np.float32, copy=False)
    dim = vectors.shape[1]

    faiss.normalize_L2(vectors)

    print("[INFO] Loading metadata from SQLite...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT vector_index, video_id, frame_idx FROM keyframes ORDER BY"
        " video_id, frame_idx ASC"
    )
    rows = cur.fetchall()
    conn.close()

    video_buckets = defaultdict(list)
    for idx, vid, frame_idx in rows:
        if idx < ntotal:
            video_buckets[str(vid)].append((int(idx), int(frame_idx)))

    kept_indices = []

    print("[INFO] Filtering near-duplicates by per-video window...")
    for vid, frames in tqdm(video_buckets.items(), desc="Filtering"):
        if not frames:
            continue

        # Sort chắc chắn theo frame_idx
        frames = sorted(frames, key=lambda x: x[1])

        stats = video_stats.get(vid, {})
        raw_window = int(
            stats.get(
                use_stats_key,
                stats.get("mean_dup_window", default_window_frames),
            )
        )
        
        # Kẹp trần window_frames trong khoảng [1, max_window_frames]
        window_frames = min(max(1, raw_window), max_window_frames)

        # Luôn giữ frame đầu tiên
        last_kept_vector_idx = frames[0][0]
        last_kept_pos = 0  # Vị trí index trong danh sách keyframe
        kept_indices.append(last_kept_vector_idx)

        for current_pos in range(1, len(frames)):
            curr_vector_idx, _ = frames[current_pos]
            step_gap = current_pos - last_kept_pos  # Số lượng keyframe chênh lệch

            # Nếu khoảng cách bước vượt quá window_frames kẹp trần -> Giữ luôn
            if step_gap > window_frames:
                kept_indices.append(curr_vector_idx)
                last_kept_vector_idx = curr_vector_idx
                last_kept_pos = current_pos
                continue

            # Kiểm tra similarity với frame được giữ gần nhất
            sim = float(
                np.dot(vectors[curr_vector_idx], vectors[last_kept_vector_idx])
            )

            if sim >= sim_threshold:
                # Duplicate quá giống nhau -> Bỏ qua
                continue
            else:
                # Khác biệt đủ lớn -> Giữ lại và cập nhật anchor
                kept_indices.append(curr_vector_idx)
                last_kept_vector_idx = curr_vector_idx
                last_kept_pos = current_pos

    kept_arr = np.array(kept_indices, dtype=np.int64)
    print(
        f"\n[RESULTS] Original: {ntotal} | Kept: {len(kept_arr)} "
        f"({len(kept_arr) / ntotal * 100:.2f}%) | Removed: {ntotal - len(kept_arr)}"
    )

    base_index = faiss.IndexFlatIP(dim)
    new_index = faiss.IndexIDMap(base_index)
    new_index.add_with_ids(vectors[kept_arr], kept_arr)

    faiss.write_index(new_index, new_index_path)
    print(f"[INFO] Saved new index to: {new_index_path}")


def main():
    filter_index_simple_per_video(
        old_index_path="data/indexes/dfn5b_clip_vit_h14_keyframes_new.index",
        db_path="data/indexes/keyframes_new_kf.db",
        video_stats_path="data/indexes/dfn5b_clip_vit_h14_dup_stats.json",
        new_index_path="data/indexes/dfn5b_clip_vit_h14_keyframes_new_rag.index",
        sim_threshold=0.94,        # Cập nhật 0.96
        default_window_frames=2,
        max_window_frames=3,       # Giới hạn nhảy tối đa 3 keyframes
        use_stats_key="adaptive_window_frames",
    )
    filter_index_simple_per_video(
        old_index_path="data/indexes/siglip2_keyframes_new.index",
        db_path="data/indexes/keyframes_new_kf.db",
        video_stats_path="data/indexes/siglip2_dup_stats.json",
        new_index_path="data/indexes/siglip2_keyframes_new_rag.index",
        sim_threshold=0.94,        # Cập nhật 0.96
        default_window_frames=2,
        max_window_frames=3,       # Giới hạn nhảy tối đa 3 keyframes
        use_stats_key="adaptive_window_frames",
    )


if __name__ == "__main__":
    main()