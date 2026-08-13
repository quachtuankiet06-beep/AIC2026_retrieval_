import faiss
import torch
import numpy as np
from pathlib import Path

# 1. Load thử index
index_path = Path("data/indexes/beit3.index")
index = faiss.read_index(str(index_path))
print(f"Tổng số vector trong index: {index.ntotal}")
print(f"Chiều vector (dimension): {index.d}")

# 2. Lấy thử vector đầu tiên trong index ra kiểm tra xem đã chuẩn hóa L2 chưa (Norm có bằng 1.0 không)
vec_0 = index.reconstruct(0) # Lấy vector gốc ở index 0
norm_val = np.linalg.norm(vec_0)
print(f"Norm của vector ảnh số 0 trong index: {norm_val:.4f} (Lý tưởng phải bằng 1.0 nếu đã chuẩn hóa L2)")