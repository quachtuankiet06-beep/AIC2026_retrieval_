# 04. Build Vector Database

## 1. Purpose

Module này chịu trách nhiệm xây dựng Vector Database từ toàn bộ CLIP Features do BTC cung cấp.

Vector Database sẽ được sử dụng để thực hiện truy xuất (retrieval) các keyframe tương đồng với truy vấn ngôn ngữ tự nhiên.

Việc xây dựng chỉ thực hiện **một lần** trong giai đoạn offline.

---

## 2. Input

```
data/
└── clip_features/
    ├── L01/
    │   ├── L01_V001.npy
    │   ├── L01_V002.npy
    │   └── ...
    ├── L02/
    └── ...
```

Mỗi file `.npy` chứa embedding CLIP của toàn bộ keyframe trong một video.

Ví dụ

```
L01_V001.npy

Shape

(523, 512)
```

Trong đó

- 523 : số lượng keyframe
- 512 : chiều embedding của CLIP ViT-B/32

Mỗi dòng tương ứng với **một keyframe**.

---

## 3. Output

```
data/
└── indexes/
    └── faiss.index
```

Đây là Vector Database của toàn bộ dataset.

Tất cả embedding của tất cả video sẽ được gộp thành một FAISS Index duy nhất.

---

## 4. Workflow

```
Load CLIP Features
        │
        ▼
Read every .npy file
        │
        ▼
Merge all embeddings
        │
        ▼
Normalize embeddings
        │
        ▼
Build FAISS Index
        │
        ▼
Save faiss.index
```

---

## 5. Processing Logic

### Step 1

Duyệt toàn bộ thư mục `clip_features`.

Ví dụ

```
L01_V001.npy

L01_V002.npy

...

L22_V198.npy
```

---

### Step 2

Load từng file `.npy`.

Ví dụ

```
(523,512)
```

---

### Step 3

Ghép toàn bộ embedding thành một ma trận lớn.

Ví dụ

```
Video 1

523 vectors

+

Video 2

642 vectors

+

...

↓

Global Embedding Matrix

(N,512)
```

Trong đó

```
N = Tổng số keyframe của toàn bộ dataset
```

---

### Step 4

Chuẩn hóa embedding.

Sử dụng L2 Normalization trước khi xây dựng FAISS.

Việc chuẩn hóa giúp Inner Product tương đương với Cosine Similarity.

---

### Step 5

Xây dựng FAISS Index.

Version 1 sử dụng

```
faiss.IndexFlatIP
```

để đảm bảo độ chính xác cao.

---

### Step 6

Lưu index xuống đĩa.

```
faiss.index
```

---

## 6. File Structure

Module dự kiến

```
src/

└── indexing/

    build_vector_db.py
```

Module này chịu trách nhiệm

- Load CLIP Features
- Merge Embeddings
- Normalize
- Build FAISS
- Save Index

---

## 7. Notes

- Chỉ xây dựng một lần.
- Không lưu metadata trong FAISS.
- Không lưu object trong FAISS.
- FAISS chỉ chứa embedding.

Thông tin của từng vector sẽ được lưu trong Mapping Database.

---

## 8. Future Improvements

Các phiên bản sau có thể thay thế

```
IndexFlatIP
```

bằng

- IVF
- HNSW
- PQ

nếu kích thước dataset tăng đáng kể.