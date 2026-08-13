# 12. Retrieval Pipeline

## 1. Mục đích

Module Retrieval chịu trách nhiệm chuyển đổi truy vấn văn bản của người dùng thành vector embedding, tìm kiếm trên FAISS Vector Database và trả về danh sách các vector phù hợp nhất.

Module này **không thực hiện Mapping** và **không thực hiện Re-ranking**.

Đầu ra của Retrieval chỉ bao gồm:

- Vector Index
- Similarity Score

Các bước Mapping và Re-ranking sẽ được thực hiện ở các module tiếp theo.

---

## 2. Input

### User Query

Ví dụ:

```
xe cứu hỏa đang chạy trên đường

người đang chơi bóng đá

tàu thuyền trên biển
```

---

## 3. Output

Danh sách các vector phù hợp nhất sau khi áp dụng Similarity Threshold.

Ví dụ:

```json
[
    {
        "rank": 1,
        "vector_index": 15230,
        "score": 0.9134
    },
    {
        "rank": 2,
        "vector_index": 86452,
        "score": 0.9017
    }
]
```

Module Retrieval không trả về:

- frame_idx
- keyframe_path
- metadata
- object

Các thông tin này sẽ được bổ sung ở bước Mapping.

---

## 4. Workflow

```
User Query
      │
      ▼
CLIP ViT-B/32 Text Encoder
      │
      ▼
L2 Normalize
      │
      ▼
FAISS Search
      │
      ▼
Top-K Retrieval
      │
      ▼
Similarity Threshold
      │
      ▼
Return (vector_index, score)
```

---

## 5. Processing Logic

### Step 1

Nhận câu truy vấn từ người dùng.

Ví dụ:

```
xe buýt màu đỏ
```

---

### Step 2

Sử dụng **OpenAI CLIP ViT-B/32 Text Encoder** để chuyển câu truy vấn thành vector embedding.

```
Query

↓

512-dimensional embedding
```

---

### Step 3

Chuẩn hóa vector bằng **L2 Normalization**.

Việc chuẩn hóa giúp đảm bảo phép đo độ tương đồng giữa truy vấn và CLIP Feature trong FAISS được nhất quán.

---

### Step 4

Thực hiện tìm kiếm trên FAISS Vector Database.

FAISS sẽ trả về Top-K vector có độ tương đồng cao nhất.

Ví dụ:

```
Top-K = 100
```

---

### Step 5

Áp dụng Similarity Threshold.

Những vector có Similarity Score nhỏ hơn ngưỡng cấu hình sẽ bị loại bỏ.

Ví dụ:

```
score >= retrieval_threshold
```

Sau bước này số lượng kết quả có thể nhỏ hơn Top-K.

---

### Step 6

Trả về danh sách Retrieval Results.

Ví dụ:

```json
[
    {
        "rank": 1,
        "vector_index": 15230,
        "score": 0.9134
    },
    {
        "rank": 2,
        "vector_index": 86452,
        "score": 0.9017
    }
]
```

Module Retrieval kết thúc tại đây.

---

## 6. Configuration

Các tham số của Retrieval được cấu hình trong file:

```
configs/retrieval.yaml
```

Ví dụ:

```yaml
model: "ViT-B/32"

normalize: true

top_k: 100

retrieval_threshold: 0.25
```

Trong đó:

| Tham số | Ý nghĩa |
|---------|----------|
| model | Mô hình CLIP sử dụng để sinh embedding |
| normalize | Bật/Tắt chuẩn hóa L2 |
| top_k | Số lượng vector tối đa lấy từ FAISS |
| retrieval_threshold | Ngưỡng Similarity tối thiểu để giữ lại kết quả |

---

## 7. Design Notes

Module Retrieval chỉ chịu trách nhiệm tìm kiếm theo đặc trưng thị giác (Visual Similarity).

Nó không sử dụng:

- Object Detection
- Metadata
- OCR
- Caption

Điều này giúp module luôn đơn giản, độc lập và có thể thay thế mô hình embedding hoặc Vector Database mà không ảnh hưởng đến các bước phía sau.

---

## 8. Next Step

Đầu ra của Retrieval sẽ được chuyển sang **Mapping Module**.

Mapping Module sẽ sử dụng `vector_index` để tra cứu Mapping Database và xây dựng Candidate List chứa đầy đủ thông tin của từng keyframe.

Sau khi hoàn thành Mapping, Candidate List sẽ được chuyển sang **Re-ranking Module** để thực hiện xếp hạng lại dựa trên Object và Metadata.

---

## 9. Future Improvements

Module Retrieval có thể được mở rộng trong tương lai bằng các hướng sau:

- Hỗ trợ nhiều mô hình embedding (SigLIP, EVA-CLIP, OpenCLIP,...)
- Hỗ trợ truy vấn đa ngôn ngữ
- Hybrid Retrieval (Vector + BM25)
- Batch Retrieval
- Query Expansion
- GPU FAISS Index