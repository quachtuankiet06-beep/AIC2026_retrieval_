# 14. Re-ranking Pipeline

## 1. Mục đích

Re-ranking Pipeline chịu trách nhiệm đánh giá lại danh sách Candidate sau bước Mapping nhằm cải thiện chất lượng kết quả truy xuất.

Thay vì chỉ dựa vào Visual Similarity của CLIP, hệ thống sẽ khai thác thêm:

- Object Detection
- Metadata

để tính toán điểm phù hợp cuối cùng cho từng Candidate.

---

## 2. Input

### Candidate List

Đầu ra của Mapping Pipeline.

Mỗi Candidate bao gồm:

- Retrieval Score
- Object Cache
- Metadata Cache
- Thông tin keyframe

Ví dụ:

```json
{
    "rank": 1,
    "vector_index": 15230,
    "retrieval_score": 0.9134,
    "video_id": "L21_V001",
    "frame_idx": 1078,
    "keyframe_path": "...",
    "object_entities": [
        "tower",
        "building",
        "boat"
    ],
    "metadata": {
        "title": "...",
        "description": "...",
        "keywords": [
            ...
        ]
    }
}
```

---

## 3. Output

Danh sách Candidate sau khi được sắp xếp lại.

Ví dụ:

```json
[
    {
        "rank": 1,

        "video_id": "L21_V001",

        "frame_idx": 1078,

        "pts_time": 35.93,

        "keyframe_path": "keyframes/L21/L21_V001/000035.jpg",

        "retrieval_score": 0.9134,

        "object_score": 0.82,

        "metadata_score": 0.74,

        "final_score": 0.87
    },
    {
        "rank": 2,

        "video_id": "L21_V015",

        "frame_idx": 528,

        "pts_time": 17.60,

        "keyframe_path": "keyframes/L21/L21_V015/000018.jpg",

        "retrieval_score": 0.9011,

        "object_score": 0.75,

        "metadata_score": 0.68,

        "final_score": 0.83
    }
]
```


---

## 4. Workflow

```
Candidate List
        │
        │
        ├───────────────┐
        │               │
        ▼               ▼
Extract Query      Metadata BM25
Objects (spaCy)         │
        │               │
        ▼               ▼
Object Matching   Metadata Matching
        │               │
        ▼               ▼
Object Score      Metadata Score
        └──────┬────────┘
               │
               ▼
Normalize Scores
               │
               ▼
Weighted Score Fusion
               │
               ▼
Sort Candidates
               │
               ▼
Final Ranking
```

---

## 5. Processing Logic

### Step 1

Nhận Candidate List từ Mapping Pipeline.

---

### Step 2

Trích xuất Object từ Query.

Sử dụng spaCy để phân tích câu truy vấn và lấy các danh từ (Nouns) làm Object Query.

Ví dụ:

```
Query

xe cứu hỏa đang chạy trên đường
```

↓

```
Object Query

fire truck

road
```

Danh sách này sẽ được sử dụng trong bước Object Matching.

---

### Step 3

Object Matching.

So sánh Object Query với danh sách Object của từng Candidate.

Ví dụ:

```
Object Query

person
motorcycle
```

Candidate:

```
person

motorcycle

building
```

↓

```
Matched Objects

person

motorcycle
```

Sau đó tính Object Score.

Trong Version 1 sử dụng Exact Matching.

---

### Step 4

Metadata Matching.

Đối với mỗi Candidate, tạo một văn bản gồm:

```
title

+

description

+

keywords
```

Sau đó sử dụng BM25 để tính Metadata Score giữa Query và Metadata.

---

### Step 5

Normalize Score.

Ba loại điểm sẽ được chuẩn hóa về cùng một thang đo:

```
[0, 1]
```

Bao gồm:

- Retrieval Score
- Object Score
- Metadata Score

Việc chuẩn hóa giúp các điểm có thể kết hợp với nhau một cách hợp lý.

---

### Step 6

Weighted Score Fusion.

Điểm cuối cùng được tính bằng tổng có trọng số:

```
Final Score

=

Clip Weight × Retrieval Score

+

Object Weight × Object Score

+

Metadata Weight × Metadata Score
```

Các trọng số được cấu hình trong file cấu hình của hệ thống.

---

### Step 7

Sắp xếp Candidate theo Final Score giảm dần.

Danh sách sau khi sắp xếp sẽ trở thành kết quả cuối cùng của hệ thống.

---

## 6. Configuration

Các tham số được lưu trong:

```
configs/reranking.yaml
```

Ví dụ:

```yaml
clip_weight: 0.6

object_weight: 0.2

metadata_weight: 0.2
```

Các trọng số có thể điều chỉnh mà không cần thay đổi mã nguồn.

---

## 7. Candidate Score Schema

Sau Re-ranking, mỗi Candidate sẽ bao gồm:

| Trường | Ý nghĩa |
|----------|----------|
| retrieval_score | Điểm từ CLIP Retrieval |
| object_score | Điểm Object Matching |
| metadata_score | Điểm BM25 Metadata |
| final_score | Điểm cuối cùng dùng để xếp hạng |

Việc lưu riêng từng loại điểm giúp:

- Dễ debug
- Dễ đánh giá
- Dễ tuning trọng số
- Dễ mở rộng hệ thống

---

## 8. Design Notes

Re-ranking không thực hiện tìm kiếm mới.

Module này chỉ đánh giá lại Candidate đã được Retrieval tìm thấy.

Việc tách Retrieval và Re-ranking giúp:

- Dễ thay thế mô hình Retrieval
- Dễ thử nghiệm nhiều chiến lược Re-ranking
- Giảm sự phụ thuộc giữa các module

---

## 9. Future Improvements

Pipeline có thể mở rộng bằng nhiều nguồn thông tin khác như:

- OCR
- Image Caption
- Audio Transcript
- Scene Classification
- Face Recognition

Mỗi nguồn thông tin chỉ cần sinh thêm một loại Score mới và bổ sung vào bước Score Fusion mà không cần thay đổi kiến trúc tổng thể của hệ thống.