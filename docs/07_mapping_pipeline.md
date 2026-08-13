# 13. Mapping Pipeline

## 1. Mục đích

Mapping Pipeline chịu trách nhiệm chuyển đổi kết quả từ Retrieval thành Candidate List phục vụ cho Re-ranking.

Module này không thực hiện:

- Tìm kiếm Vector
- Tính Similarity
- Re-ranking
- Chấm điểm

Nó chỉ thực hiện việc tra cứu Mapping Database và bổ sung đầy đủ thông tin cho từng kết quả Retrieval.

---

## 2. Input

### Retrieval Results

Đầu ra của Retrieval Pipeline.

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

---

### Mapping Database

Được xây dựng từ module:

```
11_build_mapping_database.md
```

Mapping Database lưu toàn bộ thông tin liên quan đến từng `vector_index`.

---

## 3. Output

Candidate List.

Ví dụ:

```json
[
    {
        "rank": 1,

        "vector_index": 15230,

        "score": 0.9134,

        "video_id": "L21_V001",

        "keyframe_index": 35,

        "frame_idx": 1078,

        "pts_time": 35.93,

        "fps": 30,

        "keyframe_path": "keyframes/L21/L21_V001/000035.jpg",

        "object_path": "data/objects/L21_V001/000035.json",

        "metadata_path": "data/metadata/L21_V001.json",

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
            ],
            "publish_date": "01/08/2024"
        }
    }
]
```

Candidate List sẽ được chuyển sang Re-ranking Pipeline.

---

## 4. Workflow

```
Retrieval Results
        │
        ▼
Lookup Mapping Database
        │
        ▼
Merge Retrieval Result
với Mapping Record
        │
        ▼
Candidate List
```

---

## 5. Processing Logic

### Step 1

Nhận Retrieval Results từ Retrieval Pipeline.

Mỗi kết quả bao gồm:

- rank
- vector_index
- score

---

### Step 2

Sử dụng `vector_index` để tra cứu Mapping Database.

Ví dụ:

```
vector_index = 15230

↓

mapping_database[15230]
```

Do mỗi `vector_index` là duy nhất nên việc tra cứu có độ phức tạp:

```
O(1)
```

---

### Step 3

Lấy Mapping Record tương ứng.

Ví dụ:

```
video_id

frame_idx

keyframe_path

object_entities

metadata

...
```

---

### Step 4

Ghép Retrieval Result và Mapping Record thành một Candidate.

```
Candidate

=

Retrieval Result

+

Mapping Record
```

---

### Step 5

Lặp lại cho toàn bộ Retrieval Results.

Sau khi hoàn thành sẽ thu được Candidate List.

---

### Step 6

Chuyển Candidate List sang Re-ranking Pipeline.

Mapping Pipeline kết thúc tại đây.

---

## 6. Candidate Schema

Mỗi Candidate bao gồm:

| Trường | Ý nghĩa |
|---------|----------|
| rank | Thứ hạng ban đầu từ Retrieval |
| vector_index | Chỉ số vector trong FAISS |
| score | Similarity Score |
| video_id | ID video |
| keyframe_index | Thứ tự keyframe trong video |
| frame_idx | Frame gốc trong video |
| pts_time | Thời gian của frame |
| fps | FPS của video |
| keyframe_path | Đường dẫn ảnh keyframe |
| object_path | Đường dẫn Object JSON |
| metadata_path | Đường dẫn Metadata JSON |
| object_entities | Cache object của keyframe |
| metadata | Cache metadata của video |

---

## 7. Design Notes

Mapping Pipeline chỉ thực hiện chuyển đổi dữ liệu (Data Transformation).

Module này không:

- tính Similarity
- thay đổi điểm số
- lọc kết quả
- sắp xếp lại Candidate

Mọi Candidate đều giữ nguyên Similarity Score từ Retrieval.

---

## 8. Hiệu năng

Mapping Database được tổ chức theo `vector_index`.

Ví dụ:

```python
mapping_database = {

    0: {...},

    1: {...},

    2: {...},

    ...

}
```

Do đó việc tra cứu Mapping Record chỉ cần:

```
mapping_database[vector_index]
```

với độ phức tạp:

```
O(1)
```

Điều này giúp Mapping Pipeline gần như không tạo thêm chi phí tính toán trong quá trình truy vấn.

---

## 9. Next Step

Candidate List sẽ được chuyển sang Re-ranking Pipeline.

Tại đây hệ thống sẽ khai thác:

- Object Detection
- Metadata
- BM25

để đánh giá lại mức độ phù hợp của từng Candidate và tạo ra danh sách kết quả cuối cùng.

---

## 10. Future Improvements

Trong các phiên bản tiếp theo, Candidate có thể được mở rộng thêm các thông tin như:

- OCR
- Caption
- Scene Classification
- Face Recognition
- Audio Transcript

mà không cần thay đổi Retrieval Pipeline hay Vector Database.