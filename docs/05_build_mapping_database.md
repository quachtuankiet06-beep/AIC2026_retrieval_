# 11. Build Mapping Database

## 1. Purpose

Module này chịu trách nhiệm xây dựng Mapping Database nhằm ánh xạ giữa các vector trong FAISS Index và toàn bộ thông tin liên quan đến từng keyframe.

Mapping Database đóng vai trò là cầu nối giữa:

- FAISS Vector Database
- Keyframe Images
- Frame Index
- Object Detection
- Video Metadata

Module này chỉ được thực hiện **một lần** trong giai đoạn offline sau khi xây dựng Vector Database.

---

## 2. Input

```
data/

├── clip_features/
│   ├── L21/
│   │   ├── L21_V001.npy
│   │   ├── L21_V002.npy
│   │   └── ...
│
├── map/
│   ├── L21/
│   │   ├── L21_V001.csv
│   │   ├── L21_V002.csv
│   │   └── ...
│
├── keyframes/
│   ├── keyframes_L21/
│   │   ├── L21_V001/
│   │   ├── L21_V002/
│   │   └── ...
│
├── objects/
│   ├── L21_V001/
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   └── ...
│
└── metadata/
    ├── L21_V001.json
    ├── L21_V002.json
    └── ...
```

---

## 3. Output

```
data/

└── mapping/

    mapping_database.json
```

Mapping Database lưu toàn bộ thông tin cần thiết để truy xuất và re-ranking cho từng vector trong FAISS.

---

## 4. Data Relationship

Đối với một video, tất cả dữ liệu đều được tổ chức theo cùng một thứ tự keyframe.

```
L21_V001.npy
        │
        ▼
Embedding #i
        │
        ▼
Map CSV Row #i
        │
        ▼
Keyframe #i
        │
        ▼
Object JSON #i
```

Trong khi đó

```
Metadata

↓

L21_V001.json
```

được chia sẻ cho toàn bộ keyframe của video.

Điều này cho phép xây dựng Mapping Database bằng cách duyệt tuần tự từng embedding mà không cần thực hiện tìm kiếm giữa các nguồn dữ liệu.

---

## 5. Workflow

```
Load CLIP Features
        │
        ▼
Load Map CSV
        │
        ▼
Load Metadata
        │
        ▼
For each Keyframe
        │
        ▼
Build Mapping Record
        │
        ▼
Append to Mapping Database
        │
        ▼
Save mapping_database.json
```

---

## 6. Processing Logic

### Step 1

Duyệt toàn bộ video trong dataset.

Ví dụ

```
L21_V001

L21_V002

...

L22_V198
```

---

### Step 2

Load the resources of the current video:

- Map CSV
- Metadata JSON

Optionally load the corresponding CLIP Feature file to verify that the number of embeddings matches the number of keyframes in the Map CSV.

---

### Step 3

Duyệt từng embedding trong file `.npy`.

Đối với mỗi embedding:

- Lấy thông tin frame từ Map CSV.
- Xác định đường dẫn keyframe.
- Xác định đường dẫn Object JSON.
- Tham chiếu Metadata của video.

---

### Step 4

Đọc Object JSON.

Trích xuất danh sách object từ trường

```
detection_class_entities
```

Sau đó:

- chuyển về lowercase;
- loại bỏ object trùng lặp;
- lưu dưới dạng cache để phục vụ re-ranking.

---

### Step 5

Đọc Metadata JSON.

Lưu cache các trường thường xuyên sử dụng:

- title
- description
- keywords
- publish_date

Đồng thời lưu đường dẫn đến Metadata gốc để phục vụ các phiên bản re-ranking trong tương lai.

---

### Step 6

Tạo Mapping Record.

Ví dụ

```json
{
    "vector_index": 15230,

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
        "boat",
        "vehicle"
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
```

---

### Step 7

Lưu toàn bộ Mapping Database.

Mỗi record tương ứng với đúng một vector trong FAISS Index.

```
vector_index

↓

mapping_record
```

---

## 7. File Structure

```
src/

└── indexing/

    build_mapping_database.py
```

Module chịu trách nhiệm:

- Load toàn bộ dữ liệu của BTC.
- Xây dựng Mapping Database.
- Tiền xử lý object cache.
- Tiền xử lý metadata cache.
- Lưu Mapping Database.

---

## 8. Design Notes

Mapping Database lưu hai loại thông tin:

### 1. Reference

Đường dẫn đến dữ liệu gốc.

Ví dụ

- keyframe_path
- object_path
- metadata_path

Điều này giúp các phiên bản re-ranking trong tương lai có thể khai thác thêm thông tin mà không cần xây dựng lại Mapping Database.

---

### 2. Cache

Các thông tin thường xuyên sử dụng trong quá trình re-ranking.

Bao gồm:

- object_entities
- title
- description
- keywords
- publish_date

Việc cache giúp giảm số lần đọc file JSON trong quá trình truy vấn, đồng thời vẫn giữ khả năng truy cập dữ liệu gốc khi cần.

---

## 9. Future Improvements

Các phiên bản tiếp theo có thể mở rộng Mapping Database bằng cách bổ sung thêm:

- OCR text
- Image Caption
- Scene Classification
- Face Recognition
- Audio Transcript

mà không cần thay đổi cấu trúc của FAISS Vector Database.