# 📊 AIC 2026 — Data Architecture & Schema Specification

Tài liệu này mô tả chi tiết kiến trúc dữ liệu, cấu trúc schema và cách thức tổ chức các file chỉ mục (`.index`), cơ sở dữ liệu (`.db`), và file ánh xạ (`.json`) phục vụ cho hệ thống truy vấn và xếp hạng video (**Video Retrieval & Multi-Score Reranking**).

Mục tiêu của tài liệu giúp người đọc và nhà phát triển:
1. Nắm rõ cách hệ thống ánh xạ vector embedding sang từng khung hình video.
2. Hiểu cấu trúc từng trường dữ liệu để có thể tự viết mã trích xuất đặc trưng và tái tạo dữ liệu tương thích 100% với mã nguồn hệ thống mà không cần phụ thuộc vào file nhị phân có sẵn.

---

## 1. Cấu trúc thư mục dữ liệu tổng quát (`data/`)

Toàn bộ dữ liệu phục vụ runtime được đặt trong thư mục `data/` ở thư mục gốc của dự án:

```
data/
├── indexes/
│   ├── siglip2_keyframes_b1_b2.index           # FAISS Index cho mô hình SigLIP2
│   ├── dfn5b_clip_vit_h14_keyframes_b1_b2.index # FAISS Index cho mô hình DFN5B ViT-H/14
│   ├── object_IVFPQ_b1_b2.index                # FAISS Index cho vector đối tượng (E5 embedding)
│   ├── metadata_IVFPQ_b1_b2.index              # FAISS Index cho vector OCR/ASR (E5 embedding)
│   ├── keyframes_b1_b2.db                      # SQLite DB chứa thông tin keyframes & metadata
│   ├── object_mapping_b1_b2_kf.json            # Mapping {video_id_kfidx} -> danh sách object vector index
│   └── metadata_mapping_b1_b2_kf.json          # Mapping {video_id_kfidx} -> metadata vector index
├── mapping/
│   └── video_fps_mapping.json                  # Mapping FPS của từng video gốc
├── keyframes/
│   └── <tên_nhóm_keyframe>/<video_id>/<tên_ảnh>.jpg
└── videos/                                     # (Tùy chọn) Chứa video .mp4 gốc để stream trực tiếp
```

---

## 2. Nguyên tắc đồng bộ và liên kết dữ liệu (Unified ID Linking)

Hệ thống sử dụng cơ chế liên kết **1:1** giữa vị trí vector trong FAISS và chỉ mục dòng trong cơ sở dữ liệu:

```
FAISS Index (Vector thứ i: [0 .. N-1])
   │
   ▼
SQLite DB (keyframes_b1_b2.db: dòng có vector_index = i)
   │
   ├── video_id, keyframe_index, frame_idx, pts_time, fps, keyframe_path
   ├── object_entities: Danh sách các thực thể Object được nhận diện
   ├── ocr_texts: Danh sách từ ngữ OCR xuất hiện trên khung hình
   ├── asr_text: Đoạn phụ đề / lời thoại audio tại thời điểm khung hình
   └── metadata: Thông tin mô tả video (title, description, tags)
```

Nhờ cơ chế này:
- Quá trình Retrieval chỉ cần trả về danh sách `vector_index` (mảng số nguyên `int64`).
- Tầng Mapping tra cứu O(1) từ SQLite để lấy toàn bộ thông tin ngữ cảnh đầy đủ phục vụ cho tầng Reranking và hiển thị Web.

---

## 3. Chi tiết Schema từng file dữ liệu

### 3.1. Visual FAISS Indexes (`.index`)

#### a. `siglip2_keyframes_b1_b2.index`
- **Mô hình trích xuất**: `google/siglip2-base-patch16-256`
- **Kiểu FAISS**: `faiss.IndexFlatIP` (Inner Product, tương đương Cosine Similarity khi vector đã L2-normalized).
- **Kích thước vector (dimension)**: `d = 768`
- **Số lượng vector**: `540,856`
- **Chuẩn hóa**: Tất cả vector đặc trưng ảnh được chuẩn hóa L2 (`faiss.normalize_L2`) trước khi đưa vào index.
- **Phân bố dải chỉ mục theo chủ đề (Topic Filter Range)**:
  FAISS hỗ trợ tìm kiếm có điều kiện qua `faiss.IDSelectorRange` / `faiss.IDSelectorOr` bằng cách giới hạn phạm vi `vector_index`:

  | Nhóm chủ đề | Ký hiệu Video | Dải Vector Index `[start, end)` |
  | :--- | :---: | :--- |
  | **Thời sự 60 Giây** | L21, L22, M | `[0, 65830)` và `[358639, 475164)` |
  | **Đua xe đạp theo chặng** | L23, S | `[65830, 72220)` và `[475164, 496777)` |
  | **Lân sư rồng** | L24 | `[72220, 85982)` |
  | **Bài giảng ôn thi THPT** | L25 | `[85982, 125529)` |
  | **Nấu ăn ViVU TV** | L26 | `[125529, 292636)` |
  | **Du lịch miền Tây** | L27–L29 | `[292636, 341643)` |
  | **Lan tỏa năng lượng tích cực** | L30 | `[341643, 358639)` |
  | **Camera giao thông** | N | `[496777, 540856)` |

#### b. `dfn5b_clip_vit_h14_keyframes_b1_b2.index`
- **Mô hình trích xuất**: OpenCLIP `ViT-H-14` (pretrained `dfn5b`)
- **Kiểu FAISS**: `faiss.IndexFlatIP`
- **Kích thước vector (dimension)**: `d = 1024`
- **Số lượng vector**: `540,856`
- **Quy ước đồng bộ**: Vector thứ `i` của DFN5B đại diện cho cùng một keyframe với vector thứ `i` của SigLIP2.

---

### 3.2. SQLite Keyframe Database (`keyframes_b1_b2.db`)

Tệp cơ sở dữ liệu SQLite chứa thông tin chi tiết từng keyframe.

- **Tên bảng**: `keyframes`
- **Cấu trúc cột (Table Schema)**:

| Tên cột | Kiểu dữ liệu | Ràng buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `vector_index` | `INTEGER` | `PRIMARY KEY` | Chỉ số thứ tự vector trong FAISS (0 .. N-1) |
| `video_id` | `TEXT` | `NOT NULL` | Mã định danh video (ví dụ: `L21_V001`) |
| `keyframe_index`| `INTEGER`| `NOT NULL` | Số thứ tự keyframe trong video (1, 2, 3...) |
| `frame_idx` | `INTEGER` | `NOT NULL` | Chỉ số frame thực tế trong luồng video gốc |
| `pts_time` | `REAL` | `NOT NULL` | Timestamp thời gian của frame tính bằng giây |
| `fps` | `INTEGER` | | Tốc độ khung hình của video (ví dụ: 25, 30) |
| `keyframe_path` | `TEXT` | | Đường dẫn tương đối tới ảnh keyframe |
| `object_path` | `TEXT` | | Đường dẫn tới file nhãn object gốc |
| `metadata_path` | `TEXT` | | Đường dẫn tới file metadata thô |
| `object_entities`| `TEXT` | | JSON string danh sách nhãn đối tượng (YOLO/OpenImages) |
| `metadata` | `TEXT` | | JSON string chứa tiêu đề, mô tả, từ khóa video |
| `ocr_texts` | `TEXT` | | JSON string danh sách text nhận dạng được qua OCR |
| `asr_text` | `TEXT` | | Chuỗi văn bản giọng nói (speech-to-text) tại khung hình |

- **Câu lệnh tạo bảng (DDL)**:

```sql
CREATE TABLE keyframes (
    vector_index INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    keyframe_index INTEGER NOT NULL,
    frame_idx INTEGER NOT NULL,
    pts_time REAL NOT NULL,
    fps INTEGER DEFAULT 25,
    keyframe_path TEXT,
    object_path TEXT,
    metadata_path TEXT,
    object_entities TEXT,
    metadata TEXT,
    ocr_texts TEXT,
    asr_text TEXT
);

CREATE INDEX idx_keyframes_video_id ON keyframes(video_id);
CREATE INDEX idx_keyframes_video_kf ON keyframes(video_id, keyframe_index);
```

- **Ví dụ bản ghi thực tế (Sample Record)**:

```json
{
  "vector_index": 0,
  "video_id": "L21_V001",
  "keyframe_index": 1,
  "frame_idx": 0,
  "pts_time": 0.0,
  "fps": 30,
  "keyframe_path": "Custom_Keyframes/L21_V001/000000.jpg",
  "object_path": "data/object/custom_objects/L21_V001/001.json",
  "metadata_path": "data/metadata/media-info/L21_V001.json",
  "object_entities": "[\"Poster\", \"Skyscraper\", \"Balloon\"]",
  "metadata": "{\"title\": \"Thời sự 60 Giây Sáng\", \"description\": \"Bản tin tổng hợp...\", \"keywords\": [\"tin tức\", \"60 giây\"]}",
  "ocr_texts": "[{\"text\": \"HTV9\", \"score\": 0.96}, {\"text\": \"06:30:11\", \"score\": 0.94}]",
  "asr_text": "Chào mừng quý vị và các bạn đang theo dõi bản tin 60 giây"
}
```

---

### 3.3. Object Semantic Index & Mapping

Phục vụ việc so khớp đối tượng nâng cao bằng mô hình ngôn ngữ (**Hybrid Object Matching**).

#### a. `object_IVFPQ_b1_b2.index`
- **Mô hình trích xuất**: `intfloat/multilingual-e5-base` (hoặc `multilingual-e5-large`)
- **Kiểu FAISS**: `faiss.IndexIVFPQ` hoặc `faiss.IndexFlatIP`
- **Kích thước vector (dimension)**: `d = 768` (với e5-base)
- **Mục đích**: Chứa vector embedding của các thực thể nhãn đối tượng riêng lẻ. Mỗi vector đại diện cho một đối tượng phát hiện được (ví dụ: `"person"`, `"red car"`, `"frying pan"`).

#### b. `object_mapping_b1_b2_kf.json`
- **Cấu trúc**: JSON Dictionary với key là định danh khung hình và value là danh sách chỉ số vector đối tượng:
  ```json
  {
    "L21_V001_0001": [0, 1, 2],
    "L21_V001_0002": [3, 4],
    "L21_V001_0003": [5, 6, 7, 8]
  }
  ```
- **Ý nghĩa**: Key có định dạng `"{video_id}_{keyframe_index:04d}"`. Value là mảng các `vector_index` trong file `object_IVFPQ_b1_b2.index`. Nhờ đó, khi cần tính điểm Object của keyframe `L21_V001_0001`, hệ thống reconstruct trực tiếp các vector này từ FAISS để tính độ tương đồng ngữ nghĩa với query tiếng Anh.

---

### 3.4. Metadata Semantic Index & Mapping

Phục vụ việc đối chiếu ngữ nghĩa sâu giữa câu hỏi với toàn bộ văn bản OCR xuất hiện trên màn hình và phụ đề ASR.

#### a. `metadata_IVFPQ_b1_b2.index`
- **Mô hình trích xuất**: `intfloat/multilingual-e5-base`
- **Kiểu FAISS**: `faiss.IndexIVFPQ` hoặc `faiss.IndexFlatIP`
- **Kích thước vector (dimension)**: `d = 768`
- **Mục đích**: Chứa vector ngữ nghĩa của đoạn text tổng hợp gồm `[OCR text] + " " + [ASR text]` cho từng keyframe.

#### b. `metadata_mapping_b1_b2_kf.json`
- **Cấu trúc**: JSON Dictionary ánh xạ khung hình tới vector metadata tương ứng:
  ```json
  {
    "L21_V001_0001": { "vector_index": 0 },
    "L21_V001_0002": { "vector_index": 1 },
    "L21_V001_0003": { "vector_index": 2 }
  }
  ```

---

### 3.5. Video FPS Mapping (`video_fps_mapping.json`)

- **Đường dẫn**: `data/mapping/video_fps_mapping.json`
- **Cấu trúc**: Danh sách các đối tượng JSON khai báo tốc độ khung hình chuẩn của từng file video:
  ```json
  [
    {
      "video_id": "L21_V001",
      "fps": 30.0,
      "csv_filename": "L21_V001.csv"
    },
    {
      "video_id": "L26_V001",
      "fps": 25.0,
      "csv_filename": "L26_V001.csv"
    }
  ]
  ```
- **Ứng dụng**: Phục vụ API chuyển đổi qua lại giữa thời gian phát video (`pts_time`), thời gian thực tế `HH:MM:SS` và số thứ tự khung hình `frame_idx` chính xác tuyệt đối.

---

## 4. Hướng dẫn quy trình tái tạo dữ liệu (Data Reproduction Workflow)

Nếu bạn có một tập video mới hoặc muốn tái tạo toàn bộ dữ liệu từ đầu, hãy thực hiện theo 4 bước chuẩn sau:

```
[Video gốc .mp4]
       │
       ▼ (1. Trích xuất Keyframes & Thông số FPS)
[Ảnh Keyframe .jpg] ───► Tính pts_time & frame_idx ───► video_fps_mapping.json
       │
       ├───► (2. Trích xuất OCR, ASR, YOLO Objects)
       │         │
       │         └───► Gom vào file keyframes.db (SQLite)
       │
       ├───► (3. Trích xuất Visual Features)
       │         ├── SigLIP2 (768d)  ───► siglip2_keyframes.index
       │         └── DFN5B (1024d)   ───► dfn5b_keyframes.index
       │
       └───► (4. Trích xuất Text Semantic Features bằng E5)
                 ├── Object Labels   ───► object_IVFPQ.index + object_mapping.json
                 └── OCR + ASR Text  ───► metadata_IVFPQ.index + metadata_mapping.json
```

### Bước 1: Trích xuất Keyframes
Sử dụng `ffmpeg` hoặc OpenCV để trích xuất keyframe theo scene change hoặc định kỳ. Lưu lại các trường `pts_time`, `frame_idx`, `fps`.

### Bước 2: Nhận diện nội dung đa phương tiện
- **OCR**: Sử dụng PaddleOCR hoặc EasyOCR trên từng ảnh keyframe, trích xuất text và confidence score.
- **ASR**: Sử dụng Whisper để trích xuất transcript có timestamp theo từng đoạn video, ánh xạ lời thoại vào keyframe theo cửa sổ thời gian `[pts_time - 2s, pts_time + 2s]`.
- **Object**: Sử dụng YOLOv8 (hoặc mô hình OpenImages) để nhận diện các đối tượng và lưu nhãn thực thể.

### Bước 3: Tạo FAISS Visual Indexes
```python
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Ví dụ cho SigLIP 2
model_name = "google/siglip2-base-patch16-256"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).cuda().eval()

# Trích xuất embeddings cho toàn bộ keyframes theo thứ tự
embeddings = []
for img_path in sorted_keyframe_paths:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        feat = model.get_image_features(**inputs).cpu().numpy()[0]
    embeddings.append(feat)

embeddings = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(768)
index.add(embeddings)
faiss.write_index(index, "data/indexes/siglip2_keyframes_b1_b2.index")
```

### Bước 4: Tạo SQLite Database
Đổ toàn bộ metadata thu thập được từ Bước 1 và 2 vào SQLite theo đúng Schema bảng `keyframes` ở Mục 3.2 để đảm bảo liên kết `vector_index = 0, 1, ..., N-1` đồng bộ với FAISS Index.
