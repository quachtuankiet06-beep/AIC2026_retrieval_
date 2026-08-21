# 🎬 AIC 2026 — Video Search & Frame Retrieval System

Hệ thống truy xuất video đa mô hình cho cuộc thi **AIC 2026**, sử dụng:
- **SigLIP2** + **DFN5B ViT-H/14** (CLIP-based visual retrieval)
- **FAISS** vector search
- **Reranking** đa tầng: OCR · ASR · Object · Metadata · CLIP
- **FastAPI** backend + giao diện web

---

## 📁 Cấu trúc thư mục

```
AIC_2026/
├── fastapi_app/          # FastAPI backend
│   ├── main.py           # Routes / endpoints
│   ├── services.py       # Business logic, pipeline
│   ├── video_stream.py   # HTTP Range video streaming
│   ├── templates/        # Jinja2 HTML templates
│   └── static/           # CSS, JS assets
├── src/
│   ├── retrieval/        # CLIP retrieval pipelines (SigLIP2, DFN5B, multi-model)
│   ├── reranking/        # Multi-score reranking pipelines
│   ├── mapping/          # Keyframe mapping
│   ├── fusion/           # Score fusion
│   └── indexing/         # FAISS index building
├── configs/
│   ├── retrieval.yaml    # Cấu hình retrieval (top_k, model weights)
│   ├── reranking.yaml    # Cấu hình reranking (weights OCR/ASR/Object...)
│   └── reranking_temporal.yaml
├── data/                 # ⚠️ KHÔNG có trong repo — xem hướng dẫn bên dưới
│   ├── indexes/          # FAISS .index files + JSON mappings
│   ├── keyframes/        # Ảnh keyframe .jpg
│   ├── mapping/          # video_fps_mapping.json
│   ├── metadata/
│   └── object/
├── run_fastapi.py        # Entry point
└── requirements_docker.txt
```

---

## ⚙️ Cài đặt & Chạy thủ công

### Yêu cầu

- Python **3.12**
- NVIDIA GPU (khuyến nghị) hoặc CPU
- CUDA 12.x (nếu dùng GPU)

### Bước 1 — Clone repo

```bash
git clone https://github.com/quachtuankiet06-beep/AIC2026_retrieval_.git
cd AIC2026_retrieval_
```

### Bước 2 — Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Bước 3 — Cài dependencies

```bash
# Cài PyTorch với CUDA 12.1 (tương thích CUDA 12.x)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Nếu chỉ dùng CPU:
# pip install torch==2.4.1 torchvision==0.19.1

# Cài các package còn lại
pip install -r requirements_docker.txt
```

### Bước 4 — Chuẩn bị Data ⚠️

> Data **không** có trong repo vì quá lớn (~20GB+). Liên hệ team để nhận link tải.

Sau khi tải về, giải nén và đặt đúng cấu trúc:

```
data/
├── indexes/
│   ├── siglip2.index
│   ├── dfn5b_clip_vit_h14.index
│   ├── keyframes_mapping.json
│   ├── metadata_mapping.json
│   ├── object_mapping.json
│   └── ...
├── keyframes/
│   └── Keyframes_L21/keyframes/<video_id>/*.jpg
├── mapping/
│   └── video_fps_mapping.json
├── metadata/
└── object/
```

### Bước 5 — Chuẩn bị Video (tùy chọn)

Đặt các file video `.mp4` vào thư mục mặc định hoặc chỉ định qua biến môi trường:

```bash
# Windows (mặc định)
# C:\Users\Public\Documents\<video_id>.mp4

# Linux / macOS — đặt biến môi trường
export VIDEO_DIR=/path/to/your/videos
```

### Bước 6 — Chạy server

```bash
python run_fastapi.py
```

Truy cập:
- 🌐 Giao diện: http://localhost:8000
- 📖 Swagger API: http://localhost:8000/docs

---

## 🔧 Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `VIDEO_DIR` | `C:\Users\Public\Documents` | Thư mục chứa video gốc |
| `PORT` | `8000` | Cổng FastAPI |
| `HOST` | `0.0.0.0` | Host bind |
| `ENV` | `production` | `production` tắt hot-reload, `development` bật |

---

## 🔍 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Giao diện tìm kiếm |
| `POST` | `/api/search` | Tìm kiếm video (Standard / Temporal) |
| `GET` | `/api/keyframe/{video_id}/{img}` | Lấy ảnh keyframe |
| `GET` | `/api/video/{video_id}` | Stream video (HTTP Range) |
| `POST` | `/api/convert_time` | Chuyển timestamp → frame index |
| `POST` | `/api/calculate_frame` | Tính frame index từ thời điểm dừng video |
| `GET` | `/api/fps/{video_id}` | Lấy FPS của video |

### Ví dụ request tìm kiếm

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "người phụ nữ đang nấu ăn trong bếp",
    "search_type": "standard",
    "retrieval_mode": "multi",
    "top_k": 20
  }'
```

---

## 🧠 Kiến trúc Pipeline

```
Query (tiếng Việt / Anh)
    │
    ▼
[Auto Translate] ──── deep-translator (Google)
    │
    ▼
[CLIP Retrieval] ─────┬── SigLIP2 (HuggingFace)
                      └── DFN5B ViT-H/14 (OpenCLIP)
    │
    ▼
[Fusion] ──────────── RRF (Reciprocal Rank Fusion)
    │
    ▼
[Mapping] ─────────── keyframes_mapping.json → thông tin keyframe
    │
    ▼
[Reranking] ──────────┬── OCR score (EasyOCR / PaddleOCR)
                      ├── ASR score (transcript)
                      ├── Object score (YOLO detections)
                      ├── Metadata score (BM25)
                      └── CLIP score
    │
    ▼
Kết quả trả về (JSON)
```

---

## 📄 License

Dự án phục vụ nghiên cứu cho **AI Challenge (AIC) 2026**.
