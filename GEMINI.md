# General Project Rules
- Luôn trả lời bằng tiếng Việt.
- Ưu tiên viết code Python theo chuẩn PEP 8.
- Khi viết giải thuật hoặc sửa code, hãy giải thích ngắn gọn lý do trước khi đưa ra code block.
- Muốn biết tổng quan dự án hãy đọc README.md
# Agent Skills Integration

Hệ thống có các Agent Skills được định nghĩa tại thư mục `.agent/skills/`.
Khi thực hiện các tác vụ liên quan, hãy tuân thủ nghiêm ngặt các hướng dẫn và quy trình trong các file Skill tương ứng:

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