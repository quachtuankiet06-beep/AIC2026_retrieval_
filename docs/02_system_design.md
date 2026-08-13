# AIC 2026 - Video Retrieval System V1

## 1. Overview

Mục tiêu của hệ thống là xây dựng một pipeline truy xuất keyframe từ truy vấn ngôn ngữ tự nhiên cho vòng sơ tuyển AIC 2026.

Triết lý thiết kế của Version 1:

- Simple
- Fast
- Modular
- Easy to Extend

Version đầu tiên chỉ tập trung vào:

- CLIP Retrieval
- Object Re-ranking
- Metadata Re-ranking

Các module phức tạp hơn (OCR, Captioning, Cross-Encoder, LLM, Learning-to-Rank...) sẽ được bổ sung ở các phiên bản sau.

---

# 2. System Architecture

Offline

CLIP Features
        │
        ▼
 Build FAISS Index

Metadata
Object
Map CSV
        │
        ▼
 Build Mapping Database


Online

Natural Language Query
        │
        ▼
CLIP ViT-B/32 Text Encoder
        │
        ▼
FAISS Retrieval
        │
      Top-K
        │
        ▼
Load Mapping Database
        │
        ├── Metadata
        ├── Object
        │
        ▼
Re-ranking
        │
        ▼
Top-N Results
        │
        ▼
(video_id, frame_idx)

---

# 3. Offline Components

## 3.1 Build FAISS Index

Input

- CLIP Features (.npy)

Output

- faiss.index

Description

- Load every CLIP feature from all videos.
- Merge into a single embedding matrix.
- Normalize embeddings.
- Build FAISS IndexFlatIP.

---

## 3.2 Build Mapping Database

Input

- Map CSV
- Metadata
- Object JSON

Output

- mapping.db

Each vector stored in FAISS corresponds to one mapping record.

Schema

{
    vector_index,
    video_id,
    keyframe_id,
    frame_idx,
    pts_time,
    fps,
    paths,
    metadata
}

---

# 4. Online Pipeline

## Step 1

Encode user query using

CLIP ViT-B/32

Output

Query Embedding

---

## Step 2

Search in FAISS

Return

Top-K candidates

Default

Top-100

---

## Step 3

Lookup Mapping

Retrieve

- frame_idx
- metadata
- object json path

---

## Step 4

Re-ranking

Compute

Image Score

+

Object Score

+

Metadata Score

↓

Final Score

Sort candidates again.

---

# 5. Re-ranking

## 5.1 CLIP Score

Cosine similarity returned by FAISS.

---

## 5.2 Object Score

Extract object keywords from query.

Compare with detected objects.

Simple Matching Ratio

matched_objects / query_objects

---

## 5.3 Metadata Score

Metadata fields

- title
- keywords
- description

Compute simple token overlap.

Example

Query

"HTV tin tức"

Metadata

"HTV Tin Tức Mới Nhất"

Metadata Score = High

---

## 5.4 Final Score

Version 1

FinalScore

=

0.8 × CLIP

+

0.1 × Object

+

0.1 × Metadata

Weights will be tuned later.

---

# 6. Future Improvements

Version 2

- BM25 Metadata
- Better Object Matching

Version 3

- Metadata Embedding

Version 4

- OCR

Version 5

- Caption Generation

Version 6

- Learning-to-Rank

---

# Design Principles

1. Keep retrieval simple.

2. Every module should be replaceable.

3. Offline preprocessing is preferred over online computation.

4. Re-ranking should remain lightweight.

5. Build a strong baseline before introducing complex models.