# 03. Project Structure

## Overview

This document describes the directory organization of the AIC 2026 Video Retrieval System.

The project follows a modular architecture where each directory has a single responsibility. This design improves maintainability, extensibility, and collaboration.

---

# Root Structure

```
project/

├── configs/
├── data/
├── docs/
├── src/
└── requirements.txt
```

---

# configs/

```
configs/

    config.yaml
```

Purpose

Store all configurable parameters of the system.

Examples

- dataset paths
- FAISS index path
- retrieval Top-K
- CLIP model
- re-ranking weights

The source code should never hardcode these values.

---

# data/

```
data/

    clip_features/

    keyframes/

    maps/

    metadata/

    objects/
```

This directory stores all datasets provided by the competition.

## clip_features/

Contains CLIP embeddings extracted by the organizers.

Example

```
L01/
    L01_V001.npy
    L01_V002.npy
```

Each row corresponds to one keyframe embedding.

---

## keyframes/

Contains extracted keyframe images.

Example

```
keyframes_L01/

    L01_V001/

        0000.jpg

        0001.jpg
```

---

## maps/

Contains CSV files mapping keyframes to original video frames.

Used when generating the final submission.

---

## metadata/

Contains metadata of each video.

Typical fields include

- title
- description
- keywords
- publish date

Used during metadata re-ranking.

---

## objects/

Contains object detection results for every keyframe.

Each keyframe has one JSON file.

Used during object re-ranking.

---

# docs/

Project documentation.

Current documents

```
01_dataset_structure.md

02_system_design.md

03_project_structure.md
```

---

# src/

Contains all source code.

Modules are organized according to the retrieval pipeline.

```
src/

    common/

    evaluation/

    indexing/

    retrieval/

    reranking/

    submission/
```

---

## common/

Shared utilities.

Examples

- logger
- file IO
- constants
- helper functions

---

## indexing/

Responsible for building offline indexes.

Typical tasks

- Build FAISS index
- Build Mapping Database

---

## retrieval/

Responsible for candidate retrieval.

Main responsibilities

- Encode text query
- Search FAISS
- Return Top-K candidates

---

## reranking/

Improve candidate ranking.

Current Version

- Object Matching
- Metadata Matching

Future versions may include

- BM25
- Metadata Embedding
- OCR
- Caption Matching

---

## evaluation/

Evaluation scripts.

Metrics include

- Recall@1
- Recall@5
- Recall@20
- Recall@50
- Recall@100

---

## submission/

Generate submission files.

Convert retrieved keyframes into

(video_id, frame_idx)

following the competition format.

---

# Design Principles

The project follows several principles.

1. Keep data and source code separated.

2. Each module has a single responsibility.

3. Offline preprocessing is preferred over online computation.

4. Components should be independently replaceable.

5. The retrieval pipeline should be easy to extend without changing the overall architecture.