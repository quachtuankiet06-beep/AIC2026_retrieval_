# Dataset Structure

## Overview

AIC 2026 preliminary dataset consists of the following components:

- Keyframes
- CLIP Features
- Mapping CSV
- Object Detection Results
- Metadata

Each component serves a different purpose in the retrieval pipeline.

---

# Directory Structure

dataset/

    keyframes/
        keyframes_L01/
        keyframes_L02/
        ...

    clip_features/
        L01/
        L02/
        ...

    maps/
        L01/
        L02/
        ...

    metadata/
        L01/
        L02/
        ...

    objects/
        L01/
        L02/

---

# Keyframes

Each dataset folder contains multiple videos.

Example

keyframes_L01/

    L01_V001/
        0000.jpg
        0001.jpg
        ...

    L01_V002/
        ...

Each image corresponds to one extracted keyframe.

---

# CLIP Features

Each video has one feature file.

Example

L01_V001.npy

Shape

(number_of_keyframes, embedding_dimension)

Example

(523, 512)

The i-th embedding corresponds to the i-th keyframe.

---

# Mapping CSV

Each video contains one mapping file.

Purpose

Convert keyframe index into original video frame index.

Example

n, pts_time, fps, frame_idx

1,0,30,0

2,3,30,90

...

Used only during submission.

---

# Metadata

Each video contains one metadata file.

Contains

- title
- description
- keywords
- publish date
- channel

Used during metadata re-ranking.

---

# Object Detection

Each keyframe has one JSON file.

Example

0000.json

Contains

Detected objects

Bounding boxes

Confidence scores

Used during object re-ranking.

---

# Relationship

One Video

│

├── Keyframes

├── CLIP Feature

├── Metadata

└── Mapping CSV

Each Keyframe

│

├── Image

├── Embedding

└── Object JSON