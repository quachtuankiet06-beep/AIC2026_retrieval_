import json
from pathlib import Path


# ==========================================================
# Load Mapping Database
# ==========================================================

_MAPPING_CACHE = {}

def load_mapping_database(mapping_path):
    mapping_path = str(Path(mapping_path).resolve())

    if mapping_path in _MAPPING_CACHE:
        return _MAPPING_CACHE[mapping_path]

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_list = json.load(f)

    mapping_dict = {
        item["vector_index"]: item
        for item in mapping_list
    }

    _MAPPING_CACHE[mapping_path] = mapping_dict

    return mapping_dict


# ==========================================================
# Mapping Pipeline
# ==========================================================

def mapping_pipeline(
    retrieval_results,
    mapping_path,
):
    """
    Parameters
    ----------
    retrieval_results

    [
        {
            "rank":1,
            "vector_index":1523,
            "score":0.92
        }
    ]

    Returns
    -------
    candidate_list

    [
        {
            "rank":1,
            "score":0.92,
            ...
            video_id
            frame_idx
            metadata
            object_entities
        }
    ]
    """

    if len(retrieval_results) == 0:
        print("[INFO] Empty retrieval results.")
        return []

    mapping_db = load_mapping_database(mapping_path)

    candidate_list = []

    missing = 0

    for item in retrieval_results:

        vector_index = item["vector_index"]

        if vector_index not in mapping_db:
            missing += 1
            continue

        mapping_record = mapping_db[vector_index]
        ##################################################################
        # Chuẩn hóa lại ocr_texts để đảm bảo đồng bộ định dạng với reranking_pipeline
        raw_ocr = mapping_record.get("ocr_texts", [])
        normalized_ocr = []
        
        for ocr_item in raw_ocr:
            if isinstance(ocr_item, dict):
                # Đã chuẩn dạng mới {"text": ..., "score": ...}
                normalized_ocr.append({
                    "text": str(ocr_item.get("text", "")).strip().lower(),
                    "score": float(ocr_item.get("score", 1.0))
                })
            elif isinstance(ocr_item, str):
                # Hỗ trợ tương thích ngược nếu record cũ còn lưu dạng string thô
                text_val = str(ocr_item).strip().lower()
                if text_val:
                    normalized_ocr.append({
                        "text": text_val,
                        "score": 1.0
                    })
        #################################################################
        candidate = {

            # Retrieval information
            "rank": item.get("rank", 1),
            "vector_index": vector_index,
            "score": item["score"],

            # Mapping information
            **mapping_record
        }

        candidate_list.append(candidate)

    if missing > 0:
        print(f"[WARNING] Missing {missing} vector indices.")

    print(f"[INFO] Generated {len(candidate_list)} candidates.")

    return candidate_list


# ==========================================================
# Debug
# ==========================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    MAPPING_PATH = (
        BASE_DIR /
        "data" /
        "indexes" /
        "keyframes_mapping_siglib2.json"
    )

    fake_retrieval = [

        {
            "rank": 1,
            "vector_index": 0,
            "score": 0.97
        },

        {
            "rank": 2,
            "vector_index": 25,
            "score": 0.95
        },

        {
            "rank": 3,
            "vector_index": 80,
            "score": 0.94
        }

    ]

    candidates = mapping_pipeline(
        retrieval_results=fake_retrieval,
        mapping_path=str(MAPPING_PATH)
    )

    print("\n")

    print("=" * 70)
    print("TOP 3 CANDIDATES")
    print("=" * 70)

    for candidate in candidates[:3]:

        print(f"Rank            : {candidate['rank']}")
        print(f"Score           : {candidate['score']:.4f}")
        print(f"Video           : {candidate['video_id']}")
        print(f"Frame           : {candidate['frame_idx']}")
        print(f"Keyframe Index  : {candidate['keyframe_index']}")
        print(f"PTS Time        : {candidate['pts_time']}")
        print(f"Image           : {candidate['keyframe_path']}")
        print(f"Objects         : {candidate['object_entities'][:5]}")
        print("-" * 70)

