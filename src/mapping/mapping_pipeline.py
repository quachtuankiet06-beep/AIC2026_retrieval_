from pathlib import Path

from src.utils.sqlite_keyframe_db import get_keyframe_database


# ==========================================================
# Mapping Pipeline
# ==========================================================

def mapping_pipeline(
    retrieval_results,
    mapping_db_path,
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

    if not retrieval_results:
        print("[INFO] Empty retrieval results.")
        return []

    ##########################################################
    # Open SQLite database
    ##########################################################

    mapping_db = get_keyframe_database(
        mapping_db_path
    )

    ##########################################################
    # Batch lookup
    ##########################################################

    vector_indices = [

        item["vector_index"]

        for item in retrieval_results

    ]

    mapping_records = mapping_db.get_batch(
        vector_indices
    )

    ##########################################################

    candidate_list = []

    missing = 0

    for item in retrieval_results:

        vector_index = item["vector_index"]

        mapping_record = mapping_records.get(
            vector_index
        )

        if mapping_record is None:

            missing += 1

            continue

        ######################################################
        # Normalize OCR format
        ######################################################

        raw_ocr = mapping_record.get(
            "ocr_texts",
            []
        )

        normalized_ocr = []

        for ocr_item in raw_ocr:

            if isinstance(
                ocr_item,
                dict
            ):

                normalized_ocr.append(

                    {

                        "text":
                        str(
                            ocr_item.get(
                                "text",
                                ""
                            )
                        ).strip().lower(),

                        "score":
                        float(
                            ocr_item.get(
                                "score",
                                1.0
                            )
                        )

                    }

                )

            elif isinstance(
                ocr_item,
                str
            ):

                text_val = str(
                    ocr_item
                ).strip().lower()

                if text_val:

                    normalized_ocr.append(

                        {

                            "text":
                            text_val,

                            "score":
                            1.0

                        }

                    )

        mapping_record["ocr_texts"] = normalized_ocr

        ######################################################

        candidate = {

            "rank":
            item.get(
                "rank",
                1
            ),

            "vector_index":
            vector_index,

            "score":
            item["score"],

            **mapping_record

        }

        candidate_list.append(
            candidate
        )

    ##########################################################

    if missing > 0:

        print(
            f"[WARNING] Missing {missing} vector indices."
        )

    print(
        f"[INFO] Generated {len(candidate_list)} candidates."
    )

    return candidate_list


# ==========================================================
# Debug
# ==========================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]

    DB_PATH = (
        BASE_DIR
        / "data"
        / "indexes"
        / "keyframes.db"
    )

    fake_retrieval = [

        {

            "rank": 1,

            "vector_index": 40,

            "score": 0.97

        },

        {

            "rank": 2,

            "vector_index": 41,

            "score": 0.95

        },

        {

            "rank": 3,

            "vector_index": 42,

            "score": 0.94

        }

    ]

    candidates = mapping_pipeline(

        retrieval_results=fake_retrieval,

        mapping_db_path=DB_PATH

    )

    print()

    print("=" * 70)
    print("TOP CANDIDATES")
    print("=" * 70)

    for candidate in candidates:

        print(f"Rank            : {candidate['rank']}")
        print(f"Score           : {candidate['score']:.4f}")
        print(f"Video           : {candidate['video_id']}")
        print(f"Frame           : {candidate['frame_idx']}")
        print(f"Keyframe Index  : {candidate['keyframe_index']}")
        print(f"PTS Time        : {candidate['pts_time']}")
        print(f"Image           : {candidate['keyframe_path']}")
        print(f"Objects         : {candidate['object_entities'][:5]}")
        print("-" * 70)