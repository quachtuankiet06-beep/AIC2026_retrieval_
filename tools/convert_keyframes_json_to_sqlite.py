import json
import sqlite3
from pathlib import Path
from tqdm import tqdm


# ==========================================================
# PATH
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

JSON_PATH = (
    ROOT /
    "data" /
    "indexes" /
    "keyframes_mapping_new.json"
)

SQLITE_PATH = (
    ROOT /
    "data" /
    "indexes" /
    "keyframes_new_kf.db"
)

BATCH_SIZE = 1000


# ==========================================================
# CREATE DATABASE
# ==========================================================

def create_database(conn):

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS keyframes(

        vector_index INTEGER PRIMARY KEY,

        video_id TEXT,

        keyframe_index INTEGER,

        frame_idx INTEGER,

        pts_time REAL,

        fps INTEGER,

        keyframe_path TEXT,

        object_path TEXT,

        metadata_path TEXT,

        object_entities TEXT,

        metadata TEXT,

        ocr_texts TEXT,

        asr_text TEXT

    )
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_video
    ON keyframes(video_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_frame
    ON keyframes(frame_idx)
    """)

    conn.commit()


# ==========================================================
# INSERT
# ==========================================================

INSERT_SQL = """
INSERT INTO keyframes(

    vector_index,
    video_id,
    keyframe_index,
    frame_idx,
    pts_time,
    fps,
    keyframe_path,
    object_path,
    metadata_path,
    object_entities,
    metadata,
    ocr_texts,
    asr_text

)

VALUES(

    ?,?,?,?,?,?,?,?,?,?,?,?,?

)
"""


def convert():

    if not JSON_PATH.exists():
        raise FileNotFoundError(JSON_PATH)

    print(f"[INFO] Loading JSON : {JSON_PATH}")

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    print(
        f"[INFO] Total records : {len(records):,}"
    )

    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()

    conn = sqlite3.connect(SQLITE_PATH)

    create_database(conn)

    cursor = conn.cursor()

    batch = []

    for item in tqdm(records):

        batch.append(

            (

                item["vector_index"],

                item["video_id"],

                item["keyframe_index"],

                item["frame_idx"],

                item["pts_time"],

                item["fps"],

                item["keyframe_path"],

                item["object_path"],

                item["metadata_path"],

                json.dumps(
                    item.get(
                        "object_entities",
                        []
                    ),
                    ensure_ascii=False
                ),

                json.dumps(
                    item.get(
                        "metadata",
                        {}
                    ),
                    ensure_ascii=False
                ),

                json.dumps(
                    item.get(
                        "ocr_texts",
                        []
                    ),
                    ensure_ascii=False
                ),

                item.get(
                    "asr_text",
                    ""
                )

            )

        )

        if len(batch) >= BATCH_SIZE:

            cursor.executemany(
                INSERT_SQL,
                batch
            )

            conn.commit()

            batch.clear()

    if batch:

        cursor.executemany(
            INSERT_SQL,
            batch
        )

        conn.commit()

    conn.close()

    print()

    print("=" * 60)

    print(
        "[SUCCESS] SQLite database created."
    )

    print(
        SQLITE_PATH
    )

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    convert()