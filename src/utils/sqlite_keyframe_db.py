import sqlite3
import json
from pathlib import Path


class KeyframeDatabase:
    """
    SQLite interface for keyframes.db

    Supports:

        db.get(vector_index)

        db.get_batch(vector_indices)

        db.close()

    Automatically converts JSON TEXT columns back to Python objects.
    """

    def __init__(self, db_path):

        self.db_path = str(Path(db_path).resolve())

        self.conn = sqlite3.connect(self.db_path)

        self.conn.row_factory = sqlite3.Row

    # ======================================================
    # PRIVATE
    # ======================================================

    @staticmethod
    def _row_to_dict(row):

        if row is None:
            return None

        item = dict(row)

        json_fields = [
            "object_entities",
            "metadata",
            "ocr_texts",
        ]

        for field in json_fields:

            value = item.get(field)

            if value is None:
                continue

            try:
                item[field] = json.loads(value)

            except Exception:
                pass

        return item

    # ======================================================
    # SINGLE LOOKUP
    # ======================================================

    def get(
        self,
        vector_index,
    ):

        cursor = self.conn.execute(

            """
            SELECT *
            FROM keyframes
            WHERE vector_index = ?
            """,

            (
                int(vector_index),
            )

        )

        row = cursor.fetchone()

        return self._row_to_dict(row)

    # ======================================================
    # BATCH LOOKUP
    # ======================================================

    def get_batch(
        self,
        vector_indices,
    ):

        if len(vector_indices) == 0:
            return {}

        placeholders = ",".join(
            ["?"] * len(vector_indices)
        )

        query = f"""

        SELECT *
        FROM keyframes
        WHERE vector_index IN ({placeholders})

        """

        cursor = self.conn.execute(

            query,

            [
                int(v)
                for v in vector_indices
            ]

        )

        rows = cursor.fetchall()

        records = {}

        for row in rows:

            item = self._row_to_dict(row)

            records[
                item["vector_index"]
            ] = item

        return records

    # ======================================================
    # EXIST
    # ======================================================

    def exists(
        self,
        vector_index,
    ):

        cursor = self.conn.execute(

            """
            SELECT 1
            FROM keyframes
            WHERE vector_index=?
            LIMIT 1
            """,

            (
                int(vector_index),
            )

        )

        return cursor.fetchone() is not None

    # ======================================================
    # COUNT
    # ======================================================

    def count(self):

        cursor = self.conn.execute(

            """
            SELECT COUNT(*)
            FROM keyframes
            """

        )

        return cursor.fetchone()[0]

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        self.conn.close()

    # ======================================================
    # CONTEXT MANAGER
    # ======================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):

        self.close()


# ==========================================================
# GLOBAL CACHE
# ==========================================================

_DB_CACHE = {}


def get_keyframe_database(db_path):

    db_path = str(Path(db_path).resolve())

    if db_path not in _DB_CACHE:

        _DB_CACHE[db_path] = KeyframeDatabase(
            db_path
        )

    return _DB_CACHE[db_path]


# ==========================================================
# DEBUG
# ==========================================================

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]

    DB_PATH = (
        ROOT /
        "data" /
        "indexes" /
        "keyframes.db"
    )

    db = get_keyframe_database(DB_PATH)

    print("=" * 70)
    print("Database Statistics")
    print("=" * 70)

    print(
        "Total Records:",
        db.count()
    )

    print()

    print("=" * 70)
    print("Single Lookup")
    print("=" * 70)

    record = db.get(40)

    print(record["video_id"])
    print(record["frame_idx"])
    print(record["keyframe_path"])

    print()

    print("=" * 70)
    print("Batch Lookup")
    print("=" * 70)

    records = db.get_batch(
        [
            40,
            41,
            42,
        ]
    )

    for idx, rec in records.items():

        print(
            idx,
            rec["video_id"],
            rec["frame_idx"],
        )