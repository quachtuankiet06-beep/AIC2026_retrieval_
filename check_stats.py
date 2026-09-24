import sys
import os
import faiss
import sqlite3
import json
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

idx_dir = Path("data/indexes")

print("=" * 70)
print("FAISS INDEX STATS:")
print("=" * 70)
for f in sorted(idx_dir.glob("*.index")):
    try:
        idx = faiss.read_index(str(f))
        print(f"{f.name:<45}: {idx.ntotal:>10,} vectors (dim={idx.d})")
    except Exception as e:
        print(f"{f.name:<45}: ERROR ({e})")

print("\n" + "=" * 70)
print("SQLITE (.db) STATS:")
print("=" * 70)
for f in sorted(idx_dir.glob("*.db")):
    try:
        conn = sqlite3.connect(str(f))
        cnt = conn.cursor().execute("SELECT COUNT(1) FROM keyframes").fetchone()[0]
        print(f"{f.name:<45}: {cnt:>10,} rows")
    except Exception as e:
        print(f"{f.name:<45}: ERROR ({e})")

print("\n" + "=" * 70)
print("JSON MAPPING STATS:")
print("=" * 70)
for f in sorted(idx_dir.glob("*.json")):
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            print(f"{f.name:<45}: {len(data):>10,} items (type={type(data).__name__})")
    except Exception as e:
        print(f"{f.name:<45}: ERROR ({e})")
