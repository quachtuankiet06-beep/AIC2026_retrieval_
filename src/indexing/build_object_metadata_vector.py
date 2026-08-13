import json
from pathlib import Path
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Cấu hình thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Sử dụng thiết bị: {device}")

# Load mô hình multilingual-e5-base
model_name = "intfloat/multilingual-e5-base"
print(f"[INFO] Đang tải mô hình {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MAPPING_PATH = BASE_DIR / "data" / "indexes" / "keyframes_mapping.json"
def average_pool(
    last_hidden_state,
    attention_mask
):

    last_hidden = last_hidden_state.masked_fill(
        ~attention_mask[..., None].bool(),
        0.0
    )

    return last_hidden.sum(dim=1) / attention_mask.sum(
        dim=1
    )[..., None]
    
def build_faiss_index_incrementally(text_list, dimension=768, batch_size=64, desc="Encoding"):
    """Encode theo batch và add thẳng vào FAISS Index để tiết kiệm RAM tối đa"""
    print(f"[INFO] Bắt đầu xây dựng FAISS Index cho {len(text_list)} items...")
    
    index = faiss.IndexFlatIP(dimension)
    formatted_texts = [f"passage: {t}" for t in text_list if t.strip()]
    
    for i in tqdm(range(0, len(formatted_texts), batch_size), desc=desc):
        batch = formatted_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = average_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Chuyển sang numpy float32 và add thẳng vào FAISS index
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            index.add(embeddings_np)
            
    return index

def main():
    if not MAPPING_PATH.exists():
        print(f"[ERROR] Không tìm thấy file mapping tại: {MAPPING_PATH}")
        return

    print(f"[INFO] Đang đọc file mapping từ {MAPPING_PATH}...")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    object_texts_to_embed = []
    object_mapping = []

    metadata_texts_to_embed = []
    metadata_mapping = []

    print("[INFO] Đang xử lý bóc tách dữ liệu Object và Metadata...")
    for item in tqdm(mapping_data, desc="Parsing JSON"):
        video_id = item.get("video_id", "")
        kf_idx = item.get("keyframe_index", 0)

        # 1. Xử lý OBJECT ENTITIES (Enrich để E5 hiểu ngữ cảnh)
        objects = item.get("object_entities", [])
        for obj in objects:
            # Biến đổi object đơn lẻ thành cụm ngữ cảnh ngắn
            enriched_obj = f"This image contains {obj}"
            object_texts_to_embed.append(enriched_obj)
            object_mapping.append({
                "vector_index": len(object_mapping),
                "video_id": video_id,
                "keyframe_index": kf_idx,
                "entity": obj
            })

        # 2. Xử lý METADATA (Merge các trường con)
        meta = item.get("metadata", {})
        if meta:
            title = meta.get("title", "")
            desc = meta.get("description", "")
            keywords = ", ".join(meta.get("keywords", []))
            
            # Gộp các trường thành một đoạn văn bản mô tả tổng quan thống nhất
            meta_text = f"Title: {title}. Description: {desc}. Keywords: {keywords}"
            
            metadata_texts_to_embed.append(meta_text)
            metadata_mapping.append({
                "vector_index": len(metadata_mapping),
                "video_id": video_id,
                "keyframe_index": kf_idx,
                "text": meta_text
            })

   # --- BUILD OBJECT INDEX ---
    obj_index_path = BASE_DIR / "data" / "indexes" / "object.index"
    obj_map_path = BASE_DIR / "data" / "indexes" / "object_mapping.json"
    
    # Tạo và add thẳng vào index không cần qua np.vstack RAM lớn
    object_index = build_faiss_index_incrementally(
        object_texts_to_embed, dimension = model.config.hidden_size, batch_size=64, desc="Encoding Objects"
    )
    
    faiss.write_index(object_index, str(obj_index_path))
    with open(obj_map_path, "w", encoding="utf-8") as f:
        json.dump(object_mapping, f, ensure_ascii=False, indent=4)
    print(f"[SUCCESS] Đã lưu thành công: {obj_index_path} và {obj_map_path}")

    # --- BUILD METADATA INDEX ---
    meta_index_path = BASE_DIR / "data" / "indexes" / "metadata.index"
    meta_map_path = BASE_DIR / "data" / "indexes" / "metadata_mapping.json"
    
    metadata_index = build_faiss_index_incrementally(
        metadata_texts_to_embed, dimension = model.config.hidden_size, batch_size=64, desc="Encoding Metadata"
    )
    
    faiss.write_index(metadata_index, str(meta_index_path))
    with open(meta_map_path, "w", encoding="utf-8") as f:
        json.dump(metadata_mapping, f, ensure_ascii=False, indent=4)
    print(f"[SUCCESS] Đã lưu thành công: {meta_index_path} và {meta_map_path}")
if __name__ == "__main__":
    main()