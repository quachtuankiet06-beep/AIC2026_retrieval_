import os
import json
import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# ==========================================================
# Cấu hình đường dẫn
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  
DATA_DIR = BASE_DIR / "data"

# Trỏ thẳng đến file keyframes_mapping.json chứa thông tin OCR của bạn
OCR_SOURCE_PATH = DATA_DIR / "indexes" / "keyframes_mapping.json" 

INDEX_OUTPUT_PATH = DATA_DIR / "indexes" / "ocr.index"
MAPPING_OUTPUT_PATH = DATA_DIR / "indexes" / "ocr_mapping.json"

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
MAX_LENGTH = 512

def build_ocr_index():
    print(f"[INFO] Loading model {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if not os.path.exists(OCR_SOURCE_PATH):
        raise FileNotFoundError(f"Không tìm thấy file nguồn tại: {OCR_SOURCE_PATH}.")

    print(f"[INFO] Loading data from {OCR_SOURCE_PATH}...")
    with open(OCR_SOURCE_PATH, "r", encoding="utf-8") as f:
        keyframes_data = json.load(f)

    texts_to_embed = []
    mapping_list = []

    print("[INFO] Extracting and preparing OCR texts from keyframes...")
    for item in keyframes_data:
        video_id = item.get("video_id")
        keyframe_index = item.get("keyframe_index")
        
        # Lấy mảng ocr_texts từ object (phòng hờ trường hợp không có thì gán list rỗng)
        ocr_texts_list = item.get("ocr_texts", [])
        
        # Gom các đoạn text OCR lại thành một chuỗi duy nhất cho keyframe này (nối bằng khoảng trắng)
        if ocr_texts_list and isinstance(ocr_texts_list, list):
            combined_ocr_text = " ".join([str(t).strip() for t in ocr_texts_list if str(t).strip()])
        else:
            combined_ocr_text = ""

        # Chuẩn bị định dạng theo chuẩn E5 document passage
        formatted_text = f"passage: {combined_ocr_text}"
        texts_to_embed.append(formatted_text)

        # Lưu mapping tương ứng để sau này tra cứu ngược lại
        mapping_list.append({
            "vector_index": len(mapping_list),
            "video_id": video_id,
            "keyframe_index": keyframe_index,
            "ocr_text": combined_ocr_text
        })

    print(f"[INFO] Encoding {len(texts_to_embed)} OCR documents using E5...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE)):
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # L2 Normalization để dùng Inner Product (Cosine Similarity)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    import numpy as np
    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    dimension = embeddings_matrix.shape[1]

    print(f"[INFO] Building FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # Lưu index và file mapping OCR
    os.makedirs(INDEX_OUTPUT_PATH.parent, exist_ok=True)
    faiss.write_index(index, str(INDEX_OUTPUT_PATH))
    print(f"[INFO] Saved FAISS index to {INDEX_OUTPUT_PATH}")

    with open(MAPPING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_list, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Saved OCR mapping to {MAPPING_OUTPUT_PATH}")
    print("[INFO] Done!")

if __name__ == "__main__":
    build_ocr_index()