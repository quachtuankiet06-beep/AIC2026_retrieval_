
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from deep_translator import GoogleTranslator
import faiss
import clip  # Thư viện openai-clip cho ViT-B/32
import open_clip
_SIGLIP2_CACHE = {
    "processor": None,
    "model": None
}

_DFN5B_CACHE = {
    "model": None,
    "tokenizer": None
}

# 1. Tự động nhận diện nếu có GPU, nếu không thì dùng CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# NEW
_FAISS_CACHE = {}
def get_faiss_index(index_path):
    """
    Load FAISS index một lần duy nhất.
    Những lần sau sẽ lấy từ RAM.
    """

    index_path = str(Path(index_path).resolve())

    if index_path not in _FAISS_CACHE:

        print(f"[INFO] Loading FAISS index:\n{index_path}")

        _FAISS_CACHE[index_path] = faiss.read_index(index_path)

    return _FAISS_CACHE[index_path]
def expand_and_translate_query(query_text):
    """
    Tự động dịch query tiếng Việt sang tiếng Anh và chuẩn hóa để tối ưu cho mô hình.
    """
    try:
        translated = GoogleTranslator(source='auto', target='english').translate(query_text)
        print(f"[INFO] Query Expansion: '{query_text}' -> '{translated}'")
        return translated if translated else query_text
    except Exception as e:
        print(f"[WARNING] Lỗi dịch query ({e}), giữ nguyên query gốc.")
        return query_text

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================================
# 1. SIGLIP 2 PIPELINE
# ==========================================================

_SIGLIP2_CACHE = {"processor": None, "model": None}

def get_siglip2_model(device="cuda"):
    if _SIGLIP2_CACHE["model"] is None:
        model_name = "google/siglip2-base-patch16-256"
        print(f"[INFO] Loading SigLIP 2 ({model_name})...")
        _SIGLIP2_CACHE["processor"] = AutoProcessor.from_pretrained(model_name)
        _SIGLIP2_CACHE["model"] = AutoModel.from_pretrained(model_name).to(device)
        _SIGLIP2_CACHE["model"].eval()
    return _SIGLIP2_CACHE["processor"], _SIGLIP2_CACHE["model"]

def siglip2_retrieval_pipeline(query_text, index_path, config_path):
    query_text = expand_and_translate_query(query_text)
    cfg = load_config(config_path)

    top_k = cfg.get("top_k", 100)
    threshold = cfg.get("retrieval_threshold", 0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = get_siglip2_model(device)
    
    max_length = model.config.text_config.max_position_embeddings
    inputs = processor(
        text=[query_text], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        text_outputs = model.get_text_features(**inputs)
        if hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        elif hasattr(text_outputs, "last_hidden_state"):
            text_features = text_outputs.last_hidden_state.mean(dim=1)
        else:
            text_features = text_outputs

        text_features = F.normalize(text_features, p=2, dim=1)

    query_embedding = text_features.cpu().numpy().astype(np.float32)

    print("[INFO] Loading SigLIP 2 FAISS index...")
    index = get_faiss_index(index_path)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
        rank += 1

    print(f"[INFO] SigLIP 2 Retrieval returned {len(results)} candidates.")
    return results


# ==========================================================
# 2. CLIP ViT-B/32 PIPELINE (MỚI BỔ SUNG)
# ==========================================================

# _CLIP_CACHE = {"model": None}

# def get_clip_b32_model(device="cuda"):
#     if _CLIP_CACHE["model"] is None:
#         print("[INFO] Loading OpenAI CLIP ViT-B/32...")
#         model, _ = clip.load("ViT-B/32", device=device)
#         model.eval()
#         _CLIP_CACHE["model"] = model
#     return _CLIP_CACHE["model"]

# def clip_b32_retrieval_pipeline(query_text, index_path, config_path):
#     query_text = expand_and_translate_query(query_text)
#     cfg = load_config(config_path)

#     top_k = cfg.get("top_k", 100)
#     threshold = cfg.get("retrieval_threshold", 0.0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = get_clip_b32_model(device)
    
#     # Tokenize query theo chuẩn CLIP
#     try:
#         text_tokens = clip.tokenize([query_text], truncate=True).to(device)
#     except TypeError:
#         # Phòng hờ version clip cũ không hỗ trợ tham số truncate
#         words = query_text.split()
#         if len(words) > 55:
#             query_text = " ".join(words[:55])
#         text_tokens = clip.tokenize([query_text]).to(device)

#     with torch.no_grad():
#         text_features = model.encode_text(text_tokens)
#         # BẮT BUỘC chuẩn hóa L2 tuyệt đối cho CLIP ViT-B/32
#         text_features = F.normalize(text_features, p=2, dim=-1)

#     query_embedding = text_features.cpu().numpy().astype(np.float32)

#     print("[INFO] Loading CLIP ViT-B/32 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] CLIP ViT-B/32 Retrieval returned {len(results)} candidates.")
#     return results
# ==========================================================
# 3. DFN5B-CLIP-ViT-H-14 PIPELINE (MỚI BỔ SUNG)
# ==========================================================

_DFN5B_CACHE = {"model": None, "tokenizer": None}

def get_dfn5b_vit_h14_model(device="cuda"):
    if _DFN5B_CACHE["model"] is None:
        model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
        print(f"[INFO] Loading DFN5B-CLIP-ViT-H-14 ({model_name})...")
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        
        _DFN5B_CACHE["model"] = model
        _DFN5B_CACHE["tokenizer"] = tokenizer
    return _DFN5B_CACHE["model"], _DFN5B_CACHE["tokenizer"]

def dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path):
    query_text = expand_and_translate_query(query_text)
    cfg = load_config(config_path)

    top_k = cfg.get("top_k", 100)
    threshold = cfg.get("retrieval_threshold", 0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = get_dfn5b_vit_h14_model(device)
    
    # Tokenize câu truy vấn theo chuẩn open_clip
    text_tokens = tokenizer([query_text]).to(device)

    with torch.no_grad(), torch.amp.autocast(device):
        text_features = model.encode_text(text_tokens)
        # BẮT BUỘC chuẩn hóa L2 tuyệt đối để tương thích IndexFlatIP (Cosine Similarity)
        text_features = F.normalize(text_features, p=2, dim=-1)

    query_embedding = text_features.cpu().float().numpy().astype(np.float32)

    print("[INFO] Loading DFN5B-CLIP-ViT-H-14 FAISS index...")
    index = faiss.read_index(index_path)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
        rank += 1

    print(f"[INFO] DFN5B-CLIP-ViT-H-14 Retrieval returned {len(results)} candidates.")
    return results


# ==========================================================
# Main Execution Block (Ví dụ kiểm thử)
# ==========================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Đường dẫn trỏ tới file index của DFN5B-CLIP-ViT-H-14 mà bạn vừa tạo
    DFN5B_INDEX = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index"
    CONFIG_PATH = BASE_DIR / "configs" / "retrieval.yaml"

    query = "a person walking"

    results = dfn5b_vit_h14_retrieval_pipeline(
        query_text=query,
        index_path=str(DFN5B_INDEX),
        config_path=str(CONFIG_PATH),
    )

    print("=" * 60)
    for r in results[:5]:
        print(r)