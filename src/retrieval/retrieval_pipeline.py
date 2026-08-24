
import yaml
import re
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from deep_translator import GoogleTranslator
import faiss
import clip  # Thư viện openai-clip cho ViT-B/32
import open_clip
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time


# Cache cho NLLB-200 để không phải load đi load lại nặng VRAM
_NLLB_CACHE = {"model": None, "tokenizer": None}
def get_nllib_translator(device="cpu"):
    """
    Load NLLB-200 chạy hoàn toàn trên CPU để tiết kiệm VRAM cho các model Vision.
    """
    if _NLLB_CACHE["model"] is None:
        model_name = "facebook/nllb-200-distilled-1.3B"
        print(f"[INFO] Loading Fallback Local Translator ({model_name}) on CPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Force load model sang CPU với float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        
        _NLLB_CACHE["tokenizer"] = tokenizer
        _NLLB_CACHE["model"] = model
    return _NLLB_CACHE["tokenizer"], _NLLB_CACHE["model"]


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

def decompose_standard_narrative_query(query_text):
    """
    Phân rã câu truy vấn văn xuôi / có cấu trúc chuỗi thời gian trong Standard Query thành:
    - q_full: Câu truy vấn đầy đủ
    - q_main: Mệnh đề / hành động trực quan chính (Anchor)
    - q_context: Mệnh đề diễn biến bổ trợ (Context Clue) nếu có
    """
    if not query_text or not isinstance(query_text, str):
        return [query_text]

    # Các từ nối chỉ mốc thời gian / diễn biến thường gặp trong đề KIS
    temporal_splitters = [
        r"(?i)\bbiết sau đó\b",
        r"(?i)\bvà sau đó\b",
        r"(?i)\bsau đó\b",
        r"(?i)\btiếp theo\b",
        r"(?i)\bkế tiếp\b",
        r"(?i)\bđoạn sau\b",
        r"(?i)\brồi sau đó\b"
    ]
    
    # 1. Thử tách theo từ nối thời gian
    for pattern in temporal_splitters:
        parts = re.split(pattern, query_text)
        if len(parts) > 1 and len(parts[0].strip()) > 10:
            q_main = parts[0].strip().rstrip(".,; ")
            q_context = " ".join([p.strip() for p in parts[1:] if p.strip()])
            print(f"[INFO] Narrative Query Decomposition (Standard Search):")
            print(f"  -> Q_Full   : '{query_text}'")
            print(f"  -> Q_Anchor : '{q_main}'")
            print(f"  -> Q_Context: '{q_context}'")
            return [query_text, q_main, q_context]
            
    # 2. Thử tách theo dấu câu nếu câu dài chứa nhiều mệnh đề
    sentences = [s.strip() for s in re.split(r"[.\n]+", query_text) if len(s.strip()) > 10]
    if len(sentences) >= 2 and len(query_text.split()) > 15:
        q_main = sentences[0]
        q_context = " ".join(sentences[1:])
        print(f"[INFO] Narrative Query Decomposition (Standard Search):")
        print(f"  -> Q_Full   : '{query_text}'")
        print(f"  -> Q_Anchor : '{q_main}'")
        print(f"  -> Q_Context: '{q_context}'")
        return [query_text, q_main, q_context]
        
    return [query_text]

# def expand_and_translate_query(query_text):
#     """
#     Tự động dịch query tiếng Việt sang tiếng Anh và chuẩn hóa để tối ưu cho mô hình.
#     """
#     try:
#         translated = GoogleTranslator(source='vi', target='en').translate(query_text)
#         print(f"[INFO] Query Expansion: '{query_text}' -> '{translated}'")
#         return translated if translated else query_text
#     except Exception as e:
#         print(f"[WARNING] Lỗi dịch query ({e}), giữ nguyên query gốc.")
#         return query_text
# import time

def expand_and_translate_query(query_text):
    """
    Chiến thuật Fallback Translator 3 tầng:
    Tầng 1: Google Translate (Chủ đạo)
    Tầng 2: NLLB-200 Distilled 1.3B (Local Offline)
    Tầng 3: Giữ nguyên query gốc
    """
    if not query_text or not query_text.strip():
        return query_text

    # # --- TẦNG 1: Thử Google Translate trước ---
    # try:
    #     translator = GoogleTranslator(source='vi', target='en')
    #     translated = translator.translate(query_text)
        
    #     if translated and translated.strip():
    #         print(f"[INFO] Google Translate Success: '{query_text}' -> '{translated}'")
    #         time.sleep(1) # Nghỉ nhẹ chống spam
    #         return translated
    # except Exception as e:
    #     print(f"[WARNING] Google Translate thất bại ({e}), chuyển sang Local Fallback...")

    # --- TẦNG 2: Fallback sang NLLB-200 Distilled 1.3B ---
    try:
        import gc
        import torch

        device = "cpu"
        tokenizer, model = get_nllib_translator(device)
        
        # NLLB yêu cầu chỉ định rõ mã ngôn ngữ (Tiếng Việt: vie_Latn, Tiếng Anh: eng_Latn)
        tokenizer.src_lang = "vie_Latn"
        inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), 
                max_new_tokens=128
            )
            
        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        # GIẢI PHÓNG TRIỆT ĐỂ KHỎI RAM NGAY LẬP TỨC SAU KHI DỊCH XONG
        del model
        del tokenizer
        del inputs
        del translated_tokens
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if translated and translated.strip():
            print(f"[INFO] NLLB-200 Fallback Success: '{query_text}' -> '{translated}'")
            return translated
            
    except Exception as e:
        print(f"[WARNING] NLLB-200 Fallback cũng thất bại ({e}).")
    print(f"[WARNING] Giữ nguyên query gốc: '{query_text}'")
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
    sub_queries = decompose_standard_narrative_query(query_text)
    translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

    cfg = load_config(config_path)
    top_k = cfg.get("top_k", 100)
    threshold = cfg.get("retrieval_threshold", 0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = get_siglip2_model(device)
    max_length = model.config.text_config.max_position_embeddings

    # Tokenize và trích xuất embedding cho từng sub-query
    embeddings_list = []
    with torch.no_grad():
        for t_query in translated_sub_queries:
            inputs = processor(
                text=[t_query], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            text_outputs = model.get_text_features(**inputs)
            if hasattr(text_outputs, "pooler_output"):
                tf = text_outputs.pooler_output
            elif hasattr(text_outputs, "last_hidden_state"):
                tf = text_outputs.last_hidden_state.mean(dim=1)
            else:
                tf = text_outputs
            tf = F.normalize(tf, p=2, dim=1)
            embeddings_list.append(tf)

    # Gộp trọng số nếu có Sub-query: 50% Anchor chính, 35% Full query, 15% Context phụ
    if len(embeddings_list) == 3:
        combined_tf = 0.35 * embeddings_list[0] + 0.50 * embeddings_list[1] + 0.15 * embeddings_list[2]
        combined_tf = F.normalize(combined_tf, p=2, dim=1)
    elif len(embeddings_list) == 2:
        combined_tf = 0.40 * embeddings_list[0] + 0.60 * embeddings_list[1]
        combined_tf = F.normalize(combined_tf, p=2, dim=1)
    else:
        combined_tf = embeddings_list[0]

    query_embedding = combined_tf.cpu().numpy().astype(np.float32)

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
    sub_queries = decompose_standard_narrative_query(query_text)
    translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

    cfg = load_config(config_path)
    top_k = cfg.get("top_k", 100)
    threshold = cfg.get("retrieval_threshold", 0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = get_dfn5b_vit_h14_model(device)

    embeddings_list = []
    with torch.no_grad(), torch.amp.autocast(device):
        for t_query in translated_sub_queries:
            text_tokens = tokenizer([t_query]).to(device)
            tf = model.encode_text(text_tokens)
            tf = F.normalize(tf, p=2, dim=-1)
            embeddings_list.append(tf)

    if len(embeddings_list) == 3:
        combined_tf = 0.35 * embeddings_list[0] + 0.50 * embeddings_list[1] + 0.15 * embeddings_list[2]
        combined_tf = F.normalize(combined_tf, p=2, dim=-1)
    elif len(embeddings_list) == 2:
        combined_tf = 0.40 * embeddings_list[0] + 0.60 * embeddings_list[1]
        combined_tf = F.normalize(combined_tf, p=2, dim=-1)
    else:
        combined_tf = embeddings_list[0]

    query_embedding = combined_tf.cpu().float().numpy().astype(np.float32)

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