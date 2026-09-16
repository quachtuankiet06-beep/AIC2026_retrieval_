import os
import yaml
import json
import re
import unicodedata
import numpy as np
import faiss
import torch
import torch.nn.functional as F
import psutil
from rank_bm25 import BM25Plus
import faiss
from sentence_transformers import CrossEncoder
from .domain_booster import DomainKeywordBooster
booster = DomainKeywordBooster()


try:
    from rapidfuzz import fuzz
except ImportError:
    import difflib
    class DummyFuzz:
        @staticmethod
        def partial_ratio(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
        @staticmethod
        def token_set_ratio(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
    fuzz = DummyFuzz()

from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

# # Khởi tạo Cross-Encoder global

# _cross_encoder = None

# def get_bge_reranker():
#     global _cross_encoder
#     if _cross_encoder is None:
#         print("[INFO] Loading BAAI/bge-reranker-v2-m3...")
#         _cross_encoder = CrossEncoder(
#             "BAAI/bge-reranker-v2-m3", 
#             max_length=1024, # bge-v2-m3 hỗ trợ tối đa 8192, để 1024-2048 là dư xài cho OCR + Metadata
#             device= "cpu"
#         )
#     return _cross_encoder

def remove_vietnamese_diacritics(text: str) -> str:
    """Chuyển tiếng Việt có dấu thành không dấu và viết thường"""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', str(text))
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text.lower().strip()

def clean_ocr_text(text: str) -> str:
    """Xóa bỏ các ký tự đặc biệt, chỉ giữ lại chữ cái và số"""
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text)).strip()

def print_ram(stage):
    process = psutil.Process(os.getpid())

    rss = process.memory_info().rss / 1024 / 1024

    print(
        f"[RAM] {stage:<35} : {rss:.2f} MB"
    )

def topk_average(scores, k=3):

    if not scores:
        return 0.0

    scores = sorted(scores, reverse=True)

    k = min(k, len(scores))

    return float(np.mean(scores[:k]))
# ==========================================================
# PATH
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ==========================================================
# CONFIG
# ==========================================================

def load_rerank_config(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Cannot find config file:\n{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg



# ==========================================================
# E5 MODEL
# ==========================================================

_e5_model = None
_e5_tokenizer = None

_device = "cuda" if torch.cuda.is_available() else "cpu"
# _device = "cpu"

# ==========================================================
# OBJECT VECTOR INDEX
# ==========================================================

_OBJECT_INDEX = None
_OBJECT_MAPPING = None

OBJECT_INDEX_PATH = (
    ROOT / "data" / "indexes" / "object_IVFPQ_new.index"
)

OBJECT_MAPPING_PATH = (
    ROOT / "data" / "indexes" / "object_mapping_new_kf.json"
)
def get_object_index():

    global _OBJECT_INDEX
    global _OBJECT_MAPPING

    if _OBJECT_INDEX is not None:
        return (
            _OBJECT_INDEX,
            _OBJECT_MAPPING
        )

    print(
        f"[INFO] Loading object FAISS index: "
        f"{OBJECT_INDEX_PATH}"
    )

    if not OBJECT_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Object index not found:\n"
            f"{OBJECT_INDEX_PATH}"
        )

    if not OBJECT_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Object mapping not found:\n"
            f"{OBJECT_MAPPING_PATH}"
        )

    # ------------------------------------------------------
    # Load FAISS index ONCE
    # ------------------------------------------------------

    _OBJECT_INDEX = faiss.read_index(
        str(OBJECT_INDEX_PATH)
    )
    _OBJECT_INDEX.make_direct_map()
    _OBJECT_INDEX.nprobe = 32
    # ------------------------------------------------------
    # Load mapping ONCE
    # ------------------------------------------------------

    with open(
        OBJECT_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        _OBJECT_MAPPING = json.load(f)

    print(
        f"[INFO] Object index loaded: "
        f"{_OBJECT_INDEX.ntotal} vectors"
    )

    print(
        f"[INFO] Object mapping loaded: "
        f"{len(_OBJECT_MAPPING)} entries"
    )

  
    return (
        _OBJECT_INDEX,
        _OBJECT_MAPPING
    )


# ==========================================================
# OBJECT MAPPING LOOKUP
# ==========================================================

_OBJECT_LOOKUP = None
def get_object_lookup():
    global _OBJECT_LOOKUP

    if _OBJECT_LOOKUP is not None:
        return _OBJECT_LOOKUP

    if not OBJECT_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Object mapping not found:\n"
            f"{OBJECT_MAPPING_PATH}"
        )

    print(
        f"[INFO] Loading object lookup mapping: {OBJECT_MAPPING_PATH}"
    )

    with open(
        OBJECT_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        _OBJECT_LOOKUP = json.load(f)

    print(
        f"[INFO] Object lookup loaded: "
        f"{len(_OBJECT_LOOKUP)} entries."
    )

    return _OBJECT_LOOKUP
def get_object_vectors(object_index, vector_indices):
    vectors = []
    for idx in vector_indices:
        idx_int = int(idx)
        # Gọi trực tiếp FAISS reconstruct (đã được bật direct map)
        vec = object_index.reconstruct(idx_int)
        vec = np.asarray(vec, dtype=np.float32)
        
        # Chuẩn hóa (Normalize) vector nếu cần thiết
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        vectors.append(vec)
        
    return np.array(vectors, dtype=np.float32)

def get_e5_components():

    global _e5_model
    global _e5_tokenizer

    if _e5_model is None:

        model_name = "intfloat/multilingual-e5-base"

        print(
            f"[INFO] Loading {model_name} for E5 reranking..."
        )

        _e5_tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        _e5_model = AutoModel.from_pretrained(
            model_name
        ).to(_device)

        _e5_model.eval()

    return _e5_tokenizer, _e5_model


# ==========================================================
# E5 ENCODING
# ==========================================================

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


def encode_e5(
    texts,
    prefix,
    batch_size=32,
):

    if not texts:

        return np.empty(
            (0, 768),
            dtype=np.float32
        )

    tokenizer, model = get_e5_components()

    embeddings = []

    formatted_texts = [
        f"{prefix}: {text}"
        for text in texts
    ]

    for start in range(
        0,
        len(formatted_texts),
        batch_size
    ):

        batch = formatted_texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(_device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            emb = average_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            emb = F.normalize(
                emb,
                p=2,
                dim=1
            )

            embeddings.append(
                emb.cpu()
            )

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    return embeddings.numpy().astype(
        np.float32
    )


# ==========================================================
# QUERY EMBEDDING
# ==========================================================

def encode_query(
    query_text
):

    return encode_e5(
        [query_text],
        prefix="query",
        batch_size=1
    )[0]

def get_translated_query_en(query_text: str, query_en: str = None) -> str:
    """Tận dụng hàm dịch có sẵn từ retrieval_pipeline để lấy query tiếng Anh cho Object matching"""
    if query_en and isinstance(query_en, str) and query_en.strip():
        return query_en.strip()
    try:
        from src.retrieval.retrieval_pipeline import expand_and_translate_query
        translated = expand_and_translate_query(query_text)
        if translated and translated.strip():
            return translated.strip()
    except Exception as e:
        print(f"[WARNING] Không thể dịch query sang tiếng Anh cho Object reranking: {e}")
    return query_text

def compute_object_scores(
    query_emb_en,
    candidate_list,
    query_en=""
):
    print(
        "[INFO] Computing Hybrid Object scores (Semantic E5 with English query + Entity Lexical Match)..."
    )

    if not candidate_list:
        return candidate_list

    # ------------------------------------------------------
    # 1. Chuẩn bị Query tiếng Anh cho Lexical / Entity Matching
    # ------------------------------------------------------
    clean_query_en = ""
    q_tokens_en = []
    if query_en:
        clean_query_en = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(query_en).lower()).strip()
        q_tokens_en = [w for w in clean_query_en.split() if len(w) >= 2]

    # ------------------------------------------------------
    # Load index + lookup
    # ------------------------------------------------------
    object_index, _ = get_object_index()
    object_lookup = get_object_lookup()

    # Prepare query embedding
    query_emb_en = np.asarray(
        query_emb_en,
        dtype=np.float32
    ).reshape(1, -1)

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------
    total_object_vectors = 0
    candidates_with_object = 0
    candidates_without_object = 0
    total_reconstructed = 0

    semantic_scores = [0.0] * len(candidate_list)
    lexical_scores = [0.0] * len(candidate_list)

    # ------------------------------------------------------
    # Process candidates
    # ------------------------------------------------------
    for cand_idx, cand in enumerate(candidate_list):
        video_id = cand.get("video_id", "")
        keyframe_index = cand.get("keyframe_index")

        # --------------------------------------------------
        # A. LEXICAL ENTITY MATCHING VỚI cand["object_entities"]
        # --------------------------------------------------
        obj_entities = cand.get("object_entities", [])
        if not isinstance(obj_entities, list):
            obj_entities = []

        if q_tokens_en and obj_entities:
            best_entity_score = 0.0
            matched_count = 0

            clean_entities = [
                re.sub(r'[^a-zA-Z0-9\s]', ' ', str(e).lower()).strip()
                for e in obj_entities if str(e).strip()
            ]

            for ent in clean_entities:
                if not ent:
                    continue

                # 1. Exact phrase match (VD: "car", "skyscraper", "land vehicle")
                if ent in clean_query_en:
                    best_entity_score = max(best_entity_score, 1.0)
                    matched_count += 1
                elif any(w == ent for w in q_tokens_en):
                    best_entity_score = max(best_entity_score, 1.0)
                    matched_count += 1
                elif any(w in ent for w in q_tokens_en if len(w) >= 3):
                    best_entity_score = max(best_entity_score, 0.8)
                    matched_count += 1
                else:
                    # Fuzzy match
                    p_ratio = fuzz.partial_ratio(ent, clean_query_en)
                    if p_ratio >= 85.0:
                        best_entity_score = max(best_entity_score, p_ratio / 100.0)
                        matched_count += 1

            if best_entity_score > 0:
                coverage_bonus = min(0.2, (matched_count / max(1, len(q_tokens_en))) * 0.2)
                lexical_scores[cand_idx] = min(1.0, best_entity_score + coverage_bonus)

        # --------------------------------------------------
        # B. SEMANTIC SIMILARITY TỪ FAISS RECONSTRUCT
        # --------------------------------------------------
        if keyframe_index is None:
            candidates_without_object += 1
            continue

        key = f"{video_id}_{int(keyframe_index):04d}"
        vector_indices = object_lookup.get(key, [])

        if not vector_indices:
            candidates_without_object += 1
            continue

        candidates_with_object += 1
        total_object_vectors += len(vector_indices)
        total_reconstructed += len(vector_indices)

        # Get object vectors (Gọi trực tiếp FAISS reconstruct)
        object_vectors = get_object_vectors(
            object_index,
            vector_indices
        )

        # Cosine similarity với query embedding tiếng Anh
        similarities = (
            object_vectors @ query_emb_en.T
        ).reshape(-1)

        # Top-k average
        score = topk_average(
            similarities.tolist(),
            k=3
        )
        semantic_scores[cand_idx] = max(0.0, float(score))

    # ------------------------------------------------------
    # C. KẾT HỢP HYBRID (Semantic + Entity Lexical Boost)
    # ------------------------------------------------------
    for cand_idx, cand in enumerate(candidate_list):
        s_sem = semantic_scores[cand_idx]
        s_lex = lexical_scores[cand_idx]

        # Nếu có entity trùng khớp mạnh (s_lex >= 0.70)
        if s_lex >= 0.70:
            final_obj = min(1.0, max(s_sem, s_lex) + 0.10 * min(s_sem, s_lex))
        elif s_lex > 0.0:
            final_obj = max(s_sem, 0.7 * s_sem + 0.3 * s_lex)
        else:
            final_obj = s_sem

        cand["object_semantic_score"] = round(float(s_sem), 4)
        cand["object_lexical_score"] = round(float(s_lex), 4)
        cand["object_score"] = round(float(final_obj), 4)

    # ------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------
    scores = [
        cand.get("object_score", 0.0)
        for cand in candidate_list
    ]

    print("[DEBUG OBJECT]")
    print(f"Candidates                 : {len(candidate_list)}")
    print(f"Candidates with object     : {candidates_with_object}")
    print(f"Candidates without object  : {candidates_without_object}")
    print(f"Object vectors evaluated   : {total_object_vectors}")
    print(f"Vectors reconstructed      : {total_reconstructed}")
    print(
        f"Score range                : "
        f"{min(scores):.4f} -> {max(scores):.4f}"
    )

    return candidate_list
# ==========================================================
# METADATA VECTOR INDEX
# ==========================================================

_METADATA_INDEX = None
_METADATA_MAPPING = None

METADATA_INDEX_PATH = (
    ROOT / "data" / "indexes" / "metadata_IVFPQ_new.index"
)

METADATA_MAPPING_PATH = (
    ROOT / "data" / "indexes" / "metadata_mapping_new_kf.json"
)

# ==========================================================
# RUNTIME CACHE
# ==========================================================

_OBJECT_INDEX = None
_OBJECT_MAPPING = None
_OBJECT_LOOKUP = None

_METADATA_INDEX = None
_METADATA_MAPPING = None
_METADATA_LOOKUP = None



def get_metadata_index():

    global _METADATA_INDEX
    global _METADATA_MAPPING

    if _METADATA_INDEX is not None:

        return (
            _METADATA_INDEX,
            _METADATA_MAPPING
        )

    print(
        f"[INFO] Loading metadata FAISS index: "
        f"{METADATA_INDEX_PATH}"
    )

    if not METADATA_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Metadata index not found:\n"
            f"{METADATA_INDEX_PATH}"
        )

    if not METADATA_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Metadata mapping not found:\n"
            f"{METADATA_MAPPING_PATH}"
        )

    _METADATA_INDEX = faiss.read_index(
        str(METADATA_INDEX_PATH)
    )
    _METADATA_INDEX.make_direct_map()
    _METADATA_INDEX.nprobe = 32
    with open(
        METADATA_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        _METADATA_MAPPING = json.load(f)

    print(
        f"[INFO] Metadata index loaded: "
        f"{_METADATA_INDEX.ntotal} vectors"
    )

    print(
        f"[INFO] Metadata mapping loaded: "
        f"{len(_METADATA_MAPPING)} entries"
    )

   

    return (
        _METADATA_INDEX,
        _METADATA_MAPPING
    )

# ==========================================================
# METADATA MAPPING LOOKUP
# ==========================================================

_METADATA_LOOKUP = None
def get_metadata_lookup():
    global _METADATA_LOOKUP

    if _METADATA_LOOKUP is not None:
        return _METADATA_LOOKUP

    if not METADATA_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Metadata mapping not found:\n"
            f"{METADATA_MAPPING_PATH}"
        )

    print(
        f"[INFO] Loading metadata lookup mapping: {METADATA_MAPPING_PATH}"
    )

    with open(
        METADATA_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        _METADATA_LOOKUP = json.load(f)

    print(
        f"[INFO] Metadata lookup loaded: "
        f"{len(_METADATA_LOOKUP)} entries."
    )

    return _METADATA_LOOKUP

def get_metadata_vector(metadata_index, vector_index):
    vector_index = int(vector_index)
    
    # Gọi trực tiếp FAISS reconstruct thay vì tra cứu qua cache dict
    vector = metadata_index.reconstruct(vector_index)
    vector = np.asarray(vector, dtype=np.float32)
    
    # Chuẩn hóa (Normalize) vector
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
        
    return vector

# ==========================================================
# HYBRID OCR SCORE (SEMANTIC E5 + ROBUST FUZZY LEXICAL MATCH)
# ==========================================================

def compute_ocr_scores(
    query_emb,
    candidate_list,
    query_text="",
    ocr_threshold=0.5,
    batch_size=32
):
    print(
        f"[INFO] Computing Hybrid OCR scores (Semantic E5 + Robust Fuzzy Match, threshold > {ocr_threshold})..."
    )

    if not candidate_list:
        return candidate_list

    query_emb = query_emb.reshape(1, -1)

    # Chuẩn bị Query cho Fuzzy Matching (bỏ dấu và lọc ký tự đặc biệt)
    clean_query = ""
    q_tokens = []
    if query_text:
        norm_query = remove_vietnamese_diacritics(query_text)
        clean_query = clean_ocr_text(norm_query)
        q_tokens = [w for w in clean_query.split() if len(w) >= 2]

    ocr_texts = []
    ocr_candidate_ids = []
    cand_clean_ocr_map = [[] for _ in candidate_list]

    for cand_idx, cand in enumerate(candidate_list):
        texts_info = cand.get("ocr_texts", [])
        if not isinstance(texts_info, list):
            texts_info = []

        for item in texts_info:
            # Hỗ trợ tương thích ngược: item có thể là dict {"text": ..., "score": ...} hoặc chuỗi
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                score = float(item.get("score", 1.0))
            elif isinstance(item, str):
                text = str(item).strip()
                score = 1.0
            else:
                continue

            if not text:
                continue

            # LỌC THEO THRESHOLD: Chỉ đưa vào xử lý nếu score vượt ngưỡng
            if score >= ocr_threshold:
                ocr_texts.append(text)
                ocr_candidate_ids.append(cand_idx)

                # Lưu text sạch cho Fuzzy Matching
                c_text = clean_ocr_text(remove_vietnamese_diacritics(text))
                if c_text and len(c_text) >= 2:
                    cand_clean_ocr_map[cand_idx].append(c_text)

    # ------------------------------------------------------
    # 1. TÍNH ĐIỂM NGỮ CẢNH BẰNG E5 (Semantic Similarity)
    # ------------------------------------------------------
    semantic_scores = [0.0] * len(candidate_list)

    if ocr_texts:
        ocr_embeddings = encode_e5(
            ocr_texts,
            prefix="passage",
            batch_size=batch_size
        )

        similarities = (
            ocr_embeddings @ query_emb.T
        ).reshape(-1)

        candidate_scores = [[] for _ in candidate_list]

        for sim, cand_idx in zip(similarities, ocr_candidate_ids):
            candidate_scores[cand_idx].append(float(sim))

        for cand_idx in range(len(candidate_list)):
            scores = candidate_scores[cand_idx]
            if scores:
                semantic_scores[cand_idx] = float(topk_average(scores, k=3))

    # ------------------------------------------------------
    # 2. TÍNH ĐIỂM SO KHỚP MỜ / TỪ KHÓA (Fuzzy & Exact Lexical Match)
    # ------------------------------------------------------
    for cand_idx, cand in enumerate(candidate_list):
        s_semantic = semantic_scores[cand_idx]
        s_fuzzy = 0.0
        clean_ocr_list = cand_clean_ocr_map[cand_idx]

        if clean_query and clean_ocr_list:
            best_match_val = 0.0

            for ocr_str in clean_ocr_list:
                # partial_ratio: tìm query con trong đoạn OCR
                p_ratio = fuzz.partial_ratio(clean_query, ocr_str)
                # token_set_ratio: so khớp tập hợp từ chống sai thứ tự
                t_ratio = fuzz.token_set_ratio(clean_query, ocr_str)

                # Token exact hit bonus (cho chữ số, mã số, tên riêng)
                token_hits = sum(1 for t in q_tokens if t in ocr_str)
                token_bonus = (token_hits / len(q_tokens)) * 25.0 if q_tokens else 0.0

                current_val = max(p_ratio, t_ratio) + token_bonus
                if current_val > best_match_val:
                    best_match_val = current_val

            # Chuẩn hóa về [0.0, 1.0] nếu đạt ngưỡng tin cậy >= 60%
            if best_match_val >= 60.0:
                s_fuzzy = min(best_match_val / 100.0, 1.0)

        # --------------------------------------------------
        # 3. KẾT HỢP HYBRID (Semantic + Fuzzy Boost)
        # --------------------------------------------------
        # Nếu có từ khóa/số hiệu khớp mạnh (s_fuzzy >= 0.70):
        # Ưu tiên lấy điểm cao nhất + bonus cộng hưởng
        # Nếu không khớp từ khóa: Giữ nguyên 100% điểm E5 ngữ cảnh
        if s_fuzzy >= 0.70:
            final_ocr = min(1.0, max(s_semantic, s_fuzzy) + 0.10 * min(s_semantic, s_fuzzy))
        elif s_fuzzy > 0.0:
            final_ocr = max(s_semantic, 0.7 * s_semantic + 0.3 * s_fuzzy)
        else:
            final_ocr = s_semantic

        cand["ocr_semantic_score"] = round(float(s_semantic), 4)
        cand["ocr_fuzzy_score"] = round(float(s_fuzzy), 4)
        cand["ocr_score"] = round(float(final_ocr), 4)

    # ------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------
    non_zero = sum(
        1
        for cand in candidate_list
        if cand["ocr_score"] != 0
    )

    scores = [
        cand["ocr_score"]
        for cand in candidate_list
    ]

    print("[DEBUG OCR]")
    print(f"Candidates             : {len(candidate_list)}")
    print(f"Candidates with OCR    : {non_zero}")
    print(
        f"Score range            : "
        f"{min(scores):.4f} -> {max(scores):.4f}"
    )

    return candidate_list
# def extract_char_ngrams(text: str, n: int = 3) -> list:
#     """Tách chuỗi thành các N-gram ký tự (Ví dụ: 'vinmart' -> [' vi', 'vin',
#     'inm', 'rma', 'art', 'rt '])"""
#     if not text:
#         return []
#     text = f" {text.strip()} "  # Thêm khoảng trắng đầu/cuối để bắt ranh giới từ
#     if len(text) < n:
#         return [text]
#     return [text[i : i + n] for i in range(len(text) - n + 1)]


# def compute_ocr_scores(
#     query_emb, candidate_list, query_text="", ocr_threshold=0.5, batch_size=32
# ):
#     print(
#         f"[INFO] Computing Hybrid OCR scores (Semantic E5 + Char 3-Gram BM25+, threshold > {ocr_threshold})..."
#     )

#     if not candidate_list:
#         return candidate_list

#     query_emb = query_emb.reshape(1, -1)

#     # ------------------------------------------------------
#     # Chuẩn bị Query Char 3-Gram cho BM25+
#     # ------------------------------------------------------
#     clean_query = ""
#     q_ngrams = []
#     if query_text:
#         norm_query = remove_vietnamese_diacritics(query_text)
#         clean_query = clean_ocr_text(norm_query)
#         q_ngrams = extract_char_ngrams(clean_query, n=3)

#     ocr_texts = []
#     ocr_candidate_ids = []
#     cand_clean_ocr_map = [[] for _ in candidate_list]

#     for cand_idx, cand in enumerate(candidate_list):
#         texts_info = cand.get("ocr_texts", [])
#         if not isinstance(texts_info, list):
#             texts_info = []

#         for item in texts_info:
#             if isinstance(item, dict):
#                 text = str(item.get("text", "")).strip()
#                 score = float(item.get("score", 1.0))
#             elif isinstance(item, str):
#                 text = str(item).strip()
#                 score = 1.0
#             else:
#                 continue

#             if not text:
#                 continue

#             # LỌC THEO THRESHOLD
#             if score >= ocr_threshold:
#                 ocr_texts.append(text)
#                 ocr_candidate_ids.append(cand_idx)

#                 # Lưu text sạch cho BM25
#                 c_text = clean_ocr_text(remove_vietnamese_diacritics(text))
#                 if c_text and len(c_text) >= 2:
#                     cand_clean_ocr_map[cand_idx].append(c_text)

#     # ------------------------------------------------------
#     # 1. TÍNH ĐIỂM NGỮ CẢNH BẰNG E5 (Semantic Similarity)
#     # ------------------------------------------------------
#     semantic_scores = [0.0] * len(candidate_list)

#     if ocr_texts:
#         ocr_embeddings = encode_e5(
#             ocr_texts, prefix="passage", batch_size=batch_size
#         )

#         similarities = (ocr_embeddings @ query_emb.T).reshape(-1)

#         candidate_scores = [[] for _ in candidate_list]

#         for sim, cand_idx in zip(similarities, ocr_candidate_ids):
#             candidate_scores[cand_idx].append(float(sim))

#         for cand_idx in range(len(candidate_list)):
#             scores = candidate_scores[cand_idx]
#             if scores:
#                 semantic_scores[cand_idx] = float(topk_average(scores, k=3))

#     # ------------------------------------------------------
#     # 2. TÍNH ĐIỂM LEXICAL BẰNG CHAR 3-GRAM BM25+
#     # ------------------------------------------------------
#     bm25_scores = [0.0] * len(candidate_list)

#     if q_ngrams:
#         # Xây dựng corpus Char 3-gram cho từng Candidate
#         corpus_ngrams = []
#         for cand_idx in range(len(candidate_list)):
#             clean_ocr_list = cand_clean_ocr_map[cand_idx]
#             combined_ocr_text = " ".join(clean_ocr_list)
#             doc_ngrams = extract_char_ngrams(combined_ocr_text, n=3)
#             corpus_ngrams.append(doc_ngrams)

#         # Kiểm tra nếu ít nhất 1 candidate có văn bản OCR
#         if any(len(doc) > 0 for doc in corpus_ngrams):
#             # Đưa q_ngrams làm tài liệu tham chiếu ở vị trí 0 để lấy điểm max lý tưởng
#             full_corpus = [q_ngrams] + corpus_ngrams
#             bm25_model = BM25Plus(full_corpus, delta=1.0)

#             raw_scores = bm25_model.get_scores(q_ngrams)
#             max_possible_score = raw_scores[0]  # Self-match score

#             if max_possible_score > 0:
#                 for cand_idx in range(len(candidate_list)):
#                     cand_raw_score = raw_scores[cand_idx + 1]
#                     # Chuẩn hóa điểm về khoảng [0.0, 1.0]
#                     norm_score = min(
#                         1.0, max(0.0, cand_raw_score / max_possible_score)
#                     )
#                     bm25_scores[cand_idx] = norm_score

#     # ------------------------------------------------------
#     # 3. KẾT HỢP HYBRID (Semantic + BM25+ Boost)
#     # ------------------------------------------------------
#     for cand_idx, cand in enumerate(candidate_list):
#         s_semantic = semantic_scores[cand_idx]
#         s_bm25 = bm25_scores[cand_idx]

#         # Nếu có từ khóa/chuỗi ký tự khớp mạnh (s_bm25 >= 0.50):
#         if s_bm25 >= 0.50:
#             final_ocr = min(
#                 1.0, max(s_semantic, s_bm25) + 0.10 * min(s_semantic, s_bm25)
#             )
#         elif s_bm25 > 0.0:
#             final_ocr = max(s_semantic, 0.7 * s_semantic + 0.3 * s_bm25)
#         else:
#             final_ocr = s_semantic

#         cand["ocr_semantic_score"] = round(float(s_semantic), 4)
#         cand["ocr_bm25_score"] = round(float(s_bm25), 4)
#         cand["ocr_score"] = round(float(final_ocr), 4)

#     # ------------------------------------------------------
#     # DEBUG
#     # ------------------------------------------------------
#     non_zero = sum(1 for cand in candidate_list if cand["ocr_score"] != 0)
#     scores = [cand["ocr_score"] for cand in candidate_list]

#     print("[DEBUG OCR]")
#     print(f"Candidates             : {len(candidate_list)}")
#     print(f"Candidates with OCR    : {non_zero}")
#     print(
#         f"Score range            : {min(scores):.4f} -> {max(scores):.4f}"
#     )

#     return candidate_list
# ==========================================================
# METADATA SEMANTIC SCORE
# USING PRECOMPUTED E5 METADATA INDEX
# ==========================================================


def compute_metadata_scores(
    query_emb,
    candidate_list,
    query_text=""
):
    print(
        "[INFO] Computing Hybrid Metadata scores (Semantic E5 + Title/Keywords Lexical BM25+)..."
    )

    if not candidate_list:
        return candidate_list

    # Load index + lookup
    metadata_index, _ = get_metadata_index()
    metadata_lookup = get_metadata_lookup()

    # Prepare query embedding
    query_emb = np.asarray(
        query_emb,
        dtype=np.float32
    ).reshape(1, -1)

    # Chuẩn bị Query cho Lexical BM25+ Matching
    clean_query = ""
    q_tokens = []
    if query_text:
        norm_query = remove_vietnamese_diacritics(query_text)
        clean_query = clean_ocr_text(norm_query)
        q_tokens = [w for w in clean_query.split() if len(w) >= 2]

    # Statistics
    candidates_with_metadata = 0
    candidates_without_metadata = 0
    total_reconstructed = 0

    semantic_scores = [0.0] * len(candidate_list)
    bm25_scores = [0.0] * len(candidate_list)
    cand_clean_meta_map = [[] for _ in candidate_list]

    for cand_idx, cand in enumerate(candidate_list):
        video_id = cand.get("video_id", "")
        keyframe_index = cand.get("keyframe_index")

        # Chuẩn bị văn bản Metadata (Title + Keywords + description) cho BM25+
        meta = cand.get("metadata", {})
        if isinstance(meta, dict):
            title = str(meta.get("title", "")).strip()
            description = str(meta.get("description", "")).strip()
            keywords = meta.get("keywords", [])
            if isinstance(keywords, list):
                kw_str = " ".join([str(k) for k in keywords])
            else:
                kw_str = str(keywords)
            
            clean_meta_str = clean_ocr_text(remove_vietnamese_diacritics(f"{title} {description} {kw_str}"))
            if clean_meta_str:
                cand_clean_meta_map[cand_idx] = [w for w in clean_meta_str.split() if len(w) >= 2]

        if keyframe_index is None:
            candidates_without_metadata += 1
            continue

        key = f"{video_id}_{int(keyframe_index):04d}"
        entry = metadata_lookup.get(key)

        if entry is None or "vector_index" not in entry:
            candidates_without_metadata += 1
            continue

        vector_index = entry.get("vector_index")
        if vector_index is None:
            candidates_without_metadata += 1
            continue

        candidates_with_metadata += 1
        vector_index = int(vector_index)
        total_reconstructed += 1

        # 1. SEMANTIC SIMILARITY TỪ FAISS RECONSTRUCT
        metadata_vector = get_metadata_vector(
            metadata_index,
            vector_index
        )

        similarity = float(
            metadata_vector @ query_emb[0]
        )
        semantic_scores[cand_idx] = max(0.0, float(similarity))

    # 2. TÍNH ĐIỂM LEXICAL BM25+ TRÊN TITLE VÀ KEYWORDS
    if q_tokens:
        corpus_tokens = cand_clean_meta_map
        if any(len(doc) > 0 for doc in corpus_tokens):
            full_corpus = [q_tokens] + corpus_tokens
            bm25_model = BM25Plus(full_corpus, delta=1.0)
            raw_scores = bm25_model.get_scores(q_tokens)
            max_possible_score = raw_scores[0]

            if max_possible_score > 0:
                for cand_idx in range(len(candidate_list)):
                    cand_raw_score = raw_scores[cand_idx + 1]
                    norm_score = min(
                        1.0, max(0.0, cand_raw_score / max_possible_score)
                    )
                    bm25_scores[cand_idx] = norm_score

    # 3. KẾT HỢP HYBRID (Semantic + Lexical BM25 Boost)
    # for cand_idx, cand in enumerate(candidate_list):
    #     s_sem = semantic_scores[cand_idx]
    #     s_bm25 = bm25_scores[cand_idx]

    #     if s_bm25 >= 0.40:
    #         final_meta = min(1.0, max(s_sem, s_bm25) + 0.10 * min(s_sem, s_bm25))
    #     elif s_bm25 > 0.0:
    #         final_meta = max(s_sem, 0.65 * s_sem + 0.35 * s_bm25)
    #     else:
    #         final_meta = s_sem

    #     cand["metadata_semantic_score"] = round(float(s_sem), 4)
    #     cand["metadata_bm25_score"] = round(float(s_bm25), 4)
    #     cand["metadata_score"] = round(float(final_meta), 4)
    for cand_idx, cand in enumerate(candidate_list):
        s_sem = semantic_scores[cand_idx]
        s_bm25 = bm25_scores[cand_idx]

        if s_bm25 >= 0.40:
            final_meta = min(1.0, max(s_sem, s_bm25) + 0.10 * min(s_sem, s_bm25))
        elif s_bm25 > 0.0:
            final_meta = max(s_sem, 0.65 * s_sem + 0.35 * s_bm25)
        else:
            final_meta = s_sem

        # 4. DOMAIN BOOSTER V3: chỉ dùng metadata + ASR
        meta_for_boost = cand.get("metadata", {})
        domain_boost = booster.compute_boost(query_text, meta_for_boost)
        if domain_boost < 0:
            # Phạt nặng tay hơn để kéo sập điểm metadata của video sai môn
            final_meta += 0.50 * domain_boost
        else:
            final_meta += 0.20 * domain_boost
        final_meta = min(1.0, max(0.0, final_meta))

        cand["metadata_semantic_score"] = round(float(s_sem), 4)
        cand["metadata_bm25_score"] = round(float(s_bm25), 4)
        cand["metadata_domain_boost"] = round(float(domain_boost), 4)
        cand["metadata_score"] = round(float(final_meta), 4)
    # DEBUG
    scores = [
        cand.get("metadata_score", 0.0)
        for cand in candidate_list
    ]

    print("[DEBUG METADATA]")
    print(f"Candidates                  : {len(candidate_list)}")
    print(f"Candidates with metadata    : {candidates_with_metadata}")
    print(f"Candidates without metadata : {candidates_without_metadata}")
    print(f"Vectors reconstructed       : {total_reconstructed}")
    print(
        f"Score range                 : "
        f"{min(scores):.4f} -> {max(scores):.4f}"
    )

    return candidate_list
# ==========================================================
# HYBRID ASR SCORE (SEMANTIC E5 + ROBUST FUZZY LEXICAL MATCH)
# ==========================================================

# def compute_asr_scores(
#     query_emb,
#     candidate_list,
#     query_text="",
#     batch_size=32
# ):
#     print(
#         "[INFO] Computing Hybrid ASR scores (Semantic E5 + Robust Fuzzy Match)..."
#     )

#     if not candidate_list:
#         return candidate_list

#     query_emb = query_emb.reshape(1, -1)

#     # Chuẩn bị Query cho Fuzzy Matching (bỏ dấu và lọc ký tự đặc biệt)
#     clean_query = ""
#     q_tokens = []
#     if query_text:
#         norm_query = remove_vietnamese_diacritics(query_text)
#         clean_query = clean_ocr_text(norm_query)
#         q_tokens = [w for w in clean_query.split() if len(w) >= 2]

#     asr_texts = []
#     asr_candidate_ids = []
#     cand_clean_asr_map = [[] for _ in candidate_list]

#     for cand_idx, cand in enumerate(candidate_list):
#         text_val = cand.get("asr_text", "")

#         if isinstance(text_val, list):
#             texts = text_val
#         elif isinstance(text_val, str) and text_val.strip():
#             texts = [text_val]
#         else:
#             texts = []

#         for text in texts:
#             text = str(text).strip()
#             if not text:
#                 continue

#             asr_texts.append(text)
#             asr_candidate_ids.append(cand_idx)

#             c_text = clean_ocr_text(remove_vietnamese_diacritics(text))
#             if c_text and len(c_text) >= 2:
#                 cand_clean_asr_map[cand_idx].append(c_text)

#     # ------------------------------------------------------
#     # 1. TÍNH ĐIỂM NGỮ CẢNH BẰNG E5 (Semantic Similarity)
#     # ------------------------------------------------------
#     semantic_scores = [0.0] * len(candidate_list)

#     if asr_texts:
#         asr_embeddings = encode_e5(
#             asr_texts,
#             prefix="passage",
#             batch_size=batch_size
#         )

#         similarities = (
#             asr_embeddings @ query_emb.T
#         ).reshape(-1)

#         candidate_semantic_scores = [[] for _ in candidate_list]

#         for sim, cand_idx in zip(similarities, asr_candidate_ids):
#             candidate_semantic_scores[cand_idx].append(float(sim))

#         for cand_idx in range(len(candidate_list)):
#             scores = candidate_semantic_scores[cand_idx]
#             if scores:
#                 semantic_scores[cand_idx] = float(max(scores))

#     # ------------------------------------------------------
#     # 2. TÍNH ĐIỂM SO KHỚP MỜ / TỪ KHÓA (Fuzzy & Exact Lexical Match)
#     # ------------------------------------------------------
#     for cand_idx, cand in enumerate(candidate_list):
#         s_semantic = semantic_scores[cand_idx]
#         s_fuzzy = 0.0
#         clean_asr_list = cand_clean_asr_map[cand_idx]

#         if clean_query and clean_asr_list:
#             best_match_val = 0.0

#             for asr_str in clean_asr_list:
#                 # partial_ratio: tìm query con trong đoạn ASR
#                 p_ratio = fuzz.partial_ratio(clean_query, asr_str)
#                 # token_set_ratio: so khớp tập hợp từ khóa
#                 t_ratio = fuzz.token_set_ratio(clean_query, asr_str)

#                 # Token exact hit bonus
#                 token_hits = sum(1 for t in q_tokens if t in asr_str)
#                 token_bonus = (token_hits / len(q_tokens)) * 25.0 if q_tokens else 0.0

#                 current_val = max(p_ratio, t_ratio) + token_bonus
#                 if current_val > best_match_val:
#                     best_match_val = current_val

#             # Chuẩn hóa về [0.0, 1.0] nếu đạt ngưỡng tin cậy >= 60%
#             if best_match_val >= 60.0:
#                 s_fuzzy = min(best_match_val / 100.0, 1.0)

#         # --------------------------------------------------
#         # 3. KẾT HỢP HYBRID (Semantic + Fuzzy Boost)
#         # --------------------------------------------------
#         if s_fuzzy >= 0.70:
#             final_asr = min(1.0, max(s_semantic, s_fuzzy) + 0.10 * min(s_semantic, s_fuzzy))
#         elif s_fuzzy > 0.0:
#             final_asr = max(s_semantic, 0.7 * s_semantic + 0.3 * s_fuzzy)
#         else:
#             final_asr = s_semantic

#         cand["asr_semantic_score"] = round(float(s_semantic), 4)
#         cand["asr_fuzzy_score"] = round(float(s_fuzzy), 4)
#         cand["asr_score"] = round(float(final_asr), 4)

#     # ------------------------------------------------------
#     # DEBUG
#     # ------------------------------------------------------
#     non_zero = sum(
#         1
#         for cand in candidate_list
#         if cand["asr_score"] != 0
#     )

#     scores = [
#         cand["asr_score"]
#         for cand in candidate_list
#     ]

#     print("[DEBUG ASR]")
#     print(f"Candidates             : {len(candidate_list)}")
#     print(f"Candidates with ASR    : {non_zero}")
#     print(
#         f"Score range            : "
#         f"{min(scores):.4f} -> {max(scores):.4f}"
#     )

#     return candidate_list
def compute_asr_scores(
    query_emb, candidate_list, query_text="", batch_size=32
):
    print(
        "[INFO] Computing Hybrid ASR scores (Semantic E5 + Word-Level BM25+)..."
    )

    if not candidate_list:
        return candidate_list

    query_emb = query_emb.reshape(1, -1)

    # ------------------------------------------------------
    # Chuẩn bị Query Word Tokens cho BM25+
    # ------------------------------------------------------
    clean_query = ""
    q_tokens = []
    if query_text:
        norm_query = remove_vietnamese_diacritics(query_text)
        clean_query = clean_ocr_text(norm_query)
        q_tokens = [w for w in clean_query.split() if len(w) >= 2]

    asr_texts = []
    asr_candidate_ids = []
    cand_clean_asr_map = [[] for _ in candidate_list]

    for cand_idx, cand in enumerate(candidate_list):
        text_val = cand.get("asr_text", "")

        if isinstance(text_val, list):
            texts = text_val
        elif isinstance(text_val, str) and text_val.strip():
            texts = [text_val]
        else:
            texts = []

        for text in texts:
            text = str(text).strip()
            if not text:
                continue

            asr_texts.append(text)
            asr_candidate_ids.append(cand_idx)

            c_text = clean_ocr_text(remove_vietnamese_diacritics(text))
            if c_text and len(c_text) >= 2:
                cand_clean_asr_map[cand_idx].append(c_text)

    # ------------------------------------------------------
    # 1. TÍNH ĐIỂM NGỮ CẢNH BẰNG E5 (Semantic Similarity)
    # ------------------------------------------------------
    semantic_scores = [0.0] * len(candidate_list)

    if asr_texts:
        asr_embeddings = encode_e5(
            asr_texts, prefix="passage", batch_size=batch_size
        )

        similarities = (asr_embeddings @ query_emb.T).reshape(-1)

        candidate_semantic_scores = [[] for _ in candidate_list]

        for sim, cand_idx in zip(similarities, asr_candidate_ids):
            candidate_semantic_scores[cand_idx].append(float(sim))

        for cand_idx in range(len(candidate_list)):
            scores = candidate_semantic_scores[cand_idx]
            if scores:
                semantic_scores[cand_idx] = float(max(scores))

    # ------------------------------------------------------
    # 2. TÍNH ĐIỂM TỪ KHÓA BẰNG WORD-LEVEL BM25+
    # ------------------------------------------------------
    bm25_scores = [0.0] * len(candidate_list)

    if q_tokens:
        # Tách Word Tokens cho từng Candidate ASR
        corpus_tokens = []
        for cand_idx in range(len(candidate_list)):
            clean_asr_list = cand_clean_asr_map[cand_idx]
            combined_asr_text = " ".join(clean_asr_list)
            doc_tokens = [
                w for w in combined_asr_text.split() if len(w) >= 2
            ]
            corpus_tokens.append(doc_tokens)

        # Đưa q_tokens vào đầu full_corpus làm tài liệu chuẩn (Self-match reference)
        if any(len(doc) > 0 for doc in corpus_tokens):
            full_corpus = [q_tokens] + corpus_tokens
            bm25_model = BM25Plus(full_corpus, delta=1.0)

            raw_scores = bm25_model.get_scores(q_tokens)
            max_possible_score = raw_scores[0]

            if max_possible_score > 0:
                for cand_idx in range(len(candidate_list)):
                    cand_raw_score = raw_scores[cand_idx + 1]
                    # Chuẩn hóa về thang [0.0, 1.0]
                    norm_score = min(
                        1.0, max(0.0, cand_raw_score / max_possible_score)
                    )
                    bm25_scores[cand_idx] = norm_score

    # ------------------------------------------------------
    # 3. KẾT HỢP HYBRID (Semantic + BM25 Boost)
    # ------------------------------------------------------
    for cand_idx, cand in enumerate(candidate_list):
        s_semantic = semantic_scores[cand_idx]
        s_bm25 = bm25_scores[cand_idx]

        # Nếu có từ khóa trùng khớp mạnh (s_bm25 >= 0.40)
        if s_bm25 >= 0.40:
            final_asr = min(
                1.0, max(s_semantic, s_bm25) + 0.10 * min(s_semantic, s_bm25)
            )
        elif s_bm25 > 0.0:
            final_asr = max(s_semantic, 0.7 * s_semantic + 0.3 * s_bm25)
        else:
            final_asr = s_semantic
        
        cand["asr_semantic_score"] = round(float(s_semantic), 4)
        cand["asr_bm25_score"] = round(float(s_bm25), 4)
        cand["asr_score"] = round(float(final_asr), 4)
    
    # ------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------
    non_zero = sum(1 for cand in candidate_list if cand["asr_score"] != 0)
    scores = [cand["asr_score"] for cand in candidate_list]

    print("[DEBUG ASR]")
    print(f"Candidates             : {len(candidate_list)}")
    print(f"Candidates with ASR    : {non_zero}")
    print(
        f"Score range            : {min(scores):.4f} -> {max(scores):.4f}"
    )

    return candidate_list
















# ==========================================================
# RETRIEVAL SCORE NORMALIZATION
# ==========================================================

def normalize_retrieval_scores(
    candidate_list
):

    retrieval_scores = np.array(
        [
            cand["score"]
            for cand in candidate_list
        ],
        dtype=np.float32
    )

    min_score = retrieval_scores.min()
    max_score = retrieval_scores.max()

    if (
        max_score - min_score
        > 1e-8
    ):

        normalized = (
            retrieval_scores
            - min_score
        ) / (
            max_score
            - min_score
        )

    else:

        normalized = np.ones_like(
            retrieval_scores
        )

    for i, cand in enumerate(
        candidate_list
    ):

        cand["retrieval_score"] = float(
            cand["score"]
        )

        cand[
            "retrieval_score_normalized"
        ] = round(
            float(normalized[i]),
            4
        )

    return candidate_list


def normalize_feature(candidate_list, feature_name):

    values = np.array(
        [
            cand.get(feature_name, 0.0)
            for cand in candidate_list
        ],
        dtype=np.float32
    )

    if len(values) == 0:
        return candidate_list

    min_v = values.min()
    max_v = values.max()

    if max_v - min_v > 1e-8:
        normalized = (
            values - min_v
        ) / (
            max_v - min_v
        )
    else:
        # Nếu toàn bộ score đều bằng 0
        # thì feature này không cung cấp tín hiệu
        if abs(min_v) < 1e-8:
            normalized = np.zeros_like(values)
        else:
            # tất cả có cùng score > 0
            normalized = np.ones_like(values)

    for cand, v in zip(
        candidate_list,
        normalized
    ):
        cand[
            f"{feature_name}_norm"
        ] = float(v)

    return candidate_list
# ==========================================================
# RRF
# ==========================================================

def rrf_score_fusion(
    candidate_list,
    k=60
):

    """
    RRF over:
        1. CLIP retrieval
        2. Object semantic score
        3. Metadata semantic score
        4. OCR MaxSim score
        5. ASR MaxSim score 
    """

    sorted_by_clip = sorted(
        candidate_list,
        key=lambda x: x.get(
            "score",
            0.0
        ),
        reverse=True
    )

    clip_ranks = {
        cand["vector_index"]: idx + 1
        for idx, cand in enumerate(
            sorted_by_clip
        )
    }

    sorted_by_obj = sorted(
        candidate_list,
        key=lambda x: x.get(
            "object_score",
            0.0
        ),
        reverse=True
    )

    obj_ranks = {
        cand["vector_index"]: idx + 1
        for idx, cand in enumerate(
            sorted_by_obj
        )
    }

    sorted_by_meta = sorted(
        candidate_list,
        key=lambda x: x.get(
            "metadata_score",
            0.0
        ),
        reverse=True
    )

    meta_ranks = {
        cand["vector_index"]: idx + 1
        for idx, cand in enumerate(
            sorted_by_meta
        )
    }

    sorted_by_ocr = sorted(
        candidate_list,
        key=lambda x: x.get(
            "ocr_score",
            0.0
        ),
        reverse=True
    )

    ocr_ranks = {
        cand["vector_index"]: idx + 1
        for idx, cand in enumerate(
            sorted_by_ocr
        )
    }

    sorted_by_asr = sorted(
        candidate_list,
        key=lambda x: x.get(
            "asr_score",
            0.0
        ),
        reverse=True
    )

    asr_ranks = {
        cand["vector_index"]: idx + 1
        for idx, cand in enumerate(
            sorted_by_asr
        )
    }

    for cand in candidate_list:

        v_idx = cand[
            "vector_index"
        ]

        r_clip = clip_ranks[
            v_idx
        ]

        r_obj = obj_ranks[
            v_idx
        ]

        r_meta = meta_ranks[
            v_idx
        ]

        r_ocr = ocr_ranks[
            v_idx
        ]

        r_asr = asr_ranks[
            v_idx
        ]

        score = (
            1.0 / (k + r_clip)
            + 1.0 / (k + r_obj)
            + 1.0 / (k + r_meta)
            + 1.0 / (k + r_ocr)
            + 1.0 / (k + r_asr)
        )

        cand[
            "final_score"
        ] = round(
            float(score),
            6
        )

    return sorted(
        candidate_list,
        key=lambda x: x[
            "final_score"
        ],
        reverse=True
    )



def compute_video_narrative_bonus(candidate_list: list, query_text: str = "", query_plan: dict = None) -> list:
    """
    Cộng điểm thưởng gắn kết cấp Video (Video-level Narrative/Coherence Bonus).
    Tối ưu hóa:
    1. Trích xuất thuộc tính narrative từ query_plan (dict) nếu có.
    2. Fallback kiểm tra từ khóa trên query_text gốc.
    3. Trích xuất video_id an toàn từ candidate.
    """
    if not candidate_list:
        return candidate_list

    # Step 1: Xác định tính chất narrative từ query_plan hoặc query_text
    has_narrative = False
    
    # Ưu tiên lấy query_plan từ tham số truyền vào hoặc từ candidate đầu tiên
    qp = query_plan
    if not qp and len(candidate_list) > 0:
        qp = candidate_list[0].get("query_plan")

    if isinstance(qp, dict):
        # Nếu query_plan có phân rã q_context hoặc flag narrative
        q_context = qp.get("q_context", "")
        q_main = qp.get("q_main", "")
        if q_context or qp.get("is_narrative", False):
            has_narrative = True
        elif len(f"{q_main} {q_context}".split()) > 15:
            has_narrative = True

    # Fallback kiểm tra chuỗi văn bản nếu query_plan không xác định được
    if not has_narrative and query_text:
        q_lower = str(query_text).lower()
        narrative_keywords = [
            "sau đó", "tiếp ngay sau đó", "tiếp theo", "kế tiếp", 
            "đoạn sau", "rồi", "về đích", "trước đó", "biết sau đó"
        ]
        has_narrative = any(kw in q_lower for kw in narrative_keywords) or (
            len(query_text.split()) > 15 and ("." in query_text or "," in query_text)
        )

    # Step 2: Đếm số lượng candidate của từng video_id
    video_cand_counts = {}
    for cand in candidate_list:
        # Hỗ trợ lấy video_id trực tiếp hoặc trích xuất từ metadata nếu v_id chưa được map
        v_id = cand.get("video_id") or cand.get("extra_info", {}).get("video_id", "")
        if v_id:
            video_cand_counts[v_id] = video_cand_counts.get(v_id, 0) + 1

    # Step 3: Gán điểm thưởng Narrative Bonus
    for cand in candidate_list:
        v_id = cand.get("video_id") or cand.get("extra_info", {}).get("video_id", "")
        count = video_cand_counts.get(v_id, 1) if v_id else 1
        
        if has_narrative and count >= 3:
            cand["narrative_bonus"] = 0.06
        elif has_narrative and count == 2:
            cand["narrative_bonus"] = 0.03
        else:
            cand["narrative_bonus"] = 0.0

    return candidate_list

    
# def compute_video_narrative_bonus(candidate_list, query_text):
#     """
#     Cộng điểm thưởng gắn kết cấp Video (Video-level Narrative/Coherence Bonus) cho Standard Query:
#     - Nếu câu truy vấn có tính chất diễn biến thời gian (chứa 'sau đó', 'tiếp theo', 'về đích', hoặc nhiều mệnh đề)
#     - Một video có nhiều ứng viên xuất hiện rải rác chứng minh video đó chứa cả chuỗi diễn biến
#     - Frame của video này được cộng thêm narrative_bonus (0.03 - 0.08) để đẩy lên Top 1
#     """
#     if not candidate_list or not query_text:
#         return candidate_list

#     q_lower = str(query_text).lower()
#     has_narrative = any(kw in q_lower for kw in [
#         "sau đó","tiếp ngay sau đó", "tiếp theo", "kế tiếp", "đoạn sau", "rồi", "về đích", "trước đó", "biết sau đó"
#     ]) or (len(query_text.split()) > 15 and ("." in query_text or "," in query_text))

#     # Đếm số lượng candidate của từng video_id
#     video_cand_counts = {}
#     for cand in candidate_list:
#         v_id = cand.get("video_id", "")
#         if v_id:
#             video_cand_counts[v_id] = video_cand_counts.get(v_id, 0) + 1

#     for cand in candidate_list:
#         v_id = cand.get("video_id", "")
#         count = video_cand_counts.get(v_id, 1)
#         if has_narrative and count >= 3:
#             cand["narrative_bonus"] = 0.06
#         elif has_narrative and count == 2:
#             cand["narrative_bonus"] = 0.03
#         else:
#             cand["narrative_bonus"] = 0.0

#     return candidate_list

def has_meaningful_text(cand) -> bool:
    """
    Kiem tra candidate co du lieu van ban thuc su (OCR, ASR, Metadata chi tiet) hay khong.
    Dung de phat hien candidate thuan thi giac (Visual-only) nham loai bo hinh phat thieu text.
    """
    # 1. OCR
    ocr_texts = cand.get('ocr_texts', [])
    if ocr_texts:
        for item in ocr_texts:
            t = item.get('text', '') if isinstance(item, dict) else str(item)
            if len(t.strip()) >= 3:
                return True

    # 2. ASR
    asr_val = cand.get('asr_text', '')
    if isinstance(asr_val, list):
        asr_str = ' '.join(str(x) for x in asr_val)
    else:
        asr_str = str(asr_val)
    if len(asr_str.strip().split()) >= 3:
        return True

    # 3. Metadata
    metadata = cand.get('metadata', {})
    title = str(metadata.get('title', '')).strip()
    desc = str(metadata.get('description', '')).strip()
    if len(title.split()) >= 4 or len(desc.split()) >= 6:
        return True

    return False


def build_doc_from_candidate(cand, max_ocr_words=25, max_asr_words=25, max_desc_words=35) -> str:
    doc_parts = []
    
    # 1. Metadata (Title + Description)
    metadata = cand.get('metadata', {})
    title = str(metadata.get('title', '')).strip()
    if title: 
        doc_parts.append(f"Title: {title}")
        
    description = str(metadata.get('description', '')).strip()
    if description:
        clean_desc = ' '.join(description.split()[:max_desc_words])
        doc_parts.append(f"Description: {clean_desc}")
    
    # 2. OCR (Siet gon 10-15 tu, chi lay text co confidence >= 0.5)
    ocr_texts = []
    for item in cand.get('ocr_texts', []):
        text = ""
        if isinstance(item, dict) and item.get('score', 1.0) >= 0.5:
            text = str(item.get('text', '')).strip()
        elif isinstance(item, str):
            text = item.strip()
            
        if len(text) >= 2:
            ocr_texts.append(text)
            
    clean_ocr = ' '.join(' '.join(ocr_texts).split()[:max_ocr_words])
    if clean_ocr: 
        doc_parts.append(f"OCR: {clean_ocr}")
    
    # 3. ASR (Siet gon 20-25 tu)
    asr_val = cand.get('asr_text', '')
    if isinstance(asr_val, list):
        asr_str = ' '.join([str(x).strip() for x in asr_val if str(x).strip()])
    else:
        asr_str = str(asr_val).strip()
    
    clean_asr = ' '.join(asr_str.split()[:max_asr_words])
    if clean_asr: 
        doc_parts.append(f"ASR: {clean_asr}")
        
    # 4. Objects (Toi da 8 thuc the)
    objects = cand.get('object_entities', [])
    if objects:
        clean_objs = ', '.join([str(o) for o in objects[:8]])
        doc_parts.append(f"Objects: {clean_objs}")
    
    doc_text = " | ".join(doc_parts)
    return doc_text.strip()


def rerank_with_bge_m3(query_text, candidate_list, top_n=30, alpha=0.75):
    """
    Rerank voi Cross-Encoder BGE-M3 ket hop co che Zero-Penalty for Missing Text:
    - Candidate KHONG co du lieu van ban y nghia (Visual-only Ground Truth):
      Bao toan 100% final_score tu Stage 1, tuyet doi khong phat tru diem vi thieu text.
    - Candidate CO du lieu van ban y nghia:
      Chay Cross-Encoder va hoa tron: alpha * stage1_score + (1.0 - alpha) * ce_score.
    """
    if not candidate_list or top_n <= 0:
        return candidate_list

    to_rerank = candidate_list[:top_n]
    rest = candidate_list[top_n:]

    # Loc cac candidate co text thuc su de dua vao Cross-Encoder
    text_cands_info = []
    for i, cand in enumerate(to_rerank):
        if has_meaningful_text(cand):
            doc_str = build_doc_from_candidate(cand)
            if doc_str:
                text_cands_info.append((i, cand, doc_str))

    # Neu khong candidate nao trong top_n co text -> Zero-penalty (giu nguyen Stage 1)
    if not text_cands_info:
        print("[INFO] BGE Cross-Encoder: Khong co candidate nao chua text -> Zero Penalty (Giu nguyen Stage 1).")
        for cand in to_rerank:
            cand['ce_score'] = cand['final_score']
        return candidate_list

    # Chuan bi batch cho Cross-Encoder
    pairs = [[query_text, doc_str] for (_, _, doc_str) in text_cands_info]
    model = get_bge_reranker()
    raw_logits = model.predict(pairs, batch_size=32, show_progress_bar=False)
    
    # Sigmoid ep logit ve [0, 1]
    sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_logits))

    text_indices_set = set()

    # 1. Cap nhat diem cho candidate CO text
    for pair_idx, (orig_idx, cand, _) in enumerate(text_cands_info):
        text_indices_set.add(orig_idx)
        ce_score = float(sigmoid_scores[pair_idx])
        stage1_score = cand['final_score']
        cand['ce_score'] = ce_score
        cand['final_score'] = float(alpha * stage1_score + (1.0 - alpha) * ce_score)

    # 2. ZERO-PENALTY: Candidate KHONG co text (thuan visual)
    # Bao toan tuyet doi final_score cua Stage 1, khong bi tru 25% diem vo co
    for i, cand in enumerate(to_rerank):
        if i not in text_indices_set:
            stage1_score = cand['final_score']
            cand['ce_score'] = stage1_score
            cand['final_score'] = float(stage1_score)

    return sorted(to_rerank + rest, key=lambda x: x['final_score'], reverse=True)

def detect_query_intent(query_text: str) -> str:
    """
    Phan loai y dinh truy van (Query Intent Classification) cho bai toan Video Retrieval:
    1. OCR_INTENT: Co dau ngoac kep hoac tu khoa chi dinh van ban/bang bieu ro rang.
    2. ASR_INTENT: Co tu khoa chi dinh loi noi, phat bieu, am thanh.
    3. PURE_VISUAL: Mo ta hanh dong, mau sac, khung canh thi giac thuan tuy (~80% de thi).
    """
    if not query_text or not isinstance(query_text, str):
        return "PURE_VISUAL"

    q_lower = query_text.lower()

    # 1. Kiem tra dau ngoac kep (trich dan chu cu the tren man hinh / slide)
    has_quotes = bool(re.search(r'["“][^"”]{2,}["”]', query_text)) or bool(re.search(r"'[^']{2,}'", query_text))

    # 2. Tu khoa OCR chat che (tranh tu gay nhieu nhu 'ghi hinh', 'hien thi')
    ocr_strict_kws = [
        "dòng chữ", "biển báo", "biển hiệu", "bảng tên", "bảng hiệu",
        "bảng tượng trưng", "tiêu đề", "logo", "ghi chữ", "viết chữ",
        "ghi là", "viết là", "nội dung:", "nội dung :", "áo in chữ",
        "slide bài giảng", "banner", "poster", "chữ nổi"
    ]
    has_ocr_kw = any(kw in q_lower for kw in ocr_strict_kws)
    if not has_ocr_kw:
        # Kiem tra tu 'chữ' doc lap (loai tru 'hinh chu nhat', 'chu u'...)
        if re.search(r'\b(chữ)\b', q_lower) and not any(x in q_lower for x in ["hình chữ nhật", "chữ u", "chữ v", "chữ t"]):
            has_ocr_kw = True

    if has_quotes or has_ocr_kw:
        return "OCR_INTENT"

    # 3. Tu khoa ASR chat che (tranh tu 'cau', 'goi' dung rieng)
    asr_strict_kws = [
        "nói rằng", "bảo rằng", "giọng nói", "hát", "hỏi rằng",
        "trả lời rằng", "phát biểu", "chia sẻ rằng", "lời bài hát",
        "giảng giải về", "thuyết minh rằng"
    ]
    has_asr_kw = any(kw in q_lower for kw in asr_strict_kws)
    if has_asr_kw:
        return "ASR_INTENT"

    return "PURE_VISUAL"


def weighted_score_fusion(
    candidate_list,
    query_text=None,
    weights=None,
):
    """
    Weighted semantic fusion voi Intent-based Dynamic Weights va Zero-penalty for missing text.
    """
    if weights is None:
        weights = {
            "retrieval": 0.60,
            "object": 0.15,
            "metadata": 0.15,
            "ocr": 0.05,
            "asr": 0.05,
        }
    else:
        # Tao ban sao de tranh lam thay doi dict weights goc truyen vao
        weights = weights.copy()

    # ----------------------------------------------------------
    # INTENT-BASED DYNAMIC WEIGHTS
    # ----------------------------------------------------------
    if query_text and isinstance(query_text, str):
        intent = detect_query_intent(query_text)
        print(f"[INFO] Query Intent Detected : {intent}")

        if intent == "OCR_INTENT":
            weights = {
                "retrieval": 0.4,
                "ocr": 0.25,
                "asr": 0.05,
                "metadata": 0.25,
                "object": 0.05,
            }
        elif intent == "ASR_INTENT":
            weights = {
                "retrieval": 0.4,
                "ocr": 0.05,
                "asr": 0.25,
                "metadata": 0.25,
                "object": 0.05,
            }
        else:
            # PURE_VISUAL: Uu tien toi da Visual Retrieval (SigLIP2 / DFN5B)
            # Triet tieu hoan toan nguy co candidate sai co text trung lap cuop ngoi Top 1
            weights = {
                "retrieval": 0.7,
                "object": 0.05,
                "metadata": 0.15,
                "ocr": 0.05,
                "asr": 0.05,
            }
    # --- TÍNH TOÁN ĐIỂM SỐ CHO TỪNG CANDIDATE ---
    for cand in candidate_list:
        # Sử dụng .get() linh hoạt để dự phòng cả tên có _norm lẫn không có _norm
        retrieval = cand.get("retrieval_score_normalized", cand.get("retrieval_score", 0.0))
        object_score = cand.get("object_score_norm", cand.get("object_score", 0.0))
        metadata = cand.get("metadata_score_norm", cand.get("metadata_score", 0.0))
        ocr = cand.get("ocr_score_norm", cand.get("ocr_score", 0.0))
        asr = cand.get("asr_score_norm", cand.get("asr_score", 0.0))
        narrative_bonus = cand.get("narrative_bonus", 0.0)

        ##################################################
        # ZERO-PENALTY FOR MISSING TEXT (FUSION LEVEL)
        ##################################################
        # Neu candidate khong co OCR / ASR (thuan visual),
        # tai phan bo trong so text sang retrieval de khong bi tru diem vo co
        w_ret = weights.get("retrieval", 0.4)
        w_ocr = weights.get("ocr", 0.2)
        w_asr = weights.get("asr", 0.15)
        w_meta = weights.get("metadata", 0.15)
        w_obj = weights.get("object", 0.1)

        has_cand_ocr = bool(cand.get("ocr_texts"))
        has_cand_asr = bool(cand.get("asr_text"))

        if not has_cand_ocr and ocr == 0.0:
            w_ret += w_ocr
            w_ocr = 0.0

        if not has_cand_asr and asr == 0.0:
            w_ret += w_asr
            w_asr = 0.0

        # Agreement chi tinh tren cac modality thuc su co du lieu
        active_scores = [retrieval, object_score, metadata]
        if has_cand_ocr and ocr > 0.0:
            active_scores.append(ocr)
        if has_cand_asr and asr > 0.0:
            active_scores.append(asr)
        agreement = np.mean(active_scores)

        ##################################################

        final_score = (
            w_ret * retrieval
            + w_ocr * ocr
            + w_asr * asr
            + w_meta * metadata
            + w_obj * object_score
            + 0.05 * agreement
            + narrative_bonus
        )

        cand["agreement_score"] = float(agreement)
        cand["final_score"] = float(final_score)

    return sorted(
        candidate_list,
        key=lambda x: x["final_score"],
        reverse=True
    )





# ==========================================================
# RERANKING PIPELINE
# ==========================================================

def reranking_pipeline(
    query_text,
    candidate_list,
    config_path,
    query_en=None
):
    print_ram("Start")
    if not candidate_list:
        print("[INFO] Empty candidate list.")
        return []

    cfg = load_rerank_config(config_path)

    # Lấy ocr_threshold từ file yaml, mặc định là 0.5 nếu không khai báo
    ocr_threshold = cfg.get("ocr_threshold", 0.5)

    # Chuẩn bị query tiếng Anh cho Object matching (tận dụng hàm dịch có sẵn của retrieval_pipeline)
    if not query_en:
        query_en = get_translated_query_en(query_text)

    print(f"[INFO] Reranking - VI Query: '{query_text}' | EN Query: '{query_en}'")

    query_emb = encode_query(query_text)
    query_emb_en = encode_query(query_en) if (query_en and query_en != query_text) else query_emb
    print_ram("After encode_query")

    # Thực thi tuần tự các bước chuẩn hóa và tính toán điểm số
    candidate_list = normalize_retrieval_scores(candidate_list)
    print_ram("After normalize_retrieval")

   

    # 1. Hybrid Object Score (Query tiếng Anh + Entity Lexical Match)
    candidate_list = compute_object_scores(
        query_emb_en=query_emb_en,
        candidate_list=candidate_list,
        query_en=query_en
    )
    print_ram("After object")

    # 2. Hybrid Metadata Score (Semantic E5 + Title/Keywords Lexical BM25+)
    candidate_list = compute_metadata_scores(
        query_emb=query_emb,
        candidate_list=candidate_list,
        query_text=query_text
    )
    print_ram("After metadata")

    # 3. Hybrid OCR Score (Semantic E5 + Char 3-Gram BM25+)
    candidate_list = compute_ocr_scores(
        query_emb=query_emb,
        candidate_list=candidate_list,
        query_text=query_text,
        ocr_threshold=ocr_threshold
    )
    print_ram("After OCR")

    # 4. Hybrid ASR Score (Semantic E5 + Word-Level BM25+)
    candidate_list = compute_asr_scores(
        query_emb=query_emb,
        candidate_list=candidate_list,
        query_text=query_text
    )
    print_ram("After ASR")

    # 5. Điểm thưởng nhất quán chuỗi cấp Video (Video-level Narrative Bonus)
    candidate_list = compute_video_narrative_bonus(candidate_list, query_text)

    # ==========================================================
    # CHUẨN HÓA FEATURE TRƯỚC KHI FUSION
    # ==========================================================
    candidate_list = normalize_feature(candidate_list, "object_score")
    candidate_list = normalize_feature(candidate_list, "metadata_score")
    candidate_list = normalize_feature(candidate_list, "ocr_score")
    candidate_list = normalize_feature(candidate_list, "asr_score")
  
    # ------------------------------------------------------
    # DEBUG SCORE TABLE
    # ------------------------------------------------------
    print("\n")
    print("=" * 110)
    print("RERANKING SCORE DEBUG (HYBRID MULTI-MODAL)")
    print("=" * 110)

    for i, cand in enumerate(candidate_list[:10], start=1):
        print(
            f"{i:02d} | "
            f"{cand.get('video_id', 'N/A')} | "
            f"KF={cand.get('keyframe_index', cand.get('frame_idx', 0)):4d} | "
            f"CLIP={cand.get('retrieval_score_normalized', 0.0):.4f} | "
            f"OBJ={cand.get('object_score_norm', 0.0):.4f} | "
            f"META={cand.get('metadata_score_norm', 0.0):.4f} | "
            f"OCR={cand.get('ocr_score_norm', 0.0):.4f} | "
            f"ASR={cand.get('asr_score_norm', 0.0):.4f} | "
            f"NarrBonus={cand.get('narrative_bonus', 0.0):.2f}"
        )

    print("=" * 110)

    # Thực hiện trộn điểm tổng hợp dựa trên weights trong yaml
    candidate_list = weighted_score_fusion(
        candidate_list,
        query_text = query_text,
        weights=cfg.get("weights")
    )   
    print_ram("After Fusion")
    
    # Rerank với Cross-Encoder (nếu kích hoạt)
    # candidate_list = rerank_with_bge_m3(query_text, candidate_list, top_n=30, alpha=0.75)
    # print_ram("After Cross-Encoder")

    # Đóng gói chuẩn xác lại danh sách kết quả trả về cho hệ thống (UI/FastAPI)
    results = []

    for rank, cand in enumerate(candidate_list, start=1):
        results.append({
            "rank": rank,
            "vector_index": cand["vector_index"],
            "video_id": cand["video_id"],
            "frame_idx": cand.get("frame_idx", cand.get("keyframe_index", 0)),
            "keyframe_index": cand.get("keyframe_index", 0),
            "pts_time": cand.get("pts_time", 0.0),
            "fps": cand.get("fps", 25.0),
            "keyframe_path": cand.get("keyframe_path", ""),
            "object_path": cand.get("object_path", ""),
            "metadata_path": cand.get("metadata_path", ""),
            "retrieval_score": cand["retrieval_score"],
            "retrieval_score_normalized": cand["retrieval_score_normalized"],
            "object_score": cand["object_score"],
            "object_semantic_score": cand.get("object_semantic_score", 0.0),
            "object_lexical_score": cand.get("object_lexical_score", 0.0),
            "metadata_score": cand["metadata_score"],
            "metadata_semantic_score": cand.get("metadata_semantic_score", 0.0),
            "metadata_bm25_score": cand.get("metadata_bm25_score", 0.0),
            "ocr_score": cand.get("ocr_score", 0.0),
            "ocr_semantic_score": cand.get("ocr_semantic_score", 0.0),
            "ocr_bm25_score": cand.get("ocr_bm25_score", cand.get("ocr_fuzzy_score", 0.0)),
            "asr_score": cand.get("asr_score", 0.0),
            "asr_semantic_score": cand.get("asr_semantic_score", 0.0),
            "asr_bm25_score": cand.get("asr_bm25_score", cand.get("asr_fuzzy_score", 0.0)),
            "final_score": cand["final_score"],
            "object_entities": cand.get("object_entities", []),
            "ocr_texts": cand.get("ocr_texts", []),
            "asr_text": cand.get("asr_text", ""),
            "metadata": cand.get("metadata", {})
        })

    print(
        f"[INFO] Reranking completed "
        f"({len(results)} results)."
    )

    return results
