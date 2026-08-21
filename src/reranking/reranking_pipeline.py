
# import os
# import yaml
# import json
# import numpy as np
# import faiss
# import torch
# import torch.nn.functional as F

# from transformers import AutoTokenizer, AutoModel
# from src.retrieval.retrieval_pipeline import (
#     siglip2_retrieval_pipeline,
#     clip_b32_retrieval_pipeline,
# )
# import sys
# from pathlib import Path

# def topk_average(scores, k=3):

#     if not scores:
#         return 0.0

#     scores = sorted(scores, reverse=True)

#     k = min(k, len(scores))

#     return float(np.mean(scores[:k]))
# # ==========================================================
# # PATH
# # ==========================================================

# ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT))

# # def get_object_index_components():

# #     global _object_index
# #     global _object_mapping

# #     if _object_index is None:

# #         print(
# #             f"[INFO] Loading object FAISS index: "
# #             f"{OBJECT_INDEX_PATH}"
# #         )

# #         if not OBJECT_INDEX_PATH.exists():
# #             raise FileNotFoundError(
# #                 f"Object index not found:\n"
# #                 f"{OBJECT_INDEX_PATH}"
# #             )

# #         if not OBJECT_MAPPING_PATH.exists():
# #             raise FileNotFoundError(
# #                 f"Object mapping not found:\n"
# #                 f"{OBJECT_MAPPING_PATH}"
# #             )

# #         _object_index = faiss.read_index(
# #             str(OBJECT_INDEX_PATH)
# #         )

# #         print(
# #             f"[INFO] Object index loaded: "
# #             f"{_object_index.ntotal} vectors"
# #         )

# #         with open(
# #             OBJECT_MAPPING_PATH,
# #             "r",
# #             encoding="utf-8"
# #         ) as f:

# #             _object_mapping = json.load(f)

# #         print(
# #             f"[INFO] Object mapping loaded: "
# #             f"{len(_object_mapping)} entries"
# #         )

# #         if (
# #             _object_index.ntotal
# #             != len(_object_mapping)
# #         ):

# #             raise ValueError(
# #                 "Object index and object mapping "
# #                 "have different sizes: "
# #                 f"{_object_index.ntotal} vs "
# #                 f"{len(_object_mapping)}"
# #             )

# #     return (
# #         _object_index,
# #         _object_mapping
# #     )
# # ==========================================================
# # CONFIG
# # ==========================================================

# def load_rerank_config(config_path):

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(
#             f"Cannot find config file:\n{config_path}"
#         )

#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     return cfg



# # ==========================================================
# # E5 MODEL
# # ==========================================================

# _e5_model = None
# _e5_tokenizer = None

# _device = "cuda" if torch.cuda.is_available() else "cpu"
# # ==========================================================
# # OBJECT VECTOR INDEX
# # ==========================================================

# _OBJECT_INDEX = None
# _OBJECT_MAPPING = None

# OBJECT_INDEX_PATH = (
#     ROOT / "data" / "indexes" / "object.index"
# )

# OBJECT_MAPPING_PATH = (
#     ROOT / "data" / "indexes" / "object_mapping.json"
# )
# def get_object_index():

#     global _OBJECT_INDEX
#     global _OBJECT_MAPPING

#     if _OBJECT_INDEX is not None:
#         return (
#             _OBJECT_INDEX,
#             _OBJECT_MAPPING
#         )

#     print(
#         f"[INFO] Loading object FAISS index: "
#         f"{OBJECT_INDEX_PATH}"
#     )

#     if not OBJECT_INDEX_PATH.exists():
#         raise FileNotFoundError(
#             f"Object index not found:\n"
#             f"{OBJECT_INDEX_PATH}"
#         )

#     if not OBJECT_MAPPING_PATH.exists():
#         raise FileNotFoundError(
#             f"Object mapping not found:\n"
#             f"{OBJECT_MAPPING_PATH}"
#         )

#     # ------------------------------------------------------
#     # Load FAISS index ONCE
#     # ------------------------------------------------------

#     _OBJECT_INDEX = faiss.read_index(
#         str(OBJECT_INDEX_PATH)
#     )

#     # ------------------------------------------------------
#     # Load mapping ONCE
#     # ------------------------------------------------------

#     with open(
#         OBJECT_MAPPING_PATH,
#         "r",
#         encoding="utf-8"
#     ) as f:

#         _OBJECT_MAPPING = json.load(f)

#     print(
#         f"[INFO] Object index loaded: "
#         f"{_OBJECT_INDEX.ntotal} vectors"
#     )

#     print(
#         f"[INFO] Object mapping loaded: "
#         f"{len(_OBJECT_MAPPING)} entries"
#     )

#     # ------------------------------------------------------
#     # Safety check
#     # ------------------------------------------------------

#     if (
#         _OBJECT_INDEX.ntotal
#         != len(_OBJECT_MAPPING)
#     ):

#         raise ValueError(
#             "Object index and object mapping "
#             "have different sizes: "
#             f"{_OBJECT_INDEX.ntotal} vs "
#             f"{len(_OBJECT_MAPPING)}"
#         )

#     return (
#         _OBJECT_INDEX,
#         _OBJECT_MAPPING
#     )

# # ==========================================================
# # OBJECT MAPPING LOOKUP
# # ==========================================================

# _OBJECT_LOOKUP = None


# def get_object_lookup():

#     global _OBJECT_LOOKUP

#     if _OBJECT_LOOKUP is not None:
#         return _OBJECT_LOOKUP

#     _, object_mapping = get_object_index()

#     print(
#         "[INFO] Building object lookup table..."
#     )

#     lookup = {}

#     for item in object_mapping:

#         video_id = item.get(
#             "video_id",
#             ""
#         )

#         keyframe_index = item.get(
#             "keyframe_index",
#             0
#         )

#         vector_index = item.get(
#             "vector_index"
#         )

#         if vector_index is None:
#             continue

#         key = (
#             video_id,
#             keyframe_index
#         )

#         if key not in lookup:
#             lookup[key] = []

#         lookup[key].append(
#             int(vector_index)
#         )

#     _OBJECT_LOOKUP = lookup

#     print(
#         f"[INFO] Object lookup contains "
#         f"{len(lookup)} keyframes."
#     )

#     return _OBJECT_LOOKUP

# def get_object_vectors(
#     object_index,
#     vector_indices
# ):

#     vectors = []

#     for vector_index in vector_indices:

#         vector_index = int(vector_index)

#         # --------------------------------------------------
#         # Already reconstructed
#         # --------------------------------------------------

#         if vector_index in _OBJECT_VECTOR_CACHE:

#             vectors.append(
#                 _OBJECT_VECTOR_CACHE[
#                     vector_index
#                 ]
#             )

#             continue

#         # --------------------------------------------------
#         # Reconstruct ONCE
#         # --------------------------------------------------

#         vector = object_index.reconstruct(
#             vector_index
#         )

#         vector = np.asarray(
#             vector,
#             dtype=np.float32
#         )

#         # --------------------------------------------------
#         # Cache
#         # --------------------------------------------------

#         _OBJECT_VECTOR_CACHE[
#             vector_index
#         ] = vector

#         vectors.append(
#             vector
#         )

#     return np.vstack(
#         vectors
#     )

# def get_e5_components():

#     global _e5_model
#     global _e5_tokenizer

#     if _e5_model is None:

#         model_name = "intfloat/multilingual-e5-base"

#         print(
#             f"[INFO] Loading {model_name} for E5 reranking..."
#         )

#         _e5_tokenizer = AutoTokenizer.from_pretrained(
#             model_name
#         )

#         _e5_model = AutoModel.from_pretrained(
#             model_name
#         ).to(_device)

#         _e5_model.eval()

#     return _e5_tokenizer, _e5_model


# # ==========================================================
# # E5 ENCODING
# # ==========================================================

# def average_pool(
#     last_hidden_state,
#     attention_mask
# ):

#     last_hidden = last_hidden_state.masked_fill(
#         ~attention_mask[..., None].bool(),
#         0.0
#     )

#     return last_hidden.sum(dim=1) / attention_mask.sum(
#         dim=1
#     )[..., None]


# def encode_e5(
#     texts,
#     prefix,
#     batch_size=32,
# ):

#     if not texts:

#         return np.empty(
#             (0, 768),
#             dtype=np.float32
#         )

#     tokenizer, model = get_e5_components()

#     embeddings = []

#     formatted_texts = [
#         f"{prefix}: {text}"
#         for text in texts
#     ]

#     for start in range(
#         0,
#         len(formatted_texts),
#         batch_size
#     ):

#         batch = formatted_texts[
#             start:start + batch_size
#         ]

#         inputs = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512
#         )

#         inputs = {
#             k: v.to(_device)
#             for k, v in inputs.items()
#         }

#         with torch.no_grad():

#             outputs = model(**inputs)

#             emb = average_pool(
#                 outputs.last_hidden_state,
#                 inputs["attention_mask"]
#             )

#             emb = F.normalize(
#                 emb,
#                 p=2,
#                 dim=1
#             )

#             embeddings.append(
#                 emb.cpu()
#             )

#     embeddings = torch.cat(
#         embeddings,
#         dim=0
#     )

#     return embeddings.numpy().astype(
#         np.float32
#     )


# # ==========================================================
# # QUERY EMBEDDING
# # ==========================================================

# def encode_query(
#     query_text
# ):

#     return encode_e5(
#         [query_text],
#         prefix="query",
#         batch_size=1
#     )[0]


# def compute_object_scores(
#     query_emb,
#     candidate_list
# ):

#     print(
#         "[INFO] Computing object scores "
#         "from cached FAISS vectors..."
#     )

#     if not candidate_list:
#         return candidate_list

#     # ------------------------------------------------------
#     # Load cached index + lookup
#     # ------------------------------------------------------

#     object_index, _ = get_object_index()
#     object_lookup = get_object_lookup()

#     # ------------------------------------------------------
#     # Prepare query embedding
#     # ------------------------------------------------------

#     query_emb = np.asarray(
#         query_emb,
#         dtype=np.float32
#     ).reshape(1, -1)

#     query_norm = np.linalg.norm(
#         query_emb,
#         axis=1,
#         keepdims=True
#     )

#     query_norm = np.maximum(
#         query_norm,
#         1e-12
#     )

#     query_emb = (
#         query_emb / query_norm
#     )

#     # ------------------------------------------------------
#     # Statistics
#     # ------------------------------------------------------

#     total_object_vectors = 0

#     candidates_with_object = 0

#     candidates_without_object = 0

#     reconstructed_now = 0

#     cached_vectors = 0

#     # ------------------------------------------------------
#     # Process candidates
#     # ------------------------------------------------------

#     for cand in candidate_list:

#         video_id = cand.get(
#             "video_id",
#             ""
#         )

#         keyframe_index = int(
#             cand.get(
#                 "keyframe_index",
#                 0
#             )
#         )

#         key = (
#             video_id,
#             keyframe_index
#         )

#         vector_indices = object_lookup.get(
#             key,
#             []
#         )

#         # --------------------------------------------------
#         # No object
#         # --------------------------------------------------

#         if not vector_indices:

#             cand["object_score"] = 0.0

#             candidates_without_object += 1

#             continue

#         candidates_with_object += 1

#         total_object_vectors += len(
#             vector_indices
#         )

#         # --------------------------------------------------
#         # Check cache statistics
#         # --------------------------------------------------

#         for vector_index in vector_indices:

#             if int(vector_index) in _OBJECT_VECTOR_CACHE:

#                 cached_vectors += 1

#             else:

#                 reconstructed_now += 1

#         # --------------------------------------------------
#         # Get object vectors
#         # --------------------------------------------------

#         object_vectors = get_object_vectors(
#             object_index,
#             vector_indices
#         )

#         # --------------------------------------------------
#         # Cosine similarity
#         # --------------------------------------------------

#         similarities = (
#             object_vectors @ query_emb.T
#         ).reshape(-1)

#         # --------------------------------------------------
#         # Top-k average
#         # --------------------------------------------------

#         score = topk_average(
#             similarities.tolist(),
#             k=3
#         )

#         cand["object_score"] = round(
#             float(score),
#             4
#         )

#     # ------------------------------------------------------
#     # DEBUG
#     # ------------------------------------------------------

#     scores = [
#         cand.get(
#             "object_score",
#             0.0
#         )
#         for cand in candidate_list
#     ]

#     print(
#         "[DEBUG OBJECT]"
#     )

#     print(
#         f"Candidates                 : "
#         f"{len(candidate_list)}"
#     )

#     print(
#         f"Candidates with object     : "
#         f"{candidates_with_object}"
#     )

#     print(
#         f"Candidates without object  : "
#         f"{candidates_without_object}"
#     )

#     print(
#         f"Object vectors evaluated   : "
#         f"{total_object_vectors}"
#     )

#     print(
#         f"Vectors reconstructed now  : "
#         f"{reconstructed_now}"
#     )

#     print(
#         f"Vectors reused from cache  : "
#         f"{cached_vectors}"
#     )

#     print(
#         f"Object vector cache size   : "
#         f"{len(_OBJECT_VECTOR_CACHE)}"
#     )

#     print(
#         f"Score range                : "
#         f"{min(scores):.4f}"
#         f" -> "
#         f"{max(scores):.4f}"
#     )

#     return candidate_list
# # ==========================================================
# # METADATA VECTOR INDEX
# # ==========================================================

# _METADATA_INDEX = None
# _METADATA_MAPPING = None

# METADATA_INDEX_PATH = (
#     ROOT / "data" / "indexes" / "metadata.index"
# )

# METADATA_MAPPING_PATH = (
#     ROOT / "data" / "indexes" / "metadata_mapping.json"
# )

# # ==========================================================
# # RUNTIME CACHE
# # ==========================================================

# _OBJECT_INDEX = None
# _OBJECT_MAPPING = None
# _OBJECT_LOOKUP = None

# _METADATA_INDEX = None
# _METADATA_MAPPING = None
# _METADATA_LOOKUP = None


# _OBJECT_VECTOR_CACHE = {}
# _METADATA_VECTOR_CACHE = {}



# def get_metadata_index():

#     global _METADATA_INDEX
#     global _METADATA_MAPPING

#     if _METADATA_INDEX is not None:

#         return (
#             _METADATA_INDEX,
#             _METADATA_MAPPING
#         )

#     print(
#         f"[INFO] Loading metadata FAISS index: "
#         f"{METADATA_INDEX_PATH}"
#     )

#     if not METADATA_INDEX_PATH.exists():
#         raise FileNotFoundError(
#             f"Metadata index not found:\n"
#             f"{METADATA_INDEX_PATH}"
#         )

#     if not METADATA_MAPPING_PATH.exists():
#         raise FileNotFoundError(
#             f"Metadata mapping not found:\n"
#             f"{METADATA_MAPPING_PATH}"
#         )

#     _METADATA_INDEX = faiss.read_index(
#         str(METADATA_INDEX_PATH)
#     )

#     with open(
#         METADATA_MAPPING_PATH,
#         "r",
#         encoding="utf-8"
#     ) as f:

#         _METADATA_MAPPING = json.load(f)

#     print(
#         f"[INFO] Metadata index loaded: "
#         f"{_METADATA_INDEX.ntotal} vectors"
#     )

#     print(
#         f"[INFO] Metadata mapping loaded: "
#         f"{len(_METADATA_MAPPING)} entries"
#     )

#     if (
#         _METADATA_INDEX.ntotal
#         != len(_METADATA_MAPPING)
#     ):

#         raise ValueError(
#             "Metadata index and metadata mapping "
#             "have different sizes: "
#             f"{_METADATA_INDEX.ntotal} vs "
#             f"{len(_METADATA_MAPPING)}"
#         )

#     return (
#         _METADATA_INDEX,
#         _METADATA_MAPPING
#     )
# # ==========================================================
# # METADATA MAPPING LOOKUP
# # ==========================================================

# _METADATA_LOOKUP = None

# def get_metadata_lookup():

#     global _METADATA_LOOKUP

#     if _METADATA_LOOKUP is not None:
#         return _METADATA_LOOKUP

#     _, metadata_mapping = get_metadata_index()

#     print(
#         "[INFO] Building metadata lookup table..."
#     )

#     lookup = {}

#     for item in metadata_mapping:

#         video_id = item.get(
#             "video_id",
#             ""
#         )

#         keyframe_index = item.get(
#             "keyframe_index"
#         )

#         vector_index = item.get(
#             "vector_index"
#         )

#         if (
#             keyframe_index is None
#             or vector_index is None
#         ):
#             continue

#         key = (
#             video_id,
#             int(keyframe_index)
#         )

#         lookup[key] = int(
#             vector_index
#         )

#     _METADATA_LOOKUP = lookup

#     print(
#         f"[INFO] Metadata lookup contains "
#         f"{len(lookup)} keyframes."
#     )

#     return _METADATA_LOOKUP

# def get_metadata_vector(
#     metadata_index,
#     vector_index
# ):

#     vector_index = int(
#         vector_index
#     )

#     if vector_index in _METADATA_VECTOR_CACHE:

#         return _METADATA_VECTOR_CACHE[
#             vector_index
#         ]

#     vector = metadata_index.reconstruct(
#         vector_index
#     )

#     vector = np.asarray(
#         vector,
#         dtype=np.float32
#     )

#     # Normalize once
#     norm = np.linalg.norm(vector)

#     if norm > 1e-12:

#         vector = vector / norm

#     _METADATA_VECTOR_CACHE[
#         vector_index
#     ] = vector

#     return vector
# # ==========================================================
# # OCR SEMANTIC SCORE - MAXSIM
# # ==========================================================

# def compute_ocr_scores(
#     query_emb,
#     candidate_list,
#     batch_size=32
# ):

#     print(
#         "[INFO] Computing OCR scores using E5..."
#     )

#     if not candidate_list:
#         return candidate_list


#     query_emb = query_emb.reshape(
#         1,
#         -1
#     )

#     ocr_texts = []
#     ocr_candidate_ids = []

#     for cand_idx, cand in enumerate(
#         candidate_list
#     ):

#         texts = cand.get(
#             "ocr_texts",
#             []
#         )

#         if not isinstance(
#             texts,
#             list
#         ):

#             texts = []

#         for text in texts:

#             text = str(text).strip()

#             if not text:
#                 continue

#             ocr_texts.append(
#                 text
#             )

#             ocr_candidate_ids.append(
#                 cand_idx
#             )

#     if not ocr_texts:

#         for cand in candidate_list:

#             cand["ocr_score"] = 0.0

#         print(
#             "[DEBUG OCR] No OCR texts found."
#         )

#         return candidate_list

#     ocr_embeddings = encode_e5(
#         ocr_texts,
#         prefix="passage",
#         batch_size=batch_size
#     )

#     similarities = (
#         ocr_embeddings
#         @ query_emb.T
#     ).reshape(-1)

#     candidate_scores = [
#         []
#         for _ in candidate_list
#     ]

#     for sim, cand_idx in zip(
#         similarities,
#         ocr_candidate_ids
#     ):

#         candidate_scores[
#             cand_idx
#         ].append(
#             float(sim)
#         )

#     for cand_idx, cand in enumerate(
#         candidate_list
#     ):

#         scores = candidate_scores[
#             cand_idx
#         ]

#         # if scores:
#         #     score = max(scores)
#         # else:
#         #     score = 0.0
#         scores = candidate_scores[cand_idx]

#         score = topk_average(scores, k=3)

#         cand["ocr_score"] = round(score, 4)
        

#     # ------------------------------------------------------
#     # DEBUG
#     # ------------------------------------------------------

#     non_zero = sum(
#         1
#         for cand in candidate_list
#         if cand["ocr_score"] != 0
#     )

#     scores = [
#         cand["ocr_score"]
#         for cand in candidate_list
#     ]

#     print(
#         "[DEBUG OCR]"
#     )

#     print(
#         f"Candidates             : {len(candidate_list)}"
#     )

#     print(
#         f"Candidates with OCR    : {non_zero}"
#     )

#     print(
#         f"Score range            : "
#         f"{min(scores):.4f}"
#         f" -> "
#         f"{max(scores):.4f}"
#     )

#     return candidate_list

# # ==========================================================
# # METADATA SEMANTIC SCORE
# # USING PRECOMPUTED E5 METADATA INDEX
# # ==========================================================
# def compute_metadata_scores(
#     query_emb,
#     candidate_list
# ):

#     print(
#         "[INFO] Computing metadata scores "
#         "from cached FAISS vectors..."
#     )

#     if not candidate_list:
#         return candidate_list

#     # ------------------------------------------------------
#     # Load cached index + lookup
#     # ------------------------------------------------------

#     metadata_index, _ = get_metadata_index()

#     metadata_lookup = get_metadata_lookup()

#     # ------------------------------------------------------
#     # Prepare query
#     # ------------------------------------------------------

#     query_emb = np.asarray(
#         query_emb,
#         dtype=np.float32
#     ).reshape(1, -1)

#     query_norm = np.linalg.norm(
#         query_emb,
#         axis=1,
#         keepdims=True
#     )

#     query_norm = np.maximum(
#         query_norm,
#         1e-12
#     )

#     query_emb = (
#         query_emb / query_norm
#     )

#     # ------------------------------------------------------
#     # Statistics
#     # ------------------------------------------------------

#     candidates_with_metadata = 0

#     candidates_without_metadata = 0

#     reconstructed_now = 0

#     cached_vectors = 0

#     # ------------------------------------------------------
#     # Process candidates
#     # ------------------------------------------------------

#     for cand in candidate_list:

#         video_id = cand.get(
#             "video_id",
#             ""
#         )

#         keyframe_index = cand.get(
#             "keyframe_index"
#         )

#         if keyframe_index is None:

#             cand["metadata_score"] = 0.0

#             candidates_without_metadata += 1

#             continue

#         key = (
#             video_id,
#             int(keyframe_index)
#         )

#         vector_index = metadata_lookup.get(
#             key
#         )

#         # --------------------------------------------------
#         # No metadata
#         # --------------------------------------------------

#         if vector_index is None:

#             cand["metadata_score"] = 0.0

#             candidates_without_metadata += 1

#             continue

#         candidates_with_metadata += 1

#         vector_index = int(
#             vector_index
#         )

#         # --------------------------------------------------
#         # Cache statistics
#         # --------------------------------------------------

#         if vector_index in _METADATA_VECTOR_CACHE:

#             cached_vectors += 1

#         else:

#             reconstructed_now += 1

#         # --------------------------------------------------
#         # Get cached metadata vector
#         # --------------------------------------------------

#         metadata_vector = get_metadata_vector(
#             metadata_index,
#             vector_index
#         )

#         # --------------------------------------------------
#         # Cosine similarity
#         # --------------------------------------------------

#         similarity = float(
#             metadata_vector @ query_emb[0]
#         )

#         cand["metadata_score"] = round(
#             similarity,
#             4
#         )

#     # ------------------------------------------------------
#     # DEBUG
#     # ------------------------------------------------------

#     scores = [
#         cand.get(
#             "metadata_score",
#             0.0
#         )
#         for cand in candidate_list
#     ]

#     print(
#         "[DEBUG METADATA]"
#     )

#     print(
#         f"Candidates                  : "
#         f"{len(candidate_list)}"
#     )

#     print(
#         f"Candidates with metadata    : "
#         f"{candidates_with_metadata}"
#     )

#     print(
#         f"Candidates without metadata : "
#         f"{candidates_without_metadata}"
#     )

#     print(
#         f"Vectors reconstructed now   : "
#         f"{reconstructed_now}"
#     )

#     print(
#         f"Vectors reused from cache   : "
#         f"{cached_vectors}"
#     )

#     print(
#         f"Metadata vector cache size  : "
#         f"{len(_METADATA_VECTOR_CACHE)}"
#     )

#     print(
#         f"Score range                 : "
#         f"{min(scores):.4f}"
#         f" -> "
#         f"{max(scores):.4f}"
#     )

#     return candidate_list

# # ==========================================================
# # ASR SEMANTIC SCORE - MAXSIM
# # ==========================================================

# def compute_asr_scores(
#     query_emb,
#     candidate_list,
#     batch_size=32
# ):

#     print(
#         "[INFO] Computing ASR scores using E5..."
#     )

#     if not candidate_list:
#         return candidate_list


#     query_emb = query_emb.reshape(
#         1,
#         -1
#     )

#     asr_texts = []
#     asr_candidate_ids = []

#     for cand_idx, cand in enumerate(
#         candidate_list
#     ):

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

#             asr_texts.append(
#                 text
#             )

#             asr_candidate_ids.append(
#                 cand_idx
#             )

#     if not asr_texts:

#         for cand in candidate_list:

#             cand["asr_score"] = 0.0

#         print(
#             "[DEBUG ASR] No ASR texts found."
#         )

#         return candidate_list

#     asr_embeddings = encode_e5(
#         asr_texts,
#         prefix="passage",
#         batch_size=batch_size
#     )

#     similarities = (
#         asr_embeddings
#         @ query_emb.T
#     ).reshape(-1)

#     candidate_scores = [
#         []
#         for _ in candidate_list
#     ]

#     for sim, cand_idx in zip(
#         similarities,
#         asr_candidate_ids
#     ):

#         candidate_scores[
#             cand_idx
#         ].append(
#             float(sim)
#         )

#     for cand_idx, cand in enumerate(
#         candidate_list
#     ):

#         scores = candidate_scores[
#             cand_idx
#         ]

#         if scores:
#             score = max(scores)
#         else:
#             score = 0.0

#         cand["asr_score"] = round(
#             float(score),
#             4
#         )

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

#     print(
#         "[DEBUG ASR]"
#     )

#     print(
#         f"Candidates             : {len(candidate_list)}"
#     )

#     print(
#         f"Candidates with ASR    : {non_zero}"
#     )

#     print(
#         f"Score range            : "
#         f"{min(scores):.4f}"
#         f" -> "
#         f"{max(scores):.4f}"
#     )

#     return candidate_list


# # ==========================================================
# # RETRIEVAL SCORE NORMALIZATION
# # ==========================================================

# def normalize_retrieval_scores(
#     candidate_list
# ):

#     retrieval_scores = np.array(
#         [
#             cand["score"]
#             for cand in candidate_list
#         ],
#         dtype=np.float32
#     )

#     min_score = retrieval_scores.min()
#     max_score = retrieval_scores.max()

#     if (
#         max_score - min_score
#         > 1e-8
#     ):

#         normalized = (
#             retrieval_scores
#             - min_score
#         ) / (
#             max_score
#             - min_score
#         )

#     else:

#         normalized = np.ones_like(
#             retrieval_scores
#         )

#     for i, cand in enumerate(
#         candidate_list
#     ):

#         cand["retrieval_score"] = float(
#             cand["score"]
#         )

#         cand[
#             "retrieval_score_normalized"
#         ] = round(
#             float(normalized[i]),
#             4
#         )

#     return candidate_list

# # def normalize_feature(candidate_list, feature_name):
# #     values = np.array(
# #         [cand.get(feature_name, 0.0) for cand in candidate_list],
# #         dtype=np.float32
# #     )

# #     if len(values) == 0:
# #         return candidate_list

# #     min_v = values.min()
# #     max_v = values.max()

# #     if max_v - min_v > 1e-8:
# #         values = (values - min_v) / (max_v - min_v)
# #     else:
# #         values = np.ones_like(values)

# #     for cand, v in zip(candidate_list, values):
# #         cand[f"{feature_name}_norm"] = float(v)

# #     return candidate_list
# def normalize_feature(candidate_list, feature_name):

#     values = np.array(
#         [
#             cand.get(feature_name, 0.0)
#             for cand in candidate_list
#         ],
#         dtype=np.float32
#     )

#     if len(values) == 0:
#         return candidate_list

#     min_v = values.min()
#     max_v = values.max()

#     if max_v - min_v > 1e-8:
#         normalized = (
#             values - min_v
#         ) / (
#             max_v - min_v
#         )
#     else:
#         # Nếu toàn bộ score đều bằng 0
#         # thì feature này không cung cấp tín hiệu
#         if abs(min_v) < 1e-8:
#             normalized = np.zeros_like(values)
#         else:
#             # tất cả có cùng score > 0
#             normalized = np.ones_like(values)

#     for cand, v in zip(
#         candidate_list,
#         normalized
#     ):
#         cand[
#             f"{feature_name}_norm"
#         ] = float(v)

#     return candidate_list
# # ==========================================================
# # RRF
# # ==========================================================

# def rrf_score_fusion(
#     candidate_list,
#     k=60
# ):

#     """
#     RRF over:
#         1. CLIP retrieval
#         2. Object semantic score
#         3. Metadata semantic score
#         4. OCR MaxSim score
#         5. ASR MaxSim score 
#     """

#     sorted_by_clip = sorted(
#         candidate_list,
#         key=lambda x: x.get(
#             "score",
#             0.0
#         ),
#         reverse=True
#     )

#     clip_ranks = {
#         cand["vector_index"]: idx + 1
#         for idx, cand in enumerate(
#             sorted_by_clip
#         )
#     }

#     sorted_by_obj = sorted(
#         candidate_list,
#         key=lambda x: x.get(
#             "object_score",
#             0.0
#         ),
#         reverse=True
#     )

#     obj_ranks = {
#         cand["vector_index"]: idx + 1
#         for idx, cand in enumerate(
#             sorted_by_obj
#         )
#     }

#     sorted_by_meta = sorted(
#         candidate_list,
#         key=lambda x: x.get(
#             "metadata_score",
#             0.0
#         ),
#         reverse=True
#     )

#     meta_ranks = {
#         cand["vector_index"]: idx + 1
#         for idx, cand in enumerate(
#             sorted_by_meta
#         )
#     }

#     sorted_by_ocr = sorted(
#         candidate_list,
#         key=lambda x: x.get(
#             "ocr_score",
#             0.0
#         ),
#         reverse=True
#     )

#     ocr_ranks = {
#         cand["vector_index"]: idx + 1
#         for idx, cand in enumerate(
#             sorted_by_ocr
#         )
#     }

#     sorted_by_asr = sorted(
#         candidate_list,
#         key=lambda x: x.get(
#             "asr_score",
#             0.0
#         ),
#         reverse=True
#     )

#     asr_ranks = {
#         cand["vector_index"]: idx + 1
#         for idx, cand in enumerate(
#             sorted_by_asr
#         )
#     }

#     for cand in candidate_list:

#         v_idx = cand[
#             "vector_index"
#         ]

#         r_clip = clip_ranks[
#             v_idx
#         ]

#         r_obj = obj_ranks[
#             v_idx
#         ]

#         r_meta = meta_ranks[
#             v_idx
#         ]

#         r_ocr = ocr_ranks[
#             v_idx
#         ]

#         r_asr = asr_ranks[
#             v_idx
#         ]

#         score = (
#             1.0 / (k + r_clip)
#             + 1.0 / (k + r_obj)
#             + 1.0 / (k + r_meta)
#             + 1.0 / (k + r_ocr)
#             + 1.0 / (k + r_asr)
#         )

#         cand[
#             "final_score"
#         ] = round(
#             float(score),
#             6
#         )

#     return sorted(
#         candidate_list,
#         key=lambda x: x[
#             "final_score"
#         ],
#         reverse=True
#     )

# def weighted_score_fusion(
#     candidate_list,
#     weights=None,
# ):
#     """
#     Weighted semantic fusion.

#     weights mặc định:

#     retrieval : 0.40
#     OCR       : 0.20
#     ASR       : 0.15
#     metadata  : 0.15
#     object    : 0.10
#     """

#     if weights is None:
#         weights = {
#             "retrieval": 0.40,
#             "ocr": 0.20,
#             "asr": 0.15,
#             "metadata": 0.15,
#             "object": 0.10,
#         }

#     for cand in candidate_list:

#         retrieval = cand["retrieval_score_normalized"]

#         object_score = cand["object_score_norm"]

#         metadata = cand["metadata_score_norm"]

#         ocr = cand["ocr_score_norm"]

#         asr = cand["asr_score_norm"]

#         ##################################################
#         # agreement bonus
#         ##################################################

#         agreement = np.mean([
#             retrieval,
#             object_score,
#             metadata,
#             ocr,
#             asr,
#         ])

#         ##################################################

#         final_score = (
#             weights["retrieval"] * retrieval
#             + weights["ocr"] * ocr
#             + weights["asr"] * asr
#             + weights["metadata"] * metadata
#             + weights["object"] * object_score
#             + 0.05 * agreement
#         )

#         cand["agreement_score"] = float(agreement)

#         cand["final_score"] = float(final_score)

#     return sorted(
#         candidate_list,
#         key=lambda x: x["final_score"],
#         reverse=True
#     )
# # ==========================================================
# # RERANKING PIPELINE
# # ==========================================================

# def reranking_pipeline(
#     query_text,
#     candidate_list,
#     config_path
# ):

#     if not candidate_list:

#         print(
#             "[INFO] Empty candidate list."
#         )

#         return []

#     cfg = load_rerank_config(
#         config_path
#     )
#     query_emb = encode_query(query_text)
#     candidate_list = (
#         normalize_retrieval_scores(
#             candidate_list
#         )
#     )

#     candidate_list = (
#         compute_object_scores(
#             query_emb,
#             candidate_list
#         )
#     )

#     candidate_list = (
#         compute_metadata_scores(
#             query_emb,
#             candidate_list
#         )
#     )

#     candidate_list = (
#         compute_ocr_scores(
#             query_emb,
#             candidate_list
#         )
#     )

#     candidate_list = (
#         compute_asr_scores(
#             query_emb,
#             candidate_list
#         )
#     )
#     candidate_list = normalize_feature(
#     candidate_list,
#     "object_score"
#     )

#     candidate_list = normalize_feature(
#         candidate_list,
#         "metadata_score"
#     )

#     candidate_list = normalize_feature(
#         candidate_list,
#         "ocr_score"
#     )

#     candidate_list = normalize_feature(
#         candidate_list,
#         "asr_score"
#     )

#     # ------------------------------------------------------
#     # DEBUG SCORE TABLE
#     # ------------------------------------------------------

#     print("\n")
#     print("=" * 90)
#     print("RERANKING SCORE DEBUG")
#     print("=" * 90)

#     for i, cand in enumerate(
#         candidate_list[:10],
#         start=1
#     ):

#         print(
#             f"{i:02d} | "
#             f"{cand['video_id']} | "
#             f"KF={cand['keyframe_index']:4d} | "
#             f"CLIP={cand['score']:.4f} | "
#             f"OBJ={cand['object_score']:.4f} | "
#             f"META={cand['metadata_score']:.4f} | "
#             f"OCR={cand['ocr_score']:.4f} | "
#             f"ASR={cand.get('asr_score', 0.0):.4f}"
#         )

#     print("=" * 90)

#     # rrf_k = cfg.get(
#     #     "rrf_k",
#     #     60
#     # )

#     candidate_list = weighted_score_fusion(
#         candidate_list,
#         weights=cfg.get("weights")
#     )

#     results = []

#     for rank, cand in enumerate(
#         candidate_list,
#         start=1
#     ):

#         cand["rank"] = rank

#         results.append({

#             "rank": rank,

#             "vector_index": cand["vector_index"],

#             "video_id": cand["video_id"],

#             "frame_idx": cand.get("frame_idx", cand.get("keyframe_index", 0)),

#             "keyframe_index": cand.get("keyframe_index", 0),

#             "pts_time": cand.get("pts_time", 0.0),

#             "fps": cand.get("fps", 25.0),

#             "keyframe_path": cand.get("keyframe_path", ""),

#             "object_path": cand.get("object_path", ""),

#             "metadata_path": cand.get("metadata_path", ""),

#             "retrieval_score": cand["retrieval_score"],

#             "retrieval_score_normalized": cand[
#                 "retrieval_score_normalized"
#             ],

#             "object_score": cand[
#                 "object_score"
#             ],

#             "metadata_score": cand[
#                 "metadata_score"
#             ],

#             "ocr_score": cand.get(
#                 "ocr_score",
#                 0.0
#             ),

#             "asr_score": cand.get(
#                 "asr_score",
#                 0.0
#             ),

#             "final_score": cand[
#                 "final_score"
#             ],

#             "object_entities": cand.get(
#                 "object_entities",
#                 []
#             ),

#             "ocr_texts": cand.get(
#                 "ocr_texts",
#                 []
#             ),

#             "asr_text": cand.get(
#                 "asr_text",
#                 ""
#             ),

#             "metadata": cand.get(
#                 "metadata",
#                 {}
#             )
#         })

#     print(
#         f"[INFO] Reranking completed "
#         f"({len(results)} results)."
#     )

#     return results









import os
import yaml
import json
import numpy as np
import faiss
import torch
import torch.nn.functional as F
import psutil

from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

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
# ==========================================================
# OBJECT VECTOR INDEX
# ==========================================================

_OBJECT_INDEX = None
_OBJECT_MAPPING = None

OBJECT_INDEX_PATH = (
    ROOT / "data" / "indexes" / "object_IVFPQ.index"
)

OBJECT_MAPPING_PATH = (
    ROOT / "data" / "indexes" / "object_mapping_new.json"
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

def compute_object_scores(
    query_emb,
    candidate_list
):

    print(
        "[INFO] Computing object scores "
        "directly from FAISS reconstruct..."
    )

    if not candidate_list:
        return candidate_list

    # ------------------------------------------------------
    # Load index + lookup
    # ------------------------------------------------------

    object_index, _ = get_object_index()
    object_lookup = get_object_lookup()

    # ------------------------------------------------------
    # Prepare query embedding
    # ------------------------------------------------------

    query_emb = np.asarray(
        query_emb,
        dtype=np.float32
    ).reshape(1, -1)

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    total_object_vectors = 0
    candidates_with_object = 0
    candidates_without_object = 0
    total_reconstructed = 0  # Tổng số vector gọi reconstruct

    # ------------------------------------------------------
    # Process candidates
    # ------------------------------------------------------

    for cand in candidate_list:

        video_id = cand.get(
            "video_id",
            ""
        )

        keyframe_index = cand.get(
            "keyframe_index"
        )

        if keyframe_index is None:
            cand["object_score"] = 0.0
            candidates_without_object += 1
            continue

        # Sửa lại: Tạo key dạng chuỗi khớp với object_mapping_new.json (ví dụ: L21_V001_0001)
        key = f"{video_id}_{int(keyframe_index):04d}"

        vector_indices = object_lookup.get(
            key,
            []
        )

        # --------------------------------------------------
        # No object
        # --------------------------------------------------

        if not vector_indices:

            cand["object_score"] = 0.0

            candidates_without_object += 1

            continue

        candidates_with_object += 1

        total_object_vectors += len(
            vector_indices
        )
        
        total_reconstructed += len(vector_indices)

        # --------------------------------------------------
        # Get object vectors (Gọi trực tiếp FAISS reconstruct)
        # --------------------------------------------------

        object_vectors = get_object_vectors(
            object_index,
            vector_indices
        )

        # --------------------------------------------------
        # Cosine similarity
        # --------------------------------------------------

        similarities = (
            object_vectors @ query_emb.T
        ).reshape(-1)

        # --------------------------------------------------
        # Top-k average
        # --------------------------------------------------

        score = topk_average(
            similarities.tolist(),
            k=3
        )

        cand["object_score"] = round(
            float(score),
            4
        )

    # ------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------

    scores = [
        cand.get(
            "object_score",
            0.0
        )
        for cand in candidate_list
    ]

    print(
        "[DEBUG OBJECT]"
    )

    print(
        f"Candidates                 : "
        f"{len(candidate_list)}"
    )

    print(
        f"Candidates with object     : "
        f"{candidates_with_object}"
    )

    print(
        f"Candidates without object  : "
        f"{candidates_without_object}"
    )

    print(
        f"Object vectors evaluated   : "
        f"{total_object_vectors}"
    )

    print(
        f"Vectors reconstructed      : "
        f"{total_reconstructed}"
    )

    print(
        f"Score range                : "
        f"{min(scores):.4f}"
        f" -> "
        f"{max(scores):.4f}"
    )

    return candidate_list
# ==========================================================
# METADATA VECTOR INDEX
# ==========================================================

_METADATA_INDEX = None
_METADATA_MAPPING = None

METADATA_INDEX_PATH = (
    ROOT / "data" / "indexes" / "metadata_IVFPQ.index"
)

METADATA_MAPPING_PATH = (
    ROOT / "data" / "indexes" / "metadata_mapping_new.json"
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
# OCR SEMANTIC SCORE 
# ==========================================================

def compute_ocr_scores(
    query_emb,
    candidate_list,
    ocr_threshold=0.5,
    batch_size=32
):
    print(
        f"[INFO] Computing OCR scores using E5 (threshold > {ocr_threshold})..."
    )

    if not candidate_list:
        return candidate_list

    query_emb = query_emb.reshape(
        1,
        -1
    )

    ocr_texts = []
    ocr_candidate_ids = []

    for cand_idx, cand in enumerate(
        candidate_list
    ):
        texts_info = cand.get(
            "ocr_texts",
            []
        )

        if not isinstance(
            texts_info,
            list
        ):
            texts_info = []

        for item in texts_info:
            # Hỗ trợ tương thích ngược: item có thể là dict {"text": ..., "score": ...} hoặc chuỗi cũ
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                score = float(item.get("score", 1.0))
            elif isinstance(item, str):
                text = str(item).strip()
                score = 1.0 # Dữ liệu cũ không có score thì coi như pass threshold
            else:
                continue

            if not text:
                continue

            # LỌC THEO THRESHOLD: Chỉ đưa vào xử lý nếu score vượt ngưỡng
            if score >= ocr_threshold:
                ocr_texts.append(text)
                ocr_candidate_ids.append(cand_idx)

    if not ocr_texts:

        for cand in candidate_list:

            cand["ocr_score"] = 0.0

        print(
            "[DEBUG OCR] No OCR texts found passing the confidence threshold."
        )

        return candidate_list

    ocr_embeddings = encode_e5(
        ocr_texts,
        prefix="passage",
        batch_size=batch_size
    )

    similarities = (
        ocr_embeddings
        @ query_emb.T
    ).reshape(-1)

    candidate_scores = [
        []
        for _ in candidate_list
    ]

    for sim, cand_idx in zip(
        similarities,
        ocr_candidate_ids
    ):
        candidate_scores[
            cand_idx
        ].append(
            float(sim)
        )

    for cand_idx, cand in enumerate(
        candidate_list
    ):
        scores = candidate_scores[
            cand_idx
        ]

        if scores:
            score = topk_average(scores, k=3)
        else:
            score = 0.0

        cand["ocr_score"] = round(float(score), 4)

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

    print(
        "[DEBUG OCR]"
    )

    print(
        f"Candidates             : {len(candidate_list)}"
    )

    print(
        f"Candidates with OCR    : {non_zero}"
    )

    print(
        f"Score range            : "
        f"{min(scores):.4f}"
        f" -> "
        f"{max(scores):.4f}"
    )

    return candidate_list
# ==========================================================
# METADATA SEMANTIC SCORE
# USING PRECOMPUTED E5 METADATA INDEX
# ==========================================================


def compute_metadata_scores(
    query_emb,
    candidate_list
):

    print(
        "[INFO] Computing metadata scores "
        "directly from FAISS reconstruct..."
    )

    if not candidate_list:
        return candidate_list

    # Load index + lookup
    metadata_index, _ = get_metadata_index()
    metadata_lookup = get_metadata_lookup()

    # Prepare query
    query_emb = np.asarray(
        query_emb,
        dtype=np.float32
    ).reshape(1, -1)

    # Statistics
    candidates_with_metadata = 0
    candidates_without_metadata = 0
    total_reconstructed = 0

    for cand in candidate_list:
        video_id = cand.get("video_id", "")
        keyframe_index = cand.get("keyframe_index")

        if keyframe_index is None:
            cand["metadata_score"] = 0.0
            candidates_without_metadata += 1
            continue

        # Đổi cách tạo key từ tuple sang chuỗi format khớp với metadata_mapping_new.json (ví dụ: L21_V001_0001)
        key = f"{video_id}_{int(keyframe_index):04d}"
        
        # metadata_lookup trả về dict, ví dụ: {"vector_index": 0}
        entry = metadata_lookup.get(key)

        if entry is None or "vector_index" not in entry:
            cand["metadata_score"] = 0.0
            candidates_without_metadata += 1
            continue

        # Lấy vector_index từ trong dict ra
        vector_index = entry.get("vector_index")
        if vector_index is None:
            cand["metadata_score"] = 0.0
            candidates_without_metadata += 1
            continue

        candidates_with_metadata += 1
        vector_index = int(vector_index)
        total_reconstructed += 1

        # Lấy vector trực tiếp từ FAISS
        metadata_vector = get_metadata_vector(
            metadata_index,
            vector_index
        )

        # Cosine similarity
        similarity = float(
            metadata_vector @ query_emb[0]
        )

        cand["metadata_score"] = round(
            similarity,
            4
        )

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
# ASR SEMANTIC SCORE - MAXSIM
# ==========================================================

def compute_asr_scores(
    query_emb,
    candidate_list,
    batch_size=32
):

    print(
        "[INFO] Computing ASR scores using E5..."
    )

    if not candidate_list:
        return candidate_list


    query_emb = query_emb.reshape(
        1,
        -1
    )

    asr_texts = []
    asr_candidate_ids = []

    for cand_idx, cand in enumerate(
        candidate_list
    ):

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

            asr_texts.append(
                text
            )

            asr_candidate_ids.append(
                cand_idx
            )

    if not asr_texts:

        for cand in candidate_list:

            cand["asr_score"] = 0.0

        print(
            "[DEBUG ASR] No ASR texts found."
        )

        return candidate_list

    asr_embeddings = encode_e5(
        asr_texts,
        prefix="passage",
        batch_size=batch_size
    )

    similarities = (
        asr_embeddings
        @ query_emb.T
    ).reshape(-1)

    candidate_scores = [
        []
        for _ in candidate_list
    ]

    for sim, cand_idx in zip(
        similarities,
        asr_candidate_ids
    ):

        candidate_scores[
            cand_idx
        ].append(
            float(sim)
        )

    for cand_idx, cand in enumerate(
        candidate_list
    ):

        scores = candidate_scores[
            cand_idx
        ]

        if scores:
            score = max(scores)
        else:
            score = 0.0

        cand["asr_score"] = round(
            float(score),
            4
        )

    # ------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------

    non_zero = sum(
        1
        for cand in candidate_list
        if cand["asr_score"] != 0
    )

    scores = [
        cand["asr_score"]
        for cand in candidate_list
    ]

    print(
        "[DEBUG ASR]"
    )

    print(
        f"Candidates             : {len(candidate_list)}"
    )

    print(
        f"Candidates with ASR    : {non_zero}"
    )

    print(
        f"Score range            : "
        f"{min(scores):.4f}"
        f" -> "
        f"{max(scores):.4f}"
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
def weighted_score_fusion(
    candidate_list,
    weights=None,
):
    """
    Weighted semantic fusion.

    weights mặc định:

    retrieval : 0.40
    OCR       : 0.20
    ASR       : 0.15
    metadata  : 0.15
    object    : 0.10
    """

    if weights is None:
        weights = {
            "retrieval": 0.40,
            "ocr": 0.20,
            "asr": 0.15,
            "metadata": 0.15,
            "object": 0.10,
        }

    for cand in candidate_list:
        # Sử dụng .get() linh hoạt để dự phòng cả tên có _norm lẫn không có _norm
        retrieval = cand.get("retrieval_score_normalized", cand.get("retrieval_score", 0.0))
        object_score = cand.get("object_score_norm", cand.get("object_score", 0.0))
        metadata = cand.get("metadata_score_norm", cand.get("metadata_score", 0.0))
        ocr = cand.get("ocr_score", 0.0)
        asr = cand.get("asr_score", 0.0)

        ##################################################
        # agreement bonus
        ##################################################

        agreement = np.mean([
            retrieval,
            object_score,
            metadata,
            ocr,
            asr,
        ])

        ##################################################

        final_score = (
            weights.get("retrieval", 0.4) * retrieval
            + weights.get("ocr", 0.2) * ocr
            + weights.get("asr", 0.15) * asr
            + weights.get("metadata", 0.15) * metadata
            + weights.get("object", 0.1) * object_score
            + 0.05 * agreement
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
    config_path
):
    print_ram("Start")
    if not candidate_list:
        print("[INFO] Empty candidate list.")
        return []

    cfg = load_rerank_config(config_path)

    # Lấy ocr_threshold từ file yaml, mặc định là 0.5 nếu không khai báo
    ocr_threshold = cfg.get("ocr_threshold", 0.5)

    query_emb = encode_query(query_text)
    print_ram("After encode_query")

    # Thực thi tuần tự các bước chuẩn hóa và tính toán điểm số
    candidate_list = normalize_retrieval_scores(candidate_list)
    print_ram("After normalize_retrieval")

    candidate_list = compute_object_scores(query_emb, candidate_list)
    print_ram("After object")

    candidate_list = compute_metadata_scores(query_emb, candidate_list)
    print_ram("After metadata")
    # Truyền ocr_threshold đã lấy từ cấu hình vào hàm tính điểm OCR
    candidate_list = compute_ocr_scores(
        query_emb=query_emb,
        candidate_list=candidate_list,
        ocr_threshold=ocr_threshold
    )
    print_ram("After OCR")
    candidate_list = compute_asr_scores(query_emb, candidate_list)
    print_ram("After ASR")

    # ------------------------------------------------------
    # DEBUG SCORE TABLE
    # ------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("RERANKING SCORE DEBUG")
    print("=" * 90)

    for i, cand in enumerate(candidate_list[:10], start=1):
        print(
            f"{i:02d} | "
            f"{cand.get('video_id', 'N/A')} | "
            f"KF={cand.get('keyframe_index', cand.get('frame_idx', 0)):4d} | "
            f"CLIP={cand.get('score', 0.0):.4f} | "
            f"OBJ={cand.get('object_score', 0.0):.4f} | "
            f"META={cand.get('metadata_score', 0.0):.4f} | "
            f"OCR={cand.get('ocr_score', 0.0):.4f} | "
            f"ASR={cand.get('asr_score', 0.0):.4f}"
        )

    print("=" * 90)

    # Thực hiện trộn điểm tổng hợp dựa trên weights trong yaml
    candidate_list = weighted_score_fusion(
        candidate_list,
        weights=cfg.get("weights")
    )
    print_ram("After Fusion")
    # Đóng gói chuẩn xác lại danh sách kết quả trả về cho hệ thống (UI/Streamlit)
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
            "metadata_score": cand["metadata_score"],
            "ocr_score": cand.get("ocr_score", 0.0),
            "asr_score": cand.get("asr_score", 0.0),
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