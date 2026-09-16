
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
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
import torch
import json
import time
import gc
import random
from huggingface_hub import hf_hub_download
import sentencepiece as spm


# bộ dịch envit5
# Biến toàn cục lưu giữ Instance của EnViT5 cố định trong RAM
_ENVIT5_MODEL = None
_ENVIT5_TOKENIZER = None
def get_envit5_translator(device: str = "cpu"):
    global _ENVIT5_MODEL, _ENVIT5_SPM

    if _ENVIT5_MODEL is None or _ENVIT5_SPM is None:
        model_name = "VietAI/envit5-translation"
        print(
            f"[INFO] Loading {model_name} into memory on {device} (One-time load)..."
        )

        # 1. Tải file spiece.model từ Hub
        spm_path = hf_hub_download(repo_id=model_name, filename="spiece.model")

        # 2. Dùng trực tiếp SentencePieceProcessor chuẩn của Google (Không thông qua transformers tokenizer)
        _ENVIT5_SPM = spm.SentencePieceProcessor()
        _ENVIT5_SPM.load(spm_path)

        # 3. Nạp Model EnViT5
        _ENVIT5_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        _ENVIT5_MODEL.eval()

        print("[SUCCESS] VietAI EnViT5 loaded into RAM successfully!")

    return _ENVIT5_MODEL, _ENVIT5_SPM


def translate_with_envit5(query_text: str, device: str = "cpu") -> str:
    """
    Hàm dịch tối ưu cho EnViT5 với Beam Search nhẹ và quản lý thiết bị tốt hơn.
    """
    if not query_text or not query_text.strip():
        return ""

    model, sp = get_envit5_translator(device=device)

    # Đảm bảo model đã nằm đúng device
    model.to(device)
    model.eval()

    # Chuẩn hóa input format của EnViT5
    input_text = f"vi: {query_text.strip()}"

    # Mã hóa trực tiếp bằng SentencePiece
    input_ids = sp.encode(input_text) + [1] # Thêm EOS token (ID 1)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tensor,
            max_new_tokens=256,        # Dùng max_new_tokens thay vì max_length để tránh bị cắt cụt câu dài
            num_beams=4,               # Tăng num_beams lên 4 để cải thiện chất lượng dịch (giảm lủng củng)
            early_stopping=True,       # Dừng sớm khi gặp token kết thúc beam search
            do_sample=False,
            decoder_start_token_id=0,  # Token bắt đầu giải mã của EnViT5
            eos_token_id=1,
            pad_token_id=0,
        )

    # Giải mã ID token ra chuỗi văn bản
    output_ids = outputs[0].tolist()
    translated = sp.decode(output_ids).strip()

    # Loại bỏ tiền tố "en:" hoặc các biến thể khoảng trắng bằng Regex cho triệt để
    translated = re.sub(r'^(en\s*:\s*)', '', translated, flags=re.IGNORECASE).strip()

    return translated
    
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


# ==========================================================
# QWEN-1.5B LOAD & DECOMPOSITION HELPERS
# ==========================================================

def load_gemini_api_key(api_file=None):
    """
    Đọc Gemini API key từ file .api tại thư mục gốc của project.
    """
    if api_file:
        path = Path(api_file)
    else:
        root = Path(__file__).resolve().parent.parent.parent
        path = root / ".api"
        if not path.exists():
            path = Path(".api")

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                return line
    return os.environ.get("GEMINI_API_KEY", "")


def decompose_with_gemini_api(query_text: str, api_key: str = None) -> dict:
    """
    Gọi Gemini API (model gemini-3.1-flash-lite) với Smart Scene-Aware Prompt.
    - Phân biệt cảnh tĩnh đồng hiện (giữ nguyên toàn bộ chi tiết) và chuỗi hành động tuần tự.
    - Bảo toàn trọn vẹn số lượng, tên đường, tên riêng và chữ viết.
    """
    if not api_key:
        api_key = load_gemini_api_key()
    if not api_key:
        raise ValueError("Không tìm thấy Gemini API key trong file .api hoặc biến môi trường.")

    prompt = f"""You are an expert query understanding engine for a video keyframe search system (CLIP/SigLIP).
Your goal is to parse a search query into JSON with 3 fields:
- "q_full": The original query text.
- "q_main": The primary search text (Anchor).
- "q_context": Secondary/sequential context. IF NOT APPLICABLE, SET TO "".

### CORE RULES (CRITICAL):
1. STATIC SCENE vs TEMPORAL SEQUENCE:
   - If the query describes a STATIC SCENE, A SINGLE FRAME, A PRESENTATION SLIDE, or CO-OCCURRING OBJECTS (e.g. a slide with multiple boxes and characters, a person dressed in specific clothes with background props):
     -> DO NOT SPLIT co-occurring objects! Keep ALL descriptive visual details in "q_main".
     -> Set "q_context" to "".
   - ONLY extract "q_context" if there is an explicit TEMPORAL PROGRESSION, SEQUENTIAL ACTION, or SECONDARY BACKGROUND ACTION (e.g., "A is eating watermelon, behind her mother is cleaning", "first shows X, then transitions to Y").

2. PRESERVE SPECIFIC IDENTIFIERS:
   - Always preserve numbers, counts, street/place names (e.g. "Hồ Tùng Mậu"), signs, text, and countdown timers (e.g. "13 giây") in "q_main".

3. REMOVE FILLER WORDS:
   - Strip generic filler phrases like "Đoạn clip là", "Tìm video về", "Cảnh phim quay cảnh" from "q_main", but KEEP all actual visual content.

### EXAMPLES

Query: "Một slide bài giảng hiển thị hình ảnh một nhóm nhân vật người ba chiều màu trắng vây quanh một nhân vật màu đỏ ở chính giữa, cùng với hai nhân vật hoạt hình nam đang trong tư thế thi đấu kéo co đối đầu nhau qua một sợi dây thừng"
{{
  "q_full": "Một slide bài giảng hiển thị hình ảnh một nhóm nhân vật người ba chiều màu trắng vây quanh một nhân vật màu đỏ ở chính giữa, cùng với hai nhân vật hoạt hình nam đang trong tư thế thi đấu kéo co đối đầu nhau qua một sợi dây thừng",
  "q_main": "slide bài giảng hiển thị nhóm nhân vật người ba chiều màu trắng vây quanh nhân vật màu đỏ ở chính giữa cùng với hai nhân vật hoạt hình nam đang trong tư thế thi đấu kéo co đối đầu nhau qua sợi dây thừng",
  "q_context": ""
}}

Query: "giáo viên nam mặc sơ mi trắng và thắt cà vạt tối màu nổi bật trên phông nền xanh dương đậm có hoa văn mờ, kết hợp với slide bài giảng có nền trắng và khung viền màu hồng tím, phía trên có thanh tiêu đề xanh dương chứa họa tiết địa cầu cùng mũi tên vàng xanh ngọc, bên dưới hiển thị sơ đồ ba tầng liên kết bởi các mũi tên xanh ngọc trỏ xuống, bao gồm tầng một với hai khối hộp nằm trong một khối hộp cam, tầng hai với một khối hộp lớn màu xanh dương đậm ở chính giữa và tầng ba với hai khối hộp nằm trong một khối hộp xanh lá cây."
{{
  "q_full": "giáo viên nam mặc sơ mi trắng và thắt cà vạt tối màu nổi bật trên phông nền xanh dương đậm có hoa văn mờ, kết hợp với slide bài giảng có nền trắng và khung viền màu hồng tím, phía trên có thanh tiêu đề xanh dương chứa họa tiết địa cầu cùng mũi tên vàng xanh ngọc, bên dưới hiển thị sơ đồ ba tầng liên kết bởi các mũi tên xanh ngọc trỏ xuống, bao gồm tầng một với hai khối hộp nằm trong một khối hộp cam, tầng hai với một khối hộp lớn màu xanh dương đậm ở chính giữa và tầng ba với hai khối hộp nằm trong một khối hộp xanh lá cây.",
  "q_main": "giáo viên nam mặc sơ mi trắng thắt cà vạt tối màu trên phông nền xanh dương đậm hoa văn mờ kết hợp slide bài giảng nền trắng khung viền hồng tím thanh tiêu đề xanh dương họa tiết địa cầu và sơ đồ ba tầng liên kết mũi tên xanh ngọc trỏ xuống",
  "q_context": ""
}}

Query: "Đoạn phim quay từ phía sau nhóm dẫn đầu, gồm 1 tay đua dẫn trước và 3 tay đua bám phía sau, khi cả nhóm rẽ phải vào đường Hồ Tùng Mậu tại giao lộ có đèn xanh đang đếm ngược đến 13 giây."
{{
  "q_full": "Đoạn phim quay từ phía sau nhóm dẫn đầu, gồm 1 tay đua dẫn trước và 3 tay đua bám phía sau, khi cả nhóm rẽ phải vào đường Hồ Tùng Mậu tại giao lộ có đèn xanh đang đếm ngược đến 13 giây.",
  "q_main": "nhóm gồm 1 tay đua dẫn trước và 3 tay đua bám phía sau rẽ phải vào đường Hồ Tùng Mậu tại giao lộ có đèn xanh đếm ngược 13 giây",
  "q_context": "quay từ phía sau nhóm dẫn đầu"
}}

Query: "Hai bạn học sinh mặc đồng phục áo trắng, quần xanh, quàng khăn đỏ đang làm MC trên một sân khấu tại trường học, phía sau là một bộ trống cơ màu đỏ và một cây đàn piano."
{{
  "q_full": "Hai bạn học sinh mặc đồng phục áo trắng, quần xanh, quàng khăn đỏ đang làm MC trên một sân khấu tại trường học, phía sau là một bộ trống cơ màu đỏ và một cây đàn piano.",
  "q_main": "hai bạn học sinh mặc đồng phục áo trắng quần xanh quàng khăn đỏ làm MC trên sân khấu trường học",
  "q_context": "phía sau là bộ trống cơ màu đỏ và cây đàn piano"
}}

Query: "Đoạn video mô tả quá trình làm bánh, bánh được tạo ra có màu tím, nguyên liệu bên trong có giá, cà rốt, và bên trong mỗi bánh đều có 1 hạt sen."
{{
  "q_full": "Đoạn video mô tả quá trình làm bánh, bánh được tạo ra có màu tím, nguyên liệu bên trong có giá, cà rốt, và bên trong mỗi bánh đều có 1 hạt sen.",
  "q_main": "làm bánh có màu tím nguyên liệu bên trong có giá cà rốt và bên trong mỗi bánh có 1 hạt sen",
  "q_context": ""
}}

### INPUT QUERY
Query: "{query_text}"

### OUTPUT (Valid JSON only):
"""

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash"
    ]

    import urllib.request
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
    }).encode("utf-8")

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(reply.strip())
                return {
                    "q_full": parsed.get("q_full", query_text),
                    "q_main": parsed.get("q_main", query_text),
                    "q_context": parsed.get("q_context", "")
                }
        except Exception:
            continue

    raise RuntimeError("Tất cả candidate models của Gemini API đều không phản hồi.")


_QWEN_MODEL_CACHE = {
    "model": None,
    "tokenizer": None
}
def get_qwen_decomposer_model(device: str = "cpu"):
    """
    Load Qwen2.5-1.5B-Instruct lên RAM/GPU một lần duy nhất (Dự phòng cục bộ).
    """
    global _QWEN_MODEL_CACHE
    if _QWEN_MODEL_CACHE["model"] is None:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"[INFO] Loading Qwen model ({model_name}) into memory on {device} (One-time load)...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to("cpu")
        model.eval()

        _QWEN_MODEL_CACHE["model"] = model
        _QWEN_MODEL_CACHE["tokenizer"] = tokenizer
        print("[SUCCESS] Qwen model loaded into RAM/GPU successfully!")

    return _QWEN_MODEL_CACHE["model"], _QWEN_MODEL_CACHE["tokenizer"]

def decompose_standard_narrative_query(query_text):
    """
    Phân rã câu truy vấn văn xuôi / có cấu trúc chuỗi thời gian thành:
    - q_full: Câu truy vấn đầy đủ
    - q_main: Mệnh đề / hành động trực quan chính (Anchor)
    - q_context: Mệnh đề diễn biến bổ trợ (Context Clue) nếu có
    
    Cơ chế: Chạy Regex trước. Chỉ khi Regex không khớp,
    hệ thống mới đẩy câu query sang Qwen-1.5B. Luôn trả về dạng dict chuẩn.
    """
    if not query_text or not isinstance(query_text, str):
        return {
            "q_full": query_text,
            "q_main": query_text,
            "q_context": ""
        }

    # Các từ nối chỉ mốc thời gian / diễn biến thường gặp trong đề KIS
    temporal_splitters = [
        r"(?i)\bbiết sau đó\b",
        r"(?i)\bvà sau đó\b",
        r"(?i)\bsau đó\b",
        r"(?i)\bsau cùng\b",
        r"(?i)\btiếp theo\b",
        r"(?i)\bkế tiếp\b",
        r"(?i)\bđoạn sau\b",
        r"(?i)\blúc sau\b",
        r"(?i)\bở phần sau\b",
        r"(?i)\brồi sau đó\b",
        r"(?i)\bgiảng bài về\b",
        r"(?i)\bgiảng về\b",
        r"(?i)\bnói về\b",
        r"(?i)\btrình bày về\b",
        r"(?i)\btrên slide\b",
        r"(?i)\bslide chứa\b",
        r"(?i)\bvới nội dung\b"
    ]
    
    # 1. Thử tách theo từ nối thời gian bằng Regex
    for pattern in temporal_splitters:
        parts = re.split(pattern, query_text)
        if len(parts) > 1 and len(parts[0].strip()) > 10:
            q_main = parts[0].strip().rstrip(".,; ")
            q_context = " ".join([p.strip() for p in parts[1:] if p.strip()])
            print(f"[INFO] Narrative Query Decomposition (Regex - Temporal Match):")
            print(f"  -> Q_Full   : '{query_text}'")
            print(f"  -> Q_Anchor : '{q_main}'")
            print(f"  -> Q_Context: '{q_context}'")
            return {
                "q_full": query_text,
                "q_main": q_main,
                "q_context": q_context
            }
            
    # 2. Thử tách theo dấu câu nếu câu dài chứa nhiều mệnh đề
    sentences = [s.strip() for s in re.split(r"[.\n]+", query_text) if len(s.strip()) > 10]
    if len(sentences) >= 2 and len(query_text.split()) > 15:
        q_main = sentences[0]
        q_context = " ".join(sentences[1:])
        print(f"[INFO] Narrative Query Decomposition (Regex - Sentence Split Match):")
        print(f"  -> Q_Full   : '{query_text}'")
        print(f"  -> Q_Anchor : '{q_main}'")
        print(f"  -> Q_Context: '{q_context}'")
        return {
            "q_full": query_text,
            "q_main": q_main,
            "q_context": q_context
        }
        
    # =========================================================================
    # 3. FALLBACK SANG GEMINI 3.1 FLASH LITE API (Smart Scene-Aware)
    # =========================================================================
    print(f"[INFO] Regex did not split the query. Fallback to Gemini 3.1 Flash Lite API...")
    try:
        gemini_res = decompose_with_gemini_api(query_text)
        q_full = gemini_res.get("q_full", query_text)
        q_main = gemini_res.get("q_main", query_text)
        q_context = gemini_res.get("q_context", "")

        if q_main and q_main.strip():
            print(f"[INFO] Narrative Query Decomposition (Gemini Flash Lite Success):")
            print(f"  -> Q_Full   : '{q_full}'")
            print(f"  -> Q_Anchor : '{q_main}'")
            print(f"  -> Q_Context: '{q_context}'")
            return {
                "q_full": q_full,
                "q_main": q_main,
                "q_context": q_context
            }
    except Exception as e:
        print(f"[WARNING] Gemini API decomposition failed: {e}. Falling back to original query.")

    # Mặc định cuối cùng nếu mọi cách đều không tách được
    return {
        "q_full": query_text,
        "q_main": query_text,
        "q_context": ""
    }

# def decompose_standard_narrative_query(query_text):
#     """
#     Phân rã câu truy vấn văn xuôi / có cấu trúc chuỗi thời gian trong Standard Query thành:
#     - q_full: Câu truy vấn đầy đủ
#     - q_main: Mệnh đề / hành động trực quan chính (Anchor)
#     - q_context: Mệnh đề diễn biến bổ trợ (Context Clue) nếu có
#     """
#     if not query_text or not isinstance(query_text, str):
#         return [query_text]

#     # Các từ nối chỉ mốc thời gian / diễn biến thường gặp trong đề KIS
#     temporal_splitters = [
#         r"(?i)\bbiết sau đó\b",
#         r"(?i)\bvà sau đó\b",
#         r"(?i)\bsau đó\b",
#         r"(?i)\bsau cùng\b",
#         r"(?i)\btiếp theo\b",
#         r"(?i)\bkế tiếp\b",
#         r"(?i)\bđoạn sau\b",
#         r"(?i)\blúc sau\b",
#         r"(?i)\bở phần sau\b",
#         r"(?i)\brồi sau đó\b",
#         r"(?i)\bgiảng bài về\b",
#         r"(?i)\bgiảng về\b",
#         r"(?i)\bnói về\b",
#         r"(?i)\btrình bày về\b",
#         r"(?i)\btrên slide\b",
#         r"(?i)\bslide chứa\b",
#         r"(?i)\bvới nội dung\b"
#     ]
    
#     # 1. Thử tách theo từ nối thời gian
#     for pattern in temporal_splitters:
#         parts = re.split(pattern, query_text)
#         if len(parts) > 1 and len(parts[0].strip()) > 10:
#             q_main = parts[0].strip().rstrip(".,; ")
#             q_context = " ".join([p.strip() for p in parts[1:] if p.strip()])
#             print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#             print(f"  -> Q_Full   : '{query_text}'")
#             print(f"  -> Q_Anchor : '{q_main}'")
#             print(f"  -> Q_Context: '{q_context}'")
#             return [query_text, q_main, q_context]
            
#     # # 2. Thử tách theo dấu câu nếu câu dài chứa nhiều mệnh đề
#     sentences = [s.strip() for s in re.split(r"[.\n]+", query_text) if len(s.strip()) > 10]
#     if len(sentences) >= 2 and len(query_text.split()) > 15:
#         q_main = sentences[0]
#         q_context = " ".join(sentences[1:])
#         print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#         print(f"  -> Q_Full   : '{query_text}'")
#         print(f"  -> Q_Anchor : '{q_main}'")
#         print(f"  -> Q_Context: '{q_context}'")
#         return [query_text, q_main, q_context]
        
#     return [query_text]


MAX_RETRY = 1
def expand_and_translate_query(query_text, device="cpu"):

    if not query_text or not query_text.strip():
        return query_text

    
    try:

        translated = translate_with_envit5(
            query_text,
            device="cpu"
        )

        if translated and translated.strip():

            print("[INFO]  Fallback Success")
            print(translated)

            return translated

    except Exception as e:

        print(f"[WARNING] cũng lỗi: {e}")

    # ======================================================
    # TẦNG 3: ORIGINAL
    # ======================================================

    print("[WARNING] Sử dụng query gốc.")

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

# def siglip2_retrieval_pipeline(query_text, index_path, config_path):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     cfg = load_config(config_path)
#     top_k = cfg.get("top_k", 100)
#     threshold = cfg.get("retrieval_threshold", 0.0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     processor, model = get_siglip2_model(device)
#     max_length = model.config.text_config.max_position_embeddings

#     # Tokenize và trích xuất embedding cho từng sub-query
#     embeddings_list = []
#     with torch.no_grad():
#         for t_query in translated_sub_queries:
#             inputs = processor(
#                 text=[t_query], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#             text_outputs = model.get_text_features(**inputs)
#             if hasattr(text_outputs, "pooler_output"):
#                 tf = text_outputs.pooler_output
#             elif hasattr(text_outputs, "last_hidden_state"):
#                 tf = text_outputs.last_hidden_state.mean(dim=1)
#             else:
#                 tf = text_outputs
#             tf = F.normalize(tf, p=2, dim=1)
#             embeddings_list.append(tf)

#     # Gộp trọng số nếu có Sub-query: 50% Anchor chính, 35% Full query, 15% Context phụ
#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().numpy().astype(np.float32)

#     print("[INFO] Loading SigLIP 2 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] SigLIP 2 Retrieval returned {len(results)} candidates.")
#     return results


def siglip2_retrieval_pipeline(query_text, index_path, config_path, query_plan=None):
    # 1. Nếu query_plan chưa được truyền từ trên xuống, ta tiến hành phân rã
    if query_plan is None:
        query_plan = decompose_standard_narrative_query(query_text)
    
    # 2. Trích xuất các sub_queries từ query_plan (hỗ trợ cả định dạng dict hoặc list)
    if isinstance(query_plan, dict):
        q_full = query_plan.get("q_full", query_text)
        q_main = query_plan.get("q_main", "")
        q_context = query_plan.get("q_context", "")

        if q_context and q_context.strip():
            sub_queries = [q_full, q_main, q_context]
        else:
            # Cảnh tĩnh hoặc không tách context: Dùng 1 query duy nhất để giữ trọn vẹn đặc trưng visual
            target_q = q_main if (q_main and q_main.strip()) else q_full
            sub_queries = [target_q]
    elif isinstance(query_plan, list):
        sub_queries = query_plan
    else:
        sub_queries = [query_text]

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

    # Gộp trọng số nếu có Sub-query: 75% Anchor chính, 15% Full query, 10% Context phụ
    if len(embeddings_list) == 3:
        combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
        combined_tf = F.normalize(combined_tf, p=2, dim=1)
    elif len(embeddings_list) == 2:
        combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
        combined_tf = F.normalize(combined_tf, p=2, dim=1)
    elif len(embeddings_list) == 1:
        combined_tf = embeddings_list[0]
    else:
        # Fallback an toàn nếu list rỗng
        inputs = processor(text=[query_text], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        text_outputs = model.get_text_features(**inputs)
        combined_tf = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
        combined_tf = F.normalize(combined_tf, p=2, dim=1)

    query_embedding = combined_tf.cpu().numpy().astype(np.float32)

    print("[INFO] Loading SigLIP 2 FAISS index...")
    index = get_faiss_index(index_path)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append({"rank": rank, "vector_index": int(idx), "score": float(score), "query_plan": query_plan})
        rank += 1

    print(f"[INFO] SigLIP 2 Retrieval returned {len(results)} candidates.")
    return results


# def siglip2_retrieval_pipeline(query_text, index_path, config_path):
#     # Dịch Q_full trực tiếp bằng EnViT5
#     q_full_en = expand_and_translate_query(query_text, device="cpu")

#     cfg = load_config(config_path)
#     top_k = cfg.get("top_k", 100)
#     threshold = cfg.get("retrieval_threshold", 0.0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     processor, model = get_siglip2_model(device)
#     max_length = model.config.text_config.max_position_embeddings

#     # Embed Q_full_en
#     with torch.no_grad():
#         inputs = processor(
#             text=[q_full_en], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#         text_outputs = model.get_text_features(**inputs)
        
#         if hasattr(text_outputs, "pooler_output"):
#             tf = text_outputs.pooler_output
#         elif hasattr(text_outputs, "last_hidden_state"):
#             tf = text_outputs.last_hidden_state.mean(dim=1)
#         else:
#             tf = text_outputs
            
#         tf = F.normalize(tf, p=2, dim=1)

#     query_embedding = tf.cpu().numpy().astype(np.float32)

#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

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

# def dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     cfg = load_config(config_path)
#     top_k = cfg.get("top_k", 100)
#     threshold = cfg.get("retrieval_threshold", 0.0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model, tokenizer = get_dfn5b_vit_h14_model(device)

#     embeddings_list = []
#     with torch.no_grad(), torch.amp.autocast(device):
#         for t_query in translated_sub_queries:
#             text_tokens = tokenizer([t_query]).to(device)
#             tf = model.encode_text(text_tokens)
#             tf = F.normalize(tf, p=2, dim=-1)
#             embeddings_list.append(tf)

#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.1 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.8 * embeddings_list[0] + 0.2 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().float().numpy().astype(np.float32)

#     print("[INFO] Loading DFN5B-CLIP-ViT-H-14 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] DFN5B-CLIP-ViT-H-14 Retrieval returned {len(results)} candidates.")
#     return results


def dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path, query_plan=None):
    # 1. Nếu query_plan chưa được truyền từ trên xuống, ta mới tiến hành phân rã
    if query_plan is None:
        query_plan = decompose_standard_narrative_query(query_text)
    
    # 2. Trích xuất các sub_queries từ query_plan (đã được chuẩn hóa dạng dict hoặc list)
    if isinstance(query_plan, dict):
        q_full = query_plan.get("q_full", query_text)
        q_main = query_plan.get("q_main", "")
        q_context = query_plan.get("q_context", "")

        if q_context and q_context.strip():
            sub_queries = [q_full, q_main, q_context]
        else:
            # Cảnh tĩnh hoặc không tách context: Dùng 1 query duy nhất để giữ trọn vẹn đặc trưng visual
            target_q = q_main if (q_main and q_main.strip()) else q_full
            sub_queries = [target_q]
    elif isinstance(query_plan, list):
        sub_queries = query_plan
    else:
        sub_queries = [query_text]

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
        combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.1 * embeddings_list[2]
        combined_tf = F.normalize(combined_tf, p=2, dim=-1)
    elif len(embeddings_list) == 2:
        combined_tf = 0.8 * embeddings_list[0] + 0.2 * embeddings_list[1]
        combined_tf = F.normalize(combined_tf, p=2, dim=-1)
    elif len(embeddings_list) == 1:
        combined_tf = embeddings_list[0]
    else:
        # Fallback an toàn nếu list rỗng
        text_tokens = tokenizer([query_text]).to(device)
        combined_tf = F.normalize(model.encode_text(text_tokens), p=2, dim=-1)

    query_embedding = combined_tf.cpu().float().numpy().astype(np.float32)

    print("[INFO] Loading DFN5B-CLIP-ViT-H-14 FAISS index...")
    index = get_faiss_index(index_path)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append({"rank": rank, "vector_index": int(idx), "score": float(score),"query_plan": query_plan})
        rank += 1

    print(f"[INFO] DFN5B-CLIP-ViT-H-14 Retrieval returned {len(results)} candidates.")
    return results


# def dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path):
#     # Dịch Q_full trực tiếp bằng EnViT5
#     q_full_en = expand_and_translate_query(query_text, device="cpu")

#     cfg = load_config(config_path)
#     top_k = cfg.get("top_k", 100)
#     threshold = cfg.get("retrieval_threshold", 0.0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model, tokenizer = get_dfn5b_vit_h14_model(device)

#     # Embed Q_full_en
#     with torch.no_grad(), torch.amp.autocast(device):
#         text_tokens = tokenizer([q_full_en]).to(device)
#         tf = model.encode_text(text_tokens)
#         tf = F.normalize(tf, p=2, dim=-1)

#     query_embedding = tf.cpu().float().numpy().astype(np.float32)

#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     return results

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












































# import yaml
# import re
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from transformers import AutoModel, AutoProcessor
# from deep_translator import GoogleTranslator
# import faiss
# import clip  # Thư viện openai-clip cho ViT-B/32
# import open_clip
# from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
# import torch
# import time
# import gc
# import random
# from huggingface_hub import hf_hub_download
# import sentencepiece as spm


# # bộ dịch envit5
# # Biến toàn cục lưu giữ Instance của EnViT5 cố định trong RAM
# _ENVIT5_MODEL = None
# _ENVIT5_TOKENIZER = None
# def get_envit5_translator(device: str = "cpu"):
#     global _ENVIT5_MODEL, _ENVIT5_SPM

#     if _ENVIT5_MODEL is None or _ENVIT5_SPM is None:
#         model_name = "VietAI/envit5-translation"
#         print(
#             f"[INFO] Loading {model_name} into memory on {device} (One-time load)..."
#         )

#         # 1. Tải file spiece.model từ Hub
#         spm_path = hf_hub_download(repo_id=model_name, filename="spiece.model")

#         # 2. Dùng trực tiếp SentencePieceProcessor chuẩn của Google (Không thông qua transformers tokenizer)
#         _ENVIT5_SPM = spm.SentencePieceProcessor()
#         _ENVIT5_SPM.load(spm_path)

#         # 3. Nạp Model EnViT5
#         _ENVIT5_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float32 if device == "cpu" else torch.float16,
#         ).to(device)
#         _ENVIT5_MODEL.eval()

#         print("[SUCCESS] VietAI EnViT5 loaded into RAM successfully!")

#     return _ENVIT5_MODEL, _ENVIT5_SPM


# def translate_with_envit5(query_text: str, device: str = "cpu") -> str:
#     """
#     Hàm dịch tối ưu cho EnViT5 với Beam Search nhẹ và quản lý thiết bị tốt hơn.
#     """
#     if not query_text or not query_text.strip():
#         return ""

#     model, sp = get_envit5_translator(device=device)

#     # Đảm bảo model đã nằm đúng device
#     model.to(device)
#     model.eval()

#     # Chuẩn hóa input format của EnViT5
#     input_text = f"vi: {query_text.strip()}"

#     # Mã hóa trực tiếp bằng SentencePiece
#     input_ids = sp.encode(input_text) + [1] # Thêm EOS token (ID 1)
#     input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_tensor,
#             max_new_tokens=256,        # Dùng max_new_tokens thay vì max_length để tránh bị cắt cụt câu dài
#             num_beams=4,               # Tăng num_beams lên 4 để cải thiện chất lượng dịch (giảm lủng củng)
#             early_stopping=True,       # Dừng sớm khi gặp token kết thúc beam search
#             do_sample=False,
#             decoder_start_token_id=0,  # Token bắt đầu giải mã của EnViT5
#             eos_token_id=1,
#             pad_token_id=0,
#         )

#     # Giải mã ID token ra chuỗi văn bản
#     output_ids = outputs[0].tolist()
#     translated = sp.decode(output_ids).strip()

#     # Loại bỏ tiền tố "en:" hoặc các biến thể khoảng trắng bằng Regex cho triệt để
#     translated = re.sub(r'^(en\s*:\s*)', '', translated, flags=re.IGNORECASE).strip()

#     return translated
    
# _SIGLIP2_CACHE = {
#     "processor": None,
#     "model": None
# }

# _DFN5B_CACHE = {
#     "model": None,
#     "tokenizer": None
# }

# # 1. Tự động nhận diện nếu có GPU, nếu không thì dùng CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[INFO] Using device: {device}")

# # NEW
# _FAISS_CACHE = {}
# def get_faiss_index(index_path):
#     """
#     Load FAISS index một lần duy nhất.
#     Những lần sau sẽ lấy từ RAM.
#     """

#     index_path = str(Path(index_path).resolve())

#     if index_path not in _FAISS_CACHE:

#         print(f"[INFO] Loading FAISS index:\n{index_path}")

#         _FAISS_CACHE[index_path] = faiss.read_index(index_path)

#     return _FAISS_CACHE[index_path]

# def decompose_standard_narrative_query(query_text):
#     """
#     Phân rã câu truy vấn văn xuôi / có cấu trúc chuỗi thời gian trong Standard Query thành:
#     - q_full: Câu truy vấn đầy đủ
#     - q_main: Mệnh đề / hành động trực quan chính (Anchor)
#     - q_context: Mệnh đề diễn biến bổ trợ (Context Clue) nếu có
#     """
#     if not query_text or not isinstance(query_text, str):
#         return [query_text]

#     # Các từ nối chỉ mốc thời gian / diễn biến thường gặp trong đề KIS
#     temporal_splitters = [
#         r"(?i)\bbiết sau đó\b",
#         r"(?i)\bvà sau đó\b",
#         r"(?i)\bsau đó\b",
#         r"(?i)\bsau cùng\b",
#         r"(?i)\btiếp theo\b",
#         r"(?i)\bkế tiếp\b",
#         r"(?i)\bđoạn sau\b",
#         r"(?i)\blúc sau\b",
#         r"(?i)\bở phần sau\b",
#         r"(?i)\brồi sau đó\b",
#         r"(?i)\bgiảng bài về\b",
#         r"(?i)\bgiảng về\b",
#         r"(?i)\bnói về\b",
#         r"(?i)\btrình bày về\b",
#         r"(?i)\btrên slide\b",
#         r"(?i)\bslide chứa\b",
#         r"(?i)\bvới nội dung\b"
#     ]
    
#     # 1. Thử tách theo từ nối thời gian
#     for pattern in temporal_splitters:
#         parts = re.split(pattern, query_text)
#         if len(parts) > 1 and len(parts[0].strip()) > 10:
#             q_main = parts[0].strip().rstrip(".,; ")
#             q_context = " ".join([p.strip() for p in parts[1:] if p.strip()])
#             print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#             print(f"  -> Q_Full   : '{query_text}'")
#             print(f"  -> Q_Anchor : '{q_main}'")
#             print(f"  -> Q_Context: '{q_context}'")
#             return [query_text, q_main, q_context]
            
#     # # 2. Thử tách theo dấu câu nếu câu dài chứa nhiều mệnh đề
#     sentences = [s.strip() for s in re.split(r"[.\n]+", query_text) if len(s.strip()) > 10]
#     if len(sentences) >= 2 and len(query_text.split()) > 15:
#         q_main = sentences[0]
#         q_context = " ".join(sentences[1:])
#         print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#         print(f"  -> Q_Full   : '{query_text}'")
#         print(f"  -> Q_Anchor : '{q_main}'")
#         print(f"  -> Q_Context: '{q_context}'")
#         return [query_text, q_main, q_context]
        
#     return [query_text]


# MAX_RETRY = 1
# def expand_and_translate_query(query_text, device="cpu"):

#     if not query_text or not query_text.strip():
#         return query_text

    
#     try:

#         translated = translate_with_envit5(
#             query_text,
#             device="cpu"
#         )

#         if translated and translated.strip():

#             print("[INFO]  Fallback Success")
#             print(translated)

#             return translated

#     except Exception as e:

#         print(f"[WARNING] cũng lỗi: {e}")

#     # ======================================================
#     # TẦNG 3: ORIGINAL
#     # ======================================================

#     print("[WARNING] Sử dụng query gốc.")

#     return query_text

# def load_config(config_path):
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)
# # ==========================================================
# # 1. SIGLIP 2 PIPELINE
# # ==========================================================

# _SIGLIP2_CACHE = {"processor": None, "model": None}

# def get_siglip2_model(device="cuda"):
#     if _SIGLIP2_CACHE["model"] is None:
#         model_name = "google/siglip2-base-patch16-256"
#         print(f"[INFO] Loading SigLIP 2 ({model_name})...")
#         _SIGLIP2_CACHE["processor"] = AutoProcessor.from_pretrained(model_name)
#         _SIGLIP2_CACHE["model"] = AutoModel.from_pretrained(model_name).to(device)
#         _SIGLIP2_CACHE["model"].eval()
#     return _SIGLIP2_CACHE["processor"], _SIGLIP2_CACHE["model"]



# # ==========================================================
# # 3. DFN5B-CLIP-ViT-H-14 PIPELINE (MỚI BỔ SUNG)
# # ==========================================================

# _DFN5B_CACHE = {"model": None, "tokenizer": None}

# def get_dfn5b_vit_h14_model(device="cuda"):
#     if _DFN5B_CACHE["model"] is None:
#         model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
#         print(f"[INFO] Loading DFN5B-CLIP-ViT-H-14 ({model_name})...")
#         model, _, _ = open_clip.create_model_and_transforms(model_name)
#         model = model.to(device)
#         model.eval()
#         tokenizer = open_clip.get_tokenizer(model_name)
        
#         _DFN5B_CACHE["model"] = model
#         _DFN5B_CACHE["tokenizer"] = tokenizer
#     return _DFN5B_CACHE["model"], _DFN5B_CACHE["tokenizer"]

# # ==============================
# # 1) ADD THESE IMPORTS
# # ==============================
# import json
# from collections import defaultdict


# # ==============================
# # 2) ADD THESE HELPERS
# # ==============================
# _VECTOR_META_CACHE = {}

# import sqlite3
# import os

# def load_vector_meta_map(mapping_path: str):
#     """
#     Hỗ trợ đọc mapping từ file SQLite (.db) hoặc file JSON cũ.
#     """
#     if not mapping_path or not os.path.exists(mapping_path):
#         return {}

#     # Kiểm tra nếu là file SQLite (.db)
#     if mapping_path.endswith('.db') or mapping_path.endswith('.sqlite'):
#         meta_map = {}
#         try:
#             conn = sqlite3.connect(mapping_path)
#             cursor = conn.cursor()
#             # Giả định bảng tên là keyframes, có các cột vector_index, video_id, frame_idx,...
#             cursor.execute("SELECT vector_index, video_id, frame_idx FROM keyframes")
#             for row in cursor.fetchall():
#                 v_idx, vid, f_idx = row
#                 meta_map[int(v_idx)] = {
#                     "video_id": vid,
#                     "frame_idx": f_idx
#                 }
#             conn.close()
#         except Exception as e:
#             print(f"[ERROR] Không thể đọc dữ liệu từ SQLite database {mapping_path}: {e}")
#         return meta_map
#     else:
#         # Fallback đọc file JSON cũ nếu cần
#         import json
#         with open(mapping_path, "r", encoding="utf-8") as f:
#             return json.load(f)

# def _search_faiss_hits(index_path, query_embedding, top_k, threshold):
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     hits = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         hits.append({
#             "rank": rank,
#             "vector_index": int(idx),
#             "score": float(score),
#         })
#         rank += 1

#     return hits


# def _aggregate_hits_to_videos(hits, vector_meta_map, allowed_videos=None):
#     """
#     Gộp frame-level hits thành score theo video_id.

#     video_score = mean(top-3 frame scores) trong video đó.
#     """
#     allowed_set = set(allowed_videos) if allowed_videos is not None else None

#     video_dict = defaultdict(lambda: {
#         "video_id": "",
#         "best_score": -1e9,
#         "best_vector_index": None,
#         "best_frame_idx": None,
#         "best_keyframe_path": "",
#         "best_object_path": "",
#         "best_metadata_path": "",
#         "best_pts_time": 0.0,
#         "best_fps": 25.0,
#         "frame_scores": [],
#         "hit_count": 0,
#     })

#     for hit in hits:
#         v_idx = hit["vector_index"]
#         meta = vector_meta_map.get(v_idx)
#         if not meta:
#             continue

#         video_id = str(meta.get("video_id", ""))
#         if not video_id:
#             continue

#         if allowed_set is not None and video_id not in allowed_set:
#             continue

#         cur_score = float(hit["score"])
#         item = video_dict[video_id]
#         item["video_id"] = video_id
#         item["frame_scores"].append(cur_score)
#         item["hit_count"] += 1

#         if cur_score > item["best_score"]:
#             item["best_score"] = cur_score
#             item["best_vector_index"] = v_idx
#             item["best_frame_idx"] = int(meta.get("frame_idx", 0))
#             item["best_keyframe_path"] = meta.get("keyframe_path", "")
#             item["best_object_path"] = meta.get("object_path", "")
#             item["best_metadata_path"] = meta.get("metadata_path", "")
#             item["best_pts_time"] = float(meta.get("pts_time", 0.0))
#             item["best_fps"] = float(meta.get("fps", 25.0))

#     for vid, item in video_dict.items():
#         top_scores = sorted(item["frame_scores"], reverse=True)[:3]
#         item["video_score"] = float(np.mean(top_scores)) if top_scores else 0.0
#         item["consensus_hits"] = len(top_scores)

#     return video_dict


# def _video_consensus_retrieval(query_text, index_path, config_path, mapping_path, embed_fn):
#     """
#     Generic video-consensus retrieval:
#       q_main selects candidate videos
#       q_full / q_context only verify within those videos
#     """
#     cfg = load_config(config_path)
#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     consensus_top_videos = int(cfg.get("consensus_top_videos", 50))

#     part_weights_cfg = cfg.get("query_part_weights") or {}
#     part_weights = {
#         "main": float(part_weights_cfg.get("main", 0.70)),
#         "full": float(part_weights_cfg.get("full", 0.20)),
#         "context": float(part_weights_cfg.get("context", 0.10)),
#     }

#     sub_queries = decompose_standard_narrative_query(query_text)
#     q_full = sub_queries[0] if len(sub_queries) > 0 else query_text
#     q_main = sub_queries[1] if len(sub_queries) > 1 else q_full
#     q_context = sub_queries[2] if len(sub_queries) > 2 else ""

#     q_full_en = expand_and_translate_query(q_full, device="cpu")
#     q_main_en = expand_and_translate_query(q_main, device="cpu")
#     q_context_en = expand_and_translate_query(q_context, device="cpu") if q_context.strip() else ""

#     emb_main = embed_fn(q_main_en)
#     emb_full = embed_fn(q_full_en)
#     emb_context = embed_fn(q_context_en) if q_context_en.strip() else None

#     hits_main = _search_faiss_hits(
#         index_path,
#         emb_main.cpu().numpy().astype(np.float32),
#         top_k,
#         threshold,
#     )
#     hits_full = _search_faiss_hits(
#         index_path,
#         emb_full.cpu().numpy().astype(np.float32),
#         top_k,
#         threshold,
#     )
#     hits_context = []
#     if emb_context is not None:
#         hits_context = _search_faiss_hits(
#             index_path,
#             emb_context.cpu().numpy().astype(np.float32),
#             top_k,
#             threshold,
#         )

#     vector_meta_map = load_vector_meta_map(mapping_path)

#     # 1. Gom hits thành video_score cho q_main
#     main_video_map = _aggregate_hits_to_videos(hits_main, vector_meta_map)
#     full_video_map_raw = _aggregate_hits_to_videos(hits_full, vector_meta_map)

#     # 2. STRICT CONSENSUS: Chỉ lấy top video từ q_main làm ứng viên duy nhất
#     top_main_vids = [
#         v["video_id"] for v in sorted(main_video_map.values(), key=lambda x: x["video_score"], reverse=True)[:consensus_top_videos]
#     ]
    
#     # Loại bỏ hoàn toàn việc union với top_full_vids
#     allowed_videos = top_main_vids

#     if not allowed_videos:
#         print("[INFO] No video candidates found from q_main.")
#         return []

#     allowed_set = set(allowed_videos)

#     # 3. q_full và q_context chỉ chấm điểm giới hạn trong tập video của q_main
#     full_video_map = _aggregate_hits_to_videos(hits_full, vector_meta_map, allowed_videos=allowed_set)
#     context_video_map = _aggregate_hits_to_videos(hits_context, vector_meta_map, allowed_videos=allowed_set)

#     final_videos = []
#     for vid in allowed_videos:
#         main_item = main_video_map.get(vid)
#         full_item = full_video_map.get(vid)
#         ctx_item = context_video_map.get(vid)

#         main_score = main_item["video_score"] if main_item else 0.0
#         full_score = full_item["video_score"] if full_item else 0.0
#         ctx_score = ctx_item["video_score"] if ctx_item else 0.0

#         part_hit_count = sum([
#             1 if main_score > 0 else 0,
#             1 if full_score > 0 else 0,
#             1 if ctx_score > 0 else 0,
#         ])

#         final_score = (
#             part_weights["main"] * main_score +
#             part_weights["full"] * full_score +
#             part_weights["context"] * ctx_score
#         )

#         # bonus nhỏ cho consensus
#         if part_hit_count >= 2:
#             final_score *= 1.03
#         if part_hit_count == 3:
#             final_score *= 1.05

#         best_candidates = [x for x in [main_item, full_item, ctx_item] if x is not None]
#         best_ref = max(best_candidates, key=lambda x: x["best_score"]) if best_candidates else None

#         final_videos.append({
#             "video_id": vid,
#             "rank": 0,  # gán sau
#             "score": float(final_score),       # để compatible với multi-model file
#             "final_score": float(final_score),  # rõ nghĩa hơn
#             "main_score": float(main_score),
#             "full_score": float(full_score),
#             "context_score": float(ctx_score),
#             "consensus_hits": int(part_hit_count),

#             # frame đại diện tốt nhất trong video
#             "vector_index": int(best_ref["best_vector_index"]) if best_ref and best_ref["best_vector_index"] is not None else -1,
#             "frame_idx": int(best_ref["best_frame_idx"]) if best_ref and best_ref["best_frame_idx"] is not None else 0,
#             "keyframe_path": best_ref["best_keyframe_path"] if best_ref else "",
#             "object_path": best_ref["best_object_path"] if best_ref else "",
#             "metadata_path": best_ref["best_metadata_path"] if best_ref else "",
#             "pts_time": float(best_ref["best_pts_time"]) if best_ref else 0.0,
#             "fps": float(best_ref["best_fps"]) if best_ref else 25.0,
#         })

#     final_videos.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, item in enumerate(final_videos, start=1):
#         item["rank"] = i

#     print(f"[INFO] Video-consensus retrieval returned {len(final_videos)} candidates.")
#     return final_videos



# # ==============================
# # 3) REPLACE siglip2_retrieval_pipeline()
# # ==============================
# def siglip2_retrieval_pipeline(
#     query_text,
#     index_path,
#     config_path,
#     mapping_path=None,
#     use_video_consensus=None,
# ):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     cfg = load_config(config_path)

#     if use_video_consensus is None:
#         use_video_consensus = bool(cfg.get("use_video_consensus", False)) and mapping_path is not None

#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     processor, model = get_siglip2_model(device)
#     max_length = model.config.text_config.max_position_embeddings

#     # -------- video consensus path --------
#     if use_video_consensus:
#         if not mapping_path:
#             raise ValueError("mapping_path is required when use_video_consensus=True")

#         def _embed_one(text):
#             with torch.no_grad():
#                 inputs = processor(
#                     text=[text],
#                     return_tensors="pt",
#                     padding="max_length",
#                     truncation=True,
#                     max_length=max_length,
#                 )
#                 inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#                 text_outputs = model.get_text_features(**inputs)

#                 if hasattr(text_outputs, "pooler_output"):
#                     tf = text_outputs.pooler_output
#                 elif hasattr(text_outputs, "last_hidden_state"):
#                     tf = text_outputs.last_hidden_state.mean(dim=1)
#                 else:
#                     tf = text_outputs

#                 return F.normalize(tf, p=2, dim=1)

#         results = _video_consensus_retrieval(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#             mapping_path=mapping_path,
#             embed_fn=_embed_one,
#         )
#         return results

#     # -------- original vector-level fallback --------
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     embeddings_list = []
#     with torch.no_grad():
#         for t_query in translated_sub_queries:
#             inputs = processor(
#                 text=[t_query],
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=max_length,
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#             text_outputs = model.get_text_features(**inputs)

#             if hasattr(text_outputs, "pooler_output"):
#                 tf = text_outputs.pooler_output
#             elif hasattr(text_outputs, "last_hidden_state"):
#                 tf = text_outputs.last_hidden_state.mean(dim=1)
#             else:
#                 tf = text_outputs

#             tf = F.normalize(tf, p=2, dim=1)
#             embeddings_list.append(tf)

#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().numpy().astype(np.float32)

#     print("[INFO] Loading SigLIP 2 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] SigLIP 2 Retrieval returned {len(results)} candidates.")
#     return results



# # ==============================
# # 4) REPLACE dfn5b_vit_h14_retrieval_pipeline()
# # ==============================
# def dfn5b_vit_h14_retrieval_pipeline(
#     query_text,
#     index_path,
#     config_path,
#     mapping_path=None,
#     use_video_consensus=None,
# ):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     cfg = load_config(config_path)

#     if use_video_consensus is None:
#         use_video_consensus = bool(cfg.get("use_video_consensus", False)) and mapping_path is not None

#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model, tokenizer = get_dfn5b_vit_h14_model(device)

#     # -------- video consensus path --------
#     if use_video_consensus and mapping_path and Path(mapping_path).exists():
#         def _embed_one(text):
#             with torch.no_grad(), torch.amp.autocast(device):
#                 text_tokens = tokenizer([text]).to(device)
#                 tf = model.encode_text(text_tokens)
#                 return F.normalize(tf, p=2, dim=-1)

#         results = _video_consensus_retrieval(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#             mapping_path=mapping_path,
#             embed_fn=_embed_one,
#         )
#         return results
#     elif use_video_consensus:
#         print("[WARNING] mapping_path is missing or invalid. Falling back to standard vector search.")
#     # -------- original vector-level fallback --------
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     embeddings_list = []
#     with torch.no_grad(), torch.amp.autocast(device):
#         for t_query in translated_sub_queries:
#             text_tokens = tokenizer([t_query]).to(device)
#             tf = model.encode_text(text_tokens)
#             tf = F.normalize(tf, p=2, dim=-1)
#             embeddings_list.append(tf)

#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().float().numpy().astype(np.float32)

#     print("[INFO] Loading DFN5B-CLIP-ViT-H-14 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] DFN5B-CLIP-ViT-H-14 Retrieval returned {len(results)} candidates.")
#     return results


# # ==========================================================
# # Main Execution Block (Ví dụ kiểm thử)
# # ==========================================================

# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).resolve().parent.parent.parent

#     # Đường dẫn trỏ tới file index của DFN5B-CLIP-ViT-H-14 mà bạn vừa tạo
#     DFN5B_INDEX = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index"
#     CONFIG_PATH = BASE_DIR / "configs" / "retrieval.yaml"

#     query = "a person walking"

#     results = dfn5b_vit_h14_retrieval_pipeline(
#         query_text=query,
#         index_path=str(DFN5B_INDEX),
#         config_path=str(CONFIG_PATH),
#     )

#     print("=" * 60)
#     for r in results[:5]:
#         print(r)


























# import yaml
# import re
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from transformers import AutoModel, AutoProcessor
# from deep_translator import GoogleTranslator
# import faiss
# import clip  # Thư viện openai-clip cho ViT-B/32
# import open_clip
# from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
# import torch
# import time
# import gc
# import random
# from huggingface_hub import hf_hub_download
# import sentencepiece as spm
# try:
#     from src.retrieval.vlm import GeminiVLM, load_vlm_from_project_root
# except Exception:
#     GeminiVLM = None
#     load_vlm_from_project_root = None


# # bộ dịch envit5
# # Biến toàn cục lưu giữ Instance của EnViT5 cố định trong RAM
# _ENVIT5_MODEL = None
# _ENVIT5_TOKENIZER = None
# def get_envit5_translator(device: str = "cpu"):
#     global _ENVIT5_MODEL, _ENVIT5_SPM

#     if _ENVIT5_MODEL is None or _ENVIT5_SPM is None:
#         model_name = "VietAI/envit5-translation"
#         print(
#             f"[INFO] Loading {model_name} into memory on {device} (One-time load)..."
#         )

#         # 1. Tải file spiece.model từ Hub
#         spm_path = hf_hub_download(repo_id=model_name, filename="spiece.model")

#         # 2. Dùng trực tiếp SentencePieceProcessor chuẩn của Google (Không thông qua transformers tokenizer)
#         _ENVIT5_SPM = spm.SentencePieceProcessor()
#         _ENVIT5_SPM.load(spm_path)

#         # 3. Nạp Model EnViT5
#         _ENVIT5_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float32 if device == "cpu" else torch.float16,
#         ).to(device)
#         _ENVIT5_MODEL.eval()

#         print("[SUCCESS] VietAI EnViT5 loaded into RAM successfully!")

#     return _ENVIT5_MODEL, _ENVIT5_SPM


# def translate_with_envit5(query_text: str, device: str = "cpu") -> str:
#     """
#     Hàm dịch tối ưu cho EnViT5 với Beam Search nhẹ và quản lý thiết bị tốt hơn.
#     """
#     if not query_text or not query_text.strip():
#         return ""

#     model, sp = get_envit5_translator(device=device)

#     # Đảm bảo model đã nằm đúng device
#     model.to(device)
#     model.eval()

#     # Chuẩn hóa input format của EnViT5
#     input_text = f"vi: {query_text.strip()}"

#     # Mã hóa trực tiếp bằng SentencePiece
#     input_ids = sp.encode(input_text) + [1] # Thêm EOS token (ID 1)
#     input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_tensor,
#             max_new_tokens=256,        # Dùng max_new_tokens thay vì max_length để tránh bị cắt cụt câu dài
#             num_beams=4,               # Tăng num_beams lên 4 để cải thiện chất lượng dịch (giảm lủng củng)
#             early_stopping=True,       # Dừng sớm khi gặp token kết thúc beam search
#             do_sample=False,
#             decoder_start_token_id=0,  # Token bắt đầu giải mã của EnViT5
#             eos_token_id=1,
#             pad_token_id=0,
#         )

#     # Giải mã ID token ra chuỗi văn bản
#     output_ids = outputs[0].tolist()
#     translated = sp.decode(output_ids).strip()

#     # Loại bỏ tiền tố "en:" hoặc các biến thể khoảng trắng bằng Regex cho triệt để
#     translated = re.sub(r'^(en\s*:\s*)', '', translated, flags=re.IGNORECASE).strip()

#     return translated
    
# _SIGLIP2_CACHE = {
#     "processor": None,
#     "model": None
# }

# _DFN5B_CACHE = {
#     "model": None,
#     "tokenizer": None
# }

# # 1. Tự động nhận diện nếu có GPU, nếu không thì dùng CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[INFO] Using device: {device}")

# # NEW
# _FAISS_CACHE = {}
# def get_faiss_index(index_path):
#     """
#     Load FAISS index một lần duy nhất.
#     Những lần sau sẽ lấy từ RAM.
#     """

#     index_path = str(Path(index_path).resolve())

#     if index_path not in _FAISS_CACHE:

#         print(f"[INFO] Loading FAISS index:\n{index_path}")

#         _FAISS_CACHE[index_path] = faiss.read_index(index_path)

#     return _FAISS_CACHE[index_path]

# def decompose_standard_narrative_query(query_text):
#     """
#     Phân rã câu truy vấn văn xuôi / có cấu trúc chuỗi thời gian trong Standard Query thành:
#     - q_full: Câu truy vấn đầy đủ
#     - q_main: Mệnh đề / hành động trực quan chính (Anchor)
#     - q_context: Mệnh đề diễn biến bổ trợ (Context Clue) nếu có
#     """
#     if not query_text or not isinstance(query_text, str):
#         return [query_text]

#     # Các từ nối chỉ mốc thời gian / diễn biến thường gặp trong đề KIS
#     temporal_splitters = [
#         r"(?i)\bbiết sau đó\b",
#         r"(?i)\bvà sau đó\b",
#         r"(?i)\bsau đó\b",
#         r"(?i)\bsau cùng\b",
#         r"(?i)\btiếp theo\b",
#         r"(?i)\bkế tiếp\b",
#         r"(?i)\bđoạn sau\b",
#         r"(?i)\blúc sau\b",
#         r"(?i)\bở phần sau\b",
#         r"(?i)\brồi sau đó\b",
#         r"(?i)\bgiảng bài về\b",
#         r"(?i)\bgiảng về\b",
#         r"(?i)\bnói về\b",
#         r"(?i)\btrình bày về\b",
#         r"(?i)\btrên slide\b",
#         r"(?i)\bslide chứa\b",
#         r"(?i)\bvới nội dung\b"
#     ]
    
#     # 1. Thử tách theo từ nối thời gian
#     for pattern in temporal_splitters:
#         parts = re.split(pattern, query_text)
#         if len(parts) > 1 and len(parts[0].strip()) > 10:
#             q_main = parts[0].strip().rstrip(".,; ")
#             q_context = " ".join([p.strip() for p in parts[1:] if p.strip()])
#             print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#             print(f"  -> Q_Full   : '{query_text}'")
#             print(f"  -> Q_Anchor : '{q_main}'")
#             print(f"  -> Q_Context: '{q_context}'")
#             return [query_text, q_main, q_context]
            
#     # # 2. Thử tách theo dấu câu nếu câu dài chứa nhiều mệnh đề
#     sentences = [s.strip() for s in re.split(r"[.\n]+", query_text) if len(s.strip()) > 10]
#     if len(sentences) >= 2 and len(query_text.split()) > 15:
#         q_main = sentences[0]
#         q_context = " ".join(sentences[1:])
#         print(f"[INFO] Narrative Query Decomposition (Standard Search):")
#         print(f"  -> Q_Full   : '{query_text}'")
#         print(f"  -> Q_Anchor : '{q_main}'")
#         print(f"  -> Q_Context: '{q_context}'")
#         return [query_text, q_main, q_context]
        
#     return [query_text]


# MAX_RETRY = 1
# def expand_and_translate_query(query_text, device="cpu"):

#     if not query_text or not query_text.strip():
#         return query_text

    
#     try:

#         translated = translate_with_envit5(
#             query_text,
#             device="cpu"
#         )

#         if translated and translated.strip():

#             print("[INFO]  Fallback Success")
#             print(translated)

#             return translated

#     except Exception as e:

#         print(f"[WARNING] cũng lỗi: {e}")

#     # ======================================================
#     # TẦNG 3: ORIGINAL
#     # ======================================================

#     print("[WARNING] Sử dụng query gốc.")

#     return query_text

# def load_config(config_path):
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def _coerce_query_plan(query_plan, query_text=None):
#     """
#     Normalize query_plan into a plain dict.
#     Accepts:
#       - dict
#       - JSON string
#       - dataclass / pydantic-like object with to_dict/model_dump
#     """
#     if query_plan is None:
#         return None

#     if hasattr(query_plan, "model_dump"):
#         query_plan = query_plan.model_dump()
#     elif hasattr(query_plan, "to_dict"):
#         query_plan = query_plan.to_dict()
#     elif isinstance(query_plan, str):
#         try:
#             query_plan = json.loads(query_plan)
#         except Exception:
#             return None

#     if not isinstance(query_plan, dict):
#         return None

#     plan = dict(query_plan)
#     if query_text and not plan.get("main_query"):
#         plan["main_query"] = query_text
#     if not plan.get("context_query"):
#         plan["context_query"] = ""
#     if not plan.get("sub_queries"):
#         main_query = plan.get("main_query", query_text or "")
#         plan["sub_queries"] = [main_query] if main_query else []
#     if not plan.get("hypotheses"):
#         plan["hypotheses"] = []
#     return plan
# # ==========================================================
# # 1. SIGLIP 2 PIPELINE
# # ==========================================================

# _SIGLIP2_CACHE = {"processor": None, "model": None}

# def get_siglip2_model(device="cuda"):
#     if _SIGLIP2_CACHE["model"] is None:
#         model_name = "google/siglip2-base-patch16-256"
#         print(f"[INFO] Loading SigLIP 2 ({model_name})...")
#         _SIGLIP2_CACHE["processor"] = AutoProcessor.from_pretrained(model_name)
#         _SIGLIP2_CACHE["model"] = AutoModel.from_pretrained(model_name).to(device)
#         _SIGLIP2_CACHE["model"].eval()
#     return _SIGLIP2_CACHE["processor"], _SIGLIP2_CACHE["model"]


# # def siglip2_retrieval_pipeline(query_text, index_path, config_path):
# #     # Dịch Q_full trực tiếp bằng EnViT5
# #     q_full_en = expand_and_translate_query(query_text, device="cpu")

# #     cfg = load_config(config_path)
# #     top_k = cfg.get("top_k", 100)
# #     threshold = cfg.get("retrieval_threshold", 0.0)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     processor, model = get_siglip2_model(device)
# #     max_length = model.config.text_config.max_position_embeddings

# #     # Embed Q_full_en
# #     with torch.no_grad():
# #         inputs = processor(
# #             text=[q_full_en], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
# #         )
# #         inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
# #         text_outputs = model.get_text_features(**inputs)
        
# #         if hasattr(text_outputs, "pooler_output"):
# #             tf = text_outputs.pooler_output
# #         elif hasattr(text_outputs, "last_hidden_state"):
# #             tf = text_outputs.last_hidden_state.mean(dim=1)
# #         else:
# #             tf = text_outputs
            
# #         tf = F.normalize(tf, p=2, dim=1)

# #     query_embedding = tf.cpu().numpy().astype(np.float32)

# #     index = get_faiss_index(index_path)
# #     scores, indices = index.search(query_embedding, top_k)

# #     results = []
# #     rank = 1
# #     for score, idx in zip(scores[0], indices[0]):
# #         if idx == -1 or score < threshold:
# #             continue
# #         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
# #         rank += 1

# #     return results

# # ==========================================================
# # 3. DFN5B-CLIP-ViT-H-14 PIPELINE (MỚI BỔ SUNG)
# # ==========================================================

# _DFN5B_CACHE = {"model": None, "tokenizer": None}

# def get_dfn5b_vit_h14_model(device="cuda"):
#     if _DFN5B_CACHE["model"] is None:
#         model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
#         print(f"[INFO] Loading DFN5B-CLIP-ViT-H-14 ({model_name})...")
#         model, _, _ = open_clip.create_model_and_transforms(model_name)
#         model = model.to(device)
#         model.eval()
#         tokenizer = open_clip.get_tokenizer(model_name)
        
#         _DFN5B_CACHE["model"] = model
#         _DFN5B_CACHE["tokenizer"] = tokenizer
#     return _DFN5B_CACHE["model"], _DFN5B_CACHE["tokenizer"]


# # def dfn5b_vit_h14_retrieval_pipeline(query_text, index_path, config_path):
# #     # Dịch Q_full trực tiếp bằng EnViT5
# #     q_full_en = expand_and_translate_query(query_text, device="cpu")

# #     cfg = load_config(config_path)
# #     top_k = cfg.get("top_k", 100)
# #     threshold = cfg.get("retrieval_threshold", 0.0)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     model, tokenizer = get_dfn5b_vit_h14_model(device)

# #     # Embed Q_full_en
# #     with torch.no_grad(), torch.amp.autocast(device):
# #         text_tokens = tokenizer([q_full_en]).to(device)
# #         tf = model.encode_text(text_tokens)
# #         tf = F.normalize(tf, p=2, dim=-1)

# #     query_embedding = tf.cpu().float().numpy().astype(np.float32)

# #     index = get_faiss_index(index_path)
# #     scores, indices = index.search(query_embedding, top_k)

# #     results = []
# #     rank = 1
# #     for score, idx in zip(scores[0], indices[0]):
# #         if idx == -1 or score < threshold:
# #             continue
# #         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
# #         rank += 1

# #     return results


# # ==============================
# # 1) ADD THESE IMPORTS
# # ==============================
# import json
# from collections import defaultdict


# # ==============================
# # 2) ADD THESE HELPERS
# # ==============================
# _VECTOR_META_CACHE = {}


# def load_vector_meta_map(mapping_path: str):
#     """
#     Load file mapping vector_index -> metadata.

#     Hỗ trợ format:
#       - list[dict]
#       - dict[str/int -> dict]

#     Mong đợi các field (best effort):
#       - vector_index
#       - video_id / video
#       - frame_idx / keyframe_index
#       - keyframe_path
#       - object_path / object_json_path
#       - metadata_path / metadata_cache
#       - pts_time
#       - fps
#     """
#     mapping_path = str(Path(mapping_path).resolve())

#     if mapping_path in _VECTOR_META_CACHE:
#         return _VECTOR_META_CACHE[mapping_path]

#     with open(mapping_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     meta_map = {}

#     if isinstance(data, list):
#         for item in data:
#             if not isinstance(item, dict):
#                 continue
#             if "vector_index" not in item:
#                 continue

#             v_idx = int(item["vector_index"])
#             meta_map[v_idx] = {
#                 "video_id": str(item.get("video_id", item.get("video", ""))),
#                 "frame_idx": int(item.get("frame_idx", item.get("keyframe_index", 0))),
#                 "keyframe_path": item.get("keyframe_path", ""),
#                 "object_path": item.get("object_json_path", item.get("object_path", "")),
#                 "metadata_path": item.get("metadata_cache", item.get("metadata_path", "")),
#                 "pts_time": float(item.get("pts_time", 0.0)),
#                 "fps": float(item.get("fps", 25.0)),
#             }

#     elif isinstance(data, dict):
#         for k, item in data.items():
#             if not isinstance(item, dict):
#                 continue
#             v_idx = int(item.get("vector_index", k))
#             meta_map[v_idx] = {
#                 "video_id": str(item.get("video_id", item.get("video", ""))),
#                 "frame_idx": int(item.get("frame_idx", item.get("keyframe_index", 0))),
#                 "keyframe_path": item.get("keyframe_path", ""),
#                 "object_path": item.get("object_json_path", item.get("object_path", "")),
#                 "metadata_path": item.get("metadata_cache", item.get("metadata_path", "")),
#                 "pts_time": float(item.get("pts_time", 0.0)),
#                 "fps": float(item.get("fps", 25.0)),
#             }

#     _VECTOR_META_CACHE[mapping_path] = meta_map
#     return meta_map


# def _search_faiss_hits(index_path, query_embedding, top_k, threshold):
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     hits = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         hits.append({
#             "rank": rank,
#             "vector_index": int(idx),
#             "score": float(score),
#         })
#         rank += 1

#     return hits


# def _aggregate_hits_to_videos(hits, vector_meta_map, allowed_videos=None):
#     """
#     Gộp frame-level hits thành score theo video_id.

#     video_score = mean(top-3 frame scores) trong video đó.
#     """
#     allowed_set = set(allowed_videos) if allowed_videos is not None else None

#     video_dict = defaultdict(lambda: {
#         "video_id": "",
#         "best_score": -1e9,
#         "best_vector_index": None,
#         "best_frame_idx": None,
#         "best_keyframe_path": "",
#         "best_object_path": "",
#         "best_metadata_path": "",
#         "best_pts_time": 0.0,
#         "best_fps": 25.0,
#         "frame_scores": [],
#         "hit_count": 0,
#     })

#     for hit in hits:
#         v_idx = hit["vector_index"]
#         meta = vector_meta_map.get(v_idx)
#         if not meta:
#             continue

#         video_id = str(meta.get("video_id", ""))
#         if not video_id:
#             continue

#         if allowed_set is not None and video_id not in allowed_set:
#             continue

#         cur_score = float(hit["score"])
#         item = video_dict[video_id]
#         item["video_id"] = video_id
#         item["frame_scores"].append(cur_score)
#         item["hit_count"] += 1

#         if cur_score > item["best_score"]:
#             item["best_score"] = cur_score
#             item["best_vector_index"] = v_idx
#             item["best_frame_idx"] = int(meta.get("frame_idx", 0))
#             item["best_keyframe_path"] = meta.get("keyframe_path", "")
#             item["best_object_path"] = meta.get("object_path", "")
#             item["best_metadata_path"] = meta.get("metadata_path", "")
#             item["best_pts_time"] = float(meta.get("pts_time", 0.0))
#             item["best_fps"] = float(meta.get("fps", 25.0))

#     for vid, item in video_dict.items():
#         top_scores = sorted(item["frame_scores"], reverse=True)[:3]
#         item["video_score"] = float(np.mean(top_scores)) if top_scores else 0.0
#         item["consensus_hits"] = len(top_scores)

#     return video_dict


# def _video_consensus_retrieval(query_text, index_path, config_path, mapping_path, embed_fn, query_plan=None):
#     """
#     Generic video-consensus retrieval:
#       q_main selects candidate videos
#       q_full / q_context only verify within those videos
#     """
#     cfg = load_config(config_path)
#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     consensus_top_videos = int(cfg.get("consensus_top_videos", 50))

#     part_weights_cfg = cfg.get("query_part_weights") or {}
#     part_weights = {
#         "main": float(part_weights_cfg.get("main", 0.70)),
#         "full": float(part_weights_cfg.get("full", 0.20)),
#         "context": float(part_weights_cfg.get("context", 0.10)),
#     }

#     plan = _coerce_query_plan(query_plan, query_text=query_text)
#     if plan:
#         q_full = query_text
#         q_main = plan.get("main_query") or query_text
#         q_context = plan.get("context_query") or ""
#     else:
#         sub_queries = decompose_standard_narrative_query(query_text)
#         q_full = sub_queries[0] if len(sub_queries) > 0 else query_text
#         q_main = sub_queries[1] if len(sub_queries) > 1 else q_full
#         q_context = sub_queries[2] if len(sub_queries) > 2 else ""

#     q_full_en = expand_and_translate_query(q_full, device="cpu")
#     q_main_en = expand_and_translate_query(q_main, device="cpu")
#     q_context_en = expand_and_translate_query(q_context, device="cpu") if q_context.strip() else ""

#     emb_main = embed_fn(q_main_en)
#     emb_full = embed_fn(q_full_en)
#     emb_context = embed_fn(q_context_en) if q_context_en.strip() else None

#     hits_main = _search_faiss_hits(
#         index_path,
#         emb_main.cpu().numpy().astype(np.float32),
#         top_k,
#         threshold,
#     )
#     hits_full = _search_faiss_hits(
#         index_path,
#         emb_full.cpu().numpy().astype(np.float32),
#         top_k,
#         threshold,
#     )
#     hits_context = []
#     if emb_context is not None:
#         hits_context = _search_faiss_hits(
#             index_path,
#             emb_context.cpu().numpy().astype(np.float32),
#             top_k,
#             threshold,
#         )

#     vector_meta_map = load_vector_meta_map(mapping_path)

#     # 1. Gom hits thành video_score cho q_main
#     main_video_map = _aggregate_hits_to_videos(hits_main, vector_meta_map)
#     full_video_map_raw = _aggregate_hits_to_videos(hits_full, vector_meta_map)

#     # 2. STRICT CONSENSUS: Chỉ lấy top video từ q_main làm ứng viên duy nhất
#     top_main_vids = [
#         v["video_id"] for v in sorted(main_video_map.values(), key=lambda x: x["video_score"], reverse=True)[:consensus_top_videos]
#     ]
    
#     # Loại bỏ hoàn toàn việc union với top_full_vids
#     allowed_videos = top_main_vids

#     if not allowed_videos:
#         print("[INFO] No video candidates found from q_main.")
#         return []

#     allowed_set = set(allowed_videos)

#     # 3. q_full và q_context chỉ chấm điểm giới hạn trong tập video của q_main
#     full_video_map = _aggregate_hits_to_videos(hits_full, vector_meta_map, allowed_videos=allowed_set)
#     context_video_map = _aggregate_hits_to_videos(hits_context, vector_meta_map, allowed_videos=allowed_set)

#     final_videos = []
#     for vid in allowed_videos:
#         main_item = main_video_map.get(vid)
#         full_item = full_video_map.get(vid)
#         ctx_item = context_video_map.get(vid)

#         main_score = main_item["video_score"] if main_item else 0.0
#         full_score = full_item["video_score"] if full_item else 0.0
#         ctx_score = ctx_item["video_score"] if ctx_item else 0.0

#         part_hit_count = sum([
#             1 if main_score > 0 else 0,
#             1 if full_score > 0 else 0,
#             1 if ctx_score > 0 else 0,
#         ])

#         final_score = (
#             part_weights["main"] * main_score +
#             part_weights["full"] * full_score +
#             part_weights["context"] * ctx_score
#         )

#         # bonus nhỏ cho consensus
#         if part_hit_count >= 2:
#             final_score *= 1.03
#         if part_hit_count == 3:
#             final_score *= 1.05

#         best_candidates = [x for x in [main_item, full_item, ctx_item] if x is not None]
#         best_ref = max(best_candidates, key=lambda x: x["best_score"]) if best_candidates else None

#         final_videos.append({
#             "video_id": vid,
#             "rank": 0,  # gán sau
#             "score": float(final_score),       # để compatible với multi-model file
#             "final_score": float(final_score),  # rõ nghĩa hơn
#             "main_score": float(main_score),
#             "full_score": float(full_score),
#             "context_score": float(ctx_score),
#             "consensus_hits": int(part_hit_count),

#             # frame đại diện tốt nhất trong video
#             "vector_index": int(best_ref["best_vector_index"]) if best_ref and best_ref["best_vector_index"] is not None else -1,
#             "frame_idx": int(best_ref["best_frame_idx"]) if best_ref and best_ref["best_frame_idx"] is not None else 0,
#             "keyframe_path": best_ref["best_keyframe_path"] if best_ref else "",
#             "object_path": best_ref["best_object_path"] if best_ref else "",
#             "metadata_path": best_ref["best_metadata_path"] if best_ref else "",
#             "pts_time": float(best_ref["best_pts_time"]) if best_ref else 0.0,
#             "fps": float(best_ref["best_fps"]) if best_ref else 25.0,
#         })

#     final_videos.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, item in enumerate(final_videos, start=1):
#         item["rank"] = i

#     print(f"[INFO] Video-consensus retrieval returned {len(final_videos)} candidates.")
#     return final_videos



# # ==============================
# # 3) REPLACE siglip2_retrieval_pipeline()
# # ==============================
# def siglip2_retrieval_pipeline(
#     query_text,
#     index_path,
#     config_path,
#     mapping_path=None,
#     use_video_consensus=None,
#     query_plan=None,
# ):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     cfg = load_config(config_path)

#     if use_video_consensus is None:
#         use_video_consensus = bool(cfg.get("use_video_consensus", False)) and mapping_path is not None

#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     processor, model = get_siglip2_model(device)
#     max_length = model.config.text_config.max_position_embeddings

#     # -------- video consensus path --------
#     if use_video_consensus:
#         if not mapping_path:
#             raise ValueError("mapping_path is required when use_video_consensus=True")

#         def _embed_one(text):
#             with torch.no_grad():
#                 inputs = processor(
#                     text=[text],
#                     return_tensors="pt",
#                     padding="max_length",
#                     truncation=True,
#                     max_length=max_length,
#                 )
#                 inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#                 text_outputs = model.get_text_features(**inputs)

#                 if hasattr(text_outputs, "pooler_output"):
#                     tf = text_outputs.pooler_output
#                 elif hasattr(text_outputs, "last_hidden_state"):
#                     tf = text_outputs.last_hidden_state.mean(dim=1)
#                 else:
#                     tf = text_outputs

#                 return F.normalize(tf, p=2, dim=1)

#         results = _video_consensus_retrieval(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#             mapping_path=mapping_path,
#             embed_fn=_embed_one,
#             query_plan=query_plan,
#         )
#         return results

#     # -------- original vector-level fallback --------
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     embeddings_list = []
#     with torch.no_grad():
#         for t_query in translated_sub_queries:
#             inputs = processor(
#                 text=[t_query],
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=max_length,
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
#             text_outputs = model.get_text_features(**inputs)

#             if hasattr(text_outputs, "pooler_output"):
#                 tf = text_outputs.pooler_output
#             elif hasattr(text_outputs, "last_hidden_state"):
#                 tf = text_outputs.last_hidden_state.mean(dim=1)
#             else:
#                 tf = text_outputs

#             tf = F.normalize(tf, p=2, dim=1)
#             embeddings_list.append(tf)

#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().numpy().astype(np.float32)

#     print("[INFO] Loading SigLIP 2 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] SigLIP 2 Retrieval returned {len(results)} candidates.")
#     return results



# # ==============================
# # 4) REPLACE dfn5b_vit_h14_retrieval_pipeline()
# # ==============================
# def dfn5b_vit_h14_retrieval_pipeline(
#     query_text,
#     index_path,
#     config_path,
#     mapping_path=None,
#     use_video_consensus=None,
#     query_plan=None,
# ):
#     sub_queries = decompose_standard_narrative_query(query_text)
#     cfg = load_config(config_path)

#     if use_video_consensus is None:
#         use_video_consensus = bool(cfg.get("use_video_consensus", False)) and mapping_path is not None

#     top_k = int(cfg.get("top_k", 100))
#     threshold = float(cfg.get("retrieval_threshold", 0.0))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model, tokenizer = get_dfn5b_vit_h14_model(device)

#     # -------- video consensus path --------
#     if use_video_consensus and mapping_path and Path(mapping_path).exists():
#         def _embed_one(text):
#             with torch.no_grad(), torch.amp.autocast(device):
#                 text_tokens = tokenizer([text]).to(device)
#                 tf = model.encode_text(text_tokens)
#                 return F.normalize(tf, p=2, dim=-1)

#         results = _video_consensus_retrieval(
#             query_text=query_text,
#             index_path=index_path,
#             config_path=config_path,
#             mapping_path=mapping_path,
#             embed_fn=_embed_one,
#             query_plan=query_plan,
#         )
#         return results
#     elif use_video_consensus:
#         print("[WARNING] mapping_path is missing or invalid. Falling back to standard vector search.")
#     # -------- original vector-level fallback --------
#     translated_sub_queries = [expand_and_translate_query(q) for q in sub_queries if q.strip()]

#     embeddings_list = []
#     with torch.no_grad(), torch.amp.autocast(device):
#         for t_query in translated_sub_queries:
#             text_tokens = tokenizer([t_query]).to(device)
#             tf = model.encode_text(text_tokens)
#             tf = F.normalize(tf, p=2, dim=-1)
#             embeddings_list.append(tf)

#     if len(embeddings_list) == 3:
#         combined_tf = 0.75 * embeddings_list[0] + 0.15 * embeddings_list[1] + 0.10 * embeddings_list[2]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     elif len(embeddings_list) == 2:
#         combined_tf = 0.80 * embeddings_list[0] + 0.20 * embeddings_list[1]
#         combined_tf = F.normalize(combined_tf, p=2, dim=-1)
#     else:
#         combined_tf = embeddings_list[0]

#     query_embedding = combined_tf.cpu().float().numpy().astype(np.float32)

#     print("[INFO] Loading DFN5B-CLIP-ViT-H-14 FAISS index...")
#     index = get_faiss_index(index_path)
#     scores, indices = index.search(query_embedding, top_k)

#     results = []
#     rank = 1
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or score < threshold:
#             continue
#         results.append({"rank": rank, "vector_index": int(idx), "score": float(score)})
#         rank += 1

#     print(f"[INFO] DFN5B-CLIP-ViT-H-14 Retrieval returned {len(results)} candidates.")
#     return results


# # ==========================================================
# # Main Execution Block (Ví dụ kiểm thử)
# # ==========================================================

# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).resolve().parent.parent.parent

#     # Đường dẫn trỏ tới file index của DFN5B-CLIP-ViT-H-14 mà bạn vừa tạo
#     DFN5B_INDEX = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index"
#     CONFIG_PATH = BASE_DIR / "configs" / "retrieval.yaml"

#     query = "a person walking"

#     results = dfn5b_vit_h14_retrieval_pipeline(
#         query_text=query,
#         index_path=str(DFN5B_INDEX),
#         config_path=str(CONFIG_PATH),
#     )

#     print("=" * 60)
#     for r in results[:5]:
#         print(r)