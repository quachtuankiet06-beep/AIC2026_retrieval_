
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
    Đọc Gemini API key từ file .api hoặc .api_dich tại thư mục gốc của project.
    """
    if api_file:
        candidate_paths = [Path(api_file)]
    else:
        root = Path(__file__).resolve().parent.parent.parent
        candidate_paths = [
            root / ".api",
            Path(".api"),
            root / ".api_dich",
            Path(".api_dich")
        ]

    for path in candidate_paths:
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
    Gọi Gemini API với Smart Scene-Aware Prompt để phân tách intent (q_main, q_context).
    - Phân biệt cảnh tĩnh đồng hiện và chuỗi hành động tuần tự.
    - Bảo toàn trọn vẹn số lượng, tên đường, tên riêng và chữ viết.
    - Hỗ trợ cơ chế Retry khi gặp Rate Limit (HTTP 429) và tự động bóc tách Markdown JSON.
    """
    if not api_key:
        api_key = load_gemini_api_key()
    if not api_key:
        raise ValueError("Không tìm thấy Gemini API key trong file .api, .api_dich hoặc biến môi trường.")

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
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest"
    ]

    import urllib.request
    import urllib.error
    import time
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
    }).encode("utf-8")

    last_error = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    # Bóc tách Markdown JSON nếu có (vd: ```json ... ```)
                    if "```" in reply:
                        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", reply, re.DOTALL)
                        if m:
                            reply = m.group(1).strip()

                    parsed = json.loads(reply)
                    return {
                        "q_full": parsed.get("q_full", query_text),
                        "q_main": parsed.get("q_main", query_text),
                        "q_context": parsed.get("q_context", "")
                    }
            except urllib.error.HTTPError as he:
                last_error = f"{model_name}: HTTP {he.code} ({he.reason})"
                if he.code == 429 and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except Exception as e:
                last_error = f"{model_name}: {repr(e)}"
                break

    raise RuntimeError(f"Tất cả candidate models của Gemini API đều không phản hồi ({last_error}).")


_QWEN_MODEL_CACHE = {
    "model": None,
    "tokenizer": None
}
def get_qwen_decomposer_model(device: str = "cuda"):
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

_TRANSLATION_CACHE = {}

def load_gemini_translation_api_key(api_file=None):
    """
    Đọc Gemini API key dành cho dịch thuật từ file .api_dich.
    Nếu không có, fallback sang .api hoặc biến môi trường.
    """
    if api_file:
        candidate_paths = [Path(api_file)]
    else:
        root = Path(__file__).resolve().parent.parent.parent
        candidate_paths = [
            root / ".api_dich",
            Path(".api_dich"),
            root / ".api",
            Path(".api")
        ]

    for path in candidate_paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    return line
    return os.environ.get("GEMINI_API_KEY_DICH") or os.environ.get("GEMINI_API_KEY", "")

def translate_with_gemini_visual(query_text: str, api_key: str = None) -> str:
    """
    Dịch câu truy vấn tiếng Việt sang tiếng Anh với Chuẩn hóa từ vựng thị giác (Visual Vocabulary Standardization).
    - Giữ trọn nghĩa gốc, không tự ý suy diễn hoặc bịa thêm các thực thể/nguyên liệu không có trong câu gốc.
    - Chuẩn hóa chính xác các thực thể thị giác, dụng cụ bếp, vật thể, hành động sang tiếng Anh tự nhiên cho CLIP/SigLIP.
    """
    if not query_text or not query_text.strip():
        return query_text

    if not api_key:
        api_key = load_gemini_translation_api_key()
    if not api_key:
        raise ValueError("Không tìm thấy Gemini API key cho dịch thuật trong .api_dich hoặc .api.")

    system_instruction = (
        "You are an expert translator specializing in Video Retrieval (CLIP/SigLIP models).\n"
        "Your task is to convert Vietnamese search queries into concise, visually-dense English descriptions.\n\n"
        "RULES:\n"
        "1. STRIP FILLER & META-TALK: NEVER include or start with phrases like 'A video showing', 'A scene of', "
        "'The clip begins with', 'Footage of', 'Find a video about', 'A news report about', 'The shot shows'. "
        "Start directly with the core visual subject, setting, or action.\n"
        "2. VISUAL CONCISENESS (< 40 words): Keep the translation dense, compact, and strictly under 40 words "
        "so it fits completely within the 64-token limit of the visual model without truncation.\n"
        "3. FAITHFUL & NO HALLUCINATION: Translate only what is explicitly in the query. Do not invent entities or seasonings.\n"
        "4. VISUAL VOCABULARY STANDARDIZATION:\n"
        "   - 'vá', 'muôi' in cooking context -> 'ladle' (NEVER 'patch')\n"
        "   - 'bột nhào', 'bột lỏng' in food context -> 'dough' or 'batter' (NEVER 'play-dough')\n"
        "   - 'nồi', 'chảo', 'bếp', 'nắp' -> 'pot', 'pan', 'stove', 'pot lid'\n"
        "   - 'sợi trắng' -> 'white noodles / white strands'\n"
        "   - 'vỏ', 'lõi' of food -> 'outer layer / peel / shell', 'inner core / filling'\n"
        "5. Output ONLY the clean English sentence."
    )

    prompt = f"{system_instruction}\n\nQuery to translate: \"{query_text.strip()}\"\n\nEnglish Translation:"

    models_to_try = [
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest"
    ]

    import urllib.request
    import urllib.error
    import time
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }).encode("utf-8")

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if reply.startswith('"') and reply.endswith('"') and len(reply) > 2:
                        reply = reply[1:-1].strip()
                    return reply
            except urllib.error.HTTPError as he:
                if he.code == 429 and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except Exception:
                break

    raise RuntimeError("Tất cả candidate models của Gemini API dịch thuật đều không phản hồi.")


def translate_with_qwen(query_text: str, device: str = "cuda") -> str:
    """
    Dịch và chuẩn hóa từ vựng thị giác bằng Qwen2.5-1.5B-Instruct chạy Local 100%.
    Tái sử dụng trực tiếp instance Qwen đã có trong RAM, không tốn thêm tài nguyên.
    """
    if not query_text or not query_text.strip():
        return ""

    model, tokenizer = get_qwen_decomposer_model(device=device)

    system_instruction = (
        "You are an expert translator specializing in Video Retrieval (CLIP/SigLIP).\n"
        "Translate the following Vietnamese search query into concise, visually-dense English.\n"
        "RULES:\n"
        "1. Start directly with the core visual action or subject (no filler like 'A video of').\n"
        "2. Keep it compact (< 35 words).\n"
        "3. Standardize cooking/visual terms:\n"
        "   - 'vá', 'muôi' -> 'ladle'\n"
        "   - 'bột nhào', 'bột lỏng' -> 'dough' or 'batter'\n"
        "   - 'nồi', 'chảo', 'nắp' -> 'pot', 'pan', 'pot lid'\n"
        "   - 'sợi trắng' -> 'white noodles / white strands'\n"
        "4. Output ONLY the clean English sentence."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": f"Translate to visual English: \"Một người đang đổ nguyên liệu vào nồi thủy tinh, dùng đũa đảo\"",
        },
        {
            "role": "assistant",
            "content": "A person pouring ingredients into a glass pot, stirring with chopsticks.",
        },
        {"role": "user", "content": f'Translate to visual English: "{query_text.strip()}"'},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,  # Greedy decoding để câu dịch ổn định và chuẩn xác nhất
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Cắt bỏ phần prompt, chỉ lấy text sinh ra
    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    translated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Làm sạch dấu ngoặc kép thừa nếu có
    if (
        translated.startswith('"')
        and translated.endswith('"')
        and len(translated) > 2
    ):
        translated = translated[1:-1].strip()

    return translated


def expand_and_translate_query(query_text: str, device: str = "cpu") -> str:
    """
    Dịch và chuẩn hóa câu truy vấn:
    1. Ưu tiên: Gemini 3.1/2.5 Flash Lite với Chuẩn hóa từ vựng thị giác (key .api_dich)
    2. Fallback: Mô hình EnViT5 cục bộ hiện tại nếu Gemini gặp sự cố
    3. Cuối cùng: Trả về câu gốc nếu tất cả đều lỗi
    Kết quả dịch được lưu cache để tái sử dụng xuyên suốt pipeline.
    """
    if not query_text or not query_text.strip():
        return query_text

    clean_q = query_text.strip()
    if clean_q in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[clean_q]

    # ======================================================
    # TẦNG 1: GEMINI VISUAL TRANSLATION (.api_dich)
    # ======================================================
    try:
        translated = translate_with_gemini_visual(clean_q)
        if translated and translated.strip():
            print(f"[INFO] Gemini Visual Translation Success:")
            print(f"  -> VI: '{clean_q}'")
            print(f"  -> EN: '{translated}'")
            _TRANSLATION_CACHE[clean_q] = translated.strip()
            return translated.strip()
    except Exception as e:
        print(f"[WARNING] Gemini Visual Translation failed ({e}). Fallback sang EnViT5.")

    # ======================================================
    # TẦNG 2: FALLBACK SANG ENVIT5 (MẶC ĐỊNH HIỆN TẠI)
    # ======================================================
    try:
        translated = translate_with_envit5(clean_q, device=device)
        if translated and translated.strip():
            print(f"[INFO] EnViT5 Translation Fallback Success:")
            print(f"  -> VI: '{clean_q}'")
            print(f"  -> EN: '{translated}'")
            _TRANSLATION_CACHE[clean_q] = translated.strip()
            return translated.strip()
    except Exception as e:
        print(f"[WARNING] EnViT5 translation cũng gặp lỗi: {e}")

    # ======================================================
    # TẦNG 3: ORIGINAL QUERY
    # ======================================================
    print("[WARNING] Sử dụng query gốc.")
    _TRANSLATION_CACHE[clean_q] = clean_q
    return clean_q

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


# ==========================================================
# 3. DFN5B-CLIP-ViT-H-14 PIPELINE (MỚI BỔ SUNG)
# ==========================================================

_DFN5B_CACHE = {"model": None, "tokenizer": None}

def get_dfn5b_vit_h14_model(device="cuda"):
    if _DFN5B_CACHE["model"] is None:
        model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
        print(f"[INFO] Loading DFN5B-CLIP-ViT-H-14 ({model_name})...")
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        
        # Tối ưu triệt để: Xóa bỏ hoàn toàn Visual Encoder (ViT-Huge ~632M params)
        # khỏi cả RAM lẫn VRAM vì Online Retrieval chỉ cần duy nhất Text Encoder
        if hasattr(model, "visual"):
            del model.visual
            model.visual = None
            import gc
            gc.collect()

        # Nạp phần Text Encoder còn lại lên GPU ở định dạng FP16
        is_cuda = device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")
        if is_cuda:
            model = model.to(device).half()
            torch.cuda.empty_cache()
        else:
            model = model.to(device)
            
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        
        _DFN5B_CACHE["model"] = model
        _DFN5B_CACHE["tokenizer"] = tokenizer
    return _DFN5B_CACHE["model"], _DFN5B_CACHE["tokenizer"]


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








































