# import unicodedata

# def remove_vietnamese_diacritics(text: str) -> str:
#     """Chuẩn hóa chuỗi: chuyển về lowercase và bỏ dấu tiếng Việt"""
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = unicodedata.normalize('NFD', text)
#     text = ''.join([c for c in text if not unicodedata.combining(c)])
#     text = text.replace('đ', 'd')
#     return text

# class DomainKeywordBooster:
#     def __init__(self):
#         raw_domains = {
#             "education": {
#                 "general": ["giáo viên", "thầy giáo", "cô giáo", "bài giảng", "ôn thi", "thpt", "chấm điểm", "slide bài giảng"],
#                 "subdomains": {
#                     "literature": ["ngữ văn", "văn học", "nghị luận", "nhân vật", "tác phẩm", "mở bài", "thân bài", "kết bài"],
#                     "math": ["toán", "hình học", "đại số", "tích phân", "đạo hàm", "phương trình", "số học"],
#                     "english": ["tiếng anh", "english", "ngữ pháp", "từ vựng", "grammar", "động từ"],
#                     "science": ["vật lý", "vật lí", "hóa học", "sinh học"]
#                 }
#             },
#             "sports": {
#                 "general": ["thể thao", "vận động viên", "thi đấu", "giải đấu", "cúp", "huy chương"],
#                 "subdomains": {
#                     "cycling": ["đua xe đạp", "xe đạp", "tay đua", "chặng đua", "vạch đích", "áo vàng", "áo xanh"],
#                     "football": ["bóng đá", "cầu thủ", "bàn thắng", "thủ môn", "trọng tài", "sân cỏ"]
#                 }
#             },
#             "culinary": {
#                 "general": ["đầu bếp", "nấu ăn", "chế biến", "món ăn", "ẩm thực", "nguyên liệu", "gia vị", "hấp", "nướng", "chiên", "xào", "nước dùng"],
#                 "subdomains": {}
#             }
#         }
        
#         self.domains = {}
#         for dom_key, dom_val in raw_domains.items():
#             self.domains[dom_key] = {
#                 "general": [remove_vietnamese_diacritics(kw) for kw in dom_val.get("general", [])],
#                 "subdomains": {
#                     sub_k: [remove_vietnamese_diacritics(kw) for kw in sub_list]
#                     for sub_k, sub_list in dom_val.get("subdomains", {}).items()
#                 }
#             }

#     def _extract_domain_context(self, clean_query: str):
#         detected_domain = None
#         detected_subdomain = None
        
#         for dom_key, dom_data in self.domains.items():
#             for sub_key, sub_keywords in dom_data["subdomains"].items():
#                 if any(kw in clean_query for kw in sub_keywords):
#                     return dom_key, sub_key
#             if any(kw in clean_query for kw in dom_data["general"]):
#                 detected_domain = dom_key
                
#         return detected_domain, detected_subdomain

#     def compute_boost(self, query_text: str, cand_metadata: dict) -> float:
#         clean_query = remove_vietnamese_diacritics(query_text)
#         q_domain, q_subdomain = self._extract_domain_context(clean_query)
        
#         if not q_domain:
#             return 0.0
            
#         title = remove_vietnamese_diacritics(cand_metadata.get("title", ""))
#         desc = remove_vietnamese_diacritics(cand_metadata.get("description", ""))
#         ocr = remove_vietnamese_diacritics(cand_metadata.get("ocr_text", ""))
#         asr = remove_vietnamese_diacritics(cand_metadata.get("asr_text", ""))
#         cand_text = f"{title} {desc} {ocr} {asr}"
        
#         boost_score = 0.0
        
#         if q_subdomain:
#             sub_keywords = self.domains[q_domain]["subdomains"][q_subdomain]
#             matched_count = sum(1 for kw in sub_keywords if kw in cand_text)
            
#             if matched_count > 0:
#                 boost_score += 0.30
#             else:
#                 other_subdomains_kw = [
#                     kw for sub_k, kw_list in self.domains[q_domain]["subdomains"].items() 
#                     if sub_k != q_subdomain for kw in kw_list
#                 ]
#                 if any(kw in cand_text for kw in other_subdomains_kw):
#                     return -0.35
                    
#         gen_keywords = self.domains[q_domain]["general"]
#         if any(kw in cand_text for kw in gen_keywords):
#             boost_score += 0.10
            
#         other_domains = [d for d in self.domains.keys() if d != q_domain]
#         for od in other_domains:
#             od_data = self.domains[od]
#             all_od_keywords = od_data["general"] + [
#                 kw for sub_list in od_data["subdomains"].values() for kw in sub_list
#             ]
#             if any(kw in title or kw in ocr for kw in all_od_keywords):
#                 boost_score -= 0.30
#                 break

#         return max(-0.40, min(0.40, boost_score))








import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def remove_vietnamese_diacritics(text: str) -> str:
    """Lowercase + bỏ dấu tiếng Việt + chuẩn hóa khoảng trắng."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d")
    return " ".join(text.split())


class DomainKeywordBooster:
    """
    Booster chuyên cho MÔN HỌC.

    Mục tiêu:
    - Query đã chỉ ra môn/chủ đề học thuật thì candidate phải khớp đúng môn.
    - Cue chung như "giáo viên", "bài giảng", "ôn thi" chỉ là tín hiệu phụ.
    - Chỉ dùng metadata + ASR (không dùng OCR vì OCR dễ sai chính tả).

    Triết lý:
    - Nếu query có subject rõ: subdomain match là điều kiện chính.
    - Nếu không có subject rõ: chỉ dùng general education nhẹ.
    - Sai môn trong cùng nhóm education sẽ bị phạt mạnh.
    """

    def __init__(self):
        raw_domains = {
            "education": {
                "general": [
                    "giao vien", "thay giao", "co giao", "bai giang", "on thi",
                    "thpt", "cham diem", "chuong trinh", "tai lieu hoc tap",
                    "lop hoc", "giang bai", "hoc sinh", "mon hoc", "bai hoc",
                    "de cuong", "kiem tra", "on tap", "lop 12", "lop 11", "lop 10"
                ],
                "subdomains": {
                    "literature": [
                        "ngu van", "van hoc", "nghi luan van hoc", "nghi luan",
                        "mo bai", "than bai", "ket bai", "phan tich nhan vat",
                        "tac pham", "nhan vat", "phan tich tac pham", "bai van",
                        "tac pham van hoc", "nghi luan xa hoi", "nghi luan van hoc"
                    ],
                    "math": [
                        "toan", "dai so", "hinh hoc", "dao ham", "tich phan",
                        "phuong trinh", "bat phuong trinh", "so hoc", "ham so",
                        "vector", "toa do", "giai tich"
                    ],
                    "english": [
                        "tieng anh", "english", "ngu phap", "tu vung",
                        "grammar", "verb", "adjective", "noun", "reading",
                        "listening", "speaking", "writing"
                    ],
                    "science": [
                        "vat ly", "vat li", "hoa hoc", "sinh hoc",
                        "thi nghiem", "cong thuc", "thuc hanh", "chat", "nang luong",
                        "dien", "quang hoc"
                    ],
                    "civics": [
                        "giao duc cong dan", "cong dan", "phap luat",
                        "quyen", "nghia vu", "xa hoi", "phap ly"
                    ],
                    "history": [
                        "lich su", "su kien", "chien tranh", "trieu dai",
                        "niem dai", "cach mang", "khang chien"
                    ],
                    "geography": [
                        "dia ly", "ban do", "khi hau", "thien nhien",
                        "dan cu", "kinh te", "vung mien", "dia hinh"
                    ]
                },
            }
        }

        self.domains: Dict[str, Dict[str, Any]] = {}
        for dom, val in raw_domains.items():
            self.domains[dom] = {
                "general": [remove_vietnamese_diacritics(x) for x in val.get("general", [])],
                "subdomains": {
                    sk: [remove_vietnamese_diacritics(x) for x in lst]
                    for sk, lst in val.get("subdomains", {}).items()
                },
            }

    def _flatten_text(self, obj: Any) -> str:
        """
        Gom text từ nhiều kiểu metadata:
        - str
        - list[str]
        - list[dict(text=...)]
        - dict có keys: title, description, asr_text, keywords, text, caption
        """
        if obj is None:
            return ""

        if isinstance(obj, str):
            return obj

        if isinstance(obj, (int, float)):
            return str(obj)

        if isinstance(obj, dict):
            parts = []
            for k in ["title", "description", "asr_text", "keywords", "text", "caption"]:
                if k in obj and obj[k] is not None:
                    parts.append(self._flatten_text(obj[k]))
            return " ".join(p for p in parts if p)

        if isinstance(obj, (list, tuple)):
            parts = []
            for item in obj:
                if isinstance(item, dict) and "text" in item:
                    parts.append(self._flatten_text(item["text"]))
                else:
                    parts.append(self._flatten_text(item))
            return " ".join(p for p in parts if p)

        return str(obj)

    def _detect_domain(self, clean_query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect subject từ query.
        Ưu tiên subdomain nếu query có tín hiệu rõ.
        """
        found_domain = None
        found_subdomain = None

        for dom, dom_data in self.domains.items():
            for sub, kws in dom_data["subdomains"].items():
                if any(kw in clean_query for kw in kws):
                    return dom, sub

            if any(kw in clean_query for kw in dom_data["general"]):
                found_domain = dom

        return found_domain, found_subdomain

    def _count_hits(self, text: str, keywords: List[str]) -> int:
        return sum(1 for kw in keywords if kw in text)

    def compute_boost(self, query_text: str, cand_metadata: Dict[str, Any]) -> float:
        """
        Trả về boost trong [-0.80, +0.80]

        Quy tắc:
        - Query có subject rõ -> candidate phải khớp subject.
        - Title + ASR là tín hiệu chính.
        - Description chỉ phụ.
        - General education chỉ cộng nhẹ.
        - Sai môn trong cùng education bị phạt mạnh.
        """
        q = remove_vietnamese_diacritics(query_text)
        q_domain, q_subdomain = self._detect_domain(q)

        if not q_domain:
            return 0.0

        title = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("title", "")))
        asr = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("asr_text", "")))
        desc = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("description", "")))
        keywords = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("keywords", "")))

        # Title + ASR + keywords là lõi chính
        cand_text_main = " ".join([title, asr, keywords]).strip()
        cand_text_all = " ".join([title, asr, keywords, desc]).strip()

        score = 0.0

        # 1) General education: chỉ là tín hiệu phụ
        dom_general = self.domains[q_domain]["general"]
        dom_hits = self._count_hits(cand_text_all, dom_general)

        if dom_hits > 0:
            score += min(0.10, 0.03 * dom_hits)
        else:
            # query đã rõ học thuật mà candidate không có chút dấu hiệu giáo dục nào
            score -= 0.08

        # 2) Subject/subdomain: điều kiện chính
        if q_subdomain:
            sub_keywords = self.domains[q_domain]["subdomains"].get(q_subdomain, [])
            sub_hits = self._count_hits(cand_text_main, sub_keywords)

            if sub_hits > 0:
                # Cộng mạnh theo số keyword khớp, nhưng có chặn trần
                if sub_hits >= 1:
                    score += 0.22
                if sub_hits >= 3:
                    score += 0.18
                if sub_hits >= 5:
                    score += 0.12
            else:
                # Query đã chỉ rõ môn/chủ đề mà candidate không có dấu hiệu đúng môn
                # => phạt mạnh
                score -= 0.40

                # Nếu candidate lại có dấu hiệu của môn khác trong cùng education,
                # phạt thêm để tránh nhầm môn.
                other_sub_kws = [
                    kw
                    for sk, kws in self.domains[q_domain]["subdomains"].items()
                    if sk != q_subdomain
                    for kw in kws
                ]
                other_hits = self._count_hits(cand_text_main, other_sub_kws)
                if other_hits > 0:
                    score -= min(0.20, 0.05 * other_hits)

                # Nếu chỉ có cue chung giáo dục mà không có subject đúng, giảm nhẹ thêm
                if dom_hits > 0:
                    score -= 0.08

        else:
            # Không có subdomain rõ: chỉ dùng general domain nhẹ
            score += min(0.08, 0.02 * dom_hits)

        # 3) Phạt nếu title/ASR có dấu hiệu môn khác rất rõ
        for odom, od_data in self.domains.items():
            if odom == q_domain:
                continue

            other_kws = od_data["general"][:]
            for lst in od_data["subdomains"].values():
                other_kws.extend(lst)

            if any(kw in cand_text_main for kw in other_kws):
                score -= 0.08
                break

        # 4) Phạt rất nhẹ nếu description có dấu hiệu khác môn, nhưng không để lấn át chính
        if desc:
            desc_other_hits = 0
            for odom, od_data in self.domains.items():
                if odom == q_domain:
                    continue
                other_kws = od_data["general"][:]
                for lst in od_data["subdomains"].values():
                    other_kws.extend(lst)
                desc_other_hits += self._count_hits(desc, other_kws)
            if desc_other_hits > 0:
                score -= min(0.06, 0.02 * desc_other_hits)

        return max(-0.80, min(0.80, score))
