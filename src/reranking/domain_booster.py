



# import unicodedata
# from typing import Any, Dict, List, Optional, Tuple


# def remove_vietnamese_diacritics(text: str) -> str:
#     """Lowercase + bỏ dấu tiếng Việt + chuẩn hóa khoảng trắng."""
#     if not isinstance(text, str):
#         return ""
#     text = text.lower().strip()
#     text = unicodedata.normalize("NFD", text)
#     text = "".join(c for c in text if not unicodedata.combining(c))
#     text = text.replace("đ", "d")
#     return " ".join(text.split())


# class DomainKeywordBooster:
#     """
#     Booster chuyên cho MÔN HỌC.

#     Mục tiêu:
#     - Query đã chỉ ra môn/chủ đề học thuật thì candidate phải khớp đúng môn.
#     - Cue chung như "giáo viên", "bài giảng", "ôn thi" chỉ là tín hiệu phụ.
#     - Chỉ dùng metadata + ASR (không dùng OCR vì OCR dễ sai chính tả).

#     Triết lý:
#     - Nếu query có subject rõ: subdomain match là điều kiện chính.
#     - Nếu không có subject rõ: chỉ dùng general education nhẹ.
#     - Sai môn trong cùng nhóm education sẽ bị phạt mạnh.
#     """

#     def __init__(self):
#         raw_domains = {
#             "education": {
#                 "general": [
#                     "giao vien", "thay giao", "co giao", "bai giang", "on thi",
#                     "thpt", "cham diem", "chuong trinh", "tai lieu hoc tap",
#                     "lop hoc", "giang bai", "hoc sinh", "mon hoc", "bai hoc",
#                     "de cuong", "kiem tra", "on tap", "lop 12", "lop 11", "lop 10"
#                 ],
#                 "subdomains": {
#                     "literature": [
#                         "ngu van", "van hoc", "nghi luan van hoc", "nghi luan",
#                         "mo bai", "than bai", "ket bai", "phan tich nhan vat",
#                         "tac pham", "nhan vat", "phan tich tac pham", "bai van",
#                         "tac pham van hoc", "nghi luan xa hoi", "nghi luan van hoc"
#                     ],
#                     "math": [
#                         "toan", "dai so", "hinh hoc", "dao ham", "tich phan",
#                         "phuong trinh", "bat phuong trinh", "so hoc", "ham so",
#                         "vector", "toa do", "giai tich"
#                     ],
#                     "english": [
#                         "tieng anh", "english", "ngu phap", "tu vung",
#                         "grammar", "verb", "adjective", "noun", "reading",
#                         "listening", "speaking", "writing"
#                     ],
#                     "science": [
#                         "vat ly", "vat li", "hoa hoc", "sinh hoc",
#                         "thi nghiem", "cong thuc", "thuc hanh", "chat", "nang luong",
#                         "dien", "quang hoc"
#                     ],
#                     "civics": [
#                         "giao duc cong dan", "cong dan", "phap luat",
#                         "quyen", "nghia vu", "xa hoi", "phap ly"
#                     ],
#                     "history": [
#                         "lich su", "su kien", "chien tranh", "trieu dai",
#                         "niem dai", "cach mang", "khang chien"
#                     ],
#                     "geography": [
#                         "dia ly", "ban do", "khi hau", "thien nhien",
#                         "dan cu", "kinh te", "vung mien", "dia hinh"
#                     ]
#                 },
#             }
#         }

#         self.domains: Dict[str, Dict[str, Any]] = {}
#         for dom, val in raw_domains.items():
#             self.domains[dom] = {
#                 "general": [remove_vietnamese_diacritics(x) for x in val.get("general", [])],
#                 "subdomains": {
#                     sk: [remove_vietnamese_diacritics(x) for x in lst]
#                     for sk, lst in val.get("subdomains", {}).items()
#                 },
#             }

#     def _flatten_text(self, obj: Any) -> str:
#         """
#         Gom text từ nhiều kiểu metadata:
#         - str
#         - list[str]
#         - list[dict(text=...)]
#         - dict có keys: title, description, asr_text, keywords, text, caption
#         """
#         if obj is None:
#             return ""

#         if isinstance(obj, str):
#             return obj

#         if isinstance(obj, (int, float)):
#             return str(obj)

#         if isinstance(obj, dict):
#             parts = []
#             for k in ["title", "description", "asr_text", "keywords", "text", "caption"]:
#                 if k in obj and obj[k] is not None:
#                     parts.append(self._flatten_text(obj[k]))
#             return " ".join(p for p in parts if p)

#         if isinstance(obj, (list, tuple)):
#             parts = []
#             for item in obj:
#                 if isinstance(item, dict) and "text" in item:
#                     parts.append(self._flatten_text(item["text"]))
#                 else:
#                     parts.append(self._flatten_text(item))
#             return " ".join(p for p in parts if p)

#         return str(obj)

#     def _detect_domain(self, clean_query: str) -> Tuple[Optional[str], Optional[str]]:
#         """
#         Detect subject từ query.
#         Ưu tiên subdomain nếu query có tín hiệu rõ.
#         """
#         found_domain = None
#         found_subdomain = None

#         for dom, dom_data in self.domains.items():
#             for sub, kws in dom_data["subdomains"].items():
#                 if any(kw in clean_query for kw in kws):
#                     return dom, sub

#             if any(kw in clean_query for kw in dom_data["general"]):
#                 found_domain = dom

#         return found_domain, found_subdomain

#     def _count_hits(self, text: str, keywords: List[str]) -> int:
#         return sum(1 for kw in keywords if kw in text)

#     def compute_boost(self, query_text: str, cand_metadata: Dict[str, Any]) -> float:
#         """
#         Trả về boost trong [-0.80, +0.80]

#         Quy tắc:
#         - Query có subject rõ -> candidate phải khớp subject.
#         - Title + ASR là tín hiệu chính.
#         - Description chỉ phụ.
#         - General education chỉ cộng nhẹ.
#         - Sai môn trong cùng education bị phạt mạnh.
#         """
#         q = remove_vietnamese_diacritics(query_text)
#         q_domain, q_subdomain = self._detect_domain(q)

#         if not q_domain:
#             return 0.0

#         title = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("title", "")))
#         asr = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("asr_text", "")))
#         desc = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("description", "")))
#         keywords = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("keywords", "")))

#         # Title + ASR + keywords là lõi chính
#         cand_text_main = " ".join([title, asr, keywords]).strip()
#         cand_text_all = " ".join([title, asr, keywords, desc]).strip()

#         score = 0.0

#         # 1) General education: chỉ là tín hiệu phụ
#         dom_general = self.domains[q_domain]["general"]
#         dom_hits = self._count_hits(cand_text_all, dom_general)

#         if dom_hits > 0:
#             score += min(0.10, 0.03 * dom_hits)
#         else:
#             # query đã rõ học thuật mà candidate không có chút dấu hiệu giáo dục nào
#             score -= 0.08

#         # 2) Subject/subdomain: điều kiện chính
#         if q_subdomain:
#             sub_keywords = self.domains[q_domain]["subdomains"].get(q_subdomain, [])
#             sub_hits = self._count_hits(cand_text_main, sub_keywords)

#             if sub_hits > 0:
#                 # Cộng mạnh theo số keyword khớp, nhưng có chặn trần
#                 if sub_hits >= 1:
#                     score += 0.22
#                 if sub_hits >= 3:
#                     score += 0.18
#                 if sub_hits >= 5:
#                     score += 0.12
#             else:
#                 # Query đã chỉ rõ môn/chủ đề mà candidate không có dấu hiệu đúng môn
#                 # => phạt mạnh
#                 score -= 0.40

#                 # Nếu candidate lại có dấu hiệu của môn khác trong cùng education,
#                 # phạt thêm để tránh nhầm môn.
#                 other_sub_kws = [
#                     kw
#                     for sk, kws in self.domains[q_domain]["subdomains"].items()
#                     if sk != q_subdomain
#                     for kw in kws
#                 ]
#                 other_hits = self._count_hits(cand_text_main, other_sub_kws)
#                 if other_hits > 0:
#                     score -= min(0.20, 0.05 * other_hits)

#                 # Nếu chỉ có cue chung giáo dục mà không có subject đúng, giảm nhẹ thêm
#                 if dom_hits > 0:
#                     score -= 0.08

#         else:
#             # Không có subdomain rõ: chỉ dùng general domain nhẹ
#             score += min(0.08, 0.02 * dom_hits)

#         # 3) Phạt nếu title/ASR có dấu hiệu môn khác rất rõ
#         for odom, od_data in self.domains.items():
#             if odom == q_domain:
#                 continue

#             other_kws = od_data["general"][:]
#             for lst in od_data["subdomains"].values():
#                 other_kws.extend(lst)

#             if any(kw in cand_text_main for kw in other_kws):
#                 score -= 0.08
#                 break

#         # 4) Phạt rất nhẹ nếu description có dấu hiệu khác môn, nhưng không để lấn át chính
#         if desc:
#             desc_other_hits = 0
#             for odom, od_data in self.domains.items():
#                 if odom == q_domain:
#                     continue
#                 other_kws = od_data["general"][:]
#                 for lst in od_data["subdomains"].values():
#                     other_kws.extend(lst)
#                 desc_other_hits += self._count_hits(desc, other_kws)
#             if desc_other_hits > 0:
#                 score -= min(0.06, 0.02 * desc_other_hits)

#         return max(-0.80, min(0.80, score))










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
    Booster chuyên cho MÔN HỌC / chủ đề nội dung.

    Mục tiêu:
    - Query có subject rõ thì candidate phải khớp đúng subject.
    - Cue chung như "giáo viên", "bài giảng", "nấu ăn" chỉ là tín hiệu phụ.
    - Chỉ dùng title + description + keywords + ASR (không dùng OCR vì OCR dễ sai chính tả).

    Mở rộng thêm:
    - art/crafts: cho các query về trình diễn thời trang, đồ thủ công, patchwork,
      trưng bày ngoài trời, textile art...
    - cooking-process / recipe steps: cho các query mô tả quy trình nấu ăn, sơ chế,
      cho nguyên liệu, đảo/chiên/đun, định lượng gia vị.
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
                        "tac pham van hoc", "nghi luan xa hoi"
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
                        "cach mang", "khang chien"
                    ],
                    "geography": [
                        "dia ly", "ban do", "khi hau", "thien nhien",
                        "dan cu", "kinh te", "vung mien", "dia hinh"
                    ],
                },
            },

            # NEW: art / crafts
            "art_crafts": {
                "general": [
                    "art", "arts", "craft", "crafts", "handmade", "handcrafted",
                    "arts and crafts", "exhibition", "installation", "outdoor exhibition",
                    "textile art", "fabric art", "patchwork", "sewing", "embroidery",
                    "fashion show", "runway", "model", "garment", "designer",
                    "do thu cong", "thu cong", "nghe thuat", "trien lam", "trung bay",
                    "trinh dien", "trang phuc", "vai", "hoa van", "hoa tiet", "hinh hoc",
                    "bup be", "qua cau", "tho cam", "det", "may", "theu"
                ],
                "subdomains": {
                    "textile_art": [
                        "textile", "textile art", "fabric", "fabric art", "patchwork",
                        "cloth pieces", "colorful fabric", "patterned fabric", "geometric pattern",
                        "woven", "woven fabric", "embroidery", "stitch", "thread", "tho cam",
                        "vai mau", "manh vai", "hoa tiet hinh hoc", "hoa van"
                    ],
                    "fashion_show": [
                        "fashion show", "runway", "model", "models", "outfit", "garment",
                        "wide silhouette", "cream colored", "collection", "designer collection",
                        "trinh dien", "nguoi mau", "trang phuc", "bo trang phuc", "dang rong"
                    ],
                    "handmade_objects": [
                        "handmade", "craft", "doll", "toy", "ornament", "ball",
                        "fabric doll", "craft doll", "handmade toy", "installation",
                        "do thu cong", "bup be", "qua cau", "vat lieu", "san co"
                    ],
                },
            },

            "culinary": {
                "general": [
                    "dau bep", "nau an", "che bien", "mon an", "am thuc", "nguyen lieu",
                    "gia vi", "hap", "nuong", "chien", "xao", "nuoc dung", "cong thuc",
                    "recipe", "cooking", "cook", "kitchen", "dish", "food"
                ],
                "subdomains": {
                    "cooking_process": [
                        "so che", "rinse", "wash", "drain", "de rao",
                        "pour", "add", "mix", "stir", "stir fry", "boil", "simmer", "fry",
                        "saute", "steam", "bake", "knead", "rest", "set aside", "prepare",
                        "season", "marinate", "heat oil", "preheat", "cook", "process",
                        "recipe steps", "step", "steps", "nguyen lieu",
                        "cho lan luot", "dao", "dao deu",
                        "thot", "noi thuy tinh", "chao", "chen", "muong", "muoi", "lit",
                        "gram", "ml", "kg", "1/2", "1.5l", "1,5l", "muong ca phe"
                    ],
                    "food_prep": [
                        "prepare", "preparation", "prepped", "ingredient", "ingredients",
                        "slice", "chop", "mince", "peel", "cut", "clean", "wash",
                        "blend", "grind", "marinate", "season"
                    ]
                }
            }
        }
        self.domains: Dict[str, Dict[str, Any]] = {}
        for dom_key, dom_val in raw_domains.items():
            self.domains[dom_key] = {
                "general": [remove_vietnamese_diacritics(kw) for kw in dom_val.get("general", [])],
                "subdomains": {
                    sub_k: [remove_vietnamese_diacritics(kw) for kw in sub_list]
                    for sub_k, sub_list in dom_val.get("subdomains", {}).items()
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
        Detect domain / subdomain từ query.
        Ưu tiên subdomain nếu query có tín hiệu rõ.
        """
        found_domain = None
        found_subdomain = None

        for dom_key, dom_data in self.domains.items():
            for sub_key, kws in dom_data["subdomains"].items():
                if any(kw in clean_query for kw in kws):
                    return dom_key, sub_key

            if any(kw in clean_query for kw in dom_data["general"]):
                found_domain = dom_key

        return found_domain, found_subdomain

    def _count_hits(self, text: str, keywords: List[str]) -> int:
        return sum(1 for kw in keywords if kw in text)

    
    # def compute_boost(self, query_text: str, cand_metadata: Dict[str, Any]) -> float:
    #     """
    #     Trả về boost trong [-0.80, +0.80]

    #     Triết lý:
    #     - Match rõ thì cộng rõ.
    #     - Xung đột rõ thì trừ rõ.
    #     - Mơ hồ / metadata nghèo thì gần trung tính.
    #     - Không coi thiếu keyword là sai ngay.
    #     """
    #     q = remove_vietnamese_diacritics(query_text)
    #     q_domain, q_subdomain = self._detect_domain(q)

    #     if not q_domain:
    #         return 0.0

    #     title = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("title", "")))
    #     asr = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("asr_text", "")))
    #     desc = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("description", "")))
    #     keywords = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("keywords", "")))

    #     # Title + ASR + keywords là lõi chính
    #     cand_text_main = " ".join([title, asr, keywords]).strip()
    #     cand_text_all = " ".join([title, asr, keywords, desc]).strip()

    #     score = 0.0

    #     # --------------------------------------------------
    #     # 1) General domain cues: chỉ cộng rất nhẹ
    #     # --------------------------------------------------
    #     dom_general = self.domains[q_domain]["general"]
    #     dom_hits = self._count_hits(cand_text_all, dom_general)

    #     if dom_hits >= 3:
    #         score += min(0.10, 0.03 * dom_hits)
    #     elif dom_hits == 2:
    #         score += 0.03
    #     elif dom_hits == 1:
    #         score += 0.01
    #     # dom_hits == 0: trung tính

    #     # --------------------------------------------------
    #     # 2) Subject / subdomain
    #     #    - chỉ boost mạnh khi có dấu hiệu rõ
    #     #    - không phạt mạnh chỉ vì thiếu keyword
    #     # --------------------------------------------------
    #     if q_subdomain:
    #         sub_keywords = self.domains[q_domain]["subdomains"].get(q_subdomain, [])
    #         sub_hits = self._count_hits(cand_text_main, sub_keywords)

    #         other_sub_kws = [
    #             kw
    #             for sk, kws in self.domains[q_domain]["subdomains"].items()
    #             if sk != q_subdomain
    #             for kw in kws
    #         ]
    #         other_hits = self._count_hits(cand_text_main, other_sub_kws)

    #         if sub_hits >= 3:
    #             # tín hiệu rõ
    #             score += 0.22
    #             if sub_hits >= 5:
    #                 score += 0.10
    #             if sub_hits >= 7:
    #                 score += 0.06

    #         elif sub_hits == 2:
    #             # tín hiệu khá rõ nhưng chưa thật mạnh
    #             score += 0.12

    #         elif sub_hits == 1:
    #             # tín hiệu yếu, chỉ cộng nhẹ
    #             score += 0.05

    #         else:
    #             # Không có dấu hiệu đúng subdomain:
    #             # chỉ phạt khi có dấu hiệu lệch đủ rõ
    #             if other_hits >= 4:
    #                 score -= 0.20
    #             elif other_hits >= 2:
    #                 score -= 0.10
    #             elif other_hits == 1:
    #                 score -= 0.03
    #             # nếu metadata quá nghèo thì gần như trung tính

    #     else:
    #         # Không có subdomain rõ: chỉ dùng general rất nhẹ
    #         if dom_hits >= 3:
    #             score += min(0.08, 0.02 * dom_hits)
    #         elif dom_hits == 2:
    #             score += 0.03
    #         elif dom_hits == 1:
    #             score += 0.01

    #     # --------------------------------------------------
    #     # 3) Phạt nếu title / ASR có dấu hiệu domain khác rất rõ
    #     #    Chỉ phạt khi dấu hiệu đối nghịch khá rõ, không phạt vì 1 keyword lẻ.
    #     # --------------------------------------------------
    #     for odom, od_data in self.domains.items():
    #         if odom == q_domain:
    #             continue

    #         other_kws = od_data["general"][:]
    #         for lst in od_data["subdomains"].values():
    #             other_kws.extend(lst)

    #         other_main_hits = self._count_hits(cand_text_main, other_kws)

    #         if other_main_hits >= 4:
    #             score -= 0.08
    #             break
    #         elif other_main_hits >= 2:
    #             score -= 0.05
    #             break
    #         elif other_main_hits == 1:
    #             score -= 0.02
    #             break

    #     # --------------------------------------------------
    #     # 4) Phạt rất nhẹ nếu description có dấu hiệu khác domain
    #     #    Description chỉ là tín hiệu phụ nên phạt cực nhẹ.
    #     # --------------------------------------------------
    #     if desc:
    #         desc_other_hits = 0
    #         for odom, od_data in self.domains.items():
    #             if odom == q_domain:
    #                 continue
    #             other_kws = od_data["general"][:]
    #             for lst in od_data["subdomains"].values():
    #                 other_kws.extend(lst)
    #             desc_other_hits += self._count_hits(desc, other_kws)

    #         if desc_other_hits >= 4:
    #             score -= min(0.03, 0.01 * desc_other_hits)
    #         elif desc_other_hits == 2 or desc_other_hits == 3:
    #             score -= 0.01

    #     return max(-0.80, min(0.80, score))
    def compute_boost(self, query_text: str, cand_metadata: Dict[str, Any]) -> float:
        """
        Trả về boost trong [-0.80, +0.80]

        Rule:
        - education: scoring chặt
        * general chỉ là tín hiệu phụ
        * subdomain match phải đủ rõ mới boost mạnh
        * sai môn/subdomain phạt rõ hơn
        - culinary/cooking_process: safe mode mềm hơn
        * thiếu keyword không bị phạt mạnh
        * chỉ phạt khi lệch domain khác khá rõ
        * ưu tiên không làm tụt candidate đúng visual nhưng metadata nghèo
        - các domain khác: dùng mode mặc định, nằm giữa strict và safe
        """
        q = remove_vietnamese_diacritics(query_text)
        q_domain, q_subdomain = self._detect_domain(q)

        if not q_domain:
            return 0.0

        title = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("title", "")))
        asr = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("asr_text", "")))
        desc = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("description", "")))
        keywords = remove_vietnamese_diacritics(self._flatten_text(cand_metadata.get("keywords", "")))

        cand_text_main = " ".join([title, asr, keywords]).strip()
        cand_text_all = " ".join([title, asr, keywords, desc]).strip()

        score = 0.0

        # Mode selection
        if q_domain == "education":
            mode = "strict"
        elif q_domain == "culinary" and q_subdomain == "cooking_process":
            mode = "safe"
        else:
            mode = "default"

        # --------------------------------------------------
        # 1) General domain cues
        # --------------------------------------------------
        dom_general = self.domains[q_domain]["general"]
        dom_hits = self._count_hits(cand_text_all, dom_general)

        if mode == "strict":
            # education: chỉ cộng khi tín hiệu đủ rõ
            if dom_hits >= 3:
                score += min(0.10, 0.03 * dom_hits)
            elif dom_hits == 2:
                score += 0.03
            elif dom_hits == 1:
                score += 0.01
            else:
                score -= 0.04  # nhẹ hơn bản cũ
        elif mode == "safe":
            # culinary/cooking_process: mềm hơn
            if dom_hits >= 3:
                score += min(0.08, 0.02 * dom_hits)
            elif dom_hits == 2:
                score += 0.02
            elif dom_hits == 1:
                score += 0.00
            # dom_hits == 0: trung tính
        else:
            # default mode: giữa strict và safe
            if dom_hits >= 3:
                score += min(0.08, 0.025 * dom_hits)
            elif dom_hits == 2:
                score += 0.02
            elif dom_hits == 1:
                score += 0.01

        # --------------------------------------------------
        # 2) Subject / subdomain
        # --------------------------------------------------
        if q_subdomain:
            sub_keywords = self.domains[q_domain]["subdomains"].get(q_subdomain, [])
            sub_hits = self._count_hits(cand_text_main, sub_keywords)

            other_sub_kws = [
                kw
                for sk, kws in self.domains[q_domain]["subdomains"].items()
                if sk != q_subdomain
                for kw in kws
            ]
            other_hits = self._count_hits(cand_text_main, other_sub_kws)

            if mode == "strict":
                # education: phải khớp đủ rõ mới được cộng mạnh
                if sub_hits >= 3:
                    score += 0.22
                    if sub_hits >= 5:
                        score += 0.10
                    if sub_hits >= 7:
                        score += 0.06
                elif sub_hits == 2:
                    score += 0.12
                elif sub_hits == 1:
                    score += 0.05
                else:
                    # chỉ phạt khi có lệch rõ
                    if other_hits >= 4:
                        score -= 0.20
                    elif other_hits >= 2:
                        score -= 0.10
                    elif other_hits == 1:
                        score -= 0.03

                    # nếu có cue general nhưng subdomain vẫn sai -> phạt nhẹ thêm
                    if dom_hits >= 2:
                        score -= 0.03

            elif mode == "safe":
                # culinary/cooking_process: mềm hơn, tránh tụt query process-heavy
                if sub_hits >= 3:
                    score += 0.16
                    if sub_hits >= 5:
                        score += 0.06
                elif sub_hits == 2:
                    score += 0.08
                elif sub_hits == 1:
                    score += 0.03
                else:
                    # metadata nghèo thì gần trung tính
                    if other_hits >= 5:
                        score -= 0.03
                    elif other_hits >= 3:
                        score -= 0.01
                    # elif other_hits == 2:
                    #     score -= 0.03
                    # other_hits 0 or 1: không phạt mạnh

            else:
                # default mode
                if sub_hits >= 3:
                    score += 0.20
                    if sub_hits >= 5:
                        score += 0.08
                elif sub_hits == 2:
                    score += 0.10
                elif sub_hits == 1:
                    score += 0.04
                else:
                    if other_hits >= 4:
                        score -= 0.15
                    elif other_hits >= 2:
                        score -= 0.08
                    elif other_hits == 1:
                        score -= 0.02

        else:
            # Không có subdomain rõ:
            # education vẫn chặt hơn, culinary safe mode vẫn mềm
            if mode == "strict":
                if dom_hits >= 3:
                    score += min(0.08, 0.02 * dom_hits)
                elif dom_hits == 2:
                    score += 0.03
                elif dom_hits == 1:
                    score += 0.01
            elif mode == "safe":
                if dom_hits >= 3:
                    score += min(0.06, 0.015 * dom_hits)
                elif dom_hits == 2:
                    score += 0.02
                elif dom_hits == 1:
                    score += 0.00
            else:
                if dom_hits >= 3:
                    score += min(0.07, 0.02 * dom_hits)
                elif dom_hits == 2:
                    score += 0.02
                elif dom_hits == 1:
                    score += 0.01

        # --------------------------------------------------
        # 3) Phạt nếu title / ASR có dấu hiệu domain khác
        # --------------------------------------------------
        for odom, od_data in self.domains.items():
            if odom == q_domain:
                continue

            other_kws = od_data["general"][:]
            for lst in od_data["subdomains"].values():
                other_kws.extend(lst)

            other_main_hits = self._count_hits(cand_text_main, other_kws)

            if mode == "strict":
                if other_main_hits >= 4:
                    score -= 0.08
                    break
                elif other_main_hits >= 2:
                    score -= 0.05
                    break
                elif other_main_hits == 1:
                    score -= 0.02
                    break

            elif mode == "safe":
                # safe mode: chỉ phạt khi lệch khá rõ
                if other_main_hits >= 5:
                    score -= 0.05
                    break
                elif other_main_hits >= 3:
                    score -= 0.03
                    break
                elif other_main_hits == 2:
                    score -= 0.01
                    break

            else:
                if other_main_hits >= 4:
                    score -= 0.06
                    break
                elif other_main_hits >= 2:
                    score -= 0.03
                    break
                elif other_main_hits == 1:
                    score -= 0.01
                    break

        # --------------------------------------------------
        # 4) Phạt nếu description có dấu hiệu khác domain
        # --------------------------------------------------
        if desc:
            desc_other_hits = 0
            for odom, od_data in self.domains.items():
                if odom == q_domain:
                    continue
                other_kws = od_data["general"][:]
                for lst in od_data["subdomains"].values():
                    other_kws.extend(lst)
                desc_other_hits += self._count_hits(desc, other_kws)

            if mode == "strict":
                if desc_other_hits >= 4:
                    score -= min(0.03, 0.01 * desc_other_hits)
                elif desc_other_hits == 2 or desc_other_hits == 3:
                    score -= 0.01
            elif mode == "safe":
                # description chỉ là tín hiệu phụ rất yếu
                if desc_other_hits >= 5:
                    score -= 0.02
                elif desc_other_hits >= 3:
                    score -= 0.01
            else:
                if desc_other_hits >= 4:
                    score -= min(0.025, 0.008 * desc_other_hits)
                elif desc_other_hits == 2 or desc_other_hits == 3:
                    score -= 0.005

        return max(-0.80, min(0.80, score))