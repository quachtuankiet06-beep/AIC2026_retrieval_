from typing import Optional, List, Dict, Any, Tuple
import faiss

# ==========================================================
# BỘ LỌC CHỦ ĐỀ VIDEO (TOPIC GROUPS FILTER)
# Các dải vector_index được xác định chính xác từ keyframes_b1_b2.db
# ==========================================================

TOPIC_GROUPS: Dict[str, Dict[str, Any]] = {
    "all": {
        "label": "Tất cả chủ đề (Không lọc)",
        "icon": "fa-globe",
        "prefixes": [],
        "ranges": [],
    },
    "thoi_su": {
        "label": "Thời sự 60 Giây (L21, L22, M)",
        "icon": "fa-newspaper",
        "prefixes": ["L21", "L22", "M"],
        "ranges": [(0, 65830), (358639, 475164)],
    },
    "dua_xe_dap": {
        "label": "Đua xe đạp theo chặng (L23, S)",
        "icon": "fa-person-biking",
        "prefixes": ["L23", "S"],
        "ranges": [(65830, 72220), (475164, 496777)],
    },
    "lan_su_rong": {
        "label": "Lân sư rồng (L24)",
        "icon": "fa-dragon",
        "prefixes": ["L24"],
        "ranges": [(72220, 85982)],
    },
    "on_thi_thpt": {
        "label": "Bài giảng ôn thi THPT (L25)",
        "icon": "fa-graduation-cap",
        "prefixes": ["L25"],
        "ranges": [(85982, 125529)],
    },
    "nau_an_vivu": {
        "label": "Nấu ăn ViVU TV (L26)",
        "icon": "fa-utensils",
        "prefixes": ["L26"],
        "ranges": [(125529, 292636)],
    },
    "du_lich_mientay": {
        "label": "Du lịch miền Tây (L27–L29)",
        "icon": "fa-mountain-sun",
        "prefixes": ["L27", "L28", "L29"],
        "ranges": [(292636, 341643)],
    },
    "lan_toa_tich_cuc": {
        "label": "Lan tỏa năng lượng tích cực (L30)",
        "icon": "fa-heart",
        "prefixes": ["L30"],
        "ranges": [(341643, 358639)],
    },
    "camera_giao_thong": {
        "label": "Camera giao thông (N)",
        "icon": "fa-video",
        "prefixes": ["N"],
        "ranges": [(496777, 540856)],
    },
}


def get_topic_selector(topic_key: Optional[str]) -> Optional[faiss.IDSelector]:
    """
    Tạo FAISS IDSelector tối ưu từ topic_key.
    Hỗ trợ IDSelectorRange (cho 1 dải) hoặc IDSelectorOr (cho 2 dải).
    Nếu topic_key rỗng hoặc 'all' -> trả về None (quét toàn bộ).
    """
    if not topic_key or topic_key == "all":
        return None

    group_info = TOPIC_GROUPS.get(topic_key)
    if not group_info:
        # Thử tìm theo nhãn nếu truyền label
        for k, v in TOPIC_GROUPS.items():
            if topic_key.lower() in [k.lower(), v["label"].lower()]:
                group_info = v
                break

    if not group_info or not group_info.get("ranges"):
        return None

    ranges = group_info["ranges"]
    if len(ranges) == 1:
        start_id, end_id = ranges[0]
        return faiss.IDSelectorRange(int(start_id), int(end_id))
    elif len(ranges) == 2:
        sel1 = faiss.IDSelectorRange(int(ranges[0][0]), int(ranges[0][1]))
        sel2 = faiss.IDSelectorRange(int(ranges[1][0]), int(ranges[1][1]))
        return faiss.IDSelectorOr(sel1, sel2)
    else:
        current_sel = faiss.IDSelectorRange(int(ranges[0][0]), int(ranges[0][1]))
        for r in ranges[1:]:
            next_sel = faiss.IDSelectorRange(int(r[0]), int(r[1]))
            current_sel = faiss.IDSelectorOr(current_sel, next_sel)
        return current_sel


def get_search_parameters(topic_key: Optional[str]) -> Optional[faiss.SearchParameters]:
    """Tạo FAISS SearchParameters có gắn IDSelector của topic_filter."""
    sel = get_topic_selector(topic_key)
    if sel is not None:
        return faiss.SearchParameters(sel=sel)
    return None


def get_topic_list() -> List[Dict[str, Any]]:
    """Trả về danh sách topic cho Frontend UI."""
    return [
        {
            "key": k,
            "label": v["label"],
            "icon": v["icon"],
            "prefixes": v["prefixes"],
        }
        for k, v in TOPIC_GROUPS.items()
    ]
