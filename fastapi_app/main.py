import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi_app.services import (
    execute_search,
    convert_time_to_frame,
    get_keyframe_path,
    get_video_path,
    get_video_fps,
    format_pts_to_hms
)
from fastapi_app.video_stream import range_requests_response

# Khởi tạo FastAPI App
app = FastAPI(
    title="AIC 2026 - Video Search & Frame Index Submission System",
    description="Hệ thống truy xuất video đa mô hình, reranking đa tầng, và bắt current_time sinh frame_idx nộp bài",
    version="1.0.0"
)

# Kích hoạt CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thiết lập thư mục tĩnh và templates
BASE_APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_APP_DIR / "static"
TEMPLATES_DIR = BASE_APP_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="Nội dung câu truy vấn tìm kiếm")
    search_type: str = Field("standard", description="'standard' hoặc 'temporal'")
    retrieval_mode: str = Field("multi", description="'multi' (Ensemble) hoặc 'single'")
    single_model_choice: str = Field("siglip2", description="'siglip2' hoặc 'dfn5b_vit_h14'")
    use_siglip2: bool = Field(True, description="Sử dụng SigLIP 2")
    use_dfn5b: bool = Field(True, description="Sử dụng DFN5B ViT-H/14")
    top_k: int = Field(20, ge=1, le=100, description="Số lượng kết quả hiển thị")
    max_kf_gap: int = Field(150, ge=1, le=500, description="Khoảng cách keyframe tối đa")
    min_kf_gap: int = Field(0, ge=0, le=50, description="Khoảng cách keyframe tối thiểu")
    beam_width: int = Field(5, ge=1, le=100, description="Beam Width cho Temporal Search")
    topic_filter: str = Field("all", description="Bộ lọc nhóm chủ đề video, vd: 'thoi_su', 'lan_su_rong', ...")


class ConvertTimeRequest(BaseModel):
    converter_input: str = Field(..., description="Chuỗi dạng 'video_id / time_str', vd: 'L22_V001 / 2:16'")


class CalculateFrameRequest(BaseModel):
    video_id: str = Field(..., description="Mã Video ID, vd: 'L22_V001'")
    current_time: float = Field(..., ge=0.0, description="Mốc thời gian hiện tại (giây)")


# -------------------------------------------------------------
# ROUTERS / ENDPOINTS
# -------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

import time

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render trang chủ giao diện tìm kiếm video"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"version": int(time.time())}
    )


@app.get("/api/topics")
async def api_get_topics():
    """Trả về danh sách 8 nhóm chủ đề video hỗ trợ lọc truy vấn."""
    from src.retrieval.topic_filter import get_topic_list
    return JSONResponse(status_code=status.HTTP_200_OK, content={"topics": get_topic_list()})


@app.post("/api/search")
async def api_search(req: SearchRequest):
    """
    Endpoint thực hiện tìm kiếm video (Standard Retrieval hoặc Temporal Sequence)
    kèm mapping và reranking đa tầng.
    """
    try:
        data = execute_search(
            query=req.query,
            search_type=req.search_type,
            retrieval_mode=req.retrieval_mode,
            single_model_choice=req.single_model_choice,
            use_siglip2=req.use_siglip2,
            use_dfn5b=req.use_dfn5b,
            top_k=req.top_k,
            max_kf_gap=req.max_kf_gap,
            min_kf_gap=req.min_kf_gap,
            beam_width=req.beam_width,
            topic_filter=req.topic_filter,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=data)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        import traceback
        print(f"[ERROR] /api/search failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi thực hiện pipeline tìm kiếm: {str(e)}"
        )


@app.get("/api/keyframe/{video_id}/{img_name}")
async def api_get_keyframe(video_id: str, img_name: str):
    """Stream ảnh keyframe JPEG từ thư mục data/keyframes/"""
    img_path = get_keyframe_path(video_id, img_name)
    if not img_path or not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file ảnh keyframe: {video_id}/{img_name}"
        )
    return FileResponse(
        img_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=604800, immutable",
        },
    )


@app.get("/api/video/{video_id}")
async def api_get_video(video_id: str, request: Request):
    """
    Stream video gốc từ C:\\Users\\Public\\Documents
    Hỗ trợ HTTP Range Requests để tua tức thì tới bất kỳ thời điểm nào.
    """
    video_path = get_video_path(video_id)
    if not video_path or not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy video '{video_id}' tại thư mục Public Documents"
        )
    return range_requests_response(request, video_path)


@app.post("/api/convert_time")
async def api_convert_time(req: ConvertTimeRequest):
    """Công cụ chuyển đổi nhanh chuỗi Video/Time -> Frame Index"""
    try:
        res = convert_time_to_frame(req.converter_input)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/fps/{video_id}")
async def api_get_fps(video_id: str):
    """Lấy FPS của một video cụ thể"""
    fps = get_video_fps(video_id)
    return {"video_id": video_id, "fps": fps}


@app.post("/api/calculate_frame")
async def api_calculate_frame(req: CalculateFrameRequest):
    """
    Module tính toán frame_idx tức thời khi người dùng dừng video tại current_time
    và trả về chuỗi nộp bài chuẩn video_id / frame_idx
    """
    fps = get_video_fps(req.video_id)
    frame_idx = int(round(req.current_time * fps))
    hms = format_pts_to_hms(req.current_time)
    
    return {
        "video_id": req.video_id,
        "current_time": req.current_time,
        "formatted_time": hms,
        "fps": fps,
        "frame_idx": frame_idx,
        "submission_string": f"{req.video_id} / {frame_idx}"
    }
