import os
from pathlib import Path
from typing import Generator
from fastapi import Request, HTTPException, status
from fastapi.responses import StreamingResponse

def range_requests_response(
    request: Request,
    file_path: Path,
    content_type: str = "video/mp4"
) -> StreamingResponse:
    """
    Trả về StreamingResponse hỗ trợ HTTP 206 Partial Content
    cho phép HTML5 video player tua (seek) tức thì đến bất kỳ giây nào.
    """
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file không tồn tại: {file_path.name}"
        )

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        # Xử lý Range header: bytes=start-end
        range_value = range_header.strip().lower().replace("bytes=", "")
        parts = range_value.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Dải byte yêu cầu không hợp lệ"
            )

        content_length = (end - start) + 1

        def iterfile() -> Generator[bytes, None, None]:
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = content_length
                chunk_size = 1024 * 1024  # 1MB chunk
                while bytes_left > 0:
                    read_size = min(chunk_size, bytes_left)
                    data = f.read(read_size)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            iterfile(),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers
        )
    else:
        def iterfile_full() -> Generator[bytes, None, None]:
            with open(file_path, "rb") as f:
                chunk_size = 1024 * 1024
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            iterfile_full(),
            status_code=status.HTTP_200_OK,
            headers=headers
        )
