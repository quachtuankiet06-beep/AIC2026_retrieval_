import uvicorn
import os
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc được nhận diện
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("\n" + "=" * 60)
    print("🚀 ĐANG KHỞI CHẠY HỆ THỐNG FASTAPI VIDEO SEARCH - AIC 2026")
    print(f"🔗 Mở trình duyệt và truy cập: http://127.0.0.1:{port}")
    print(f"📖 Swagger API Docs: http://127.0.0.1:{port}/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        "fastapi_app.main:app",
        host=host,
        port=port,
        reload=True
    )
