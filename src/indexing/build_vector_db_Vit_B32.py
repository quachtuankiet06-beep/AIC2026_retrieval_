import os
import glob
import torch
import clip
import numpy as np
import faiss
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def build_clip_vit_b32_index(
    keyframes_dir="data/keyframes", 
    output_index_path="data/indexes/vit_b32.index",
    batch_size=64
):
    print("=" * 60)
    print("Starting Building CLIP ViT-B/32 FAISS Index (Sequential Keyframes xxx.jpg)...")
    print("=" * 60)
    
    # 1. Thiết bị chạy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Sử dụng thiết bị: {device}")

    # 2. Load model CLIP ViT-B/32 chính hãng OpenAI
    print("[Info] Đang load model CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Chiều embedding của ViT-B/32 là 512
    dimension = 512 

    # 3. Tìm kiếm toàn bộ các file ảnh có định dạng .jpg (hỗ trợ pattern xxx.jpg như 001.jpg, 002.jpg...)
    search_path = os.path.join(keyframes_dir, "**", "*.jpg")
    img_files = glob.glob(search_path, recursive=True)

    if not img_files:
        print(f"[Error] Không tìm thấy file ảnh .jpg nào trong đường dẫn: {keyframes_dir}")
        return

    # Sắp xếp danh sách file theo thứ tự bảng chữ cái/đường dẫn để đảm bảo tuần tự chính xác theo video và frame
    img_files = sorted(img_files)
    total_files = len(img_files)
    print(f"[Info] Tìm thấy tổng số {total_files} keyframes. Bắt đầu trích xuất...")

    # 4. Khởi tạo FAISS IndexFlatIP (Inner Product)
    index = faiss.IndexFlatIP(dimension)
    
    batch_images = []
    all_embeddings = []

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(img_files, desc="Encoding Keyframes")):
            try:
                # Mở và tiền xử lý ảnh theo chuẩn CLIP
                image = Image.open(file_path).convert("RGB")
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)

                # Khi đủ batch hoặc đến file cuối cùng thì đẩy qua GPU
                if len(batch_images) == batch_size or (i == total_files - 1):
                    tensor_stack = torch.stack(batch_images).to(device)
                    
                    # Trích xuất vector đặc trưng hình ảnh
                    features = model.encode_image(tensor_stack)
                    
                    # ----------------------------------------------------
                    # BẮT BUỘC CHUẨN HÓA L2 NORM = 1.0 TRƯỚC KHI ĐƯA VÀO FAISS
                    # ----------------------------------------------------
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                    
                    # Chuyển đổi sang numpy float32
                    features_np = features.cpu().numpy().astype(np.float32)
                    all_embeddings.append(features_np)
                    
                    batch_images = []
            except Exception as e:
                print(f"\n[Warning] Lỗi khi xử lý file {file_path}: {e}")

    if not all_embeddings:
        print("[Error] Không thu được vector embedding nào hợp lệ.")
        return

    # Ghép toàn bộ các batch thành ma trận tổng (N, 512)
    global_matrix = np.vstack(all_embeddings)
    N, dim = global_matrix.shape
    
    print(f"\n[Info] Hoàn tất encode! Kích thước ma trận tổng: {N} x {dim}")

    # Kiểm tra nhanh mẫu norm của vector đầu tiên để chắc chắn đã chuẩn hóa = 1.0
    sample_norm = np.linalg.norm(global_matrix[0])
    print(f"[Check] Norm của vector đầu tiên sau chuẩn hóa: {sample_norm:.4f} (Lý tưởng = 1.0)")

    # 5. Thêm dữ liệu vào FAISS Index
    print(f"[Info] Đang thêm {N} vector vào FAISS IndexFlatIP...")
    index.add(global_matrix)
    print(f"[Info] Đã thêm thành công {index.ntotal} vector.")

    # 6. Lưu file index xuống đĩa
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    
    print("=" * 60)
    print(f"[Success] Đã lưu file index thành công tại: {output_index_path}")
    print("=" * 60)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # Đường dẫn trỏ tới thư mục chứa keyframes của bạn
    KEYFRAMES_DIR = BASE_DIR / "data" / "keyframes"
    OUTPUT_INDEX = BASE_DIR / "data" / "indexes" / "vit_b32.index"
    
    build_clip_vit_b32_index(
        keyframes_dir=str(KEYFRAMES_DIR), 
        output_index_path=str(OUTPUT_INDEX),
        batch_size=64
    )