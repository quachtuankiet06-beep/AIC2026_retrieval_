import os
import glob
import torch
import open_clip
import numpy as np
import faiss
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def build_dfn5b_clip_vit_h14_index(
    keyframes_dir="data/keyframes", 
    output_index_path="data/indexes/dfn5b_clip_vit_h14.index",
    batch_size=16  # Giảm batch size xuống 16 để an toàn với 8GB VRAM của RTX 5060 Laptop
):
    print("=" * 60)
    print("Starting Building DFN5B-CLIP-ViT-H-14 FAISS Index...")
    print("=" * 60)
    
    # 1. Thiết bị chạy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Sử dụng thiết bị: {device}")

    # 2. Load model DFN5B-CLIP-ViT-H-14 từ Open_CLIP qua HuggingFace Hub
    model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
    print(f"[Info] Đang load model {model_name}...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()
    
    # Chiều embedding của dòng ViT-H-14 (DFN5B) là 1024 chiều
    dimension = 1024 

    # 3. Tìm kiếm toàn bộ các file ảnh có định dạng .jpg
    search_path = os.path.join(keyframes_dir, "**", "*.jpg")
    img_files = glob.glob(search_path, recursive=True)

    if not img_files:
        print(f"[Error] Không tìm thấy file ảnh .jpg nào trong đường dẫn: {keyframes_dir}")
        return

    # Sắp xếp danh sách file theo thứ tự đường dẫn để đảm bảo tuần tự chính xác
    img_files = sorted(img_files)
    total_files = len(img_files)
    print(f"[Info] Tìm thấy tổng số {total_files} keyframes. Bắt đầu trích xuất...")

    # 4. Khởi tạo FAISS IndexFlatIP (Inner Product) với chiều 1024
    index = faiss.IndexFlatIP(dimension)
    
    batch_images = []
    all_embeddings = []

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(img_files, desc="Encoding Keyframes")):
            try:
                # Mở và tiền xử lý ảnh theo chuẩn DFN5B-CLIP
                image = Image.open(file_path).convert("RGB")
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)

                # Khi đủ batch_size hoặc đến file cuối cùng thì đẩy qua GPU
                if len(batch_images) == batch_size or (i == total_files - 1):
                    tensor_stack = torch.stack(batch_images).to(device)
                    
                    # Trích xuất vector đặc trưng hình ảnh với Mixed Precision (tiết kiệm VRAM)
                    with torch.amp.autocast(device):
                        features = model.encode_image(tensor_stack)
                    
                    # ----------------------------------------------------
                    # BẮT BUỘC CHUẨN HÓA L2 NORM = 1.0 TRƯỚC KHI ĐƯA VÀO FAISS
                    # ----------------------------------------------------
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                    
                    # Chuyển đổi sang numpy float32
                    features_np = features.cpu().float().numpy().astype(np.float32)
                    all_embeddings.append(features_np)
                    
                    batch_images = []
            except Exception as e:
                print(f"\n[Warning] Lỗi khi xử lý file {file_path}: {e}")

        # Xả sạch bộ nhớ đệm GPU sau khi chạy xong
        if device == "cuda":
            torch.cuda.empty_cache()

    if not all_embeddings:
        print("[Error] Không thu được vector embedding nào hợp lệ.")
        return

    # Ghép toàn bộ các batch thành ma trận tổng (N, 1024)
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
    
    # Đường dẫn trỏ tới thư mục chứa keyframes và file index đầu ra mới
    KEYFRAMES_DIR = BASE_DIR / "data" / "keyframes"
    OUTPUT_INDEX = BASE_DIR / "data" / "indexes" / "dfn5b_clip_vit_h14.index"
    
    build_dfn5b_clip_vit_h14_index(
        keyframes_dir=str(KEYFRAMES_DIR), 
        output_index_path=str(OUTPUT_INDEX),
        batch_size=16
    )