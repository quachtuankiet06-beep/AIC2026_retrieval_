import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

print("[INFO] Đang tải mô hình YOLOv8x-OIV7 (601 classes) từ Ultralytics...")
model = YOLO("yolov8x-oiv7.pt")

def detect_objects_single_image(image_path, conf=0.1):
    """
    Trích xuất thực thể bằng YOLOv8x-OIV7 chạy trên GPU.
    """
    try:
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        
        entities = set()
        for result in results:
            classes_ids = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            for cls_id, score in zip(classes_ids, scores):
                if score >= conf:
                    class_name = model.names[int(cls_id)]
                    entities.add(str(class_name).strip())

        return list(entities)

    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý ảnh {image_path}: {e}")
        return []

def update_mapping_with_yolov8_oiv7_full(mapping_json_path, custom_keyframes_dir, conf=0.1, save_every=1000):
    mapping_json_path = Path(mapping_json_path)
    custom_keyframes_dir = Path(custom_keyframes_dir)

    if not mapping_json_path.exists():
        print(f"[ERROR] Không tìm thấy file mapping: {mapping_json_path}")
        return

    with open(mapping_json_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    print(f"[INFO] Bắt đầu quét lại từ đầu với YOLOv8x-OIV7 cho {len(mapping_data)} keyframes (conf={conf})...")

    updated_count = 0
    for i, item in enumerate(tqdm(mapping_data, desc="YOLOv8x-OIV7 Full Processing")):
        
        rel_path = item["keyframe_path"]
        img_abs_path = BASE_DIR / "data" / rel_path
        
        if img_abs_path.exists():
            entities = detect_objects_single_image(img_abs_path, conf=conf)
            item["object_entities"] = entities

            video_id = item["video_id"]
            n = item["keyframe_index"]
            item["object_path"] = f"data/object/custom_objects/{video_id}/{n:03d}.json"
            
            updated_count += 1
        else:
            item["object_entities"] = []

        if (i + 1) % save_every == 0:
            with open(mapping_json_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=4)

    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Đã xử lý lại từ đầu thành công cho {updated_count} keyframes!")
    print(f"[OUTPUT] File mapping đã lưu tại: {mapping_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    BASE_DIR = Path(r"D:\tai_lieu_hoc_tap\AIC_2026")

    update_mapping_with_yolov8_oiv7_full(
        mapping_json_path=BASE_DIR / "data" / "indexes" / "keyframes_mapping_new.json",
        custom_keyframes_dir=BASE_DIR / "data" / "Custom_Keyframes",
        conf=0.1,
        save_every=1000
    )