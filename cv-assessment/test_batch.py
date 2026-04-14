import os
import glob
import random
from tqdm import tqdm
from src.pipeline.pipeline import process_image

def main():
    model_path = "models/best_map_model_backup.pth"
    image_dir = r"d:\sotatek\datasets\BOM-Folder- BOM-Dataset.coco\train"
    out_dir = "outputs/test_batch"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    # Pick 5 random images
    random.seed(42)
    selected_images = random.sample(image_paths, min(5, len(image_paths)))
    
    print(f"Bắt đầu test trên {len(selected_images)} ảnh ngẫu nhiên...")
    
    for i, img_path in enumerate(selected_images):
        name = os.path.basename(img_path)
        print(f"[{i+1}/{len(selected_images)}] Đang xử lý: {name}")
        out_base = os.path.join(out_dir, f"result_{name.split('.')[0]}")
        process_image(img_path, out_base)
        print(f" -> Đã lưu kết quả tại: {out_base}.jpg / .json")
        
    print("\n✅ HOÀN TẤT! Hãy mở thư mục cv-assessment/outputs/test_batch để xem ảnh.")

if __name__ == "__main__":
    main()
