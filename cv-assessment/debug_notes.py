"""Debug: check what binary looks like at the small Note location."""
import sys; sys.path.insert(0, '.')
import numpy as np, cv2
from src.detection.inference import load_model, preprocess_image, detect_objects
from PIL import Image

model = load_model('models/best_model.pth', device='cuda')
img_path = 'D:/sotatek/datasets/BOM-Folder- BOM-Dataset.coco/train/5_jpg.rf.EAoozrpzo12Bv7IVb4kI.jpg'
tensor, img_np = preprocess_image(img_path)
boxes, labels, scores = detect_objects(model, tensor, 'cuda', conf_threshold=0.5)

h, w = img_np.shape[:2]
print(f"Image: {w}x{h}")

# The small Note is at approximately [765,204,834,248] (from model at 0.05 conf)
# Check if it's inside any covered region
for box, lbl in zip(boxes, labels):
    x1, y1, x2, y2 = map(int, box)
    from src.detection.inference import CLASS_NAMES
    print(f"Detection: {CLASS_NAMES[lbl]} [{x1},{y1},{x2},{y2}]")
    # Check if Note area overlaps
    note_x1, note_y1, note_x2, note_y2 = 765, 204, 834, 248
    ix1 = max(x1, note_x1)
    iy1 = max(y1, note_y1)
    ix2 = min(x2, note_x2)
    iy2 = min(y2, note_y2)
    if ix1 < ix2 and iy1 < iy2:
        inter = (ix2-ix1)*(iy2-iy1)
        note_area = (note_x2-note_x1)*(note_y2-note_y1)
        print(f"  -> Overlaps with Note region! overlap={inter}/{note_area} = {inter/note_area:.2f}")

# Check the actual content in that region
roi = img_np[200:255, 760:840]
Image.fromarray(roi).save('outputs/debug_small_note_roi.png')
print(f"\nSmall Note ROI saved")
