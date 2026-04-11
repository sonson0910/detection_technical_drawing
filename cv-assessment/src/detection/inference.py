"""
Inference Pipeline for Engineering Drawing Detection.
Loads trained model and performs object detection + cropping.
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.detection.model import get_model
from src.detection.dataset import EngineeringDrawingDataset


CLASS_NAMES = EngineeringDrawingDataset.CLASS_NAMES
CLASS_COLORS = {
    "PartDrawing": (0, 255, 0),    # Green
    "Note": (255, 165, 0),          # Orange
    "Table": (0, 120, 255),         # Blue
}


def load_model(checkpoint_path, num_classes=4, device="cuda"):
    """Load trained model from checkpoint."""
    model = get_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess image for inference."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)
    return image_tensor, image_np


def detect_objects(model, image_tensor, device, conf_threshold=0.5, nms_threshold=0.3):
    """Run detection with per-class confidence thresholds.
    
    Uses a lower threshold for Note class (0.08) since the model has limited
    training data for Notes (only 41 annotations), while PartDrawing and Table
    use the normal threshold.
    """
    # Per-class thresholds: Note gets much lower threshold
    CLASS_THRESHOLDS = {
        1: conf_threshold,        # PartDrawing - use normal threshold
        2: 0.08,                  # Note - very low due to class imbalance
        3: conf_threshold,        # Table - use normal threshold
    }
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])[0]

    # Use minimum threshold for initial filtering
    min_threshold = min(CLASS_THRESHOLDS.values())
    keep = predictions["scores"] >= min_threshold
    boxes = predictions["boxes"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()
    scores = predictions["scores"][keep].cpu().numpy()

    # Apply per-class confidence filtering and NMS
    nms_boxes, nms_labels, nms_scores = [], [], []
    for cls_id in np.unique(labels):
        cls_threshold = CLASS_THRESHOLDS.get(cls_id, conf_threshold)
        cls_mask = (labels == cls_id) & (scores >= cls_threshold)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        # NMS
        keep_idx = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(),
            cls_scores.tolist(),
            cls_threshold,
            nms_threshold,
        )
        if len(keep_idx) > 0:
            keep_idx = keep_idx.flatten()
            nms_boxes.extend(cls_boxes[keep_idx])
            nms_labels.extend([cls_id] * len(keep_idx))
            nms_scores.extend(cls_scores[keep_idx])

    # Post-processing: filter out tiny/invalid detections
    MIN_SIZE = {
        1: {"min_w": 80, "min_h": 60, "min_area": 5000},   # PartDrawing - must be substantial
        2: {"min_w": 30, "min_h": 15, "min_area": 800},     # Note - can be small annotations
        3: {"min_w": 50, "min_h": 25, "min_area": 2000},    # Table - includes title blocks
    }
    final_boxes, final_labels, final_scores = [], [], []
    for box, lbl, scr in zip(nms_boxes, nms_labels, nms_scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        area = w * h
        thresholds = MIN_SIZE.get(lbl, {"min_w": 20, "min_h": 20, "min_area": 500})
        if w >= thresholds["min_w"] and h >= thresholds["min_h"] and area >= thresholds["min_area"]:
            final_boxes.append(box)
            final_labels.append(lbl)
            final_scores.append(scr)

    return (
        np.array(final_boxes) if final_boxes else np.zeros((0, 4)),
        np.array(final_labels) if final_labels else np.zeros((0,), dtype=int),
        np.array(final_scores) if final_scores else np.zeros((0,)),
    )


def detect_note_regions(image_np, existing_boxes, existing_labels):
    """Hybrid Note detection: find text-heavy regions NOT covered by PartDrawing/Table.
    
    Uses morphological text region analysis to find text blocks in remaining space
    after removing detected PartDrawing and Table regions.
    """
    h, w = image_np.shape[:2]
    
    # Create mask of already-detected regions (with padding)
    covered_mask = np.zeros((h, w), dtype=np.uint8)
    for box, lbl in zip(existing_boxes, existing_labels):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
        x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
        covered_mask[y1:y2, x1:x2] = 255
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Adaptive threshold to detect text pixels
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 10)
    
    # Remove already-detected regions
    binary[covered_mask == 255] = 0
    
    # Step 1: Connect characters horizontally into text lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
    line_connected = cv2.dilate(binary, kernel_h, iterations=1)
    
    # Step 2: Close small gaps within lines
    kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    line_connected = cv2.morphologyEx(line_connected, cv2.MORPH_CLOSE, kernel_close_h)
    
    # Step 3: Connect nearby lines into paragraphs (gentle vertical merge)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
    paragraph = cv2.dilate(line_connected, kernel_v, iterations=2)
    
    # Step 4: Close remaining gaps within paragraphs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    paragraph = cv2.morphologyEx(paragraph, cv2.MORPH_CLOSE, kernel_close)
    
    # Enforce mask again to prevent morphological operations from bleeding into existing boxes
    paragraph[covered_mask == 255] = 0
    
    # Find contours of paragraph regions
    contours, _ = cv2.findContours(paragraph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    note_boxes = []
    min_note_area = h * w * 0.002   # At least 0.2% of image
    max_note_area = h * w * 0.35    # At most 35% of image
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        # Size filter
        if area < min_note_area or area > max_note_area:
            continue
        if cw < 40 or ch < 20:
            continue
        
        # Aspect ratio: Notes are typically wider than tall or roughly square
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.3 or aspect > 10:
            continue
            
        # Check text density in the original binary (before morphology)
        roi_binary = binary[y:y+ch, x:x+cw]
        text_density = np.sum(roi_binary > 0) / (cw * ch) if cw * ch > 0 else 0
        
        # Notes should have real text (1.5% - 65%)
        if text_density < 0.015 or text_density > 0.65:
            continue
        
        # Skip regions that significantly overlap with existing detections
        overlap = False
        for box in existing_boxes:
            bx1, by1, bx2, by2 = map(int, box)
            ix1 = max(x, bx1)
            iy1 = max(y, by1)
            ix2 = min(x + cw, bx2)
            iy2 = min(y + ch, by2)
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area / area > 0.2:
                    overlap = True
                    break
        
        if not overlap:
            # Add small padding around the Note
            px, py = int(cw * 0.03), int(ch * 0.05)
            note_boxes.append([
                max(0, x - px), max(0, y - py),
                min(w, x + cw + px), min(h, y + ch + py)
            ])
    
    # Merge overlapping Note boxes
    note_boxes = _merge_overlapping_boxes(note_boxes)
    
    return note_boxes


def _merge_overlapping_boxes(boxes, iou_threshold=0.3):
    """Merge boxes that overlap significantly."""
    if not boxes:
        return boxes
    
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    merged = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
        x1, y1, x2, y2 = boxes[i]
        
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            
            # Check overlap
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_j = (bx2 - bx1) * (by2 - by1)
                if inter / area_j > iou_threshold:
                    # Merge
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used.add(j)
        
        merged.append([x1, y1, x2, y2])
    
    return merged


def crop_objects(image_np, boxes, labels, scores, output_dir, image_name):
    """Crop detected objects and save as individual images."""
    os.makedirs(output_dir, exist_ok=True)
    crops = []

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = map(int, box)
        # Clamp to image bounds
        h, w = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image_np[y1:y2, x1:x2]
        class_name = CLASS_NAMES[label]

        # Create class-specific directory
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        crop_filename = f"{os.path.splitext(image_name)[0]}_{class_name}_{i+1}.png"
        crop_path = os.path.join(class_dir, crop_filename)
        Image.fromarray(crop).save(crop_path)

        crops.append({
            "id": i + 1,
            "class": class_name,
            "confidence": round(float(score), 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_path": crop_path,
        })

    return crops


def draw_detections(image_np, boxes, labels, scores):
    """Draw bounding boxes on image with class-specific colors."""
    image_vis = image_np.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[label]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw box
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)

        # Draw label
        text = f"{class_name}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image_vis


def generate_json_output(image_name, detections, ocr_results=None):
    """Generate JSON output for a single image."""
    objects = []
    for det in detections:
        obj = {
            "id": det["id"],
            "class": det["class"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
        }

        # Add OCR content if available
        if ocr_results and det["id"] in ocr_results:
            obj["ocr_content"] = ocr_results[det["id"]]
        elif det["class"] in ["Note", "Table"]:
            obj["ocr_content"] = None
        # PartDrawing doesn't need OCR

        objects.append(obj)

    return {"image": image_name, "objects": objects}


def run_inference(model, image_path, output_dir, device, conf_threshold=0.5, nms_threshold=0.3):
    """Run full inference pipeline on a single image."""
    image_name = os.path.basename(image_path)
    image_tensor, image_np = preprocess_image(image_path)

    # Detect
    boxes, labels, scores = detect_objects(model, image_tensor, device, conf_threshold, nms_threshold)

    if len(boxes) == 0:
        print(f"  No objects detected in {image_name}")
        return None, image_np

    # Crop
    crops_dir = os.path.join(output_dir, "crops")
    detections = crop_objects(image_np, boxes, labels, scores, crops_dir, image_name)

    # Visualize
    vis_image = draw_detections(image_np, boxes, labels, scores)

    # Save visualization
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, f"vis_{image_name}")
    Image.fromarray(vis_image).save(vis_path)

    # Generate JSON (OCR will be added later in pipeline)
    json_output = generate_json_output(image_name, detections)

    # Save JSON
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"{os.path.splitext(image_name)[0]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"  Detected {len(detections)} objects in {image_name}")
    return json_output, vis_image


def batch_inference(model_path, input_dir, output_dir, device="cuda", conf_threshold=0.5):
    """Run inference on a directory of images."""
    model = load_model(model_path, device=device)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {input_dir}")
    results = []

    for img_file in sorted(image_files):
        img_path = os.path.join(input_dir, img_file)
        result, _ = run_inference(model, img_path, output_dir, device, conf_threshold)
        if result:
            results.append(result)

    # Save all results
    all_results_path = os.path.join(output_dir, "all_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {all_results_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run detection inference")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_inference(args.model, args.input, args.output, args.device, args.confidence)
    else:
        model = load_model(args.model, device=args.device)
        run_inference(model, args.input, args.output, args.device, args.confidence)
