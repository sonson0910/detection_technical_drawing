"""Test detection on the reference image - match ground truth: 3 PartDrawing, 2 Note, 2 Table.
Includes post-processing to fix 'lem' (overlap/imprecision) issues."""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import torch
import cv2
from src.detection.inference import (
    load_model, preprocess_image, detect_objects, 
    detect_note_regions, draw_detections, CLASS_NAMES
)
from src.detection.postprocess import post_process_detections
from PIL import Image

# Load model
model = load_model('models/best_model.pth', device='cuda')

# Test on reference image
img_path = 'D:/sotatek/datasets/BOM-Folder- BOM-Dataset.coco/train/5_jpg.rf.EAoozrpzo12Bv7IVb4kI.jpg'
tensor, img_np = preprocess_image(img_path)

# Model detection (with per-class thresholds: Note uses 0.08)
boxes, labels, scores = detect_objects(model, tensor, 'cuda', conf_threshold=0.5)

print("=== Model Detections (raw) ===")
for box, lbl, scr in zip(boxes, labels, scores):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    print(f"  {CLASS_NAMES[lbl]} ({scr:.3f}): [{x1},{y1},{x2},{y2}] size={w}x{h}")

# Hybrid Note detection (for text blocks missed by model)
# Only pass non-Note model detections as "covered" areas
non_note_mask = labels != 2
non_note_boxes = boxes[non_note_mask] if len(boxes) > 0 else np.zeros((0, 4))
non_note_labels = labels[non_note_mask] if len(labels) > 0 else np.zeros((0,), dtype=int)

note_boxes = detect_note_regions(img_np, non_note_boxes, non_note_labels)
print(f"\n=== Hybrid Note Detection ===")
for i, box in enumerate(note_boxes):
    x1, y1, x2, y2 = box
    print(f"  Note #{i+1}: [{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1}")

# Merge: add hybrid Notes that don't overlap with model-detected Notes
model_note_mask = labels == 2
model_note_boxes = boxes[model_note_mask] if np.any(model_note_mask) else np.zeros((0, 4))

all_boxes = list(boxes)
all_labels = list(labels)
all_scores = list(scores)

for nb in note_boxes:
    # Check if hybrid Note overlaps with any model-detected Note
    is_duplicate = False
    for mnb in model_note_boxes:
        ix1 = max(nb[0], mnb[0])
        iy1 = max(nb[1], mnb[1])
        ix2 = min(nb[2], mnb[2])
        iy2 = min(nb[3], mnb[3])
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2-ix1) * (iy2-iy1)
            nb_area = (nb[2]-nb[0]) * (nb[3]-nb[1])
            if inter / nb_area > 0.3:
                is_duplicate = True
                break
    
    if not is_duplicate:
        all_boxes.append(np.array(nb, dtype=float))
        all_labels.append(2)
        all_scores.append(0.85)

all_boxes = np.array(all_boxes) if all_boxes else np.zeros((0, 4))
all_labels = np.array(all_labels)
all_scores = np.array(all_scores)

print(f"\n=== Before Post-Processing ===")
for box, lbl, scr in zip(all_boxes, all_labels, all_scores):
    x1, y1, x2, y2 = map(int, box)
    print(f"  {CLASS_NAMES[int(lbl)]} ({scr:.3f}): [{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1}")

# POST-PROCESSING: Fix "lem" issues
refined_boxes, refined_labels, refined_scores = post_process_detections(
    img_np, all_boxes, all_labels, all_scores,
    footer_ratio=0.06,
    overlap_iou_thresh=0.02
)

print(f"\n=== After Post-Processing (Refined) ===")
for box, lbl, scr in zip(refined_boxes, refined_labels, refined_scores):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    print(f"  {CLASS_NAMES[int(lbl)]} ({scr:.3f}): [{x1},{y1},{x2},{y2}] size={w}x{h}")

# Summary
print(f"\n=== Final Summary ===")
class_counts = {}
for lbl in refined_labels:
    cname = CLASS_NAMES[int(lbl)]
    class_counts[cname] = class_counts.get(cname, 0) + 1
print(f"Total: {len(refined_boxes)} detections")
print(f"By class: {class_counts}")

# Expected: 3 PartDrawing, 2 Note, 2 Table
for cls, expected in [('PartDrawing', 3), ('Note', 2), ('Table', 2)]:
    actual = class_counts.get(cls, 0)
    status = 'OK' if actual == expected else 'MISS'
    print(f"  [{status}] {cls}: {actual}/{expected}")

# Check overlap reduction
print(f"\n=== Overlap Analysis ===")
for i in range(len(refined_boxes)):
    for j in range(i+1, len(refined_boxes)):
        b1 = refined_boxes[i]
        b2 = refined_boxes[j]
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2-ix1) * (iy2-iy1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            o1 = inter/area1 * 100 if area1 > 0 else 0
            o2 = inter/area2 * 100 if area2 > 0 else 0
            if max(o1, o2) > 1:
                print(f"  OVERLAP: {CLASS_NAMES[int(refined_labels[i])]} #{i+1} vs "
                      f"{CLASS_NAMES[int(refined_labels[j])]} #{j+1}: "
                      f"{o1:.1f}%/{o2:.1f}%")

# Save raw vs refined visualizations
vis_raw = draw_detections(img_np, all_boxes, all_labels, all_scores)
Image.fromarray(vis_raw).save('outputs/test_reference_raw.png')
print(f"\nRaw visualization: outputs/test_reference_raw.png")

vis_refined = draw_detections(img_np, refined_boxes, refined_labels, refined_scores)
Image.fromarray(vis_refined).save('outputs/test_reference_vis.png')
print(f"Refined visualization: outputs/test_reference_vis.png")
