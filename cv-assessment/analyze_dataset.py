"""Analyze COCO dataset annotations."""
import json
import numpy as np
from collections import Counter

with open('D:/sotatek/datasets/BOM-Folder- BOM-Dataset.coco/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)

print('=== CATEGORIES ===')
for cat in data['categories']:
    print(f"  ID {cat['id']}: {cat['name']}")

print(f"\n=== IMAGES: {len(data['images'])} ===")
print(f"=== ANNOTATIONS: {len(data['annotations'])} ===")

# Build mapping
cat_map = {c['id']: c['name'] for c in data['categories']}

# Count per category ID
cat_counts = Counter()
for ann in data['annotations']:
    cat_counts[ann['category_id']] += 1
for cid, cnt in sorted(cat_counts.items()):
    print(f"  CatID={cid} ({cat_map.get(cid, '?')}): {cnt} annotations")

# Show bbox sizes per class
for cid in sorted(cat_counts.keys()):
    bboxes = [ann['bbox'] for ann in data['annotations'] if ann['category_id'] == cid]
    widths = [float(b[2]) for b in bboxes]
    heights = [float(b[3]) for b in bboxes]
    areas = [w*h for w, h in zip(widths, heights)]
    print(f"\n  {cat_map.get(cid)} (CatID={cid}) bbox stats ({len(bboxes)} boxes):")
    print(f"    Width:  min={min(widths):.0f}, max={max(widths):.0f}, avg={np.mean(widths):.0f}")
    print(f"    Height: min={min(heights):.0f}, max={max(heights):.0f}, avg={np.mean(heights):.0f}")
    print(f"    Area:   min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")

# Show annotation for reference image (5_jpg)
print("\n=== REFERENCE IMAGE (5_jpg) ===")
for img in data['images']:
    if '5_jpg' in img['file_name']:
        print(f"Image: {img['file_name']} ({img['width']}x{img['height']}), ID={img['id']}")
        anns = [a for a in data['annotations'] if a['image_id'] == img['id']]
        for a in anns:
            cname = cat_map[a['category_id']]
            x, y, w, h = [float(v) for v in a['bbox']]
            print(f"  CatID={a['category_id']} {cname}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}] ({w:.0f}x{h:.0f})")
        break

# Show annotation for image 10_png (tested)
print("\n=== IMAGE 10_png ===")
for img in data['images']:
    if '10_png' in img['file_name']:
        print(f"Image: {img['file_name']} ({img['width']}x{img['height']}), ID={img['id']}")
        anns = [a for a in data['annotations'] if a['image_id'] == img['id']]
        for a in anns:
            cname = cat_map[a['category_id']]
            x, y, w, h = [float(v) for v in a['bbox']]
            print(f"  CatID={a['category_id']} {cname}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}] ({w:.0f}x{h:.0f})")
        break
