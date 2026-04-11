"""Analyze the COCO annotations to understand class distribution."""
import json

ann_file = "D:/sotatek/datasets/BOM-Folder- BOM-Dataset.coco/train/_annotations.coco.json"
with open(ann_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== Categories ===")
for cat in data["categories"]:
    print(f"  ID {cat['id']}: {cat['name']}")

print(f"\n=== Dataset Stats ===")
print(f"Total images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")

# Count per category
cat_counts = {}
cat_areas = {}
for ann in data["annotations"]:
    cid = ann["category_id"]
    cat_name = next((c["name"] for c in data["categories"] if c["id"] == cid), f"unknown_{cid}")
    cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    
    x, y, w, h = [float(v) for v in ann["bbox"]]
    area = w * h
    if cat_name not in cat_areas:
        cat_areas[cat_name] = []
    cat_areas[cat_name].append({"w": w, "h": h, "area": area})

print("\n=== Annotations per Category ===")
for name, count in sorted(cat_counts.items()):
    areas = cat_areas[name]
    avg_w = sum(a["w"] for a in areas) / len(areas)
    avg_h = sum(a["h"] for a in areas) / len(areas)
    avg_area = sum(a["area"] for a in areas) / len(areas)
    min_area = min(a["area"] for a in areas)
    max_area = max(a["area"] for a in areas)
    print(f"  {name}: {count} annotations")
    print(f"    Avg size: {avg_w:.0f}x{avg_h:.0f}, Avg area: {avg_area:.0f}")
    print(f"    Area range: {min_area:.0f} - {max_area:.0f}")

# Map to our classes
print("\n=== Mapped to Model Classes ===")
CATEGORY_MAP = {0: "PartDrawing", 1: "Note", 2: "PartDrawing", 3: "Table"}
mapped_counts = {}
for ann in data["annotations"]:
    mapped = CATEGORY_MAP.get(ann["category_id"], "Unknown")
    mapped_counts[mapped] = mapped_counts.get(mapped, 0) + 1

for name, count in sorted(mapped_counts.items()):
    print(f"  {name}: {count}")
