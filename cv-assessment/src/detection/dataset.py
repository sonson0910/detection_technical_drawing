"""
COCO Dataset Loader for Engineering Drawing Detection.
Loads images and annotations in COCO format for training Faster R-CNN.

CV Expert optimizations (v2 - Enhanced for 58-image dataset):
- Aggressive augmentation to combat severe overfitting
- Copy-Paste augmentation for Note class (41 → ~200+ effective annotations)
- CLAHE, perspective, coarse dropout, quality degradation
- Multi-scale training via RandomScale
- Class-aware oversampling (4x for Note, 2x for Table)
"""
import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class EngineeringDrawingDataset(Dataset):
    """Dataset for engineering drawing object detection in COCO format."""

    # Map original category IDs to contiguous IDs (1-indexed for Faster R-CNN)
    CATEGORY_MAP = {
        0: 1,  # partdrawing -> 1
        1: 2,  # note -> 2
        2: 1,  # partdrawing -> 1 (duplicate in dataset)
        3: 3,  # table -> 3
    }
    CLASS_NAMES = ["__background__", "PartDrawing", "Note", "Table"]

    def __init__(self, root_dir, annotation_file, transforms=None, indices=None,
                 copy_paste=False):
        self.root_dir = root_dir
        self.transforms = transforms
        self.copy_paste = copy_paste

        with open(annotation_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        if indices is not None:
            self.images = [self.images[i] for i in indices]

        # Build image_id -> annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Pre-extract Note + small Table regions for Copy-Paste augmentation
        if self.copy_paste:
            self._note_crops = self._extract_note_crops()
            self._small_table_crops = self._extract_small_table_crops()

    def _extract_note_crops(self):
        """Pre-extract Note region crops from ALL images for Copy-Paste."""
        crops = []
        # Use ALL images in the dataset (not just filtered self.images)
        all_images = self.coco_data["images"]
        for img_info in all_images:
            img_id = img_info["id"]
            anns = self.img_to_anns.get(img_id, [])
            note_anns = [a for a in anns
                         if self.CATEGORY_MAP.get(a["category_id"], 1) == 2]
            if not note_anns:
                continue

            img_path = os.path.join(self.root_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                ih, iw = image.shape[:2]
                for ann in note_anns:
                    x, y, w, h = [int(float(v)) for v in ann["bbox"]]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(iw, x + w), min(ih, y + h)
                    if x2 - x1 > 10 and y2 - y1 > 5:
                        crop = image[y1:y2, x1:x2].copy()
                        crops.append(crop)
            except Exception:
                continue

        print(f"  [Copy-Paste] Extracted {len(crops)} Note crops")
        return crops

    def _extract_small_table_crops(self):
        """Pre-extract small Table region crops for Copy-Paste augmentation."""
        crops = []
        all_images = self.coco_data["images"]
        for img_info in all_images:
            img_id = img_info["id"]
            anns = self.img_to_anns.get(img_id, [])
            table_anns = [a for a in anns
                          if self.CATEGORY_MAP.get(a["category_id"], 1) == 3]
            if not table_anns:
                continue

            img_path = os.path.join(self.root_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                ih, iw = image.shape[:2]
                for ann in table_anns:
                    x, y, w, h = [int(float(v)) for v in ann["bbox"]]
                    # Only extract SMALL tables (footer/title-block type)
                    if h > 60 and w > 150:
                        continue  # Skip large tables — already well-detected
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(iw, x + w), min(ih, y + h)
                    if x2 - x1 > 10 and y2 - y1 > 5:
                        crop = image[y1:y2, x1:x2].copy()
                        crops.append(crop)
            except Exception:
                continue

        print(f"  [Copy-Paste] Extracted {len(crops)} small Table crops")
        return crops

    def __len__(self):
        return len(self.images)

    def get_class_weights(self):
        """Compute per-image sampling weights based on class presence.
        Images with Note annotations get highest weight to combat imbalance.
        """
        weights = []
        for img_info in self.images:
            img_id = img_info["id"]
            anns = self.img_to_anns.get(img_id, [])
            has_note = any(
                self.CATEGORY_MAP.get(a["category_id"], 1) == 2 for a in anns
            )
            has_table = any(
                self.CATEGORY_MAP.get(a["category_id"], 1) == 3 for a in anns
            )
            # Give higher weight to images with Note (4x) and Table (2x)
            w = 1.0
            if has_note:
                w = 4.0
            elif has_table:
                w = 2.0
            weights.append(w)
        return weights

    def _apply_copy_paste(self, image, boxes, labels, areas):
        """Copy-Paste augmentation: paste Note + small Table crops into empty regions."""
        has_notes = self._note_crops and len(self._note_crops) > 0
        has_tables = hasattr(self, '_small_table_crops') and self._small_table_crops and len(self._small_table_crops) > 0
        
        if (not has_notes and not has_tables) or random.random() > 0.4:
            return image, boxes, labels, areas

        ih, iw = image.shape[:2]

        # Create occupied mask from existing boxes
        occupied = np.zeros((ih, iw), dtype=bool)
        for bx in boxes:
            x1, y1, x2, y2 = map(int, bx)
            occupied[max(0,y1):min(ih,y2), max(0,x1):min(iw,x2)] = True

        # Build paste candidates: [(crop, class_id), ...]
        paste_pool = []
        if has_notes:
            paste_pool.extend([(c, 2) for c in self._note_crops])  # Note = class 2
        if has_tables:
            # Oversample small tables (3x weight) since they're harder to detect
            paste_pool.extend([(c, 3) for c in self._small_table_crops] * 3)

        # Paste 1-3 random crops
        n_paste = random.randint(1, 3)
        for _ in range(n_paste):
            crop, cls_id = random.choice(paste_pool)
            ch, cw = crop.shape[:2]

            # Random scale (0.6x - 1.4x)
            scale = random.uniform(0.6, 1.4)
            new_w = max(20, int(cw * scale))
            new_h = max(10, int(ch * scale))
            if new_w >= iw or new_h >= ih:
                continue

            resized = cv2.resize(crop, (new_w, new_h))

            # For small tables, prefer footer region (bottom 30%)
            for attempt in range(10):
                px = random.randint(0, iw - new_w)
                py_max = ih - new_h
                if py_max < 0:
                    break  # Crop taller than image, skip
                if cls_id == 3 and attempt < 6:  # 60% chance of footer placement
                    py_min = max(0, int(ih * 0.7))
                    if py_min > py_max:
                        py_min = 0  # Fall back to full-image placement
                    py = random.randint(py_min, py_max)
                else:
                    py = random.randint(0, py_max)

                # Check overlap with existing boxes
                roi = occupied[py:py+new_h, px:px+new_w]
                if np.sum(roi) / roi.size < 0.1:
                    image[py:py+new_h, px:px+new_w] = resized
                    occupied[py:py+new_h, px:px+new_w] = True

                    boxes.append([px, py, px + new_w, py + new_h])
                    labels.append(cls_id)
                    areas.append(new_w * new_h)
                    break

        return image, boxes, labels, areas

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.root_dir, img_info["file_name"])

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Get annotations
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        img_h, img_w = image.shape[:2]
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            # Clamp to image bounds
            x1 = max(0, min(x, img_w - 1))
            y1 = max(0, min(y, img_h - 1))
            x2 = max(0, min(x + w, img_w))
            y2 = max(0, min(y + h, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.CATEGORY_MAP.get(ann["category_id"], 1))
            areas.append((x2 - x1) * (y2 - y1))

        # Copy-Paste augmentation for Note class
        if self.copy_paste and len(boxes) > 0:
            image, boxes, labels, areas = self._apply_copy_paste(
                image.copy(), boxes, labels, areas
            )

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        # Apply augmentations
        if self.transforms:
            bboxes_list = boxes.numpy().tolist() if len(boxes) > 0 else []
            labels_list = labels.numpy().tolist() if len(labels) > 0 else []

            transformed = self.transforms(
                image=image,
                bboxes=bboxes_list,
                labels=labels_list,
            )
            image = transformed["image"]
            if len(transformed["bboxes"]) > 0:
                boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                areas = torch.zeros((0,), dtype=torch.float32)
        else:
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        return image, target


def get_train_transforms():
    """Training augmentations - AGGRESSIVE for 58-image dataset.

    Strategy: maximize diversity while keeping drawings recognizable.
    Key additions vs v1:
    - CLAHE: enhances faded scan lines
    - Perspective: simulates different scan angles
    - CoarseDropout: forces robust feature learning (CutOut regularization)
    - ImageCompression: simulates JPEG artifacts from scans
    - Downscale: simulates low-res scans
    - Sharpen: counteracts blur
    - Multi-scale via RandomScale
    """
    return A.Compose(
        [
            # --- Geometric augmentations ---
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            A.Affine(
                scale=(0.8, 1.2),
                rotate=(-8, 8),
                shear=(-5, 5),
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.2),

            # --- Quality / scan simulation ---
            A.OneOf([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.0), p=1.0),
                A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=1.0),
            ], p=0.4),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # --- Color/brightness (scan quality variation) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.5,
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.2),

            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.15,
            ),

            # --- Quality degradation (simulate low quality scans) ---
            A.ImageCompression(quality_range=(60, 100), p=0.2),
            A.Downscale(scale_range=(0.5, 0.9), p=0.15),

            # --- CutOut regularization ---
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=255,  # White fill (matches drawing background)
                p=0.25,
            ),

            # --- Normalize ---
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=100,
            min_visibility=0.3,
        ),
    )


def get_val_transforms():
    """Validation transforms (no augmentation, just normalize)."""
    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=0,
            min_visibility=0.0,
        ),
    )


def collate_fn(batch):
    """Custom collate function for variable-size targets."""
    return tuple(zip(*batch))
