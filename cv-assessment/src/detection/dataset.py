"""
COCO Dataset Loader for Engineering Drawing Detection.
Loads images and annotations in COCO format for training Faster R-CNN.

CV Expert optimizations:
- Class-aware oversampling to fix Note class imbalance (41 vs 278 PartDrawing)
- Targeted augmentations for engineering drawing characteristics
- Proper bbox clamping and validation
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

    def __init__(self, root_dir, annotation_file, transforms=None, indices=None):
        self.root_dir = root_dir
        self.transforms = transforms

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

    def __len__(self):
        return len(self.images)

    def get_class_weights(self):
        """Compute per-image sampling weights based on class presence.
        Images with Note annotations get higher weight to combat imbalance.
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
            # Give higher weight to images with Note (3x) and Table (1.5x)
            w = 1.0
            if has_note:
                w = 3.0
            elif has_table:
                w = 1.5
            weights.append(w)
        return weights

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
    """Training augmentations optimized for engineering drawings.
    
    Key decisions:
    - Mild geometric transforms only (drawings must stay readable)
    - Color/brightness jitter to handle scan quality variations
    - No heavy cropping (would lose annotations)
    """
    return A.Compose(
        [
            # Color augmentations (simulate different scan/photo qualities)
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.GaussNoise(p=0.15),
            # Mild geometric (engineering drawings shouldn't be heavily distorted)
            A.Affine(
                scale=(0.85, 1.15),
                rotate=(-5, 5),
                shear=(-3, 3),
                p=0.4,
            ),
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
