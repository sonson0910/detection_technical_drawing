"""
Training Script for Engineering Drawing Object Detection (v2).
Fine-tunes Faster R-CNN on the engineering drawing dataset.

V2 enhancements:
- AdamW optimizer (better generalization on small datasets)
- CosineAnnealingWarmRestarts (periodic LR restarts escape local minima)
- Copy-Paste augmentation integration
- mAP evaluation using torchvision COCO evaluator
- Detailed training log with per-class metrics
- 3x oversampling with weighted sampler
"""
import os
import sys
import time
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detection.dataset import (
    EngineeringDrawingDataset,
    get_train_transforms,
    get_val_transforms,
    collate_fn,
)
from src.detection.model import get_model


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Linear warmup scheduler."""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


@torch.no_grad()
def compute_map(model, data_loader, device, iou_threshold=0.5):
    """Compute mAP@50 per class using simple AP calculation.

    Returns overall mAP and per-class AP dict.
    """
    model.eval()
    class_names = EngineeringDrawingDataset.CLASS_NAMES
    # per-class: list of (confidence, is_tp)
    all_preds = {c: [] for c in range(1, len(class_names))}
    all_gt_counts = {c: 0 for c in range(1, len(class_names))}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()

            # Count GT per class
            for gl in gt_labels:
                all_gt_counts[int(gl)] += 1

            # Match predictions to GT
            gt_matched = set()
            # Sort by score descending
            score_order = np.argsort(-pred_scores)

            for idx in score_order:
                plabel = int(pred_labels[idx])
                pbox = pred_boxes[idx]
                pscore = pred_scores[idx]

                if plabel < 1 or plabel >= len(class_names):
                    continue

                best_iou = 0
                best_gt = -1
                for gi, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
                    if int(glabel) != plabel or gi in gt_matched:
                        continue
                    iou = compute_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                is_tp = best_iou >= iou_threshold and best_gt >= 0
                if is_tp:
                    gt_matched.add(best_gt)
                all_preds[plabel].append((pscore, is_tp))

    # Compute AP per class
    class_ap = {}
    for c in range(1, len(class_names)):
        preds = sorted(all_preds[c], key=lambda x: -x[0])
        n_gt = all_gt_counts[c]
        if n_gt == 0:
            class_ap[class_names[c]] = 0.0
            continue

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        for score, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec = tp_cumsum / n_gt
            precisions.append(prec)
            recalls.append(rec)

        # 11-point interpolation AP
        if len(recalls) == 0:
            class_ap[class_names[c]] = 0.0
            continue

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_at_r = 0
            for p, r in zip(precisions, recalls):
                if r >= t:
                    p_at_r = max(p_at_r, p)
            ap += p_at_r / 11
        class_ap[class_names[c]] = ap

    mean_ap = np.mean(list(class_ap.values())) if class_ap else 0.0
    return mean_ap, class_ap


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Warmup for first epoch
    lr_scheduler = None
    if epoch == 1:
        warmup_iters = min(500, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.001)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with no valid annotations
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"  WARNING: Non-finite loss, skipping batch")
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += losses.item()
        num_batches += 1

        pbar.set_postfix(
            loss=f"{losses.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """Evaluate validation loss."""
    model.train()  # Keep in train mode to get losses
    total_loss = 0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if torch.isfinite(losses):
            total_loss += losses.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset paths
    root_dir = config["dataset"]["root"]
    ann_file = config["dataset"]["annotations"]

    # Load full dataset info to split
    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    num_images = len(coco_data["images"])
    indices = list(range(num_images))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = int(num_images * config["dataset"]["val_split"])
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"Images: {num_images} total, {len(train_indices)} train, {len(val_indices)} val")

    # Create datasets
    use_copy_paste = config["dataset"].get("copy_paste", False)
    train_dataset = EngineeringDrawingDataset(
        root_dir, ann_file,
        transforms=get_train_transforms(),
        indices=train_indices,
        copy_paste=use_copy_paste,
    )
    val_dataset = EngineeringDrawingDataset(
        root_dir, ann_file, transforms=get_val_transforms(), indices=val_indices
    )

    # WeightedRandomSampler to oversample Note-containing images
    oversample_factor = config["training"].get("oversample_factor", 3)
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset) * oversample_factor,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    num_classes = config["dataset"]["num_classes"]
    model = get_model(num_classes=num_classes, pretrained=config["model"]["pretrained"])
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Optimizer selection
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = config["training"].get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        print(f"Optimizer: AdamW (lr={config['training']['learning_rate']})")
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=config["training"]["learning_rate"],
            momentum=config["training"]["momentum"],
            weight_decay=config["training"]["weight_decay"],
        )
        print(f"Optimizer: SGD (lr={config['training']['learning_rate']})")

    # LR Scheduler
    num_epochs = config["training"]["num_epochs"]
    sched_name = config["training"].get("scheduler", "cosine_warm_restarts")

    if sched_name == "cosine_warm_restarts":
        T_0 = config["training"].get("T_0", 20)
        T_mult = config["training"].get("T_mult", 2)
        eta_min = config["training"].get("eta_min", 1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
        )
        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("lr_step_size", 30),
            gamma=config["training"].get("lr_gamma", 0.1),
        )
        print(f"Scheduler: StepLR")

    # Training
    save_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models")
    )
    os.makedirs(save_dir, exist_ok=True)

    # Training log
    log_path = os.path.join(save_dir, "..", "saved_models", "training_log_v2.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    training_log = []

    best_val_loss = float("inf")
    best_map = 0.0
    patience = config["training"].get("patience", 35)
    patience_counter = 0
    save_every = config["training"].get("save_every", 30)

    print(f"\n{'='*70}")
    print(f"Training {num_epochs} epochs | Faster R-CNN ResNet50-FPN-v2")
    print(f"Classes: {EngineeringDrawingDataset.CLASS_NAMES}")
    print(f"Augmentation: Aggressive + Copy-Paste={'ON' if use_copy_paste else 'OFF'}")
    print(f"Oversampling: {oversample_factor}x ({len(train_dataset) * oversample_factor} samples/epoch)")
    print(f"Early stopping patience: {patience}")
    print(f"{'='*70}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Evaluate loss
        val_loss = evaluate_loss(model, val_loader, device)

        # Compute mAP every 5 epochs (expensive)
        map_50 = 0.0
        class_ap = {}
        if epoch % 5 == 0 or epoch == 1:
            map_50, class_ap = compute_map(model, val_loader, device)

        lr_scheduler.step()
        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "map_50": round(map_50, 4),
            "class_ap": {k: round(v, 4) for k, v in class_ap.items()},
            "time_s": round(elapsed, 1),
        }
        training_log.append(log_entry)

        # Print
        map_str = f" | mAP@50: {map_50:.4f}" if map_50 > 0 else ""
        class_str = ""
        if class_ap:
            parts = [f"{cls[:4]}={ap:.2f}" for cls, ap in class_ap.items()]
            class_str = f" ({', '.join(parts)})"
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            f"{map_str}{class_str} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s"
        )

        # Save best by val_loss
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_loss, "map_50": map_50},
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

        # Save best by mAP
        if map_50 > best_map and map_50 > 0:
            best_map = map_50
            improved = True
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "map_50": map_50},
                os.path.join(save_dir, "best_map_model.pth"),
            )
            print(f"  -> Best mAP model saved (mAP@50={map_50:.4f})")

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        # Checkpoint
        if epoch % save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "map_50": map_50},
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

        # Save log periodically
        if epoch % 5 == 0:
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(),
         "val_loss": val_loss, "map_50": map_50},
        os.path.join(save_dir, "final_model.pth"),
    )

    # Save final log
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best mAP@50:   {best_map:.4f}")
    print(f"  Models saved:  {save_dir}")
    print(f"  Training log:  {log_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
