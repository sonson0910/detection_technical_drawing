"""
Training Script for Engineering Drawing Object Detection.
Fine-tunes Faster R-CNN on the engineering drawing dataset.

CV Expert optimizations:
- WeightedRandomSampler to oversample images with Note annotations
- Warmup LR + StepLR schedule (better than cosine for small datasets)
- Early stopping with patience to prevent overfitting
- Gradient clipping for training stability
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
def evaluate(model, data_loader, device):
    """Evaluate on validation set."""
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
    train_dataset = EngineeringDrawingDataset(
        root_dir, ann_file, transforms=get_train_transforms(), indices=train_indices
    )
    val_dataset = EngineeringDrawingDataset(
        root_dir, ann_file, transforms=get_val_transforms(), indices=val_indices
    )

    # WeightedRandomSampler to oversample Note-containing images
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset) * 2,  # 2x oversampling
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,  # Use weighted sampler instead of shuffle
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

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Step LR (simpler, better for small datasets)
    num_epochs = config["training"]["num_epochs"]
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"],
    )

    # Training
    save_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models")
    )
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience = 25
    patience_counter = 0

    print(f"\nTraining {num_epochs} epochs | Faster R-CNN ResNet50-FPN-v2")
    print(f"Classes: {EngineeringDrawingDataset.CLASS_NAMES}")
    print(f"Note oversampling: 3x weight")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        lr_scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss},
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss},
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
        os.path.join(save_dir, "final_model.pth"),
    )

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Models: {save_dir}")


if __name__ == "__main__":
    main()
