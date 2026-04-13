"""
Resume Training: Fine-tune with Small Table Copy-Paste Augmentation.

Loads the best_map_model.pth checkpoint and continues training for 30 more
epochs with enhanced augmentation targeting small footer tables.

Key changes from initial training:
- Lower LR (1e-4) for fine-tuning stability
- Copy-Paste now includes small Table crops (3x oversampled)
- Small tables pasted preferentially in footer region (bottom 30%)
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detection.dataset import (
    EngineeringDrawingDataset,
    get_train_transforms,
    get_val_transforms,
    collate_fn,
)
from src.detection.model import get_model
from src.detection.train import compute_map, train_one_epoch, evaluate_loss


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resume config ───────────────────────────────────────────────
    RESUME_EPOCHS = 5           # Extra epochs (shorter, to avoid destroying other classes)
    RESUME_LR     = 5e-5        # Lower LR for fine-tuning
    CHECKPOINT    = os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_map_model.pth")
    # ────────────────────────────────────────────────────────────────

    # Dataset
    root_dir = config["dataset"]["root"]
    ann_file = config["dataset"]["annotations"]

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

    # Datasets
    train_dataset = EngineeringDrawingDataset(
        root_dir, ann_file,
        transforms=get_train_transforms(),
        indices=train_indices,
        copy_paste=False,
    )
    val_dataset = EngineeringDrawingDataset(
        root_dir, ann_file, transforms=get_val_transforms(), indices=val_indices
    )

    # Weighted sampler
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

    # Load model + checkpoint
    num_classes = config["dataset"]["num_classes"]
    model = get_model(num_classes=num_classes, pretrained=False)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        prev_map = ckpt.get("map_50", 0.0)
    else:
        model.load_state_dict(ckpt)
        start_epoch = 0
        prev_map = 0.0

    model.to(device)
    print(f"Loaded checkpoint: epoch={start_epoch}, mAP@50={prev_map:.4f}")

    # AdamW with lower LR
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=RESUME_LR, weight_decay=0.005)
    print(f"Optimizer: AdamW (lr={RESUME_LR})")

    # Cosine annealing
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RESUME_EPOCHS, eta_min=1e-6,
    )

    # Paths
    save_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models")
    )
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "..", "saved_models", "training_log_resume.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    training_log = []

    best_map = prev_map
    best_val_loss = float("inf")

    print(f"\n{'='*70}")
    print(f"RESUME Training: {RESUME_EPOCHS} epochs | Small Table Copy-Paste ON")
    print(f"Starting from epoch {start_epoch} | Previous best mAP@50: {prev_map:.4f}")
    print(f"{'='*70}\n")

    for epoch_i in range(1, RESUME_EPOCHS + 1):
        epoch = start_epoch + epoch_i
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)

        # mAP every 5 epochs + first and last
        map_50 = 0.0
        class_ap = {}
        if epoch_i % 5 == 0 or epoch_i == 1 or epoch_i == RESUME_EPOCHS:
            map_50, class_ap = compute_map(model, val_loader, device)

        lr_scheduler.step()
        elapsed = time.time() - t0

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

        map_str = f" | mAP@50: {map_50:.4f}" if map_50 > 0 else ""
        class_str = ""
        if class_ap:
            parts = [f"{cls[:4]}={ap:.2f}" for cls, ap in class_ap.items()]
            class_str = f" ({', '.join(parts)})"
        print(
            f"Epoch {epoch:3d} ({epoch_i}/{RESUME_EPOCHS}) | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            f"{map_str}{class_str} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s"
        )

        # Save best by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "map_50": map_50},
                os.path.join(save_dir, "best_map_model.pth"),
            )
            print(f"  -> Best mAP model saved (mAP@50={map_50:.4f})")

        # Checkpoint every 10
        if epoch_i % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "map_50": map_50},
                os.path.join(save_dir, f"resume_ckpt_epoch_{epoch}.pth"),
            )

        # Save log
        if epoch_i % 5 == 0:
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

    # Save final
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(),
         "val_loss": val_loss, "map_50": map_50},
        os.path.join(save_dir, "resume_final_model.pth"),
    )
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Resume training complete!")
    print(f"  Previous best mAP@50: {prev_map:.4f}")
    print(f"  New best mAP@50:      {best_map:.4f}")
    print(f"  Best val_loss:        {best_val_loss:.4f}")
    print(f"  Models saved:         {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
