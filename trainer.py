"""
Fine-tuning Pipeline for WasteViTClassifier

Dataset structure expected:
    data/
      train/
        Recyclable/  (plastic bottles, clean cans, paper, cardboard …)
        Compost/     (fruit peels, food scraps, coffee grounds …)
        Landfill/    (styrofoam, contaminated packaging, chip bags …)
      val/
        Recyclable/
        Compost/
        Landfill/

Run:
    python trainer.py --data_dir data --epochs 20 --batch_size 16
"""

import os
import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np

from classifier import WasteViTClassifier, CLASSES, CLASS_TO_IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "google/vit-base-patch16-224"

# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224):
    """
    Training augmentation is aggressive to combat the diverse, uncontrolled
    camera angles and lighting conditions of a real cafeteria deployment.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (ViT was pretrained on these)
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.05),       # rare but realistic (dim lighting)
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1),          # simulate partial occlusion
        ])
    else:  # val / test — no augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Weighted sampler (handles class imbalance) ─────────────────────────────────
def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    class_counts = np.bincount([s[1] for s in dataset.samples])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "train"),
        transform=get_transforms("train"),
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "val"),
        transform=get_transforms("val"),
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    logger.info(f"Class mapping: {train_dataset.class_to_idx}")

    sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = WasteViTClassifier(num_classes=len(CLASSES), dropout=args.dropout)
    model.to(device)

    # Differential learning rates: backbone gets 10× lower lr than head
    backbone_params = list(model.vit.parameters())
    head_params     = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs // 3 + 1, T_mult=2
    )

    # Compute class weights for loss (inverse frequency)
    class_counts = np.bincount([s[1] for s in train_dataset.samples]).astype(np.float32)
    class_weights = torch.tensor(1.0 / (class_counts / class_counts.sum())).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Mixed precision scaler ─────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ── Training ───────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train epoch ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()

            # Gradient clipping prevents explosive gradients in transformer layers
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (step + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        class_correct = [0] * len(CLASSES)
        class_total   = [0] * len(CLASSES)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(images)
                    loss   = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

                for i, (p, l) in enumerate(zip(preds, labels)):
                    class_total[l.item()]   += 1
                    class_correct[l.item()] += (p == l).item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total

        # Per-class accuracy
        per_class_acc = {
            CLASSES[i]: round(class_correct[i] / max(class_total[i], 1), 4)
            for i in range(len(CLASSES))
        }

        logger.info(
            f"\nEpoch {epoch}/{args.epochs} Summary:\n"
            f"  Train Loss : {avg_train_loss:.4f}\n"
            f"  Val Loss   : {avg_val_loss:.4f}\n"
            f"  Val Acc    : {val_acc:.4f}\n"
            f"  Per-class  : {per_class_acc}"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "val_loss":         avg_val_loss,
                "class_to_idx":     train_dataset.class_to_idx,
            }, ckpt_path)
            logger.info(f"  ✅ New best model saved → {ckpt_path}")

    # Save full training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    return history


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT for waste classification")
    parser.add_argument("--data_dir",    type=str,   default="data",         help="Root of train/val splits")
    parser.add_argument("--output_dir",  type=str,   default="checkpoints",  help="Where to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    args = parser.parse_args()

    train(args)
