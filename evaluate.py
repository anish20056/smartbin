"""
Evaluation Script
Runs the trained model on a test set and reports:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Confusion matrix
  - Top mis-classified examples (for error analysis)

Usage:
    python evaluate.py --data_dir data/test --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classifier import WasteViTClassifier, CLASSES
from trainer import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    model = WasteViTClassifier(num_classes=len(CLASSES))
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_transforms("val"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    logger.info(f"Test samples: {len(dataset)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=-1).cpu()
            preds  = probs.argmax(dim=-1)
            confs  = probs.max(dim=-1).values

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean()
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        digits=4,
    )

    logger.info(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc:.2%})")
    logger.info(f"\nClassification Report:\n{report}")

    # ── Confidence analysis ───────────────────────────────────────────────────
    uncertain_mask = all_confs < 0.55
    uncertain_pct  = uncertain_mask.mean()
    logger.info(f"Uncertain predictions (<55% confidence): {uncertain_pct:.2%}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — Accuracy: {overall_acc:.2%}", fontsize=13, pad=12)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix saved → {cm_path}")
    plt.close()

    # ── Save full report ──────────────────────────────────────────────────────
    summary = {
        "overall_accuracy":      round(float(overall_acc), 6),
        "uncertain_predictions": round(float(uncertain_pct), 6),
        "confusion_matrix":      cm.tolist(),
    }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_dir}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Smart Bin classifier")
    parser.add_argument("--data_dir",   type=str, required=True,             help="Test data folder (ImageFolder format)")
    parser.add_argument("--checkpoint", type=str, required=True,             help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()
    evaluate(args)
