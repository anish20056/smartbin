"""
Smart Bin Waste Classifier
Uses Vision Transformer (ViT) from Hugging Face for 3-class waste classification:
  - Recyclable
  - Compost
  - Landfill

Architecture:
  - Backbone: google/vit-base-patch16-224 (pretrained on ImageNet-21k)
  - Head:     Custom MLP classifier (768 → 256 → 3)
  - Training: Fine-tuned with class-weighted cross-entropy (handles imbalanced data)
"""

import torch
import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor
from PIL import Image
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────
CLASSES = ["Recyclable", "Compost", "Landfill"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Contamination keywords used to flag ambiguous/contaminated items
CONTAMINATION_FLAGS = {
    "Recyclable": "Contains food residue — please rinse before recycling.",
    "Compost":    "Ensure no plastics or metals mixed in.",
    "Landfill":   "Non-recyclable waste — no action needed.",
}

# ── Model Definition ───────────────────────────────────────────────────────────
class WasteViTClassifier(nn.Module):
    """
    Fine-tunable ViT backbone + custom classification head.

    The ViT splits the input image into 16×16 patches, embeds them as tokens,
    and passes them through 12 transformer encoder layers. The [CLS] token
    output (768-dim) is fed into our 3-class MLP head.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        # Load pretrained ViT backbone
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            add_pooling_layer=False,   # we'll pool [CLS] ourselves
        )
        hidden_size = self.vit.config.hidden_size  # 768

        # Classification head: 768 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) normalised image tensor
        Returns:
            logits: (B, num_classes)
        """
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_token)
        return logits


# ── Inference Engine ───────────────────────────────────────────────────────────
class WasteClassifierInference:
    """
    Wraps WasteViTClassifier for single-image and batch inference.
    Handles preprocessing, device placement, and confidence scoring.
    """

    MODEL_ID = "google/vit-base-patch16-224"

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")

        # Image processor (normalises to ViT's expected stats)
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)

        # Build model
        self.model = WasteViTClassifier(num_classes=len(CLASSES))

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            logger.warning(
                "No checkpoint provided — running with pretrained ViT backbone only. "
                "Predictions are based on fine-tuned head random weights (for demo/testing). "
                "Train the model using trainer.py before production use."
            )

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        # Support both raw state_dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """
        Classify a single PIL image.

        Returns a dict with:
          - label:        predicted class name
          - confidence:   probability of the predicted class (0–1)
          - probabilities: dict of all class probabilities
          - contamination_tip: actionable disposal advice
          - is_uncertain: True if max confidence < 0.55 (ambiguous item)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # Forward pass
        logits = self.model(pixel_values)                        # (1, 3)
        probs  = F.softmax(logits, dim=-1).squeeze(0).cpu()     # (3,)

        pred_idx    = probs.argmax().item()
        confidence  = probs[pred_idx].item()
        label       = IDX_TO_CLASS[pred_idx]

        return {
            "label":             label,
            "confidence":        round(confidence, 4),
            "probabilities":     {
                cls: round(probs[i].item(), 4) for i, cls in IDX_TO_CLASS.items()
            },
            "contamination_tip": CONTAMINATION_FLAGS[label],
            "is_uncertain":      confidence < 0.55,
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """Batch predict for a list of PIL images."""
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values)
        probs  = F.softmax(logits, dim=-1).cpu()

        results = []
        for i in range(len(images)):
            pred_idx   = probs[i].argmax().item()
            confidence = probs[i][pred_idx].item()
            label      = IDX_TO_CLASS[pred_idx]
            results.append({
                "label":             label,
                "confidence":        round(confidence, 4),
                "probabilities":     {
                    cls: round(probs[i][j].item(), 4) for j, cls in IDX_TO_CLASS.items()
                },
                "contamination_tip": CONTAMINATION_FLAGS[label],
                "is_uncertain":      confidence < 0.55,
            })
        return results
