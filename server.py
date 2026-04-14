"""
Smart Bin FastAPI Backend
Exposes REST endpoints consumed by the Streamlit dashboard or any frontend.

Endpoints:
  POST /classify/image   → classify an uploaded image file
  POST /classify/base64  → classify a base64-encoded image (for webcam frames)
  GET  /health           → service health + model status
  GET  /classes          → returns label list and disposal tips
"""

import base64
import io
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# ── Add project root to path so imports work when run directly ─────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import WasteClassifierInference, CLASSES, CONTAMINATION_FLAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configure Gemini Vision ────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCnUp9Iz6d3I7g_j608pMP81tplAqZTiSo")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── Global inference engine (loaded once at startup) ──────────────────────────
classifier: Optional[WasteClassifierInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, release on shutdown."""
    global classifier
    try:
        logger.info("Loading WasteViT classifier…")
        checkpoint = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
        classifier = WasteClassifierInference(checkpoint_path=checkpoint)
        logger.info("Model ready.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Bin Waste Classifier API",
    description="ViT-based real-time waste classification: Recyclable / Compost / Landfill",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard and any local client to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ────────────────────────────────────────────────────────────
class ClassificationResult(BaseModel):
    label:             str
    confidence:        float
    probabilities:     dict
    contamination_tip: str
    is_uncertain:      bool
    inference_ms:      float
    gemini_description: str = ""


class Base64Request(BaseModel):
    image_b64: str
    filename:  str = "frame.jpg"


# ── Helper ─────────────────────────────────────────────────────────────────────
def _pil_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def gemini_identify(image: Image.Image) -> tuple:
    """
    Use Gemini Vision to identify the object and suggest waste category.
    Returns (description, suggested_label)
    """
    try:
       prompt = """You are a waste classification expert. Look at this image and classify the waste item.

RULES:
- Paper, newspaper, cardboard, books, notebooks, question papers = Recyclable
- Food waste, fruit peels, vegetables, cotton, wool = Compost  
- Plastic bags, styrofoam, broken glass, electronics = Landfill
- Plastic bottles, cans, glass bottles, metal = Recyclable

Respond in this exact format:
OBJECT: <what you see>
CATEGORY: <Recyclable or Compost or Landfill>
REASON: <one sentence why>"""

        response = gemini_model.generate_content([prompt, image])
        text = response.text.strip()

        # Parse response
        lines = {line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
                 for line in text.split("\n") if ":" in line}

        description = lines.get("OBJECT", "Unknown object")
        gemini_label = lines.get("CATEGORY", "").strip()
        reason = lines.get("REASON", "")

        # Validate label
        if gemini_label not in CLASSES:
            gemini_label = None

        full_description = f"{description} — {reason}"
        logger.info(f"Gemini identified: {full_description} → {gemini_label}")
        return full_description, gemini_label

    except Exception as e:
        logger.error(f"Gemini Vision failed: {e}")
        return "Could not identify object", None


def smart_classify(image: Image.Image) -> dict:
    """
    Combine Gemini Vision + ViT classifier for better accuracy.
    - If both agree → high confidence result
    - If they disagree → trust Gemini (it can actually see the object)
    """
    # Run both models
    vit_result = classifier.predict(image)
    gemini_description, gemini_label = gemini_identify(image)

    vit_label = vit_result["label"]
    vit_confidence = vit_result["confidence"]

    # Decision logic
if gemini_label:
    logger.info(f"Gemini decision: {gemini_label} (ViT said: {vit_label})")
    final_label = gemini_label
    final_confidence = 0.90
    vit_result["probabilities"][gemini_label] = final_confidence
else:
    logger.info(f"Gemini failed, using ViT: {vit_label}")
    final_label = vit_label
    final_confidence = vit_confidence

    return {
        "label": final_label,
        "confidence": round(final_confidence, 4),
        "probabilities": vit_result["probabilities"],
        "contamination_tip": CONTAMINATION_FLAGS[final_label],
        "is_uncertain": final_confidence < 0.55,
        "gemini_description": gemini_description,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": classifier is not None,
        "device":      classifier.device if classifier else "N/A",
        "classes":     CLASSES,
        "gemini":      "enabled" if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE" else "not configured",
    }


@app.get("/classes")
def list_classes():
    return {
        "classes": [
            {"label": label, "tip": tip}
            for label, tip in CONTAMINATION_FLAGS.items()
        ]
    }


@app.post("/classify/image", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    raw = await file.read()
    image = _pil_from_bytes(raw)

    t0 = time.perf_counter()
    result = smart_classify(image)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(f"classify/image → {result['label']} ({result['confidence']:.2%}) in {elapsed_ms} ms")
    return ClassificationResult(**result, inference_ms=elapsed_ms)


@app.post("/classify/base64", response_model=ClassificationResult)
async def classify_base64(body: Base64Request):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    b64 = body.image_b64
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string.")

    image = _pil_from_bytes(raw)

    t0 = time.perf_counter()
    result = smart_classify(image)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(f"classify/base64 → {result['label']} ({result['confidence']:.2%}) in {elapsed_ms} ms")
    return ClassificationResult(**result, inference_ms=elapsed_ms)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )
