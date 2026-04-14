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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import WasteClassifierInference, CLASSES, CONTAMINATION_FLAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global inference engine ────────────────────────────────────────────────────
classifier: Optional[WasteClassifierInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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


class Base64Request(BaseModel):
    image_b64: str
    filename:  str = "frame.jpg"


# ── Helper ─────────────────────────────────────────────────────────────────────
def _pil_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": classifier is not None,
        "device":      classifier.device if classifier else "N/A",
        "classes":     CLASSES,
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
    result = classifier.predict(image)
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
    result = classifier.predict(image)
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
