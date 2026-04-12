"""
Edge Camera Feed Processor
Captures frames from a webcam using OpenCV, runs the ViT classifier
on each frame, and overlays the classification result in real-time.

This script runs directly on the Smart Bin hardware (Raspberry Pi / Jetson Nano).

Usage:
    python camera_processor.py --camera 0 --interval 1.5
    python camera_processor.py --camera 0 --api_url http://localhost:8000

Two modes:
  - LOCAL:  runs classifier in-process (requires GPU/fast CPU)
  - API:    posts frames to the FastAPI backend (recommended for Pi)
"""

import argparse
import base64
import io
import time
import logging
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Visual overlay constants ───────────────────────────────────────────────────
LABEL_COLORS = {
    "Recyclable": (72, 199, 116),    # green  (BGR)
    "Compost":    (255, 180, 50),    # amber
    "Landfill":   (80,  80, 220),    # red-ish
    "Unknown":    (180, 180, 180),   # grey
}
FONT            = cv2.FONT_HERSHEY_DUPLEX
OVERLAY_ALPHA   = 0.55
BOX_PADDING     = 14


def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Draw classification result as a semi-transparent banner at the bottom
    of the frame. Shows label, confidence bar, and disposal tip.
    """
    h, w = frame.shape[:2]
    label       = result.get("label",       "Unknown")
    confidence  = result.get("confidence",  0.0)
    tip         = result.get("contamination_tip", "")
    is_uncertain= result.get("is_uncertain", False)
    color       = LABEL_COLORS.get(label, LABEL_COLORS["Unknown"])

    # Semi-transparent banner
    banner_h = 120
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    # Label text
    label_text = f"{'[?] ' if is_uncertain else ''}{label.upper()}"
    cv2.putText(frame, label_text,
                (BOX_PADDING, h - banner_h + 38),
                FONT, 1.1, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 2 * BOX_PADDING) * confidence)
    bar_y = h - banner_h + 55
    cv2.rectangle(frame, (BOX_PADDING, bar_y), (w - BOX_PADDING, bar_y + 12), (60, 60, 60), -1)
    cv2.rectangle(frame, (BOX_PADDING, bar_y), (BOX_PADDING + bar_w, bar_y + 12), color, -1)
    cv2.putText(frame, f"{confidence:.0%}",
                (w - BOX_PADDING - 55, bar_y + 11),
                FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Disposal tip
    tip_display = tip[:80] + "…" if len(tip) > 80 else tip
    cv2.putText(frame, tip_display,
                (BOX_PADDING, h - BOX_PADDING),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Corner label box
    cv2.rectangle(frame, (w - 160, 10), (w - 10, 50), color, -1)
    cv2.putText(frame, "SMART BIN", (w - 152, 38),
                FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    return frame


def frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV BGR frame to JPEG base64 for API submission."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── API mode ───────────────────────────────────────────────────────────────────
def classify_via_api(b64: str, api_url: str) -> Optional[dict]:
    try:
        resp = requests.post(
            f"{api_url}/classify/base64",
            json={"image_b64": b64, "filename": "frame.jpg"},
            timeout=3.0,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning(f"API request failed: {e}")
        return None


# ── Local mode ────────────────────────────────────────────────────────────────
def get_local_classifier(checkpoint: Optional[str]):
    from models.classifier import WasteClassifierInference
    return WasteClassifierInference(checkpoint_path=checkpoint)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    local_clf = None
    if not args.api_url:
        logger.info("Running in LOCAL mode (in-process inference)")
        local_clf = get_local_classifier(args.checkpoint)
    else:
        logger.info(f"Running in API mode → {args.api_url}")

    last_classify_time = 0.0
    last_result        = {"label": "Scanning…", "confidence": 0.0,
                          "contamination_tip": "", "is_uncertain": False}
    fps_counter        = []

    logger.info("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Frame capture failed")
            break

        now = time.time()
        fps_counter.append(now)
        fps_counter = [t for t in fps_counter if now - t < 1.0]

        # ── Classify at the configured interval ────────────────────────────────
        if (now - last_classify_time) >= args.interval:
            last_classify_time = now

            if args.api_url:
                b64    = frame_to_b64(frame)
                result = classify_via_api(b64, args.api_url)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result  = local_clf.predict(pil_img)

            if result:
                last_result = result
                logger.info(
                    f"{result['label']} ({result['confidence']:.2%}) | "
                    f"ms={result.get('inference_ms', 'N/A')}"
                )

        # ── Draw overlay ───────────────────────────────────────────────────────
        display = draw_overlay(frame.copy(), last_result)
        cv2.putText(display, f"FPS: {len(fps_counter)}", (10, 28),
                    FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Smart Bin — Real-time Classifier", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera released. Exiting.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Bin live camera classifier")
    parser.add_argument("--camera",     type=int,   default=0,              help="Camera index")
    parser.add_argument("--interval",   type=float, default=1.5,            help="Seconds between classifications")
    parser.add_argument("--api_url",    type=str,   default=None,           help="FastAPI URL (omit for local mode)")
    parser.add_argument("--checkpoint", type=str,   default=None,           help="Path to .pt checkpoint (local mode)")
    args = parser.parse_args()
    run(args)
