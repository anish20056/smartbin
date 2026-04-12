# Smart Bin — Waste Classification Backend

ViT-based backend for real-time, 3-class waste classification:
**Recyclable · Compost · Landfill**

---

## Architecture

```
Camera Frame (224×224 RGB)
        │
        ▼
┌─────────────────────────────────────┐
│  AutoImageProcessor (ViT stats)     │  normalise & patch-ify
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  ViT-Base/16 Backbone               │  12 transformer encoder layers
│  (google/vit-base-patch16-224)      │  patches: 14×14 grid (196 tokens)
│  → [CLS] token  (768-dim)           │  attention sees the whole image
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Classification Head (MLP)          │
│  LayerNorm → Linear(768→256)        │
│  → GELU → Dropout(0.3)              │
│  → Linear(256→3)                    │
└─────────────────────────────────────┘
        │
        ▼
  Softmax → [Recyclable, Compost, Landfill]
```

### Why ViT over CNN?

| Property | CNN (ResNet) | ViT |
|---|---|---|
| Receptive field | Local (grows slowly) | Global from layer 1 |
| Contamination detection | Misses context | Sees food residue in relation to whole item |
| Transfer learning | Strong | Stronger (ImageNet-21k pre-training) |
| Edge efficiency | Faster | Acceptable with INT8 quantisation |

The attention mechanism lets ViT reason about **relationships** — e.g. a bottle is recyclable, but the attention heads can also "look at" the liquid inside it and flag uncertainty.

---

## Project Structure

```
smartbin/
├── models/
│   └── classifier.py       # WasteViTClassifier + WasteClassifierInference
├── api/
│   └── server.py           # FastAPI REST API
├── trainer.py              # Fine-tuning pipeline
├── evaluate.py             # Metrics + confusion matrix
├── camera_processor.py     # OpenCV live feed overlay
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Organise your images in ImageFolder format:

```
data/
  train/
    Recyclable/   # clean plastic, glass, paper, cans
    Compost/      # fruit peels, food scraps, coffee grounds
    Landfill/     # styrofoam, chip bags, contaminated packaging
  val/
    Recyclable/
    Compost/
    Landfill/
  test/
    ...
```

**Recommended dataset sizes per class:** ≥500 train, ≥100 val.
Public datasets to supplement: TrashNet, TACO, WasteNet.

---

## Training

```bash
# Fine-tune for 20 epochs (GPU recommended)
python trainer.py \
  --data_dir data \
  --output_dir checkpoints \
  --epochs 20 \
  --batch_size 16 \
  --lr 2e-4

# Lighter run for testing (CPU)
python trainer.py --epochs 3 --batch_size 8 --num_workers 0
```

Key training design choices:
- **Differential LR**: backbone gets lr/10, head gets full lr (preserves pretrained features)
- **WeightedRandomSampler**: counteracts class imbalance automatically
- **Label smoothing (0.1)**: reduces overconfidence on messy real-world images
- **Mixed precision (AMP)**: ~2× speedup on GPU with no accuracy loss
- **Cosine annealing with warm restarts**: avoids sharp LR drops

---

## Running the API Server

```bash
# Without a checkpoint (demo mode — random head weights)
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# With a trained checkpoint
MODEL_CHECKPOINT=checkpoints/best_model.pt \
  python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model readiness + device info |
| GET | `/classes` | All class labels + disposal tips |
| POST | `/classify/image` | Upload an image file |
| POST | `/classify/base64` | Submit a base64-encoded frame |

### Example — Upload a file

```bash
curl -X POST http://localhost:8000/classify/image \
  -F "file=@my_bottle.jpg"
```

### Example — Response

```json
{
  "label": "Recyclable",
  "confidence": 0.8923,
  "probabilities": {
    "Recyclable": 0.8923,
    "Compost":    0.0512,
    "Landfill":   0.0565
  },
  "contamination_tip": "Contains food residue — please rinse before recycling.",
  "is_uncertain": false,
  "inference_ms": 47.3
}
```

---

## Live Camera Feed (Edge Device)

```bash
# API mode (recommended for Raspberry Pi — offloads compute to server)
python camera_processor.py --camera 0 --api_url http://localhost:8000

# Local mode (Jetson Nano / laptop with GPU)
python camera_processor.py --camera 0 --checkpoint checkpoints/best_model.pt

# Classify every 2 seconds
python camera_processor.py --camera 0 --interval 2.0 --api_url http://localhost:8000
```

---

## Evaluation

```bash
python evaluate.py \
  --data_dir data/test \
  --checkpoint checkpoints/best_model.pt \
  --output_dir eval_results
```

Outputs:
- `eval_results/confusion_matrix.png`
- `eval_results/eval_summary.json`

---

## Uncertainty Handling

Any prediction with confidence < 55% is flagged as `is_uncertain: true`.
The Streamlit dashboard should prompt the user to manually confirm the bin
when this flag is set — acting as a human-in-the-loop safety net.

---

## Integration with Streamlit Dashboard

The dashboard POSTs webcam frames to `/classify/base64` every ~1.5 seconds:

```python
import requests, base64, cv2

ret, frame = cap.read()
_, buf = cv2.imencode(".jpg", frame)
b64 = base64.b64encode(buf).decode()

resp = requests.post("http://localhost:8000/classify/base64",
                     json={"image_b64": b64})
result = resp.json()
# result["label"], result["confidence"], result["contamination_tip"]
```
