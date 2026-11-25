"""
DermaCLIP Backend — FastAPI + WebSocket skin analysis server
Model: ViT-L-14-quickgelu with FusedQKV LoRA (open_clip) → 10-class skin classifier
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dermaclip")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "../best_macro_f1.pt"))
PORT = int(os.environ.get("PORT", 8081))
DEVICE_STR = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = torch.device(DEVICE_STR)

# Default class names (overridden by checkpoint)
CLASS_NAMES = [
    "Acne", "Blackheads", "Whiteheads", "Wrinkles",
    "Pigmentation", "Freckles", "Dark Circles", "Pores",
    "Redness", "Puffy Eyes"
]

# Map class → UI metric id used by the frontend
CLASS_TO_METRIC_ID: dict[str, str] = {
    "Acne":        "acne",
    "Pores":       "pores",
    "Wrinkles":    "wrinkles",
    "Dark Circles":"dark_circles",
    "Pigmentation":"pigmentation",
    "Freckles":    "pigmentation",
    "Blackheads":  "acne",
    "Whiteheads":  "acne",
    "Redness":     "acne",
    "Puffy Eyes":  "dark_circles",
}

SEVERITY_THRESHOLDS = [(0.6, "High"), (0.4, "Moderate"), (0.2, "Low"), (0.0, "Minimal")]

RECOMMENDATIONS: dict[str, list[dict]] = {
    "Acne":        [{"name": "Salicylic Acid Cleanser", "timing": "Morning & Evening", "active": "2% BHA", "icon": "flask"}],
    "Blackheads":  [{"name": "BHA Exfoliant", "timing": "2-3×/week", "active": "2% Salicylic Acid", "icon": "flask"}],
    "Whiteheads":  [{"name": "Gentle Retinol Serum", "timing": "Evening", "active": "0.3% Retinol", "icon": "flask"}],
    "Pores":       [{"name": "Niacinamide Serum", "timing": "Morning", "active": "10% Niacinamide", "icon": "droplets"}],
    "Wrinkles":    [{"name": "Retinol Night Cream", "timing": "Evening", "active": "0.5% Retinol", "icon": "flask"},
                   {"name": "SPF 50 Sunscreen", "timing": "Morning", "active": "Broad Spectrum", "icon": "sun"}],
    "Pigmentation":[{"name": "Vitamin C Serum", "timing": "Morning", "active": "15% Ascorbic Acid", "icon": "flask"},
                   {"name": "SPF 50+ Sunscreen", "timing": "Morning", "active": "Broad Spectrum", "icon": "sun"}],
    "Freckles":    [{"name": "SPF 50+ Sunscreen", "timing": "Every 2 hours", "active": "Broad Spectrum", "icon": "sun"}],
    "Dark Circles":[{"name": "Caffeine Eye Serum", "timing": "Morning & Evening", "active": "5% Caffeine", "icon": "droplets"}],
    "Redness":     [{"name": "Centella Asiatica Cream", "timing": "Morning & Evening", "active": "Cica", "icon": "flask"}],
    "Puffy Eyes":  [{"name": "Cooling Eye Gel", "timing": "Morning", "active": "Green Tea Extract", "icon": "droplets"}],
}

# ---------------------------------------------------------------------------
# Model loader — reuses the exact lora_fused_qkv module from training
# ---------------------------------------------------------------------------

class SkinClassifier:
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.preprocess = None
        self.class_names: list[str] = CLASS_NAMES
        self.text_features: Optional[torch.Tensor] = None
        self.ready = False

    def load(self, checkpoint_path: Path) -> None:
        from lora_fused_qkv import apply_fused_qkv_lora  # exact same module used at training time

        needs_download = not checkpoint_path.exists()
        if not needs_download:
            with open(checkpoint_path, "rb") as f:
                if f.read(1) == b"<":
                    log.warning("Existing file at %s is HTML, not a PyTorch checkpoint; re-downloading …", checkpoint_path)
                    needs_download = True

        if needs_download:
            download_url = os.environ.get("MODEL_DOWNLOAD_URL", "").strip()
            if download_url:
                log.info("Downloading checkpoint to %s from MODEL_DOWNLOAD_URL …", checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                if "drive.google.com" in download_url or "drive.usercontent.google.com" in download_url:
                    import gdown
                    gdown.download(download_url, str(checkpoint_path), quiet=False, fuzzy=True)
                else:
                    import urllib.request
                    urllib.request.urlretrieve(download_url, checkpoint_path)
                log.info("Downloaded checkpoint to %s", checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found (or invalid) at {checkpoint_path} and MODEL_DOWNLOAD_URL is not set."
                )

        log.info("Loading checkpoint %s on %s …", checkpoint_path, DEVICE_STR)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "ViT-L-14-quickgelu")
        pretrained = model_cfg.get("pretrained", "openai")
        lora_rank = int(model_cfg.get("lora_rank", 8))
        lora_alpha = float(model_cfg.get("lora_alpha", 16.0))

        self.class_names = ckpt.get("class_names", CLASS_NAMES)

        # Build base model + preprocessing (same call as training)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Inject LoRA adapters — MUST match training exactly
        apply_fused_qkv_lora(model, rank=lora_rank, alpha=lora_alpha)

        # Load checkpoint weights
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            log.warning("Missing state_dict keys (%d): %s …", len(missing), missing[:5])
        if unexpected:
            log.warning("Unexpected state_dict keys (%d): %s …", len(unexpected), unexpected[:5])

        model = model.to(DEVICE).eval()
        self.model = model
        self.preprocess = preprocess

        # Encode class-name text prompts once and cache
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = [f"A face with {c.lower()}." for c in self.class_names]
        tokens = tokenizer(prompts).to(DEVICE)
        with torch.inference_mode():
            text_feat = model.encode_text(tokens)
            text_feat = F.normalize(text_feat.float(), dim=-1)
        self.text_features = text_feat  # (num_classes, D)

        self.ready = True
        log.info("Model ready — %d classes: %s", len(self.class_names), self.class_names)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return per-class softmax probabilities (0–1, sum≈1)."""
        if not self.ready:
            return {}
        img_tensor = self.preprocess(image).unsqueeze(0).to(DEVICE)
        image_feat = self.model.encode_image(img_tensor, normalize=False)
        image_feat = F.normalize(image_feat.float(), dim=-1)
        logits = image_feat @ self.text_features.T
        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {cls: float(prob) for cls, prob in zip(self.class_names, probs)}


classifier = SkinClassifier()


# ---------------------------------------------------------------------------
# Session store — rolling buffer of recent predictions per session
# ---------------------------------------------------------------------------

class SessionStore:
    def __init__(self, window: int = 10):
        self._window = window
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def push(self, session_id: str, probs: dict[str, float]) -> None:
        self._buffers[session_id].append(probs)

    def averaged(self, session_id: str) -> dict[str, float]:
        buf = list(self._buffers[session_id])
        if not buf:
            return {}
        keys = buf[0].keys()
        return {k: float(np.mean([d[k] for d in buf])) for k in keys}

    def has_data(self, session_id: str) -> bool:
        return bool(self._buffers[session_id])


store = SessionStore()


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

METRIC_SIDE: dict[str, str] = {
    "pores":        "left",
    "wrinkles":     "left",
    "acne":         "left",
    "dark_circles": "right",
    "pigmentation": "right",
}


def severity_label(prob: float) -> str:
    for threshold, label in SEVERITY_THRESHOLDS:
        if prob >= threshold:
            return label
    return "Minimal"


def map_to_semantic(avg_probs: dict[str, float]) -> dict[str, float]:
    """Reproduce the semantic concept aggregation from the original analyzer."""
    acne_v   = avg_probs.get("Acne", 0.0)
    bh_v     = avg_probs.get("Blackheads", 0.0)
    wh_v     = avg_probs.get("Whiteheads", 0.0)
    wrink_v  = avg_probs.get("Wrinkles", 0.0)
    pigm_v   = avg_probs.get("Pigmentation", 0.0)
    freck_v  = avg_probs.get("Freckles", 0.0)
    dc_v     = avg_probs.get("Dark Circles", 0.0)
    pores_v  = avg_probs.get("Pores", 0.0)
    red_v    = avg_probs.get("Redness", 0.0)
    puffy_v  = avg_probs.get("Puffy Eyes", 0.0)

    sem_acne   = float(np.clip(acne_v + 0.5*bh_v + 0.5*wh_v, 0, 1))
    sem_pigm   = float(np.clip(pigm_v + 0.5*freck_v, 0, 1))
    sem_wrink  = float(np.clip(wrink_v, 0, 1))
    sem_dark   = float(np.clip(dc_v + 0.5*puffy_v, 0, 1))
    sem_red    = float(np.clip(red_v, 0, 1))
    sem_tex    = float(np.clip(pores_v + 0.3*bh_v, 0, 1))
    overall    = sem_acne*0.3 + sem_pigm*0.25 + sem_wrink*0.2 + sem_dark*0.15 + sem_red*0.1
    sem_health = float(np.clip(1.0 - overall, 0.05, 1.0))

    return {
        "acne": sem_acne, "pigmentation": sem_pigm, "wrinkles": sem_wrink,
        "dark_circles": sem_dark, "redness": sem_red, "texture": sem_tex,
        "healthy": sem_health,
    }


def build_response(session_id: str) -> dict:
    if not store.has_data(session_id):
        return {
            "has_inference": False,
            "status": "Awaiting analysis …",
            "metrics": [],
            "recommendations": [],
        }

    avg_probs = store.averaged(session_id)
    semantic  = map_to_semantic(avg_probs)

    # Build UI metric callout list
    metric_map: dict[str, dict] = {}
    for cls, prob in avg_probs.items():
        mid  = CLASS_TO_METRIC_ID.get(cls, cls.lower().replace(" ", "_"))
        sev  = severity_label(prob)
        # Score: 100 means no concern, 0 means maximum concern
        score = round((1.0 - prob) * 100)
        if mid not in metric_map or prob > (1.0 - metric_map[mid]["score"] / 100):
            metric_map[mid] = {
                "id":       mid,
                "name":     cls,
                "score":    score,
                "severity": sev,
                "note":     f"{sev} concern",
                "side":     METRIC_SIDE.get(mid, "left"),
            }

    metrics = list(metric_map.values())

    # Overall skin health score
    overall = round(semantic["healthy"] * 100)

    # Convenience fields expected by existing UI template
    acne_prob   = avg_probs.get("Acne", avg_probs.get("Blackheads", 0.0))
    wrinkle_prob= avg_probs.get("Wrinkles", 0.0)
    pigment_prob= avg_probs.get("Pigmentation", avg_probs.get("Freckles", 0.0))

    # Recommendations: collect unique items for top conditions (prob ≥ 0.25)
    recs: list[dict] = []
    seen_recs: set[str] = set()
    sorted_classes = sorted(avg_probs.items(), key=lambda x: -x[1])
    for cls, prob in sorted_classes:
        if prob < 0.25:
            break
        for rec in RECOMMENDATIONS.get(cls, []):
            if rec["name"] not in seen_recs:
                seen_recs.add(rec["name"])
                recs.append(rec)
        if len(recs) >= 5:
            break

    top_issues = [cls for cls, p in sorted_classes if p >= 0.35][:3]
    rec_text = (
        f"Your analysis highlights {', '.join(top_issues).lower()}. "
        "Focus on a gentle routine with targeted actives."
        if top_issues
        else "Your skin looks healthy! Maintain with SPF and hydration daily."
    )

    return {
        "has_inference":     True,
        "status":            "Analysis complete",
        "score":             overall,
        "skin_type":         "Oily" if acne_prob > 0.4 else ("Dry" if overall > 75 else "Combination"),
        "hydration":         "Low" if acne_prob > 0.5 else "Normal",
        "acne_severity":     severity_label(acne_prob),
        "wrinkle_risk":      severity_label(wrinkle_prob),
        "pigmentation_risk": severity_label(pigment_prob),
        "metrics":           metrics,
        "recommendations":   recs,
        "recommendation_text": rec_text,
    }


# ---------------------------------------------------------------------------
# FastAPI lifespan — load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, classifier.load, MODEL_PATH)
    yield


app = FastAPI(title="DermaCLIP API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": classifier.ready, "device": DEVICE_STR}


@app.get("/api/scores")
async def get_scores(session_id: str = "default"):
    return JSONResponse(build_response(session_id))


# ---------------------------------------------------------------------------
# WebSocket — receives JPEG frames as base64, runs inference
# ---------------------------------------------------------------------------

@app.websocket("/ws/frame")
async def ws_frame(websocket: WebSocket, session_id: str = "default"):
    await websocket.accept()
    log.info("WS connected: session=%s", session_id)
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_text()
            if not classifier.ready:
                await websocket.send_text(json.dumps({"error": "Model not ready yet"}))
                continue

            try:
                img_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                log.warning("Frame decode error: %s", exc)
                continue

            try:
                probs    = await loop.run_in_executor(None, classifier.predict, image)
                store.push(session_id, probs)
                response = build_response(session_id)
                await websocket.send_text(json.dumps(response))
            except Exception as exc:
                log.error("Inference error: %s", exc, exc_info=True)

    except WebSocketDisconnect:
        log.info("WS disconnected: session=%s", session_id)
    except Exception as exc:
        log.error("WS error: %s", exc)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
