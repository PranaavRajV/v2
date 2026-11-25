# DermaCLIP — Skin Analysis Smart Mirror

A full-stack skin analysis app: **React/Vite** frontend + **FastAPI** backend powered by a fine-tuned **ViT-L-14 LoRA** model classifying 10 skin conditions in real time from the webcam.

## Project structure

```
untitled folder/
├── best_macro_f1.pt          ← trained model checkpoint (153 MB)
├── render.yaml               ← Render.com deployment config
├── .gitignore
├── backend/
│   ├── main.py               ← FastAPI server (WebSocket + REST)
│   ├── lora_fused_qkv.py     ← LoRA module (must match training)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── railway.toml          ← Railway deployment config
│   └── start.sh
└── skin-analyser-ui/
    ├── src/App.jsx           ← React UI (reads VITE_API_BASE)
    ├── .env.example
    └── ...
```

## Running locally

### 1 — Backend

```bash
cd backend

# Optional: create a venv
python -m venv venv && source venv/bin/activate

pip install -r requirements.txt

# MODEL_PATH must point to the .pt file
MODEL_PATH=../best_macro_f1.pt python main.py
# → listening on http://localhost:8081
```

### 2 — Frontend

```bash
cd skin-analyser-ui

# Copy and optionally edit the env file
cp .env.example .env.local   # defaults to localhost:8081

npm install
npm run dev
# → http://localhost:5173
```

Open the app, click **Enable camera**, and the UI will connect via WebSocket to the backend.

---

## Deploying to Render.com

> **Before you push:** add `best_macro_f1.pt` to the backend folder (or mount it via a Render Disk — see notes in `render.yaml`).

1. Push the repo to GitHub.
2. In the Render dashboard → **New Blueprint** → connect your repo.
3. Render reads `render.yaml` and creates two services:
   - `dermaclip-api` — Docker backend
   - `dermaclip-ui` — static React site
4. After the backend deploys, copy its public URL (e.g. `https://dermaclip-api.onrender.com`).
5. In the `dermaclip-ui` service → **Environment** → set:
   ```
   VITE_API_BASE = https://dermaclip-api.onrender.com
   ```
6. Trigger a manual redeploy of the UI service.

> **Model size note:** The 150 MB `.pt` file exceeds Render's free-tier image cache. Use a Render Disk or the standard plan for production.

---

## Deploying to Railway

Railway deploys only the backend (the frontend can be served by Netlify/Vercel or as a static export from Railway itself).

```bash
cd backend
railway login
railway init
railway up
```

Railway reads `railway.toml`. Set the following environment variable in the Railway dashboard:
```
MODEL_PATH = /app/best_macro_f1.pt
```

Bake the model into the image by uncommenting the `COPY` line in `Dockerfile` and placing `best_macro_f1.pt` next to it:
```bash
cp ../best_macro_f1.pt ./best_macro_f1.pt
railway up
```

After the backend is live, set `VITE_API_BASE` in the frontend's deployment to the Railway public URL and rebuild.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check; returns `model_ready` bool |
| `GET` | `/api/scores?session_id=X` | Latest analysis result for session |
| `WS`  | `/ws/frame?session_id=X` | Send base64-JPEG frames, receive JSON analysis |

### WebSocket frame format
- **Send:** raw base64-encoded JPEG string (no data-URI prefix)
- **Receive:** JSON matching the `/api/scores` schema

### Scores response schema
```json
{
  "has_inference": true,
  "status": "Analysis complete",
  "score": 72,
  "skin_type": "Combination",
  "hydration": "Normal",
  "acne_severity": "Low",
  "wrinkle_risk": "Minimal",
  "pigmentation_risk": "Moderate",
  "metrics": [
    { "id": "acne", "name": "Acne", "score": 85, "severity": "Low", "note": "Low concern", "side": "left" },
    ...
  ],
  "recommendations": [
    { "name": "Niacinamide Serum", "timing": "Morning", "active": "10% Niacinamide", "icon": "droplets" },
    ...
  ],
  "recommendation_text": "..."
}
```

## Model details

- **Architecture:** OpenCLIP ViT-L-14-quickgelu (OpenAI pretrained)
- **Adaptation:** Fused QKV LoRA (rank 8, alpha 16)
- **Classes (10):** Acne, Blackheads, Whiteheads, Wrinkles, Pigmentation, Freckles, Dark Circles, Pores, Redness, Puffy Eyes
- **Inference:** softmax over CLIP similarity logits against per-class text prompts

> AI-generated skin screening estimate — not a medical diagnosis.
