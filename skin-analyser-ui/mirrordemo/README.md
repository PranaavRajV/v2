# DermaCLIP Mirror — Complete Standalone & Local AI Diagnostic Package

This repository contains the complete codebase, fine-tuned model checkpoint, and standalone browser distribution for the **DermaCLIP Clinical Skincare Mirror**.

---

## 📁 Package Contents

```
mirrordemo/
├── backend/                              # Full Python FastAPI Inference Server
│   ├── main.py                           # Multi-session async WebSocket & REST API
│   ├── models/
│   │   └── best_macro_f1.pt              # Fine-tuned DermaCLIP ViT-L-14 LoRA checkpoint (146 MB)
│   ├── dermaclip_analyzer.py             # CLIP ViT-L-14 model loader & feature extractor
│   ├── face_points.py                    # Landmark transposition & 3-point clinical spotlight
│   ├── face_regions.py                   # 9-zone anatomical face region extractor
│   ├── patch_analyzer.py                 # Low-level biometric skin feature analysis
│   ├── scoring_engine.py                 # Clinical score calculation & smoothing
│   ├── llm_recommender.py                # Formulation recommendations (Gemini + fallback)
│   ├── concept_fusion.py                 # Dermatology concept fusion
│   └── requirements.txt                  # Python dependencies
├── frontend/
│   └── dist/
│       ├── mirror_v2.html                # Enhanced interactive WebGL + WebSocket UI
│       └── assets/                       # Bundled static assets
├── index.html                            # Zero-install standalone browser edition (runs on any device)
├── run_local.sh                          # One-click startup script for local AI server
├── Dockerfile                            # Production container build
└── railway.json                          # Cloud deployment configuration
```

---

## 🚀 How to Run

### Method 1: Full AI Backend (Uses fine-tuned `best_macro_f1.pt` model)
1. Ensure Python 3.10+ is installed.
2. Run the startup script:
   ```bash
   ./run_local.sh
   ```
   Or manually:
   ```bash
   cd backend
   pip install -r requirements.txt
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8081
   ```
3. Open your browser to:
   ```
   http://localhost:8081/mirror_v2.html
   ```
4. Click **Start Mirror Diagnostic** to begin live assessment.

---

### Method 2: Zero-Install Standalone Browser Mirror (Works on ANY Device)
- Double-click `index.html` to open it directly in Chrome, Edge, Safari, or Brave.
- No terminal, no Python, and no installations needed.
- Uses client-side WebGL MediaPipe Face Tracking + client-side biometric vision pipeline.
- Automatically connects to the local Python backend if running, or runs in autonomous standalone mode if offline.

---

## 🎯 Clinical Scoring & 3-Point Diagnostic Display

- **Comprehensive 9-Zone Calculation**: Unlike flat patch averaging, the backend analyzes all 9 anatomical facial zones (forehead, glabella, both cheeks, nose/T-zone, chin/jawline, both under-eyes, and temples) and all ~90 candidate landmark positions.
- **Localized Severity Weighting**: Local conditions (acne bumps, wrinkles, pigmentation) use peak-weighted clinical aggregation ($70\% \text{ peak} + 30\% \text{ distribution}$) so breakouts are never diluted by clear skin elsewhere.
- **Top 3 Focal Points Display**: To maintain clean reflection visibility without visual clutter, the mirror displays **only the 3 most critical diagnostic focal points** at any time. Clicking any condition spotlights the 3 points for that condition; tapping a point opens its formulation protocol.
