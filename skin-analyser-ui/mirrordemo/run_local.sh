#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=================================================="
echo " Starting DermaCLIP Mirror Local Server..."
echo " Model: ViT-L-14 LoRA (backend/models/best_macro_f1.pt)"
echo "=================================================="

cd "$SCRIPT_DIR/backend"
python3 -m pip install -q -r requirements.txt || true
echo "Launching FastAPI server on http://localhost:8081 ..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8081
