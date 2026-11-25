# ── DermaCLIP Backend ─────────────────────────────────────────────────────
# The model file (best_macro_f1.pt) is NOT committed to git (153 MB).
# It is downloaded at Docker build time from MODEL_DOWNLOAD_URL.
#
# How to set MODEL_DOWNLOAD_URL:
#   1. Upload best_macro_f1.pt to Google Drive → Share → "Anyone with link"
#      Then convert the share URL to a direct download URL:
#      https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t
#   2. Or upload to Hugging Face Hub and use the raw file URL.
#   3. Or set MODEL_DOWNLOAD_URL as a build arg in Render/Railway dashboard.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM deps AS runtime
WORKDIR /app

COPY . .

# Download model at build time.
# Pass MODEL_DOWNLOAD_URL as a Docker build arg in Render/Railway (defaults to hosted checkpoint).
ARG MODEL_DOWNLOAD_URL="https://drive.usercontent.google.com/download?id=1sd8QNsqvKoeihHgGLHudcCFAlNZefB3C&export=download&confirm=t"
RUN if [ -n "$MODEL_DOWNLOAD_URL" ]; then \
        echo "Downloading model from $MODEL_DOWNLOAD_URL …"; \
        curl -fL "$MODEL_DOWNLOAD_URL" -o /app/best_macro_f1.pt; \
        echo "Model downloaded: $(du -sh /app/best_macro_f1.pt)"; \
    else \
        echo "⚠ MODEL_DOWNLOAD_URL not set — model must be mounted at runtime"; \
    fi

ENV MODEL_PATH=/app/best_macro_f1.pt
ENV PORT=8081

EXPOSE 8081

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
