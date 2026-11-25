# ── DermaCLIP Backend ─────────────────────────────────────────────────────
# Multi-stage Docker build: keeps the final image lean.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps needed by Pillow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM deps AS runtime
WORKDIR /app

# Copy backend source
COPY . .

# Model file is expected at /app/best_macro_f1.pt
# On Render/Railway you can mount it via a persistent disk or bake it in.
# To bake it in (larger image), uncomment the COPY line below and place the
# .pt file next to this Dockerfile before building.
# COPY best_macro_f1.pt /app/best_macro_f1.pt

ENV MODEL_PATH=/app/best_macro_f1.pt
ENV PORT=8081

EXPOSE 8081

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
