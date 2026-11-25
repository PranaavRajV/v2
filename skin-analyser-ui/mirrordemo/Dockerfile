# ==============================================================================
# DermaCLIP Mirror — Production Dockerfile for Railway.app
# ==============================================================================
FROM python:3.10-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered output for live logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MEDIAPIPE_DISABLE_GPU=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies required for OpenCV, MediaPipe, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend code, models, configs, and frontend static distribution
COPY backend/ ./backend/
COPY frontend/dist/ ./frontend/dist/

# Health check using FastAPI /api/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/health || exit 1

WORKDIR /app/backend

# Railway dynamically assigns $PORT at runtime
CMD ["sh", "-c", "python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
