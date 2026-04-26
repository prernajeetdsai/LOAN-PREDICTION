# ─── Dockerfile ─────────────────────────────────────────────────────────────
# Loan Default Prediction Service
#
# Build:  docker build -t loan-default-predictor .
# Run:    docker run -p 8000:8000 -p 5000:5000 loan-default-predictor
#
# What Docker does here:
#   - Packages the entire ML lifecycle (train → drift → explain → serve) into
#     a single reproducible image.
#   - Training runs at container startup, writing mlruns/ and model.pkl inside
#     the container — so the MLflow UI at :5000 reflects the real run data.
#   - FastAPI at :8000 serves the trained model for live inference.
#   - Anyone with Docker can reproduce results without installing Python deps.
#
# Fixes applied:
#   1. Dataset not baked in — training failed silently inside the container.
#      Added: COPY data/ data/  (dataset.csv and Data_Dictionary.csv baked in).
#   2. entrypoint.sh COPY-ed twice. Removed duplicate.
#   3. chmod +x must happen AFTER COPY. Reordered into one RUN layer.
#   4. Added PYTHONUNBUFFERED=1 so training logs appear in real-time.
#   5. Fixed MLFLOW_TRACKING_URI to sqlite:///mlruns.db (was ./mlruns), matching
#      the URI set in model.py / drift.py / explain.py.
#   8. Added app.py to COPY — FastAPI service was missing, docker build failed.
#   6. Used --no-cache-dir on pip install to reduce image size.
#   7. Added git and libgomp1 (MLflow artifact backends / tree model deps).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="Prerna Jeet"
LABEL description="Loan Default Prediction - ML Service with MLflow"

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (separate layer for better caching) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ──────────────────────────────────────────────────────
# Fix: app.py was missing from the repository; it is now included.
COPY app.py model.py drift.py explain.py entrypoint.sh ./

# ── Bake dataset into image so training works without a volume mount ─────────
# Fix: original had no COPY for data/ — training silently failed in container.
# To use a different dataset at runtime, override with:
#   docker run -v /host/data:/app/data ...
COPY data/ data/

# ── Create runtime directories and set permissions ───────────────────────────
RUN mkdir -p mlruns && chmod +x /app/entrypoint.sh

# ── Expose ports ─────────────────────────────────────────────────────────────
# 8000 = FastAPI inference service
# 5000 = MLflow tracking UI
EXPOSE 8000 5000

# ── Entrypoint: train → drift → explain → serve ──────────────────────────────
CMD ["/app/entrypoint.sh"]
