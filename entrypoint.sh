set -euo pipefail

echo "============================================"
echo "  Loan Default Prediction - Starting Up"
echo "============================================"

# ── Guard: dataset must exist ────────────────────────────────────────────────
if [ ! -f "data/dataset.csv" ]; then
    echo ""
    echo "ERROR: data/dataset.csv not found."
    echo "Either:"
    echo "  1. The dataset was not baked into the image (check Dockerfile COPY data/)."
    echo "  2. Mount your data at runtime:"
    echo "       docker run -v /host/path/to/data:/app/data -p 8000:8000 -p 5000:5000 loan-default-predictor"
    exit 1
fi

# ── Step 1: Train two MLflow experiments ────────────────────────────────────
echo ""
echo "[1/4] Training models (2 MLflow experiments)..."
python model.py || { echo "ERROR: model.py failed"; exit 1; }

# ── Step 2: Data drift detection ────────────────────────────────────────────
echo ""
echo "[2/4] Running data drift detection (PSI, CSI, KS)..."
python drift.py || { echo "ERROR: drift.py failed"; exit 1; }

# ── Step 3: SHAP explainability ──────────────────────────────────────────────
echo ""
echo "[3/4] Running SHAP explainability analysis..."
python explain.py || { echo "ERROR: explain.py failed"; exit 1; }

# ── Step 4: MLflow UI (background) ──────────────────────────────────────────
echo ""
echo "[4/4] Starting MLflow UI on port 5000..."
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db &

# ── Step 5: FastAPI (foreground, PID 1 via exec) ─────────────────────────────
echo ""
echo "Starting FastAPI inference service on port 8000..."
echo "  API docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo "  MLflow:   http://localhost:5000"
exec uvicorn app:app --host 0.0.0.0 --port 8000
