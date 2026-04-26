# 🏦 Loan Default Prediction — ML System

> End-to-end machine learning pipeline for predicting the probability of a borrower defaulting on a loan.  
> Includes feature engineering, SMOTE-balanced training, MLflow experiment tracking, SHAP explainability, data drift monitoring, and a FastAPI inference service — all containerised with Docker.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Dataset & Features](#4-dataset--features)
5. [ML Pipeline](#5-ml-pipeline)
6. [Model Results](#6-model-results)
7. [Data Drift Detection](#7-data-drift-detection)
8. [SHAP Explainability](#8-shap-explainability)
9. [FastAPI Inference Service](#9-fastapi-inference-service)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Setup & Installation](#11-setup--installation)
12. [Running Locally (VS Code)](#12-running-locally-vs-code)
13. [Running with Docker](#13-running-with-docker)
14. [API Reference](#14-api-reference)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

Traditional loan risk assessment using credit scores and income metrics alone fails to capture complex non-linear relationships in borrower behaviour. This project builds a machine learning system that:

- Predicts the **probability of loan default** for any applicant
- Handles **class imbalance** (9,845 defaults vs 112,011 non-defaults) via SMOTE oversampling
- Handles **missing values and outliers** through imputation and winsorisation
- Provides **model explanations** at both global and local levels using SHAP
- Monitors **data drift** using PSI, CSI, and KS tests logged to MLflow
- Serves predictions via a **production-ready REST API** (FastAPI + uvicorn)
- Is fully **reproducible via Docker** — one command runs the entire lifecycle

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Container                          │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ model.py │ → │ drift.py │ → │explain.py│ → │  app.py    │  │
│  │          │   │          │   │          │   │  FastAPI   │  │
│  │ Train RF │   │ PSI/CSI  │   │  SHAP    │   │  :8000     │  │
│  │ Train HGB│   │ KS Test  │   │  Plots   │   │            │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────────────┘  │
│       │              │              │                           │
│       └──────────────┴──────────────┘                          │
│                       │                                         │
│              ┌─────────────────┐                                │
│              │  mlruns.db      │  ← SQLite MLflow backend       │
│              │  MLflow UI :5000│                                │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

**Startup sequence** (managed by `entrypoint.sh`):
1. Train 2 MLflow experiments (RF + HistGBM)
2. Run data drift detection → log PSI/CSI/KS metrics
3. Run SHAP analysis → log importance plots as artifacts
4. Start MLflow UI on port `5000`
5. Start FastAPI inference service on port `8000`

---

## 3. Project Structure

```
loan-default/
│
├── app.py                  # FastAPI inference service
├── model.py                # ML pipeline: train, evaluate, log to MLflow
├── drift.py                # Data drift: PSI, CSI, KS tests
├── explain.py              # SHAP explainability: global + local
├── entrypoint.sh           # Docker startup script (runs full pipeline)
│
├── Dockerfile              # Single-stage Python 3.10 image
├── docker-compose.yml      # Compose config with volume persistence
├── requirements.txt        # Pinned Python dependencies
│
├── EDA.ipynb               # Exploratory Data Analysis notebook
│
├── data/
│   ├── dataset.csv         # Raw loan application data (121,856 rows)
│   └── Data_Dictionary.csv # Feature descriptions
│
├── model.pkl               # Saved pipeline (preprocessor + SMOTE + classifier)
├── preprocessor.pkl        # Fitted ColumnTransformer (for SHAP)
└── mlruns.db               # SQLite MLflow tracking database
```

---

## 4. Dataset & Features

**Source:** `data/dataset.csv` — 121,856 loan applications, 39 features + 1 target.

**Class distribution:**
| Class | Count | Percentage |
|---|---|---|
| No Default (0) | 112,011 | 91.9% |
| Default (1) | 9,845 | 8.1% |

### Feature Groups

**Numerical (19 features)**
| Feature | Description |
|---|---|
| `Client_Income` | Monthly income (string → coerced to float) |
| `Credit_Amount` | Total loan amount |
| `Loan_Annuity` | Annual repayment amount |
| `Age_Days` | Client age in days (negative values) |
| `Employed_Days` | Employment duration in days (365243 = pensioner sentinel) |
| `Score_Source_1/2/3` | External credit bureau scores |
| `Population_Region_Relative` | Regional population density |
| `Child_Count` | Number of children |
| `Registration_Days`, `ID_Days` | Document registration ages |
| `Own_House_Age`, `Phone_Change` | Asset and contact stability |
| `Social_Circle_Default`, `Credit_Bureau` | Social risk indicators |

**Categorical (11 features)**
`Client_Income_Type`, `Client_Education`, `Client_Marital_Status`, `Client_Gender`, `Loan_Contract_Type`, `Client_Housing_Type`, `Client_Occupation`, `Type_Organization`, `Accompany_Client`, `Client_Permanent_Match_Tag`, `Client_Contact_Work_Tag`

**Binary (7 features)**
`Car_Owned`, `Bike_Owned`, `Active_Loan`, `House_Own`, `Mobile_Tag`, `Homephone_Tag`, `Workphone_Working`

**Engineered (7 features)**
| Feature | Formula |
|---|---|
| `Age_Years` | `abs(Age_Days) / 365` |
| `Employed_Years` | `abs(Employed_Days) / 365` (NaN for pensioners) |
| `Income_to_Credit` | `Client_Income / Credit_Amount` |
| `Annuity_to_Income` | `Loan_Annuity / Client_Income` |
| `Credit_to_Annuity` | `Credit_Amount / Loan_Annuity` |
| `Employment_to_Age` | `Employed_Years / Age_Years` |
| `Family_per_Income` | `Client_Family_Members / Client_Income` |

---

## 5. ML Pipeline

### Preprocessing

```
Numerical  → Median Imputation → StandardScaler
Categorical → Mode Imputation  → OneHotEncoder (handle_unknown='ignore')
Binary     → Mode Imputation
                    ↓
             ColumnTransformer
                    ↓
                  SMOTE
           (oversamples minority class on training set only)
                    ↓
               Classifier
```

Key design decisions:
- **`engineer_features()` runs before `cap_outliers()`** so engineered features are also winsorised
- **SMOTE runs inside the pipeline** and only on training data — no leakage into validation folds
- **Cross-validation uses pre-transformed data without SMOTE** to give an unbiased CV score
- **Outlier capping** uses 1st–99th percentile winsorisation on all numerical + engineered features

### Experiments

Two experiments are run with different algorithms and hyperparameters:

| Parameter | Experiment 1 (RF) | Experiment 2 (HistGBM) |
|---|---|---|
| Algorithm | `RandomForestClassifier` | `HistGradientBoostingClassifier` |
| Estimators / Iterations | 50 | 100 |
| Max Depth | 8 | 6 |
| Learning Rate | — | 0.05 |
| SMOTE Sampling | 0.5 | 0.6 |
| Class Weight | balanced | balanced |
| CV Folds | 3 | 3 |

---

## 6. Model Results

### Experiment 1 — Random Forest Baseline

```
Run name : RF_baseline_50est_depth8
```

| Metric | Value |
|---|---|
| Accuracy | 0.6921 |
| ROC-AUC | 0.7337 |
| F1 Score | 0.5730 |
| Precision | 0.5281 |
| Recall | 0.6262 |
| F1 Weighted | 0.6978 |
| CV ROC-AUC | 0.7311 ± 0.0065 |

### Experiment 2 — HistGradientBoosting (Tuned) ✅ Best Model

```
Run name : HistGBM_tuned_100iter_depth6_lr005
```

| Metric | Value |
|---|---|
| Accuracy | 0.6986 |
| **ROC-AUC** | **0.7435** |
| F1 Score | 0.5724 |
| Precision | 0.5380 |
| Recall | 0.6115 |
| F1 Weighted | 0.7030 |
| **CV ROC-AUC** | **0.7448 ± 0.0070** |

**HistGBM wins** on both test ROC-AUC (+0.0098) and cross-validated ROC-AUC (+0.0137). The saved `model.pkl` is the HistGBM pipeline.

> ROC-AUC is the primary evaluation metric for this imbalanced classification problem. It measures the model's ability to rank defaulters above non-defaulters regardless of threshold.

---

## 7. Data Drift Detection

**Script:** `drift.py`  
**MLflow run:** `drift_analysis_v1`

Drift is simulated by splitting the dataset into a reference set (70%) and a current set (30%), then measuring distributional shift per feature.

### Methods

| Method | Applied To | Interpretation |
|---|---|---|
| **PSI** (Population Stability Index) | Numerical features | < 0.1 Stable · 0.1–0.2 Moderate · > 0.2 Significant |
| **CSI** (Characteristic Stability Index) | Categorical features | Same thresholds as PSI |
| **KS Test** (Kolmogorov-Smirnov) | Numerical features | p > 0.05 Stable · p 0.01–0.05 Moderate · p < 0.01 Significant |

### Results

All features are **STABLE** — no significant drift detected.

**PSI — Numerical Features**
| Feature | PSI | Status |
|---|---|---|
| Client_Income | 0.0002 | ✅ STABLE |
| Credit_Amount | 0.0003 | ✅ STABLE |
| Loan_Annuity | 0.0005 | ✅ STABLE |
| Age_Days | 0.0008 | ✅ STABLE |
| Employed_Days | 0.0002 | ✅ STABLE |
| Score_Source_2 | 0.0003 | ✅ STABLE |
| Population_Region_Relative | 0.0006 | ✅ STABLE |
| Child_Count | 0.0002 | ✅ STABLE |
| **Average PSI** | **0.0004** | ✅ STABLE |

**CSI — Categorical Features**
| Feature | CSI | Status |
|---|---|---|
| Client_Income_Type | 0.0006 | ✅ STABLE |
| Client_Education | 0.0001 | ✅ STABLE |
| Client_Marital_Status | 0.0000 | ✅ STABLE |
| Client_Gender | 0.0001 | ✅ STABLE |
| Loan_Contract_Type | 0.0001 | ✅ STABLE |
| **Average CSI** | **0.0002** | ✅ STABLE |

**KS Test — Numerical Features (all p > 0.05)**
| Feature | KS Statistic | p-value | Status |
|---|---|---|---|
| Client_Income | 0.0033 | 0.9541 | ✅ STABLE |
| Credit_Amount | 0.0054 | 0.4658 | ✅ STABLE |
| Loan_Annuity | 0.0049 | 0.5866 | ✅ STABLE |
| Age_Days | 0.0070 | 0.1700 | ✅ STABLE |
| Employed_Days | 0.0037 | 0.8911 | ✅ STABLE |
| Score_Source_2 | 0.0069 | 0.1987 | ✅ STABLE |
| **Average KS** | **0.0053** | — | ✅ STABLE |

---

## 8. SHAP Explainability

**Script:** `explain.py`  
**MLflow run:** `shap_explainability_v1`  
**Explainer:** `TreeExplainer` (300 test samples)  
**Metric logged:** Mean Absolute SHAP Value per feature

### Global Feature Importance (Mean Absolute SHAP Value)

| Rank | Feature | Mean Abs SHAP |
|:---:|---|:---:|
| 1 | Score_Source_2 | 0.3169 |
| 2 | Score_Source_3 | 0.3120 |
| 3 | Client_Gender_Female | 0.2444 |
| 4 | Client_Gender_Male | 0.1336 |
| 5 | Score_Source_1 | 0.1119 |
| 6 | Credit_to_Annuity | 0.1052 |
| 7 | Phone_Change | 0.0611 |
| 8 | Car_Owned | 0.0664 |
| 9 | Loan_Annuity | 0.0507 |
| 10 | Employed_Days | 0.0467 |

**Key insight:** External credit bureau scores (`Score_Source_1/2/3`) are the strongest predictors of default, followed by gender and the credit-to-annuity ratio. This aligns with domain knowledge — credit scores directly reflect repayment history.

### Three plots logged as MLflow artifacts:
- **`shap_bar.png`** — Global feature importance bar chart (top 20)
- **`shap_summary.png`** — Beeswarm plot showing direction of feature impact
- **`shap_local.png`** — Waterfall plot for a single prediction (sample 0)

---

## 9. FastAPI Inference Service

**Script:** `app.py`  
**Port:** `8000`  
**Docs:** `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API root — version info and endpoint list |
| `GET` | `/health` | Liveness probe — confirms model is loaded |
| `GET` | `/predict` | Usage instructions (browser-friendly) |
| `POST` | `/predict` | Predict default probability for one applicant |

### Prediction Response

```json
{
  "default_probability": 0.111,
  "prediction": 0,
  "risk_label": "LOW RISK",
  "model_version": "1.0.0"
}
```

**Risk tiers:**
| Probability | Risk Label |
|---|---|
| < 0.30 | LOW RISK |
| 0.30 – 0.60 | MEDIUM RISK |
| ≥ 0.60 | HIGH RISK |

### Verified Test Results

**Low-risk applicant** (high income, good scores, employed, married):
```json
{ "default_probability": 0.111, "prediction": 0, "risk_label": "LOW RISK" }
```

**High-risk applicant** (low income, poor scores, active loan, unemployed):
```json
{ "default_probability": 0.566, "prediction": 1, "risk_label": "MEDIUM RISK" }
```

---

## 10. MLflow Experiment Tracking

**Experiment name:** `Loan_Default_Prediction`  
**Backend:** `sqlite:///mlruns.db`  
**UI:** `http://localhost:5000`

### Runs Summary

| Run Name | Type | Key Metric |
|---|---|---|
| `RF_baseline_50est_depth8` | Training | ROC-AUC: 0.7337 |
| `HistGBM_tuned_100iter_depth6_lr005` | Training | ROC-AUC: 0.7435 |
| `drift_analysis_v1` | Drift | avg_psi: 0.0004 |
| `shap_explainability_v1` | XAI | 25 SHAP metrics + 3 plots |

Every run logs: parameters, all evaluation metrics, model artifact, and run tags. The model is also registered in the **MLflow Model Registry** under `LoanDefaultPredictor`.

---

## 11. Setup & Installation

### Prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org/downloads) |
| Docker Desktop | Latest | [docker.com](https://docker.com/products/docker-desktop) |
| VS Code | Latest | [code.visualstudio.com](https://code.visualstudio.com) |

### VS Code Extensions (recommended)

- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **Docker** (Microsoft)

### Clone / Unzip the Project

```bash
# If using git
git clone <repo-url>
cd loan-default

# Or unzip and open folder in VS Code
```

### Folder structure required before running

```
loan-default/
├── data/
│   ├── dataset.csv          ← required
│   └── Data_Dictionary.csv
├── app.py
├── model.py
├── drift.py
├── explain.py
├── EDA.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── entrypoint.sh
```

---

## 12. Running Locally (VS Code)

### Step 1 — Create Virtual Environment

Open the VS Code terminal (`Ctrl+` ` `) and run:

```bash
# Create venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `venv`.

### Step 2 — Run EDA Notebook

Open `EDA.ipynb` → click **Select Kernel** (top right) → choose `venv` → **Run All**.

### Step 3 — Train Models

```bash
python model.py
```

Expected output:
```
===== Experiment 1: Random Forest (baseline) =====
  [Subsampled: 9,845 defaults + 20,000 non-defaults | ratio=0.49]
  Training random_forest...
  Running 3-fold CV (no SMOTE)...
  ====================================================
  Run: RF_baseline_50est_depth8
  ====================================================
  accuracy              : 0.6921
  roc_auc               : 0.7337
  f1_score              : 0.5730
  ...
  Model saved to model.pkl ✓

===== Experiment 2: Histogram Gradient Boosting (tuned) =====
  ...
  roc_auc               : 0.7435
  cv_roc_auc_mean       : 0.7448 +/- 0.0070
  Model saved to model.pkl ✓
```

### Step 4 — Run Drift Detection

```bash
python drift.py
```

Expected output:
```
DATA DRIFT REPORT
Reference size : 85,299 | Current size: 36,557

--- PSI (Numerical Features) ---
  Client_Income        PSI=0.0002  [STABLE]
  ...

SUMMARY
  Average PSI : 0.0004  [STABLE]
  Average CSI : 0.0002  [STABLE]
  Average KS  : 0.0053
Drift metrics logged to MLflow ✓
```

### Step 5 — Run SHAP Analysis

```bash
python explain.py
```

Expected output:
```
Computing SHAP values for HistGradientBoostingClassifier on 500 samples...

Top 10 SHAP Features:
Score_Source_2          0.3169
Score_Source_3          0.3120
Client_Gender_Female    0.2444
...
SHAP analysis (global + local) logged to MLflow ✓
```

### Step 6 — View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

Open **http://localhost:5000** — you'll see the `Loan_Default_Prediction` experiment with all 4 runs, metrics, and SHAP artifact images.

### Step 7 — Start FastAPI Service

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** to use the interactive Swagger UI.

---

## 13. Running with Docker

### Prerequisites
- Docker Desktop must be running

### Build and Run

```bash
# Build image and start all services
docker compose up --build
```

This automatically runs the full pipeline: train → drift → SHAP → serve.

Wait for the line:
```
Starting FastAPI inference service on port 8000...
  API docs: http://localhost:8000/docs
  MLflow:   http://localhost:5000
```

Then open:
- **http://localhost:8000/docs** — FastAPI Swagger UI
- **http://localhost:5000** — MLflow Experiment Tracker

### Other Docker Commands

```bash
# Run with existing image (no rebuild)
docker compose up

# Stop and remove containers
docker compose down

# View live logs
docker compose logs -f

# Use your own dataset instead of the baked-in one
docker run -v /your/path/to/data:/app/data -p 8000:8000 -p 5000:5000 loan-default-predictor
```

### Health Check

Docker will poll `GET /health` every 30 seconds (with 120s startup grace period). The container is marked healthy once the model is loaded and the API responds.

---

## 14. API Reference

### POST /predict

**Request body** (all fields shown — optional fields can be omitted or sent as `null`):

```json
{
  "Client_Income": 180000,
  "Car_Owned": 1,
  "Bike_Owned": 0,
  "Active_Loan": 0,
  "House_Own": 1,
  "Child_Count": 1,
  "Credit_Amount": 200000,
  "Loan_Annuity": 10000,
  "Accompany_Client": "Unaccompanied",
  "Client_Income_Type": "Commercial associate",
  "Client_Education": "Higher education",
  "Client_Marital_Status": "Married",
  "Client_Gender": "Female",
  "Loan_Contract_Type": "Cash loans",
  "Client_Housing_Type": "House / apartment",
  "Population_Region_Relative": 0.035,
  "Age_Days": -16000,
  "Employed_Days": -4000,
  "Registration_Days": -5000,
  "ID_Days": -2000,
  "Own_House_Age": 10.0,
  "Mobile_Tag": 1,
  "Homephone_Tag": 1,
  "Workphone_Working": 1,
  "Client_Occupation": "Managers",
  "Client_Family_Members": 3,
  "Cleint_City_Rating": 2,
  "Application_Process_Day": 2,
  "Application_Process_Hour": 10.0,
  "Score_Source_1": 0.72,
  "Score_Source_2": 0.68,
  "Score_Source_3": 0.75,
  "Social_Circle_Default": 0.0,
  "Phone_Change": 500.0,
  "Credit_Bureau": 0.0
}
```

**Response:**

```json
{
  "default_probability": 0.111,
  "prediction": 0,
  "risk_label": "LOW RISK",
  "model_version": "1.0.0"
}
```

**Notes on input values:**
- `Age_Days` and `Employed_Days` are negative integers (days before today)
- `Employed_Days = 365243` means pensioner (handled as NaN internally)
- Binary fields (`Car_Owned`, etc.) accept `0` or `1`
- All optional fields default to `null` and are imputed by the pipeline

---

## 15. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥2.1, <2.3 | Data manipulation |
| `numpy` | ≥1.26, <2.0 | Numerical operations |
| `scipy` | ≥1.13, <2.0 | KS test (drift) |
| `scikit-learn` | ≥1.4, <1.6 | ML pipeline, models, metrics |
| `imbalanced-learn` | ≥0.12, <0.13 | SMOTE oversampling |
| `mlflow` | ≥2.16, <3.0 | Experiment tracking & model registry |
| `shap` | ≥0.45, <0.48 | Model explainability |
| `fastapi` | ≥0.111, <0.120 | REST API framework |
| `uvicorn` | ≥0.30, <0.35 | ASGI server |
| `pydantic` | ≥2.7, <3.0 | Request/response validation |
| `joblib` | ≥1.4, <2.0 | Model serialisation |
| `matplotlib` | ≥3.9, <4.0 | SHAP plots |
| `seaborn` | ≥0.13, <0.14 | EDA visualisations |
| `jupyter` | ≥1.0 | EDA notebook |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not activated | Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) |
| `data/dataset.csv not found` | Missing data folder | Ensure `data/` folder is in the project root with `dataset.csv` inside |
| `model.pkl not found` on API start | Model not trained yet | Run `python model.py` before starting uvicorn |
| `Port 8000 already in use` | Another process on the port | Run `lsof -i :8000` then `kill <PID>` (Mac/Linux) |
| `MLflow UI shows no runs` | Wrong tracking URI | Use `--backend-store-uri sqlite:///mlruns.db` — must match your project root |
| Docker `connection refused` | Docker Desktop not running | Open Docker Desktop and wait for it to start before running `docker compose up` |
| `SMOTE ValueError` | Sampling strategy < current ratio | Already fixed in `model.py` — ratio is checked before SMOTE runs |
| `sparse_output` TypeError | Old sklearn version | Upgrade: `pip install scikit-learn>=1.4` |
