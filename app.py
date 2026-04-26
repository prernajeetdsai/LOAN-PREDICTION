"""
app.py
Loan Default Prediction - FastAPI Inference Service
Exposes /predict (GET + POST), /health, and / endpoints.

"""

import os
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import uvicorn

warnings.filterwarnings("ignore")

MODEL_PATH    = "model.pkl"
MODEL_VERSION = "1.0.0"

LOW_RISK_THRESHOLD  = 0.30
HIGH_RISK_THRESHOLD = 0.60

model_pipeline = None


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH}")
    else:
        print("[startup] WARNING: model.pkl not found. Run model.py first.")
    yield
    model_pipeline = None


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Predicts the probability of a borrower defaulting on a loan.\n\n"
        "Run **model.py** first to train and save the model.\n\n"
        "Interactive docs: **/docs** | Health: **/health** | Predict: **POST /predict**"
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    """All fields that the model was trained on."""
    Client_Income:              float          = Field(..., description="Monthly client income")
    Car_Owned:                  int            = Field(..., ge=0, le=1)
    Bike_Owned:                 int            = Field(..., ge=0, le=1)
    Active_Loan:                int            = Field(..., ge=0, le=1)
    House_Own:                  Optional[float]= Field(None)
    Child_Count:                int            = Field(..., ge=0)
    Credit_Amount:              float          = Field(..., description="Loan credit amount")
    Loan_Annuity:               float          = Field(..., description="Annual loan payment")
    Accompany_Client:           Optional[str]  = Field(None)
    Client_Income_Type:         Optional[str]  = Field(None)
    Client_Education:           Optional[str]  = Field(None)
    Client_Marital_Status:      Optional[str]  = Field(None)
    Client_Gender:              Optional[str]  = Field(None)
    Loan_Contract_Type:         Optional[str]  = Field(None)
    Client_Housing_Type:        Optional[str]  = Field(None)
    Population_Region_Relative: Optional[float]= Field(None)
    Age_Days:                   int            = Field(..., description="Age in days (negative)")
    Employed_Days:              int            = Field(..., description="Employment days (negative or 365243 for pensioner)")
    Registration_Days:          Optional[float]= Field(None)
    ID_Days:                    Optional[int]  = Field(None)
    Own_House_Age:              Optional[float]= Field(None)
    Mobile_Tag:                 int            = Field(..., ge=0, le=1)
    Homephone_Tag:              int            = Field(..., ge=0, le=1)
    Workphone_Working:          int            = Field(..., ge=0, le=1)
    Client_Occupation:          Optional[str]  = Field(None)
    Client_Family_Members:      int            = Field(..., ge=1)
    Cleint_City_Rating:         int            = Field(..., ge=1, le=3)
    Application_Process_Day:    int            = Field(..., ge=0, le=6)
    Application_Process_Hour:   Optional[float]= Field(None)
    Client_Permanent_Match_Tag: Optional[str]  = Field(None)
    Client_Contact_Work_Tag:    Optional[str]  = Field(None)
    Type_Organization:          Optional[str]  = Field(None)
    Score_Source_1:             Optional[float]= Field(None)
    Score_Source_2:             Optional[float]= Field(None)
    Score_Source_3:             Optional[float]= Field(None)
    Social_Circle_Default:      Optional[float]= Field(None)
    Phone_Change:               Optional[float]= Field(None)
    Credit_Bureau:              Optional[float]= Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Client_Income": 50000,
                "Car_Owned": 1,
                "Bike_Owned": 0,
                "Active_Loan": 1,
                "Child_Count": 2,
                "Credit_Amount": 120000,
                "Loan_Annuity": 6500,
                "Age_Days": -14000,
                "Employed_Days": -2000,
                "Mobile_Tag": 1,
                "Homephone_Tag": 0,
                "Workphone_Working": 1,
                "Client_Family_Members": 3,
                "Cleint_City_Rating": 2,
                "Application_Process_Day": 2,
            }
        }
    }

    def to_dict(self) -> dict:
        """Pydantic v1/v2 compat."""
        if int(pydantic.VERSION.split(".")[0]) >= 2:
            return self.model_dump()
        return self.dict()  # type: ignore[attr-defined]


class PredictionResponse(BaseModel):
    default_probability: float
    prediction:          int
    risk_label:          str
    model_version:       str = MODEL_VERSION


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_input(data: dict) -> pd.DataFrame:
    """
    Replicate engineer_features() for a single inference row.
    Uses pd.isna() for safe NaN check (works with None and np.nan).
    """
    df = pd.DataFrame([data])

    age_days      = abs(df["Age_Days"].iloc[0])
    employed_days = df["Employed_Days"].iloc[0]

    if pd.isna(employed_days) or employed_days == 365243:
        employed_days_years = np.nan
    else:
        employed_days_years = round(abs(float(employed_days)) / 365, 2)

    df["Age_Years"]         = round(age_days / 365, 2)
    df["Employed_Years"]    = employed_days_years
    df["Income_to_Credit"]  = df["Client_Income"] / df["Credit_Amount"].replace(0, np.nan)
    df["Annuity_to_Income"] = df["Loan_Annuity"]  / df["Client_Income"].replace(0, np.nan)
    df["Credit_to_Annuity"] = df["Credit_Amount"] / df["Loan_Annuity"].replace(0, np.nan)
    df["Employment_to_Age"] = df["Employed_Years"] / (df["Age_Years"].replace(0, np.nan))
    df["Family_per_Income"] = df["Client_Family_Members"] / df["Client_Income"].replace(0, np.nan)
    df["Employed_Days"]     = df["Employed_Days"].replace(365243, np.nan)

    return df


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", summary="API root")
def root():
    return {
        "message":       "Loan Default Prediction API",
        "version":       MODEL_VERSION,
        "endpoints": {
            "docs":    "GET  /docs   — interactive Swagger UI",
            "health":  "GET  /health — model health check",
            "predict": "POST /predict — predict loan default probability",
        },
    }


@app.get("/health", summary="Health check")
def health():
    return {
        "status":       "ok",
        "model_loaded": model_pipeline is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/predict", summary="Predict usage instructions")
def predict_get():
    """
    GET /predict returns usage instructions.
    Use POST /predict with a JSON body to get a prediction.
    The browser always hits GET when you visit a URL — that was causing 405.
    """
    return {
        "message": "Use POST /predict with a JSON body to get a loan default prediction.",
        "docs":    "/docs",
        "example_request": {
            "method":  "POST",
            "url":     "/predict",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "Client_Income": 50000,
                "Car_Owned": 1,
                "Bike_Owned": 0,
                "Active_Loan": 1,
                "Child_Count": 2,
                "Credit_Amount": 120000,
                "Loan_Annuity": 6500,
                "Age_Days": -14000,
                "Employed_Days": -2000,
                "Mobile_Tag": 1,
                "Homephone_Tag": 0,
                "Workphone_Working": 1,
                "Client_Family_Members": 3,
                "Cleint_City_Rating": 2,
                "Application_Process_Day": 2,
            },
        },
        "example_response": {
            "default_probability": 0.1234,
            "prediction": 0,
            "risk_label": "LOW RISK",
            "model_version": MODEL_VERSION,
        },
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict loan default")
def predict(application: LoanApplication):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run model.py first.",
        )

    input_data = application.to_dict()
    df = engineer_input(input_data)

    try:
        prob = float(model_pipeline.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)

        if prob < LOW_RISK_THRESHOLD:
            risk_label = "LOW RISK"
        elif prob < HIGH_RISK_THRESHOLD:
            risk_label = "MEDIUM RISK"
        else:
            risk_label = "HIGH RISK"

        return PredictionResponse(
            default_probability=round(prob, 4),
            prediction=pred,
            risk_label=risk_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
