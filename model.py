"""
model.py
Loan Default Prediction - ML Pipeline
Handles preprocessing, feature engineering, training, evaluation, and MLflow logging.

Fixes applied:
  1. cap_outliers called BEFORE engineer_features — swapped order: engineer → cap.
  2. SMOTE inside cross_val_score leaks synthetic data into validation folds.
     CV now uses pre-transformed X_train without SMOTE.
  3. classifier.__class__(**clf.get_params()) crashes for HistGBM.
     Replaced with _build_classifier() helper using explicit params.
  4. joblib.dump(preprocessor) saved the unfitted object.
     Now saves pipeline.named_steps["preprocessor"] (fitted).
  5. sparse_output=False deprecated in sklearn >=1.2. Version-guarded with _ohe_kwargs().
  6. MLflow tracking URI explicitly set to ./mlruns so runs/models always appear locally.
  7. mlflow.sklearn.log_model now uses registered_model_name so models appear
     in the MLflow Model Registry ("Models" tab).
  8. Application_Process_Day added to numeric coercion list.
"""

import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings("ignore")

# ─── Paths & Constants ───────────────────────────────────────────────────────

DATA_PATH             = "data/dataset.csv"
MODEL_PATH            = "model.pkl"
PREPROCESSOR_PATH     = "preprocessor.pkl"
MLFLOW_TRACKING_URI   = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT     = "Loan_Default_Prediction"
MODEL_VERSION         = "1.0.0"
REGISTERED_MODEL_NAME = "LoanDefaultPredictor"

# Explicitly point MLflow at ./mlruns so the UI and this script agree on location
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Feature Definitions ────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "Client_Income", "Credit_Amount", "Loan_Annuity",
    "Population_Region_Relative", "Age_Days", "Employed_Days",
    "Registration_Days", "ID_Days", "Own_House_Age",
    "Score_Source_1", "Score_Source_2", "Score_Source_3",
    "Social_Circle_Default", "Phone_Change", "Credit_Bureau",
    "Child_Count", "Client_Family_Members", "Cleint_City_Rating",
    "Application_Process_Hour",
]

CATEGORICAL_FEATURES = [
    "Accompany_Client", "Client_Income_Type", "Client_Education",
    "Client_Marital_Status", "Client_Gender", "Loan_Contract_Type",
    "Client_Housing_Type", "Client_Occupation", "Type_Organization",
    "Client_Permanent_Match_Tag", "Client_Contact_Work_Tag",
]

BINARY_FEATURES = [
    "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own",
    "Mobile_Tag", "Homephone_Tag", "Workphone_Working",
]

ENGINEERED_FEATURES = [
    "Age_Years", "Employed_Years", "Income_to_Credit",
    "Annuity_to_Income", "Credit_to_Annuity", "Employment_to_Age",
    "Family_per_Income",
]

TARGET = "Default"


# ─── Data Loading & Feature Engineering ─────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=["ID"], errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features and coerce mixed-type columns to numeric.
    Must be called BEFORE cap_outliers so engineered cols exist for capping.
    """
    df = df.copy()

    numeric_cols = [
        "Age_Days", "Employed_Days", "Client_Income", "Credit_Amount",
        "Loan_Annuity", "Client_Family_Members", "Registration_Days", "ID_Days",
        "Score_Source_1", "Score_Source_2", "Score_Source_3",
        "Population_Region_Relative", "Social_Circle_Default",
        "Phone_Change", "Credit_Bureau", "Own_House_Age",
        "Application_Process_Hour", "Application_Process_Day", "Child_Count",
        "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own",
        "Mobile_Tag", "Homephone_Tag", "Workphone_Working",
        "Cleint_City_Rating",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Age_Years"] = (df["Age_Days"].abs() / 365).round(2)

    df["Employed_Years"] = df["Employed_Days"].apply(
        lambda x: round(abs(x) / 365, 2) if pd.notna(x) and x != 365243 else np.nan
    )

    df["Income_to_Credit"]   = df["Client_Income"]  / df["Credit_Amount"].replace(0, np.nan)
    df["Annuity_to_Income"]  = df["Loan_Annuity"]   / df["Client_Income"].replace(0, np.nan)
    df["Credit_to_Annuity"]  = df["Credit_Amount"]  / df["Loan_Annuity"].replace(0, np.nan)
    df["Employment_to_Age"]  = df["Employed_Years"] / df["Age_Years"].replace(0, np.nan)
    df["Family_per_Income"]  = df["Client_Family_Members"] / df["Client_Income"].replace(0, np.nan)

    # Replace pensioner sentinel AFTER using raw value for Employed_Years
    df["Employed_Days"] = df["Employed_Days"].replace(365243, np.nan)

    return df


def cap_outliers(
    df: pd.DataFrame,
    cols: list,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorise numeric columns at given quantile bounds."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
            df[col] = df[col].clip(lo, hi)
    return df


# ─── Preprocessing Pipeline ─────────────────────────────────────────────────

def _ohe_kwargs() -> dict:
    """sparse_output replaces sparse in scikit-learn >= 1.2."""
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_version >= (1, 2):
        return {"handle_unknown": "ignore", "sparse_output": False}
    return {"handle_unknown": "ignore", "sparse": False}


def build_preprocessor() -> ColumnTransformer:
    all_num = NUMERICAL_FEATURES + ENGINEERED_FEATURES

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(**_ohe_kwargs())),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline,                            all_num),
            ("cat", cat_pipeline,                            CATEGORICAL_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_classifier(model_type: str, n_estimators: int, max_depth: int,
                       min_samples_split: int, learning_rate: float):
    """
    Construct a fresh, unfitted classifier from explicit params.
    Avoids clf.__class__(**clf.get_params()) which breaks for HistGBM
    due to internal read-only params.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ─── Model Training ──────────────────────────────────────────────────────────

def train_model(
    model_type:        str   = "random_forest",
    n_estimators:      int   = 100,
    max_depth:         int   = 10,
    min_samples_split: int   = 5,
    learning_rate:     float = 0.1,
    test_size:         float = 0.2,
    smote_sampling:    float = 0.3,
    sample_size:       int   = 40000,
    cv_folds:          int   = 3,
    experiment_name:   str   = MLFLOW_EXPERIMENT,
    run_name:          str   = None,
):
    """
    Full ML pipeline: load → engineer → cap_outliers → split →
    preprocess → SMOTE (train only) → train → evaluate → log to MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Fix: set experiment BEFORE starting run
    mlflow.set_experiment(experiment_name)

    # Fix: correct order — engineer first, then cap
    df = load_data()
    df = engineer_features(df)
    df = cap_outliers(df, NUMERICAL_FEATURES + ENGINEERED_FEATURES)

    if sample_size and len(df) > sample_size:
        df_default    = df[df[TARGET] == 1]
        n_no_default  = min(20000, len(df[df[TARGET] == 0]))
        df_no_default = df[df[TARGET] == 0].sample(n=n_no_default, random_state=42)
        df = pd.concat([df_default, df_no_default]).sample(frac=1, random_state=42)
        ratio = len(df_default) / n_no_default
        print(f"  [Subsampled: {len(df_default):,} defaults + "
              f"{n_no_default:,} non-defaults | ratio={ratio:.2f}]")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf          = _build_classifier(model_type, n_estimators, max_depth,
                                     min_samples_split, learning_rate)
    preprocessor = build_preprocessor()

    # Full training pipeline: preprocess → SMOTE → classify
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(sampling_strategy=smote_sampling, random_state=42)),
        ("classifier",   clf),
    ])

    run_name = run_name or f"{model_type}_{n_estimators}est_{max_depth}depth"

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "model_type":              model_type,
            "n_estimators":            n_estimators,
            "max_depth":               max_depth,
            "min_samples_split":       min_samples_split,
            "learning_rate":           learning_rate,
            "smote_sampling_strategy": smote_sampling,
            "test_size":               test_size,
            "sample_size":             sample_size or "full",
            "cv_folds":                cv_folds,
        })

        print(f"  Training {model_type}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "roc_auc":     roc_auc_score(y_test, y_prob),
            "f1_score":    f1_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred),
            "recall":      recall_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)

        # CV WITHOUT SMOTE — no data leakage into validation folds
        print(f"  Running {cv_folds}-fold CV (no SMOTE)...")
        cv_clf = _build_classifier(model_type, n_estimators, max_depth,
                                   min_samples_split, learning_rate)
        X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            cv_clf, X_train_transformed, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std",  cv_scores.std())

        # Log model → appears in MLflow "Models" tab via registered_model_name
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        mlflow.set_tag("model_version", MODEL_VERSION)
        mlflow.set_tag("model_type", model_type)

        print(f"\n{'='*52}")
        print(f"  Run: {run_name}")
        print(f"{'='*52}")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v:.4f}")
        print(f"  {'cv_roc_auc_mean':22s}: {cv_scores.mean():.4f} "
              f"+/- {cv_scores.std():.4f}")
        print()
        print(classification_report(y_test, y_pred,
                                    target_names=["No Default", "Default"]))

        # Save locally — fitted preprocessor (not unfitted)
        joblib.dump(pipeline, MODEL_PATH)
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)
        print(f"  Model saved to {MODEL_PATH}")
        print(f"  Preprocessor saved to {PREPROCESSOR_PATH}")

        return pipeline, X_test, y_test, metrics, run.info.run_id


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n===== Experiment 1: Random Forest (baseline) =====")
    train_model(
        model_type="random_forest",
        n_estimators=50,
        max_depth=8,
        min_samples_split=10,
        smote_sampling=0.5,
        sample_size=40000,
        cv_folds=3,
        run_name="RF_baseline_50est_depth8",
    )

    print("\n===== Experiment 2: Histogram Gradient Boosting (tuned) =====")
    train_model(
        model_type="hist_gradient_boosting",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        smote_sampling=0.6,
        sample_size=40000,
        cv_folds=3,
        run_name="HistGBM_tuned_100iter_depth6_lr005",
    )
