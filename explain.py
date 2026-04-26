"""
explain.py
XAI (Explainable AI) using SHAP — logs global & local explanations to MLflow.

"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import shap
import joblib
from sklearn.model_selection import train_test_split

from model import (
    load_data, engineer_features, cap_outliers,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    ENGINEERED_FEATURES, TARGET, MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)

warnings.filterwarnings("ignore")

SHAP_BAR_PATH     = "shap_bar.png"
SHAP_SUMMARY_PATH = "shap_summary.png"
SHAP_LOCAL_PATH   = "shap_local.png"


# ─── Feature name extraction ─────────────────────────────────────────────────

def get_feature_names(
    preprocessor,
    num_cols:  list,
    cat_cols:  list,
    bin_cols:  list,
) -> list:
    """
    Reconstruct ordered feature names after ColumnTransformer.transform().
    Fix: use .named_steps["encoder"] to traverse the Pipeline inside the
    "cat" transformer.
    """
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_feature_names = []

    return num_cols + cat_feature_names + bin_cols


# ─── SHAP analysis ───────────────────────────────────────────────────────────

def run_shap_analysis(n_samples: int = 500, run_name: str = "shap_explainability"):
    """
    Compute SHAP values for the saved model and log results to MLflow.
    """
    if not os.path.exists("model.pkl"):
        print("model.pkl not found. Please run model.py first.")
        return None

    # Fix: explicit tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    pipeline     = joblib.load("model.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    df = load_data()
    df = engineer_features(df)
    df = cap_outliers(df, NUMERICAL_FEATURES + ENGINEERED_FEATURES)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fix: reset index for safe positional indexing
    y_test = y_test.reset_index(drop=True)

    X_test_transformed = preprocessor.transform(X_test)

    rng        = np.random.default_rng(42)
    n          = min(n_samples, X_test_transformed.shape[0])
    sample_idx = rng.choice(X_test_transformed.shape[0], size=n, replace=False)
    X_sample   = X_test_transformed[sample_idx]

    model_name = type(classifier).__name__
    print(f"Computing SHAP values for {model_name} on {n} samples...")

    if "RandomForest" in model_name or "GradientBoosting" in model_name:
        explainer   = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample)

        # Fix: handle list-of-arrays (RF) and 3-D array (HistGBM)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
    else:
        explainer = shap.LinearExplainer(classifier, X_sample)
        sv        = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]

    all_num       = NUMERICAL_FEATURES + ENGINEERED_FEATURES
    feature_names = get_feature_names(preprocessor, all_num, CATEGORICAL_FEATURES, BINARY_FEATURES)
    n_features    = X_sample.shape[1]
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Fix: set experiment before start_run
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    temp_files = [SHAP_BAR_PATH, SHAP_SUMMARY_PATH, SHAP_LOCAL_PATH]

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params({
                "model_type":     model_name,
                "shap_samples":   n,
                "explainer_type": (
                    "TreeExplainer"
                    if any(k in model_name for k in ("Tree", "Forest", "Boosting"))
                    else "LinearExplainer"
                ),
            })

            mean_shap          = np.abs(sv).mean(axis=0)
            feature_importance = (
                pd.Series(mean_shap, index=feature_names)
                .sort_values(ascending=False)
            )

            for feat, val in feature_importance.head(20).items():
                safe = (
                    feat.replace(" ", "_")
                        .replace("(", "").replace(")", "")
                        .replace("/", "_")[:50]
                )
                mlflow.log_metric(f"shap_importance_{safe}", float(val))

            # Plot 1: SHAP Bar (global importance)
            plt.figure(figsize=(10, 8))
            top20 = feature_importance.head(20)
            plt.barh(top20.index[::-1], top20.values[::-1], color="steelblue")
            plt.xlabel("Mean |SHAP Value|")
            plt.title(f"SHAP Feature Importance (Top 20) — {model_name}")
            plt.tight_layout()
            plt.savefig(SHAP_BAR_PATH, dpi=120, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(SHAP_BAR_PATH)

            # Plot 2: SHAP Beeswarm (direction of impact)
            # Fix: guard against duplicate OHE feature names
            top20_names  = list(dict.fromkeys(top20.index.tolist()))
            feat_indices = []
            seen         = set()
            for name in top20_names:
                if name in feature_names and name not in seen:
                    feat_indices.append(feature_names.index(name))
                    seen.add(name)
            feat_indices = feat_indices[:20]

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv[:, feat_indices],
                X_sample[:, feat_indices],
                feature_names=[feature_names[i] for i in feat_indices],
                show=False,
                max_display=20,
            )
            plt.title(f"SHAP Summary Plot — {model_name}")
            plt.tight_layout()
            plt.savefig(SHAP_SUMMARY_PATH, dpi=120, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(SHAP_SUMMARY_PATH)

            # Plot 3: Local explanation waterfall for sample 0
            local_sv  = sv[0]
            local_imp = pd.Series(local_sv, index=feature_names).abs().sort_values(ascending=False)
            for feat, val in local_imp.head(5).items():
                safe = feat.replace(" ", "_").replace("/", "_")[:50]
                mlflow.log_metric(f"local_shap_{safe}", float(val))

            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev_arr   = np.atleast_1d(ev)
                base_val = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
            else:
                base_val = float(ev)

            exp_obj = shap.Explanation(
                values=sv[0],
                base_values=base_val,
                data=X_sample[0],
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(exp_obj, max_display=15, show=False)
            plt.title(f"SHAP Local Explanation (sample 0) — {model_name}")
            plt.tight_layout()
            plt.savefig(SHAP_LOCAL_PATH, dpi=120, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(SHAP_LOCAL_PATH)

            print(f"\nTop 10 SHAP Features:")
            print(feature_importance.head(10).to_string())
            print("\nSHAP analysis (global + local) logged to MLflow ✓")

        finally:
            # Fix: always clean up temp files even if log_artifact raises
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

    return feature_importance


if __name__ == "__main__":
    run_shap_analysis(n_samples=500, run_name="shap_explainability_v1")
