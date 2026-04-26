"""
drift.py
Data Drift Detection: PSI, CSI, and KS Test
Logs all drift metrics to MLflow for monitoring.

Fixes applied:
  1. MLflow experiment must be set BEFORE mlflow.start_run().
     Moved mlflow.set_experiment() outside the run block.
  2. compute_psi() denominator used bucket count in epsilon term instead of
     array length. Fixed normalisation formula.
  3. compute_cat_psi() only used categories from expected; new categories in
     actual were ignored. Now uses union of both category sets.
  4. Numeric columns not coerced before drift computation (mixed string/float).
     Added pd.to_numeric(errors="coerce").
  5. avg_psi/avg_csi could be NaN if results dict was empty. Added guards.
  6. MLflow tracking URI now explicitly set to ./mlruns (same as model.py)
     so runs appear in the same UI.
"""

import warnings
import numpy as np
import pandas as pd
import mlflow
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH           = "data/dataset.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT   = "Loan_Default_Prediction"

# Explicitly point MLflow at ./mlruns
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ─── PSI ─────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → STABLE
    PSI 0.1–0.2 → MODERATE
    PSI > 0.2  → SIGNIFICANT
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0].astype(float)

    # Fix: epsilon on counts only; denominator is total obs count
    eps          = 1e-6
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct   = (actual_counts   + eps) / (len(actual)   + eps * len(actual_counts))

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


# ─── CSI ─────────────────────────────────────────────────────────────────────

def compute_cat_psi(expected: pd.Series, actual: pd.Series) -> float:
    """
    Characteristic Stability Index for categorical features.
    Fix: uses UNION of categories so new categories in actual are captured.
    """
    all_cats = pd.unique(pd.concat([expected.dropna(), actual.dropna()]))
    if len(all_cats) == 0:
        return 0.0

    eps   = 1e-6
    e_raw = expected.value_counts(normalize=True).reindex(all_cats, fill_value=0.0)
    a_raw = actual.value_counts(normalize=True).reindex(all_cats, fill_value=0.0)

    e_pct = (e_raw + eps) / (e_raw + eps).sum()
    a_pct = (a_raw + eps) / (a_raw + eps).sum()

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# ─── KS ──────────────────────────────────────────────────────────────────────

def compute_ks(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Kolmogorov-Smirnov two-sample test."""
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0}
    ks_stat, ks_pval = stats.ks_2samp(expected, actual)
    return {"ks_statistic": float(ks_stat), "ks_pvalue": float(ks_pval)}


# ─── Severity helpers ────────────────────────────────────────────────────────

def psi_severity(psi: float) -> str:
    if psi < 0.1:
        return "STABLE"
    elif psi < 0.2:
        return "MODERATE"
    return "SIGNIFICANT"


def ks_severity(ks_pval: float) -> str:
    if ks_pval > 0.05:
        return "STABLE"
    elif ks_pval > 0.01:
        return "MODERATE"
    return "SIGNIFICANT"


# ─── Feature lists ───────────────────────────────────────────────────────────

NUMERICAL_COLS = [
    "Client_Income", "Credit_Amount", "Loan_Annuity",
    "Age_Days", "Employed_Days", "Score_Source_2",
    "Population_Region_Relative", "Child_Count",
]

CATEGORICAL_COLS = [
    "Client_Income_Type", "Client_Education",
    "Client_Marital_Status", "Client_Gender",
    "Loan_Contract_Type",
]


# ─── Main ────────────────────────────────────────────────────────────────────

def run_drift_detection(run_name: str = "drift_analysis"):
    """
    Simulate drift by splitting dataset into reference (70%) and current (30%),
    then computing PSI / CSI / KS per feature. All metrics logged to MLflow.
    """
    # Tracking URI already set at module level; only set experiment here.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Fix: coerce numeric drift columns to proper types
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    reference, current = train_test_split(df, test_size=0.3, random_state=42)

    print(f"\n{'='*60}")
    print("DATA DRIFT REPORT")
    print(f"Reference size : {len(reference):,} | Current size: {len(current):,}")
    print(f"{'='*60}\n")

    psi_results: dict = {}
    csi_results: dict = {}
    ks_results:  dict = {}

    # ── PSI for numerical ──
    print("--- PSI (Numerical Features) ---")
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        psi_val = compute_psi(reference[col].values, current[col].values)
        psi_results[col] = psi_val
        print(f"  {col:35s} PSI={psi_val:.4f}  [{psi_severity(psi_val)}]")

    # ── CSI for categorical ──
    print("\n--- CSI (Categorical Features) ---")
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        csi_val = compute_cat_psi(reference[col], current[col])
        csi_results[col] = csi_val
        print(f"  {col:35s} CSI={csi_val:.4f}  [{psi_severity(csi_val)}]")

    # ── KS for numerical ──
    print("\n--- KS Test (Numerical Features) ---")
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        ks = compute_ks(reference[col].values, current[col].values)
        ks_results[col] = ks
        print(f"  {col:35s} KS={ks['ks_statistic']:.4f}  "
              f"p={ks['ks_pvalue']:.4f}  [{ks_severity(ks['ks_pvalue'])}]")

    # ── Log to MLflow ──
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "reference_size": len(reference),
            "current_size":   len(current),
            "method":         "PSI+CSI+KS",
        })

        for col, val in psi_results.items():
            mlflow.log_metric(f"psi_{col}", val)
        for col, val in csi_results.items():
            mlflow.log_metric(f"csi_{col}", val)
        for col, ks in ks_results.items():
            mlflow.log_metric(f"ks_stat_{col}", ks["ks_statistic"])
            mlflow.log_metric(f"ks_pval_{col}", ks["ks_pvalue"])

        # Fix: guard against empty dicts
        avg_psi = float(np.mean(list(psi_results.values()))) if psi_results else 0.0
        avg_csi = float(np.mean(list(csi_results.values()))) if csi_results else 0.0
        avg_ks  = float(np.mean([v["ks_statistic"] for v in ks_results.values()])) \
                  if ks_results else 0.0

        mlflow.log_metrics({
            "avg_psi":          avg_psi,
            "avg_csi":          avg_csi,
            "avg_ks_statistic": avg_ks,
        })

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"  Average PSI : {avg_psi:.4f}  [{psi_severity(avg_psi)}]")
        print(f"  Average CSI : {avg_csi:.4f}  [{psi_severity(avg_csi)}]")
        print(f"  Average KS  : {avg_ks:.4f}")
        print(f"{'='*60}")
        print("Drift metrics logged to MLflow ✓")

    return psi_results, csi_results, ks_results


if __name__ == "__main__":
    run_drift_detection(run_name="drift_analysis_v1")
