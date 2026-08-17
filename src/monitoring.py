"""
Phase 5b — Population Stability Index (PSI) drift monitoring.

Compares the distribution of features in RECENT production predictions
(logged to MySQL by the API) against the ORIGINAL training distribution.
A high PSI means production data has drifted away from what the model was
trained on -- a signal the model may need retraining.

Run: python3 src/monitoring.py
"""

import json
import numpy as np
import pandas as pd
import mysql.connector

from db_logger import DB_CONFIG

# PSI is normally computed on NUMERIC features (bucketed into deciles).
# Categorical drift is a separate, simpler comparison (% share per category).
NUMERIC_DRIFT_FEATURES = ["tenure", "monthly_charges", "total_charges"]


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )  across buckets.

    Interpretation (industry standard thresholds):
      PSI < 0.1        -> no significant shift
      0.1 <= PSI < 0.25 -> moderate shift, worth watching
      PSI >= 0.25       -> significant drift, investigate / consider retraining
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.unique(np.percentile(expected, breakpoints))
    if len(bucket_edges) < 3:
        return 0.0  # not enough variation to bucket meaningfully

    bucket_edges[0] -= 1e-6
    bucket_edges[-1] += 1e-6

    expected_counts = pd.cut(expected, bucket_edges).value_counts(normalize=True).sort_index()
    actual_counts = pd.cut(actual, bucket_edges).value_counts(normalize=True).sort_index()

    expected_pct = expected_counts.reindex(expected_counts.index, fill_value=0).replace(0, 1e-6)
    actual_pct = actual_counts.reindex(expected_counts.index, fill_value=0).replace(0, 1e-6)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_verdict(psi: float) -> str:
    if psi < 0.1:
        return "no significant shift"
    elif psi < 0.25:
        return "moderate shift -- monitor"
    else:
        return "significant drift -- investigate/retrain"


def load_production_features() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT features, created_at FROM predictions")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    features_df = pd.json_normalize([json.loads(r["features"]) for r in rows])
    features_df["created_at"] = [r["created_at"] for r in rows]
    return features_df


def run_drift_report():
    training_df = pd.read_csv("data/spark_cleaned_churn.csv")
    production_df = load_production_features()

    print(f"Training rows: {len(training_df)}, Production rows logged: {len(production_df)}\n")

    if len(production_df) < 10:
        print("Not enough production predictions yet for a meaningful PSI calculation "
              "(need at least ~10-20 for stable bucket estimates). Exiting.")
        return {}

    results = {}
    for feature in NUMERIC_DRIFT_FEATURES:
        psi = calculate_psi(training_df[feature], production_df[feature])
        verdict = psi_verdict(psi)
        results[feature] = {"psi": round(psi, 4), "verdict": verdict}
        print(f"  {feature:20s} PSI={psi:.4f}  ->  {verdict}")

    return results


if __name__ == "__main__":
    run_drift_report()