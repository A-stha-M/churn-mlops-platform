"""
Phase 5c — Streamlit operational dashboard.

Shows prediction volume, churn probability distribution, and PSI drift
status, all reading live from the same MySQL table the API writes to.

Run: streamlit run src/dashboard.py
"""

import json
import sys

import pandas as pd
import mysql.connector
import streamlit as st

sys.path.insert(0, "src")
from db_logger import DB_CONFIG
from monitoring import calculate_psi, psi_verdict, NUMERIC_DRIFT_FEATURES

st.set_page_config(page_title="Churn Model Dashboard", layout="wide")
st.title("Churn Prediction — Operational Dashboard")


@st.cache_data(ttl=30)  # re-query at most every 30s, not on every widget interaction
def load_predictions() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    features_expanded = pd.json_normalize(df["features"].apply(json.loads))
    df = pd.concat([df.drop(columns=["features"]), features_expanded], axis=1)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=30)
def load_training_data() -> pd.DataFrame:
    return pd.read_csv("data/spark_cleaned_churn.csv")


df = load_predictions()

if df.empty:
    st.warning(
        "No predictions logged yet. Send some requests to the /predict endpoint "
        "(via the FastAPI app) and refresh this page."
    )
    st.stop()

# --- Top-line metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total predictions logged", len(df))
col2.metric("Mean churn probability", f"{df['churn_probability'].mean():.1%}")
high_risk_pct = (df["churn_probability"] > 0.5).mean()
col3.metric("Flagged high-risk (>50%)", f"{high_risk_pct:.1%}")

st.divider()

# --- Prediction volume + probability over time ---
st.subheader("Churn probability over time")
chart_df = df.set_index("created_at")[["churn_probability"]].sort_index()
st.line_chart(chart_df)

st.divider()

# --- PSI drift section ---
st.subheader("Production drift monitoring (PSI)")
st.caption(
    "Compares the distribution of recent production requests against the "
    "original training data. PSI < 0.1: stable. 0.1-0.25: moderate shift. "
    "> 0.25: significant drift, consider retraining."
)

training_df = load_training_data()

if len(df) < 10:
    st.info(f"Only {len(df)} predictions logged so far -- need at least ~10-20 for a stable PSI estimate.")
else:
    psi_cols = st.columns(len(NUMERIC_DRIFT_FEATURES))
    for col, feature in zip(psi_cols, NUMERIC_DRIFT_FEATURES):
        psi = calculate_psi(training_df[feature], df[feature])
        verdict = psi_verdict(psi)
        if psi >= 0.25:
            col.error(f"**{feature}**\n\nPSI = {psi:.3f}\n\n{verdict}")
        elif psi >= 0.1:
            col.warning(f"**{feature}**\n\nPSI = {psi:.3f}\n\n{verdict}")
        else:
            col.success(f"**{feature}**\n\nPSI = {psi:.3f}\n\n{verdict}")

st.divider()

# --- Recent predictions table ---
st.subheader("Recent predictions")
display_cols = ["created_at", "churn_probability", "contract_type", "tenure", "monthly_charges", "top_drivers"]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(df[display_cols].head(20), width="stretch")