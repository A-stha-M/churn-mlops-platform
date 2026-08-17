"""
Phase 5c (upgraded) — Streamlit dashboard with three tabs:
  1. Make a Prediction -- a real form that calls the live FastAPI /predict
     endpoint, so you can send actual queries from a UI instead of curl/docs.
  2. Operational Overview -- volume, churn rate, PSI drift (same as before).
  3. Database Explorer -- shows exactly which database/table entries are
     landing in, with a live, refreshable row browser.

Run: streamlit run src/dashboard.py
Requires: the FastAPI app (src/serve.py) running separately on port 8000.
"""

import os
import json
import sys

import pandas as pd
import mysql.connector
import requests
import streamlit as st

# IMPORTANT: Streamlit Community Cloud secrets are only accessible via
# st.secrets -- they are NOT automatically exposed as OS environment
# variables. db_logger.DB_CONFIG reads from os.environ at import time, so we
# have to copy any matching secrets into os.environ BEFORE importing it, or
# a deployed dashboard would silently fall back to the localhost defaults
# and fail to connect to the real production database.
try:
    for key in ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE", "API_URL"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # no secrets.toml file at all -- expected for local development

sys.path.insert(0, "src")
from db_logger import DB_CONFIG
from monitoring import calculate_psi, psi_verdict, NUMERIC_DRIFT_FEATURES

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Model Dashboard", layout="wide")
st.title("Churn Prediction — Operational Dashboard")


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


@st.cache_data(ttl=10)
def load_predictions() -> pd.DataFrame:
    conn = get_connection()
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


tab_predict, tab_overview, tab_db = st.tabs(
    ["🔮 Make a Prediction", "📊 Operational Overview", "🗄️ Database Explorer"]
)

# ============================================================
# TAB 1 — Live prediction form, calls the real FastAPI endpoint
# ============================================================
with tab_predict:
    st.subheader("Send a real prediction request")
    st.caption(
        f"This form POSTs to {API_URL}/predict — the actual FastAPI serving "
        "layer — and the result gets logged to MySQL exactly like a real "
        "production request. Make sure `uvicorn src.serve:app` is running."
    )

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("model_loaded"):
            st.success("API is reachable and the model is loaded.")
        else:
            st.warning("API is reachable but the model isn't loaded yet.")
    except requests.exceptions.RequestException:
        st.error(
            f"Can't reach the API at {API_URL}. Start it with: "
            "`uvicorn src.serve:app --port 8000`"
        )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, 840.0)
            senior_citizen = st.selectbox("Senior citizen", [0, 1])
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        with c3:
            online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

        c4, c5 = st.columns(2)
        with c4:
            contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        with c5:
            payment_method = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        payload = {
            "tenure": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
            "senior_citizen": senior_citizen, "gender": gender, "partner": partner,
            "dependents": dependents, "phone_service": phone_service, "multiple_lines": multiple_lines,
            "internet_service": internet_service, "online_security": online_security,
            "online_backup": online_backup, "device_protection": device_protection,
            "tech_support": tech_support, "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
            "contract_type": contract_type, "paperless_billing": paperless_billing,
            "payment_method": payment_method,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            prob = result["churn_probability"]
            st.divider()
            rcol1, rcol2 = st.columns([1, 2])
            with rcol1:
                if prob >= 0.5:
                    st.error(f"### Churn risk: {prob:.1%}")
                else:
                    st.success(f"### Churn risk: {prob:.1%}")
            with rcol2:
                st.write("**Top drivers of this prediction (SHAP values):**")
                for feature, value in result["top_drivers"].items():
                    direction = "increases" if value > 0 else "decreases"
                    st.write(f"- `{feature}` **{direction}** risk (contribution: {value:+.3f})")

            st.caption("This request has been logged to MySQL — check the Database Explorer tab.")
            load_predictions.clear()  # invalidate cache so the new row shows up immediately
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# ============================================================
# TAB 2 — Operational overview (volume, churn rate, PSI drift)
# ============================================================
with tab_overview:
    df = load_predictions()

    if df.empty:
        st.warning("No predictions logged yet. Use the 'Make a Prediction' tab to send one.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total predictions logged", len(df))
        col2.metric("Mean churn probability", f"{df['churn_probability'].mean():.1%}")
        high_risk_pct = (df["churn_probability"] > 0.5).mean()
        col3.metric("Flagged high-risk (>50%)", f"{high_risk_pct:.1%}")

        st.divider()
        st.subheader("Churn probability over time")
        chart_df = df.set_index("created_at")[["churn_probability"]].sort_index()
        st.line_chart(chart_df)

        st.divider()
        st.subheader("Production drift monitoring (PSI)")
        st.caption(
            "Compares recent production requests against the original training "
            "distribution. PSI < 0.1: stable. 0.1-0.25: moderate. > 0.25: significant drift."
        )
        training_df = load_training_data()
        if len(df) < 10:
            st.info(f"Only {len(df)} predictions logged -- need ~10-20+ for a stable PSI estimate.")
        else:
            psi_cols = st.columns(len(NUMERIC_DRIFT_FEATURES))
            for col, feature in zip(psi_cols, NUMERIC_DRIFT_FEATURES):
                psi = calculate_psi(training_df[feature], df[feature])
                verdict = psi_verdict(psi)
                text = f"**{feature}**\n\nPSI = {psi:.3f}\n\n{verdict}"
                if psi >= 0.25:
                    col.error(text)
                elif psi >= 0.1:
                    col.warning(text)
                else:
                    col.success(text)

# ============================================================
# TAB 3 — Database explorer: shows exactly where data is landing
# ============================================================
with tab_db:
    st.subheader("Where this is actually logging to")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        current_db = cursor.fetchone()[0]
        cursor.execute("SELECT @@hostname, @@port;")
        host_info = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        row_count = cursor.fetchone()[0]
        cursor.execute("SHOW TABLES;")
        tables = [t[0] for t in cursor.fetchall()]
        cursor.close()
        conn.close()

        m1, m2, m3 = st.columns(3)
        m1.metric("Connected database", current_db)
        m2.metric("Table", "predictions")
        m3.metric("Row count", row_count)

        st.caption(
            f"Connection: host=`{DB_CONFIG['host']}`, port=`{DB_CONFIG['port']}`, "
            f"user=`{DB_CONFIG['user']}`  |  Tables in `{current_db}`: {', '.join(tables)}"
        )
    except mysql.connector.Error as e:
        st.error(f"Could not connect to MySQL: {e}")
        st.stop()

    st.divider()
    st.subheader("Live table contents")
    if st.button("Refresh"):
        load_predictions.clear()

    df = load_predictions()
    if df.empty:
        st.info("No rows yet.")
    else:
        st.dataframe(
            df[["id", "created_at", "churn_probability", "contract_type", "tenure", "top_drivers"]],
            width="stretch",
        )

        st.divider()
        st.subheader("Raw row inspector")
        selected_id = st.selectbox("Pick a row to see the full logged record", df["id"].tolist())
        row = df[df["id"] == selected_id].iloc[0]
        st.json({
            "id": int(row["id"]),
            "created_at": str(row["created_at"]),
            "churn_probability": float(row["churn_probability"]),
            "top_drivers": row["top_drivers"],
        })