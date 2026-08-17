import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, "src")
from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from db_logger import init_db, log_prediction, close_pool

MODEL_NAME = "churn_xgboost_model"
DEPLOYED_MODEL_PATH = Path(__file__).resolve().parent.parent / "deployed_model"
ml_models = {}


def load_champion_pipeline():
    """
    Prefers the committed deployed_model/ folder (what a deployed server has
    available) and only falls back to the live MLflow registry if that
    folder doesn't exist -- which is the normal case for local development,
    where mlflow.db + mlruns/ are present and up to date with every trial.
    """
    if DEPLOYED_MODEL_PATH.exists():
        print(f"Loading model from committed folder: {DEPLOYED_MODEL_PATH}")
        return mlflow.sklearn.load_model(str(DEPLOYED_MODEL_PATH))

    print("No deployed_model/ folder found -- loading live from MLflow registry (local dev mode).")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")


@asynccontextmanager
async def lifespan(app: FastAPI):
    full_pipeline = load_champion_pipeline()
    preprocessor = full_pipeline.named_steps["preprocessor"]
    classifier = full_pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)

    ml_models["pipeline"] = full_pipeline
    ml_models["preprocessor"] = preprocessor
    ml_models["explainer"] = explainer
    ml_models["feature_names"] = preprocessor.get_feature_names_out()

    await init_db()
    print(f"Loaded '{MODEL_NAME}@champion' and built SHAP explainer.")
    yield
    await close_pool()
    ml_models.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)

# Allows the Streamlit dashboard (running on a different port) to call this API
# directly from the browser if needed -- permissive here since this is a local
# demo project, not a public-facing production deployment.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    senior_citizen: int = Field(..., ge=0, le=1)
    gender: str
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract_type: str
    paperless_billing: str
    payment_method: str


class ChurnResponse(BaseModel):
    churn_probability: float
    top_drivers: dict[str, float]


def get_top_drivers(input_df: pd.DataFrame, top_n: int = 3) -> dict:
    transformed = ml_models["preprocessor"].transform(input_df)
    row_shap = ml_models["explainer"].shap_values(transformed)[0]
    feature_names = ml_models["feature_names"]
    contributions = sorted(zip(feature_names, row_shap), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {name.split("__", 1)[-1]: round(float(val), 4) for name, val in contributions}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "pipeline" in ml_models}


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest):
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_df = pd.DataFrame([request.model_dump()])
    input_df = input_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prob = ml_models["pipeline"].predict_proba(input_df)[0, 1]
    top_drivers = get_top_drivers(input_df)

    await log_prediction(request.model_dump(), float(prob), top_drivers)
    return ChurnResponse(churn_probability=round(float(prob), 4), top_drivers=top_drivers)