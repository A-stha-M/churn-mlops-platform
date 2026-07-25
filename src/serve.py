"""
Phase 4 — FastAPI serving layer with SHAP-based explainability.

Run: uvicorn src.serve:app --reload --port 8000
Then: POST http://localhost:8000/predict  (see example payload in README)
"""

import sys
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, "src")
from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_NAME = "churn_xgboost_model"

# Populated at startup, not per-request -- loading a model on every single
# request would add seconds of latency to what should be a fast API call.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Load the FULL pipeline (preprocessor + classifier) via mlflow.sklearn,
    # NOT mlflow.pyfunc -- we need direct access to the individual pipeline
    # steps below, which the generic pyfunc wrapper doesn't expose.
    full_pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    preprocessor = full_pipeline.named_steps["preprocessor"]
    classifier = full_pipeline.named_steps["classifier"]

    # TreeExplainer is the fast, EXACT SHAP method for tree-based models
    # (XGBoost, LightGBM, random forests) -- don't reach for the generic
    # KernelExplainer here, it's slow and only needed for black-box models.
    explainer = shap.TreeExplainer(classifier)

    ml_models["pipeline"] = full_pipeline
    ml_models["preprocessor"] = preprocessor
    ml_models["explainer"] = explainer
    ml_models["feature_names"] = preprocessor.get_feature_names_out()

    print(f"Loaded '{MODEL_NAME}@champion' and built SHAP explainer.")
    yield
    # --- SHUTDOWN --- (nothing to clean up here, but this is where you would)
    ml_models.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


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

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 2, "monthly_charges": 95.0, "total_charges": 190.0,
                "senior_citizen": 0, "gender": "Female", "partner": "No",
                "dependents": "No", "phone_service": "Yes", "multiple_lines": "No",
                "internet_service": "Fiber optic", "online_security": "No",
                "online_backup": "No", "device_protection": "No", "tech_support": "No",
                "streaming_tv": "Yes", "streaming_movies": "Yes",
                "contract_type": "Month-to-month", "paperless_billing": "Yes",
                "payment_method": "Electronic check",
            }
        }


class ChurnResponse(BaseModel):
    churn_probability: float
    top_drivers: dict[str, float]


def get_top_drivers(input_df: pd.DataFrame, top_n: int = 3) -> dict:
    transformed = ml_models["preprocessor"].transform(input_df)
    row_shap = ml_models["explainer"].shap_values(transformed)[0]
    feature_names = ml_models["feature_names"]

    contributions = sorted(
        zip(feature_names, row_shap), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]
    # sklearn's ColumnTransformer prefixes names with the transformer they
    # came from (e.g. "cat__contract_type_Month-to-month") -- clean, useful
    # internally, but noisy in an API response meant for a human/dashboard
    # to read. Strip the "num__"/"cat__" prefix for a nicer output.
    cleaned = {name.split("__", 1)[-1]: round(float(val), 4) for name, val in contributions}
    return cleaned


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "pipeline" in ml_models}


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest):
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_df = pd.DataFrame([request.model_dump()])

    # NUMERIC_FEATURES/CATEGORICAL_FEATURES order must match training exactly --
    # ColumnTransformer selects by NAME internally so order in the dict doesn't
    # actually matter here, but keeping columns limited to exactly what the
    # model was trained on avoids silently passing through unexpected extras.
    input_df = input_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prob = ml_models["pipeline"].predict_proba(input_df)[0, 1]
    top_drivers = get_top_drivers(input_df)

    return ChurnResponse(churn_probability=round(float(prob), 4), top_drivers=top_drivers)