"""
Phase 2 — Optuna-tuned XGBoost training, tracked in MLflow.
Run: python3 src/train.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # headless plotting, no display needed
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_FEATURES = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract_type",
    "paperless_billing", "payment_method",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def load_data(path: str):
    df = pd.read_csv(path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    X_train, X_test, y_train, y_test = load_data("data/spark_cleaned_churn.csv")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance -> negative: {neg}, positive: {pos}, scale_pos_weight: {scale_pos_weight:.3f}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local SQLite store — the filesystem
    # store ("./mlruns") is now in MLflow's maintenance mode as of recent
    # versions, so SQLite is the standard lightweight local alternative that
    # doesn't require running a separate tracking server.
    mlflow.set_experiment("churn_prediction")

    best_run_id = {"value": None}
    best_auc = {"value": -1.0}

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        with mlflow.start_run(nested=True) as run:
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                random_state=42,
            )
            model.fit(X_train_t, y_train)
            probs = model.predict_proba(X_test_t)[:, 1]
            auc = roc_auc_score(y_test, probs)

            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", auc)

            # Feature importance plot -> logged as an MLflow artifact
            fig, ax = plt.subplots(figsize=(8, 6))
            xgb.plot_importance(model, max_num_features=15, ax=ax)
            fig.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)

            mlflow.xgboost.log_model(model, name="model")

            if auc > best_auc["value"]:
                best_auc["value"] = auc
                best_run_id["value"] = run.info.run_id

        return auc

    study = optuna.create_study(direction="maximize", study_name="churn_xgboost_optuna")
    study.optimize(objective, n_trials=30)  # 30 trials keeps this fast for a one-day sprint

    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best MLflow run_id: {best_run_id['value']}")

    # Save run_id to a file so the next phase (registry) can pick up the winner
    with open("data/best_run_id.txt", "w") as f:
        f.write(best_run_id["value"])

    return best_run_id["value"], study.best_value


if __name__ == "__main__":
    main()