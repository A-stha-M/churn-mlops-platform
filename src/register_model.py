"""
Phase 3a — Register the best training run into MLflow's Model Registry.
Run: python3 src/register_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn_xgboost_model"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    with open("data/best_run_id.txt") as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from run {run_id} ...")

    registered = mlflow.register_model(model_uri, MODEL_NAME)
    version = registered.version
    print(f"Registered as '{MODEL_NAME}' version {version}")

    # Every newly trained model starts life as a "challenger" -- it has NOT
    # proven itself against the current production model yet. Promotion to
    # "champion" only happens via the evaluation gate (tests/evaluate_and_promote.py),
    # which is exactly what GitHub Actions will run on every push.
    client.set_registered_model_alias(name=MODEL_NAME, alias="challenger", version=version)
    print(f"Set alias 'challenger' -> version {version}")

    # Bootstrap case: if NO version has the "champion" alias yet (this is the
    # very first model ever registered), there's nothing to compare against.
    # In that case, and only that case, we also make this version the champion
    # so the system has a working baseline from day one.
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        print("A 'champion' already exists -- leaving it as-is. "
              "Run tests/evaluate_and_promote.py to decide if the challenger should replace it.")
    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=version)
        print(f"No champion existed yet -- bootstrapped 'champion' -> version {version} too.")


if __name__ == "__main__":
    main()