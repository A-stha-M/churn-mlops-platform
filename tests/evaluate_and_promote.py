"""
Phase 3b — Evaluation gate: compares @challenger against @champion on a fixed
held-out set, and promotes the challenger ONLY if it wins.

This is the exact script GitHub Actions runs on every push (see
.github/workflows/model-evaluation.yml). It exits with a non-zero status if
anything goes wrong, so a broken pipeline visibly fails in CI rather than
silently doing nothing.

Run: python3 tests/evaluate_and_promote.py
"""

import sys
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # reuse the same feature lists

MODEL_NAME = "churn_xgboost_model"


def load_holdout_set():
    """
    Uses the SAME random_state=42 split as training, so this is a stable,
    reproducible "golden" hold-out set every time this script runs -- not a
    fresh random sample that could vary between CI runs.
    """
    df = pd.read_csv("data/spark_cleaned_churn.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


def evaluate(model, X_test, y_test) -> float:
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    X_test, y_test = load_holdout_set()

    try:
        challenger_version = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    except mlflow.exceptions.MlflowException:
        print("No challenger registered -- nothing to evaluate. Exiting cleanly.")
        return

    challenger_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger")
    challenger_auc = evaluate(challenger_model, X_test, y_test)
    print(f"Challenger (version {challenger_version.version}) ROC-AUC: {challenger_auc:.4f}")

    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        # No champion exists at all -- this challenger becomes champion by default.
        print("No champion exists yet -- promoting challenger unconditionally.")
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version.version)
        return

    # BOOTSTRAP CASE: register_model.py sets @champion == @challenger on the
    # very first-ever registration (nothing to compare against). Comparing a
    # model to itself would always tie and look like a failed promotion --
    # not a real evaluation outcome, so we detect and skip it cleanly instead
    # of treating it as CI failure.
    if challenger_version.version == champion_version.version:
        print(
            f"Challenger and champion are both version {challenger_version.version} "
            "(first-time bootstrap) -- nothing to compare yet. Exiting cleanly."
        )
        return

    champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    champion_auc = evaluate(champion_model, X_test, y_test)
    print(f"Champion  (version {champion_version.version}) ROC-AUC: {champion_auc:.4f}")

    if challenger_auc > champion_auc:
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version.version)
        print(
            f"\nPROMOTED: challenger (AUC={challenger_auc:.4f}) beat "
            f"champion (AUC={champion_auc:.4f}). Version {challenger_version.version} is now @champion."
        )
    else:
        print(
            f"\nNOT PROMOTED: challenger (AUC={challenger_auc:.4f}) did not beat "
            f"champion (AUC={champion_auc:.4f}). @champion remains version {champion_version.version}."
        )
        # Non-zero exit is a DELIBERATE choice for CI: it makes "did not improve"
        # visible as a distinct outcome in GitHub Actions' run history, rather
        # than looking identical to a successful promotion in the logs.
        sys.exit(1)


if __name__ == "__main__":
    main()