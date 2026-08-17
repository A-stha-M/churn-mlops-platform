"""
Exports the current @champion model as a small, self-contained, committable
folder -- decoupled from mlflow.db / mlruns/, which are gitignored and only
exist on your machine.

WHY THIS IS NEEDED FOR DEPLOYMENT:
Locally, serve.py loads the model live from the MLflow registry (sqlite +
local mlruns/ artifacts). A deployed server (Render, etc.) starts from a
fresh git checkout -- it has neither of those files, so it would crash on
startup trying to reach a registry that doesn't exist there.

The fix: export just the ONE model that's currently @champion into a small
folder (deployed_model/), commit THAT to git (a few MB, not the full
multi-trial mlruns/ history), and have serve.py load from it directly when
running in a deployed environment.

Run this once now, and again any time you want to ship a newly-promoted
champion to production.
"""

import shutil
from pathlib import Path

import mlflow

MODEL_NAME = "churn_xgboost_model"
EXPORT_PATH = Path("deployed_model")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    if EXPORT_PATH.exists():
        shutil.rmtree(EXPORT_PATH)

    mlflow.sklearn.save_model(champion, path=str(EXPORT_PATH), serialization_format="cloudpickle")
    size_mb = sum(f.stat().st_size for f in EXPORT_PATH.rglob("*")) / 1e6
    print(f"Exported current @champion -> {EXPORT_PATH}/ ({size_mb:.1f} MB)")
    print("Commit this folder to git -- it's what the deployed API will load.")


if __name__ == "__main__":
    main()