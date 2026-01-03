# register model with input_example fix

import json
import mlflow
import logging
from mlflow.tracking import MlflowClient
import yaml
import numpy as np

# -------------------------
# Load model registry config from params.yaml
# -------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["model_register"]

MLFLOW_TRACKING_URI = params["mlflow_tracking_uri"]
MODEL_NAME = params["model_name"]
EXPERIMENT_NAME = params["mlflow_experiment_name"]
MODEL_VERSION_FILE = params["model_version_file"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -------------------------
# Logging configuration
# -------------------------
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------
# Helper: get latest run
# -------------------------
def get_latest_run_id():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.end_time DESC"],
        max_results=20,
    )

    for run in runs:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        try:
            mlflow.models.get_model_info(model_uri)
            logger.info(f"Found valid model in run: {run_id}")
            return run_id
        except Exception:
            continue

    raise RuntimeError("No MLflow run contains a valid logged model")

# -------------------------
# Register model & save input_example
# -------------------------
def register_model(model_name: str):
    try:
        run_id = get_latest_run_id()
        logger.debug(f"Latest Run ID: {run_id}")

        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered with version: {model_version.version}")

        # Move to Staging environment
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"Model transitioned to 'Staging': {model_name} v{model_version.version}")

        # -------------------------
        # PATCH: Save input_example for pyfunc flavor
        # -------------------------
        try:
            import os
            import numpy as np
            # Load a sample input (same as used during training)
            sample_input_file = os.path.join("data", "features", "features_train.npz")
            sample_input = np.load(sample_input_file)["X"][:1].astype(np.float32)

            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            pyfunc_model.metadata.save_input_example(sample_input)
            logger.info("Input example saved to pyfunc metadata successfully.")
        except Exception as e:
            logger.error(f"Failed to save input example: {e}")

        # Save version info
        version_info = {"model_name": model_name, "model_version": model_version.version}
        with open(MODEL_VERSION_FILE, "w") as f:
            json.dump(version_info, f, indent=4)
        logger.info(f"Model version saved to {MODEL_VERSION_FILE}")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise

# -------------------------
# Main
# -------------------------
def main():
    try:
        register_model(MODEL_NAME)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
