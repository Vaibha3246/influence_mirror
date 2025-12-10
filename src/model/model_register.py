# register model

import json
import mlflow
import logging
from mlflow.tracking import MlflowClient
import yaml

# Load model registry config from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["model_register"]

MLFLOW_TRACKING_URI = params["mlflow_tracking_uri"]
MODEL_NAME = params["model_name"]
EXPERIMENT_NAME = params["mlflow_experiment_name"]
MODEL_VERSION_FILE = params["model_version_file"]

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Logging configuration
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


def get_latest_run_id():
    """Fetch the latest MLflow run ID from experiment"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow!")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.end_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No MLflow runs found for model registration!")

    latest_run = runs[0]
    return latest_run.info.run_id


def register_model(model_name: str):
    """Register the latest trained model into MLflow Model Registry"""

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

        # Save version info for deployment stage
        version_info = {"model_name": model_name, "model_version": model_version.version}

        with open(MODEL_VERSION_FILE, "w") as f:
            json.dump(version_info, f, indent=4)

        logger.info(f"Model version saved to {MODEL_VERSION_FILE}")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise


def main():
    try:
        register_model(MODEL_NAME)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
