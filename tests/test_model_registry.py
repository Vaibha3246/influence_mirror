import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/"
    )
)

@pytest.mark.parametrize(
    "model_name, preferred_stage",
    [
        ("yt_chrome_plugin_model", "Staging"),
    ]
)
def test_load_latest_model_from_registry(model_name, preferred_stage):
    try:
        client = MlflowClient()
    except MlflowException as e:
        pytest.fail(f"Failed to connect to MLflow server: {e}")

    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException as e:
        pytest.fail(f"Error fetching model versions: {e}")

    if not versions:
        pytest.fail(f"No versions found for model '{model_name}'. Check MLflow registry or tracking URI.")

    # Try preferred stage first, fallback to any available stage
    stage_versions = [v for v in versions if v.current_stage == preferred_stage]
    if not stage_versions:
        stage_versions = versions
        print(f"WARNING: No model in '{preferred_stage}' stage. Using latest available stage.")

    # Pick the latest version by version number
    latest_version = sorted(stage_versions, key=lambda v: int(v.version))[-1]
    model_uri = f"models:/{model_name}/{latest_version.version}"

    # Metadata checks
    try:
        model_info = mlflow.models.get_model_info(model_uri)
    except MlflowException as e:
        pytest.fail(f"Failed to load model info: {e}")

    assert model_info is not None, "Model info is None"
    assert model_info.signature is not None, "Model signature is missing"
    assert "lightgbm" in model_info.flavors or "python_function" in model_info.flavors, \
        f"Expected flavors not found in model: {model_info.flavors}"
