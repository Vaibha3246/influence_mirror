import pytest
import mlflow
from mlflow.tracking import MlflowClient

@pytest.mark.parametrize(
    "model_name, preferred_stage",
    [
        ("yt_chrome_plugin_model", "Staging"),
    ]
)
def test_load_latest_model_from_registry(model_name, preferred_stage):
    client = MlflowClient()

    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    assert len(versions) > 0, f"No versions found for model {model_name}"

    # Try preferred stage first, fallback to any available stage
    stage_versions = [v for v in versions if v.current_stage == preferred_stage]
    if not stage_versions:
        stage_versions = versions  # fallback
        print(f"WARNING: No model in '{preferred_stage}' stage. Using latest available stage.")

    # Pick the latest version by version number
    latest_version = sorted(stage_versions, key=lambda v: int(v.version))[-1]
    model_uri = f"models:/{model_name}/{latest_version.version}"

    # Metadata checks
    model_info = mlflow.models.get_model_info(model_uri)
    assert model_info is not None
    assert model_info.signature is not None
    assert "lightgbm" in model_info.flavors or "python_function" in model_info.flavors
