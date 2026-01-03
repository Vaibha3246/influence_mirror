import pytest
import mlflow
from mlflow.tracking import MlflowClient

@pytest.mark.parametrize(
    "model_name, stage",
    [
        ("yt_chrome_plugin_model", "Staging"),
    ]
)
def test_load_latest_model_from_registry(model_name, stage):
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    stage_versions = [
        v for v in versions if v.current_stage == stage
    ]

    assert len(stage_versions) > 0, f"No model found in {stage} stage"

    latest_version = sorted(
        stage_versions, key=lambda v: int(v.version)
    )[-1]

    model_uri = f"models:/{model_name}/{latest_version.version}"

    # ONLY METADATA CHECKS — 
    model_info = mlflow.models.get_model_info(model_uri)

    assert model_info is not None
    assert model_info.signature is not None
    assert "lightgbm" in model_info.flavors or "python_function" in model_info.flavors
