import mlflow
import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import os

# Tracking URI 
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/")
)

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

    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None
