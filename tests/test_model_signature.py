import mlflow
import numpy as np
import tempfile
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri(
    "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/"
)

MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_mlflow_model_signature_ci_safe():
    client = MlflowClient()

    #  Get latest model in Staging
    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    assert versions, "No model found in Staging"

    mv = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{mv.version}"

    #  Load model
    model = mlflow.pyfunc.load_model(model_uri)

    # Load model signature
    model_info = mlflow.models.get_model_info(model_uri)
    sig = model_info.signature
    assert sig is not None, "Model signature missing"

    #  Download sample input from MODEL ARTIFACTS
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path="artifacts/sample_input.npy",
            dst_path=tmpdir,
        )

        sample_input = np.load(local_path)

    #  Validate input shape
    assert sample_input.ndim == 2
    assert sample_input.shape[1] == len(sig.inputs.inputs)

    # Run prediction
    preds = model.predict(sample_input)
    assert len(preds) == sample_input.shape[0]
