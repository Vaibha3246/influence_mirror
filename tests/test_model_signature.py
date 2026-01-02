import mlflow
import numpy as np
from mlflow.tracking import MlflowClient


# MLflow tracking server
mlflow.set_tracking_uri(
    "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/"
)

MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_mlflow_model_signature_ci_safe():
    client = MlflowClient()

    #  Get latest model from Staging
    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    assert versions, " No model found in Staging stage"

    mv = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{mv.version}"

    #  Load model
    model = mlflow.pyfunc.load_model(model_uri)

    #  Get model info + signature
    model_info = mlflow.models.get_model_info(model_uri)
    sig = model_info.signature
    assert sig is not None, " Model signature missing"

    #  Load CI-safe input example (saved during training)
    sample_input = np.array(model_info.saved_input_example)

    #  Validate input shape
    assert sample_input.ndim == 2, " Input must be 2D"
    assert sample_input.shape[1] == len(sig.inputs.inputs), (
        " Feature count mismatch with model signature"
    )

    #  Run prediction
    preds = model.predict(sample_input)

    #  Validate output
    assert len(preds) == sample_input.shape[0], (
        "Output length mismatch"
    )
