import mlflow
import os
import pytest
import numpy as np
import tempfile
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/")

MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging" 

def test_mlflow_model_signature_ci_safe():
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    assert versions, " No model in Staging"

    mv = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{mv.version}"

    model = mlflow.pyfunc.load_model(model_uri)

    tmp_dir = tempfile.mkdtemp()

    #  FIX: correct artifact path
    client.download_artifacts(
        mv.run_id,
        "artifacts/sample_input.npy",
        tmp_dir
    )

    sample_input = np.load(os.path.join(tmp_dir, "artifacts", "sample_input.npy"))

    # Signature check
    model_info = mlflow.models.get_model_info(model_uri)
    sig = model_info.signature
    assert sig is not None, " Signature missing"

    assert sample_input.shape[1] == len(sig.inputs.inputs), \
        " Feature mismatch between training and inference"

    preds = model.predict(sample_input)

    assert preds is not None
    assert len(preds) == sample_input.shape[0]

    #  Safer output check
    assert isinstance(preds[0], (int, np.integer, np.int64)), \
        " Prediction type invalid"

    print(f"CI SAFE MODEL TEST PASSED | v{mv.version}")
