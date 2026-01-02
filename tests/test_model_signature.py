import os
import mlflow
import numpy as np
import tempfile
import shutil
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/")

MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"

def test_mlflow_model_signature_ci_safe():
    client = MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=[STAGE])[0]

    model_uri = f"models:/{MODEL_NAME}/{mv.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    tmp_dir = tempfile.mkdtemp()

    local_path = client.download_artifacts(
        mv.run_id,
        "artifacts/sample_input.npy",
        tmp_dir
    )

    sample_input = np.load(local_path)

    model_info = mlflow.models.get_model_info(model_uri)
    sig = model_info.signature
    assert sig is not None

    assert sample_input.shape[1] == len(sig.inputs.inputs)

    preds = model.predict(sample_input)
    assert len(preds) == sample_input.shape[0]

    shutil.rmtree(tmp_dir)
