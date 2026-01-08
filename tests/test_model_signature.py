import mlflow
import numpy as np
import os

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_model_loads():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    model = mlflow.pyfunc.load_model(model_uri)
    assert model is not None, "Model failed to load"


def test_model_has_signature():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    info = mlflow.models.get_model_info(model_uri)
    assert info.signature is not None, "Model signature missing"


def test_model_has_input_example_artifact():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{model_uri}/model/input_example.json"
    )

    assert os.path.exists(local_path), " input_example.json missing"
    assert os.path.getsize(local_path) > 0, " input_example.json is empty"


def test_model_prediction_with_signature_shape():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    model = mlflow.pyfunc.load_model(model_uri)
    info = mlflow.models.get_model_info(model_uri)

    signature = info.signature
    assert signature is not None, "Signature missing"

    #  CORRECT WAY (tensor-based model)
    tensor_spec = signature.inputs[0]
    n_features = tensor_spec.shape[1]

    X = np.random.rand(5, n_features).astype(np.float32)
    preds = model.predict(X)

    assert preds is not None
    assert len(preds) == 5
