# tests/test_model_signature.py

import mlflow
import numpy as np

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_model_loads():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None, "Model failed to load from MLflow"


def test_model_has_signature():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    info = mlflow.models.get_model_info(model_uri)

    assert info.signature is not None, "Model signature is missing"


def test_model_has_input_example():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    info = mlflow.models.get_model_info(model_uri)

    assert info.input_example is not None, " Input example missing in MLflow model"
    assert len(info.input_example) > 0, " Input example is empty"


def test_model_prediction_with_signature_shape():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"

    model = mlflow.pyfunc.load_model(model_uri)
    info = mlflow.models.get_model_info(model_uri)

    signature = info.signature
    assert signature is not None, "Signature missing"

    n_features = len(signature.inputs.input_names)

    X = np.random.rand(5, n_features).astype(np.float32)
    preds = model.predict(X)

    assert preds is not None, " Prediction returned None"
    assert len(preds) == 5, " Prediction count mismatch"
