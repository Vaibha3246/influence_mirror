import mlflow
import pytest
import numpy as np

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"

@pytest.fixture(scope="session")
def model_info():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    return mlflow.models.get_model_info(model_uri)

@pytest.fixture(scope="session")
def model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    return mlflow.pyfunc.load_model(model_uri)

def test_model_loads(model):
    assert model is not None

def test_model_has_signature(model_info):
    signature = model_info.signature
    assert signature is not None, "Model signature missing"

def test_model_has_input_example(model_info):
    # Access input_example from model_info
    input_example = None
    if model_info.metadata and hasattr(model_info.metadata, "input_example"):
        input_example = model_info.metadata.input_example

    assert input_example is not None, "Input example missing"

def test_model_prediction(model, model_info):
    # Load input example
    input_example = None
    if model_info.metadata and hasattr(model_info.metadata, "input_example"):
        input_example = model_info.metadata.input_example

    assert input_example is not None, "Input example missing"

    X = np.array(input_example)
    preds = model.predict(X)

    assert preds is not None
    assert len(preds) == X.shape[0]
