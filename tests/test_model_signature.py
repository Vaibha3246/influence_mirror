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
    info = mlflow.models.get_model_info(model_uri)
    return info

@pytest.fixture(scope="session")
def model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model

def test_model_loads(model):
    assert model is not None

def test_model_has_signature(model_info):
    signature = model_info.signature
    assert signature is not None, "Model signature missing"

def test_model_has_input_example(model):
    # Load input example from model object itself, not ModelInfo
    input_example = model.metadata.get_input_example()
    assert input_example is not None, "Input example missing"

def test_model_prediction(model):
    input_example = model.metadata.get_input_example()
    X = np.array(input_example)

    preds = model.predict(X)
    assert preds is not None
    assert len(preds) == X.shape[0]
