# tests/test_model_registry.py

import mlflow
import numpy as np
import pytest

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


@pytest.fixture(scope="session")
def model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    return mlflow.pyfunc.load_model(model_uri)


def test_model_loads(model):
    assert model is not None


def test_model_has_signature(model):
    signature = model.metadata.signature
    assert signature is not None, " Model signature missing"


def test_model_has_input_example(model):
    input_example = model.metadata.load_input_example()
    assert input_example is not None, " Input example missing"


def test_model_prediction(model):
    input_example = model.metadata.load_input_example()
    X = np.array(input_example)

    preds = model.predict(X)

    assert preds is not None
    assert len(preds) == X.shape[0]
