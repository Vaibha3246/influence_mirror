# tests/test_model_signature.py

import mlflow
import pytest
import numpy as np

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"

@pytest.fixture(scope="session")
def model_info():
    """Get MLflow ModelInfo object for metadata checks"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    info = mlflow.models.get_model_info(model_uri)
    return info

@pytest.fixture(scope="session")
def model():
    """Load the MLflow model for prediction tests"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    return mlflow.pyfunc.load_model(model_uri)

def test_model_loads(model):
    """Test that model loads successfully"""
    assert model is not None, "Failed to load model"

def test_model_has_signature(model_info):
    """Test that model has a valid signature"""
    signature = getattr(model_info, "signature", None)
    assert signature is not None, "Model signature missing"

def test_model_has_input_example(model_info):
    """Test that model has input example if provided"""
    input_example = None
    if hasattr(model_info, "input_example") and model_info.input_example is not None:
        input_example = model_info.input_example

    if input_example is None:
        pytest.skip("Input example not available for this model")

    # Optional: ensure input_example is non-empty
    assert len(input_example) > 0, "Input example is empty"

def test_model_prediction(model, model_info):
    """Test that model can make predictions using input example"""
    input_example = getattr(model_info, "input_example", None)
    
    if input_example is None:
        pytest.skip("Input example not available, skipping prediction test")

    # Convert input example to NumPy array
    X = np.array(input_example)
    preds = model.predict(X)

    assert preds is not None, "Predictions returned None"
    assert len(preds) == X.shape[0], "Number of predictions does not match input"
