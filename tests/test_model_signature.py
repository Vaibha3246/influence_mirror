import mlflow
import numpy as np

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_model_loads():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    assert model is not None, "Model failed to load"


def test_model_has_signature():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{STAGE}")

    assert info.signature is not None, "Model signature missing"


def test_signature_matches_numpy_tensor():
    """
    Signature must represent a 2D numeric tensor:
    (batch_size, n_features)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{STAGE}")

    signature = info.signature
    assert signature is not None

    # MLflow Schema API (correct for >=2.x)
    input_schema = signature.inputs
    assert len(input_schema.inputs) == 1, "Expected single tensor input"

    tensor_spec = input_schema.inputs[0]

    # Shape must be (None, n_features)
    assert tensor_spec.shape is not None
    assert len(tensor_spec.shape) == 2
    assert tensor_spec.shape[0] is None
    assert tensor_spec.shape[1] > 0


def test_model_prediction_respects_signature_shape():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    info = mlflow.models.get_model_info(model_uri)

    signature = info.signature
    tensor_spec = signature.inputs.inputs[0]

    n_features = tensor_spec.shape[1]

    # Create valid dummy input (same as training style)
    X = np.random.rand(5, n_features).astype(np.float32)

    preds = model.predict(X)

    assert preds is not None
    assert len(preds) == 5
