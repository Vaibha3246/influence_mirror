import mlflow
import numpy as np

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_model_loads():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    assert model is not None


def test_model_has_signature():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{STAGE}")
    assert info.signature is not None


def test_signature_matches_numpy_tensor():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{STAGE}")

    signature = info.signature
    input_schema = signature.inputs

    # Exactly ONE tensor input
    assert len(input_schema.inputs) == 1

    tensor_spec = input_schema.inputs[0]

    # MLflow uses -1 for dynamic batch dimension
    assert tensor_spec.shape[0] == -1
    assert tensor_spec.shape[1] > 0

    # MLflow inferred dtype
    assert str(tensor_spec.type) == "float64"


def test_model_prediction_respects_signature_shape():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    info = mlflow.models.get_model_info(model_uri)

    tensor_spec = info.signature.inputs.inputs[0]
    n_features = tensor_spec.shape[1]

    # IMPORTANT: must be float64 (schema enforced)
    X = np.random.rand(5, n_features).astype(np.float64)

    preds = model.predict(X)

    assert preds is not None
    assert len(preds) == 5
