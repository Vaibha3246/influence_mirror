import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"
STAGE = "Staging"


def test_model_signature_and_prediction():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=[STAGE])[0]

    model_uri = f"models:/{MODEL_NAME}/{mv.version}"

    #  Load model
    model = mlflow.pyfunc.load_model(model_uri)

    # Check signature exists
    model_info = mlflow.models.get_model_info(model_uri)
    signature = model_info.signature
    assert signature is not None, "Model signature missing"

    #  Get input example from model metadata
    input_example = model.metadata.get_input_example()
    assert input_example is not None, "Input example missing"

    # Ensure numpy array
    sample_input = np.array(input_example)

    #  Validate shape vs signature
    assert sample_input.shape[1] == len(signature.inputs.inputs)

    #  Run prediction
    preds = model.predict(sample_input)

    assert preds is not None
    assert len(preds) == sample_input.shape[0]
