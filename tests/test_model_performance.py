import mlflow
import pytest

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"

# ---- Absolute thresholds (minimum quality) ----
MIN_ACCURACY = 0.90
MIN_MACRO_F1 = 0.88

# ---- Regression tolerance ----
MAX_ALLOWED_DROP = 0.02  # 2% degradation allowed


@pytest.fixture(scope="session")
def mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def get_run_by_stage(mlflow_client):
    def _get(stage):
        versions = mlflow_client.get_latest_versions(MODEL_NAME, stages=[stage])
        if not versions:
            pytest.skip(f"No model found in stage: {stage}")
        return mlflow_client.get_run(versions[0].run_id)
    return _get


def test_required_metrics_exist(get_run_by_stage):
    run = get_run_by_stage("Staging")
    metrics = run.data.metrics

    assert "accuracy" in metrics, "accuracy not logged"
    assert "macro_f1" in metrics, "macro_f1 not logged"


def test_absolute_performance_threshold(get_run_by_stage):
    run = get_run_by_stage("Staging")
    metrics = run.data.metrics

    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]

    print(f"\n[STAGING] Accuracy: {acc}")
    print(f"[STAGING] Macro F1: {macro_f1}")

    assert acc >= MIN_ACCURACY, "Accuracy below minimum threshold"
    assert macro_f1 >= MIN_MACRO_F1, "Macro F1 below minimum threshold"


def test_regression_against_production(get_run_by_stage):
    staging_run = get_run_by_stage("Staging")
    prod_run = get_run_by_stage("Production")

    s_metrics = staging_run.data.metrics
    p_metrics = prod_run.data.metrics

    print("\n--- Regression Check ---")
    print(f"Production Macro F1: {p_metrics['macro_f1']}")
    print(f"Staging Macro F1: {s_metrics['macro_f1']}")

    assert (
        s_metrics["macro_f1"]
        >= p_metrics["macro_f1"] - MAX_ALLOWED_DROP
    ), "Macro F1 regression beyond allowed tolerance"
