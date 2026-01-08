import mlflow
import pytest

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
EXPERIMENT_NAME = "yt_chrome_plugin_experiment"

# ---- Quality gates ----
MIN_ACCURACY = 0.90
MIN_MACRO_F1 = 0.88

# ---- Regression tolerance ----
MAX_ALLOWED_DROP = 0.02  # 2%


@pytest.fixture(scope="session")
def mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def latest_eval_runs(mlflow_client):
    """
    Always return evaluation runs or FAIL
    """
    exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp is not None, "MLflow experiment NOT FOUND"

    runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name = 'model_evaluation_sbert_lgbm'",
        order_by=["attributes.start_time DESC"],
        max_results=2,
    )

    assert len(runs) >= 1, "No evaluation runs found in MLflow"

    return runs


# -------------------------------------------------
def test_required_metrics_exist(latest_eval_runs):
    run = latest_eval_runs[0]
    metrics = run.data.metrics

    assert "accuracy" in metrics, "accuracy metric missing"
    assert "macro_f1" in metrics, "macro_f1 metric missing"


def test_absolute_performance_threshold(latest_eval_runs):
    metrics = latest_eval_runs[0].data.metrics

    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]

    print(f"\n[EVAL] Accuracy: {acc}")
    print(f"[EVAL] Macro F1: {macro_f1}")

    assert acc >= MIN_ACCURACY, f"Accuracy below threshold: {acc}"
    assert macro_f1 >= MIN_MACRO_F1, f"Macro F1 below threshold: {macro_f1}"


def test_regression_against_previous_eval(latest_eval_runs):
    """
    Regression check only if previous run exists
    """
    if len(latest_eval_runs) == 1:
        # First production model → acceptable
        pytest.xfail("⚠ First evaluation run – no regression baseline yet")

    latest = latest_eval_runs[0].data.metrics
    previous = latest_eval_runs[1].data.metrics

    print("\n--- Regression Check ---")
    print(f"Previous Macro F1: {previous['macro_f1']}")
    print(f"Latest Macro F1: {latest['macro_f1']}")

    assert (
        latest["macro_f1"]
        >= previous["macro_f1"] - MAX_ALLOWED_DROP
    ), "Macro F1 regression beyond allowed tolerance"
