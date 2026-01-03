import mlflow
import pytest

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
EXPERIMENT_NAME = "yt_chrome_plugin_experiment"

# ---- Absolute thresholds (quality gates) ----
MIN_ACCURACY = 0.90
MIN_MACRO_F1 = 0.88

# ---- Regression tolerance ----
MAX_ALLOWED_DROP = 0.02  # 2%

# -------------------------------------------------
@pytest.fixture(scope="session")
def mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def get_latest_eval_run(mlflow_client):
    """
    Fetch latest MODEL EVALUATION run (not training run)
    """
    exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        pytest.skip("MLflow experiment not found")

    runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name = 'model_evaluation_sbert_lgbm'",
        order_by=["attributes.start_time DESC"],
        max_results=2,
    )

    if not runs:
        pytest.skip("No evaluation runs found")

    return runs


# -------------------------------------------------
def test_required_metrics_exist(get_latest_eval_run):
    run = get_latest_eval_run[0]
    metrics = run.data.metrics

    assert "accuracy" in metrics, "accuracy not logged in evaluation run"
    assert "macro_f1" in metrics, "macro_f1 not logged in evaluation run"


def test_absolute_performance_threshold(get_latest_eval_run):
    run = get_latest_eval_run[0]
    metrics = run.data.metrics

    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]

    print(f"\n[EVAL] Accuracy: {acc}")
    print(f"[EVAL] Macro F1: {macro_f1}")

    assert acc >= MIN_ACCURACY, "Accuracy below minimum threshold"
    assert macro_f1 >= MIN_MACRO_F1, "Macro F1 below minimum threshold"


def test_regression_against_previous_eval(get_latest_eval_run):
    """
    Compare latest eval vs previous eval
    """
    if len(get_latest_eval_run) < 2:
        pytest.skip("Not enough evaluation runs for regression check")

    latest = get_latest_eval_run[0].data.metrics
    previous = get_latest_eval_run[1].data.metrics

    print("\n--- Regression Check ---")
    print(f"Previous Macro F1: {previous['macro_f1']}")
    print(f"Latest Macro F1: {latest['macro_f1']}")

    assert (
        latest["macro_f1"]
        >= previous["macro_f1"] - MAX_ALLOWED_DROP
    ), "Macro F1 regression beyond allowed tolerance"
