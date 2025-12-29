import os
import json
import numpy as np
import joblib
import mlflow
import mlflow.lightgbm
import logging
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- Logging Setup -------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ------------------------- Load params.yaml -------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)["model_evaluation"]

MODEL_DIR = params["model_dir"]
TEST_FEATURES_PATH = params.get("test_features_path", "data/features/features_test.npz")
MLFLOW_TRACKING_URI = params["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT_NAME = params["mlflow_experiment_name"]

LABELS = ["negative", "neutral", "positive"]  # consistent labeling for confusion matrix

# ------------------------- Load Evaluation Features -------------------------
def load_eval_features(path):
    logger.info(f"Loading evaluation features from: {path}")

    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError("Evaluation requires labeled data (X, y)")

    X = np.array(data["X"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.int64)

    if y.size == 0:
        raise ValueError("y is empty. Cannot evaluate model.")

    logger.info(f"Eval shape → X: {X.shape}, y: {y.shape}")
    return X, y

# ------------------------- Load Trained Model -------------------------
def load_model():
    model_path = os.path.join(MODEL_DIR, "lightgbm_best_model.pkl")
    logger.info(f"Loading trained model from: {model_path}")
    return joblib.load(model_path)

# ------------------------- Save Confusion Matrix -------------------------
def save_confusion_matrix(y_true, y_pred, labels=LABELS):
    logger.info("Saving confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    return path

# ------------------------- Main -------------------------
def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load features & model
    X_eval, y_eval = load_eval_features(TEST_FEATURES_PATH)
    model = load_model()

    logger.info("Running model evaluation...")
    y_pred = model.predict(X_eval)

    # ------------------------- Metrics -------------------------
    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "macro_precision": precision_score(y_eval, y_pred, average="macro"),
        "macro_recall": recall_score(y_eval, y_pred, average="macro"),
        "macro_f1": f1_score(y_eval, y_pred, average="macro")
    }
    logger.info(f"Evaluation Metrics: {metrics}")

    # Classification report
    report = classification_report(y_eval, y_pred, output_dict=True)
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    cm_path = save_confusion_matrix(y_eval, y_pred)

    # ------------------------- MLflow Logging -------------------------
    with mlflow.start_run(run_name="model_evaluation_sbert_lgbm"):
        # Log overall metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log per-class metrics
        for cls, scores in report.items():
            if isinstance(scores, dict) and "precision" in scores:
                mlflow.log_metric(f"{cls}_precision", scores["precision"])
                mlflow.log_metric(f"{cls}_recall", scores["recall"])
                mlflow.log_metric(f"{cls}_f1", scores["f1-score"])

        # Log artifacts
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

    logger.info("Model evaluation completed successfully!")

if __name__ == "__main__":
    main()
