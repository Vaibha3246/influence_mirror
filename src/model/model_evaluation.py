import os
import json
import numpy as np
import joblib
import mlflow
import mlflow.lightgbm
import logging
import yaml
import pandas as pd
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

# -------------------------
# Logging Setup
# -------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -------------------------
# Load params.yaml
# -------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)["model_evaluation"]

MODEL_DIR = params["model_dir"]
MLFLOW_TRACKING_URI = params["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT_NAME = params["mlflow_experiment_name"]

# -------------------------
# Load Feature Engineered Test Data
# -------------------------
def load_test_features():
    logger.info("Loading test features...")
    test_data = np.load("data/features/features_test.npz")
    X_test, y_test = test_data["X"], test_data["y"]
    logger.info(f"Test shape: {X_test.shape}, Labels: {len(y_test)}")
    return X_test, y_test

# -------------------------
# Load Model
# -------------------------
def load_model():
    model_path = os.path.join(MODEL_DIR, "lightgbm_best_model.pkl")
    logger.info(f"Loading saved trained model: {model_path}")
    return joblib.load(model_path)

# -------------------------
# Save Confusion Matrix Plot
# -------------------------
def save_confusion_matrix(y_true, y_pred):
    logger.info("Generating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = "artifacts/confusion_matrix.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    return cm_path

# -------------------------
# Main Evaluation
# -------------------------
def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_test, y_test = load_test_features()
    model = load_model()

    logger.info("Starting model evaluation...")

    y_pred = model.predict(X_test)

    # Metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro"),
        "macro_recall": recall_score(y_test, y_pred, average="macro"),
        "macro_f1": f1_score(y_test, y_pred, average="macro")
    }

    logger.info(f"Evaluation Results: {results}")

    # Save classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=4)

    # Confusion Matrix & Save
    cm_path = save_confusion_matrix(y_test, y_pred)

    # MLflow Logging
    with mlflow.start_run(run_name="model_evaluation_sbert_lgbm"):
        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        # Log class-wise metrics
        for cls, scores in class_report.items():
            if cls.isdigit() or cls.startswith("-"):  # Only class labels
                mlflow.log_metric(f"{cls}_precision", scores["precision"])
                mlflow.log_metric(f"{cls}_recall", scores["recall"])
                mlflow.log_metric(f"{cls}_f1", scores["f1-score"])
    

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

    logger.info("Model Evaluation Completed & Logged Successfully!")

if __name__ == "__main__":
    main()
