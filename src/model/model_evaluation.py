import os
import joblib
import yaml
import logging
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import dagshub
import json

# Initialize MLflow tracking
dagshub.init(repo_owner='Vaibha3246', repo_name='influence_mirror', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/influence_mirror.mlflow")

# Logging setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream.setFormatter(fmt)
    logger.addHandler(stream)


def get_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    root = get_root_dir()
    cfg = load_yaml(os.path.join(root, "params.yaml")).get("model_evaluation", {})

    # Load test features
    feature_dir = os.path.join(root, "data", "features")
    test_data_path = os.path.join(feature_dir, "test_features.pkl")

    if not os.path.exists(test_data_path):
        logger.error(f" Test feature file not found at: {test_data_path}")
        return

    test_data = joblib.load(test_data_path)
    X_test, y_test = test_data["X"], np.asarray(test_data["y"])

    # Check label distribution
    print("\n Checking label distribution in test set...")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} samples")

    # Load trained model
    model_path = os.path.join(root, "models", "xgboost_best.pkl")
    if not os.path.exists(model_path):
        logger.error(f" Model file not found at: {model_path}")
        return

    model = joblib.load(model_path)
    logger.info(f" Loaded trained model from {model_path}")

    # Make predictions
    logger.info(" Running predictions on test set...")
    y_pred = model.predict(X_test)

    # Debug info
    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    print("\n Unique predictions and counts:")
    for u, c in zip(unique_preds, pred_counts):
        print(f"  Predicted {u}: {c} samples")

    print("\n Sample comparison:")
    print("First 20 predictions:", y_pred[:20])
    print("First 20 actual labels:", y_test[:20])

    # Handle unseen labels
    unseen_labels = [label for label in np.unique(y_pred) if label not in np.unique(y_test)]
    if unseen_labels:
        logger.warning(f" Unseen predicted labels detected: {unseen_labels}")

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Create reports directory
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Log to MLflow
    mlflow.set_experiment(cfg.get("mlflow_experiment", "influence_mirror"))
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("test_accuracy", acc)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    mlflow.log_metric(f"{label}_{k}", float(v))

        # Save and log confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Model Evaluation")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_path = os.path.join(reports_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save metrics.json for DVC tracking
        metrics_path = os.path.join(reports_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"test_accuracy": float(acc)}, f, indent=4)
        mlflow.log_artifact(metrics_path)

    logger.info(f" Model evaluation completed successfully | Test Accuracy: {acc:.4f}")
    logger.info(f" Saved metrics report to {metrics_path}")


if __name__ == "__main__":
    main()
