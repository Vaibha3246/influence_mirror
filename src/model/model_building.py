# src/model/model_building.py
"""
Model Building for Influence Mirror (Production-Ready)

Highlights:
- Uses Optuna for hyperparameter tuning.
- SMOTE for class imbalance.
- Deterministic training with fixed random seed.
- Logs metrics, confusion matrix, and model artifacts to MLflow (DAGsHub integrated).
- No data leakage: validation set used for tuning, test set only for final evaluation.
"""

import os
import joblib
import yaml
import logging
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import random


# DagsHub + MLflow Tracking Setup

import dagshub
import mlflow

dagshub.init(repo_owner='Vaibha3246', repo_name='influence_mirror', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/influence_mirror.mlflow")


# Logging config

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler("model_building_errors.log")
fh.setLevel(logging.ERROR)

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)

if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)


# Helper Functions

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_features(base_dir: str, split: str) -> dict:
    file_path = os.path.join(base_dir, f"{split}_features.pkl")
    logger.info(f"Loading features from {file_path}")
    return joblib.load(file_path)


def normalize_labels(y_train, y_test):
    """Shift labels if they include negative values (e.g. -1 -> 0)."""
    mn = int(np.min(y_train))
    if mn < 0:
        shift = -mn
        y_train = y_train + shift
        y_test = y_test + shift
        logger.info(f"Shifted labels by +{shift}; classes now: {np.unique(y_train)}")
    return y_train, y_test


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



# SMOTE wrapper

def apply_smote_safe(X, y):
    """Apply SMOTE safely, converting to dense if needed."""
    sm = SMOTE(random_state=42)
    try:
        X_res, y_res = sm.fit_resample(X, y)
    except Exception:
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X_res, y_res = sm.fit_resample(X_dense, y)
    return X_res, y_res



# Optuna Tuning

def optimize_model(X_train, y_train, X_val, y_val, params_cfg, n_trials=1):
    def objective(trial):
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "objective": params_cfg.get("objective", "multi:softprob"),
            "num_class": int(np.unique(y_train).shape[0]),
            "random_state": params_cfg.get("random_state", 42),
            "n_jobs": -1,
        }

        model = XGBClassifier(**trial_params)
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=False)
        except TypeError:
            model.fit(X_train, y_train)

        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best parameters found: {study.best_params}")
    return study.best_params



# Train + MLflow Logging

def train_and_log(X_train, y_train, X_val, y_val, X_test, y_test, best_params, cfg):
    model = XGBClassifier(**best_params)
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=False)
    except TypeError:
        model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info(f"Accuracies → Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    mlflow.set_experiment(cfg.get("mlflow_experiment", "influence_mirror"))

    with mlflow.start_run(run_name=cfg.get("run_name", "XGBoost_Final")):
        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc
        })

        # Classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for m_name, m_val in metrics.items():
                    mlflow.log_metric(f"{label}_{m_name}", float(m_val))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "xgboost_best.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

    return model, {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc}



# Main Pipeline

def main():
    try:
        set_seed(42)

        root = get_project_root()
        params_all = load_yaml(os.path.join(root, "params.yaml"))
        model_cfg = params_all.get("model_building", {})

        feat_dir = os.path.join(root, "data", "features")
        train_obj = load_features(feat_dir, "train")
        test_obj = load_features(feat_dir, "test")

        X_all, y_all = train_obj["X"], np.asarray(train_obj["y"])
        X_test, y_test = test_obj["X"], np.asarray(test_obj["y"])

        # Normalize if labels include negatives
        y_all, y_test = normalize_labels(y_all, y_test)

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        logger.info(f"Train shape: {getattr(X_train, 'shape', None)}, Val shape: {getattr(X_val, 'shape', None)}")

        # Apply SMOTE on training partition only
        X_train_res, y_train_res = apply_smote_safe(X_train, y_train)
        logger.info(f"After SMOTE → {np.unique(y_train_res, return_counts=True)}")

        # Hyperparameter tuning
        best_params = optimize_model(X_train_res, y_train_res, X_val, y_val, model_cfg, n_trials=model_cfg.get("optuna_trials", 25))

        # Final training + MLflow logging
        model, metrics = train_and_log(X_train_res, y_train_res, X_val, y_val, X_test, y_test, best_params, model_cfg)

        logger.info(f" Pipeline completed successfully. Test accuracy: {metrics['test_acc']:.4f}")

    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
