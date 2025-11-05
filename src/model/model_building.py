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
import mlflow
import dagshub
from xgboost.callback import EarlyStopping
import json
import subprocess
import sys
import pkg_resources
from mlflow.utils.environment import _mlflow_conda_env

# Initialize MLflow tracking (DagsHub)
dagshub.init(repo_owner='Vaibha3246', repo_name='influence_mirror', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/influence_mirror.mlflow")

# LOGGING SETUP
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream.setFormatter(fmt)
file_handler.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(stream)
    logger.addHandler(file_handler)

# UTILS
def get_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_features(base_dir: str, split: str) -> dict:
    path = os.path.join(base_dir, f"{split}_features.pkl")
    logger.info(f"Loading features: {path}")
    return joblib.load(path)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def normalize_labels(y_train, y_test):
    mn = int(np.min(y_train))
    if mn < 0:
        shift = -mn
        y_train += shift
        y_test += shift
        logger.info(f"Shifted labels by +{shift}")
    return y_train, y_test

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    try:
        return sm.fit_resample(X, y)
    except Exception:
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return sm.fit_resample(X_dense, y)

# OPTUNA OPTIMIZATION
def optimize_xgb(X_train, y_train, X_val, y_val, cfg, trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": cfg.get("objective", "multi:softprob"),
            "num_class": int(len(np.unique(y_train))),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[EarlyStopping(rounds=25)],
            verbose=False
        )
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("Running Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    logger.info(f"Best Parameters: {study.best_params}")
    return study.best_params

# FUNCTION TO EXPORT PYTHON ENV MANUALLY (FOR DVC)
def export_python_env_yaml(output_path="models/env/python_env.yaml"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env_data = {
        "python": sys.version.split()[0],
        "dependencies": [str(pkg) for pkg in pkg_resources.working_set]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(env_data, f)
    logger.info(f" Saved python environment to {output_path}")

# TRAIN + LOG MODEL
def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params, cfg):
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[EarlyStopping(rounds=25)],
        verbose=False
    )

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_val = accuracy_score(y_val, y_val_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    logger.info(f"Accuracy - Train: {acc_train:.4f} | Val: {acc_val:.4f} | Test: {acc_test:.4f}")

    mlflow.set_experiment(cfg.get("mlflow_experiment", "influence_mirror"))
    with mlflow.start_run(run_name=cfg.get("run_name", "XGBoost_Final")) as run:
        # Log hyperparameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({
            "train_acc": acc_train,
            "val_acc": acc_val,
            "test_acc": acc_test
        })

        # Classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    mlflow.log_metric(f"{label}_{k}", float(v))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save model locally and log as artifact
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "xgboost_best.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Save metrics.json for DVC
        metrics_dict = {"train_acc": float(acc_train), "val_acc": float(acc_val), "test_acc": float(acc_test)}
        with open("metrics.json", "w", encoding="utf-8") as mf:
            json.dump(metrics_dict, mf, indent=4)
        mlflow.log_artifact("metrics.json")

        # Save experiment info
        model_info = {
            "run_id": run.info.run_id,
            "model_path": model_path
        }
        with open("experiment_info.json", "w") as f:
            json.dump(model_info, f, indent=4)
        mlflow.log_artifact("experiment_info.json")

        
        # Always create environment folder and required files
        env_dir = os.path.join("models", "env")
        os.makedirs(env_dir, exist_ok=True)

        conda_env_path = os.path.join(env_dir, "conda.yaml")
        python_env_path = os.path.join(env_dir, "python_env.yaml")
        req_path = os.path.join(env_dir, "requirements.txt")

        # Write conda.yaml
        conda_env = _mlflow_conda_env(additional_pip_deps=["dagshub"])
        with open(conda_env_path, "w", encoding="utf-8") as f:
            yaml.dump(conda_env, f)

        # Write python_env.yaml manually
        export_python_env_yaml(python_env_path)

        # Write requirements.txt
        subprocess.run(["pip", "freeze"], stdout=open(req_path, "w", encoding="utf-8"), check=False)

        # Log all 3 artifacts
        for path in [conda_env_path, python_env_path, req_path]:
            if os.path.exists(path):
                mlflow.log_artifact(path)
            else:
                logger.error(f" Missing expected file: {path}")

            
    return model, acc_test

# MAIN EXECUTION
def main():
    try:
        set_seed(42)
        root = get_root_dir()
        cfg = load_yaml_config(os.path.join(root, "params.yaml")).get("model_building", {})

        feature_dir = os.path.join(root, "data", "features")
        train_data = load_features(feature_dir, "train")
        test_data = load_features(feature_dir, "test")

        X_all, y_all = train_data["X"], np.asarray(train_data["y"])
        X_test, y_test = test_data["X"], np.asarray(test_data["y"])

        y_all, y_test = normalize_labels(y_all, y_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        X_train_res, y_train_res = apply_smote(X_train, y_train)

        best_params = optimize_xgb(X_train_res, y_train_res, X_val, y_val, cfg, trials=cfg.get("optuna_trials", 15))
        model, test_acc = train_and_log_model(X_train_res, y_train_res, X_val, y_val, X_test, y_test, best_params, cfg)

        logger.info(f"Pipeline completed successfully | Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        logger.exception(f"Model building failed: {e}")
        raise

if __name__ == "__main__":
    main()
