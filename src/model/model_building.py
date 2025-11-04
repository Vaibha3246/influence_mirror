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



# Initialize MLflow tracking (DagsHub)
dagshub.init(repo_owner='Vaibha3246', repo_name='influence_mirror', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/influence_mirror.mlflow")


# LOGGING
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


#  UTILS
def get_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_features(base_dir: str, split: str) -> dict:
    """Load preprocessed features from .pkl file"""
    path = os.path.join(base_dir, f"{split}_features.pkl")
    logger.info(f"Loading features: {path}")
    return joblib.load(path)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_labels(y_train, y_test):
    """Shift labels if negatives (-1) are present."""
    mn = int(np.min(y_train))
    if mn < 0:
        shift = -mn
        y_train += shift
        y_test += shift
        logger.info(f"Shifted labels by +{shift}")
    return y_train, y_test


#  SMOTE
def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    try:
        return sm.fit_resample(X, y)
    except Exception:
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return sm.fit_resample(X_dense, y)


#  OPTUNA
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
            callbacks=[EarlyStopping(rounds=25, save_best=True)],  # fixed
            verbose=False
        )
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("Running Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    logger.info(f"Best Parameters: {study.best_params}")
    return study.best_params


#  TRAINING + LOGGING
def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params, cfg):
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[EarlyStopping(rounds=25, save_best=True)],   #  fixed
        verbose=False
    )

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_val = accuracy_score(y_val, y_val_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    logger.info(f"Accuracy - Train: {acc_train:.4f} | Val: {acc_val:.4f} | Test: {acc_test:.4f}")

    # Debug info
    unique_preds, pred_counts = np.unique(y_test_pred, return_counts=True)
    logger.info(f"Unique predictions: {dict(zip(unique_preds, pred_counts))}")
    logger.info(f"Sample Predictions: {y_test_pred[:20]}")
    logger.info(f"Sample Actuals: {y_test[:20]}")

    mlflow.set_experiment(cfg.get("mlflow_experiment", "influence_mirror"))
    with mlflow.start_run(run_name=cfg.get("run_name", "XGBoost_Final")):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "train_acc": acc_train,
            "val_acc": acc_val,
            "test_acc": acc_test
        })

        report = classification_report(y_test, y_test_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    mlflow.log_metric(f"{label}_{k}", float(v))

        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "xgboost_best.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

    return model, acc_test


#  MAIN
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

        logger.info(f"Unique train labels: {np.unique(y_all)}")
        logger.info(f"Unique test labels: {np.unique(y_test)}")

        missing_labels = [l for l in np.unique(y_test) if l not in np.unique(y_all)]
        if missing_labels:
            logger.warning(f"Test has unseen labels: {missing_labels}. Mapping them to nearest train class.")
            y_test = np.clip(y_test, np.min(y_all), np.max(y_all))

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        logger.info(f"Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        X_train_res, y_train_res = apply_smote(X_train, y_train)
        logger.info("Applied SMOTE successfully.")

        best_params = optimize_xgb(X_train_res, y_train_res, X_val, y_val, cfg, trials=cfg.get("optuna_trials", 25))
        model, test_acc = train_and_log_model(X_train_res, y_train_res, X_val, y_val, X_test, y_test, best_params, cfg)

        logger.info(f"Pipeline completed successfully | Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        logger.exception(f"Model building failed: {e}")
        raise


if __name__ == "__main__":
    main()
