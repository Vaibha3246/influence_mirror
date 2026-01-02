import os
import numpy as np
import joblib
import mlflow
import lightgbm as lgb
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, classification_report
import yaml
from mlflow.models.signature import infer_signature

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("model_building")
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
    params = yaml.safe_load(f)["model_building"]

RANDOM_STATE = params["random_state"]
N_FOLDS = params["n_folds"]
N_TRIALS = params["n_trials"]
EARLY_STOPPING_ROUNDS = params["early_stopping_rounds"]
MODEL_DIR = params["model_dir"]
MLFLOW_TRACKING_URI = params["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT_NAME = params["mlflow_experiment_name"]

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load feature-engineered train data
# -------------------------
def load_features():
    logger.info("Loading feature-engineered train data...")
    train_data = np.load("data/features/features_train.npz")
    X_train, y_train = train_data["X"], train_data["y"]
    logger.info(f"Train shape: {X_train.shape}, Labels: {len(y_train)}")
    return X_train, y_train

# -------------------------
# Train LightGBM model using class weights + Optuna
# -------------------------
def train_model(X_train, y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    logger.info(f"Class weights: {cw_dict}")

    def objective(trial):
        params_lgb = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": len(classes),
            "metric": "multi_logloss",
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "class_weight": cw_dict,
            "random_state": RANDOM_STATE
        }

        recalls = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMClassifier(**params_lgb)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                           lgb.log_evaluation(0)]
            )
            preds = model.predict(X_val)
            recalls.append(recall_score(y_val, preds, average="macro"))

        return float(np.mean(recalls))

    import optuna
    logger.info("Running Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params.copy()
    best_params.update({
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "class_weight": cw_dict,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    })

    logger.info(f"Best Params: {best_params}")

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # -------------------------
    # Evaluate on train for per-class recall
    # -------------------------
    train_preds = final_model.predict(X_train)
    report = classification_report(y_train, train_preds, output_dict=True)
    logger.info("Per-class recall on training data:")
    for cls in report:
        if cls.isdigit():  # Only class labels
            logger.info(f"Class {cls} → Recall: {report[cls]['recall']:.4f}")

    return final_model, cw_dict, best_params, report

# -------------------------
# Main
# -------------------------
def main():
    X_train, y_train = load_features()
    model, cw_dict, best_params, report = train_model(X_train, y_train)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model_building_sbert_lgbm_classweight"):
        # Log parameters
        for key, value in best_params.items():
            mlflow.log_param(key, value)
        mlflow.log_dict(cw_dict, "class_weights.json")
        mlflow.log_dict(report, "train_classification_report.json")

        # Save model
        model_path = os.path.join(MODEL_DIR, "lightgbm_best_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        #  CI-SAFE model logging 
        sample_input = X_train[:1].astype(np.float32)

        signature = infer_signature(
            sample_input,
            model.predict(sample_input)
        )

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input
        )

        # Log requirements
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

    logger.info("Model Building & MLflow logging completed successfully!")

if __name__ == "__main__":
    main()
