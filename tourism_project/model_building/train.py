"""Model building: tune XGBoost, track with MLflow, save best model for deployment."""
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "tourism_project/deployment/best_model.joblib"
THRESHOLD = 0.45  # classification threshold, tuned for recall on the minority (buyer) class

# MLflow: use a server if one is running (the workflow starts one), else a local file store
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism-package-prediction")

Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
ytrain, ytest = pd.read_csv("ytrain.csv").squeeze(), pd.read_csv("ytest.csv").squeeze()

numeric = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
           "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
           "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(handle_unknown="ignore"), categorical),
)

# Handle class imbalance (~19% buyers)
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42,
                              eval_metric="logloss", n_jobs=-1)
pipe = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run(run_name="xgboost-gridsearch"):
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every tuned combination as a nested run
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metric("mean_cv_recall", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_recall", results["std_test_score"][i])

    best = grid.best_estimator_
    mlflow.log_params({k.split("__")[1]: v for k, v in grid.best_params_.items()})
    mlflow.log_param("classification_threshold", THRESHOLD)

    def evaluate(X, y, prefix):
        proba = best.predict_proba(X)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        return {
            f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_precision": precision_score(y, pred),
            f"{prefix}_recall": recall_score(y, pred),
            f"{prefix}_f1": f1_score(y, pred),
            f"{prefix}_roc_auc": roc_auc_score(y, proba),
        }, pred

    train_metrics, train_pred = evaluate(Xtrain, ytrain, "train")
    test_metrics, test_pred = evaluate(Xtest, ytest, "test")
    mlflow.log_metrics({**train_metrics, **test_metrics})

    print("Best parameters:", grid.best_params_)
    print("\nTrain report:\n", classification_report(ytrain, train_pred))
    print("Test report:\n", classification_report(ytest, test_pred))
    print({k: round(v, 4) for k, v in test_metrics.items()})

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": best, "threshold": THRESHOLD}, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
