"""Model building: tune XGBoost on PRE-CONTACT features, choose the decision threshold from
out-of-fold predictions, compare with baselines, track everything in MLflow, apply a
promotion gate, and save the model + metadata for deployment."""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
MODEL_PATH = "tourism_project/deployment/best_model.joblib"
META_PATH = "tourism_project/deployment/model_metadata.json"
REPORT_DIR = "reports"

# Promotion gate: a newly trained model replaces the deployed one only if it passes
MIN_TEST_ROC_AUC = 0.70          # absolute floor
MAX_PR_AUC_DROP = 0.03           # may not be worse than the current model by more than this

# MLflow: use a server if one is running (the workflow starts one), else a local SQLite store
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("tourism-package-prediction")

Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
ytrain, ytest = pd.read_csv("ytrain.csv").squeeze(), pd.read_csv("ytest.csv").squeeze()

numeric = ["Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar", "NumberOfTrips",
           "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical = ["Occupation", "Gender", "MaritalStatus", "Designation"]
features = numeric + categorical
assert set(features) == set(Xtrain.columns), "feature lists must match the prepared data"


def make_preprocessor():
    # Imputation is part of the pipeline: statistics are learned from training folds only
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": round(float(thr), 2),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "f2": float(fbeta_score(y, pred, beta=2)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(estimator, X, y):
    """Pick the threshold that maximises F2 (recall weighted twice as much as precision, because
    missing a likely buyer costs more than one extra call) on OUT-OF-FOLD training predictions."""
    oof = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [fbeta_score(y, (oof >= t).astype(int), beta=2) for t in grid]
    return float(grid[int(np.argmax(scores))]), oof, grid, np.array(scores)


# ---------------------------------------------------------------- model search
scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
xgb_pipe = Pipeline([
    ("prep", make_preprocessor()),
    ("model", xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, colsample_bytree=0.8,
                                random_state=SEED, eval_metric="logloss", n_jobs=1)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [2, 3, 4, 6],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__min_child_weight": [1, 5],
    "model__reg_lambda": [1, 5],
}

def one_se_rule(res):
    """Overfitting control: among all models within one standard error of the best CV score,
    choose the one with the smallest train-vs-validation gap (the most regularised)."""
    mean, std = res["mean_test_score"], res["std_test_score"]
    ok = np.where(mean >= mean.max() - std[np.argmax(mean)])[0]
    gap = res["mean_train_score"][ok] - mean[ok]
    return int(ok[np.argmin(gap)])


with mlflow.start_run(run_name="xgboost-gridsearch"):
    # Threshold-independent tuning metric (PR-AUC) because the final decision threshold is chosen separately
    grid = GridSearchCV(xgb_pipe, param_grid, cv=cv, scoring="average_precision",
                        n_jobs=-1, return_train_score=True, refit=one_se_rule)
    grid.fit(Xtrain, ytrain)

    res = grid.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metric("cv_pr_auc_validation", res["mean_test_score"][i])
            mlflow.log_metric("cv_pr_auc_train", res["mean_train_score"][i])

    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame({**{k.split("__")[1]: [pr[k] for pr in res["params"]] for k in grid.best_params_},
                  "cv_pr_auc_validation": res["mean_test_score"],
                  "cv_pr_auc_train": res["mean_train_score"]}).to_csv(f"{REPORT_DIR}/cv_results.csv", index=False)

    best = grid.best_estimator_
    best_params = {k.split("__")[1]: v for k, v in grid.best_params_.items()}
    cv_val = float(res["mean_test_score"][grid.best_index_])
    cv_train = float(res["mean_train_score"][grid.best_index_])
    cv_best_raw = float(res["mean_test_score"].max())
    mlflow.log_metric("cv_pr_auc_validation_max", cv_best_raw)
    mlflow.log_params(best_params)
    mlflow.log_metrics({"cv_pr_auc_validation_best": cv_val, "cv_pr_auc_train_best": cv_train,
                        "cv_overfit_gap": cv_train - cv_val})

    # ------------------------------------------------ threshold from out-of-fold predictions
    threshold, oof, thr_grid, thr_scores = choose_threshold(best, Xtrain, ytrain)
    mlflow.log_param("threshold", threshold)

    # ------------------------------------------------ evaluation (test set used once)
    proba_test = best.predict_proba(Xtest)[:, 1]
    proba_train = best.predict_proba(Xtrain)[:, 1]
    test_m = metrics_at(ytest, proba_test, threshold)
    train_m = metrics_at(ytrain, proba_train, threshold)
    oof_m = metrics_at(ytrain, oof, threshold)
    default_m = metrics_at(ytest, proba_test, 0.5)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
    mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items() if k != "threshold"})

    # ------------------------------------------------ baselines (same data, same threshold procedure)
    lr = Pipeline([("prep", make_preprocessor()),
                   ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xtrain, ytrain)
    lr_thr, _, _, _ = choose_threshold(lr, Xtrain, ytrain)
    lr_m = metrics_at(ytest, lr.predict_proba(Xtest)[:, 1], lr_thr)
    base_rate = float(ytest.mean())
    baselines = {
        "contact_everyone": {"roc_auc": 0.5, "pr_auc": base_rate, "precision": base_rate,
                             "recall": 1.0, "f1": 2 * base_rate / (1 + base_rate)},
        "logistic_regression": lr_m,
    }
    mlflow.log_metrics({"baseline_lr_roc_auc": lr_m["roc_auc"], "baseline_lr_pr_auc": lr_m["pr_auc"]})

    # ------------------------------------------------ business view: cumulative gains
    order = np.argsort(-proba_test)
    hits = ytest.to_numpy()[order]
    gains = {}
    for pct in range(10, 101, 10):
        k = int(round(len(hits) * pct / 100))
        gains[pct] = float(hits[:k].sum() / hits.sum())

    # ------------------------------------------------ reports
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(ytest, (proba_test >= threshold).astype(int)),
                           display_labels=["No purchase", "Purchase"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Test confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(ytest, proba_test, ax=ax, name="XGBoost")
    PrecisionRecallDisplay.from_predictions(ytest, lr.predict_proba(Xtest)[:, 1], ax=ax, name="Logistic regression")
    ax.axhline(base_rate, ls="--", c="grey", label="Contact everyone")
    ax.plot(test_m["recall"], test_m["precision"], "ro", label=f"Chosen threshold {threshold:.2f}")
    ax.legend(); ax.set_title("Precision-recall (test)")
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/precision_recall_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(ytest, proba_test, ax=ax, name="XGBoost")
    RocCurveDisplay.from_predictions(ytest, lr.predict_proba(Xtest)[:, 1], ax=ax, name="Logistic regression")
    ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set_title("ROC (test)")
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thr_grid, thr_scores, label="F2 (out-of-fold)")
    ax.axvline(threshold, c="r", ls="--", label=f"Chosen {threshold:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F2"); ax.legend(); ax.set_title("Threshold selection (training OOF)")
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/threshold_selection.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [0] + list(gains.keys()); ys = [0] + list(gains.values())
    ax.plot(xs, [x / 100 for x in xs], "--", c="grey", label="Random contact")
    ax.plot(xs, ys, marker="o", label="Model ranking")
    ax.set_xlabel("% of customers contacted (highest score first)"); ax.set_ylabel("Share of buyers reached")
    ax.legend(); ax.set_title("Cumulative gains (test)")
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/cumulative_gains.png", dpi=150); plt.close(fig)

    for f in os.listdir(REPORT_DIR):
        mlflow.log_artifact(os.path.join(REPORT_DIR, f))

    # ------------------------------------------------ console report
    print("Best parameters:", best_params)
    print(f"Selection: simplest model within 1 SE of the best CV PR-AUC ({cv_best_raw:.3f})")
    print(f"CV PR-AUC  train {cv_train:.3f}  vs  validation {cv_val:.3f}  (gap {cv_train - cv_val:.3f})")
    print(f"Threshold chosen on out-of-fold training predictions (max F2): {threshold:.2f}")
    show = lambda name, m: print(f"{name:<28} ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
                                  f"precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    show("XGBoost - train", train_m)
    show("XGBoost - out-of-fold", oof_m)
    show("XGBoost - TEST", test_m)
    show("Logistic regression - TEST", lr_m)
    print(f"{'Contact everyone - TEST':<28} ROC-AUC 0.500 | PR-AUC {base_rate:.3f} | precision {base_rate:.3f} | recall 1.000")
    print("Test confusion matrix [tn fp fn tp]:", [test_m[k] for k in ("tn", "fp", "fn", "tp")])
    print("Cumulative gains (share of buyers reached by contacting top x%):",
          {k: round(v, 2) for k, v in gains.items() if k in (10, 20, 30, 40, 50)})

    # ------------------------------------------------ promotion gate
    failures = []
    if test_m["roc_auc"] < MIN_TEST_ROC_AUC:
        failures.append(f"test ROC-AUC {test_m['roc_auc']:.3f} < minimum {MIN_TEST_ROC_AUC}")
    if os.path.exists(META_PATH):
        prev = json.load(open(META_PATH))["test_metrics"]["pr_auc"]
        if test_m["pr_auc"] < prev - MAX_PR_AUC_DROP:
            failures.append(f"test PR-AUC {test_m['pr_auc']:.3f} is more than {MAX_PR_AUC_DROP} below "
                            f"the deployed model ({prev:.3f})")
    promoted = not failures
    mlflow.set_tag("promotion", "accepted" if promoted else "rejected")

    if promoted:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"model": best, "threshold": threshold, "features": features}, MODEL_PATH)
        sha = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
        metadata = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_sha256": sha,
            "purpose": "Propensity to buy a tourism package (historical ProdTaken label) using "
                       "pre-contact customer attributes only; not Wellness-specific.",
            "features": features, "threshold": threshold,
            "threshold_method": "max F2 on out-of-fold training predictions",
            "best_params": best_params,
            "n_train": int(len(Xtrain)), "n_test": int(len(Xtest)),
            "cv_pr_auc_train": cv_train, "cv_pr_auc_validation": cv_val,
            "test_metrics": test_m, "train_metrics": train_m,
            "test_metrics_at_0.5": default_m, "baselines": baselines,
            "cumulative_gains": gains,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "xgboost": xgb.__version__, "pandas": pd.__version__},
        }
        json.dump(metadata, open(META_PATH, "w"), indent=2)
        mlflow.log_artifact(MODEL_PATH); mlflow.log_artifact(META_PATH)
        print(f"\nPROMOTED: saved {MODEL_PATH} (sha256 {sha[:12]}) and {META_PATH}")
    else:
        print("\nREJECTED by promotion gate:", *failures, sep="\n - ")

if not promoted:
    sys.exit(1)
