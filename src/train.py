"""Model zoo, cross-validated comparison and hyperparameter tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def get_models(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """The four model families I compare, with sensible starting settings."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        ),
    }


def param_grids() -> dict[str, dict]:
    """Search spaces for the tuning step."""
    return {
        "Logistic Regression": {
            "C": np.logspace(-2, 1.5, 12),
            "class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [6, 8, 10, 14, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.3, 0.5],
        },
        "XGBoost": {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
    }


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Stratified k-fold comparison. Returns one row per model with mean and std of
    every metric in SCORING (accuracy, precision, recall, f1, roc_auc)."""
    cv = cv or make_cv()
    rows = []
    for name, model in models.items():
        result = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            scores = result[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        row["fit_time_s"] = result["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def tune_model(
    estimator: BaseEstimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[BaseEstimator, pd.DataFrame]:
    """Random search over ``param_grid``.

    Each candidate is fitted on the training data and scored by ROC-AUC on the
    held-out data; the best-scoring candidate is refitted and returned together
    with the full search log.
    """
    sampler = ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)
    log = []
    candidates = []
    for i, params in enumerate(sampler):
        candidate = clone(estimator).set_params(**params)
        candidate.fit(X_train, y_train)
        auc = roc_auc_score(y_eval, candidate.predict_proba(X_eval)[:, 1])
        log.append({"iter": i, "roc_auc": auc, **params})
        candidates.append((auc, params))

    log_df = pd.DataFrame(log).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_params = max(candidates, key=lambda c: c[0])[1]
    best = clone(estimator).set_params(**best_params)
    best.fit(X_train, y_train)
    return best, log_df
