"""Model zoo, cross-validated comparison and hyperparameter tuning.

Every estimator is wrapped in an imbalanced-learn pipeline
``preprocessor -> SMOTE -> classifier`` so that, inside cross-validation, the scaler and
encoder are fit and the synthetic churners are generated from the training folds only.
The test split never sees any of it until the final scoring pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.features import build_preprocessor

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
    """Search spaces for the tuning step, keyed for the ``clf`` step of the pipeline."""
    grids = {
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
    return {name: {f"clf__{k}": v for k, v in grid.items()} for name, grid in grids.items()}


def make_pipeline(model: BaseEstimator, random_state: int = RANDOM_STATE) -> Pipeline:
    """preprocessor -> SMOTE -> model. SMOTE is a sampler step, so it only acts during
    ``fit``; ``predict`` / ``predict_proba`` go straight from the preprocessor to the model."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", clone(model)),
        ]
    )


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Stratified k-fold comparison of the full pipelines. Returns one row per model
    with mean and std of every metric in SCORING (accuracy, precision, recall, f1, roc_auc)."""
    cv = cv or make_cv()
    rows = []
    for name, model in models.items():
        result = cross_validate(make_pipeline(model), X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            scores = result[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        row["fit_time_s"] = result["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def tune_model(
    model: BaseEstimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
    n_iter: int = 20,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame]:
    """Random search over ``param_grid`` using cross-validation on the training split.

    Returns the pipeline refitted on all training rows with the best parameters, and the
    search log sorted by mean CV score.
    """
    search = RandomizedSearchCV(
        make_pipeline(model),
        param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv or make_cv(),
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    log = pd.DataFrame(search.cv_results_)
    keep = ["mean_test_score", "std_test_score", "mean_fit_time"] + [c for c in log.columns if c.startswith("param_")]
    log = log[keep].rename(columns=lambda c: c.replace("param_clf__", "")).sort_values("mean_test_score", ascending=False)
    return search.best_estimator_, log.reset_index(drop=True)
