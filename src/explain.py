"""SHAP explanations for the saved pipeline, shared by notebook 05 and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline

from src.features import add_features, feature_names


def encode(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Run engineered customer rows through the pipeline's fitted preprocessor. The
    sampler step is skipped, which is what we want outside of training."""
    return pipeline.named_steps["prep"].transform(df)


def make_explainer(pipeline: Pipeline, background: pd.DataFrame | None = None) -> shap.Explainer:
    """TreeExplainer for the tree ensembles, a linear/kernel fallback otherwise.

    ``background`` is a frame of *engineered* rows (post ``add_features``); it is only
    needed for the non-tree fallback.
    """
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "estimators_") or hasattr(clf, "get_booster") or hasattr(clf, "booster_"):
        return shap.TreeExplainer(clf)
    if background is None:
        raise ValueError("a background sample is needed for non-tree models")
    return shap.Explainer(clf, encode(pipeline, background))


def shap_values_for(explainer: shap.Explainer, X_enc: pd.DataFrame) -> np.ndarray:
    """SHAP values towards the churn class as a 2-D array (rows x features).

    Different explainers return different shapes: TreeExplainer on a binary sklearn forest
    gives (rows, features, classes), the boosters give (rows, features)."""
    values = explainer(X_enc).values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def contributions(pipeline: Pipeline, explainer: shap.Explainer, raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-feature SHAP contribution for each raw customer row, as a frame with the same
    index as ``raw_rows`` and one column per encoded feature. Positive pushes towards churn."""
    fe = add_features(raw_rows)
    X_enc = encode(pipeline, fe)
    values = shap_values_for(explainer, X_enc)
    return pd.DataFrame(values, index=raw_rows.index, columns=feature_names(pipeline.named_steps["prep"]))


def top_drivers(row: pd.Series, n: int = 3) -> str:
    """Compact text like ``Contract_Two year (-0.12), tenure (-0.08)`` for a table cell."""
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
    return ", ".join(f"{name} ({value:+.2f})" for name, value in top.items())
