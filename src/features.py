"""Feature engineering: derived columns plus the encoding/scaling preprocessor.

`add_features` is a plain row-wise transformation with no fitted state, so it is safe to
apply to the whole dataset. Everything that *learns* from the data (scaler statistics,
one-hot categories) lives in the ColumnTransformer from `build_preprocessor`, which is fit
on the training split only and saved with the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

ENGINEERED_NUMERIC = ["avg_monthly_charge", "num_services", "charge_ratio"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

_ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns I use on top of the raw ones.

    avg_monthly_charge  what the customer actually paid per month over their tenure
                        (first-month customers have no history yet, so it is their
                        current monthly price)
    charge_ratio        current monthly price relative to that average (price creep)
    num_services        how many add-on services are switched on
    tenure_group        coarse tenure buckets for the EDA and for tree splits
    """
    out = df.copy()
    has_history = out["tenure"] > 0
    out["avg_monthly_charge"] = np.where(
        has_history, out["TotalCharges"] / out["tenure"].where(has_history, 1), out["MonthlyCharges"]
    )
    out["charge_ratio"] = out["MonthlyCharges"] / out["avg_monthly_charge"].replace(0, np.nan)
    out["charge_ratio"] = out["charge_ratio"].fillna(1.0)
    out["num_services"] = (out[_ADDON_SERVICES] == "Yes").sum(axis=1)
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return out


def numeric_features() -> list[str]:
    return NUMERIC_COLUMNS + ENGINEERED_NUMERIC


def categorical_features() -> list[str]:
    return CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """Unfitted encoder + scaler.

    Numeric columns are standardised; categoricals are one-hot encoded with the first level
    dropped (so `Contract` becomes `Contract_One year` / `Contract_Two year`). Unknown levels
    at scoring time encode as all-zeros instead of raising, which matters for the batch
    upload in the app. Fit it on the training rows only.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features()),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False, dtype=int), categorical_features()),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Column names of the matrix a fitted preprocessor produces, in order."""
    return preprocessor.get_feature_names_out().tolist()
