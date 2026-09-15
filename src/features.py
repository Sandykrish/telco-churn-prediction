"""Feature engineering: derived columns, encoding and scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import NUMERIC_COLUMNS

ENGINEERED_NUMERIC = ["avg_monthly_charge", "num_services", "charge_ratio"]

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


def encode_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """One-hot encode the categoricals and standardise the numeric columns.

    Returns the model-ready matrix (as a DataFrame so the column names survive) and
    the fitted scaler so the app can reuse it.
    """
    numeric = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[numeric]), columns=numeric, index=df.index)

    categorical = [c for c in df.columns if c not in numeric]
    dummies = pd.get_dummies(df[categorical], drop_first=True, dtype=int)

    X = pd.concat([scaled, dummies], axis=1)
    return X, scaler
