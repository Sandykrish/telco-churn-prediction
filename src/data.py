"""Loading and cleaning the raw Telco churn file.

Everything the notebooks do to the raw CSV lives here so the Streamlit app and the
tests apply exactly the same cleaning as the training run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Every non-numeric input column, in the order they appear in the raw file.
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Columns where "No internet service" / "No phone service" is really just "No".
SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded (TotalCharges arrives as text)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy of the raw frame.

    * TotalCharges is parsed to float.
    * SeniorCitizen becomes Yes/No like every other flag column.
    * "No internet service" / "No phone service" collapse to "No" - they carry no extra
      information once InternetService / PhoneService are known.
    * Churn is mapped to 0/1.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)

    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    for col in SERVICE_COLUMNS:
        out[col] = out[col].replace({"No internet service": "No", "No phone service": "No"})

    if TARGET in out.columns:
        out[TARGET] = out[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    out = out.drop_duplicates().reset_index(drop=True)
    return out


def load_clean(path: Path | str = CLEAN_PATH) -> pd.DataFrame:
    """Read the cleaned dataset written by notebook 01."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the id column and separate the target."""
    X = df.drop(columns=[c for c in (ID_COLUMN, TARGET) if c in df.columns])
    y = df[TARGET]
    return X, y
