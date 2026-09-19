import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """Forty real customers from the raw file, half of them churners."""
    return pd.read_csv(PROJECT_ROOT / "data" / "sample_customers.csv")


@pytest.fixture(scope="session")
def pipeline():
    import joblib

    return joblib.load(PROJECT_ROOT / "models" / "churn_pipeline.joblib")
