import numpy as np
import pandas as pd

from src.data import SERVICE_COLUMNS, clean, missing_total_charges, parse_total_charges


def _row(**overrides):
    base = {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "250.0",
        "Churn": "No",
    }
    base.update(overrides)
    return base


def test_blank_total_charges_are_detected_not_hidden():
    raw = pd.DataFrame([_row(TotalCharges=" ", tenure=0), _row()])
    assert parse_total_charges(raw["TotalCharges"]).isna().sum() == 1
    assert len(missing_total_charges(raw)) == 1


def test_first_month_customer_gets_zero_total_charges():
    df = clean(pd.DataFrame([_row(TotalCharges=" ", tenure=0)]))
    assert df.loc[0, "TotalCharges"] == 0.0


def test_blank_total_charges_with_history_stays_missing():
    df = clean(pd.DataFrame([_row(TotalCharges=" ", tenure=9)]))
    assert np.isnan(df.loc[0, "TotalCharges"])


def test_flags_and_service_levels_are_normalised():
    df = clean(pd.DataFrame([_row(SeniorCitizen=1, InternetService="No", OnlineSecurity="No internet service", PhoneService="No", MultipleLines="No phone service", Churn="Yes")]))
    assert df.loc[0, "SeniorCitizen"] == "Yes"
    assert df.loc[0, "Churn"] == 1
    assert all(df.loc[0, c] in {"Yes", "No"} for c in SERVICE_COLUMNS)


def test_clean_keeps_every_customer(raw_sample):
    assert len(clean(raw_sample)) == len(raw_sample)
