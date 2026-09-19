import numpy as np
import pandas as pd

from src.data import clean
from src.features import add_features, build_preprocessor, feature_names


def test_avg_monthly_charge_is_total_over_tenure(raw_sample):
    fe = add_features(clean(raw_sample))
    with_history = fe[fe["tenure"] > 0]
    np.testing.assert_allclose(with_history["avg_monthly_charge"], with_history["TotalCharges"] / with_history["tenure"])


def test_first_month_customer_uses_current_price():
    df = clean(pd.DataFrame(
        {"tenure": [0], "MonthlyCharges": [70.0], "TotalCharges": [" "], "SeniorCitizen": [0],
         **{c: ["No"] for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]}}
    ))
    fe = add_features(df)
    assert fe.loc[0, "avg_monthly_charge"] == 70.0
    assert fe.loc[0, "charge_ratio"] == 1.0


def test_num_services_counts_addons(raw_sample):
    fe = add_features(clean(raw_sample))
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    assert (fe["num_services"] == (fe[addons] == "Yes").sum(axis=1)).all()
    assert fe["num_services"].between(0, 6).all()


def test_tenure_groups_cover_the_full_range():
    df = pd.DataFrame({"tenure": [0, 12, 13, 24, 25, 48, 49, 72], "MonthlyCharges": 1.0, "TotalCharges": 1.0,
                       **{c: "No" for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}})
    groups = add_features(df)["tenure_group"].tolist()
    assert groups == ["0-12", "0-12", "13-24", "13-24", "25-48", "25-48", "49-72", "49-72"]


def test_preprocessor_column_order_is_stable_and_unknown_levels_are_ignored(raw_sample):
    fe = add_features(clean(raw_sample)).drop(columns=["customerID", "Churn"])
    prep = build_preprocessor().fit(fe)
    cols = feature_names(prep)
    assert cols[:3] == ["tenure", "MonthlyCharges", "TotalCharges"]
    shuffled = fe[list(reversed(fe.columns))]
    pd.testing.assert_frame_equal(prep.transform(shuffled), prep.transform(fe))
    odd = fe.head(1).assign(PaymentMethod="Carrier pigeon")
    assert prep.transform(odd).shape == (1, len(cols))
