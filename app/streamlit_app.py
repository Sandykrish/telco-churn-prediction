"""Streamlit front-end for the churn model.

Three things a retention analyst needs: score one customer from a form, score a CSV of
customers, and see how good the model is.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import NUMERIC_COLUMNS, clean  # noqa: E402
from src.features import ENGINEERED_NUMERIC, add_features  # noqa: E402

MODEL_PATH = PROJECT_ROOT / os.environ.get("CHURN_MODEL_PATH", "models/churn_model.joblib")
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
FIGURES = PROJECT_ROOT / "reports" / "figures"

# Model input layout: the encoded columns produced in notebook 03.
FEATURE_COLUMNS = [
    "Contract_One year",
    "Contract_Two year",
    "Dependents_Yes",
    "DeviceProtection_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "MonthlyCharges",
    "MultipleLines_Yes",
    "OnlineBackup_Yes",
    "OnlineSecurity_Yes",
    "PaperlessBilling_Yes",
    "Partner_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "PhoneService_Yes",
    "SeniorCitizen_Yes",
    "StreamingMovies_Yes",
    "StreamingTV_Yes",
    "TechSupport_Yes",
    "TotalCharges",
    "avg_monthly_charge",
    "charge_ratio",
    "gender_Male",
    "num_services",
    "tenure",
    "tenure_group_13-24",
    "tenure_group_25-48",
    "tenure_group_49-72",
]

YES_NO = ["No", "Yes"]

st.set_page_config(page_title="Telco churn scoring", page_icon=":telephone_receiver:", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def encode_rows(df: pd.DataFrame, scaler) -> np.ndarray:
    """Apply the notebook-03 transformation to raw customer rows and return the matrix
    the model expects."""
    fe = add_features(df)
    numeric = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    scaled = pd.DataFrame(scaler.transform(fe[numeric]), columns=numeric, index=fe.index)
    categorical = [c for c in fe.columns if c not in numeric]
    dummies = pd.get_dummies(fe[categorical], dtype=int)
    encoded = pd.concat([scaled, dummies], axis=1)
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return encoded.values


def customer_form() -> pd.DataFrame:
    """Collect one customer's attributes and return a single-row raw frame."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", YES_NO)
        partner = st.selectbox("Partner", YES_NO)
        dependents = st.selectbox("Dependents", YES_NO)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", YES_NO, index=1)
        multiple = st.selectbox("Multiple lines", YES_NO)
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=1)
        security = st.selectbox("Online security", YES_NO)
        backup = st.selectbox("Online backup", YES_NO)
        protection = st.selectbox("Device protection", YES_NO)
        support = st.selectbox("Tech support", YES_NO)
        tv = st.selectbox("Streaming TV", YES_NO)
        movies = st.selectbox("Streaming movies", YES_NO)
    with c3:
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", YES_NO, index=1)
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 79.9, step=0.5)
        total = st.number_input("Total charges ($)", 0.0, 10000.0, float(round(79.9 * max(tenure, 1), 2)), step=1.0)

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )


def risk_band(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def page_single(model, scaler, threshold: float):
    st.header("Score a customer")
    row = customer_form()
    if st.button("Predict churn", type="primary"):
        proba = float(model.predict_proba(encode_rows(row, scaler))[0][1])
        label = "likely to churn" if proba >= threshold else "likely to stay"
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Decision", label)
        m3.metric("Risk band", risk_band(proba))
        st.progress(min(max(proba, 0.0), 1.0))
        with st.expander("Customer record sent to the model"):
            st.dataframe(row.T.rename(columns={0: "value"}).astype(str), use_container_width=True)


def page_batch(model, scaler, threshold: float):
    st.header("Batch scoring")
    st.write(
        "Upload a CSV with the same columns as the raw Telco file (a `customerID` column is "
        "optional, `Churn` is ignored if present)."
    )
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is None:
        return
    raw = pd.read_csv(uploaded)
    ids = raw["customerID"] if "customerID" in raw.columns else pd.Series(range(len(raw)), name="row")
    df = clean(raw).drop(columns=[c for c in ("customerID", "Churn") if c in raw.columns])
    proba = model.predict_proba(encode_rows(df, scaler))[:, 0]
    scored = pd.DataFrame(
        {
            "customerID": ids.values,
            "churn_probability": np.round(proba, 4),
            "prediction": np.where(proba >= threshold, "Churn", "Stay"),
            "risk_band": [risk_band(p) for p in proba],
        }
    ).sort_values("churn_probability", ascending=False)
    st.success(f"Scored {len(scored)} customers - {int((scored['prediction'] == 'Churn').sum())} flagged at threshold {threshold:.2f}")
    st.dataframe(scored, use_container_width=True, height=400)


def page_performance():
    st.header("Model performance")
    if not METRICS_PATH.exists():
        st.warning("reports/metrics.json not found - run notebook 04 first.")
        return
    summary = json.loads(METRICS_PATH.read_text())
    st.write(f"Final model: **{summary['model']}** with parameters `{summary['params']}`")
    st.subheader("Cross-validated comparison (5-fold, training split)")
    st.dataframe(pd.DataFrame(summary["cv_comparison"]).T.round(4), use_container_width=True)
    st.subheader("Held-out test metrics")
    st.dataframe(pd.DataFrame(summary["test_metrics"]).T.round(4), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "13_roc_curves.png"), caption="ROC curves")
    c2.image(str(FIGURES / "16_threshold_sweep.png"), caption="Threshold sweep for the tuned model")
    st.subheader("What the EDA showed")
    st.markdown(
        """
        * Month-to-month customers churn at ~43% against ~3% on two-year contracts.
        * Fiber optic customers churn twice as often as DSL customers.
        * Almost half of first-year customers leave; after four years fewer than one in ten do.
        * Electronic-check payers churn at ~45%, automatic payment methods at ~15%.
        """
    )
    st.image(str(FIGURES / "05_contract_billing.png"))


def main():
    st.title("Telco customer churn scoring")
    st.caption("Predict which customers are about to leave, and why.")
    model, scaler = load_artifacts()

    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.caption("Customers with a churn probability at or above the threshold are flagged.")
    page = st.sidebar.radio("Page", ["Score a customer", "Batch scoring", "Model performance"])

    if page == "Score a customer":
        page_single(model, scaler, threshold)
    elif page == "Batch scoring":
        page_batch(model, scaler, threshold)
    else:
        page_performance()


if __name__ == "__main__":
    main()
