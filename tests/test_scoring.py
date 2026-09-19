"""Round trip through the saved pipeline with real customers."""
import numpy as np
import pandas as pd

from src.data import clean
from src.explain import contributions, make_explainer, top_drivers
from src.features import add_features, feature_names


def _inputs(raw_sample):
    return clean(raw_sample).drop(columns=["customerID", "Churn"])


def test_probabilities_are_valid_and_separate_the_classes(raw_sample, pipeline):
    proba = pipeline.predict_proba(add_features(_inputs(raw_sample)))[:, 1]
    assert proba.shape == (len(raw_sample),)
    assert ((proba >= 0) & (proba <= 1)).all()
    churned = raw_sample["Churn"].eq("Yes").to_numpy()
    assert proba[churned].mean() > proba[~churned].mean() + 0.2


def test_known_profiles_score_in_the_right_order(pipeline):
    base = {"gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "PaperlessBilling": "Yes"}
    risky = {**base, "tenure": 2, "InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No",
             "Contract": "Month-to-month", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0}
    safe = {**base, "tenure": 70, "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
            "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3850.0}
    proba = pipeline.predict_proba(add_features(pd.DataFrame([risky, safe])))[:, 1]
    assert proba[0] > 0.6
    assert proba[1] < 0.2


def test_batch_and_single_scoring_agree(raw_sample, pipeline):
    df = _inputs(raw_sample)
    batch = pipeline.predict_proba(add_features(df))[:, 1]
    singles = np.array([pipeline.predict_proba(add_features(df.iloc[[i]]))[0, 1] for i in range(5)])
    np.testing.assert_allclose(batch[:5], singles)


def test_shap_contributions_add_up_to_the_prediction(raw_sample, pipeline):
    df = _inputs(raw_sample).head(10)
    explainer = make_explainer(pipeline)
    contrib = contributions(pipeline, explainer, df)
    assert list(contrib.columns) == feature_names(pipeline.named_steps["prep"])
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    proba = pipeline.predict_proba(add_features(df))[:, 1]
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), proba, atol=1e-6)
    assert top_drivers(contrib.iloc[0]).count("(") == 3
