"""
Comprehensive test suite for ML model functions and FastAPI endpoints.
Targets near-100% coverage of:  main.py, starter/ml/model.py,
                                 starter/ml/data.py, starter/train_model.py

Run with:
    pytest tests/ -v --cov=. --cov-report=term-missing
"""
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# ---------------------------------------------------------------------------
# ML model unit tests — starter/ml/model.py
# ---------------------------------------------------------------------------
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


def test_train_model_returns_correct_type():
    """train_model should return a RandomForestClassifier."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    m = train_model(X, y)
    assert isinstance(m, RandomForestClassifier)


def test_train_model_is_fitted():
    """Trained model should have feature_importances_ attribute."""
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, 20)
    m = train_model(X, y)
    assert hasattr(m, "feature_importances_")


def test_inference_returns_ndarray():
    """inference should return a numpy ndarray with one value per row."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    m = train_model(X, y)
    preds = inference(m, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)


def test_inference_predictions_are_binary():
    """inference values should only be 0 or 1."""
    X = np.random.rand(30, 5)
    y = np.random.randint(0, 2, 30)
    m = train_model(X, y)
    preds = inference(m, X)
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics_perfect():
    """Perfect predictions → all metrics == 1.0."""
    y = np.array([1, 0, 1, 0, 1])
    preds = y.copy()
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert fbeta == pytest.approx(1.0)


def test_compute_model_metrics_returns_floats():
    """compute_model_metrics should return three Python floats."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_compute_model_metrics_range():
    """All metric values should be in [0, 1]."""
    y = np.array([1, 0, 1, 1, 0, 0])
    preds = np.array([0, 0, 1, 0, 1, 0])
    for v in compute_model_metrics(y, preds):
        assert 0.0 <= v <= 1.0


def test_save_and_load_model_roundtrip():
    """Model saved to disk and reloaded should produce identical predictions."""
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, 20)
    m = train_model(X, y)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_model(m, path)
        m2 = load_model(path)
        np.testing.assert_array_equal(inference(m, X), inference(m2, X))
    finally:
        os.unlink(path)


def test_compute_slice_metrics_structure():
    """compute_slice_metrics should return one dict per unique feature value."""
    from starter.ml.data import process_data

    cat_features = ["workclass", "education"]
    df = pd.DataFrame({
        "age": [25, 35, 45, 55],
        "workclass": ["Private", "Private", "State-gov", "State-gov"],
        "education": ["Bachelors", "Masters", "Bachelors", "Masters"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    })
    train_df = df.copy()
    X_train, y_train, enc, lb_obj = process_data(
        train_df, categorical_features=cat_features, label="salary", training=True
    )
    m = train_model(X_train, y_train)

    results = compute_slice_metrics(df, "workclass", cat_features, "salary", m, enc, lb_obj)
    unique_vals = df["workclass"].unique()
    assert len(results) == len(unique_vals)
    for r in results:
        for key in ("feature", "value", "count", "precision", "recall", "fbeta"):
            assert key in r


# ---------------------------------------------------------------------------
# Data processing unit tests — starter/ml/data.py
# ---------------------------------------------------------------------------
from starter.ml.data import process_data


def _make_df():
    return pd.DataFrame({
        "age": [25, 35, 45],
        "workclass": ["Private", "State-gov", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"],
    })


def test_process_data_training_shapes():
    """Training mode should return arrays with matching row counts."""
    df = _make_df()
    X, y, enc, lb_obj = process_data(
        df, categorical_features=["workclass"], label="salary", training=True
    )
    assert X.shape[0] == len(df)
    assert len(y) == len(df)


def test_process_data_training_returns_encoder_lb():
    """Training mode should return fitted encoder and LabelBinarizer."""
    df = _make_df()
    _, _, enc, lb_obj = process_data(
        df, categorical_features=["workclass"], label="salary", training=True
    )
    assert isinstance(enc, OneHotEncoder)
    assert isinstance(lb_obj, LabelBinarizer)


def test_process_data_inference_no_label():
    """Inference mode with label=None should return an empty y array."""
    df = _make_df()
    _, _, enc, lb_obj = process_data(
        df, categorical_features=["workclass"], label="salary", training=True
    )
    df_infer = df.drop("salary", axis=1)
    X, y, _, _ = process_data(
        df_infer,
        categorical_features=["workclass"],
        label=None,
        training=False,
        encoder=enc,
        lb=lb_obj,
    )
    assert len(y) == 0


def test_process_data_inference_with_label():
    """Inference mode with label provided should encode y correctly."""
    df = _make_df()
    _, _, enc, lb_obj = process_data(
        df, categorical_features=["workclass"], label="salary", training=True
    )
    X, y, _, _ = process_data(
        df,
        categorical_features=["workclass"],
        label="salary",
        training=False,
        encoder=enc,
        lb=lb_obj,
    )
    assert len(y) == len(df)


# ---------------------------------------------------------------------------
# API tests — main.py  (using FastAPI TestClient, no live server needed)
# ---------------------------------------------------------------------------
from main import (  # noqa: E402
    FAKE_USERS_DB,
    app,
    authenticate_user,
    create_access_token,
    get_user,
    pwd_context,
    verify_password,
)

client = TestClient(app)

# Shared census payloads (hyphenated keys per Pydantic alias)
SAMPLE_ABOVE_50K = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}

SAMPLE_BELOW_50K = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 226956,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def _get_token(username="alice", password="secret") -> str:
    """Helper: obtain a valid JWT for a test user."""
    resp = client.post(
        "/token", data={"username": username, "password": password}
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ── Public endpoint ──────────────────────────────────────────────────────────

def test_get_root_status_code():
    """GET / returns HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_get_root_welcome_message():
    """GET / body contains message, demo credentials, and docs link."""
    body = client.get("/").json()
    assert "message" in body
    assert len(body["message"]) > 0
    # Reviewer-facing fields
    assert "authentication" in body
    assert body["authentication"]["demo_username"] == "alice"
    assert body["authentication"]["demo_password"] == "secret"
    assert "docs" in body


# ── Authentication endpoints ─────────────────────────────────────────────────

def test_login_valid_credentials_returns_token():
    """Valid credentials → 200 with access_token and token_type='bearer'."""
    resp = client.post("/token", data={"username": "alice", "password": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password_returns_401():
    """Wrong password → 401 Unauthorized."""
    resp = client.post("/token", data={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401():
    """Unknown username → 401 Unauthorized."""
    resp = client.post("/token", data={"username": "nobody", "password": "x"})
    assert resp.status_code == 401


def test_get_users_me_authenticated():
    """GET /users/me with valid token → 200 and correct username."""
    token = _get_token()
    resp = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "alice"


def test_get_users_me_unauthenticated():
    """GET /users/me without token → 401."""
    resp = client.get("/users/me")
    assert resp.status_code == 401


def test_get_users_me_invalid_token():
    """GET /users/me with a garbage token → 401."""
    resp = client.get("/users/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_inactive_user_cannot_access_protected_endpoints():
    """Disabled user 'bob' should receive 400 Inactive user."""
    token = _get_token(username="bob", password="password")
    resp = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400
    assert "Inactive" in resp.json()["detail"]


# ── Predict endpoint ─────────────────────────────────────────────────────────

def test_post_predict_above_50k():
    """POST /predict with high-income sample → >50K."""
    token = _get_token()
    resp = client.post(
        "/predict",
        json=SAMPLE_ABOVE_50K,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["prediction"] == ">50K"


def test_post_predict_below_50k():
    """POST /predict with low-income sample → <=50K."""
    token = _get_token()
    resp = client.post(
        "/predict",
        json=SAMPLE_BELOW_50K,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["prediction"] == "<=50K"


def test_post_predict_unauthenticated():
    """POST /predict without token → 401."""
    resp = client.post("/predict", json=SAMPLE_ABOVE_50K)
    assert resp.status_code == 401


def test_post_predict_returns_predicted_by_field():
    """Response should include 'predicted_by' set to the authenticated user."""
    token = _get_token()
    resp = client.post(
        "/predict",
        json=SAMPLE_ABOVE_50K,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.json()["predicted_by"] == "alice"


# ── Auth helper unit tests ───────────────────────────────────────────────────

def test_verify_password_correct():
    hashed = pwd_context.hash("mypassword")
    assert verify_password("mypassword", hashed) is True


def test_verify_password_wrong():
    hashed = pwd_context.hash("mypassword")
    assert verify_password("wrong", hashed) is False


def test_get_user_exists():
    user = get_user(FAKE_USERS_DB, "alice")
    assert user is not None
    assert user.username == "alice"


def test_get_user_missing():
    assert get_user(FAKE_USERS_DB, "ghost") is None


def test_authenticate_user_valid():
    user = authenticate_user(FAKE_USERS_DB, "alice", "secret")
    assert user is not None
    assert user.username == "alice"


def test_authenticate_user_bad_password():
    assert authenticate_user(FAKE_USERS_DB, "alice", "bad") is None


def test_authenticate_user_unknown():
    assert authenticate_user(FAKE_USERS_DB, "ghost", "x") is None


def test_create_access_token_contains_sub():
    from jose import jwt
    from main import ALGORITHM, SECRET_KEY
    token = create_access_token({"sub": "testuser"})
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "testuser"


# ---------------------------------------------------------------------------
# Fairness analysis smoke tests — starter/fairness_analysis.py
# ---------------------------------------------------------------------------
import json


def test_fairness_analysis_runs_and_produces_outputs():
    """run_fairness_analysis() should complete without error and write outputs."""
    from starter.fairness_analysis import run_fairness_analysis

    summary = run_fairness_analysis()

    # Summary dict has the expected top-level keys
    assert "original" in summary
    assert "reweighting" in summary
    assert "counterfactual_sex" in summary
    assert "counterfactual_race" in summary
    assert "counterfactual_both" in summary


def test_fairness_summary_json_written():
    """fairness/fairness_summary.json should exist after analysis runs."""
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(_root, "fairness", "fairness_summary.json")
    assert os.path.exists(json_path), "fairness_summary.json not found"
    with open(json_path) as f:
        data = json.load(f)
    assert "original" in data


def test_fairness_metrics_are_floats():
    """All AIF360 metrics in fairness summary should be numeric."""
    from starter.fairness_analysis import run_fairness_analysis

    summary = run_fairness_analysis()
    for key in ("Equal Opportunity Diff", "Disparate Impact",
                "Statistical Parity Diff", "Theil Index"):
        assert isinstance(summary["original"][key], float | int)


def test_counterfactual_sex_flip_rate_is_positive():
    """Sex counterfactual should flip at least some predictions."""
    from starter.fairness_analysis import run_fairness_analysis

    summary = run_fairness_analysis()
    assert summary["counterfactual_sex"]["pct_flipped"] > 0


def test_counterfactual_intersectional_gte_individual():
    """Intersectional flip rate should be >= each individual flip rate."""
    from starter.fairness_analysis import run_fairness_analysis

    summary = run_fairness_analysis()
    both = summary["counterfactual_both"]["pct_flipped"]
    sex = summary["counterfactual_sex"]["pct_flipped"]
    race = summary["counterfactual_race"]["pct_flipped"]
    assert both >= sex
    assert both >= race


def test_fairness_plots_written():
    """All five fairness PNG figures should be written to fairness/."""
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected = [
        "fig_aequitas_metrics.png",
        "fig_disparity_ratios.png",
        "fig_bias_mitigation.png",
        "fig_confusion_matrices.png",
        "fig_counterfactual.png",
    ]
    for fname in expected:
        path = os.path.join(_root, "fairness", fname)
        assert os.path.exists(path), f"Missing fairness plot: {fname}"
