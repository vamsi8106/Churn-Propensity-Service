# tests/test_api_contract.py
from fastapi.testclient import TestClient
from src.api import app
from pydantic import BaseModel

def test_predict_schema_and_status(monkeypatch):
    # monkeypatch a fake model to avoid requiring artifact/fit in CI
    class FakeModel:
        def predict_proba(self, X):
            # return 2-class probs; shape (n, 2)
            return [[0.3, 0.7] for _ in range(len(X))]

    import src.api as api
    monkeypatch.setattr(api, "_model", FakeModel())

    client = TestClient(app)
    payload = {
        "tenure_days": 120,
        "avg_daily_minutes": 35.2,
        "data_gb_30d": 40.5,
        "bill_amount_3m_avg": 28.9,
        "support_tickets_60d": 1,
        "plan_tier": "basic",
        "region": "south",
        "channel": "app",
        "device_type": "android"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "churn_probability" in out and 0.0 <= out["churn_probability"] <= 1.0

def test_predict_validation_error_on_bad_type(monkeypatch):
    class FakeModel:
        def predict_proba(self, X): return [[0.6, 0.4] for _ in range(len(X))]
    import src.api as api
    monkeypatch.setattr(api, "_model", FakeModel())

    client = TestClient(app)
    bad = {"tenure_days": -3, "plan_tier": "basic"}  # negative tenure triggers validator
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Pydantic validation error

def test_predict_batch(monkeypatch):
    class FakeModel:
        def predict_proba(self, X): return [[0.9, 0.1] for _ in range(len(X))]

    import src.api as api
    monkeypatch.setattr(api, "_model", FakeModel())

    client = TestClient(app)
    payload = {
        "rows": [
            {
                "tenure_days": 30,
                "avg_daily_minutes": 22.5,
                "data_gb_30d": 15.3,
                "bill_amount_3m_avg": 39.9,
                "support_tickets_60d": 1,
                "plan_tier": "plus",
                "region": "south",
                "channel": "web",
                "device_type": "android"
            },
            {
                "tenure_days": 90,
                "avg_daily_minutes": 48.1,
                "data_gb_30d": 27.7,
                "bill_amount_3m_avg": 55.2,
                "support_tickets_60d": 0,
                "plan_tier": "pro",
                "region": "north",
                "channel": "app",
                "device_type": "ios"
            }
        ]
    }

    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "churn_probabilities" in out
    assert len(out["churn_probabilities"]) == 2
    assert all(0.0 <= p <= 1.0 for p in out["churn_probabilities"])
