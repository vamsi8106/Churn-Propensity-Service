# tests/test_latency.py
import time
import statistics
from fastapi.testclient import TestClient
from src.api import app

def test_p95_latency_under_budget(monkeypatch):
    class FastFake:
        def predict_proba(self, X): return [[0.25, 0.75] for _ in range(len(X))]
    import src.api as api
    monkeypatch.setattr(api, "_model", FastFake())

    client = TestClient(app)
    payload = {"plan_tier": "basic", "region": "south", "channel": "app", "device_type": "android"}

    latencies = []
    for _ in range(150):
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p95 = statistics.quantiles(latencies, n=100)[94]  # approx p95
    assert p95 <= 120.0, f"p95 latency too high: {p95:.2f} ms"
