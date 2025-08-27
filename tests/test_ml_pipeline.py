# tests/test_ml_pipeline.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.pipeline import (
    generate_synthetic_churn, train_validate, bootstrap_auc_ci, NUM, CAT
)

def test_schema_and_training_smoke():
    df = generate_synthetic_churn(n=1200, seed=1)
    _, metrics, _ = train_validate(df, random_state=1)
    assert metrics["val_auc"] >= 0.70

def test_preprocess_handles_unseen_categories():
    df = generate_synthetic_churn(n=1000, seed=2)
    pipe, _, _ = train_validate(df, random_state=2)

    sample = df.sample(7, random_state=2).copy()
    sample.loc[sample.index[0], "region"] = "central"   # unseen
    sample.loc[sample.index[1], "plan_tier"] = "enterprise"  # unseen
    probs = pipe.predict_proba(sample[NUM + CAT])[:, 1]
    assert probs.shape == (7,)
    assert np.isfinite(probs).all()

def test_bootstrap_ci_reasonable():
    df = generate_synthetic_churn(n=2000, seed=3)
    pipe, _, _ = train_validate(df, random_state=3)
    X = df[NUM + CAT]
    y = df["churn"].values
    probs = pipe.predict_proba(X)[:, 1]
    lo, hi = bootstrap_auc_ci(y, probs, n_boot=300, seed=3)
    assert 0.60 <= lo < hi <= 1.0

def test_model_regression_guardrail(tmp_path: Path):
    # Pretend champion metrics (from prod last month)
    prev = {"val_auc": 0.78, "val_auc_ci_low": 0.74}
    (tmp_path / "prev_metrics.json").write_text(json.dumps(prev))

    df = generate_synthetic_churn(n=2500, seed=4)
    _, metrics, _ = train_validate(df, random_state=4)

    new_auc = metrics["val_auc"]
    old_auc = json.loads((tmp_path / "prev_metrics.json").read_text())["val_auc"]

    # Allow ≤0.01 degradation (tunable)
    assert new_auc >= old_auc - 0.01, f"Regression: new {new_auc:.3f} << old {old_auc:.3f}"

def test_roundtrip_save_load(tmp_path: Path):
    import joblib
    df = generate_synthetic_churn(n=1800, seed=5)
    pipe, _, _ = train_validate(df, random_state=5)
    p = tmp_path / "model.joblib"
    joblib.dump(pipe, p)
    loaded = joblib.load(p)
    X = df.sample(12, random_state=6)[NUM + CAT]
    a = loaded.predict_proba(X)[:, 1]
    b = pipe.predict_proba(X)[:, 1]
    assert a.shape == b.shape == (12,)
    assert np.allclose(a, b, atol=1e-6)
