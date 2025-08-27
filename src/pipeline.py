# src/pipeline.py
from __future__ import annotations
from typing import Tuple, Dict, List
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# --- Feature contracts (keep in one place) ---
NUM: List[str] = ["tenure_days", "avg_daily_minutes", "data_gb_30d",
                  "bill_amount_3m_avg", "support_tickets_60d"]
CAT: List[str] = ["plan_tier", "region", "channel", "device_type"]
REQUIRED: List[str] = NUM + CAT + ["churn"]

def build_pipeline() -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM),
        ("cat", cat_pipe, CAT),
    ])
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])

def _basic_schema_check(df: pd.DataFrame) -> None:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    # simple dtype guards
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in NUM if c in df):
        raise TypeError("Numeric columns must be numeric.")
    # simple domain/range sanity (tune for your business)
    if (df["tenure_days"] < 0).any():
        raise ValueError("tenure_days must be >= 0")

def train_validate(df: pd.DataFrame, random_state: int = 42
                   ) -> Tuple[Pipeline, Dict[str, float], dict]:
    """Train with a holdout and return pipeline, metrics, and holdout (for CI)."""
    _basic_schema_check(df)

    X = df[NUM + CAT].copy()
    y = df["churn"].astype(int).values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    val_probs = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    metrics = {
        "val_auc": float(auc),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val))
    }
    holdout = {"X_val": X_val, "y_val": y_val, "val_probs": val_probs}
    return pipe, metrics, holdout

def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays for CI.")
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = float(np.quantile(aucs, alpha/2))
    upper = float(np.quantile(aucs, 1 - alpha/2))
    return lower, upper

def save_artifacts(pipe: Pipeline, metrics: Dict[str, float],
                   model_path: str | Path = "model.joblib",
                   metrics_path: str | Path = "metrics.json"):
    joblib.dump(pipe, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))

def load_model(model_path: str | Path = "model.joblib") -> Pipeline:
    return joblib.load(model_path)

# --- Synthetic data generator (for demo & tests) ---
def generate_synthetic_churn(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Realistic-ish synthetic churn data with planted signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_days": rng.integers(0, 365*5, n),
        "avg_daily_minutes": rng.normal(45, 20, n).clip(0),
        "data_gb_30d": rng.gamma(2.5, 5.0, n).clip(0, 5000),
        "bill_amount_3m_avg": rng.normal(35, 18, n).clip(0),
        "support_tickets_60d": rng.poisson(0.3, n),
        "plan_tier": rng.choice(["basic", "plus", "pro"], n, p=[0.55, 0.30, 0.15]),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "channel": rng.choice(["app", "web", "retail", "callcenter"], n),
        "device_type": rng.choice(["android", "ios", "desktop"], n, p=[0.5, 0.35, 0.15]),
    })
    # planted signal: basic plan + many tickets + certain regions + low tenure
    z = (
        0.6*(df["plan_tier"] == "basic").astype(int)
        + 0.25*(df["support_tickets_60d"] >= 2).astype(int)
        + 0.2*(df["region"].isin(["south", "east"])).astype(int)
        - 0.002*(df["tenure_days"])
        - 0.005*(df["avg_daily_minutes"])
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-z))
    df["churn"] = (rng.random(n) < prob).astype(int)
    # add some messiness
    df.loc[::37, "plan_tier"] = None
    df.loc[::29, "avg_daily_minutes"] = None
    return df
