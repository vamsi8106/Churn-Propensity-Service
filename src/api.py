# src/api.py
from __future__ import annotations
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
from contextlib import asynccontextmanager
from .pipeline import NUM, CAT, load_model
import numpy as np

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        model_path = os.getenv("MODEL_PATH", "model.joblib")
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model artifact not found at {model_path}. Run `python train.py` first."
            )
        _model = load_model(model_path)
    yield
    
app = FastAPI(
    lifespan=lifespan,
    title="Churn Propensity Service",
    version="1.0.0"
)
_model = None  # loaded on startup

class PredictIn(BaseModel):
    tenure_days: Optional[float] = None
    avg_daily_minutes: Optional[float] = None
    data_gb_30d: Optional[float] = None
    bill_amount_3m_avg: Optional[float] = None
    support_tickets_60d: Optional[float] = None
    plan_tier: Optional[str] = None
    region: Optional[str] = None
    channel: Optional[str] = None
    device_type: Optional[str] = None

    @field_validator("tenure_days")
    @classmethod
    def non_negative_tenure(cls, v):
        if v is not None and v < 0:
            raise ValueError("tenure_days must be >= 0")
        return v

class PredictOut(BaseModel):
    churn_probability: float

class PredictBatchIn(BaseModel):
    rows: List[PredictIn]

class PredictBatchOut(BaseModel):
    churn_probabilities: List[float]


def _df_from_payload(payload: PredictIn | List[PredictIn]) -> pd.DataFrame:
    if isinstance(payload, PredictIn):
        data = [payload.model_dump()]
    else:
        data = [p.model_dump() for p in payload]
    df = pd.DataFrame(data)
    # ensure all expected columns exist (even if all None)
    for col in NUM + CAT:
        if col not in df.columns:
            df[col] = None
    return df[NUM + CAT]

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    X = _df_from_payload(payload)
    try:
        probs = _model.predict_proba(X)
        probs = np.asarray(probs)
        prob = float(probs[0, 1])  # Extract single probability
    except Exception as e:
        raise HTTPException(400, f"Bad request: {e}")
    return PredictOut(churn_probability=prob)


@app.post("/predict_batch", response_model=PredictBatchOut)
def predict_batch(payload: PredictBatchIn):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    X = _df_from_payload(payload.rows)
    try:
        probs = _model.predict_proba(X)
        probs = np.asarray(probs)            # <-- ADD THIS
        probs = probs[:, 1].tolist()         # <-- THEN this is safe
    except Exception as e:
        raise HTTPException(400, f"Bad request: {e}")
    return PredictBatchOut(churn_probabilities=probs)

