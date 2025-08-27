# Churn Propensity ML Service
Production-grade FastAPI service that predicts a customer’s probability of churn. It Includes a complete testing & validation suite (pytest), training script, bootstrap CIs for metrics, API contract tests, latency checks, and robust preprocessing (missing values, unseen categories, outliers).

Tech: Python, scikit-learn, FastAPI, pytest

## Features

- End-to-end pipeline: train → save → serve → test

- Robust preprocessing: imputers, scaling, OHE with handle_unknown='ignore'

- Stat gates: AUC + bootstrap 95% CI helpers

- Artifact integrity: train→save→load round-trip tests

- API contract: Pydantic schemas, single & batch prediction endpoints

- CI-friendly: tests don’t require a real model (use env flag + monkeypatch)

## 📁 Project Structure

```text
ml-churn-service/
├─ src/
│  ├─ api.py               # FastAPI app (serving)
│  ├─ pipeline.py          # training/validation utilities (sklearn)
│  └─ __init__.py
├─ train.py                # trains a model, writes artifacts/
├─ tests/
│  ├─ conftest.py          # adds repo root to sys.path, skips model load in tests
│  ├─ test_api_contract.py # API schema/contract tests
│  ├─ test_latency.py      # latency p95 test
│  └─ test_ml_pipeline.py  # data/model tests (floors, CI, round-trip)
├─ artifacts/              # (generated) model.joblib, metrics.json
├─ requirements.txt
└─ README.md
```

## Quickstart
1. **Clone the Repository**

   ```bash
   https://github.com/vamsi8106/Churn-Propensity-Service.git
   ```
2. **Install requirements**
    ```bash
    cd Churn-Propensity-Service
    ```
    ```bash
    pip install -r requirements.txt
    ```

3. **Train a model (creates artifacts/model.joblib & artifacts/metrics.json)**

    ```bash
    python train.py
    ```

4. **Run the API server:**

  ```bash
   export MODEL_PATH=artifacts/model.joblib   # Windows: set MODEL_PATH=artifacts\model.joblib
   uvicorn src.api:app --reload 
  ```
5. **Test the API:**  
   5.1 Single prediction
   
  ```bash
  curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_days": 120,
    "avg_daily_minutes": 35.2,
    "data_gb_30d": 40.5,
    "bill_amount_3m_avg": 28.9,
    "support_tickets_60d": 1,
    "plan_tier": "basic",
    "region": "south",
    "channel": "app",
    "device_type": "android"
  }'

 ```

  5.2 Batch prediction
   
  ```bash
  curl -s -X POST "http://127.0.0.1:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {"plan_tier":"plus","region":"south","channel":"web","device_type":"android",
       "tenure_days": 30, "avg_daily_minutes": 22.5, "data_gb_30d": 15.3,
       "bill_amount_3m_avg": 39.9, "support_tickets_60d": 1},
      {"plan_tier":"pro","region":"north","channel":"app","device_type":"ios",
       "tenure_days": 90, "avg_daily_minutes": 48.1, "data_gb_30d": 27.7,
       "bill_amount_3m_avg": 55.2, "support_tickets_60d": 0}
    ]
  }'

 ```

## Run the Test Suite
 **Unit & integration tests**
   ```bash
  pytest -q
 ```
You should see all tests pass:

- test_ml_pipeline.py: schema/smoke learning, unseen categories, bootstrap CI, regression guardrail, round-trip save/load

- test_api_contract.py: request/response schema, validation errors, batch contract

- test_latency.py: p95 latency budget (uses fake model)

 
## Credits

- scikit-learn team for great ML tooling

- FastAPI for a clean, modern API framework
## License

This project is licensed under the MIT License.



