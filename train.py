# train.py
from pathlib import Path
from src.pipeline import generate_synthetic_churn, train_validate, bootstrap_auc_ci, save_artifacts

if __name__ == "__main__":
    df = generate_synthetic_churn(n=6000, seed=7)
    pipe, metrics, holdout = train_validate(df, random_state=13)

    # Bootstrap CI on holdout
    lo, hi = bootstrap_auc_ci(holdout["y_val"], holdout["val_probs"], n_boot=800, seed=13)
    metrics["val_auc_ci_low"] = lo
    metrics["val_auc_ci_high"] = hi

    print("Metrics:", metrics)

    Path("artifacts").mkdir(exist_ok=True)
    save_artifacts(pipe, metrics, model_path="artifacts/model.joblib", metrics_path="artifacts/metrics.json")
    print("Saved artifacts to artifacts/model.joblib and artifacts/metrics.json")
