"""05 - Train delinquency models (baseline vs gradient boosting) and compare them.

Fits majority, logistic regression and XGBoost on EverDelinquent with a stratified
train/validation/test split, tunes the F1 threshold on validation and evaluates on test.
Leaking columns (MonthsDelinquent, MonthsInRepayment) are excluded by mbs_risk.model_frame.

    python examples/05_train_delinquency_model.py
"""
from pathlib import Path

from mbs_risk import config, load_loans, plotting, prepare, run_task

OUT = Path(__file__).parent / "output"
df = prepare(load_loans(config.SAMPLE_CSV))
res = run_task(df, config.DELINQ_TARGET, ["majority", "logreg", "hist_gb", "xgboost"])
print("\n", res.metrics[["roc_auc", "pr_auc", "f1", "precision", "recall", "brier"]].round(3).to_string())

plotting.plot_roc_pr(res, OUT / "05_roc_pr.png")
plotting.plot_before_after(res, "majority", res.best_model(), OUT / "05_before_after.png")
print(f"\nbest model on the sample: {res.best_model()} -> figures in {OUT}")
