"""mbs_risk: delinquency and prepayment-risk modelling on Freddie Mac mortgage loans.

Typical use::

    from mbs_risk import load_loans, prepare, run_task
    df = prepare(load_loans())                 # clean + features + bins + targets
    res = run_task(df, "EverDelinquent", ["logreg", "xgboost"])
    print(res.metrics)
"""
from .cleaning import clean_loans, remove_outliers_iqr
from .data import download_data, load_loans, stratified_sample
from .features import add_bins, add_features, add_targets, model_frame, monthly_payment, prepare
from .hypotheses import check_all as check_hypotheses
from .models import MODEL_NAMES, make_model
from .pipeline import TaskResult, run_task

__all__ = [
    "MODEL_NAMES", "TaskResult", "add_bins", "add_features", "add_targets", "clean_loans",
    "download_data", "load_loans", "make_model", "model_frame", "monthly_payment", "prepare",
    "remove_outliers_iqr", "run_task", "stratified_sample", "check_hypotheses",
]
__version__ = "1.0.0"
