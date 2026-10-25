"""Metrics, threshold tuning, cross-validation and permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config


def tune_threshold(y_true, proba) -> float:
    """Probability threshold that maximises F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1[:-1]))])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Ranking metrics (ROC-AUC, PR-AUC), calibration (Brier) and thresholded metrics."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    one_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if one_class else float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "prevalence": float(y_true.mean()),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float((pred == y_true).mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_val_auc(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, seed: int = config.SEED) -> dict:
    """Stratified K-fold ROC-AUC and PR-AUC (mean and std) of a fresh clone of ``model``."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr, va in skf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    return {
        "cv_roc_auc_mean": float(np.mean(aucs)), "cv_roc_auc_std": float(np.std(aucs)),
        "cv_pr_auc_mean": float(np.mean(aps)), "cv_pr_auc_std": float(np.std(aps)),
    }


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Observed frequency vs mean predicted probability per quantile bin."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed": frac_pos})


def permutation_importance_table(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 3, max_rows: int = 20000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Drop in ROC-AUC when each *input* column is shuffled (on at most ``max_rows`` rows)."""
    if len(X) > max_rows:
        idx = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    r = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
