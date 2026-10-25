"""End-to-end experiment runner shared by scripts, examples and the notebook."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .evaluation import classification_metrics, cross_val_auc, tune_threshold
from .features import model_frame
from .models import make_model


@dataclass
class TaskResult:
    """Everything produced by :func:`run_task` for one target."""

    task: str
    metrics: pd.DataFrame                      # one row per model, test-set metrics
    test_proba: dict[str, np.ndarray]          # model -> predicted probabilities on test
    y_test: pd.Series
    X_test: pd.DataFrame
    models: dict = field(default_factory=dict)  # model -> fitted pipeline

    def best_model(self, exclude=("majority",)) -> str:
        m = self.metrics.drop(index=[e for e in exclude if e in self.metrics.index])
        return str(m["roc_auc"].idxmax())


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2,
          seed: int = config.SEED):
    """Stratified train / validation / test split (validation is carved out of train)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed)
    return X_fit, X_val, X_te, y_fit, y_val, y_te


def run_task(
    df: pd.DataFrame,
    target: str,
    model_names: list[str],
    cv_models: tuple[str, ...] = (),
    cv_splits: int = 3,
    seed: int = config.SEED,
    verbose: bool = True,
) -> TaskResult:
    """Fit each model on the training split, tune its F1 threshold on validation, score on test.

    Args:
        df: prepared frame (output of :func:`mbs_risk.features.prepare`).
        target: ``EverDelinquent`` or ``Prepaid36``.
        model_names: names accepted by :func:`mbs_risk.models.make_model`.
        cv_models: subset of models that also get a stratified K-fold ROC-AUC on train+val.
    """
    X, y = model_frame(df, target)
    X_fit, X_val, X_te, y_fit, y_val, y_te = split(X, y, seed=seed)
    rows, probas, fitted = [], {}, {}
    for name in model_names:
        t0 = time.time()
        model = make_model(name, seed=seed).fit(X_fit, y_fit)
        thr = tune_threshold(y_val, model.predict_proba(X_val)[:, 1]) if name != "majority" else 0.5
        p = model.predict_proba(X_te)[:, 1]
        row = {"model": name, **classification_metrics(y_te, p, thr), "fit_seconds": time.time() - t0}
        if name in cv_models:
            row.update(cross_val_auc(make_model(name, seed=seed), pd.concat([X_fit, X_val]),
                                     pd.concat([y_fit, y_val]), n_splits=cv_splits, seed=seed))
        rows.append(row)
        probas[name], fitted[name] = p, model
        if verbose:
            print(f"[{target}] {name:14s} roc_auc={row['roc_auc']:.4f} pr_auc={row['pr_auc']:.4f} "
                  f"f1={row['f1']:.4f} ({row['fit_seconds']:.0f}s)", flush=True)
    metrics = pd.DataFrame(rows).set_index("model")
    return TaskResult(target, metrics, probas, y_te, X_te, fitted)
