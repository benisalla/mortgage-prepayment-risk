"""Figures with one consistent matplotlib style (no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from . import config

PALETTE = {
    "majority": "#9aa0a6",
    "logreg": "#4c78a8",
    "random_forest": "#72b7b2",
    "hist_gb": "#f58518",
    "xgboost": "#e45756",
}
ACCENT = "#4c78a8"
ACCENT2 = "#e45756"
NICE = {
    "majority": "Majority (prior)",
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gb": "Hist. gradient boosting",
    "xgboost": "XGBoost",
    "EverDelinquent": "Delinquency (EverDelinquent)",
    "Prepaid36": "Prepayment within 36 months (Prepaid36)",
}


def set_style() -> None:
    """Apply the project style (white background, light grid, readable fonts)."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })


def _save(fig, path: str | Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------
def plot_target_balance(df: pd.DataFrame, path=None):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, t in zip(axes, [config.DELINQ_TARGET, config.PREPAY_TARGET]):
        vc = df[t].dropna().astype(int).value_counts().sort_index()
        bars = ax.bar(["0", "1"], vc.values, color=[ACCENT, ACCENT2], width=0.6)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / vc.sum():.1%})",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_title(NICE[t], fontsize=10)
        ax.set_ylim(0, vc.max() * 1.25)
        ax.set_ylabel("loans")
    fig.suptitle("Class balance of the two targets", fontweight="bold")
    return _save(fig, path)


def plot_distributions(df: pd.DataFrame, path=None):
    set_style()
    cols = ["CreditScore", "LTV", "DTI", "OrigUPB", "OrigInterestRate", "MonthsInRepayment"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        s = df[c].dropna()
        ax.hist(s, bins=50, color=ACCENT, alpha=0.85)
        ax.axvline(s.median(), color=ACCENT2, lw=1.5, ls="--", label=f"median {s.median():,.4g}")
        ax.set_title(c)
        ax.legend()
    fig.suptitle("Distributions of the main numeric variables (missing codes removed)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation(df: pd.DataFrame, path=None):
    set_style()
    cols = ["CreditScore", "MIP", "OCLTV", "LTV", "DTI", "OrigUPB", "OrigInterestRate",
            "RateSpread", "MonthlyPayment", "EstMonthlyIncome", "VintageMonth",
            "MonthsInRepayment", "EverDelinquent", "Prepaid36"]
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Spearman correlation (Prepaid36 defined for never-delinquent loans)")
    return _save(fig, path)


def _survival(months: pd.Series, horizon: int) -> np.ndarray:
    """Share of loans still in the pool at each month 0..horizon (no censoring correction)."""
    m = months.to_numpy()
    return np.array([(m > k).mean() for k in range(horizon + 1)])


def plot_survival(df: pd.DataFrame, path=None, horizon: int = 120):
    """Share of never-delinquent loans still outstanding by months since first payment."""
    set_style()
    d = df[df[config.DELINQ_TARGET] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in [(axes[0], "CreditRange", "by credit score"),
                           (axes[1], "RateBucket", "by origination rate (quintiles)")]:
        groups = [g for g in d[col].cat.categories if (d[col] == g).sum() > 200]
        cmap = plt.get_cmap("viridis", len(groups))
        for i, g in enumerate(groups):
            s = _survival(d.loc[d[col] == g, "MonthsInRepayment"], horizon)
            ax.plot(range(horizon + 1), s, color=cmap(i), lw=2, label=str(g))
        ax.axvline(config.PREPAY_HORIZON_MONTHS, color="grey", ls=":", lw=1)
        ax.text(config.PREPAY_HORIZON_MONTHS + 2, 0.95, "36-month horizon", fontsize=8, color="grey")
        ax.set_xlabel("months since first payment")
        ax.set_ylabel("share of loans still outstanding")
        ax.set_title(f"Loan survival {title}")
        ax.legend(title=col, fontsize=8)
        ax.set_ylim(0, 1.02)
    fig.suptitle("Prepayment speed of never-delinquent loans", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_hypotheses(summary: pd.DataFrame, path=None):
    """One bar panel per hypothesis with the prepayment rate per group and the verdict."""
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    colors = {"supported": "#54a24b", "contradicted": ACCENT2, "no material effect": "#9aa0a6",
              "not testable": "#9aa0a6"}
    for ax, (key, row) in zip(axes.ravel(), summary.iterrows()):
        v = row["verdict"]
        ax.set_title(f"{key}: {row['statement']}", fontsize=9.5)
        if not isinstance(row["rates"], pd.DataFrame):
            ax.text(0.5, 0.5, "Not testable:\n" + row["detail"], ha="center", va="center",
                    fontsize=9, transform=ax.transAxes, wrap=True)
            ax.axis("off")
            continue
        g = row["rates"]
        labels = [str(i) for i in g.index]
        ax.bar(labels, 100 * g["rate"], color=colors[v], alpha=0.85)
        for i, (r, n) in enumerate(zip(g["rate"], g["n"])):
            ax.text(i, 100 * r, f"{r:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, 100 * g["rate"].max() * 1.3)
        ax.set_ylabel("prepaid within 36 months (%)")
        ax.tick_params(axis="x", labelsize=7.5, rotation=15)
        ax.text(0.02, 0.97, f"{v.upper()} ({row['effect_pp']:+.1f} pp)", transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=colors[v], va="top")
    fig.suptitle("Original hypotheses H1-H6 tested on never-delinquent loans", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def plot_roc_pr(result, path=None):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    y = result.y_test
    for name, p in result.test_proba.items():
        m = result.metrics.loc[name]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        c = PALETTE.get(name)
        axes[0].plot(fpr, tpr, color=c, lw=1.8, label=f"{NICE[name]} (AUC {m.roc_auc:.3f})")
        axes[1].plot(rec, prec, color=c, lw=1.8, label=f"{NICE[name]} (AP {m.pr_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    axes[1].axhline(y.mean(), color="grey", ls=":", lw=1)
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
    fig.suptitle(f"{NICE[result.task]}: test set (n={len(y):,})", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _confusion(ax, y, p, thr, title):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j] / cm[i].sum():.1%} of row", ha="center",
                    va="center", fontsize=8.5, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=9.5)


def plot_confusions(result, names=None, path=None):
    set_style()
    names = names or [n for n in result.test_proba if n != "majority"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        m = result.metrics.loc[n]
        _confusion(ax, result.y_test, result.test_proba[n], m.threshold,
                   f"{NICE[n]}\nthr={m.threshold:.2f}  F1={m.f1:.3f}")
    fig.suptitle(f"{NICE[result.task]}: confusion matrices at the F1-tuned threshold", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(result, path=None):
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="perfect")
    for name, p in result.test_proba.items():
        if name == "majority":
            continue
        fp, mp = calibration_curve(result.y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", ms=3.5, color=PALETTE.get(name), lw=1.6,
                label=f"{NICE[name]} (Brier {result.metrics.loc[name].brier:.4f})")
    hi = max(0.05, float(max(np.quantile(p, 0.99) for p in result.test_proba.values())))
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="mean predicted probability",
           ylabel="observed frequency", title=f"Calibration: {result.task}")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_importance(table: pd.DataFrame, title: str, path=None, top: int = 15):
    set_style()
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh(t["feature"], t["importance"], xerr=t["std"], color=ACCENT, alpha=0.9)
    ax.set_xlabel("drop in test ROC-AUC when the column is shuffled")
    ax.set_title(title)
    return _save(fig, path)


def plot_before_after(result, baseline: str, best: str, path=None):
    """Baseline vs final model: ROC, PR and the two confusion matrices side by side."""
    set_style()
    y = result.y_test
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for name, ls in [(baseline, "--"), (best, "-")]:
        p, m = result.test_proba[name], result.metrics.loc[name]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, ls=ls, lw=2, color=PALETTE[name], label=f"{NICE[name]} AUC {m.roc_auc:.3f}")
        axes[1].plot(rec, prec, ls=ls, lw=2, color=PALETTE[name], label=f"{NICE[name]} AP {m.pr_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[1].axhline(y.mean(), color="grey", ls=":", lw=1)
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    axes[0].legend(loc="lower right"), axes[1].legend(loc="upper right")
    for ax, name, tag in [(axes[2], baseline, "Before"), (axes[3], best, "After")]:
        m = result.metrics.loc[name]
        _confusion(ax, y, result.test_proba[name], m.threshold,
                   f"{tag}: {NICE[name]}\nF1={m.f1:.3f} recall={m.recall:.2f} prec={m.precision:.2f}")
    fig.suptitle(f"{NICE[result.task]}: baseline vs final model on the held-out test set",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
