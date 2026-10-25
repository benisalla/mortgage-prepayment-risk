"""Feature engineering and target definitions.

Origination-time features (safe to feed a model that scores a loan when it is originated):

* ``VintageMonth``     months since Jan 1999 of the first payment (loan cohort);
* ``MonthlyPayment``   scheduled level payment (annuity formula, see :func:`monthly_payment`);
* ``TotalInterest``    total interest over the full term if the loan runs to maturity;
* ``EstMonthlyIncome`` borrower income implied by the payment and the DTI ratio;
* ``RateSpread``       loan rate minus the median rate of its vintage month (refinance incentive
  proxy: a borrower who locked in above the market has more to gain from refinancing).

Grouping helpers reproduce the bins from the original notebook (``CreditRange``,
``LTV_Range``, ``Repay_Range``) for EDA and hypothesis tests.

Targets:

* ``EverDelinquent`` (given): the loan was 30+ days delinquent at least once.
* ``Prepaid36`` (derived, see :func:`add_targets`): the loan left the pool within 36 months of
  its first payment *and* was never delinquent, i.e. a voluntary early payoff (refinance or
  sale). Defined only for never-delinquent loans; a delinquent loan that disappears early may
  have defaulted, so those loans are excluded from the prepayment task rather than guessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CREDIT_BINS = [300, 650, 700, 750, 851]
CREDIT_LABELS = ["<650", "650-699", "700-749", "750+"]
LTV_BINS = [0, 60, 80, 90, 200]
LTV_LABELS = ["<=60", "61-80", "81-90", ">90"]
DTI_BINS = [0, 20, 30, 40, 100]
DTI_LABELS = ["<=20", "21-30", "31-40", ">40"]
REPAY_BINS = [0, 48, 96, 144, 192, 1000]
REPAY_LABELS = ["0-4y", "4-8y", "8-12y", "12-16y", "16y+"]


def monthly_payment(principal, annual_rate_pct, n_months):
    """Level monthly payment of a fully amortising loan.

    ``P * r / (1 - (1 + r) ** -n)`` with ``r`` the monthly rate. Works on scalars and arrays.
    """
    p = np.asarray(principal, dtype="float64")
    r = np.asarray(annual_rate_pct, dtype="float64") / 1200.0
    n = np.asarray(n_months, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, p * r / (1.0 - (1.0 + r) ** -n), p / n)
    return pay


def scheduled_balance(principal, annual_rate_pct, n_months, months_paid):
    """Outstanding balance after ``months_paid`` scheduled payments (no prepayment)."""
    p = np.asarray(principal, dtype="float64")
    r = np.asarray(annual_rate_pct, dtype="float64") / 1200.0
    k = np.asarray(months_paid, dtype="float64")
    pay = monthly_payment(p, annual_rate_pct, n_months)
    growth = (1.0 + r) ** k
    return np.maximum(p * growth - pay * (growth - 1.0) / np.where(r > 0, r, np.nan), 0.0)


def vintage_month(first_payment_yyyymm: pd.Series) -> pd.Series:
    """Convert a YYYYMM integer to months since January 1999 (0 = 1999-01)."""
    y = first_payment_yyyymm // 100
    m = first_payment_yyyymm % 100
    return (y - 1999) * 12 + (m - 1)


def add_features(df: pd.DataFrame, rate_reference: pd.Series | None = None) -> pd.DataFrame:
    """Add the engineered origination-time features.

    Args:
        df: cleaned loans (output of :func:`mbs_risk.cleaning.clean_loans`).
        rate_reference: optional mapping ``VintageMonth -> median OrigInterestRate`` used for
            ``RateSpread``. When ``None`` it is computed from ``df`` (it uses no target
            information, so computing it on the full frame does not leak labels).
    """
    out = df.copy()
    out["VintageMonth"] = vintage_month(out["FirstPaymentDate"])
    out["MonthlyPayment"] = monthly_payment(out["OrigUPB"], out["OrigInterestRate"], out["OrigLoanTerm"])
    out["TotalInterest"] = out["MonthlyPayment"] * out["OrigLoanTerm"] - out["OrigUPB"]
    # DTI is a percentage of gross monthly income spent on debt payments. Using the mortgage
    # payment as a lower bound of debt gives an (upper-bound) income estimate.
    out["EstMonthlyIncome"] = out["MonthlyPayment"] / (out["DTI"] / 100.0)
    if rate_reference is None:
        rate_reference = out.groupby("VintageMonth")["OrigInterestRate"].median()
    ref = out["VintageMonth"].map(rate_reference)
    ref = ref.fillna(float(np.nanmedian(rate_reference.values)))
    out["RateSpread"] = out["OrigInterestRate"] - ref
    return out


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add the readable groupings used in EDA (CreditRange, LTV_Range, DTI_Range, Repay_Range, RateBucket)."""
    out = df.copy()
    out["CreditRange"] = pd.cut(out["CreditScore"], CREDIT_BINS, labels=CREDIT_LABELS, right=False)
    out["LTV_Range"] = pd.cut(out["LTV"], LTV_BINS, labels=LTV_LABELS)
    out["DTI_Range"] = pd.cut(out["DTI"], DTI_BINS, labels=DTI_LABELS)
    out["Repay_Range"] = pd.cut(out["MonthsInRepayment"], REPAY_BINS, labels=REPAY_LABELS, right=False)
    q = out["OrigInterestRate"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    q = np.unique(q)
    labels = [f"<={q[1]:g}%"] + [f"{a:g}-{b:g}%" for a, b in zip(q[1:-2], q[2:-1])] + [f">{q[-2]:g}%"]
    out["RateBucket"] = pd.cut(out["OrigInterestRate"], q, include_lowest=True, labels=labels)
    return out


def add_targets(df: pd.DataFrame, horizon: int = config.PREPAY_HORIZON_MONTHS) -> pd.DataFrame:
    """Add the prepayment target ``Prepaid36`` (NaN for ever-delinquent loans).

    The observation window of the extract ends around November 2016 (the longest observed
    loan life is 212 months from a 1999-03 first payment), so every loan in the file had at
    least 36 months in which it could be observed; a loan observed for fewer than ``horizon``
    months therefore left the pool rather than being censored by the end of the data.
    """
    out = df.copy()
    prepaid = (out["MonthsInRepayment"] <= horizon).astype("float64")
    out[config.PREPAY_TARGET] = prepaid.where(out[config.DELINQ_TARGET] == 0)
    return out


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean + engineer features + bins + targets in one call."""
    from .cleaning import clean_loans

    return add_targets(add_bins(add_features(clean_loans(df_raw))))


def model_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Select origination-time model inputs ``X`` and the target ``y`` (rows with a target only).

    Raises if any leaking column would slip into ``X``.
    """
    rows = df[target].notna()
    X = df.loc[rows, config.MODEL_FEATURES].copy()
    forbidden = set(config.POST_ORIGINATION_COLUMNS) | set(config.ID_COLUMNS) | {
        config.DELINQ_TARGET, config.PREPAY_TARGET}
    leak = forbidden & set(X.columns)
    if leak:
        raise AssertionError(f"leaking columns in model inputs: {sorted(leak)}")
    y = df.loc[rows, target].astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)
