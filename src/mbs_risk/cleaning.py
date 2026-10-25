"""Cleaning rules for the loan export.

The raw file encodes "not available" in several ways: the string ``"X"`` (often padded with
spaces) in code columns, and ``0`` in numeric columns where zero is impossible (a credit
score, an LTV or a DTI of 0). These are turned into proper missing values so that imputers and
tree models can treat them explicitly instead of as real numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STRING_COLUMNS = [
    "FirstTimeHomebuyer", "MSA", "Occupancy", "Channel", "PPM", "ProductType",
    "PropertyState", "PropertyType", "PostalCode", "LoanSeqNum", "LoanPurpose",
    "NumBorrowers", "SellerName", "ServicerName",
]
# Zero means "not reported" for these fields.
ZERO_IS_MISSING = ["CreditScore", "OCLTV", "LTV", "DTI", "Units"]


def clean_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw export.

    Steps:
      1. strip padding from string codes and map the placeholder ``"X"`` to missing;
      2. map impossible zeros (credit score, LTV, OCLTV, DTI, units) to missing;
      3. drop exact duplicate rows (there are none in the shipped file, but the check is cheap).

    Columns are not dropped here; leakage-driven column selection happens in
    :mod:`mbs_risk.features` via :data:`mbs_risk.config.MODEL_FEATURES`.
    """
    out = df.copy()
    for col in STRING_COLUMNS:
        if col in out:
            s = out[col].astype(object).str.strip()
            out[col] = s.where(s.notna() & ~s.isin(["X", "x", ""]), np.nan).astype(object)
    for col in ZERO_IS_MISSING:
        if col in out:
            out[col] = out[col].astype("float64").mask(out[col] == 0)
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, largest first."""
    n = df.isna().sum()
    rep = pd.DataFrame({"missing": n, "pct": 100 * n / len(df)})
    return rep[rep.missing > 0].sort_values("missing", ascending=False)


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences ``[Q1 - k*IQR, Q3 + k*IQR]``."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Drop rows outside the Tukey fences of any column in ``cols`` (missing values are kept).

    This is the corrected version of the original notebook's ``remove_outlier``, which
    compared against the *builtin* ``min`` function instead of the lower fence. Fences are
    computed once on the input frame, not re-computed after each column is filtered.
    The modelling pipeline does not drop outliers (tree models do not need it and dropping
    rows would bias the test set); this helper is used for EDA only.
    """
    keep = pd.Series(True, index=df.index)
    for col in cols:
        lo, hi = iqr_bounds(df[col].dropna(), k)
        keep &= df[col].isna() | df[col].between(lo, hi)
    return df.loc[keep]
