"""Test the original notebook's hypotheses H1-H6 against the data.

All six hypotheses are about *prepayment*. They are tested on never-delinquent loans with the
``Prepaid36`` target (voluntary payoff within 36 months). With ~230k loans almost any
difference is statistically significant, so the verdict is based on effect size: the
hypothesis is *supported* when the prepayment rate moves in the claimed direction by at least
``min_effect`` (2 percentage points by default) between the extreme groups, *contradicted* when
it moves the other way by at least that much, and *no material effect* otherwise. A chi-square
test of independence and a Spearman rank correlation are reported alongside.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config

T = config.PREPAY_TARGET


@dataclass(frozen=True)
class Hypothesis:
    key: str
    statement: str
    group_col: str | None      # column holding the ordered groups
    expected: str              # "increasing" | "decreasing" | "higher_in:<value>"
    order: tuple | None = None  # explicit group order (for nominal columns)


HYPOTHESES = [
    Hypothesis("H1", "Lower credit score -> higher prepayment", "CreditRange", "decreasing"),
    Hypothesis("H2", "Higher LTV -> higher prepayment", "LTV_Range", "increasing"),
    Hypothesis("H3", "Investment properties prepay more than owner-occupied", "Occupancy",
               "higher_in:I", ("O", "S", "I")),
    Hypothesis("H4", "ARMs prepay more than FRMs", None, "n/a"),
    Hypothesis("H5", "Higher DTI -> higher prepayment", "DTI_Range", "increasing"),
    Hypothesis("H6", "Lower origination rate -> higher prepayment", "RateBucket", "decreasing"),
]


def group_rates(df: pd.DataFrame, col: str, target: str = T, order=None) -> pd.DataFrame:
    """Rate of ``target`` and count per group of ``col`` (observed rows only)."""
    d = df.loc[df[target].notna() & df[col].notna(), [col, target]]
    g = d.groupby(col, observed=True)[target].agg(["mean", "size"]).rename(columns={"mean": "rate", "size": "n"})
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    return g


def check_hypothesis(df: pd.DataFrame, h: Hypothesis, min_effect: float = 0.02) -> dict:
    """Return a dict with group rates, effect, chi-square p-value, Spearman rho and verdict."""
    if h.group_col is None:
        n_arm = int((df.get("ProductType", pd.Series(dtype=str)).astype(str).str.strip() == "ARM").sum())
        return {"key": h.key, "statement": h.statement, "verdict": "not testable",
                "detail": f"every loan in the extract is a fixed-rate mortgage ({n_arm} ARMs)",
                "effect_pp": np.nan, "p_value": np.nan, "spearman": np.nan, "rates": None}
    g = group_rates(df, h.group_col, order=h.order)
    d = df.loc[df[T].notna() & df[h.group_col].notna(), [h.group_col, T]]
    table = pd.crosstab(d[h.group_col], d[T])
    p = float(stats.chi2_contingency(table)[1]) if table.shape[0] > 1 else np.nan
    if h.expected.startswith("higher_in:"):
        target_group = h.expected.split(":", 1)[1]
        base = g.drop(index=target_group)
        base_rate = float((base["rate"] * base["n"]).sum() / base["n"].sum())
        effect = float(g.loc[target_group, "rate"] - base_rate)
        rho = np.nan
        detail = f"{target_group}: {g.loc[target_group, 'rate']:.1%} vs others {base_rate:.1%}"
    else:
        codes = d[h.group_col].cat.codes if hasattr(d[h.group_col], "cat") else d[h.group_col]
        rho = float(stats.spearmanr(codes, d[T]).statistic)
        diff = float(g["rate"].iloc[-1] - g["rate"].iloc[0])   # last group minus first
        effect = diff if h.expected == "increasing" else -diff
        detail = f"{g.index[0]}: {g['rate'].iloc[0]:.1%} -> {g.index[-1]}: {g['rate'].iloc[-1]:.1%}"
    if effect >= min_effect:
        verdict = "supported"
    elif effect <= -min_effect:
        verdict = "contradicted"
    else:
        verdict = "no material effect"
    return {"key": h.key, "statement": h.statement, "verdict": verdict, "detail": detail,
            "effect_pp": 100 * effect, "p_value": p, "spearman": rho, "rates": g}


def check_all(df: pd.DataFrame, min_effect: float = 0.02) -> pd.DataFrame:
    """Run every hypothesis; returns a summary table (group rates kept in column ``rates``)."""
    return pd.DataFrame([check_hypothesis(df, h, min_effect) for h in HYPOTHESES]).set_index("key")
