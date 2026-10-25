import numpy as np
import pandas as pd

from mbs_risk import clean_loans, config, remove_outliers_iqr
from mbs_risk.data import RAW_COLUMNS, stratified_sample


def test_sample_schema(raw_sample):
    assert list(raw_sample.columns) == RAW_COLUMNS
    assert len(raw_sample) == 5000
    # stratified: class ratio matches the full file (19.78% ever delinquent)
    assert abs(raw_sample[config.DELINQ_TARGET].mean() - 0.1978) < 0.002


def test_placeholders_become_missing(raw_sample):
    clean = clean_loans(raw_sample)
    for col in ["MSA", "FirstTimeHomebuyer", "PPM", "NumBorrowers"]:
        assert not clean[col].dropna().isin(["X", "x", ""]).any()
    assert (clean["CreditScore"].dropna() > 0).all()
    assert clean["ProductType"].dropna().eq("FRM").all()   # padding stripped


def test_remove_outliers_uses_lower_fence():
    df = pd.DataFrame({"a": [-100.0] + list(np.linspace(10, 20, 50)) + [500.0, np.nan]})
    out = remove_outliers_iqr(df, ["a"])
    assert -100.0 not in out["a"].values and 500.0 not in out["a"].values
    assert out["a"].isna().sum() == 1  # missing values are kept, not treated as outliers


def test_stratified_sample_keeps_ratio(raw_sample):
    s = stratified_sample(raw_sample, n=1000)
    assert len(s) == 1000
    assert abs(s[config.DELINQ_TARGET].mean() - raw_sample[config.DELINQ_TARGET].mean()) < 0.01
